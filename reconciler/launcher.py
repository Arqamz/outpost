"""Compile a resolved Launch Plan into a launcher's own syntax.

This is the ONLY place that knows what mpirun's flags look like. A benchmark
suite states placement semantically and the resolver turns it into exact cores
and devices; everything below this line is translation, and translation must not
change the meaning — so nothing here makes a placement decision. If a launcher
cannot express something the plan requires, it says so and the job stops, rather
than rendering a command that silently drops it.

WHY AN APPFILE RATHER THAN ONE SHARED COMMAND. Per-rank GPU visibility cannot be
expressed any other way: `mpirun ... apptainer exec --env CUDA_VISIBLE_DEVICES=X`
applies X to every rank. OpenMPI's appfile gives each rank its own argv, which is
what lets rank 0 see one device and rank 1 another. LocalHostAdapter already
takes this path for hybrid jobs; the VM path gains it here.

WHY A RANKFILE FOR CPUs. `--map-by`/`--bind-to` express policies; the plan is
already an exact mapping, and re-expressing it as a policy would let the launcher
re-derive something different. A rankfile states the mapping literally, and
`--report-bindings` makes the launcher say back what it did — which is what the
preflight then checks against reality.
"""
from __future__ import annotations

import shlex
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from .placement import LaunchPlan, RankPlacement

APPFILE = "launcher-appfile.txt"
RANKFILE = "launcher-rankfile.txt"


class LauncherUnsupported(ValueError):
    """The plan needs something this launcher cannot express."""


@dataclass(frozen=True)
class LauncherRendering:
    """Everything needed to start the job, plus the evidence of how."""
    launcher: str
    version: str
    argv: list[str]
    files: dict[str, str] = field(default_factory=dict)   # name -> content, into the workdir
    preview_argv: list[str] | None = None                 # dry mapping preview, if supported
    per_rank_env: dict = field(default_factory=dict)      # global_rank -> env, for the receipt
    notes: tuple[str, ...] = ()


def _compress(values: list[int]) -> str:
    """[0,1,2,3,8] -> '0-3,8' — the form OpenMPI's rankfile documents."""
    if not values:
        return ""
    ordered = sorted(set(values))
    runs: list[tuple[int, int]] = [(ordered[0], ordered[0])]
    for value in ordered[1:]:
        if value == runs[-1][1] + 1:
            runs[-1] = (runs[-1][0], value)
        else:
            runs.append((value, value))
    return ",".join(str(lo) if lo == hi else f"{lo}-{hi}" for lo, hi in runs)


def slot_expression(slots: tuple[tuple[int, int], ...]) -> str:
    """(socket, core) pairs -> `0:0-3;1:0-2`, OpenMPI's rankfile slot syntax.

    `;` separates sockets and `,`/`-` list cores within one (mpirun(1), rankfile
    section). The socket:core form is used rather than a bare slot list because
    the bare form's meaning depends on whether the build counts hardware threads,
    and a binding that means something different on another machine is not a
    binding an approved plan can promise.
    """
    by_socket: dict[int, list[int]] = {}
    for socket, core in slots:
        by_socket.setdefault(socket, []).append(core)
    return ";".join(f"{socket}:{_compress(cores)}"
                    for socket, cores in sorted(by_socket.items()))


class LauncherAdapter(ABC):
    name: str = "abstract"
    CAPABILITIES: dict = {}

    @abstractmethod
    def compile(self, plan: LaunchPlan, spec, *, node_ips: dict[str, str],
                workdir: str, image: str, **kw) -> LauncherRendering: ...

    def unsupported(self, plan: LaunchPlan, intent: dict) -> list[str]:
        """Requirements this launcher cannot express, as human-readable reasons.

        Checked BEFORE compiling so an unsupported capability is reported as
        itself, rather than as whatever the half-rendered command does next.
        """
        reasons: list[str] = []
        caps = self.CAPABILITIES
        wants_binding = any(r.cpu_ids for r in plan.ranks)
        if wants_binding and not caps.get("cpu_binding"):
            reasons.append(f"{self.name} cannot bind ranks to CPUs, but the plan assigns "
                           "explicit cores to at least one rank")
        if len({r.node for r in plan.ranks}) > 1 and not caps.get("explicit_rank_map"):
            reasons.append(f"{self.name} cannot place ranks across multiple nodes")
        if any(r.gpu_uuid for r in plan.ranks) and not caps.get("gpu_visibility"):
            reasons.append(f"{self.name} cannot restrict which GPU each rank sees")

        memory = intent["memory"]["strategy"]
        if memory == "interleave":
            reasons.append(
                f"{self.name} cannot express memory.strategy 'interleave': it needs an "
                "explicit memory policy (numactl --interleave), which is not installed on "
                "the nodes")
        elif memory == "local":
            # `local` IS satisfiable without a memory-policy tool, but only by
            # construction: a rank confined to one NUMA node gets node-local pages
            # from Linux's default first-touch policy. A rank spanning two nodes
            # does not, and saying otherwise would be a claim nothing enforces.
            spanning = [r.global_rank for r in plan.ranks if len(r.numa_nodes) > 1]
            unknown = [r.global_rank for r in plan.ranks if not r.numa_nodes]
            if spanning:
                reasons.append(
                    f"memory.strategy is 'local' but rank(s) {spanning} span more than one "
                    "NUMA node, so first-touch cannot guarantee local pages and no memory "
                    "policy tool is available to force it")
            if unknown:
                reasons.append(
                    f"memory.strategy is 'local' but rank(s) {unknown} have no known NUMA "
                    "node, so locality cannot be established or verified")
        return reasons


class OpenMPILauncherAdapter(LauncherAdapter):
    """OpenMPI 4.1.x: a per-rank appfile for argv and env, a rankfile for cores."""

    name = "openmpi"
    CAPABILITIES = {
        "cpu_binding": True,
        "memory_binding": True,      # by confinement only — see unsupported()
        "gpu_visibility": True,
        "explicit_rank_map": True,
        "mapping_preview": True,
        "binding_report": True,
    }

    def compile(self, plan: LaunchPlan, spec, *, node_ips: dict[str, str],
                workdir: str, image: str, mca: dict[str, str] | None = None,
                rsh_agent: str | None = None, no_tree_spawn: bool = False,
                container_argv=None) -> LauncherRendering:
        from .adapter import container_argv as default_container_argv
        build_argv = container_argv or default_container_argv
        import dataclasses

        missing = sorted({r.node for r in plan.ranks} - set(node_ips))
        if missing:
            raise LauncherUnsupported(
                f"the plan places ranks on {missing}, which are not in the allocation")

        appfile_lines: list[str] = []
        files_extra: dict[str, str] = {}
        rankfile_lines: list[str] = []
        per_rank_env: dict[int, dict] = {}
        notes: list[str] = []

        # An appfile line is split on WHITESPACE and shell quoting is not honoured
        # (mpirun(1); the same constraint adapter.write_appfile documents). A
        # rendered benchmark command is `sh -c '<multi-line probe script>'`, so
        # inlining the argv turns one rank into a dozen malformed app contexts —
        # verified against a real rendered command, which produced five lines
        # where there should have been one. Each rank therefore gets a tiny
        # script, and the appfile references it by a path with no spaces in it.
        if any(c.isspace() for c in workdir):
            raise LauncherUnsupported(
                f"workdir {workdir!r} contains whitespace; an appfile cannot reference it")

        for rank in plan.ranks:
            env = dict(spec.env)
            if rank.gpu_uuid:
                # By UUID, not by index: index order depends on the driver's
                # enumeration and on CUDA_DEVICE_ORDER, and a plan that promised
                # a specific physical device must not be re-pointed by either.
                env["CUDA_VISIBLE_DEVICES"] = rank.gpu_uuid
            per_rank_env[rank.global_rank] = env

            rank_spec = dataclasses.replace(spec, image=image, gpu=bool(rank.gpu_uuid))
            argv = build_argv(rank_spec, workdir, spec.output_dir, None, env)

            script_name = f"launcher-rank-{rank.global_rank}.sh"
            # `exec` so the rank process IS the workload: one less process in the
            # tree, and the affinity mask mpirun set is carried straight into it.
            files_extra[script_name] = (
                "#!/bin/sh\n"
                f"# global rank {rank.global_rank} (local {rank.local_rank}) on {rank.node}\n"
                f"# cpus={list(rank.cpu_ids)} gpu={rank.gpu_uuid or '-'}\n"
                f"exec {shlex.join(argv)}\n")
            appfile_lines.append(
                f"-np 1 --host {node_ips[rank.node]} /bin/sh {workdir}/{script_name}")

            if rank.cpu_slots:
                rankfile_lines.append(
                    f"rank {rank.global_rank}={node_ips[rank.node]} "
                    f"slot={slot_expression(rank.cpu_slots)}")
            elif rank.cpu_ids:
                # Cores were assigned but their socket/core identity was not
                # discoverable, so there is no address to write. Binding is
                # dropped, and dropping it silently is what this note prevents.
                notes.append(f"rank {rank.global_rank} has CPUs {list(rank.cpu_ids)} but no "
                             "socket/core identity, so it cannot be expressed in a rankfile")

        argv = ["mpirun"]
        for key, value in sorted((mca or {}).items()):
            argv += ["--mca", key, value]
        if rsh_agent:
            argv += ["--mca", "plm_rsh_agent", rsh_agent]
        if no_tree_spawn:
            argv += ["--mca", "plm_rsh_no_tree_spawn", "1"]

        files = {APPFILE: "\n".join(appfile_lines) + "\n", **files_extra}
        if rankfile_lines:
            files[RANKFILE] = "\n".join(rankfile_lines) + "\n"
            argv += ["--rankfile", f"{workdir}/{RANKFILE}"]
        # Make the launcher state what it did. This is the evidence the receipt
        # is compared against, and it costs nothing to always ask for it.
        argv += ["--report-bindings", "--display-map"]
        argv += ["--app", f"{workdir}/{APPFILE}"]

        # A dry mapping preview: same placement inputs, nothing launched.
        preview = ["mpirun", "--display-map", "--display-allocation", "--do-not-launch"]
        if rankfile_lines:
            preview += ["--rankfile", f"{workdir}/{RANKFILE}"]
        preview += ["--app", f"{workdir}/{APPFILE}"]

        return LauncherRendering(
            launcher=self.name, version=str((plan.launcher or {}).get("version") or ""),
            argv=argv, files=files, preview_argv=preview,
            per_rank_env=per_rank_env, notes=tuple(notes),
        )


class LocalProcessLauncherAdapter(LauncherAdapter):
    """A single rank, started directly. No MPI, so binding comes from taskset.

    This is the `launcher: single` path — one container on one node. It cannot
    place ranks across nodes and has no mapping report of its own, which is why
    those capabilities are false rather than approximated.
    """

    name = "local"
    CAPABILITIES = {
        "cpu_binding": True,          # taskset (util-linux, present on every node)
        "memory_binding": False,      # numactl is not installed
        "gpu_visibility": True,
        "explicit_rank_map": False,
        "mapping_preview": False,
        "binding_report": False,
    }

    def compile(self, plan: LaunchPlan, spec, *, node_ips: dict[str, str],
                workdir: str, image: str, container_argv=None, **kw) -> LauncherRendering:
        from .adapter import container_argv as default_container_argv
        import dataclasses
        build_argv = container_argv or default_container_argv

        if len(plan.ranks) != 1:
            raise LauncherUnsupported(
                f"{self.name} runs a single rank, but the plan has {len(plan.ranks)}")
        rank: RankPlacement = plan.ranks[0]

        env = dict(spec.env)
        if rank.gpu_uuid:
            env["CUDA_VISIBLE_DEVICES"] = rank.gpu_uuid
        rank_spec = dataclasses.replace(spec, image=image, gpu=bool(rank.gpu_uuid))
        argv = list(build_argv(rank_spec, workdir, spec.output_dir, None, env))
        if rank.cpu_ids:
            # taskset sets the affinity mask on itself and then execs, so every
            # descendant — apptainer, and the workload inside it — inherits it.
            argv = ["taskset", "-c", _compress(list(rank.cpu_ids))] + argv

        return LauncherRendering(
            launcher=self.name, version="", argv=argv,
            per_rank_env={rank.global_rank: env},
            notes=("binding via taskset; this launcher has no binding report of its own, "
                   "so the preflight is the only confirmation",) if rank.cpu_ids else (),
        )


LAUNCHERS: dict[str, LauncherAdapter] = {
    OpenMPILauncherAdapter.name: OpenMPILauncherAdapter(),
    LocalProcessLauncherAdapter.name: LocalProcessLauncherAdapter(),
}


def for_plan(plan: LaunchPlan) -> LauncherAdapter:
    """The adapter that compiles this plan, by the launcher the nodes reported."""
    kind = (plan.launcher or {}).get("type") or ""
    if len(plan.ranks) == 1 and kind not in LAUNCHERS:
        return LAUNCHERS[LocalProcessLauncherAdapter.name]
    if kind not in LAUNCHERS:
        raise LauncherUnsupported(
            f"no launcher adapter for {kind!r} (have {sorted(LAUNCHERS)})")
    return LAUNCHERS[kind]
