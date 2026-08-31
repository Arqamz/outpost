"""Compiling a resolved plan into launcher syntax.

The rule these all serve: translation must not change meaning. A plan that says
rank 3 gets GPU-c3d4 and cores 24-31 must produce a command that does exactly
that, or produce no command at all.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import REPO_DIR

from reconciler.launcher import (
    APPFILE,
    LAUNCHERS,
    RANKFILE,
    LauncherUnsupported,
    LocalProcessLauncherAdapter,
    OpenMPILauncherAdapter,
    _compress,
    for_plan,
    slot_expression,
)
from reconciler.models import JobSpec
from reconciler.placement import resolve
from reconciler.topology import normalize

FIXTURES = Path(REPO_DIR) / "tests" / "fixtures"
EXAMPLES = Path(REPO_DIR) / "contract" / "launch-intent" / "v1" / "examples"


def topo(name: str, node: str = "n1"):
    return normalize(json.loads((FIXTURES / f"topology_{name}.json").read_text()), node)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and \
            isinstance(out.get(key), dict) else value
    return out


def intent(**over) -> dict:
    return _merge(json.loads((EXAMPLES / "one-rank-per-gpu.json").read_text()), over)


def explicit_intent(**over) -> dict:
    return _merge(json.loads((EXAMPLES / "explicit-placement.json").read_text()), over)


def spec(**kw) -> JobSpec:
    return JobSpec(**{"name": "nccl", "image": "docker://x@sha256:" + "a" * 64,
                      "command": ["all_reduce_perf_mpi", "-b", "8"],
                      "output_dir": "/out", **kw})


def compiled(plan=None, node_ips=None, **kw):
    plan = plan or resolve(intent(), [topo("2socket_8gpu")])
    ips = node_ips or {"n1": "192.168.71.11"}
    return OpenMPILauncherAdapter().compile(
        plan, spec(), node_ips=ips, workdir="/tmp/cluster/job-1",
        image="/tmp/cluster/job-1/image.sif", **kw)


class TestSlotSyntax:
    @pytest.mark.parametrize(("values", "expected"), [
        ([0, 1, 2, 3], "0-3"),
        ([0, 2, 4], "0,2,4"),
        ([0, 1, 2, 8], "0-2,8"),
        ([5], "5"),
        ([], ""),
        ([3, 1, 2], "1-3"),
    ])
    def test_ranges_are_compressed(self, values, expected):
        assert _compress(values) == expected

    def test_one_socket(self):
        assert slot_expression(((0, 0), (0, 1), (0, 2))) == "0:0-2"

    def test_sockets_are_separated_by_semicolons(self):
        # mpirun(1) rankfile syntax: `slot=0:1;1:0-2`.
        assert slot_expression(((0, 1), (1, 0), (1, 1), (1, 2))) == "0:1;1:0-2"

    def test_socket_order_is_deterministic(self):
        assert slot_expression(((1, 0), (0, 0))) == slot_expression(((0, 0), (1, 0)))


class TestOpenMpiRendering:
    def test_one_appfile_line_per_rank(self):
        rendering = compiled()
        lines = rendering.files[APPFILE].strip().splitlines()
        assert len(lines) == 8
        assert all(line.startswith("-np 1 --host 192.168.71.11 ") for line in lines)

    def test_each_rank_sees_only_its_own_gpu(self):
        # The reason an appfile is required at all: a single shared command
        # cannot give rank 0 one device and rank 1 another.
        plan = resolve(intent(), [topo("2socket_8gpu")])
        rendering = compiled(plan)
        seen = []
        for rank in plan.ranks:
            script = rendering.files[f"launcher-rank-{rank.global_rank}.sh"]
            assert f"CUDA_VISIBLE_DEVICES={rank.gpu_uuids[0]}" in script
            seen.append(rank.gpu_uuids[0])
        assert len(set(seen)) == 8
        # and the appfile still has exactly one line per rank pointing at them
        assert len(rendering.files[APPFILE].strip().splitlines()) == 8

    def test_per_rank_env_is_exported_before_exec(self):
        # A preflight probe is spliced in as a subshell BEFORE the `exec` line
        # (adapter._with_preflight_probe), on the bare host — it never starts
        # a container. CUDA_VISIBLE_DEVICES must therefore be `export`ed into
        # the script's OWN shell, not just handed to apptainer via `--env`
        # (which only takes effect once the container actually starts), or
        # the probe reads it back unset and reports a false mismatch for a
        # correctly-applied GPU assignment. Found live on real AWS hardware.
        plan = resolve(intent(), [topo("2socket_8gpu")])
        rendering = compiled(plan)
        for rank in plan.ranks:
            script = rendering.files[f"launcher-rank-{rank.global_rank}.sh"]
            export_line = f"export CUDA_VISIBLE_DEVICES={rank.gpu_uuids[0]}"
            assert export_line in script
            assert script.index(export_line) < script.index("\nexec ")

    def test_devices_are_pinned_by_uuid_not_index(self):
        # Index order depends on driver enumeration and CUDA_DEVICE_ORDER; a plan
        # that promised a specific physical card must not be re-pointed by either.
        rendering = compiled()
        for value in rendering.per_rank_env.values():
            assert value["CUDA_VISIBLE_DEVICES"].startswith("GPU-")


class TestUcxDevicePinning:
    # Found live: on a shared box also running a kubelet/CNI workload, UCX
    # auto-selected a pod-network interface and advertised an address the
    # peer rank could never reach, segfaulting MPI_Init on both ranks
    # (job-1e25ee9612ff). btl_tcp_if_include's IP/subnet matching doesn't
    # reach UCX, which needs an actual device name — node_ifaces threads
    # that in, same shape as CUDA_VISIBLE_DEVICES above.

    def test_node_ifaces_pins_ucx_net_devices_per_rank(self):
        rendering = compiled(node_ifaces={"n1": "ens5"})
        for env in rendering.per_rank_env.values():
            assert env["UCX_NET_DEVICES"] == "ens5"
            assert env["UCX_TLS"] == "tcp,sm,self"

    def test_no_node_ifaces_means_no_ucx_pinning(self):
        # Default (no cross-node fabric info supplied) — byte-identical to
        # before this fix, no silent behavior change for single-node or
        # non-UCX-affected callers.
        rendering = compiled()
        for env in rendering.per_rank_env.values():
            assert "UCX_NET_DEVICES" not in env
            assert "UCX_TLS" not in env

    def test_unknown_node_gets_no_ucx_pinning(self):
        # A node_ifaces map that doesn't cover a rank's node degrades to "no
        # pinning for that rank" rather than a KeyError.
        rendering = compiled(node_ifaces={"some-other-node": "eth0"})
        for env in rendering.per_rank_env.values():
            assert "UCX_NET_DEVICES" not in env

    def test_ucx_pinning_is_exported_before_exec(self):
        # Same reasoning as CUDA_VISIBLE_DEVICES: a preflight probe runs as a
        # bare-host subshell before `exec`, so the value must be in the
        # script's own shell env, not only apptainer's --env.
        plan = resolve(intent(), [topo("2socket_8gpu")])
        rendering = compiled(plan, node_ifaces={"n1": "ens5"})
        for rank in plan.ranks:
            script = rendering.files[f"launcher-rank-{rank.global_rank}.sh"]
            export_line = "export UCX_NET_DEVICES=ens5"
            assert export_line in script
            assert script.index(export_line) < script.index("\nexec ")

    def test_rankfile_addresses_cores_as_socket_and_core(self):
        plan = resolve(intent(), [topo("2socket_8gpu")])
        lines = compiled(plan).files[RANKFILE].strip().splitlines()
        assert len(lines) == 8
        assert lines[0] == "rank 0=192.168.71.11 slot=0:0-7"
        # ranks 4-7 sit on the second socket, per the plan
        assert lines[4].endswith("slot=1:0-7")

    def test_the_command_references_both_files_and_asks_for_evidence(self):
        rendering = compiled()
        argv = rendering.argv
        assert argv[0] == "mpirun"
        assert "--rankfile" in argv and f"/tmp/cluster/job-1/{RANKFILE}" in argv
        assert "--app" in argv and f"/tmp/cluster/job-1/{APPFILE}" in argv
        # The launcher's own account of what it bound — what the receipt checks.
        assert "--report-bindings" in argv and "--display-map" in argv

    def test_mca_options_are_passed_through_sorted(self):
        rendering = compiled(mca={"btl": "tcp,self", "btl_tcp_if_include": "192.168.71.0/24"})
        argv = " ".join(rendering.argv)
        assert "--mca btl tcp,self" in argv
        assert "--mca btl_tcp_if_include 192.168.71.0/24" in argv

    def test_rsh_agent_and_tree_spawn_are_wired(self):
        rendering = compiled(rsh_agent="ssh -i /k -l cluster", no_tree_spawn=True)
        argv = " ".join(rendering.argv)
        assert "plm_rsh_agent" in argv and "plm_rsh_no_tree_spawn 1" in argv

    def test_a_preview_launches_nothing(self):
        preview = compiled().preview_argv
        assert "--do-not-launch" in preview and "--display-map" in preview

    def test_rendering_is_deterministic(self):
        first, second = compiled(), compiled()
        assert first.argv == second.argv and first.files == second.files

    def test_a_node_missing_from_the_allocation_is_refused(self):
        with pytest.raises(LauncherUnsupported, match="not in the allocation"):
            compiled(node_ips={"other": "10.0.0.1"})

    def test_unbindable_cores_are_noted_not_dropped_silently(self):
        # Cores assigned but with no socket/core identity cannot be addressed in a
        # rankfile. The binding is lost either way; what must not be lost is the
        # fact that it was. Explicit placement is the only way to get cores on a
        # node with no core identity — every other policy needs that identity.
        doc = json.loads((FIXTURES / "topology_dev_1gpu.json").read_text())
        for cpu in doc["cpus"]:
            cpu["socket"] = cpu["core"] = None
        topology = normalize(doc, "n1")
        plan = resolve(explicit_intent(
            process={"ranks_per_node": 1},
            cpu={"explicit_cpu_ids": [[0, 1]], "cores_per_rank": 2},
            gpu={"explicit_gpu_uuids": [[topology.gpus[0].uuid]]}), [topology])
        assert plan.ranks[0].cpu_ids == (0, 1) and plan.ranks[0].cpu_slots == ()
        rendering = compiled(plan)
        assert RANKFILE not in rendering.files
        assert any("cannot be expressed in a rankfile" in n for n in rendering.notes)


class TestAppfileQuoting:
    """OpenMPI splits an appfile line on whitespace and does NOT honour shell
    quoting (mpirun(1)). A rendered benchmark command is `sh -c '<script>'`, so
    inlining the argv silently turns one rank into several malformed app
    contexts — confirmed live, where a real command produced five lines where
    there should have been one."""

    GTL_COMMAND = ["sh", "-c",
                   "(\nprobe\n) > /out/probe.json 2>/dev/null || true\n"
                   "exec all_reduce_perf_mpi -b 8 -e 8G"]

    def _rendered(self):
        plan = resolve(intent(cpu={"cores_per_rank": 2}), [topo("dev_1gpu")])
        return plan, OpenMPILauncherAdapter().compile(
            plan, spec(command=self.GTL_COMMAND), node_ips={"n1": "10.0.0.1"},
            workdir="/tmp/w", image="/tmp/w/i.sif")

    def test_a_multiline_command_stays_one_appfile_line_per_rank(self):
        plan, rendering = self._rendered()
        assert len(rendering.files[APPFILE].strip().splitlines()) == len(plan.ranks)

    def test_every_appfile_token_is_whitespace_free(self):
        _, rendering = self._rendered()
        for line in rendering.files[APPFILE].strip().splitlines():
            assert line.split() == line.split(" "), line

    def test_the_command_survives_intact_in_the_rank_script(self):
        _, rendering = self._rendered()
        script = rendering.files["launcher-rank-0.sh"]
        assert "all_reduce_perf_mpi -b 8 -e 8G" in script
        assert script.startswith("#!/bin/sh")
        # exec, so the rank process IS the workload and keeps mpirun's mask.
        assert "\nexec apptainer exec" in script

    def test_the_script_records_what_the_rank_was_given(self):
        _, rendering = self._rendered()
        script = rendering.files["launcher-rank-0.sh"]
        assert "cpus=[0, 1]" in script and "gpu=GPU-" in script

    def test_a_workdir_with_whitespace_is_refused(self):
        # The script path itself would then break the same way.
        plan = resolve(intent(cpu={"cores_per_rank": 2}), [topo("dev_1gpu")])
        with pytest.raises(LauncherUnsupported, match="whitespace"):
            OpenMPILauncherAdapter().compile(
                plan, spec(), node_ips={"n1": "10.0.0.1"},
                workdir="/tmp/my runs", image="/i.sif")


class TestCapabilityRefusals:
    def test_interleave_is_refused_because_numactl_is_absent(self):
        plan = resolve(intent(), [topo("2socket_8gpu")])
        reasons = OpenMPILauncherAdapter().unsupported(
            plan, intent(memory={"strategy": "interleave"}))
        assert any("interleave" in r and "numactl" in r for r in reasons)

    def test_local_memory_is_satisfied_by_single_node_confinement(self):
        # No memory-policy tool is needed when a rank sits entirely on one NUMA
        # node: Linux's first-touch default already places its pages there.
        plan = resolve(intent(), [topo("2socket_8gpu")])
        assert OpenMPILauncherAdapter().unsupported(plan, intent()) == []

    def test_local_memory_is_refused_when_a_rank_spans_nodes(self):
        # closest_to_gpu confines a rank to one socket by construction, so the
        # only way to straddle two is to name the cores explicitly: 0-1 are on
        # socket 0, 40-41 on socket 1.
        t = topo("2socket_8gpu")
        plan = resolve(explicit_intent(
            process={"ranks_per_node": 1},
            cpu={"explicit_cpu_ids": [[0, 1, 40, 41]], "cores_per_rank": 4},
            gpu={"explicit_gpu_uuids": [[t.gpus[0].uuid]]}), [t])
        spanning = [r for r in plan.ranks if len(r.numa_nodes) > 1]
        assert spanning, "fixture no longer produces a socket-spanning rank"
        reasons = OpenMPILauncherAdapter().unsupported(plan, intent())
        assert any("span more than one NUMA node" in r for r in reasons)

    def test_local_process_launcher_refuses_multiple_ranks(self):
        plan = resolve(intent(), [topo("2socket_8gpu")])
        with pytest.raises(LauncherUnsupported, match="single rank"):
            LocalProcessLauncherAdapter().compile(
                plan, spec(), node_ips={"n1": "1.2.3.4"}, workdir="/w", image="/w/i.sif")

    def test_local_process_launcher_cannot_span_nodes(self):
        plan = resolve(intent(cpu={"cores_per_rank": 2}),
                       [topo("dev_1gpu", "a"), topo("dev_1gpu", "b")]) \
            if False else None
        # Built directly rather than resolved: two nodes sharing a GPU UUID is
        # refused upstream, and this is about the launcher's own capability.
        from reconciler.placement import LaunchPlan, RankPlacement
        ranks = tuple(RankPlacement(i, 0, f"n{i}", (), (), (), (), (), ())
                      for i in range(2))
        plan = LaunchPlan("lp-x", {"type": "local"}, ranks, {"warnings": ()}, {})
        reasons = LocalProcessLauncherAdapter().unsupported(plan, intent())
        assert any("multiple nodes" in r for r in reasons)


class TestLocalProcessRendering:
    def _single(self):
        plan = resolve(intent(cpu={"cores_per_rank": 2}, memory={"strategy": "none"}),
                       [topo("dev_1gpu")])
        return plan, LocalProcessLauncherAdapter().compile(
            plan, spec(), node_ips={"n1": "127.0.0.1"}, workdir="/w", image="/w/i.sif")

    def test_binding_uses_taskset(self):
        # No MPI in this path, so there is no rankfile; taskset sets the mask on
        # itself and execs, and every descendant inherits it. Same probe-splice
        # convention as OpenMPI's per-rank script, so the argv now references
        # the script rather than running taskset directly.
        plan, rendering = self._single()
        assert rendering.argv == ["/bin/sh", "/w/launcher-rank-0.sh"]
        script = rendering.files["launcher-rank-0.sh"]
        assert "exec taskset -c 0-1 " in script
        assert "apptainer" in script

    def test_the_single_rank_still_gets_its_device(self):
        plan, rendering = self._single()
        script = rendering.files["launcher-rank-0.sh"]
        assert f"CUDA_VISIBLE_DEVICES={plan.ranks[0].gpu_uuids[0]}" in script

    def test_no_binding_report_is_admitted_not_faked(self):
        _, rendering = self._single()
        assert rendering.preview_argv is None
        assert any("no binding report" in n for n in rendering.notes)

    def test_probe_rank_identity_is_exported_but_never_reaches_the_container(self):
        # The regression this guards: placement-probe.sh identifies its own
        # rank via OMPI_COMM_WORLD_RANK/PMIX_RANK/PMI_RANK — vars no launcher
        # sets here (no mpirun of any kind wraps this script). Without an
        # explicit export, the probe reports global_rank: null and the
        # receipt discards an otherwise-correct observation as unattributable
        # — confirmed live, `mismatched` for a placement that actually held.
        # This launcher is exactly one rank by construction, so the identity
        # is known, not detected. It must land in the script's own export
        # lines (for the probe) but NEVER in per_rank_env / container --env
        # (HPL's own internal mpirun inheriting a stray OMPI_COMM_WORLD_RANK
        # from a "launch" that never happened is exactly the bug class
        # _with_preflight_probe's splice-before-exec ordering exists to
        # avoid elsewhere).
        plan, rendering = self._single()
        script = rendering.files["launcher-rank-0.sh"]
        assert "export OMPI_COMM_WORLD_RANK=0" in script
        assert "export OMPI_COMM_WORLD_LOCAL_RANK=0" in script
        assert "export OMPI_COMM_WORLD_SIZE=1" in script
        assert "OMPI_COMM_WORLD_RANK" not in rendering.per_rank_env[0]
        assert not any("OMPI_COMM_WORLD_RANK" in arg for arg in rendering.argv)


class TestSelection:
    def test_the_launcher_the_nodes_reported_is_used(self):
        plan = resolve(intent(), [topo("2socket_8gpu")])
        assert isinstance(for_plan(plan), OpenMPILauncherAdapter)

    def test_a_single_rank_on_an_unknown_launcher_runs_locally(self):
        from reconciler.placement import LaunchPlan, RankPlacement
        plan = LaunchPlan("lp-x", {"type": ""},
                          (RankPlacement(0, 0, "n1", (), (), (), (), (), ()),),
                          {"warnings": ()}, {})
        assert isinstance(for_plan(plan), LocalProcessLauncherAdapter)

    def test_an_unknown_launcher_with_many_ranks_is_refused(self):
        from reconciler.placement import LaunchPlan, RankPlacement
        ranks = tuple(RankPlacement(i, i, "n1", (), (), (), (), (), ())
                      for i in range(2))
        plan = LaunchPlan("lp-x", {"type": "slurm"}, ranks, {"warnings": ()}, {})
        with pytest.raises(LauncherUnsupported, match="no launcher adapter"):
            for_plan(plan)

    def test_job_launcher_single_always_runs_locally_even_on_a_real_mpi_node(self):
        # The regression this guards: a launcher: single job's container may
        # run its OWN internal mpirun (HPL's hpl-mxp.sh, for one). Wrapping
        # that in an OUTER mpirun — which is what happens if the node's own
        # installed MPI (plan.launcher, from topology) picks the adapter
        # instead of the job's declared shape — is not just redundant but
        # broken: OpenMPI refuses a nested invocation outright ("mpirun does
        # not support recursive calls"), confirmed live running real HPL on a
        # real MPI-capable node. job_launcher="single" must win regardless of
        # what plan.launcher reports.
        plan = resolve(intent(cpu={"cores_per_rank": 2}), [topo("dev_1gpu")])
        assert plan.launcher.get("type") == "openmpi"  # the node DOES have real MPI
        assert isinstance(for_plan(plan, job_launcher="single"), LocalProcessLauncherAdapter)
        assert isinstance(for_plan(plan, job_launcher="mpi"), OpenMPILauncherAdapter)

    def test_job_launcher_single_refuses_a_multi_rank_plan(self):
        from reconciler.placement import LaunchPlan, RankPlacement
        ranks = tuple(RankPlacement(i, i, "n1", (), (), (), (), (), ())
                      for i in range(2))
        plan = LaunchPlan("lp-x", {"type": "openmpi", "version": "4.1.6"}, ranks,
                          {"warnings": ()}, {})
        with pytest.raises(LauncherUnsupported, match="resolved to 2 rank"):
            for_plan(plan, job_launcher="single")

    def test_every_registered_adapter_declares_the_same_capability_keys(self):
        # A new adapter that forgot a key would read as "false" and quietly
        # refuse work it can actually do, or worse, be assumed capable.
        keys = {frozenset(a.CAPABILITIES) for a in LAUNCHERS.values()}
        assert len(keys) == 1
        assert keys.pop() == {"cpu_binding", "memory_binding", "gpu_visibility",
                              "explicit_rank_map", "mapping_preview", "binding_report"}
