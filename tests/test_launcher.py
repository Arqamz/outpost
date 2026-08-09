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
            assert f"CUDA_VISIBLE_DEVICES={rank.gpu_uuid}" in script
            seen.append(rank.gpu_uuid)
        assert len(set(seen)) == 8
        # and the appfile still has exactly one line per rank pointing at them
        assert len(rendering.files[APPFILE].strip().splitlines()) == 8

    def test_devices_are_pinned_by_uuid_not_index(self):
        # Index order depends on driver enumeration and CUDA_DEVICE_ORDER; a plan
        # that promised a specific physical card must not be re-pointed by either.
        rendering = compiled()
        for value in rendering.per_rank_env.values():
            assert value["CUDA_VISIBLE_DEVICES"].startswith("GPU-")

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
            gpu={"explicit_gpu_uuids": [topology.gpus[0].uuid]}), [topology])
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
            gpu={"explicit_gpu_uuids": [t.gpus[0].uuid]}), [t])
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
        ranks = tuple(RankPlacement(i, 0, f"n{i}", (), (), (), None, None, None)
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
        # itself and execs, and every descendant inherits it.
        plan, rendering = self._single()
        assert rendering.argv[:3] == ["taskset", "-c", "0-1"]
        assert "apptainer" in rendering.argv

    def test_the_single_rank_still_gets_its_device(self):
        plan, rendering = self._single()
        assert f"CUDA_VISIBLE_DEVICES={plan.ranks[0].gpu_uuid}" in " ".join(rendering.argv)

    def test_no_binding_report_is_admitted_not_faked(self):
        _, rendering = self._single()
        assert rendering.preview_argv is None
        assert any("no binding report" in n for n in rendering.notes)


class TestSelection:
    def test_the_launcher_the_nodes_reported_is_used(self):
        plan = resolve(intent(), [topo("2socket_8gpu")])
        assert isinstance(for_plan(plan), OpenMPILauncherAdapter)

    def test_a_single_rank_on_an_unknown_launcher_runs_locally(self):
        from reconciler.placement import LaunchPlan, RankPlacement
        plan = LaunchPlan("lp-x", {"type": ""},
                          (RankPlacement(0, 0, "n1", (), (), (), None, None, None),),
                          {"warnings": ()}, {})
        assert isinstance(for_plan(plan), LocalProcessLauncherAdapter)

    def test_an_unknown_launcher_with_many_ranks_is_refused(self):
        from reconciler.placement import LaunchPlan, RankPlacement
        ranks = tuple(RankPlacement(i, i, "n1", (), (), (), None, None, None)
                      for i in range(2))
        plan = LaunchPlan("lp-x", {"type": "slurm"}, ranks, {"warnings": ()}, {})
        with pytest.raises(LauncherUnsupported, match="no launcher adapter"):
            for_plan(plan)

    def test_every_registered_adapter_declares_the_same_capability_keys(self):
        # A new adapter that forgot a key would read as "false" and quietly
        # refuse work it can actually do, or worse, be assumed capable.
        keys = {frozenset(a.CAPABILITIES) for a in LAUNCHERS.values()}
        assert len(keys) == 1
        assert keys.pop() == {"cpu_binding", "memory_binding", "gpu_visibility",
                              "explicit_rank_map", "mapping_preview", "binding_report"}
