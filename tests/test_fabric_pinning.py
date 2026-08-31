"""LibvirtAdapter._run_planned's cross-node MPI fabric pinning.

`job-1e25ee9612ff` (OSU, 2 real ranks across aws-gpu-1/aws-gpu-2) staged and
launched cleanly, then both ranks segfaulted inside MPI_Init's UCX/UCC
transport setup — on a shared box also running a kubelet/CNI workload, UCX
picked up a pod-network interface and advertised an address the peer rank
could never reach. Two gaps caused it: the planned path never applied the
classic TCP BTL/OOB `if_include` pinning the unplanned path already had, and
that pinning doesn't reach UCX at all (it needs a device name, not an
IP/subnet). This covers both fixes without touching real ssh/apptainer.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from reconciler.adapter import LibvirtAdapter
from reconciler.launcher import LauncherRendering
from reconciler.models import JobSpec, NodeRecord
from reconciler.placement import LaunchPlan, RankPlacement


def node(name: str, ip: str) -> NodeRecord:
    return NodeRecord(node_id=name, name=name, index=1, ip=ip, provider="static-ssh",
                      gpu=True)


def spec(**kw) -> JobSpec:
    base = dict(name="osu-affinity", image="docker://x@sha256:" + "a" * 64,
               command=["true"], launcher="mpi", node_count=2, gpu=True)
    return JobSpec(**{**base, **kw})


def two_rank_plan() -> LaunchPlan:
    ranks = (
        RankPlacement(global_rank=0, local_rank=0, node="aws-gpu-1",
                      cpu_ids=(0, 1, 2, 3), cpu_slots=((0, 0), (0, 1), (0, 2), (0, 3)),
                      numa_nodes=(0,), gpu_uuids=("GPU-aaa",), gpu_pci_bus_ids=("0000:00:1e.0",),
                      visible_gpu_indices=(0,)),
        RankPlacement(global_rank=1, local_rank=0, node="aws-gpu-2",
                      cpu_ids=(0, 1, 2, 3), cpu_slots=((0, 0), (0, 1), (0, 2), (0, 3)),
                      numa_nodes=(0,), gpu_uuids=("GPU-bbb",), gpu_pci_bus_ids=("0000:00:1e.0",),
                      visible_gpu_indices=(0,)),
    )
    return LaunchPlan(plan_id="lp-test", launcher={"type": "openmpi"}, ranks=ranks,
                      validation={}, digests={})


@pytest.fixture()
def adapter():
    return LibvirtAdapter(log=lambda msg: None)


def _wire_common_mocks(monkeypatch, adapter, compile_calls):
    """Stub every _run_planned collaborator EXCEPT the compile() call itself,
    which records its kwargs into compile_calls and returns a minimal
    rendering — that's the one thing this test cares about."""
    def fake_compile(plan, spec_, *, node_ips, workdir, image, **kw):
        compile_calls.append(kw)
        return LauncherRendering(launcher="openmpi", version="", argv=["true"],
                                 files={}, per_rank_env={r.global_rank: {} for r in plan.ranks})

    class FakeLauncherAdapter:
        def unsupported(self, plan, intent):
            return []
        compile = staticmethod(fake_compile)

    monkeypatch.setattr("reconciler.launcher.for_plan", lambda *a, **kw: FakeLauncherAdapter())
    monkeypatch.setattr("reconciler.receipt.build_receipt",
                        lambda *a, **kw: MagicMock(to_dict=lambda: {}))
    monkeypatch.setattr(adapter, "_scp_to", lambda *a, **kw: None)
    monkeypatch.setattr(adapter, "_stage_launch_artifacts", lambda *a, **kw: None)
    monkeypatch.setattr("reconciler.adapter.run_logged", lambda *a, **kw: 0)

    def fake_subprocess_run(argv, **kw):
        cmd = argv[-1] if argv else ""
        if "ip -o -4 addr show" in cmd:
            return MagicMock(returncode=0, stdout="ens5\n")
        return MagicMock(returncode=0, stdout="")  # the post-run `cat stdout_log` fetch
    monkeypatch.setattr("reconciler.adapter.subprocess.run", fake_subprocess_run)


class TestCrossNodeFabricPinning:
    def test_two_nodes_pins_both_btl_and_ucx(self, adapter, monkeypatch):
        nodes = [node("aws-gpu-1", "172.31.87.50"), node("aws-gpu-2", "172.31.94.28")]
        compile_calls: list[dict] = []
        _wire_common_mocks(monkeypatch, adapter, compile_calls)

        adapter._run_planned(nodes, "job-1", spec(), "/tmp/cluster/job-1",
                             "/tmp/outpost-sif-cache/x.sif", two_rank_plan().to_dict())

        assert len(compile_calls) == 1
        kw = compile_calls[0]
        assert kw["mca"]["btl_tcp_if_include"] == "172.31.87.0/24"
        assert kw["mca"]["oob_tcp_if_include"] == "172.31.87.50/32,172.31.94.28/32"
        assert kw["node_ifaces"] == {"aws-gpu-1": "ens5", "aws-gpu-2": "ens5"}

    def test_single_node_gets_no_fabric_pinning(self, adapter, monkeypatch):
        # No cross-node communication to protect — matches launcher: single's
        # one-container shape (HPL/NCCL today), which never hits UCX cross-node
        # address exchange at all. Must stay a no-op: pinning here would be
        # pure overhead, and calling _node_iface unconditionally would add an
        # ssh round-trip to every job, not just cross-node MPI ones.
        nodes = [node("aws-gpu-1", "172.31.87.50")]
        plan = LaunchPlan(plan_id="lp-test", launcher={"type": "openmpi"},
                          ranks=(two_rank_plan().ranks[0],), validation={}, digests={})
        compile_calls: list[dict] = []
        _wire_common_mocks(monkeypatch, adapter, compile_calls)

        adapter._run_planned(nodes, "job-1", spec(node_count=1), "/tmp/cluster/job-1",
                             "/tmp/outpost-sif-cache/x.sif", plan.to_dict())

        assert len(compile_calls) == 1
        assert "mca" not in compile_calls[0]
        assert "node_ifaces" not in compile_calls[0]
