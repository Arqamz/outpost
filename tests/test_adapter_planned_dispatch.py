"""LibvirtAdapter.run()'s planned-vs-plain dispatch gate.

Intra-node GPU affinity (e.g. one_per_rank across all 8 GPUs on a single
claimed node) has exactly one node but many ranks: a resolved plan must take
the planned rankfile/appfile path even when len(nodes) == 1, or the affinity
placement would be silently discarded. A launcher: single job with a resolved
plan (one node, one rank — HPL's own internal mpirun) takes the same planned
path: a plan present is now sufficient on its own, regardless of launcher.
StaticSshAdapter inherits run() unchanged, so this covers it too.
"""
from __future__ import annotations

import pytest
from reconciler.adapter import LibvirtAdapter
from reconciler.models import JobSpec, NodeRecord


def node(name="n1") -> NodeRecord:
    return NodeRecord(node_id=name, name=name, index=1, ip="10.0.0.1")


def spec(**kw) -> JobSpec:
    base = dict(name="affinity-smoke", image="", command=["true"], launcher="mpi")
    return JobSpec(**{**base, **kw})


@pytest.fixture()
def adapter():
    return LibvirtAdapter(log=lambda msg: None)


class TestPlannedDispatch:
    def test_single_node_with_a_plan_takes_the_planned_launch_path(self, adapter, monkeypatch):
        calls = []
        monkeypatch.setattr(adapter, "_run_launch",
                            lambda nodes, job_id, spec, remote_workdir, plan: calls.append(plan))
        adapter.run([node()], "job-1", spec(image="docker://x"), plan={"ranks": []})
        assert calls == [{"ranks": []}]

    def test_single_node_with_no_plan_falls_to_the_plain_path(self, adapter, monkeypatch):
        launch_called, plain_called = [], []
        monkeypatch.setattr(adapter, "_run_launch", lambda *a, **kw: launch_called.append(True))
        # The plain path stages the SIF itself now, which is a real ssh.
        monkeypatch.setattr(adapter, "_ensure_remote_sif", lambda *a, **kw: "/tmp/cached.sif")
        monkeypatch.setattr("reconciler.adapter.run_logged",
                            lambda *a, **kw: plain_called.append(True) or 0)
        adapter.run([node()], "job-1", spec(image="docker://x"), plan=None)
        assert launch_called == []
        assert plain_called == [True]

    def test_multi_node_with_no_plan_still_takes_the_launch_path(self, adapter, monkeypatch):
        calls = []
        monkeypatch.setattr(adapter, "_run_launch",
                            lambda nodes, job_id, spec, remote_workdir, plan: calls.append(plan))
        adapter.run([node("n1"), node("n2")], "job-1", spec(image="docker://x", node_count=2),
                   plan=None)
        assert calls == [None]

    def test_single_node_single_launcher_with_a_plan_takes_the_planned_path(
            self, adapter, monkeypatch):
        # Proves the widened `plan is not None or (...)` condition fires
        # regardless of launcher — a launcher: single job (HPL's own internal
        # mpirun, one container) with a resolved plan must not fall through
        # to the plain unplanned path.
        calls = []
        monkeypatch.setattr(adapter, "_run_launch",
                            lambda nodes, job_id, spec, remote_workdir, plan: calls.append(plan))
        adapter.run([node()], "job-1", spec(image="docker://x", launcher="single"),
                   plan={"ranks": []})
        assert calls == [{"ranks": []}]
