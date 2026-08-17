"""LibvirtAdapter.run()'s planned-vs-plain dispatch gate.

Intra-node GPU affinity (e.g. one_per_rank across all 8 GPUs on a single
claimed node) has exactly one node but many ranks: a resolved plan must take
the planned rankfile/appfile path even when len(nodes) == 1, or the affinity
placement would be silently discarded. StaticSshAdapter inherits run()
unchanged, so this covers it too.
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
    def test_single_node_with_a_plan_takes_the_planned_mpi_path(self, adapter, monkeypatch):
        calls = []
        monkeypatch.setattr(adapter, "_run_mpi",
                            lambda nodes, job_id, spec, remote_workdir, plan: calls.append(plan))
        adapter.run([node()], "job-1", spec(image="docker://x"), plan={"ranks": []})
        assert calls == [{"ranks": []}]

    def test_single_node_with_no_plan_falls_to_the_plain_path(self, adapter, monkeypatch):
        mpi_called, plain_called = [], []
        monkeypatch.setattr(adapter, "_run_mpi", lambda *a, **kw: mpi_called.append(True))
        monkeypatch.setattr("reconciler.adapter.run_logged",
                            lambda *a, **kw: plain_called.append(True) or 0)
        adapter.run([node()], "job-1", spec(image="docker://x"), plan=None)
        assert mpi_called == []
        assert plain_called == [True]

    def test_multi_node_with_no_plan_still_takes_the_mpi_path(self, adapter, monkeypatch):
        calls = []
        monkeypatch.setattr(adapter, "_run_mpi",
                            lambda nodes, job_id, spec, remote_workdir, plan: calls.append(plan))
        adapter.run([node("n1"), node("n2")], "job-1", spec(image="docker://x", node_count=2),
                   plan=None)
        assert calls == [None]
