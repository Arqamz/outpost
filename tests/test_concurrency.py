"""The stress case: many jobs contending for a small pool, ticked to completion.

This is the shape that matters most in this repo. `tick()` advances every
actionable job on its own worker thread, and per-job node loops fan out again
inside that — so the store and the registry are under genuine multi-threaded
read-modify-write pressure. Two real FileStore races (unlocked reads, and
releasing the write lock before the write was flushed) were found this way and
by nothing else; a sequential test passes straight through both.

The fake adapter's jitter is what creates the interleavings. It is seeded, so a
failure here is reproducible rather than a coin flip.

SCOPE NOTE: every multi-node case here keeps concurrent demand within the pool.
Oversubscribed multi-node demand livelocks — see
test_registry.TestCompetingMultiNodeClaims, where it is pinned deterministically
rather than left to surface as a flaky timeout in this file.
"""
from __future__ import annotations

from collections import Counter

from conftest import FakeAdapter, active, assert_pool_returned, drive

from reconciler.states import JobState, NodeState

PROMOTED = JobState.PROMOTED.value


class TestContention:
    def test_twelve_jobs_over_four_nodes_all_complete(self, store, seed, make_reconciler, spec):
        seed(cpu=4)
        rec, _ = make_reconciler(FakeAdapter(jitter=0.01))
        job_ids = [rec.submit(spec(name=f"j{i}")) for i in range(12)]
        drive(rec, max_ticks=400)

        final = {j.job_id: j.state for j in store.list_jobs()}
        stuck = {jid: state for jid, state in final.items() if state != PROMOTED}
        assert not stuck, f"jobs did not complete: {stuck}"
        assert set(final) == set(job_ids)
        assert_pool_returned(store)

    def test_multi_node_jobs_run_side_by_side(self, store, seed, make_reconciler, spec):
        # Two 2-node jobs against a 4-node pool: both claims can be satisfied at
        # once, so this exercises the concurrent multi-node path without crossing
        # into the oversubscribed case that livelocks.
        seed(cpu=4)
        rec, fake = make_reconciler(FakeAdapter(jitter=0.01, seed=7))
        ids = [rec.submit(spec(name=f"pair{i}", launcher="mpi", node_count=2))
               for i in range(2)]
        drive(rec, max_ticks=400)

        assert {j.job_id: j.state for j in store.list_jobs()} == dict.fromkeys(ids, PROMOTED)
        for job_id in ids:
            runs = [c for c in fake.calls if c[0] == "run" and c[1] == job_id]
            assert len(runs) == 1 and len(runs[0][2]) == 2
        assert_pool_returned(store)

    def test_no_node_is_ever_held_by_two_jobs(self, store, seed, make_reconciler, spec):
        # Checked at every tick boundary, not just at the end: a double-claim that
        # resolved itself before completion would still be a lost exclusive lock.
        seed(cpu=3)
        rec, _ = make_reconciler(FakeAdapter(jitter=0.01, seed=99))
        for i in range(12):
            rec.submit(spec(name=f"j{i}"))

        for _ in range(400):
            held = [n.node_id for n in store.list_nodes() if n.owner_job]
            assert len(held) == len(set(held)), "a node is owned by more than one job"
            for node in store.list_nodes():
                if node.owner_job:
                    owner = store.get_job(node.owner_job)
                    assert owner is not None, f"{node.node_id} owned by a job that does not exist"
            rec.tick()
            if not active(store):
                break
        else:
            raise AssertionError("jobs still active after 400 ticks")

        assert {j.state for j in store.list_jobs()} == {PROMOTED}
        assert_pool_returned(store)

    def test_a_failing_job_does_not_strand_the_others(self, store, seed, make_reconciler, spec):
        seed(cpu=4)
        healthy = FakeAdapter(jitter=0.01, seed=3)
        broken = FakeAdapter(jitter=0.01, seed=4, fail_on={"run"})
        rec, _ = make_reconciler(healthy)
        # Route one node through a broken adapter by changing which provider key
        # it resolves to; every other node keeps the healthy one.
        rec.adapters["static-ssh"] = broken
        node = store.list_nodes()[0]
        node.provider = "static-ssh"
        store.put_node(node)

        for i in range(6):
            rec.submit(spec(name=f"j{i}"))
        drive(rec, max_ticks=400)

        outcomes = Counter(j.state for j in store.list_jobs())
        assert outcomes[PROMOTED] + outcomes[JobState.FAILED.value] == 6
        assert outcomes[JobState.FAILED.value] >= 1, "the broken node never took a job"
        assert all(n.state in (NodeState.AVAILABLE.value, NodeState.QUARANTINED.value)
                   for n in store.list_nodes())
