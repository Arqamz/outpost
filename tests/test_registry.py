"""NodeRegistry — all-or-nothing claims and the hybrid ordering the phases rely on."""
from __future__ import annotations

import threading

import pytest

from reconciler.audit import Audit
from reconciler.models import NodeRecord
from reconciler.registry import NoCapacity, NodeRegistry
from reconciler.states import NodeState


@pytest.fixture()
def registry(store):
    return NodeRegistry(store, Audit(store))


def _seed(store, cpu: int = 4, gpu: bool = False) -> None:
    for i in range(1, cpu + 1):
        store.put_node(NodeRecord(node_id=f"n{i}", name=f"n{i}", index=i, ip=f"10.0.0.{i}"))
    if gpu:
        store.put_node(NodeRecord(node_id="host", name="host", index=0, ip="10.0.0.100",
                                  gpu=True, local=True))


class TestClaim:
    def test_claims_the_requested_count(self, store, registry):
        _seed(store, cpu=4)
        assert len(registry.claim("job-a", 3)) == 3
        assert sum(n.state == NodeState.CLAIMED.value for n in store.list_nodes()) == 3

    def test_partial_claim_is_rolled_back(self, store, registry):
        # All-or-nothing: two nodes briefly claimed for a 3-node job must go back,
        # or the pool leaks capacity to a job that never ran.
        _seed(store, cpu=2)
        with pytest.raises(NoCapacity):
            registry.claim("job-a", 3)
        assert all(n.state == NodeState.AVAILABLE.value and n.owner_job is None
                   for n in store.list_nodes())

    def test_no_capacity_names_what_was_wanted(self, store, registry):
        _seed(store, cpu=1)
        with pytest.raises(NoCapacity, match="2 CPU node"):
            registry.claim("job-a", 2)

    def test_gpu_claim_takes_only_gpu_nodes(self, store, registry):
        _seed(store, cpu=2, gpu=True)
        assert [n.node_id for n in registry.claim("job-a", 1, require_gpu=True)] == ["host"]

    def test_hybrid_puts_the_gpu_node_first(self, store, registry):
        # The phases pick the driving adapter from nodes[0]; if the ordering
        # slipped, a hybrid job would try to launch its mpirun from a VM.
        _seed(store, cpu=3, gpu=True)
        claimed = registry.claim("job-a", 3, hybrid=True)
        assert claimed[0].node_id == "host" and claimed[0].gpu
        assert [n.gpu for n in claimed[1:]] == [False, False]

    def test_hybrid_without_a_gpu_node_rolls_back(self, store, registry):
        _seed(store, cpu=3, gpu=False)
        with pytest.raises(NoCapacity, match="1 GPU"):
            registry.claim("job-a", 2, hybrid=True)
        assert all(n.state == NodeState.AVAILABLE.value for n in store.list_nodes())


class TestCompetingMultiNodeClaims:
    """A multi-node claim is all-or-nothing but NOT atomic: it acquires one node
    at a time, in no particular order, with no reservation. Two such claims can
    each take a node the other needs, both fail, and both roll back — forever."""

    @pytest.mark.xfail(strict=True, reason=(
        "BUG (livelock): concurrent multi-node claims mutually starve. Each job takes "
        "one node, neither can complete, both roll back, and the next attempt repeats "
        "it. Observed end to end: 4 jobs x 3 nodes against a 4-node pool were still "
        "SUBMITTED after 400 ticks with all 4 nodes free and unowned — so in production "
        "run_forever spins until CLUSTER_CAPACITY_WAIT_TIMEOUT and fails EVERY one of "
        "them with 'timed out waiting for capacity', despite the pool being able to "
        "satisfy any one of them serially. Needs an ordering rule or a reservation so "
        "one claimant wins; rollback alone does not break the tie. Interleaving here is "
        "forced with a barrier, so this is deterministic, not a race."))
    def test_one_competing_claim_should_win(self, store, registry):
        _seed(store, cpu=2)
        barrier = threading.Barrier(2, timeout=5)
        acquire = store.claim_node

        def claim_then_wait(job_id, require_gpu=False):
            # Hold at the barrier after the FIRST node so both jobs are holding one
            # before either asks for its second — the interleaving that starves.
            node = acquire(job_id, require_gpu=require_gpu)
            if node is not None and not getattr(threading.current_thread(), "paused", False):
                threading.current_thread().paused = True
                barrier.wait()
            return node

        store.claim_node = claim_then_wait
        outcomes: list[object] = []
        lock = threading.Lock()

        def worker(job_id: str) -> None:
            try:
                result: object = registry.claim(job_id, 2)
            except NoCapacity as e:
                result = e
            with lock:
                outcomes.append(result)

        threads = [threading.Thread(target=worker, args=(f"job-{k}",)) for k in ("a", "b")]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=10)

        winners = [o for o in outcomes if isinstance(o, list)]
        assert len(winners) == 1, (
            f"neither job made progress: {outcomes}; pool is "
            f"{[(n.node_id, n.state) for n in store.list_nodes()]}")

    def test_rollback_leaves_no_node_stranded(self, store, registry):
        # The half of the behaviour that IS correct, and worth keeping green: a
        # failed multi-node claim never leaks capacity, even though it also never
        # breaks the tie above.
        _seed(store, cpu=2)
        store.claim_node("other-job")
        with pytest.raises(NoCapacity):
            registry.claim("job-a", 2)
        owners = {n.owner_job for n in store.list_nodes()}
        assert owners == {"other-job", None}


class TestReleaseAndQuarantine:
    def test_release_clears_ownership(self, store, registry):
        _seed(store, cpu=1)
        registry.claim("job-a", 1)
        registry.release("n1")
        node = store.get_node("n1")
        assert (node.state, node.owner_job) == (NodeState.AVAILABLE.value, None)

    def test_quarantine_is_idempotent(self, store, registry):
        _seed(store, cpu=1)
        registry.claim("job-a", 1)
        registry.quarantine("n1", "died")
        registry.quarantine("n1", "died again")     # must not raise on the illegal re-entry
        assert store.get_node("n1").state == NodeState.QUARANTINED.value

    def test_unknown_node_is_a_no_op_not_a_crash(self, store, registry):
        registry.release("ghost")
        registry.quarantine("ghost", "x")
        registry.advance("ghost", NodeState.READY)

    def test_advance_walks_the_lifecycle(self, store, registry):
        _seed(store, cpu=1)
        registry.claim("job-a", 1)
        for state in (NodeState.PROVISIONED, NodeState.READY, NodeState.BUSY,
                      NodeState.DRAINING):
            registry.advance("n1", state)
        assert store.get_node("n1").state == NodeState.DRAINING.value

    def test_every_transition_is_audited(self, store, registry):
        _seed(store, cpu=1)
        registry.claim("job-a", 1)
        registry.release("n1")
        transitions = [(e["from"], e["to"]) for e in store.list_audit("n1")]
        assert (NodeState.AVAILABLE.value, NodeState.CLAIMED.value) in transitions
        assert (NodeState.CLAIMED.value, NodeState.AVAILABLE.value) in transitions
