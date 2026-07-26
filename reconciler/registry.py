"""NodeRegistry — the pool of nodes with exclusive per-job locks.

`claim` is delegated to the store's atomic primitive so two concurrent jobs can
never grab the same node. `release` and `quarantine` are the inverse paths.
"""
from __future__ import annotations

from .models import NodeRecord
from .states import NodeState, assert_node
from .store import Store
from .audit import Audit


class NoCapacity(Exception):
    pass


class NodeRegistry:
    def __init__(self, store: Store, audit: Audit):
        self.store = store
        self.audit = audit

    def _set_state(self, node: NodeRecord, to: NodeState, reason: str = "", owner: str | None = ...) -> NodeRecord:
        frm = NodeState(node.state)
        assert_node(frm, to)
        node.state = to.value
        if owner is not ...:
            node.owner_job = owner
        self.store.put_node(node)
        self.audit.record("node", node.node_id, frm.value, to.value, reason)
        return node

    def claim(self, job_id: str, count: int, require_gpu: bool = False,
              hybrid: bool = False, backend: str | None = None,
              kube_context: str | None = None) -> list[NodeRecord]:
        """Exclusively lock `count` matching nodes for job_id. All-or-nothing.

        backend set (e.g. "k8s") claims from that backend's pool of slots by
        provider, ignoring require_gpu/hybrid — a k8s job claims one slot and
        does the N-way parallelism inside the pod group, not in the node pool.
        kube_context (with backend="k8s") targets a specific cluster's slots.

        hybrid=True is the one shape that mixes pools: 1 GPU node + (count-1)
        CPU nodes, claimed GPU-first so claimed[0] is always the host — the
        phases rely on that ordering to pick the adapter that drives the job."""
        if backend is not None:
            needs = [None] * count          # provider-matched, gpu flag irrelevant
        elif hybrid:
            needs = [True] + [False] * (count - 1)
        else:
            needs = [require_gpu] * count
        claimed: list[NodeRecord] = []
        for need_gpu in needs:
            # backend claims match by provider (+ optional cluster context);
            # pool claims match by gpu flag.
            n = self.store.claim_node(job_id, require_gpu=bool(need_gpu),
                                      provider=backend,
                                      kube_context=kube_context)  # atomic AVAILABLE -> CLAIMED
            if n is None:
                for c in claimed:               # roll back partial claim
                    self.release(c.node_id, reason=f"rollback claim for {job_id}")
                ctx = f" @ {kube_context}" if kube_context else ""
                kind = (f"{count} {backend}{ctx}" if backend is not None
                        else (f"1 GPU + {count - 1} CPU" if hybrid
                              else f"{count} {'GPU' if require_gpu else 'CPU'}"))
                raise NoCapacity(f"need {kind} node(s), pool exhausted after {len(claimed)}")
            self.audit.record("node", n.node_id, NodeState.AVAILABLE.value,
                              NodeState.CLAIMED.value, f"claimed by {job_id}")
            claimed.append(n)
        return claimed

    def release(self, node_id: str, reason: str = "released") -> None:
        node = self.store.get_node(node_id)
        if node is None:
            return
        # DRAINING/CLAIMED -> AVAILABLE; also clears a quarantine on manual reuse.
        self._set_state(node, NodeState.AVAILABLE, reason, owner=None)

    def quarantine(self, node_id: str, reason: str) -> None:
        node = self.store.get_node(node_id)
        if node is None or NodeState(node.state) == NodeState.QUARANTINED:
            return  # idempotent: already quarantined
        self._set_state(node, NodeState.QUARANTINED, reason)

    def advance(self, node_id: str, to: NodeState, reason: str = "") -> None:
        node = self.store.get_node(node_id)
        if node is None:
            return
        self._set_state(node, to, reason)
