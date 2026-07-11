"""Job + node state machines and their legal transitions.

Kept declarative and dependency-free so the transition tables are the single
source of truth and are trivially unit-testable.
"""
from __future__ import annotations
from enum import Enum


class JobState(str, Enum):
    SUBMITTED = "submitted"
    PROVISIONING = "provisioning"
    BOOTSTRAPPING = "bootstrapping"
    RUNNING = "running"
    COLLECTING = "collecting"
    TEARDOWN = "teardown"
    VALIDATING = "validating"
    PROMOTED = "promoted"      # terminal: fingerprint(s) accepted
    REJECTED = "rejected"      # terminal: gate rejected the candidate
    FAILED = "failed"          # terminal: clean failure (incl. node failure)
    CANCELLED = "cancelled"    # terminal: operator cancelled


class NodeState(str, Enum):
    AVAILABLE = "available"        # in the pool, unclaimed
    CLAIMED = "claimed"           # exclusively locked by a job
    PROVISIONED = "provisioned"   # VM defined + started
    READY = "ready"               # bootstrap succeeded
    BUSY = "busy"                 # running a job
    DRAINING = "draining"         # job done, tearing down
    QUARANTINED = "quarantined"   # failed; excluded until manually cleared


JOB_TERMINAL = {JobState.PROMOTED, JobState.REJECTED, JobState.FAILED, JobState.CANCELLED}
NODE_TERMINAL = {NodeState.QUARANTINED}

# Legal job transitions. FAILED is reachable from every active state (node/phase
# failure); encoded explicitly for auditability rather than a wildcard.
JOB_TRANSITIONS: dict[JobState, set[JobState]] = {
    # FAILED here covers a capacity-wait timeout — the job never got as far as
    # claiming a node, so it can fail straight out of SUBMITTED.
    JobState.SUBMITTED:     {JobState.PROVISIONING, JobState.FAILED, JobState.CANCELLED},
    JobState.PROVISIONING:  {JobState.BOOTSTRAPPING, JobState.FAILED, JobState.CANCELLED},
    JobState.BOOTSTRAPPING: {JobState.RUNNING, JobState.FAILED, JobState.CANCELLED},
    JobState.RUNNING:       {JobState.COLLECTING, JobState.FAILED, JobState.CANCELLED},
    JobState.COLLECTING:    {JobState.TEARDOWN, JobState.FAILED, JobState.CANCELLED},
    JobState.TEARDOWN:      {JobState.VALIDATING, JobState.FAILED},
    JobState.VALIDATING:    {JobState.PROMOTED, JobState.REJECTED, JobState.FAILED},
    JobState.PROMOTED:      set(),
    JobState.REJECTED:      set(),
    JobState.FAILED:        set(),
    JobState.CANCELLED:     set(),
}

# Happy-path successor the reconciler drives toward at each active state.
JOB_NEXT: dict[JobState, JobState] = {
    JobState.SUBMITTED:     JobState.PROVISIONING,
    JobState.PROVISIONING:  JobState.BOOTSTRAPPING,
    JobState.BOOTSTRAPPING: JobState.RUNNING,
    JobState.RUNNING:       JobState.COLLECTING,
    JobState.COLLECTING:    JobState.TEARDOWN,
    JobState.TEARDOWN:      JobState.VALIDATING,
    JobState.VALIDATING:    JobState.PROMOTED,  # unless a gate rejects
}

# AVAILABLE is reachable from every *held* state: releasing a node back to the
# pool (the adapter deprovisions the VM as part of it) is legal whenever we hold
# it — the happy path still drains BUSY->DRAINING->AVAILABLE, the direct edges
# exist for abort/failure cleanup.
NODE_TRANSITIONS: dict[NodeState, set[NodeState]] = {
    NodeState.AVAILABLE:    {NodeState.CLAIMED},
    NodeState.CLAIMED:      {NodeState.PROVISIONED, NodeState.QUARANTINED, NodeState.AVAILABLE},
    NodeState.PROVISIONED:  {NodeState.READY, NodeState.QUARANTINED, NodeState.AVAILABLE},
    NodeState.READY:        {NodeState.BUSY, NodeState.DRAINING, NodeState.QUARANTINED, NodeState.AVAILABLE},
    NodeState.BUSY:         {NodeState.DRAINING, NodeState.QUARANTINED, NodeState.AVAILABLE},
    NodeState.DRAINING:     {NodeState.AVAILABLE, NodeState.QUARANTINED},
    NodeState.QUARANTINED:  {NodeState.AVAILABLE},  # only via manual clear
}


class IllegalTransition(Exception):
    pass


def can(transitions, frm, to) -> bool:
    return to in transitions.get(frm, set())


def assert_job(frm: JobState, to: JobState) -> None:
    if not can(JOB_TRANSITIONS, frm, to):
        raise IllegalTransition(f"job: {frm.value} -/-> {to.value}")


def assert_node(frm: NodeState, to: NodeState) -> None:
    if not can(NODE_TRANSITIONS, frm, to):
        raise IllegalTransition(f"node: {frm.value} -/-> {to.value}")
