"""The transition tables are the source of truth; these guard them structurally.

Every assertion here is DERIVED from the tables rather than restating a list of
states, so a state added later is covered automatically instead of quietly
sitting outside the checks.
"""
from __future__ import annotations

import pytest

from reconciler.states import (
    JOB_NEXT,
    JOB_TERMINAL,
    JOB_TRANSITIONS,
    NODE_TERMINAL,
    NODE_TRANSITIONS,
    IllegalTransition,
    JobState,
    NodeState,
    assert_job,
    assert_node,
    can,
)


class TestJobTable:
    def test_every_state_has_an_entry(self):
        assert set(JOB_TRANSITIONS) == set(JobState)

    def test_terminal_states_are_sinks(self):
        for state in JOB_TERMINAL:
            assert JOB_TRANSITIONS[state] == set(), f"{state} is terminal but has outgoing edges"

    def test_every_active_state_can_fail(self):
        # states.py encodes FAILED from every active state explicitly (for
        # auditability) rather than as a wildcard — so it is exactly the kind of
        # thing a newly added phase forgets, stranding a job with no way out.
        for state in set(JobState) - JOB_TERMINAL:
            assert JobState.FAILED in JOB_TRANSITIONS[state], state

    def test_happy_path_successors_are_legal_transitions(self):
        for frm, to in JOB_NEXT.items():
            assert can(JOB_TRANSITIONS, frm, to), f"JOB_NEXT says {frm} -> {to}, table forbids it"

    def test_every_active_state_has_a_happy_path_successor(self):
        # An active state missing from JOB_NEXT is a state a job can reach and
        # never leave.
        assert set(JOB_NEXT) == set(JobState) - JOB_TERMINAL

    def test_happy_path_terminates_at_promoted(self):
        seen, state = [], JobState.SUBMITTED
        while state in JOB_NEXT:
            state = JOB_NEXT[state]
            assert state not in seen, f"cycle in the happy path at {state}"
            seen.append(state)
        assert state is JobState.PROMOTED

    def test_illegal_transition_raises(self):
        with pytest.raises(IllegalTransition):
            assert_job(JobState.SUBMITTED, JobState.RUNNING)

    def test_legal_transition_passes(self):
        assert_job(JobState.SUBMITTED, JobState.PROVISIONING)


class TestNodeTable:
    def test_every_state_has_an_entry(self):
        assert set(NODE_TRANSITIONS) == set(NodeState)

    def test_every_held_state_can_return_to_the_pool(self):
        # Releasing a node is legal whenever we hold it (states.py); a held state
        # without that edge is capacity that can never come back.
        held = set(NodeState) - {NodeState.AVAILABLE} - NODE_TERMINAL
        for state in held:
            assert NodeState.AVAILABLE in NODE_TRANSITIONS[state], state

    def test_quarantine_is_reachable_from_every_held_state(self):
        held = set(NodeState) - {NodeState.AVAILABLE} - NODE_TERMINAL
        for state in held:
            assert NodeState.QUARANTINED in NODE_TRANSITIONS[state], state

    def test_quarantine_only_clears_to_available(self):
        assert NODE_TRANSITIONS[NodeState.QUARANTINED] == {NodeState.AVAILABLE}

    def test_available_is_only_claimable(self):
        # An available node must not jump straight to busy/ready without a claim,
        # or two jobs could use it at once.
        assert NODE_TRANSITIONS[NodeState.AVAILABLE] == {NodeState.CLAIMED}

    def test_illegal_transition_raises(self):
        with pytest.raises(IllegalTransition):
            assert_node(NodeState.AVAILABLE, NodeState.BUSY)
