"""End-to-end control plane: submit -> tick -> terminal, plus every failure path.

Runs entirely in-process against a FileStore and a fake provider — the
repeatable form of the dry-run smoke in CLAUDE.md. Nothing here touches
libvirt, ssh, ansible or apptainer.
"""
from __future__ import annotations

import pytest
from conftest import FakeAdapter, assert_pool_returned, drive, states

from reconciler.reconciler import clear_jobs
from reconciler.states import IllegalTransition, JobState, NodeState

PROMOTED = JobState.PROMOTED.value
FAILED = JobState.FAILED.value
AVAILABLE = NodeState.AVAILABLE.value


class TestHappyPath:
    def test_single_node_job_reaches_promoted(self, store, seed, make_reconciler, spec):
        seed(cpu=2)
        rec, fake = make_reconciler()
        job_id = rec.submit(spec())
        drive(rec)

        job = store.get_job(job_id)
        assert job.state == PROMOTED
        assert job.error is None
        assert len(job.assigned_nodes) == 1
        assert job.run["exit_code"] == 0
        assert job.drop_path and job.drop_path.endswith(job_id)
        assert fake.phases_for(job_id) == ["provision", "bootstrap", "run", "collect",
                                           "deprovision"]
        assert_pool_returned(store)

    def test_audit_records_the_whole_chain(self, store, seed, make_reconciler, spec):
        seed(cpu=1)
        rec, _ = make_reconciler()
        job_id = rec.submit(spec())
        drive(rec)
        assert [(e["from"], e["to"]) for e in store.list_audit(job_id)] == [
            ("-", "submitted"),
            ("submitted", "provisioning"),
            ("provisioning", "bootstrapping"),
            ("bootstrapping", "running"),
            ("running", "collecting"),
            ("collecting", "teardown"),
            ("teardown", "validating"),
            ("validating", "promoted"),
        ]

    def test_dry_run_job_completes(self, store, seed, make_reconciler, spec):
        # image: "" is the documented control-plane exercise.
        seed(cpu=1)
        rec, _ = make_reconciler()
        job_id = rec.submit(spec(image=""))
        drive(rec)
        assert store.get_job(job_id).state == PROMOTED

    def test_terminal_jobs_are_not_advanced_again(self, store, seed, make_reconciler, spec):
        seed(cpu=1)
        rec, _ = make_reconciler()
        rec.submit(spec())
        drive(rec)
        assert rec.tick() == 0


class TestRouting:
    def test_mpi_job_runs_once_across_every_claimed_node(self, store, seed, make_reconciler,
                                                         spec):
        seed(cpu=4)
        rec, fake = make_reconciler()
        job_id = rec.submit(spec(launcher="mpi", node_count=3))
        drive(rec)

        assert len(store.get_job(job_id).assigned_nodes) == 3
        runs = [c for c in fake.calls if c[0] == "run" and c[1] == job_id]
        assert len(runs) == 1, "mpi must be ONE launch from the head, not one per node"
        assert len(runs[0][2]) == 3

    def test_gpu_job_lands_on_the_gpu_node(self, store, seed, make_reconciler, spec):
        seed(cpu=2, gpu=True)
        rec, _ = make_reconciler()
        job_id = rec.submit(spec(gpu=True))
        drive(rec)
        assert store.get_job(job_id).assigned_nodes == ["cluster-host"]

    def test_cpu_job_never_takes_the_gpu_node(self, store, seed, make_reconciler, spec):
        seed(cpu=1, gpu=True)
        rec, _ = make_reconciler()
        job_id = rec.submit(spec(node_count=1))
        drive(rec)
        assert store.get_job(job_id).assigned_nodes == ["cluster-node-01"]

    def test_hybrid_job_drives_from_the_gpu_node(self, store, seed, make_reconciler, spec):
        seed(cpu=3, gpu=True)
        rec, _ = make_reconciler()
        job_id = rec.submit(spec(launcher="mpi", node_count=3, hybrid=True))
        drive(rec)
        job = store.get_job(job_id)
        assert job.state == PROMOTED
        assert job.assigned_nodes[0] == "cluster-host"


class TestFailurePaths:
    def test_hybrid_without_mpi_is_rejected_before_claiming(self, store, seed, make_reconciler,
                                                            spec):
        seed(cpu=2, gpu=True)
        rec, _ = make_reconciler()
        job_id = rec.submit(spec(hybrid=True))        # launcher defaults to "single"
        drive(rec)

        job = store.get_job(job_id)
        assert job.state == FAILED and "hybrid" in job.error
        assert job.assigned_nodes == [], "a rejected shape must not hold capacity"
        assert_pool_returned(store)

    def test_unsupported_launch_intent_is_refused_before_claiming(
            self, store, seed, make_reconciler, spec):
        # Planning is not implemented yet, so a valid intent must still stop the
        # job — running it would apply a placement the caller never asked for,
        # and the output would look entirely normal afterwards.
        import json
        from pathlib import Path

        from conftest import REPO_DIR
        intent = json.loads((Path(REPO_DIR) / "contract" / "launch-intent" / "v1" /
                             "examples" / "one-rank-per-gpu.json").read_text())
        seed(cpu=2)
        rec, fake = make_reconciler()
        job_id = rec.submit(spec(launch=intent))
        drive(rec)

        job = store.get_job(job_id)
        assert job.state == FAILED
        assert "cannot resolve a launch plan yet" in job.error
        assert job.assigned_nodes == [], "a job we cannot run must not hold capacity"
        assert fake.phases_for(job_id) == [], "nothing should have been provisioned"
        assert_pool_returned(store)

    def test_malformed_launch_intent_fails_the_job_with_the_reason(
            self, store, seed, make_reconciler, spec):
        seed(cpu=2)
        rec, _ = make_reconciler()
        job_id = rec.submit(spec(launch={"schema_version": "launch-intent/v99"}))
        drive(rec)

        job = store.get_job(job_id)
        assert job.state == FAILED
        assert "launch-intent/v99" in job.error

    def test_nonzero_container_exit_fails_the_job(self, store, seed, make_reconciler, spec):
        seed(cpu=1)
        rec, _ = make_reconciler(FakeAdapter(exit_code=3))
        job_id = rec.submit(spec())
        drive(rec)

        job = store.get_job(job_id)
        assert job.state == FAILED and "exited 3" in job.error
        # The run summary survives the failure — the exit code is the evidence.
        assert job.run["exit_code"] == 3
        assert_pool_returned(store)

    def test_phase_failure_deprovisions_healthy_nodes_rather_than_quarantining(
            self, store, seed, make_reconciler, spec):
        # Nothing implicated the nodes themselves, so they go back to the pool —
        # but via deprovision, since releasing the claim alone would orphan a
        # running VM that no job owns and nothing later tears down.
        seed(cpu=2)
        rec, fake = make_reconciler(FakeAdapter(fail_on={"bootstrap"}))
        job_id = rec.submit(spec(launcher="mpi", node_count=2))
        drive(rec)

        job = store.get_job(job_id)
        assert job.state == FAILED and "injected bootstrap failure" in job.error
        assert all(n.state == AVAILABLE for n in store.list_nodes())
        assert "deprovision" in fake.phases_for(job_id)

    def test_teardown_error_does_not_mask_the_real_failure(self, store, seed, make_reconciler,
                                                           spec):
        seed(cpu=1)
        rec, _ = make_reconciler(FakeAdapter(fail_on={"bootstrap", "deprovision"}))
        job_id = rec.submit(spec())
        drive(rec)
        # The reason on record is the bootstrap failure, not the cleanup noise.
        assert "injected bootstrap failure" in store.get_job(job_id).error

    def test_node_failure_quarantines_it_and_fails_its_job(self, store, seed, make_reconciler,
                                                           spec):
        seed(cpu=2)
        rec, _ = make_reconciler()
        job_id = rec.submit(spec(launcher="mpi", node_count=2))
        rec.tick()                                    # claim + provision
        rec.handle_node_failure("cluster-node-01", reason="host died")

        job = store.get_job(job_id)
        assert job.state == FAILED and "host died" in job.error
        assert store.get_node("cluster-node-01").state == NodeState.QUARANTINED.value
        # the innocent sibling goes back to the pool
        assert store.get_node("cluster-node-02").state == AVAILABLE

    @pytest.mark.xfail(strict=True, raises=IllegalTransition, reason=(
        "BUG: an idle node cannot be quarantined. handle_node_failure has an explicit "
        "'No active owner -> quarantine the node directly' branch, but NODE_TRANSITIONS "
        "allows AVAILABLE -> CLAIMED only, so that branch always raises. Consequence: "
        "`cluster fail-node --inject` on an unowned node hard-kills the VM first and THEN "
        "raises, leaving a destroyed machine registered as available for the scheduler to "
        "hand out. Fix is one edge in states.py; strict=True so this flips to a failure "
        "the moment it is fixed and the marker must come off."))
    def test_node_failure_without_an_owner_only_quarantines(self, store, seed, make_reconciler):
        seed(cpu=1)
        rec, _ = make_reconciler()
        rec.handle_node_failure("cluster-node-01")
        assert store.get_node("cluster-node-01").state == NodeState.QUARANTINED.value

    def test_unknown_node_failure_is_survivable(self, store, seed, make_reconciler):
        seed(cpu=1)
        rec, _ = make_reconciler()
        rec.handle_node_failure("ghost-node")         # must not raise


class TestCapacity:
    def test_job_waits_instead_of_failing(self, store, seed, make_reconciler, spec):
        # Contention is not a job defect: the job stays SUBMITTED and retries.
        seed(cpu=1)
        rec, _ = make_reconciler()
        job_id = rec.submit(spec(node_count=4))
        assert rec.tick() == 0
        assert store.get_job(job_id).state == JobState.SUBMITTED.value

    def test_wait_times_out_cleanly(self, store, seed, make_reconciler, spec):
        # A request the pool can never satisfy must not wait forever.
        seed(cpu=1)
        rec, _ = make_reconciler(capacity_wait_timeout=-1)
        job_id = rec.submit(spec(node_count=4))
        rec.tick()
        job = store.get_job(job_id)
        assert job.state == FAILED and "waiting for capacity" in job.error

    def test_waiting_job_proceeds_once_capacity_frees(self, store, seed, make_reconciler, spec):
        seed(cpu=2)
        rec, _ = make_reconciler()
        first = rec.submit(spec(launcher="mpi", node_count=2))
        second = rec.submit(spec())
        drive(rec)
        assert states(store) == {first: PROMOTED, second: PROMOTED}
        assert_pool_returned(store)


class TestClearJobs:
    def test_refuses_while_a_job_is_active(self, store, seed, make_reconciler, spec):
        seed(cpu=2)
        rec, _ = make_reconciler()
        job_id = rec.submit(spec())
        rec.tick()
        result = clear_jobs(store)
        assert result["cleared"] is False
        assert [a["job_id"] for a in result["active"]] == [job_id]
        assert store.get_job(job_id) is not None

    def test_force_clears_and_resets_the_pool(self, store, seed, make_reconciler, spec):
        seed(cpu=2)
        rec, _ = make_reconciler()
        rec.submit(spec())
        rec.tick()
        result = clear_jobs(store, force=True)
        assert result["cleared"] and result["forced_active"] == 1
        assert store.list_jobs() == []
        assert all(n.state == AVAILABLE and n.owner_job is None for n in store.list_nodes())

    def test_leaves_the_pool_topology_intact(self, store, seed, make_reconciler):
        seed(cpu=3, gpu=True)
        clear_jobs(store)
        assert len(store.list_nodes()) == 4
