"""The reconcile loop: drive submitted jobs through the state machine.

`tick()` advances every actionable job by exactly one phase and returns how many
it moved (0 == quiescent). Each phase transition is guarded by states.py and
audit-logged. With execute=False (default) phases log their intent but touch no
infrastructure — the skeleton runs end-to-end with nothing launched.

Jobs are advanced CONCURRENTLY within a tick (one worker thread per job, capped
at CLUSTER_RECONCILE_WORKERS): a phase call blocks for as long as the real command
takes (a 10-20 minute apptainer/mpirun workload blocks for 10-20 minutes), and
without concurrency that would stall every other job in the pool behind it for
the whole tick. The node registry's claim/release and the store's writes are
already safe under concurrent callers (FileStore serializes via flock, Mongo via
find_one_and_update) — that's what a multi-process CLI already required, so
multiple threads in one process are no stricter a requirement.
"""
from __future__ import annotations
import concurrent.futures
import os
import sys
import time
from datetime import datetime, timezone

from . import launch_models
from .adapter import (ProviderAdapter, NullAdapter, LibvirtAdapter,
                      LocalHostAdapter, StaticSshAdapter, REPO_ROOT, LOGS_DIR,
                      append_job_log)
from .audit import Audit
from .models import JobRecord, JobSpec, NodeRecord
from .registry import NodeRegistry, NoCapacity  # NoCapacity: a submitted job waits, it doesn't fail
from .states import JobState, NodeState, JOB_TERMINAL, assert_job
from .store import Store, now_iso

# Egress half of the open interface: collected artifacts land under here per job.
DROPZONE = os.environ.get("CLUSTER_DROPZONE") or os.path.join(REPO_ROOT, ".var", "dropzone")

# How long a job may sit retrying NoCapacity before it's failed cleanly instead
# of waiting forever (e.g. node_count bigger than the whole pool will ever have).
CAPACITY_WAIT_TIMEOUT_S = float(os.environ.get("CLUSTER_CAPACITY_WAIT_TIMEOUT", "600"))

# Cap on how many jobs' phases run concurrently in one tick. Threads, not
# processes — each phase is I/O-bound (ssh/scp/subprocess), so this is cheap;
# it exists to bound simultaneous ssh/ansible load against the VM pool, not
# because the threads themselves are expensive.
RECONCILE_WORKERS = int(os.environ.get("CLUSTER_RECONCILE_WORKERS", "8"))

# Every job's narration line also lands here — one chronological, cross-job
# timeline (.var/logs/reconciler.log) alongside each job's own full transcript
# (.var/logs/<job_id>.log, written by adapter.run_logged / append_job_log).
GLOBAL_LOG = os.path.join(LOGS_DIR, "reconciler.log")


def log_stderr(msg: str) -> None:
    """Human/progress output goes to stderr so stdout stays machine-readable
    (e.g. `jid=$(cluster submit ...)` captures only the job-id), AND is
    appended to .var/logs/reconciler.log for replay after the terminal's gone."""
    print(msg, file=sys.stderr)
    try:
        os.makedirs(LOGS_DIR, exist_ok=True)
        with open(GLOBAL_LOG, "a") as f:
            f.write(f"{now_iso()} {msg}\n")
    except OSError:
        pass


class Reconciler:
    def __init__(self, store: Store, execute: bool = False, log=log_stderr,
                 adapters: dict[str, ProviderAdapter] | None = None,
                 capacity_wait_timeout: float = CAPACITY_WAIT_TIMEOUT_S,
                 max_workers: int = RECONCILE_WORKERS):
        self.store = store
        self.audit = Audit(store)
        self.registry = NodeRegistry(store, self.audit)
        self.execute = execute
        self.log = log
        self.capacity_wait_timeout = capacity_wait_timeout
        self.max_workers = max_workers
        if adapters is None:
            if execute:
                # Real adapters, chosen per node by its provider (NodeRecord.adapter_key):
                # the control-plane host, a libvirt VM, or a static/EC2 ssh box.
                adapters = {"local": LocalHostAdapter(log), "libvirt": LibvirtAdapter(log),
                            "static-ssh": StaticSshAdapter(log)}
            else:
                null = NullAdapter(log)          # dry-run: everything no-ops
                adapters = {"local": null, "libvirt": null, "static-ssh": null}
        self.adapters = adapters

    def _adapter_for(self, node: NodeRecord) -> ProviderAdapter:
        """A node's provider (its `adapter_key`) decides which adapter drives it."""
        return self.adapters[node.adapter_key]

    def _parallel_for_each(self, items: list, fn) -> None:
        """Run fn(item) for every item concurrently (capped at self.max_workers)
        and propagate the first exception once ALL items have been attempted —
        used for per-node work within a single job's phase (provision, teardown)
        so an N-node job doesn't pay N sequential rounds of e.g. an SSH-readiness
        wait when the nodes could just as well come up in parallel."""
        if not items:
            return
        workers = min(len(items), self.max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(fn, items))

    def _capacity_wait_seconds(self, job: JobRecord) -> float:
        """How long this job has been sitting SUBMITTED, unable to claim nodes.
        created_at is stamped by the store on first write (submit()), so this
        covers the whole wait — not just since the most recent tick."""
        if not job.created_at:
            return 0.0
        return (datetime.now(timezone.utc) - datetime.fromisoformat(job.created_at)).total_seconds()

    # ── job state helper ──────────────────────────────────────────────────
    def _set_job(self, job: JobRecord, to: JobState, reason: str = "") -> None:
        frm = JobState(job.state)
        assert_job(frm, to)
        job.state = to.value
        self.store.put_job(job)
        self.audit.record("job", job.job_id, frm.value, to.value, reason)
        msg = f"[job {job.job_id}] {frm.value} -> {to.value}  {reason}"
        self.log(msg)
        append_job_log(job.job_id, f"{now_iso()} {msg}")

    # ── entrypoint used by the CLI ────────────────────────────────────────
    def submit(self, spec: JobSpec) -> str:
        job = JobRecord.create(spec)
        self.store.put_job(job)
        self.audit.record("job", job.job_id, "-", job.state, f"submitted: {spec.name}")
        append_job_log(job.job_id, f"{now_iso()} === job {job.job_id} submitted: "
                       f"{spec.name} (launcher={spec.launcher}, node_count={spec.node_count}, "
                       f"gpu={spec.gpu}, image={spec.image or '(dry-run)'}) ===")
        self.log(f"[job {job.job_id}] submitted ({spec.workload}, {spec.node_count} node/s)")
        return job.job_id

    # ── driver ────────────────────────────────────────────────────────────
    def tick(self) -> int:
        """Advance every actionable job by one phase, CONCURRENTLY — one worker
        thread per job (capped at self.max_workers). A slow job's phase call
        (a real 10-20 minute apptainer/mpirun run) blocks only its own thread;
        every other job's phase for this tick proceeds without waiting on it."""
        jobs = [j for j in self.store.list_jobs() if JobState(j.state) not in JOB_TERMINAL]
        if not jobs:
            return 0
        workers = min(len(jobs), self.max_workers)
        with concurrent.futures.ThreadPoolExecutor(max_workers=workers) as pool:
            results = pool.map(self._advance_one, jobs)
            return sum(1 for moved in results if moved)

    def _advance_one(self, job: JobRecord) -> bool:
        """One job's phase call plus its exception handling — the unit of work
        a tick() worker thread runs. Never raises: every outcome (advance,
        wait, timeout, fail) is handled here so pool.map can't abort early."""
        try:
            return self._advance(job)
        except NoCapacity as e:
            # Pool contention, not a job defect — leave the job as-is (still
            # SUBMITTED, since _phase_provision only transitions on success)
            # and retry on the next tick instead of failing it. But a request
            # the pool can never satisfy (or one stuck behind permanent
            # contention) shouldn't wait forever — time it out instead.
            waited = self._capacity_wait_seconds(job)
            if waited > self.capacity_wait_timeout:
                self._fail(job, f"timed out after {waited:.0f}s waiting for capacity: {e}")
                return True
            self.log(f"[job {job.job_id}] waiting for capacity "
                     f"({waited:.0f}s/{self.capacity_wait_timeout:.0f}s): {e}")
            return False
        except Exception as e:  # noqa: BLE001 — any other phase error is a clean job failure
            self._fail(job, f"{type(e).__name__}: {e}")
            return True

    def run_forever(self, interval: float = 2.0) -> None:
        self.log(f"[reconciler] loop start (execute={self.execute}, adapters={sorted(self.adapters)})")
        while True:
            if self.tick() == 0:
                time.sleep(interval)

    def _advance(self, job: JobRecord) -> bool:
        state = JobState(job.state)
        handler = {
            JobState.SUBMITTED:     self._phase_provision,
            JobState.PROVISIONING:  self._phase_bootstrap,
            JobState.BOOTSTRAPPING: self._phase_run,
            JobState.RUNNING:       self._phase_collect,
            JobState.COLLECTING:    self._phase_teardown,
            JobState.TEARDOWN:      self._phase_validate,
            JobState.VALIDATING:    self._phase_promote,
        }.get(state)
        if handler is None:
            return False
        handler(job)
        return True

    # ── phases (each leaves `job` in the next state) ──────────────────────
    # Dry-run vs execute lives entirely in which adapter is wired (Null vs real),
    # so the phases just call the adapter unconditionally.
    def _phase_provision(self, job: JobRecord) -> None:
        spec = JobSpec.from_dict(job.spec)
        # hybrid only makes sense as a multi-node mpirun; anything else would
        # claim mixed pools with no launch path that can span them.
        if spec.hybrid and (spec.launcher != "mpi" or spec.node_count < 2):
            raise ValueError("hybrid: true requires launcher: mpi and node_count >= 2")
        # Refuse a placement request we cannot resolve BEFORE claiming anything —
        # same reasoning as the hybrid check above: a job shape this backend can't
        # execute should never hold capacity while it fails.
        unsupported = launch_models.unsupported_reason(spec.launch)
        if unsupported:
            raise ValueError(unsupported)
        kind = (f"hybrid (1 GPU + {spec.node_count - 1} CPU)" if spec.hybrid
                else ("GPU" if spec.gpu else "CPU"))
        # Claim BEFORE transitioning state: if the pool can't cover this job
        # (NoCapacity), the job must stay untouched in SUBMITTED so tick()'s
        # wait-and-retry path has something to retry — not a half-provisioned
        # job stuck with no assigned_nodes.
        nodes = self.registry.claim(job.job_id, spec.node_count,
                                    require_gpu=spec.gpu, hybrid=spec.hybrid)
        self._set_job(job, JobState.PROVISIONING, f"claimed {spec.node_count} {kind} node(s)")
        job.assigned_nodes = [n.node_id for n in nodes]
        self.store.put_job(job)

        def _provision_one(n: NodeRecord) -> None:
            self._adapter_for(n).provision(n, job.job_id)  # blocks until truly reachable
            self.registry.advance(n.node_id, NodeState.PROVISIONED, "provisioned")

        self._parallel_for_each(nodes, _provision_one)

    def _phase_bootstrap(self, job: JobRecord) -> None:
        self._set_job(job, JobState.BOOTSTRAPPING, "bootstrapping node(s)")
        nodes = [self.store.get_node(nid) for nid in job.assigned_nodes]
        # Group by adapter INSTANCE, not by nodes[0]: a hybrid job's host and VM
        # subsets each need their own bootstrap call (ansible --limit must never
        # see cluster-host — it isn't in the inventory), while dry-run — where
        # both keys are wired to the one NullAdapter — still makes a single call.
        groups: dict[int, tuple[ProviderAdapter, list[NodeRecord]]] = {}
        for n in nodes:
            a = self._adapter_for(n)
            groups.setdefault(id(a), (a, []))[1].append(n)
        for adapter, subset in groups.values():
            adapter.bootstrap(subset, job.job_id)
        for nid in job.assigned_nodes:
            self.registry.advance(nid, NodeState.READY, "bootstrapped")

    def _phase_run(self, job: JobRecord) -> None:
        spec = JobSpec.from_dict(job.spec)
        self._set_job(job, JobState.RUNNING, f"running '{spec.image or 'dry-run'}'")
        nodes = [self.store.get_node(nid) for nid in job.assigned_nodes]
        for nid in job.assigned_nodes:
            self.registry.advance(nid, NodeState.BUSY, "job running")
        result = self._adapter_for(nodes[0]).run(nodes, job.job_id, spec)
        job.run = result.to_dict()
        self.store.put_job(job)
        # A non-zero container exit is a clean job failure (tick() catches it).
        if result.exit_code not in (None, 0):
            raise RuntimeError(f"workload exited {result.exit_code}")

    def _phase_collect(self, job: JobRecord) -> None:
        spec = JobSpec.from_dict(job.spec)
        self._set_job(job, JobState.COLLECTING, "collecting artifacts to drop-zone")
        nodes = [self.store.get_node(nid) for nid in job.assigned_nodes]
        dest = os.path.join(DROPZONE, job.job_id)
        job.drop_path = self._adapter_for(nodes[0]).collect(nodes, job.job_id, spec, dest)
        self.store.put_job(job)

    def _phase_teardown(self, job: JobRecord) -> None:
        self._set_job(job, JobState.TEARDOWN, "tearing down nodes")
        nodes = [self.store.get_node(nid) for nid in job.assigned_nodes]

        def _teardown_one(n: NodeRecord) -> None:
            self.registry.advance(n.node_id, NodeState.DRAINING, "draining")
            self._adapter_for(n).deprovision(n, job.job_id)
            self.registry.release(n.node_id, "returned to pool")

        self._parallel_for_each(nodes, _teardown_one)

    def _phase_validate(self, job: JobRecord) -> None:
        self._set_job(job, JobState.VALIDATING, "running gates")

    def _phase_promote(self, job: JobRecord) -> None:
        # Real gate logic lands with the parsers; skeleton promotes cleanly.
        self._set_job(job, JobState.PROMOTED, "gates passed (skeleton)")

    # ── failure handling ──────────────────────────────────────────────────
    def _fail(self, job: JobRecord, reason: str, bad_nodes: list[str] | None = None) -> None:
        bad = set(bad_nodes or [])
        job.error = reason
        for nid in job.assigned_nodes:
            if nid in bad:
                self.registry.quarantine(nid, f"failed job {job.job_id}: {reason}")
            else:
                # Healthy node not implicated -> tear the VM down and return to
                # pool. Deprovision, not just release: the pool is ephemeral (see
                # _phase_teardown), so releasing the claim WITHOUT destroying the
                # VM orphans a running domain that no job owns and nothing later
                # tears down. deprovision is wrapped so a teardown error on the
                # failure path never masks the job's real failure reason.
                node = self.store.get_node(nid)
                if node and NodeState(node.state) not in (NodeState.AVAILABLE, NodeState.QUARANTINED):
                    try:
                        self._adapter_for(node).deprovision(node, job.job_id)
                    except Exception as e:  # noqa: BLE001 — keep the original failure
                        self.log(f"[reconciler] post-failure deprovision of {nid} "
                                 f"failed (VM may need manual cleanup): {e}")
                    self.registry.release(nid, f"released after job {job.job_id} failure")
        self._set_job(job, JobState.FAILED, reason)

    def handle_node_failure(self, node_id: str, reason: str = "injected failure") -> None:
        """Invoked by the failure-injection harness. Quarantine the node and fail
        its owning job cleanly."""
        node = self.store.get_node(node_id)
        if node is None:
            self.log(f"[reconciler] node {node_id} unknown"); return
        owner = node.owner_job
        if owner:
            job = self.store.get_job(owner)
            if job and JobState(job.state) not in JOB_TERMINAL:
                # _fail quarantines the bad node itself (avoids double-quarantine).
                self._fail(job, f"node {node.name} failed: {reason}", bad_nodes=[node_id])
                return
        # No active owner -> quarantine the node directly.
        self.registry.quarantine(node_id, reason)


def clear_jobs(store: Store, force: bool = False) -> dict:
    """Wipe job history for a clean slate (demo reset) — NOT part of the open
    interface, an operator/demo convenience. Clears jobs + audit, resets
    every node to AVAILABLE (owner_job=None), and deletes every replay log file
    under .var/logs/. The node pool's topology (which nodes exist, their
    capabilities) is untouched — only history and ownership.

    Refuses (cleared=False) if any job is still non-terminal, since deleting an
    in-flight job's record would orphan whatever real infrastructure it's still
    driving (a VM mid-boot, a container mid-run) with no owning record left to
    finish or clean it up. force=True clears anyway — only do this once you've
    confirmed nothing you care about is actually still running (e.g. after a
    `cluster-down`, or when you know the "active" jobs are just stale demo
    leftovers, not real work)."""
    jobs = store.list_jobs()
    active = [j for j in jobs if JobState(j.state) not in JOB_TERMINAL]
    if active and not force:
        return {"cleared": False,
                "active": [{"job_id": j.job_id, "state": j.state} for j in active]}

    nodes_reset = 0
    for n in store.list_nodes():
        if n.state != NodeState.AVAILABLE.value or n.owner_job:
            n.state = NodeState.AVAILABLE.value
            n.owner_job = None
            store.put_node(n)
            nodes_reset += 1

    store.clear_jobs()

    logs_removed = 0
    if os.path.isdir(LOGS_DIR):
        for fn in os.listdir(LOGS_DIR):
            os.remove(os.path.join(LOGS_DIR, fn))
            logs_removed += 1

    return {"cleared": True, "jobs_removed": len(jobs), "logs_removed": logs_removed,
            "nodes_reset": nodes_reset, "forced_active": len(active)}
