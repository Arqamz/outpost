# 3 · Control plane (`reconciler/`)

The brain. Turns "resources exist" into "jobs get scheduled, run, and audited."

| Module | Responsibility |
|--------|----------------|
| `states.py` | the two state machines as **transition tables** (the safety rail). |
| `models.py` | `JobSpec` (incl. `launcher`), `JobRecord`, `NodeRecord`, `RunResult` + Mongo collection names. |
| `store.py` | persistence: `FileStore` (default, `fcntl` **shared/exclusive**-locked JSON) or `MongoStore` (`CLUSTER_MONGO_URI`). |
| `registry.py` | `NodeRegistry` — the pool with exclusive locks; raises `NoCapacity` (not a failure — see below) when a claim can't be satisfied right now. |
| `adapter.py` | `ProviderAdapter` interface + `Libvirt` / `LocalHost` / `Null` impls; `container_argv()`, `mpirun_argv()`; `run_logged()`/`append_job_log()` (per-job replay transcripts). |
| `audit.py` | append-only log of every transition → `audit`. |
| `reconciler.py` | the driver: `submit`, **concurrent** `tick`, the phase handlers, wait-and-retry-with-timeout, failure handling. |
| `cli.py` | `cluster` — local test client of the interface; also `logs`/`reconciler-log`. |

## State machines (`states.py`)

**Job:** `submitted → provisioning → bootstrapping → running → collecting →
teardown → validating → promoted`, plus terminal `failed` / `rejected` /
`cancelled`. **Node:** `available → claimed → provisioned → ready → busy →
draining → available`, plus `quarantined`. Illegal transitions raise — a bug can
never silently corrupt state.

`submitted → failed` is also legal (not just `submitted → provisioning`): a job
that times out waiting for capacity (see below) never got as far as claiming a
node, so it fails straight out of `submitted` rather than needing to pass
through `provisioning` first.

## Nodes, capabilities, scheduling

`NodeRecord` carries capability tags:

- `gpu: bool` — has a usable GPU (only the host does).
- `local: bool` — the control-plane host itself (never libvirt-provisioned).
- `runtime` — container runtime available (apptainer).

Scheduling is capability-aware. `store.claim_node(job_id, require_gpu)` atomically
grabs an `available` node **matching the GPU requirement**: a `gpu:true` job can
only claim GPU nodes; a CPU job can only claim non-GPU nodes. That single rule
routes **GPU jobs to the host** and **CPU jobs to VMs**, and (because there's one
GPU node) a multi-GPU request fails cleanly with `NoCapacity`.

The claim is **atomic** (fcntl lock in FileStore; `find_one_and_update` in Mongo),
so two jobs can never grab the same node — that's the exclusive lock.

**Capacity contention waits, it doesn't fail.** If `claim()` can't find enough
matching nodes right now, it raises `NoCapacity`. The reconciler catches this
specifically (not the generic "any phase error is a clean failure" path): the
job is left untouched in `submitted` and retried on every subsequent tick,
until either it succeeds or `CLUSTER_CAPACITY_WAIT_TIMEOUT` seconds (default 600)
have elapsed since it was submitted — at which point it fails cleanly with a
`"timed out ... waiting for capacity"` reason. This is why `_phase_provision`
claims *before* transitioning state: if the claim fails, the job must stay
exactly as it was, not half-provisioned with no assigned nodes.

## Adapters (`adapter.py`) — provisioning **and** running

The interface is five methods, all taking `job_id` now (used to route
per-command output into that job's replay log — see below):

```python
class ProviderAdapter(ABC):
    def provision(self, node, job_id): ...          # make the node exist AND be usable
    def deprovision(self, node, job_id): ...         # destroy it
    def bootstrap(self, nodes, job_id): ...          # configure it
    def run(self, nodes, job_id, spec): ...          # run the container -> RunResult
    def collect(self, nodes, job_id, spec, dest): ...  # gather output -> dest
```

Three implementations:

- **`LocalHostAdapter`** (`node.local`): provision/deprovision are no-ops (the
  host already exists); `run` executes `apptainer exec --nv …` in-process on the
  host GPU; `collect` copies the run's workdir into the drop-zone.
- **`LibvirtAdapter`** (VMs): `provision` shells to `vm-define.sh`, which
  blocks until the VM is genuinely SSH-reachable before returning; `bootstrap`
  regenerates the inventory and runs the ansible role scoped to just this
  job's nodes (`--limit`); `run` either ssh's into the head node and runs
  apptainer there (single-node), or — when `spec.launcher == "mpi"` — builds a
  per-job hostfile, stages the image to every claimed node, and launches real
  `mpirun` from the head (`_run_mpi`); `collect` scp's the head node's workdir
  (which includes `stdout.log`, teed there by `run`/`_run_mpi` so it lands in
  the drop-zone the same way on every path) back to the drop-zone.
- **`NullAdapter`**: logs and no-ops. This *is* dry-run.

`container_argv()` is a pure function that builds the apptainer/docker command
for one rank — the one place that knows the runtime flags (`--nv`, `--bind`,
`--env`). `mpirun_argv()` wraps a `container_argv()`-built per-rank command in
`mpirun` (`--map-by node` over the job's own hostfile, MCA options scoped to
the cluster0 subnet).

The reconciler picks the adapter **per node** via `_adapter_for(node)` →
`local` vs `remote`. In dry-run both map to `NullAdapter`; with `--execute` they
map to the real ones. That's the only place "dry-run vs real" lives.

## Concurrency

Jobs advance **concurrently within a tick**: `tick()` runs each active job's
`_advance_one()` on its own worker thread from a pool (capped at
`self.max_workers`, default `CLUSTER_RECONCILE_WORKERS=8`). A phase call blocks
for as long as the real command takes — a 10-20 minute `mpirun`/apptainer run
blocks for 10-20 minutes — and without this, that would stall every other
job's progress for the whole tick, since the reconciler is a single process.

Per-job **node loops** are also parallelized: `_phase_provision` provisions a
job's N claimed nodes concurrently (via `Reconciler._parallel_for_each`), and
`_phase_teardown` tears them down the same way — otherwise an N-node job would
pay N sequential rounds of `provision()`'s SSH-readiness wait.

This is safe because the store's writes were already required to be safe under
concurrent *processes* (two separate `cluster` invocations racing a node
claim) — `FileStore` uses `fcntl` locks around every read-modify-write
(`_locked()`, exclusive) and every read (`_read()`, shared), with the write
path flushing to disk **before** releasing the lock. Multiple threads in one
process are no stricter a requirement than that already was; getting the lock
discipline right for real (a stress test — 15 jobs contending for 8 nodes with
a jittery fake adapter — is what caught two real races: unlocked reads, and
releasing the write lock before the write was actually flushed) is what makes
threading safe here, not something special about threads vs. processes.

## Replay logging

Every job gets a self-contained transcript at `.var/logs/<job_id>.log`:
every state transition (written by `_set_job`) and every command run on that
job's behalf, interleaved chronologically, including the command's own live
output (`run_logged()` streams a subprocess's combined stdout/stderr to the
terminal AND appends it to the job's log — `append_job_log()` is the shared
sink both use). There's also one cross-job file, `.var/logs/reconciler.log`,
that every `self.log(...)` narration line lands in regardless of which job it
concerns — a single chronological view across concurrently-running jobs.
Read either via `cluster logs <job_id>` / `cluster reconciler-log`, or
the dashboard's log viewer.
