# CLAUDE.md — Outpost

Guidance for working in this repo. Read `docs/` for the full walkthrough; this
file is the dense quick-reference.

## What this repo is

A small, generic compute cluster. It **receives any container job, runs it,
and returns any output** — nothing
about what the job actually does lives here. Resources it manages: **8
libvirt/KVM VMs** (CPU work) + **the host machine's GPU** (GPU work, as a pool
node). The interface (JobSpec in → drop-zone out) is deliberately open —
anything can drive it: a script, a CI job, a bigger scheduler built later, or
just the `cluster` CLI in this repo (a local test client of that same
interface, nothing more).

Apptainer on VMs/host is the only execution backend today. KinD, Slinky/Slurm,
and multi-node distribution are planned for later.

Runs on any Linux host with libvirt/KVM (`/dev/kvm`, nested/hardware
virtualization enabled) and, optionally, an NVIDIA GPU + driver for GPU jobs
(CPU-only jobs work without one). Defaults in `infra/libvirt/config.env`
(8 VMs × 2 vCPU/3 GB) are just a launch config for a modest single dev host —
tune node count/size to fit yours. Git repo (`main` branch).

**Status: live-verified end to end.** Real containerized single-node jobs, real
multi-node MPI jobs (hybrid `mpirun` + `apptainer exec` launch), and real GPU
jobs have all run through the full reconciler on the actual 8-VM + host-GPU
cluster — provisioned, bootstrapped, scheduled under contention, executed, and
collected. See `docs/04-job-lifecycle.md` and `docs/06-interface-contract.md`.

## Golden rules

- **Nothing is pulled automatically.** A job brings its own container image;
  the cluster only runs what it's handed. The optional pre-staged runner
  image (`cluster_pull_runner_image`) stays off unless you explicitly turn it
  on — all pull switches default off.
- **Network gate:** `CLUSTER_ALLOW_NETWORK=0` blocks base-image download + guest
  first-boot apt until explicitly set to `1` (or confirmed interactively).
- **No separate executor layer** (deliberate). Running a container is what an
  adapter does once it owns a node. Flexibility comes from the generic JobSpec.
- Work **inside the nix dev shell** (`nix develop`); all tooling is pinned there.
- Reconciler defaults to **dry-run** (`NullAdapter`); `--execute` uses real adapters.
- A job that can't get capacity **waits and retries** (not an immediate failure)
  up to `CLUSTER_CAPACITY_WAIT_TIMEOUT` (default 600s), then fails cleanly. Jobs
  advance **concurrently** within a tick (thread pool, `CLUSTER_RECONCILE_WORKERS`,
  default 8) — a slow job's phase never blocks other jobs' progress.

## Layout

```
flake.nix, shell.nix        pinned dev shell (qemu, virsh, ansible, apptainer, python…)
Makefile                    convenience targets (see below)
infra/libvirt/              make VMs: config.env (SoT), lib.sh (incl. wait_ssh), net/vm/cluster scripts
infra/ansible/              configure nodes: bootstrap role (apptainer|docker|mpi|ssh-fabric…)
reconciler/                 the control plane (Python package)
  states.py                 job/node state machines (transition tables)
  models.py                 JobSpec (incl. launcher), JobRecord, NodeRecord, RunResult
  store.py                  FileStore (default, fcntl shared/exclusive locked) | MongoStore
  registry.py               NodeRegistry: claim/release/quarantine (exclusive locks)
  adapter.py                ProviderAdapter ABC + Libvirt/LocalHost/Null; container_argv(),
                             mpirun_argv(), run_logged()/append_job_log() (replay logs)
  reconciler.py             the driver: submit, concurrent tick, phases, wait/timeout, failure
  cli.py                    cluster commands (incl. logs / reconciler-log)
demo/                       mpi_demo.c/.def/.sif, run-mpi-demo.sh (bare), run-scheduling-demo.sh
viz/                        dashboard.py + cluster.html — node stats + job queue + log viewer
docs/                       01-architecture … 06-interface-contract
bin/cluster             CLI launcher (python -m reconciler)
job.example.yaml            sample generic JobSpec (dry-run)
job.mpi.example.yaml        multi-node MPI JobSpec (node_count: 4)
job.mpi.small.example.yaml  multi-node MPI JobSpec (node_count: 2, for contention demos)
job.gpu.example.yaml        GPU JobSpec (nvidia-smi via apptainer --nv)
```

## Commands

```bash
nix develop                      # enter the pinned shell

# infrastructure plane (operator-driven)
make net-up | template | cluster-up | status | bootstrap | cluster-down | net-down
make fail N=3                    # inject a node failure

# live dashboard (per-VM CPU/RAM + host GPU + job queue + per-job/cross-job
# replay logs, one page) — open http://localhost:8087
make dashboard

# control plane (job-driven; dry-run unless --execute)
make seed-nodes                  # register 8 VMs + cluster-host GPU node
cluster nodes                # pool + capabilities (cpu / gpu-local)
cluster submit --spec j.yaml # -> job-id  (job.example.yaml | job.mpi.example.yaml |
                                  #             job.mpi.small.example.yaml | job.gpu.example.yaml)
cluster reconcile [--once] [--execute] [--interval N]
cluster list | status <id> | result <id>
cluster logs <id> [--tail N]  # full per-job replay transcript (every command + output)
cluster reconciler-log        # cross-job chronological narration, all jobs interleaved
cluster fail-node <name> [--inject] [--reason ...]
cluster clear-jobs [--force]  # demo reset: wipe job/audit history + replay logs,
                                  # reset nodes to available (refuses if jobs are active
                                  # unless --force; also a button on the dashboard)

# one-shot demos
make mpi-sif                     # build demo/mpi_demo.sif (needed once for MPI job specs)
make submit-mpi                  # submit job.mpi.example.yaml
make demo-scheduling             # submit 4 jobs (2x pool-filling MPI, 1 waiting, 1 GPU), tick live
```

## Job flow (state machine)

`submitted → provisioning → bootstrapping → running → collecting → teardown →
validating → promoted` (terminal: `failed`/`rejected`/`cancelled`).
Nodes: `available → claimed → provisioned → ready → busy → draining → available`
(+ `quarantined`). Illegal transitions raise (`states.py`).

**Terminal states, what they actually mean:**
- **`promoted`** — SUCCESS. The job ran to completion with no errors and its
  output was collected to the drop-zone. (A result gets "promoted" once
  accepted. `validate` is currently a stub that promotes unconditionally —
  real gate logic, when it lands, is what would ever turn a `validating` job
  into `rejected` instead.) Safe for a caller to read `drop_path`/`stdout.log`
  and trust it's complete.
- **`failed`** — FAILURE. Something in the pipeline broke: a phase raised
  (ssh/ansible error, non-zero container exit, a node dying mid-job) or the
  job timed out waiting for capacity. `job.error` has the reason. Don't trust
  drop-zone contents as complete for a failed job — collection may not have
  run at all, depending on which phase broke.
- `rejected` — a gate explicitly rejected the output (not reachable today;
  gates are a stub). `cancelled` — an operator cancelled it.

Per phase the reconciler calls `adapter.{provision,bootstrap,run,collect,
deprovision}` (each now also takes `job_id`, for per-job replay logging).
Adapter is chosen per node: `node.local` → `LocalHostAdapter` (apptainer `--nv`
in-process), else `LibvirtAdapter` (ssh+apptainer into the VM).

`JobSpec.launcher` selects the run path: `"single"` (default) runs the
container on one node; `"mpi"` (with `node_count > 1`) builds a per-job
hostfile from exactly the nodes this job claimed, stages the image to each,
and launches real `mpirun` from the head node — a **hybrid** launch where
`mpirun`/`orted` are the host's own OpenMPI and only the per-rank command
(`apptainer exec <image> <cmd>`) runs inside the container. See
`demo/mpi_demo.def` for how the demo container is built to match.

**Jobs advance concurrently**, not one at a time: `tick()` runs each active
job's phase on its own worker thread (capped at `CLUSTER_RECONCILE_WORKERS`,
default 8), and per-job node loops (provisioning/tearing down N nodes) are
themselves parallelized across those N nodes. A 10-20 minute real workload on
one job never blocks any other job's progress.

## Scheduling

`gpu: true` jobs claim GPU nodes (only `cluster-host`); `gpu: false` jobs claim VMs.
Claims are atomic (fcntl / `find_one_and_update`) = exclusive locks. A job that
can't get enough capacity right now **waits and retries every tick** (it is
NOT failed immediately) until either it succeeds or `CLUSTER_CAPACITY_WAIT_TIMEOUT`
(default 600s) elapses since it was submitted, at which point it fails cleanly
with a `"timed out ... waiting for capacity"` reason. A multi-GPU request will
always time out this way — there is one GPU node.

## The open interface (contract with whatever drives it)

- **Intake:** a `JobSpec` doc in `jobs` — `name, image, command, runtime,
  launcher, node_count, gpu, env, output_dir, params`. `image=""` → dry-run.
  `launcher: "mpi"` + `node_count > 1` → real multi-node `mpirun`.
- **Egress:** artifacts land in `${CLUSTER_DROPZONE}/<job_id>/` — `stdout.log`
  (the container's captured stdout+stderr, guaranteed for every adapter/launch
  path) plus whatever the job wrote to `output_dir`; `jobs.drop_path`/`run`
  record where + exit code.
- **Status:** `jobs` (state), `audit` (every transition), `nodes`
  (pool). Everything else is private implementation. See
  `docs/06-interface-contract.md`.
- **Operational observability (not part of the contract, human-facing):**
  `.var/logs/<job_id>.log` (full replay transcript — every command + its live
  output) and `.var/logs/reconciler.log` (cross-job timeline), both readable
  via `cluster logs`/`reconciler-log` or the dashboard (`make dashboard`).

## Current status (real vs stub)

- Real: VM provisioning (libvirt, SSH-readiness verified before "provisioned"),
  bootstrap (ansible + apptainer, scoped per-job via `--limit`), scheduling
  (wait-and-retry + timeout), exclusive locks, concurrent tick, failure→
  quarantine, `run` (single-node AND multi-node MPI, host GPU or VMs), `collect`
  (drop-zone, `stdout.log` guaranteed on every path), audit, replay logging,
  live dashboard (node stats + job queue + log viewer).
- Stub: `validate` gates + the sampler (`nvidia-smi` bracketing) — promote is
  currently unconditional. Wire in when parsers/gates arrive (deliberately left
  as-is for now).

## Testing

No formal test suite yet. Validate changes by:
- `python -m py_compile reconciler/*.py`
- an in-process dry-run smoke (FileStore + Reconciler) — seed nodes → submit →
  tick to promoted; assert routing + locks + failure path. For anything
  touching concurrency, also run a stress test: many jobs (10-15) contending
  for a small node pool with a jittery/sleepy fake adapter, ticked to
  completion, asserting no stuck jobs and no unexpected failures — this is
  what caught two real `FileStore` race conditions (unlocked reads, and
  releasing the write lock before the write was flushed) that only showed up
  under genuine thread concurrency.
- `nix develop` sanity: `apptainer --version`, `cluster seed-nodes`, dry-run
  a job to `promoted`. Clean up `.var/` afterward (it's gitignored).
- Live: `make cluster-up && make bootstrap && make seed-nodes`, then submit a
  real job with `--execute` and confirm it reaches `promoted` with real
  artifacts in the drop-zone. `make demo-scheduling` exercises contention +
  GPU scheduling together in one shot.

## Conventions

- `infra/libvirt/config.env` is the single source of truth for cluster shape;
  bash + python both read it (python shells `lib.sh`). Don't hardcode topology.
- Reconciler logs to **stderr** AND `.var/logs/reconciler.log`; stdout stays
  machine-readable (job-ids). Every job also gets its own full transcript at
  `.var/logs/<job_id>.log` (every command run on its behalf + that command's
  live output — the two logging destinations are independent: stderr/global
  log is a cross-job narration feed, the per-job log is a self-contained
  replay of one job start to finish).
- Generated state/artifacts live under `.var/` (gitignored): `reconciler/`
  (store), `dropzone/` (egress), `logs/` (replay transcripts), `runs/` (host
  scratch — hostfiles, local workdirs), `apptainer-cache/`, `templates/`,
  `overlays/`, `seeds/`.
- **Ansible must run with `cwd=infra/ansible`** (or `cd` there first) — that's
  the only way it auto-discovers `ansible.cfg` (which sets
  `StrictHostKeyChecking=no`/`UserKnownHostsFile=/dev/null` for the cluster0
  fabric). Running it from elsewhere silently falls back to ambient SSH
  defaults and fails host-key verification unpredictably. `adapter.py`'s
  `bootstrap()` passes `cwd=ANSIBLE_DIR` explicitly for this reason.
- **`provision()` blocks until the node is actually SSH-reachable** (see
  `lib.sh:wait_ssh`), not just "virsh start"-ed — cloud-init first boot takes
  ~30-90s, and bootstrapping a node before it's really up fails with a
  misleading SSH error. Per-job node loops (provision/teardown) parallelize
  across a job's own nodes so this doesn't serialize an N-node job's boot time.
- **Ansible bootstrap runs are scoped with `--limit <job's own node names>`**
  — the regenerated inventory always lists every currently-live VM
  cluster-wide, so without `--limit` one job's bootstrap would also touch
  concurrently-provisioning nodes that belong to a different job.
- Apptainer on the VMs is installed from a **pinned, checksum-verified `.deb`**
  downloaded directly from the apptainer GitHub releases (`apptainer.yml`) —
  not a PPA, and not Ubuntu's own archive (which only has the unrelated
  `singularity-container` fork under a different binary name).
