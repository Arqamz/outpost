# 6 · The open interface

This is the **only** thing anything driving this cluster needs to know —
whether that's a shell script, a CI job, or a bigger scheduler built later.
Everything else — adapters, apptainer, scheduling, the state machine — is
private to the cluster and can change freely as long as this contract holds.

The contract has two halves: **intake** (send a job in) and **egress** (get
output out), plus a **status/audit** read side.

## Intake — how a job is sent in

Write a **JobSpec** into the `jobs` collection of the store. The `JobSpec`
is intentionally generic — the cluster runs *any* container:

| Field | Type | Meaning |
|-------|------|---------|
| `name` | str | human label |
| `image` | str | container to run: a `.sif` path or `docker://…` ref. Empty = dry-run. |
| `command` | list[str] | argv inside the container |
| `runtime` | str | `apptainer` (default) or `docker` |
| `launcher` | str | `"single"` (default) — one node, or `"mpi"` — real multi-node `mpirun` across the `node_count` claimed nodes (ignored if `node_count == 1`) |
| `node_count` | int | how many nodes to claim |
| `gpu` | bool | `true` → schedule on a GPU node (the host); `false` → VMs |
| `hybrid` | bool | `true` → claim 1 GPU node (the host) **+** `node_count-1` CPU VMs and launch ONE `mpirun` spanning them (requires `launcher: mpi`, `node_count >= 2`) — the only job shape that mixes the two pools |
| `env` | map | env vars set inside the container |
| `output_dir` | str | in-container path the job writes results to (bound to the host) |
| `params` | map | opaque passthrough (ignored by the cluster) |
| `launch` | map, optional | a semantic placement request (`launch-intent/v1`, see [`contract/README.md`](../contract/README.md)) — resolved against real topology for `launcher: mpi` jobs only; see "Placement" below. Absent = today's behavior, unchanged. |

**The cluster promises:** given a JobSpec, it schedules the right resource
(waiting for capacity to free up if needed, not failing immediately — see
below), runs the container (single-node or, for `launcher: mpi`, a real
multi-node run), captures its `output_dir` + stdout/stderr + exit code, and
moves the job to a terminal state. It makes **no assumptions** about what the
container does — nothing job-specific lives here. That matters for tools
that *print* their results rather than only writing files: a caller can parse
`stdout.log` into its own schema afterward, no guessing required.

**Capacity waiting.** If the pool can't satisfy a request right now, the job
does **not** fail immediately — it sits in `submitted` and is retried every
tick until capacity frees up, or `CLUSTER_CAPACITY_WAIT_TIMEOUT` seconds (default
600) have passed since submission, at which point it fails cleanly with
`error` containing `"timed out ... waiting for capacity"`. A caller polling
`jobs` should treat "still submitted" as normal, not stuck — check
`error` once the job reaches a terminal state to distinguish a real failure
from a capacity timeout.

Two ways to write the spec today:
- our local CLI: `cluster submit --spec job.yaml` (a client of this same store);
- directly: insert the JobSpec document into `jobs` — what any other caller
  would do, via `CLUSTER_MONGO_URI`.

## Placement — declaring and verifying where ranks actually ran

A `launcher: mpi` `JobSpec` may carry an optional `launch` block (`contract/
launch-intent/v1/launch-intent.schema.json`) — a semantic statement of the
placement a benchmark needs (ranks per node, cores per rank, GPU binding),
with no CPU ids, device indices, or launcher flags (those depend on the
allocation, which the caller cannot know at submit time). Any other launcher
shape (`launcher: single`, `backend: k8s`) refuses a job carrying one rather
than running it under a placement nobody chose.

When present, the job passes through two additional states between
`bootstrapping` and `running`:

- **`planning`** — the cluster probes the allocated nodes' real topology and
  resolves the intent into an exact per-rank plan (`jobs.plan`), or fails the
  job with the reason if the allocation cannot satisfy it.
- **`plan_ready`** — the plan is resolved. `jobs.plan_status` is one of
  `none | ready | approved | failed` — `ready` means a human must run
  `gtl approve` (or the equivalent direct store write) before the job may
  proceed; `approved` (including auto-approval, the schema default) means it
  already has. A job with no `launch` block skips both states entirely — its
  audit trail is unaffected.

Once the job runs, three additional files land in the drop-zone alongside
`stdout.log`: `launch-intent.yaml` (what was asked for), `launch-plan.yaml`
(the exact resolved per-rank mapping), and `launch-receipt.yaml` (the
resolved plan reconciled against the launcher's own `--report-bindings`
claim and, when `validation.require_preflight` is set, a real in-process
observation from inside each rank) — `receipt.status` is `verified`,
`mismatched`, or `unverified`.

The k8s backend has a separate, narrower placement concept — see
["Declaring and verifying per-rank GPU-memory affinity"](08-kubernetes-backend.md#declaring-and-verifying-per-rank-gpu-memory-affinity)
— since it has no cores/rankfile to resolve against (the kubelet owns in-pod
CPU/GPU) and exactly one physical GPU (no device to pin to).

## Egress — how output comes back

When a job reaches `collecting`, the cluster writes everything the job produced
to the **drop-zone**:

```
${CLUSTER_DROPZONE}/<job_id>/
    stdout.log          # captured stdout+stderr of the container
    <whatever the job wrote to output_dir>
```

**`stdout.log` is guaranteed here on every code path** — single-node on the
host GPU, single-node on a VM, and multi-node MPI (where it's the combined
output of every rank, since `mpirun` forwards each rank's stdout to the
launching process). This is the field to parse for tools that report results
by printing them rather than only writing a file to `output_dir` — don't
assume you need to guess which one a given job uses; both are always
populated (empty/absent if the job wrote nothing there).

The job record also records `drop_path` and a `run` summary (`exit_code`, the
node it ran on). Read it with `cluster result <job_id>` or by reading the
`jobs` document. A different backend could swap the drop-zone for an S3
prefix or similar behind this same shape — the contract doesn't change.

**Not part of the contract, but useful while integrating:** `cluster logs
<job_id>` (or the dashboard) shows the *entire* operational transcript for a
job — every command the cluster ran on its behalf and that command's live
output, which is a superset of what lands in the drop-zone and useful for
debugging a job that didn't produce the output you expected. This is a human
debugging surface, not something a caller should parse programmatically —
the drop-zone + `jobs`/`audit`/`nodes` remain the only programmatic contract.

## Status / audit — read side

- `jobs` — every job with its current `state`, `assigned_nodes`, `run`,
  `drop_path`, `error`, and (only when `launch` was set) `plan`/`plan_status` —
  see "Placement" above.
- `audit` — append-only, ordered list of every state transition (job + node)
  with timestamps and reasons. This is the source of truth for "what happened."
- `nodes` — the resource pool with capabilities and lock ownership.

## What is NOT part of the contract

Adapters, the apptainer command line, the libvirt scripts, the state-machine
internals, dry-run vs execute. Those are implementation. Anything driving
this cluster should depend only on: **the JobSpec schema, the drop-zone
layout, and the three collections above.**
