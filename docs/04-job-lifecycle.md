# 4 · Job lifecycle

## End to end

```
caller: cluster submit --spec job.yaml
          └─> JobRecord{state: submitted} written to jobs   (the intake)

reconciler: cluster reconcile   (each tick advances one phase per job,
                                     jobs advance CONCURRENTLY — see 03-control-plane.md)
  submitted     → PROVISIONING   registry.claim(N, require_gpu=spec.gpu)  [exclusive locks]
                                  · NoCapacity -> job stays SUBMITTED, retried next tick,
                                    until CLUSTER_CAPACITY_WAIT_TIMEOUT (default 600s) elapses
                                    -> then FAILED with a clear timeout reason
                                  adapter.provision() per node, in parallel across nodes
                                  (blocks until each node is genuinely SSH-reachable)
  provisioning  → BOOTSTRAPPING   adapter.bootstrap()  (gen-inventory + bootstrap role,
                                  scoped to THIS job's nodes via --limit)
  bootstrapping → RUNNING         adapter.run(nodes, job_id, spec) -> RunResult
                                     · localhost: apptainer exec --nv <image> <cmd>
                                     · libvirt, launcher=single: ssh <vm> apptainer exec ...
                                     · libvirt, launcher=mpi (node_count>1): per-job hostfile,
                                       image staged to every claimed node, real `mpirun`
                                       launched from the head (hybrid: host mpirun, per-rank
                                       apptainer exec)
                                  non-zero exit -> clean job FAILED
  running       → COLLECTING      adapter.collect() -> ${CLUSTER_DROPZONE}/<job_id>/   (the egress)
                                  stdout.log guaranteed here on every path (single-node, MPI,
                                  GPU) — teed into the workdir at run time so collect() ships it
  collecting    → TEARDOWN        adapter.deprovision() per node, in parallel; release to pool
  teardown      → VALIDATING      (gates — stub for now)
  validating    → PROMOTED        done; every transition in audit

node dies mid-job (inject-failure / fail-node):
  handle_node_failure → quarantine that node + FAIL the job cleanly,
                        release healthy peers back to the pool
```

## Dry-run vs execute

The phases are identical in both modes; the *only* difference is which adapter
is wired (`reconciler.py` constructor):

- **dry-run (default):** `NullAdapter` for every node — provision/bootstrap/run/
  collect all just log. The full state machine + scheduling + audit runs with
  nothing launched. Use it to validate control-plane logic.
- **execute (`reconcile --execute`):** `LocalHostAdapter` for the host node,
  `LibvirtAdapter` for VMs. Now provisioning really defines VMs and `run` really
  launches containers.

## What's real vs. stubbed today

| Phase | Status |
|-------|--------|
| provision / bootstrap / teardown | real (libvirt + ansible), SSH-readiness verified, parallelized across a job's nodes |
| **run** | real — apptainer on host (`--nv`), single-node ssh+apptainer on a VM, or real multi-node `mpirun` across claimed VMs |
| **collect** | real — copies/scp's the job's `output_dir` **and** `stdout.log` to the drop-zone (guaranteed on every adapter/launch path) |
| validate (gates) / sampler | **stub** — promotes cleanly; wire in when parsers/gates exist |

## Observability

Every job gets a full replay transcript (every command run on its behalf +
that command's live output) at `.var/logs/<job_id>.log`, readable via
`cluster logs <job_id>`. `.var/logs/reconciler.log` is the cross-job
chronological narration (`cluster reconciler-log`) — useful for seeing how
concurrent jobs interleaved. `make dashboard` (http://localhost:8087) shows
node stats, the live job queue, and a click-through log viewer for both, all
on one page — it reads the same store and log files the CLI does, so it can
never drift from what actually happened.

## Two ways to drive the VMs (don't be surprised)

- **Operator plane** (`make cluster-up`): brings up *all* VMs at once for manual
  bring-up/debugging.
- **Job plane** (`reconcile --execute`): the reconciler provisions *only the
  nodes a job claims*, on demand. This is the production model.

They're alternative entry points to the same VMs. For a real job-driven run you
typically don't pre-`cluster-up`; the reconciler does it per job.
