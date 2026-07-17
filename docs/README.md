# Outpost — documentation

Outpost is a generic compute cluster: it accepts *any* container job, runs it
on the resources it manages (local KVM VMs + the host GPU), and returns *any*
output it produces. The interface is deliberately open — JobSpec in,
drop-zone out — so anything can drive it, not just a single blessed caller.

Apptainer on VMs is the only backend today. KinD, Slinky/Slurm, and
multi-node distribution are planned for later — the reconciler/adapter split
means adding one won't touch the JobSpec contract.

Read in order:

| # | Doc | What it covers |
|---|-----|----------------|
| 1 | [01-architecture.md](01-architecture.md) | the mental model: layers, axes, the cluster's boundary |
| 2 | [02-infrastructure.md](02-infrastructure.md) | `infra/libvirt` (make VMs) + `infra/ansible` (configure them) |
| 3 | [03-control-plane.md](03-control-plane.md) | the reconciler: state machine, store, registry, adapters |
| 4 | [04-job-lifecycle.md](04-job-lifecycle.md) | how a job flows submit→promote; dry-run vs execute |
| 5 | [05-gpu-and-apptainer.md](05-gpu-and-apptainer.md) | the host GPU node + running containers with apptainer |
| 6 | [06-interface-contract.md](06-interface-contract.md) | the open interface anything driving this cluster targets (intake + egress) |
| 7 | [07-ubuntu-setup.md](07-ubuntu-setup.md) | running the whole thing on a native Ubuntu host (no Nix): install → cluster up → dashboard → jobs, incl. host-only and hybrid (VM + host GPU) runbooks |

Quickstart lives in the top-level [../README.md](../README.md); day-to-day
command reference lives in [../CLAUDE.md](../CLAUDE.md).

## One-paragraph summary

A **reconciler** (in `reconciler/`) reads generic `JobSpec`s from a store,
schedules each onto the right resource (**GPU jobs → the host node**, **CPU jobs
→ VMs**, waiting-and-retrying rather than failing immediately if the pool is
full) using exclusive locks, and drives it — **concurrently across jobs** —
through a state machine: `provision → bootstrap → run → collect → teardown →
validate → promote`. *How* a node is created and *how* its container runs are
both owned by a **ProviderAdapter** (`libvirt` for VMs — single-node or real
multi-node `mpirun` — `localhost` for the host GPU); there is deliberately
**no separate executor layer**. Job artifacts (including stdout, guaranteed on
every path) land in a **drop-zone**; that plus the `JobSpec` schema is the
entire contract with the outside world. A live dashboard and per-job replay
logs make the whole thing observable while you're integrating against it.
