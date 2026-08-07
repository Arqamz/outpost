# Outpost — documentation

Outpost is a generic compute cluster: it accepts *any* container job, runs it
on the resources it manages (local KVM VMs + the host GPU), and returns *any*
output it produces. The interface is deliberately open — JobSpec in,
drop-zone out — so anything can drive it, not just a single blessed caller.

Two execution backends exist: **apptainer** (on libvirt VMs, the host GPU, or
static-ssh workers — including remote GPU boxes) and **Kubernetes + KAI/HAMi**
(GPU sliced into a gang of pods). One control plane can span multiple machines
and both backends at once (see doc 9). Slinky/Slurm is still planned — the
reconciler/adapter split means adding one won't touch the JobSpec contract.

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
| 8 | [08-kubernetes-backend.md](08-kubernetes-backend.md) | the Kubernetes backend (HAMi + KAI): `backend: k8s` runs a job as a gang of N pods sharing one GPU, VRAM-capped per rank, with pod-DNS rendezvous — design, honest limits, quick start |
| 9 | [09-multi-machine-cluster.md](09-multi-machine-cluster.md) | one control plane across two machines + all fabrics: VM control plane, the PC joined as a static-ssh GPU worker AND its own k8s cluster, per-cluster slot targeting |
| 10 | [10-ssh-gateway.md](10-ssh-gateway.md) | `ssh tashkil` — a forced-command front door that puts the open interface (JobSpec in over stdin, artifacts out over a stdout tar) behind one ssh alias, job-portal only (no admin), with a client wrapper |

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
