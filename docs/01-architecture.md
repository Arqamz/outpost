# 1 · Architecture

## What this repo *is*

A small, generic compute cluster. Its entire job:

1. **RECEIVE** any job — a generic container (`image` + `command`) with resource
   needs (`gpu?`, `node_count`).
2. **RUN** it on the resources it owns (local KVM VMs, and the host machine's GPU).
3. **SEND** back any output the job produced.

That's it. It doesn't ship an orchestrator, or an opinion about who's
calling it — the only thing this repo owes the outside world is a clean
**open interface** (see [06-interface-contract.md](06-interface-contract.md)).
Anything that can write a JobSpec and read a drop-zone can drive it: a shell
script, a CI job, a bigger scheduler you build later, or just the CLI below.
Everything else — how VMs are built, how containers are launched, how nodes
are scheduled — is private implementation, free to change.

> The `cluster` **CLI in this repo** is a *local test client* of that same
> interface — one possible caller among any number of others, not a
> privileged one.

Apptainer on libvirt VMs (plus the host itself for GPU work) is the only
execution backend today. KinD, Slinky/Slurm, and distributing jobs across
multiple nodes are planned for later — see "the key invariant" below for how
another backend slots in without touching the reconciler.

## Three layers

```
┌─ reconciler/ ─────────── BRAIN   receive jobs, schedule, drive the state machine
├─ infra/ ──────────────── HANDS   libvirt = make VMs · ansible = configure them
└─ flake.nix / shell.nix ─ BENCH   pinned, reproducible tooling (incl. apptainer)
```

## The axes (what varies vs. what stays fixed)

Four things could vary independently; here two of them collapse into one:

| Axis | Choice here | Where it lives |
|------|-------------|----------------|
| **Adapter** — how you get a node **and run on it** | `libvirt` (VMs) + `localhost` (host GPU) | `reconciler/adapter.py` |
| **Fabric** — how nodes talk | plain TCP (today) | `bootstrap` role |
| **Sampler** — how you measure | host `nvidia-smi` (later) | TODO, host node |
| ~~Executor~~ | *folded into Adapter* | — |

**Why no separate Executor layer.** An "executor" would abstract *how work runs
on a node* so an orchestrator could swap it. But the only consumer of that
choice here is the cluster itself, and it has exactly one way to run a job:
reach the node, `apptainer exec` the container. So "how to run" is just what the
adapter does once it owns the node — `localhost` runs it in-process with `--nv`;
`libvirt` ssh's into the VM (or, for a multi-node MPI job, launches real
`mpirun` from the head node — still just what the adapter does, not a
separate layer). The flexibility to run *any* job comes from the **generic
JobSpec** (`launcher: single | mpi` included), not from swappable executors.

## The key invariant

The **reconciler never knows which adapter it is driving**. It calls
`adapter.provision / bootstrap / run / collect / deprovision`; the adapter is
chosen per node by a single flag (`node.local`). Add another backend later —
real cloud VMs, KinD, Slinky, whatever — by adding a class implementing the
same five methods. Nothing in the state machine changes.
