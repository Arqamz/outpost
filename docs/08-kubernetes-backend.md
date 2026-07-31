# 8 · Kubernetes backend — HAMi + KAI (simulate multi-GPU on one GPU)

> **Status: IMPLEMENTED + live-verified.** `backend: k8s` routes a job to the
> `KubernetesAdapter`, which renders a Namespace + PodGroup + N pods and drives
> them through the same reconciler/state-machine/drop-zone as every other
> backend. The cluster + KAI + HAMi are stood up out of band by
> [`infra/k8s/setup.sh`](../infra/k8s/setup.sh); the hand-run proof this grew
> out of is in [`infra/k8s/`](../infra/k8s/). Quick start at the end of this doc.

## Why this backend at all

Apptainer-on-VMs runs one container per node. It cannot answer a question we
now want to ask on a **single-GPU** dev box: *does an N-GPU distributed
workload schedule, initialize, and run correctly?* A second execution backend —
**Kubernetes with [HAMi](https://project-hami.io/) + the
[KAI Scheduler](https://github.com/NVIDIA/KAI-Scheduler)** — lets us
oversubscribe the one physical card into N virtual slices and launch a **gang
of N pods** on it, each seeing "its own GPU" with a hard VRAM cap. That is a
faithful simulator of the *scheduling + process topology + init path* of a
multi-GPU job, on hardware that has exactly one GPU.

## The key fact: HAMi and KAI compose (they don't fight)

The obvious worry — two GPU schedulers arguing over who places the pod — is a
non-issue, because the projects have converged into a **scheduler + isolation
layer** split, with an official
[integration guide](https://project-hami.io/docs/next/userguide/kai-scheduler/how-to-use-kai-scheduler):

- **KAI is the scheduler.** Gang scheduling (all ranks bind together or none),
  hierarchical queues/quotas, fractional-GPU packing, priority/fairness.
- **HAMi-core is the isolation layer *underneath* KAI.** Recent KAI ships a
  `hamicore` binder plugin + a `kai-resource-isolator` DaemonSet: the plugin
  injects `CUDA_DEVICE_MEMORY_LIMIT` into each sharing container, and the
  isolator `ld.so.preload`-injects HAMi's `libvgpu.so` so the VRAM cap is
  *hard* — one rank can't stomp another's memory even though they're on the
  same physical die.

So a pod asks for a slice with a plain annotation + `schedulerName`, and gets
KAI's scheduling with HAMi's isolation:

```yaml
metadata:
  labels: { kai.scheduler/queue: default-queue }
  annotations: { gpu-memory: "4096" }   # MiB, hard-capped by HAMi-core
spec:
  schedulerName: kai-scheduler
```

(Standalone HAMi uses `nvidia.com/gpumem` / `nvidia.com/gpucores` *resource
limits* instead; under KAI you use its *annotation* form. Pick one path — this
doc uses KAI-as-front-door.)

## What it actually buys — and the honest limit

**Buys you (real):** N-rank process topology, the distributed-init/rendezvous
path (NCCL, `torchrun`, MPI-over-TCP), gang-scheduling + queue/quota behavior
under contention, and hard per-rank VRAM partitioning. Enough to prove a
multi-GPU job *plumbs and runs*.

**Does NOT buy you (be clear-eyed):** real parallel throughput or NVLink/P2P
bandwidth. The N ranks **time-slice one SM array** — they take turns, they do
not run at once. So this is a bench for **correctness / integration / scheduler
behavior**, *not* for performance numbers. Treat a "4-GPU" run here as "4 ranks
that scheduled and completed," never as "4× the FLOPs." Unflagged, that
distinction burns weeks; flagged, the backend is exactly the right tool for the
job it's meant for.

> NCCL note: multiple ranks on one physical GPU use SHM/P2P-within-device
> transports. Some collectives assume distinct devices — expect to set
> `NCCL_P2P_DISABLE=1` / tune `NCCL_SHM_DISABLE` depending on the workload.

## How it drops into Outpost (the adapter mapping)

It fits the existing [ProviderAdapter](03-control-plane.md) model with **one
reframing of `node_count`**. Today an adapter owns *nodes*; Kubernetes owns its
own scheduling, so the whole K8s cluster is modelled as **one registry node**,
exactly like `localhost` is one node for the host GPU:

| Adapter method | Kubernetes meaning |
|----------------|--------------------|
| `provision`    | verify/ensure the cluster + HAMi + KAI are up (no-op if managed out of band) |
| `bootstrap`    | verify the `kai-scheduler` deployment, the isolator DaemonSet, and the job's Queue exist |
| `run`          | render a **Queue + PodGroup + N pods** from the JobSpec, `kubectl apply`, wait for the gang to complete, capture pod logs |
| `collect`      | gather each pod's logs + any `output_dir` PVC → the existing drop-zone |
| `deprovision`  | delete the job's namespace / PodGroup (leave the cluster standing) |

Consequences (as built):

- **Seed a small pool of interchangeable slots** `k8s-slot-0..N-1`
  (`provider: "k8s"`, `gpu: true`, no ip/ssh), gated by `CLUSTER_K8S_BACKEND=1`
  with `CLUSTER_K8S_SLOTS` (default 4). A `backend: k8s` job claims **one slot**
  (not `node_count` nodes) and does the N-way parallelism *inside* `run()` as N
  pods. A slot is a **concurrency bound**, not a physical resource: it caps how
  many gangs Outpost hands KAI at once — KAI queues the rest. This is what lets
  you submit genuinely concurrent k8s jobs and watch KAI arbitrate them on the
  one GPU (a single slot would serialize k8s jobs at the Outpost layer, so KAI
  would never see contention).
- **`node_count` is reinterpreted as gang size** (4 → an explicit PodGroup with
  `minMember: 4`; all 4 ranks bind together or none). Clean reuse of the field.
  Verified live that a pre-created PodGroup + pods carrying the `pod-group-name`
  annotation bind to *that* gang — KAI's auto-grouper does **not** override it
  with per-pod minMember-1 groups.
- **Routed by the new `backend` field on JobSpec** (`models.py`), mirroring how
  `gpu`/`hybrid` route the claim today. `registry.claim(..., backend="k8s")`
  claims by provider via `store.claim_node(provider=...)`; a normal CPU/GPU job
  explicitly **excludes** k8s slots, so the two pools never cross.
- **Everything else is reused untouched**: the state machine, audit trail,
  drop-zone egress + guaranteed `stdout.log` (here: concatenated per-rank logs,
  plus `rank-<i>.log`), replay logs, wait-and-retry, concurrent tick,
  `cluster submit/status/result/logs`. `KubernetesAdapter` is a translator to
  `kubectl`, nothing more — no new executor layer, same golden rule as every
  other adapter. It installs nothing: `provision`/`bootstrap` only *verify* the
  cluster + KAI + HAMi + queue + RuntimeClass exist and fail closed pointing at
  `setup.sh`; `run` applies the manifests and polls the gang to completion;
  `collect` ships the rank logs; `deprovision` deletes the job's namespace.
- **Distributed rendezvous is wired**, so the gang is a real multi-rank job, not
  N solo pods. Each gang gets a **headless Service** (`gang`, in its namespace)
  plus per-pod `hostname`/`subdomain`, giving every rank a stable DNS name
  `rank-<i>.gang.<ns>.svc.cluster.local`. Every rank is injected with `RANK`,
  `WORLD_SIZE`, `MASTER_ADDR` (rank-0's FQDN), and `MASTER_PORT` (default 29500,
  `params.master_port` to override) — exactly what `torchrun`/c10d/NCCL-over-TCP/
  MPI-over-TCP expect. `NCCL_P2P_DISABLE=1` is a default (all ranks share one
  physical die, so intra-device P2P is the broken transport); a job's own `env`
  overrides any of these. Verified live: both ranks resolve rank-0 to the same
  pod IP via cluster DNS before starting work.

Proposed spec shape (see [`job.k8s.example.yaml`](../job.k8s.example.yaml)):

```yaml
name: nccl-allreduce-sim-4
backend: k8s               # NEW: routes to the KubernetesAdapter / the one k8s node
image: "docker://nvidia/cuda:12.6.3-base-ubuntu24.04"
command: ["bash", "-c", "nvidia-smi -L; echo rank $RANK ok"]
node_count: 4              # gang size = 4 pods, ALL on the one physical GPU
gpu: true
params:
  gpu_memory_mb: 4096      # HAMi hard VRAM cap per rank (annotation gpu-memory)
  queue: default-queue     # KAI queue
```

## Substrate — decided: NixOS + KinD

The adapter is a few hundred lines of `kubectl` templating. **The risk is
getting the physical GPU visible to a Kubernetes device plugin on a single dev
host.** Substrate chosen 2026-07-29: **NixOS + KinD** (k3s explicitly deferred).
De-risking status on the actual box (RTX 5060 Ti):

- ✅ Host driver works; **docker already sees the GPU via CDI**
  (`docker run --device nvidia.com/gpu=all … nvidia-smi -L` → exit 0). The
  toolkit is `hardware.nvidia-container-toolkit.enable`d in CDI mode. So the
  worst-case "can the card surface to a container at all" question is *answered
  yes*, with no system change.
- ⬜ **GPU into a KinD node** is the one open step. A kind node is itself a
  docker container and KinD has no per-node `--device` flag, so the node only
  inherits the GPU if the **NVIDIA runtime is docker's default runtime** — a
  one-line `nixos-rebuild` change (`virtualisation.docker.enableNvidia`), see
  [`infra/k8s/nixos-gpu-kind.md`](../infra/k8s/nixos-gpu-kind.md). Possibly a
  second knob (`accept-nvidia-visible-devices-as-volume-mounts`) if the device
  plugin can't enumerate the card from inside the node.

Once the node sees the GPU, HAMi + KAI install identically to any cluster via
Helm (below). The full hand-run runbook lives in
[`infra/k8s/README.md`](../infra/k8s/README.md).

## Install (once a substrate is up) — from the official guide

```bash
# KAI, with GPU sharing + the HAMi-core hard-isolation plugin:
helm install kai-scheduler oci://ghcr.io/nvidia/kai-scheduler \
  --version v0.16.4 \
  --set global.gpuSharing=true \
  --set binder.plugins.hamicore.enabled=true \
  --namespace kai-scheduler --create-namespace

# The node-side isolator DaemonSet (ships libvgpu.so to GPU nodes):
helm install kai-resource-isolator oci://docker.io/projecthami/kai-resource-isolator \
  --namespace kai-resource-isolator --create-namespace \
  --version 1.0.0-chart
```

Pin these versions against what's current when you run it, and confirm the
PodGroup CRD's apiVersion against the installed KAI (`kubectl get crd | grep -i
podgroup`) before trusting the demo manifest's gang block — the sharing
annotation is stable across versions, the gang CRD group name is the thing most
likely to have moved.

## Quick start (cluster + backend)

```bash
nix develop                     # kind/kubectl/helm + the cluster CLI

# 1. Stand up the substrate + KAI + HAMi (idempotent; re-run after kind delete).
#    Prereq: docker's default runtime = nvidia (one nixos-rebuild — nixos-gpu-kind.md).
make k8s-up                     # == infra/k8s/setup.sh

# 2. Turn the backend on and seed its slots into the registry.
#    (config.env: CLUSTER_K8S_BACKEND=1; CLUSTER_K8S_SLOTS defaults to 4.)
CLUSTER_K8S_BACKEND=1 make seed-nodes
cluster nodes               # k8s-slot-0..3 appear, adapter_key=k8s

# 3. Submit a real k8s job and drive it to promoted.
cluster submit --spec job.k8s.small.example.yaml   # -> job-id
cluster reconcile --once --execute                 # (loop until terminal)
cluster result <job-id>                            # drop-zone: rank-0.log, rank-1.log, stdout.log

# 4. The "harder proof" — concurrent gangs, KAI arbitrating on the one GPU:
make demo-k8s-scheduling
```

Each rank's log shows its HAMi cap (`CUDA_DEVICE_MEMORY_LIMIT`) and the shared
GPU UUID — N ranks co-resident on one physical card. Tunables (queue, per-rank
VRAM cap, PodGroup CRD apiVersion, run timeout, kube-context) are all env-
overridable — see the `K8S_*` constants in `reconciler/adapter.py` and the
`CLUSTER_K8S_*` block in `infra/libvirt/config.env`.

## History: how this was de-risked before the adapter

The adapter was built on top of a fully hand-run spike (no Python): stand up the
substrate, get `nvidia-smi -L` in one plain GPU pod, then install KAI + isolator
and apply a gang of N pods each with `gpu-memory` set, confirming they co-reside
and the VRAM cap is enforced. That runbook + the exact gotchas it encodes live
in [`infra/k8s/setup.sh`](../infra/k8s/setup.sh),
[`infra/k8s/README.md`](../infra/k8s/README.md), and the proof writeup
[`infra/k8s/hami_kind_run.md`](../infra/k8s/hami_kind_run.md). The manifest shape
the adapter renders is exactly what that spike proved works.
