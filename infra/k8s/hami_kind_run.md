# Proof of run — 4 GPU workers sharing one physical GPU (KAI + HAMi on KinD)

**Status: ✅ PASS** &nbsp;·&nbsp; **Date:** 2026-07-29 &nbsp;·&nbsp; **Host:** NixOS 26.11, single NVIDIA RTX 5060 Ti (16 GB)

Four independent GPU worker pods were scheduled by the **KAI Scheduler** onto a
**single physical GPU** and run concurrently, each memory-isolated to a 2 GiB
slice by **HAMi-core**. This records the live evidence that the Kubernetes GPU
backend (`docs/08-kubernetes-backend.md`) works end to end on our hardware.

> **What this proves:** GPU virtualization + scheduling — one physical GPU is
> shared by N containers, each placed by KAI and given a hard per-container VRAM
> cap by HAMi. This is the "simulate an N-GPU job on one GPU" capability.
>
> **What it does NOT claim:** real parallel throughput or NVLink/P2P — the
> workers time-slice one SM array. It is a correctness/scheduling/isolation
> proof, not a performance benchmark. It is also a single-node cluster (one GPU
> node); "across a cluster" here means across the Kubernetes control plane, not
> multiple physical hosts.

---

## 1. Environment

| Component | Version / value |
|---|---|
| OS | NixOS 26.11 (`platinum`), kernel 7.1.2 |
| GPU / driver | NVIDIA GeForce RTX 5060 Ti, 16311 MiB · driver 595.84 · CUDA 13.2 |
| Cluster | KinD, node image `kindest/node:v1.35.0` (Kubernetes v1.35.0) |
| Container runtime (node) | `containerd://2.2.0`, default runtime **nvidia** (CDI mode) |
| Device plugin | `nvidia-device-plugin` 0.17.4 (`deviceIDStrategy=index`) |
| Scheduler | `kai-scheduler` **v0.16.4** (`global.gpuSharing=true`, `binder.plugins.hamicore.enabled=true`) |
| Isolation | `kai-resource-isolator` 1.0.0 (HAMi-core `libvgpu.so` via `ld.so.preload`) |

Reproducible via [`infra/k8s/setup.sh`](setup.sh) after the one-time NixOS change
in [`nixos-gpu-kind.md`](nixos-gpu-kind.md).

---

## 2. Cluster & GPU node

```console
$ kubectl get nodes -o wide
NAME                    STATUS   ROLES           AGE   VERSION   INTERNAL-IP   OS-IMAGE                         KERNEL-VERSION   CONTAINER-RUNTIME
outpost-control-plane   Ready    control-plane   39m   v1.35.0   172.20.0.2    Debian GNU/Linux 12 (bookworm)   7.1.2            containerd://2.2.0

$ kubectl get node outpost-control-plane -o json | jq .status.capacity
capacity nvidia.com/gpu = 1          # the one physical GPU, advertised to Kubernetes

# GPU-feature labels KAI reads to fractionally share the card:
nvidia.com/gpu.present = true
nvidia.com/gpu.count   = 1
nvidia.com/gpu.memory  = 16311       # MiB
```

---

## 3. Stack is up (KAI + HAMi + device plugin)

```console
$ helm ls -A
NAME                  	NAMESPACE            	REVISION	STATUS  	CHART                            	APP VERSION
kai-resource-isolator 	kai-resource-isolator	1       	deployed	kai-resource-isolator-1.0.0-chart	1.0.0
kai-scheduler         	kai-scheduler        	1       	deployed	kai-scheduler-v0.16.4            	v0.16.4
nvdp                  	nvidia-device-plugin 	2       	deployed	nvidia-device-plugin-0.17.4      	0.17.4

$ kubectl -n kai-scheduler get pods
NAME                                     READY   STATUS    RESTARTS   AGE
admission-59f59d54cb-mzlh6               1/1     Running   0          31m
binder-557f8b5d4d-28sxz                  1/1     Running   0          31m
kai-operator-7699f6fdd7-hwbcj            1/1     Running   0          31m
kai-scheduler-default-565d94d858-pxjfv   1/1     Running   0          31m
pod-grouper-659d597f7f-jdjbg             1/1     Running   0          31m
podgroup-controller-5cddb8d99c-44n5m     1/1     Running   0          31m
queue-controller-69c56d8c9-gdhsk         1/1     Running   0          31m

$ kubectl -n kai-resource-isolator get pods
NAME                                             READY   STATUS    RESTARTS   AGE
kai-resource-isolator-libsync-whlwp              1/1     Running   0          33m
kai-resource-isolator-webhook-7d8c55cf8b-xrsnn   1/1     Running   0          33m

$ kubectl get queues.scheduling.run.ai
NAME                   PARENT                 CHILDREN
default-parent-queue                          ["default-queue"]
default-queue          default-parent-queue
```

---

## 4. The 4 GPU workers — all Running on the one node/GPU

Workload: a 4-replica Deployment ([`demo/gpu-share-gang.yaml`](demo/gpu-share-gang.yaml)),
each replica requesting a `gpu-memory: "2048"` MiB slice via KAI.

```console
$ kubectl get pods -l app=gpu-share -o wide
NAME                        READY   STATUS    RESTARTS   AGE   IP            NODE
gpu-share-8997b6f87-cvrq5   1/1     Running   1          19m   10.244.0.30   outpost-control-plane
gpu-share-8997b6f87-fjtvq   1/1     Running   1          19m   10.244.0.31   outpost-control-plane
gpu-share-8997b6f87-km4xz   1/1     Running   1          19m   10.244.0.32   outpost-control-plane
gpu-share-8997b6f87-vkrd9   1/1     Running   1          19m   10.244.0.33   outpost-control-plane
```

All four land on `outpost-control-plane` — i.e. the **same single physical GPU**.

### How KAI placed them (per-pod spec + scheduling events)

```console
$ kubectl get pod gpu-share-…-cvrq5 -o json | jq '.spec, .metadata'
schedulerName:        kai-scheduler        # placed by KAI, not the default scheduler
runtimeClassName:     nvidia               # routed to the nvidia container runtime
annotation gpu-memory: 2048                # fractional GPU request (MiB)
label kai.scheduler/queue: default-queue

$ kubectl describe pod gpu-share-…-cvrq5 | grep -E 'kai-scheduler|Bound|Scheduled'
  PodScheduled  True
  PodBound      True
  Normal  Scheduled  kai-scheduler  Successfully assigned pod default/gpu-share-…-cvrq5 to node outpost-control-plane at node-pool default
  Normal  Bound      binder         Pod bound successfully to node outpost-control-plane
```

KAI also created a scheduling PodGroup per worker (gang primitive; `minMember`
is configurable for true all-or-nothing gangs):

```console
$ kubectl get podgroups.scheduling.run.ai
NAME                                                                AGE
pg-gpu-share-8997b6f87-cvrq5-7e64f4b5-…                             20m
pg-gpu-share-8997b6f87-fjtvq-3e7fc418-…                             20m
pg-gpu-share-8997b6f87-km4xz-b0052521-…                             20m
pg-gpu-share-8997b6f87-vkrd9-32ca8da4-…                             20m
```

---

## 5. Per-worker proof: HAMi cap injected + the shared GPU visible

Each worker logs the **HAMi-injected VRAM limit** and sees the **same physical
GPU** (identical UUID = one card, shared four ways):

```console
$ for p in $(kubectl get pods -l app=gpu-share -o name); do kubectl logs $p; done
rank=gpu-share-8997b6f87-cvrq5  CUDA_DEVICE_MEMORY_LIMIT=2120m
GPU 0: NVIDIA GeForce RTX 5060 Ti (UUID: GPU-b75c61a8-fba6-aa18-344a-8a8871ba6af9)
rank=gpu-share-8997b6f87-fjtvq  CUDA_DEVICE_MEMORY_LIMIT=2120m
GPU 0: NVIDIA GeForce RTX 5060 Ti (UUID: GPU-b75c61a8-fba6-aa18-344a-8a8871ba6af9)
rank=gpu-share-8997b6f87-km4xz  CUDA_DEVICE_MEMORY_LIMIT=2120m
GPU 0: NVIDIA GeForce RTX 5060 Ti (UUID: GPU-b75c61a8-fba6-aa18-344a-8a8871ba6af9)
rank=gpu-share-8997b6f87-vkrd9  CUDA_DEVICE_MEMORY_LIMIT=2120m
GPU 0: NVIDIA GeForce RTX 5060 Ti (UUID: GPU-b75c61a8-fba6-aa18-344a-8a8871ba6af9)
```

- **`CUDA_DEVICE_MEMORY_LIMIT=2120m`** — KAI's hamicore plugin injected the hard
  cap (2048 MiB request → 2120 MiB enforced ceiling). HAMi-core (`libvgpu.so`,
  preloaded by the isolator) enforces it on CUDA allocation.
- **Same GPU UUID in all four** — one physical card, four co-resident workers.

`nvidia-smi` inside a worker confirms the driver/CUDA stack is live in the
container (note: `nvidia-smi` reports the *physical* card totals — the 3583 MiB
"used" is the host's own desktop; HAMi enforces the cap at CUDA-allocation time,
not on this display):

```console
$ kubectl exec gpu-share-…-cvrq5 -- nvidia-smi
NVIDIA-SMI 595.84    Driver Version: 595.84    CUDA Version: 13.2
+-----------------------------------------+------------------------+----------------------+
|   0  NVIDIA GeForce RTX 5060 Ti     On  |   00000000:01:00.0  On |                  N/A |
| 30%   41C    P0             16W /  180W |    3583MiB /  16311MiB |      1%      Default |
+-----------------------------------------+------------------------+----------------------+
```

---

## 6. The single physical GPU (host view)

```console
$ docker exec outpost-control-plane nvidia-smi --query-gpu=name,memory.total --format=csv
name, memory.total [MiB]
NVIDIA GeForce RTX 5060 Ti, 16311 MiB
```

One 16 GB card → **4 workers × 2 GiB slices**, scheduled and isolated. ∎

---

## Appendix — how to reproduce

```bash
# one-time: nvidia = docker default runtime (see nixos-gpu-kind.md), then:
nix develop
./infra/k8s/setup.sh
kubectl apply -f infra/k8s/demo/gpu-share-gang.yaml
kubectl get pods -l app=gpu-share -o wide
kubectl logs -l app=gpu-share --tail=2
```

Key hard-won gotchas (all documented in `setup.sh` / `README.md`): IPv4-only
KinD network; inner-containerd nvidia runtime in CDI mode; a UUID-named CDI
device; the `nvidia` RuntimeClass the isolator webhook requires; node GPU labels;
device-plugin `deviceIDStrategy=index`; KAI's `ghcr.io/kai-scheduler/...` chart
path; and granting `default-queue` GPU quota (ships at 0).
