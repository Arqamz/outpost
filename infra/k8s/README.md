# infra/k8s — Kubernetes backend (HAMi + KAI), spike + proof

> **SPIKE — PROVEN 2026-07-29, not wired into the reconciler yet.** This
> directory stands up the Kubernetes backend from
> [`docs/08-kubernetes-backend.md`](../../docs/08-kubernetes-backend.md) **by
> hand** and proves N pods share one physical GPU. No reconciler code depends
> on it yet — that's the `KubernetesAdapter`, the plan's final step.

The goal: **simulate an N-GPU job on a single GPU.** HAMi slices the card,
KAI schedules pods onto the slices. See the design doc for why the two compose
and what it does / doesn't reproduce (topology + init + scheduling: yes;
parallel throughput / NVLink: no).

**Substrate: NixOS + KinD.** ✅ **Proven end to end**: 4 pods, each HAMi-capped
to 2 GiB, all `Running` on the one **RTX 5060 Ti** — every pod reports
`CUDA_DEVICE_MEMORY_LIMIT=2120m` and the same GPU UUID.

## How to run it

**Prereq (one-time, `nixos-rebuild`):** make nvidia docker's default runtime —
see [`nixos-gpu-kind.md`](nixos-gpu-kind.md). Verify:
`docker info --format '{{.DefaultRuntime}}'` → `nvidia`.

Then, from the repo root inside `nix develop`:

```bash
./infra/k8s/setup.sh                              # cluster + device plugin + KAI + HAMi + all fixes
kubectl apply -f infra/k8s/demo/gpu-share-gang.yaml   # 4 pods share the one GPU
kubectl get pods -l app=gpu-share -o wide             # all Running on the one node
kubectl logs -l app=gpu-share --tail=2                # CUDA_DEVICE_MEMORY_LIMIT + the shared GPU
```

`setup.sh` is the source of truth for the working sequence — each step is
commented with *why* it's needed. The non-obvious gotchas it encodes (all hit
live):

1. **IPv4-only kind network** — else containerd hangs pulling docker.io over
   IPv6 (Tailscale MagicDNS handed the node IPv6-only records).
2. **Inner-containerd nvidia runtime, mode=cdi** (in `kind-cluster.yaml`) —
   nested pods need the driver injected; legacy mode can't find NixOS's
   `/nix/store` driver paths, CDI can (it's the host's proven mechanism).
3. **A UUID-named CDI device** — KAI + the device plugin request
   `nvidia.com/gpu=<UUID>`, but the nixos CDI spec only names devices by index;
   unresolved → container-create fails. `setup.sh` clones device `0` under the
   UUID.
4. **`RuntimeClass nvidia`** ([`runtimeclass-nvidia.yaml`](runtimeclass-nvidia.yaml))
   — the isolator webhook injects `runtimeClassName: nvidia`; the object must exist.
5. **Node labels** `nvidia.com/gpu.present` (device-plugin DS scheduling) +
   `nvidia.com/gpu.memory` / `.count` (KAI reads these for VRAM fractioning).
6. **Device plugin `deviceIDStrategy=index`**, **KAI org path**
   `ghcr.io/kai-scheduler/...` (not `nvidia/...`), **default-queue GPU quota**
   (ships at 0 → nothing schedules).

## What's here

| File | Purpose |
|------|---------|
| `setup.sh` | one-command stand-up of the whole stack (the working recipe) |
| `kind-cluster.yaml` | cluster + inner-containerd nvidia runtime + GPU/nix mounts |
| `node-nvidia-runtime-config.toml` | the node's nvidia-container-runtime config (mode=cdi) |
| `runtimeclass-nvidia.yaml` | RuntimeClass the isolator webhook requires |
| `nixos-gpu-kind.md` | the one-time nixos-rebuild (nvidia = default docker runtime) |
| `demo/gpu-share-gang.yaml` | 4 pods, each a 2 GiB `gpu-memory` slice, one card — the proof |
| `demo/queue.yaml` | reference Queue CR (unused — KAI ships `default-queue`; `setup.sh` just grants it quota) |

## Cleanup (fully reversible)

```bash
kubectl delete -f demo/gpu-share-gang.yaml -f demo/queue.yaml
helm uninstall kai-resource-isolator -n kai-resource-isolator
helm uninstall kai-scheduler -n kai-scheduler
# then tear down the substrate (k3s-uninstall.sh / kind delete cluster / minikube delete)
```
