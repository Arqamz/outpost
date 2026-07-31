# KAI + HAMi on native Ubuntu (the L20 box)

> Companion to [`nixos-gpu-kind.md`](nixos-gpu-kind.md) / [`setup.sh`](setup.sh),
> which are **NixOS-specific** (the `/nix/store` CDI gymnastics exist only because
> NixOS keeps driver libs off the standard path). On stock Ubuntu the GPU→kind
> path is the *documented upstream one* and much shorter. **Only the
> GPU-visibility steps (1–4) differ from `setup.sh`; the KAI/HAMi/device-plugin/
> queue steps (5–9) are distro-agnostic and copied verbatim from `setup.sh`.**
>
> Status: runbook to execute on the L20 box — validate each step there. Once it's
> green, fold it into `setup.sh` behind a host check.

Assumes: Ubuntu 22.04/24.04, the L20 driver installed (`nvidia-smi` works on the
host), Docker, and `kind`/`kubectl`/`helm` on PATH (nix dev shell provides them,
or install natively).

## 1. NVIDIA Container Toolkit → Docker default runtime

```bash
# install the toolkit (NVIDIA apt repo), then make nvidia Docker's DEFAULT
# runtime so a kind node (itself a docker container) inherits the GPU.
sudo nvidia-ctk runtime configure --runtime=docker --set-as-default
# let a container claim the GPU via a volume-mount marker (how we inject it into
# the node without a per-node --device flag kind doesn't have):
sudo nvidia-ctk config --in-place --set accept-nvidia-visible-devices-as-volume-mounts=true
sudo systemctl restart docker
docker info --format '{{.DefaultRuntime}}'    # must print: nvidia
```

Sanity (no kind yet) — the GPU reaches a plain container via the volume marker:

```bash
docker run --rm -v /dev/null:/var/run/nvidia-container-devices/all \
  ubuntu:24.04 nvidia-smi -L        # lists the L20
```

## 2. IPv4-only kind network

Same reason as the NixOS box — avoid IPv6 image-pull hangs:

```bash
docker network rm kind 2>/dev/null || true
docker network create --driver bridge --subnet 172.20.0.0/16 kind
```

## 3. kind cluster with the GPU in the node

Reuse [`kind-cluster.yaml`](kind-cluster.yaml) but **drop the NixOS-only
`extraMounts`** (`/nix/store`, the CDI config file, `node-nvidia-runtime-config.toml`).
On Ubuntu you keep only:
- `extraMounts: - hostPath: /dev/null` → `containerPath: /var/run/nvidia-container-devices/all`
  (injects the GPU into the node), and
- the `containerdConfigPatches` setting the node's inner containerd
  `default_runtime_name = "nvidia"` with `BinaryName` = the host
  `nvidia-container-runtime` (so *nested* pods get the GPU too).

```bash
kind create cluster --name outpost --config infra/k8s/kind-cluster.ubuntu.yaml
kubectl wait --for=condition=Ready node/outpost-control-plane --timeout=120s
docker exec outpost-control-plane nvidia-smi -L      # node sees the L20
```

(Write `kind-cluster.ubuntu.yaml` as the trimmed copy described above.)

## 4. (Only if the device plugin can't resolve by UUID)

The NixOS box had to clone a UUID-named CDI device (`setup.sh` step 3) because
its CDI spec named devices by index. With the toolkit-generated CDI on Ubuntu
this is usually unnecessary. If you later hit `unresolvable CDI devices
nvidia.com/gpu=GPU-<uuid>`, generate a proper CDI spec on the host
(`sudo nvidia-ctk cdi generate --output=/etc/cdi/nvidia.yaml`) and mount
`/etc/cdi` into the node, or keep `deviceIDStrategy=index` (step 5) as below.

## 5–9. KAI + HAMi + device plugin + queue quota (identical to `setup.sh`)

These steps are pure `kubectl`/`helm` and **not** distro-specific — run them
exactly as in [`setup.sh`](setup.sh):

```bash
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)   # ~46068 for the L20

# 5. RuntimeClass nvidia (isolator webhook injects runtimeClassName: nvidia)
kubectl apply -f infra/k8s/runtimeclass-nvidia.yaml

# 6. NVIDIA device plugin (index strategy) + the node labels KAI reads
kubectl label node outpost-control-plane nvidia.com/gpu.present=true --overwrite
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin && helm repo update
helm upgrade -i nvdp nvdp/nvidia-device-plugin -n nvidia-device-plugin \
  --create-namespace --version 0.17.4 --set deviceIDStrategy=index
kubectl label node outpost-control-plane \
  nvidia.com/gpu.memory="$GPU_MEM" nvidia.com/gpu.count=1 --overwrite

# 7. KAI scheduler (GPU sharing + HAMi-core) + the isolator
helm upgrade -i kai-scheduler oci://ghcr.io/kai-scheduler/kai-scheduler/kai-scheduler \
  --version v0.16.4 --set global.gpuSharing=true \
  --set binder.plugins.hamicore.enabled=true -n kai-scheduler --create-namespace
helm upgrade -i kai-resource-isolator oci://docker.io/projecthami/kai-resource-isolator \
  -n kai-resource-isolator --create-namespace --version 1.0.0-chart
kubectl -n kai-scheduler wait --for=condition=Ready pod --all --timeout=180s

# 8. GPU quota on the default queues (ship at 0 -> nothing schedules)
for q in default-parent-queue default-queue; do
  kubectl patch queue "$q" --type merge \
    -p '{"spec":{"resources":{"gpu":{"quota":1,"limit":-1,"overQuotaWeight":1}}}}'
done
```

## 9. Expose the API server for a remote control plane

If Outpost's control plane is on a *different* box (the multi-machine layout —
[`docs/09-multi-machine-cluster.md`](../../docs/09-multi-machine-cluster.md)),
this cluster's `kubectl` must be reachable from there. Recreate the cluster with
`networking.apiServerAddress: <this-box-ip>` + a fixed `apiServerPort`, copy the
kubeconfig entry to the control plane, and rename its context (e.g. `kind-vm`).
Then register it: `CLUSTER_K8S_CLUSTERS="vm:kind-vm:8,..."`.

## Why the L20 is the better HAMi box

48 GB vs the 5060 Ti's 16 GB → far more headroom to slice: e.g. 8 ranks ×
4 GB, or 24 × 2 GB, gang-scheduled on one card. The "multi-GPU on one GPU"
simulation is much more convincing at L20 scale.
