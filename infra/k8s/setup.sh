#!/usr/bin/env bash
# Stand up the HAMi + KAI GPU-sharing spike on NixOS + KinD, end to end.
# PROVEN 2026-07-29: 4 pods sharing one RTX 5060 Ti, each HAMi-capped to 2 GiB.
#
# Prereq (one-time, needs a nixos-rebuild — see nixos-gpu-kind.md):
#   docker's default runtime = nvidia. Verify:
#     docker info --format '{{.DefaultRuntime}}'   # must print: nvidia
#
# Run from the repo root inside `nix develop` (kind/kubectl/helm on PATH).
# Every step below is here because a naive install fails without it — the
# comments say why. Idempotent-ish; re-run after `kind delete cluster`.
set -euo pipefail
cd "$(dirname "$0")/../.."

CLUSTER=outpost
GPU_UUID=$(nvidia-smi --query-gpu=uuid --format=csv,noheader | head -1)
GPU_MEM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
echo "GPU: $GPU_UUID (${GPU_MEM} MiB)"

# 1. IPv4-only kind network. Otherwise the node gets an IPv6 addr and containerd
#    dead-ends pulling docker.io over IPv6 (esp. behind Tailscale MagicDNS),
#    hanging image pulls indefinitely.
kind delete cluster --name "$CLUSTER" 2>/dev/null || true
docker network rm kind 2>/dev/null || true
docker network create --driver bridge --subnet 172.20.0.0/16 kind >/dev/null

# 2. Cluster. kind-cluster.yaml wires the node's inner containerd to the nvidia
#    runtime (mode=cdi) + mounts /nix/store, the runtime config, and the host
#    CDI spec — see that file. GPU reaches the node via the /dev/null ->
#    /var/run/nvidia-container-devices/all extraMount.
kind create cluster --name "$CLUSTER" --config infra/k8s/kind-cluster.yaml
kubectl wait --for=condition=Ready node/${CLUSTER}-control-plane --timeout=120s

# 3. A UUID-named CDI device. KAI's binder (and the device plugin's uuid mode)
#    request nvidia.com/gpu=<UUID>, but the nixos-generated CDI spec only names
#    devices by index ("0","all") -> "unresolvable CDI devices" at container
#    create. Clone device "0" under the UUID (top-level containerEdits carry the
#    driver libs, so the clone injects identically). /var/run/cdi is dir-mounted
#    into the node, so patching the host file propagates.
#    ⚠ spike shortcut: /var/run/cdi is root-owned + nix-regenerated on boot.
sudo cat /var/run/cdi/nvidia-container-toolkit.json \
  | GPU_UUID="$GPU_UUID" python3 -c '
import sys,json,copy,os
u=os.environ["GPU_UUID"]; d=json.load(sys.stdin)
if u not in [x["name"] for x in d["devices"]]:
    z=copy.deepcopy([x for x in d["devices"] if x["name"]=="0"][0]); z["name"]=u
    d["devices"].append(z)
print(json.dumps(d,indent=2))' | sudo tee /var/run/cdi/nvidia-container-toolkit.json >/dev/null

# 4. RuntimeClass "nvidia" — the kai-resource-isolator webhook injects
#    runtimeClassName: nvidia into every GPU pod; without the object, pod
#    creation is rejected ("RuntimeClass nvidia not found").
kubectl apply -f infra/k8s/runtimeclass-nvidia.yaml

# 5. NVIDIA device plugin, deviceIDStrategy=index (advertises nvidia.com/gpu=1
#    so KAI sees a GPU). Needs the node label or its DaemonSet won't schedule.
kubectl label node ${CLUSTER}-control-plane nvidia.com/gpu.present=true --overwrite
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin >/dev/null 2>&1 || true
helm repo update >/dev/null
helm upgrade -i nvdp nvdp/nvidia-device-plugin -n nvidia-device-plugin \
  --create-namespace --version 0.17.4 --set deviceIDStrategy=index

# 6. GPU-memory + count labels — KAI reads these (normally from GPU Feature
#    Discovery) to know the shareable VRAM per GPU; without them KAI reports
#    "not enough GPU memory".
kubectl label node ${CLUSTER}-control-plane \
  nvidia.com/gpu.memory="$GPU_MEM" nvidia.com/gpu.count=1 --overwrite

# 7. KAI scheduler (GPU sharing + HAMi-core hard isolation) + the isolator.
#    NOTE the org path: ghcr.io/kai-scheduler/... (NOT ghcr.io/nvidia/...).
helm upgrade -i kai-scheduler oci://ghcr.io/kai-scheduler/kai-scheduler/kai-scheduler \
  --version v0.16.4 --set global.gpuSharing=true \
  --set binder.plugins.hamicore.enabled=true \
  -n kai-scheduler --create-namespace
helm upgrade -i kai-resource-isolator oci://docker.io/projecthami/kai-resource-isolator \
  -n kai-resource-isolator --create-namespace --version 1.0.0-chart
kubectl -n kai-scheduler wait --for=condition=Ready pod --all --timeout=180s

# 8. Give the default queue GPU quota (ships at 0 -> non-preemptible pods are
#    "over quota" and never schedule).
for q in default-parent-queue default-queue; do
  kubectl patch queue "$q" --type merge \
    -p '{"spec":{"resources":{"gpu":{"quota":1,"limit":-1,"overQuotaWeight":1}}}}'
done

echo "== ready =="
echo "kubectl apply -f infra/k8s/demo/gpu-share-gang.yaml   # 4 pods share the one GPU"
