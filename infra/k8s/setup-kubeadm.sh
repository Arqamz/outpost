#!/usr/bin/env bash
# Wire a real kubeadm cluster (control plane + GPU workers already joined via
# `kubeadm init` / `kubeadm token create --print-join-command`) up to the
# point where KubernetesAdapter's "dedicated" GPU mode (CLUSTER_K8S_GPU_MODE=
# dedicated) can schedule against it: a CNI so pods actually get IPs, and the
# nvidia-device-plugin so nodes advertise nvidia.com/gpu. No KAI, no HAMi, no
# RuntimeClass object, no VRAM-sharing machinery at all — that's setup.sh's
# job for the KinD "shared" spike, untouched by this script.
#
# Assumes each GPU node's containerd already has the nvidia runtime installed
# and set as `default_runtime_name` (baked into the node image / done at node
# bootstrap) — this script only talks to the k8s API, it doesn't touch nodes
# directly.
#
# Run with KUBECONFIG (or --kubeconfig via kubectl's usual env vars) pointed
# at the target cluster. Idempotent: `kubectl apply`/`helm upgrade -i` are
# safe to re-run.
set -euo pipefail

CONTEXT="${CLUSTER_K8S_CONTEXT:-}"
POD_CIDR="${TASHKIL_POD_CIDR:-10.244.0.0/16}"
KCTL=(kubectl)
[ -n "$CONTEXT" ] && KCTL=(kubectl --context "$CONTEXT")

echo "cluster: $("${KCTL[@]}" config current-context 2>/dev/null || echo "$CONTEXT")"
"${KCTL[@]}" get nodes -o wide

# 1. CNI — kubeadm sets up everything except this; kubelet stays NotReady
#    (no pod IP allocation) until a CNI is applied. Flannel, matching the
#    10.244.0.0/16 pod CIDR the golden-image kubeadm-init template already
#    uses (images/ansible/roles/tashkil_bootstrap/files/templates/
#    kubeadm-init.yaml.tpl in tashkil-golden-images).
if ! "${KCTL[@]}" get daemonset -n kube-flannel kube-flannel-ds >/dev/null 2>&1; then
  echo "applying Flannel CNI (pod CIDR ${POD_CIDR})"
  "${KCTL[@]}" apply -f https://github.com/flannel-io/flannel/releases/latest/download/kube-flannel.yml
else
  echo "Flannel already present"
fi
"${KCTL[@]}" wait --for=condition=Ready nodes --all --timeout=180s

# 2. nvidia-device-plugin — advertises nvidia.com/gpu on every node whose
#    containerd has the nvidia runtime configured. Same chart/version the KinD
#    "shared" spike uses (setup.sh), default deviceIDStrategy this time (no
#    "index" override — GPUs aren't virtually shared here, one real device
#    per allocation).
helm repo add nvdp https://nvidia.github.io/k8s-device-plugin >/dev/null 2>&1 || true
helm repo update >/dev/null
helm upgrade -i nvdp nvdp/nvidia-device-plugin -n nvidia-device-plugin \
  --create-namespace --version 0.17.4 \
  ${CONTEXT:+--kube-context "$CONTEXT"}
"${KCTL[@]}" -n nvidia-device-plugin wait --for=condition=Ready pod --all --timeout=180s

echo "== ready =="
"${KCTL[@]}" get nodes -o jsonpath='{range .items[*]}{.metadata.name}{"\tnvidia.com/gpu="}{.status.allocatable.nvidia\.com/gpu}{"\n"}{end}'
echo "kubectl apply -f infra/k8s/demo/gpu-dedicated-gang.yaml   # one real GPU per pod, no sharing"
