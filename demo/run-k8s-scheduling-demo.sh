#!/usr/bin/env bash
# Live k8s-backend scheduling demo — the "harder proof" driven through Outpost's
# real control plane instead of raw kubectl. Submits several concurrent
# `backend: k8s` jobs and ticks the reconciler with --execute until they finish,
# so you can watch Outpost hand N gangs to KAI at once and KAI co-schedule them
# on the ONE physical GPU with HAMi caps per rank.
#
#   N jobs, each a gang of node_count pods, ALL sharing the single GPU. With
#   CLUSTER_K8S_SLOTS slots, up to that many run at once in Outpost terms; a
#   further job waits (Outpost wait-and-retry) OR, once handed off, KAI queues it
#   until VRAM frees. Each rank logs its HAMi cap (CUDA_DEVICE_MEMORY_LIMIT) and
#   the shared GPU UUID.
#
# Prereqs: `make k8s-up` (cluster + KAI + HAMi) and a CLUSTER_K8S_BACKEND=1 seed
# (`CLUSTER_K8S_BACKEND=1 make seed-nodes`). Run from the repo root in nix dev shell.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../infra/libvirt" && pwd)/lib.sh"

JOBS="${CLUSTER_K8S_DEMO_JOBS:-4}"
MAX_TICKS="${CLUSTER_DEMO_MAX_TICKS:-120}"
TICK_SLEEP="${CLUSTER_DEMO_TICK_SLEEP:-3}"
SPEC="${CLUSTER_K8S_DEMO_SPEC:-job.k8s.small.example.yaml}"

log "submitting ${JOBS} concurrent k8s jobs (spec ${SPEC}) — each a gang on the one GPU"
IDS=()
for _ in $(seq 1 "${JOBS}"); do IDS+=("$(bin/cluster submit --spec "${SPEC}")"); done
log "jobs: ${IDS[*]}"
log "watch KAI: kubectl get pods -A -l kai.scheduler/queue --field-selector=status.phase=Running -o wide"

all_terminal() {
  local st
  for j in "${IDS[@]}"; do
    st=$(bin/cluster list | awk -v j="$j" '$1==j{print $2}')
    case "$st" in promoted|failed|rejected|cancelled) ;; *) return 1 ;; esac
  done
  return 0
}

tick=0
while (( tick < MAX_TICKS )); do
  tick=$((tick + 1))
  echo "──────────────────────── tick ${tick} ────────────────────────"
  bin/cluster reconcile --once --execute
  all_terminal && break
  sleep "${TICK_SLEEP}"
done

echo
echo "════════════════════════ final states ════════════════════════"
for j in "${IDS[@]}"; do bin/cluster list | awk -v j="$j" '$1==j'; done
echo
echo "per-job replay transcripts: bin/cluster logs <job-id>"
echo "drop-zone (per-rank logs):  bin/cluster result <job-id>"
