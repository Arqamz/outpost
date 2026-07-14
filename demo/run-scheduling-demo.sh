#!/usr/bin/env bash
# Live scheduling demo: submits 4 jobs against the reconciler and ticks it with
# --execute until all reach a terminal state, narrating each tick so you can
# watch the scheduler (claims, waits, real command output) as it happens.
#
#   A, B: mpi jobs, node_count=4 each  -> together claim the whole 8-VM pool
#   C:    mpi job,  node_count=2       -> pool is full; waits (reconciler's
#         wait-and-retry NoCapacity handling) until A or B tears down and
#         frees nodes, then claims and runs for real
#   D:    gpu job,  node_count=1       -> the host GPU node, entirely
#         independent of the VM-pool contention above
#
# Prereqs: `make cluster-up && make bootstrap && make mpi-sif && make seed-nodes`.
# Run from the repo root inside the nix dev shell.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../infra/libvirt" && pwd)/lib.sh"
preflight

MAX_TICKS="${CLUSTER_DEMO_MAX_TICKS:-90}"
TICK_SLEEP="${CLUSTER_DEMO_TICK_SLEEP:-3}"

log "submitting 4 jobs: A,B (mpi x4 nodes each), C (mpi x2 -- should wait), D (gpu x1)"
JOB_A=$(bin/cluster submit --spec job.mpi.example.yaml)
JOB_B=$(bin/cluster submit --spec job.mpi.example.yaml)
JOB_C=$(bin/cluster submit --spec job.mpi.small.example.yaml)
JOB_D=$(bin/cluster submit --spec job.gpu.example.yaml)
log "A=${JOB_A}  B=${JOB_B}  C=${JOB_C}  D=${JOB_D}"
log "full replay after the fact: bin/cluster logs <job-id>   (or) bin/cluster reconciler-log"

all_terminal() {
  local st
  for j in "$JOB_A" "$JOB_B" "$JOB_C" "$JOB_D"; do
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
bin/cluster list | grep -E "^(${JOB_A}|${JOB_B}|${JOB_C}|${JOB_D})[[:space:]]"
echo
echo "per-job replay transcripts:"
for j in "$JOB_A" "$JOB_B" "$JOB_C" "$JOB_D"; do
  echo "  bin/cluster logs ${j}"
done
echo "cross-job chronological timeline: bin/cluster reconciler-log"
