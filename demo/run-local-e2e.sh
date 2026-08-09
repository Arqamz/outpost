#!/usr/bin/env bash
# Local end-to-end smoke of the whole control plane — as far as a single dev box
# goes: dry-run scheduling for EVERY job shape (null adapter, no infra needed),
# the real host-GPU apptainer path, and the real k8s + KAI/HAMi backend, plus the
# reconciliation/ops commands (status/result/logs/fail-node/clear-jobs).
#
# Does NOT cover the 8-VM/MPI paths (run `make cluster-up && make bootstrap`
# first for those) or remote ssh workers (see docs/09). Safe to re-run — it
# resets job history each time. Run from repo root inside `nix develop`:
#     make e2e     (or)   demo/run-local-e2e.sh
set -uo pipefail
cd "$(dirname "$0")/.."
PASS=0; FAIL=0; SKIP=0
say()   { printf '\n\033[1m== %s ==\033[0m\n' "$1"; }
check() { if [ "$2" = "$3" ]; then echo "  PASS  $1  ($2)"; PASS=$((PASS+1));
          else echo "  FAIL  $1  (got '$2' want '$3')"; FAIL=$((FAIL+1)); fi; }
state() { bin/cluster list | awk -v j="$1" '$1==j{print $2}'; }
# drive the given job-ids to a terminal state under the given reconcile flags
# ("" = dry-run/NullAdapter, "--execute" = real adapters). reconcile touches ALL
# active jobs, so always fully drain one group before submitting the next.
drive()  { local flags="$1"; shift; local ids=("$@") t j st all
  for t in $(seq 1 90); do
    bin/cluster reconcile --once $flags >/dev/null 2>&1
    all=1; for j in "${ids[@]}"; do st=$(state "$j")
      case "$st" in promoted|failed|rejected|cancelled) ;; *) all=0 ;; esac; done
    [ "$all" = 1 ] && return 0; sleep 1
  done; }

say "reset + seed (k8s backend on)"
bin/cluster clear-jobs --force >/dev/null 2>&1 || true
CLUSTER_K8S_BACKEND=1 bin/cluster seed-nodes
echo; bin/cluster nodes

# ── A. DRY-RUN SCHEDULING (NullAdapter — exercises the full state machine) ──
say "A. dry-run: every job shape reaches promoted"
A_CPU=$(bin/cluster submit --spec job.example.yaml)
A_MPI=$(bin/cluster submit --spec job.mpi.example.yaml)          # node_count 4
A_GPU=$(bin/cluster submit --spec job.gpu.example.yaml)          # gpu -> host
A_K8S=$(bin/cluster submit --spec job.k8s.example.yaml)          # backend k8s, gang 4
drive "" "$A_CPU" "$A_MPI" "$A_GPU" "$A_K8S"
check "cpu single"      "$(state "$A_CPU")" promoted
check "mpi node_count=4" "$(state "$A_MPI")" promoted
check "gpu -> host"      "$(state "$A_GPU")" promoted
check "k8s gang (dry)"   "$(state "$A_K8S")" promoted

say "B. dry-run: wait-and-retry when the pool can't cover a job"
# A job asking for more CPU nodes than exist can never be satisfied, so it stays
# SUBMITTED (waiting + retrying every tick), it is NOT failed immediately. This
# is the deterministic form of the contention behaviour (a real N-VM contention
# demo is `make demo-scheduling`, which needs the VMs actually up).
B=$(bin/cluster submit --spec job.example.yaml)
python - "$B" <<'PY'                                 # bump node_count past the pool
import sys
from reconciler.store import open_store
from reconciler.cli import DEFAULT_STATE
s = open_store(DEFAULT_STATE); j = s.get_job(sys.argv[1])
j.spec["node_count"] = 99; s.put_job(j)
PY
for t in $(seq 1 3); do bin/cluster reconcile --once >/dev/null 2>&1; sleep 1; done
check "over-capacity job keeps waiting (not failed)" "$(state "$B")" submitted
bin/cluster clear-jobs --force >/dev/null 2>&1        # drop the un-satisfiable job

say "C. dry-run: failure injection -> job failed + node quarantined"
C=$(bin/cluster submit --spec job.example.yaml)
bin/cluster reconcile --once >/dev/null 2>&1          # claim + provision a node
CN=$(bin/cluster status "$C" | sed -n "s/^nodes *: *\['\([^']*\)'.*/\1/p")
echo "  job $C claimed node: ${CN:-<none>}"
if [ -n "${CN:-}" ]; then
  bin/cluster fail-node "$CN" --reason "e2e injected" >/dev/null 2>&1
  check "job failed after node failure" "$(state "$C")" failed
  check "node quarantined" "$(bin/cluster nodes | awk -v n="$CN" '$1==n{print $4}')" quarantined
else check "failure test" skipped skipped; SKIP=$((SKIP+1)); fi

# ── D. REAL EXECUTION (host GPU + k8s) — only after all dry-run jobs terminal ──
if command -v apptainer >/dev/null && nvidia-smi -L >/dev/null 2>&1; then
  say "D. execute: real host-GPU apptainer --nv job"
  D=$(bin/cluster submit --spec job.gpu.example.yaml)
  drive "--execute" "$D"
  check "host GPU job promoted" "$(state "$D")" promoted
  echo "  --- drop-zone stdout.log ---"; sed 's/^/  /' "$(bin/cluster result "$D")/stdout.log" 2>/dev/null | head -5
else say "D. execute host-GPU — SKIPPED (no apptainer/GPU)"; SKIP=$((SKIP+1)); fi

if kubectl get nodes >/dev/null 2>&1; then
  say "E. execute: real k8s + KAI/HAMi gang (single)"
  E=$(bin/cluster submit --spec job.k8s.small.example.yaml)
  drive "--execute" "$E"
  check "k8s gang promoted" "$(state "$E")" promoted
  echo "  --- per-rank logs (HAMi cap + GPU UUID) ---"
  for f in "$(bin/cluster result "$E")"/rank-*.log; do echo "  [$(basename "$f")]"; sed 's/^/    /' "$f"; done

  say "F. execute: 3 concurrent k8s gangs, KAI arbitrating on the one GPU"
  F1=$(bin/cluster submit --spec job.k8s.small.example.yaml)
  F2=$(bin/cluster submit --spec job.k8s.small.example.yaml)
  F3=$(bin/cluster submit --spec job.k8s.small.example.yaml)
  drive "--execute" "$F1" "$F2" "$F3"
  check "concurrent gang 1" "$(state "$F1")" promoted
  check "concurrent gang 2" "$(state "$F2")" promoted
  check "concurrent gang 3" "$(state "$F3")" promoted
else say "E/F. execute k8s — SKIPPED (kubectl can't reach a cluster; run make k8s-up)"; SKIP=$((SKIP+1)); fi

# ── G. OPS / OBSERVABILITY COMMANDS ─────────────────────────────────────────
say "G. ops commands (status / result / logs / reconciler-log)"
ANY=$(bin/cluster list | awk 'NR==1{print $1}')
bin/cluster status "$ANY" | head -6
echo "  result -> $(bin/cluster result "$ANY")"
echo "  logs (last 3 lines):"; bin/cluster logs "$ANY" --tail 3 | sed 's/^/    /'
echo "  reconciler-log (last 3 lines):"; bin/cluster reconciler-log --tail 3 | sed 's/^/    /'

say "RESULT: $PASS passed, $FAIL failed, $SKIP skipped"
[ "$FAIL" = 0 ]
