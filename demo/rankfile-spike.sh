#!/usr/bin/env bash
# Does OpenMPI's --rankfile binding survive the `apptainer exec` boundary?
#
# WHY THIS EXISTS. The launcher compiler (reconciler/launcher.py) emits a
# rankfile and lets OpenMPI bind each rank to specific cores, then runs
# `apptainer exec` as the per-rank command. That assumes the affinity mask
# OpenMPI sets on the rank process is inherited by the container's processes.
# It is plausible — apptainer is not a new namespace for CPU affinity — but it
# has NOT been verified, and if it is wrong the whole CPU-binding half of the
# design has to move to per-rank `taskset` inside the appfile line instead.
#
# This answers that question and nothing else. It needs a multi-core Linux box
# with mpirun + apptainer. It does NOT need a GPU or more than one node.
#
# Usage:  demo/rankfile-spike.sh <image>
#   <image>  any SIF path or docker:// ref with a shell (e.g. demo/mpi_demo.sif,
#            or docker://alpine:3.20 — note that pulling is a deliberate act;
#            this cluster never pulls on its own).
set -uo pipefail

IMAGE="${1:-demo/mpi_demo.sif}"
WORK="$(mktemp -d)"
trap 'rm -rf "$WORK"' EXIT

say() { printf '\n\033[1m== %s\033[0m\n' "$*"; }
fail() { printf '\033[31m%s\033[0m\n' "$*"; }
pass() { printf '\033[32m%s\033[0m\n' "$*"; }

# ── prerequisites ─────────────────────────────────────────────────────────
for tool in mpirun apptainer taskset nproc; do
  command -v "$tool" >/dev/null 2>&1 || { fail "missing $tool"; exit 1; }
done
CORES="$(nproc)"
[ "$CORES" -ge 4 ] || { fail "need at least 4 processing units, found $CORES"; exit 1; }
if [ ! -e "$IMAGE" ] && [[ "$IMAGE" != *://* ]]; then
  fail "no such image: $IMAGE  (pass one as \$1, e.g. demo/mpi_demo.sif)"
  exit 1
fi

echo "mpirun : $(mpirun --version 2>&1 | head -1)"
echo "image  : $IMAGE"
echo "cpus   : $CORES"

# What each rank reports about itself. Cpus_allowed_list is the kernel's own
# answer, read from inside whatever context the process ended up in.
PROBE='printf "rank %s pid %s cpus %s\n" "${OMPI_COMM_WORLD_RANK:-?}" "$$" \
  "$(grep -i ^Cpus_allowed_list: /proc/self/status | awk "{print \$2}")"'

# ── control: no binding requested ─────────────────────────────────────────
say "CONTROL — no rankfile, no taskset (what happens by default)"
mpirun -np 2 --oversubscribe --bind-to none \
  apptainer exec "$IMAGE" sh -c "$PROBE" 2>&1 | sed 's/^/  /'

# ── A: OpenMPI rankfile, containerised ────────────────────────────────────
# socket:core form — the same form launcher.slot_expression() emits.
cat > "$WORK/rankfile" <<EOF
rank 0=localhost slot=0:0
rank 1=localhost slot=0:1
EOF
say "A — mpirun --rankfile, per-rank command is 'apptainer exec'"
echo "  rankfile:"; sed 's/^/    /' "$WORK/rankfile"
echo "  what OpenMPI says it bound, then what the container observes:"
mpirun -np 2 --rankfile "$WORK/rankfile" --report-bindings \
  apptainer exec "$IMAGE" sh -c "$PROBE" 2>&1 | sed 's/^/    /'

# ── A': the same rankfile with NO container, as the reference ─────────────
say "A' — same rankfile, no container (does binding work at all here?)"
mpirun -np 2 --rankfile "$WORK/rankfile" --report-bindings \
  sh -c "$PROBE" 2>&1 | sed 's/^/    /'

# ── B: the fallback — taskset inside a per-rank appfile line ──────────────
# The probe is written to a SCRIPT rather than inlined: an appfile line is split
# on whitespace with no shell quoting, so a multi-word command becomes several
# malformed app contexts. (Getting this wrong here is what surfaced the same bug
# in the launcher compiler.)
printf '#!/bin/sh\n%s\n' "$PROBE" > "$WORK/probe.sh"
chmod +x "$WORK/probe.sh"
cat > "$WORK/appfile" <<EOF
-np 1 taskset -c 0 apptainer exec --bind $WORK:$WORK $IMAGE /bin/sh $WORK/probe.sh
-np 1 taskset -c 1 apptainer exec --bind $WORK:$WORK $IMAGE /bin/sh $WORK/probe.sh
EOF
say "B — FALLBACK: taskset per rank in an appfile, containerised"
mpirun --app "$WORK/appfile" 2>&1 | sed 's/^/    /'

# ── C: does apptainer itself narrow the mask? ─────────────────────────────
say "C — taskset -> apptainer, no MPI at all (isolates the container boundary)"
echo -n "  outside: "; taskset -c 2 sh -c 'grep -i ^Cpus_allowed_list: /proc/self/status'
echo -n "  inside : "; taskset -c 2 apptainer exec "$IMAGE" sh -c \
  'grep -i ^Cpus_allowed_list: /proc/self/status'

cat <<'VERDICT'

== HOW TO READ THIS ==============================================
  A  shows 'cpus 0' for rank 0 and 'cpus 1' for rank 1
     -> rankfile binding SURVIVES the container. The current design in
        reconciler/launcher.py is correct; nothing changes.

  A  shows both ranks with the full cpu list, but A' shows 0 and 1
     -> binding works but is LOST at the apptainer boundary. Switch the
        OpenMPI adapter to emit taskset per appfile line (form B) and drop
        --rankfile. B's output confirms whether that fallback works.

  A' also shows the full list
     -> the rankfile is not being honoured at all on this build; check the
        slot syntax against `man mpirun` for THIS OpenMPI version before
        concluding anything about containers.

  C  differing between outside and inside
     -> apptainer itself is resetting affinity, which would invalidate both
        A and B and means binding must happen inside the container instead.
==================================================================
VERDICT
