# Outpost placement probe — runs ONCE PER RANK, under the same launcher
# rendering as the benchmark, and reports where that rank actually landed.
#
# The workload is swapped for this script and nothing else changes: same
# rankfile, same appfile, same per-rank environment. A probe launched any other
# way would be measuring a different placement than the one being verified.
#
# POSIX sh, no dependencies. Every value is read from the kernel or from the
# environment the rank was given; nothing is inferred. A source that is missing
# yields null, and the comparison then reports the gap rather than assuming the
# placement held.
#
# Writes <out>/preflight/rank-<N>.json, where <out> is $1 (default /out) — the
# job's output_dir, so the existing collect step ships it.
set -u

OUT="${1:-/out}/preflight"
mkdir -p "$OUT" 2>/dev/null || true

esc() { printf '%s' "${1:-}" | tr -d '\\"' | tr '\n' ' '; }
jstr() { if [ -n "${1:-}" ]; then printf '"%s"' "$(esc "$1")"; else printf 'null'; fi; }
jnum() { case "${1:-}" in ''|*[!0-9-]*) printf 'null' ;; *) printf '%s' "$1" ;; esac; }

field() { grep -i "^$1:" /proc/self/status 2>/dev/null | awk '{print $2}' | head -1; }

# Which rank am I? OpenMPI first, then the PMI variants a different launcher
# would set. Without one, this file would collide with every other rank's, so
# the PID is the last resort and the missing rank is recorded as such.
rank="${OMPI_COMM_WORLD_RANK:-${PMIX_RANK:-${PMI_RANK:-}}}"
local_rank="${OMPI_COMM_WORLD_LOCAL_RANK:-${MPI_LOCALRANKID:-}}"
world="${OMPI_COMM_WORLD_SIZE:-${PMI_SIZE:-}}"
name="${rank:-pid$$}"

# What the OS will actually let this process run on — the answer the plan is
# checked against. Cpus_allowed_list is the kernel's own view, read from inside
# whatever container and cgroup the rank ended up in.
allowed_cpus="$(field Cpus_allowed_list)"
allowed_mems="$(field Mems_allowed_list)"
# Which CPU it is on right now: field 39 of /proc/self/stat. A point sample, and
# only meaningful as a cross-check that it falls inside allowed_cpus.
current_cpu="$(awk '{print $39}' /proc/self/stat 2>/dev/null)"

# How wide the machine is, so an unbound rank is distinguishable from a rank
# bound to everything. Without this, "allowed = every CPU" and "binding was
# never applied" look identical.
online_cpus="$(head -1 /sys/devices/system/cpu/online 2>/dev/null | tr -d ' \t')"

# GPUs. CUDA_VISIBLE_DEVICES is what the launcher DELIVERED to this rank;
# nvidia-smi lists what the driver exposes on the node. They answer different
# questions and both are recorded — see the note in receipt.py about which
# claims each can and cannot support.
visible_env="${CUDA_VISIBLE_DEVICES:-}"
gpu_uuids=""; gpu_count=0
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_uuids="$(nvidia-smi --query-gpu=uuid --format=csv,noheader 2>/dev/null \
               | paste -sd, - 2>/dev/null)"
  gpu_count="$(nvidia-smi -L 2>/dev/null | grep -c '^GPU' || true)"
fi
[ -n "$gpu_count" ] || gpu_count=0

printf '{' > "$OUT/rank-$name.json"
{
  printf '"probe_version":"1",'
  printf '"hostname":%s,' "$(jstr "$(hostname 2>/dev/null || true)")"
  printf '"global_rank":%s,' "$(jnum "$rank")"
  printf '"local_rank":%s,' "$(jnum "$local_rank")"
  printf '"world_size":%s,' "$(jnum "$world")"
  printf '"pid":%s,' "$$"
  printf '"allowed_cpus":%s,' "$(jstr "$allowed_cpus")"
  printf '"allowed_mems":%s,' "$(jstr "$allowed_mems")"
  printf '"current_cpu":%s,' "$(jnum "$current_cpu")"
  printf '"online_cpus":%s,' "$(jstr "$online_cpus")"
  printf '"cuda_visible_devices":%s,' "$(jstr "$visible_env")"
  printf '"driver_gpu_uuids":%s,' "$(jstr "$gpu_uuids")"
  printf '"driver_gpu_count":%s' "$gpu_count"
  printf '}\n'
} >> "$OUT/rank-$name.json"

# Also to stdout, so a run whose artifacts never get collected still leaves the
# observation in the job log.
cat "$OUT/rank-$name.json"
