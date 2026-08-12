# Outpost GPU-memory probe — runs ONCE PER RANK, inside the pod, before the
# workload — reporting the VRAM cap HAMi actually enforced on this container.
#
# WHY THIS HAS TO RUN INSIDE THE POD. `kubectl get pod -o yaml` shows
# CUDA_DEVICE_MEMORY_LIMIT is wired via envFrom a generated ConfigMap — but its
# VALUE is only resolved at container start (kubelet fetches the ConfigMap),
# and ld.so.preload's libvgpu.so is what actually enforces it against CUDA
# calls — the pod SPEC alone doesn't prove the cap took effect for THIS
# process. Verified live: the value carries a trailing unit letter ("1141m"),
# not a bare integer — stripped below, not rejected.
#
# POSIX sh, no dependencies beyond nvidia-smi (best-effort — absent yields
# null, never a guess). Prints ONE JSON line prefixed with a fixed marker so
# it is reliably extractable from a rank's combined log even mixed with real
# workload stdout.
set -u

esc() { printf '%s' "${1:-}" | tr -d '\\"' | tr '\n' ' '; }
jstr() { if [ -n "${1:-}" ]; then printf '"%s"' "$(esc "$1")"; else printf 'null'; fi; }
jnum() { case "${1:-}" in ''|*[!0-9-]*) printf 'null' ;; *) printf '%s' "$1" ;; esac; }

# HAMi-core's injected cap — what this container is ACTUALLY held to, as
# opposed to what the pod annotation asked for. Observed live as e.g. "1141m"
# (MiB with a trailing unit letter); strip any trailing non-digit suffix
# before treating it as a number, rather than letting jnum reject it outright.
cap_mb="${CUDA_DEVICE_MEMORY_LIMIT:-}"
cap_mb="$(printf '%s' "$cap_mb" | sed 's/[^0-9-]*$//')"

gpu_uuid=""; gpu_total_mb=""
if command -v nvidia-smi >/dev/null 2>&1; then
  line="$(nvidia-smi --query-gpu=uuid,memory.total --format=csv,noheader 2>/dev/null | head -1)"
  gpu_uuid="$(printf '%s' "$line" | awk -F', *' '{print $1}')"
  gpu_total_mb="$(printf '%s' "$line" | awk -F', *' '{print $2}' | awk '{print $1}')"
fi

printf '===GPU_MEMORY_PROBE===%s\n' "$(
  printf '{'
  printf '"probe_version":"1",'
  printf '"rank":%s,' "$(jnum "${RANK:-}")"
  printf '"hostname":%s,' "$(jstr "$(hostname 2>/dev/null || true)")"
  printf '"cuda_device_memory_limit_mb":%s,' "$(jnum "$cap_mb")"
  printf '"gpu_uuid":%s,' "$(jstr "$gpu_uuid")"
  printf '"gpu_total_mb":%s' "$(jnum "$gpu_total_mb")"
  printf '}'
)"
