# Outpost topology probe — runs ON an allocated node and prints one JSON object
# describing the CPUs, NUMA nodes and GPUs that node can actually use.
#
# POSIX sh only, and nothing is installed to run it: the bootstrap role provides
# apptainer, MPI and the ssh fabric, but NOT hwloc or numactl, so everything here
# comes from /sys, /proc, and nvidia-smi if it happens to be present.
#
# BEST-EFFORT, NEVER INVENTS. A missing source yields an empty list or null, and
# the caller decides whether that is fatal. A probe that guessed a topology would
# produce a placement that looks resolved and binds to the wrong cores.
#
# ALLOCATION-AWARE, not host-wide: `allowed_cpus` is the effective cpuset of THIS
# process, so a job confined by a cgroup or a container reports what it may use,
# not what the machine has.
#
# COLLECT HERE, INTERPRET IN PYTHON. Stable machine-readable sources are parsed
# inline; `nvidia-smi topo -m` is a human-readable table whose columns move with
# the GPU and NIC count, so it is captured verbatim and parsed where that
# variation can be unit-tested against real fixtures instead of guessed at.
set -u

esc() { printf '%s' "${1:-}" | tr -d '\\"' | tr '\n' ' '; }
jstr() { if [ -n "${1:-}" ]; then printf '"%s"' "$(esc "$1")"; else printf 'null'; fi; }
jnum() { case "${1:-}" in ''|*[!0-9-]*) printf 'null' ;; *) printf '%s' "$1" ;; esac; }

# `nvidia-smi topo -m` wraps its header row in ANSI SGR codes (ESC [ 4 m ...
# ESC [ 0 m) UNCONDITIONALLY, even when piped to a non-tty — verified on real
# hardware. Left in, the raw ESC byte is invalid inside a JSON string (jblob's
# own escaping below only covers \, ", tab, CR), and even escaped it would
# leave "[4mGPU0" as the header cell text, which topology.py's column parser
# does not recognise as a GPU/NIC row — so the CPU-affinity table would
# silently parse as empty. Strip the whole escape sequence, not just the ESC
# byte, before any of that.
_esc_char="$(printf '\033')"

# A multi-line blob as ONE JSON string: newlines escaped rather than flattened,
# because the meaning of a topology matrix is in its rows.
jblob() {
  if [ -z "${1:-}" ]; then printf 'null'; return; fi
  printf '"'
  printf '%s' "$1" \
    | sed -e "s/${_esc_char}\\[[0-9;]*m//g" \
          -e 's/\\/\\\\/g' -e 's/"/\\"/g' -e 's/\t/    /g' -e 's/\r//g' \
    | awk 'NR>1{printf "\\n"} {printf "%s", $0}'
  printf '"'
}

read_first() { [ -r "$1" ] && head -1 "$1" 2>/dev/null | tr -d ' \t' || printf ''; }

# ── what kind of machine are we looking at ────────────────────────────────
# A VM shows a synthetic topology (flat, often single-socket, no real NUMA), and
# a container may see the host's /sys while being confined to a slice of it.
# Recording which case this is stops a plan from claiming host-level precision
# it cannot have.
scope="host"
if [ -f /.dockerenv ] || [ -n "${APPTAINER_CONTAINER:-}${SINGULARITY_CONTAINER:-}" ]; then
  scope="container"
elif grep -qE '(docker|lxc|kubepods|containerd)' /proc/1/cgroup 2>/dev/null; then
  scope="container"
else
  virt=""
  if command -v systemd-detect-virt >/dev/null 2>&1; then
    virt="$(systemd-detect-virt 2>/dev/null || true)"
  fi
  [ -z "$virt" ] || [ "$virt" = "none" ] || scope="guest"
  if [ "$scope" = "host" ] && [ -r /sys/class/dmi/id/product_name ]; then
    case "$(cat /sys/class/dmi/id/product_name 2>/dev/null)" in
      *QEMU*|*KVM*|*VirtualBox*|*VMware*|*Bochs*|*Xen*) scope="guest" ;;
    esac
  fi
fi

# ── the cpuset this job may actually use ──────────────────────────────────
# cgroup v2, then v1, then the process affinity mask. Each is narrower and more
# truthful than "every CPU in /sys".
allowed=""
for f in /sys/fs/cgroup/cpuset.cpus.effective \
         /sys/fs/cgroup/cpuset/cpuset.effective_cpus \
         /sys/fs/cgroup/cpuset/cpuset.cpus; do
  [ -r "$f" ] && allowed="$(read_first "$f")" && [ -n "$allowed" ] && break
done
if [ -z "$allowed" ]; then
  allowed="$(grep -i '^Cpus_allowed_list:' /proc/self/status 2>/dev/null \
             | awk '{print $2}' || true)"
fi
if [ -z "$allowed" ] && command -v taskset >/dev/null 2>&1; then
  allowed="$(taskset -cp $$ 2>/dev/null | sed -n 's/.*: *//p')"
fi

online="$(read_first /sys/devices/system/cpu/online)"

# ── per-processing-unit topology ──────────────────────────────────────────
# core + socket identify a PHYSICAL core; two processing units sharing both are
# SMT siblings, which is what `smt: physical_only` must not double-count. The
# cpuN/nodeM symlink gives NUMA without expanding any cpulist ranges.
cpus=""
for d in /sys/devices/system/cpu/cpu[0-9]*; do
  [ -d "$d/topology" ] || continue
  id="${d##*/cpu}"
  core="$(read_first "$d/topology/core_id")"
  sock="$(read_first "$d/topology/physical_package_id")"
  numa=""
  for n in "$d"/node[0-9]*; do
    [ -e "$n" ] || continue
    numa="${n##*/node}"
    break
  done
  [ -n "$cpus" ] && cpus="$cpus,"
  cpus="$cpus{\"id\":$(jnum "$id"),\"core\":$(jnum "$core"),\"socket\":$(jnum "$sock"),\"numa\":$(jnum "$numa")}"
done

# ── NUMA nodes and their memory ───────────────────────────────────────────
numa_nodes=""
for d in /sys/devices/system/node/node[0-9]*; do
  [ -d "$d" ] || continue
  id="${d##*/node}"
  cpulist="$(read_first "$d/cpulist")"
  # "Node 0 MemTotal:  32900000 kB" -> MiB
  mem_kb="$(awk '/MemTotal/ {print $4; exit}' "$d/meminfo" 2>/dev/null || true)"
  mem_mib=""
  [ -n "$mem_kb" ] && mem_mib="$((mem_kb / 1024))"
  [ -n "$numa_nodes" ] && numa_nodes="$numa_nodes,"
  numa_nodes="$numa_nodes{\"id\":$(jnum "$id"),\"cpulist\":$(jstr "$cpulist"),\"memory_mib\":$(jnum "$mem_mib")}"
done

# ── GPUs ──────────────────────────────────────────────────────────────────
# Identity comes from the CSV query interface, which is stable and designed to
# be parsed. NUMA from sysfs is the FALLBACK; the driver's own view in the
# topology matrix below is preferred when it is available, since that is what
# actually decides which cores are local to a device.
gpus=""
topo_matrix=""
if command -v nvidia-smi >/dev/null 2>&1; then
  gpu_lines="$(nvidia-smi --query-gpu=index,uuid,pci.bus_id,memory.total,name \
               --format=csv,noheader,nounits 2>/dev/null || true)"
  OLDIFS="$IFS"
  IFS='
'
  for line in $gpu_lines; do
    [ -n "$line" ] || continue
    idx="$(printf '%s' "$line" | awk -F', *' '{print $1}')"
    uuid="$(printf '%s' "$line" | awk -F', *' '{print $2}')"
    pci="$(printf '%s' "$line" | awk -F', *' '{print $3}')"
    mem="$(printf '%s' "$line" | awk -F', *' '{print $4}')"
    name="$(printf '%s' "$line" | awk -F', *' '{print $5}')"
    # nvidia-smi prints 00000000:17:00.0; sysfs wants 0000:17:00.0.
    bdf="$(printf '%s' "$pci" | sed 's/^0000//' | tr 'A-Z' 'a-z')"
    gnuma="$(read_first "/sys/bus/pci/devices/$bdf/numa_node")"
    # sysfs reports -1 when the platform exposes no affinity; that is "unknown",
    # not "node -1", so it must not become a placement input.
    [ "$gnuma" = "-1" ] && gnuma=""
    [ -n "$gpus" ] && gpus="$gpus,"
    gpus="$gpus{\"index\":$(jnum "$idx"),\"uuid\":$(jstr "$uuid"),\"pci_bus_id\":$(jstr "$pci"),\"memory_mib\":$(jnum "$mem"),\"name\":$(jstr "$name"),\"numa\":$(jnum "$gnuma")}"
  done
  IFS="$OLDIFS"

  # The driver's own affinity + interconnect view. This is the authoritative
  # source for `closest_to_gpu` (its "CPU Affinity" column is the cores local to
  # each device, without inferring them through NUMA), it is the ONLY source for
  # GPU-to-GPU link class — which is what an NCCL collective actually rides — and
  # its NIC rows are what a later `closest_to_nic` policy will read.
  topo_matrix="$(nvidia-smi topo -m 2>/dev/null || true)"
fi

# ── the launcher that will actually place the ranks ───────────────────────
launcher_type=""; launcher_version=""
if command -v mpirun >/dev/null 2>&1; then
  first="$(mpirun --version 2>/dev/null | head -1)"
  case "$first" in
    *"Open MPI"*|*"OpenRTE"*) launcher_type="openmpi" ;;
    *MPICH*) launcher_type="mpich" ;;
    *) launcher_type="unknown" ;;
  esac
  launcher_version="$(printf '%s' "$first" | grep -oE '[0-9]+\.[0-9]+(\.[0-9]+)?' | head -1)"
fi

printf '{'
printf '"probe_version":"1",'
printf '"hostname":%s,' "$(jstr "$(hostname 2>/dev/null || true)")"
printf '"scope":%s,' "$(jstr "$scope")"
printf '"allowed_cpus":%s,' "$(jstr "$allowed")"
printf '"online_cpus":%s,' "$(jstr "$online")"
printf '"cpus":[%s],' "$cpus"
printf '"numa":[%s],' "$numa_nodes"
printf '"gpus":[%s],' "$gpus"
printf '"topo_matrix":%s,' "$(jblob "$topo_matrix")"
printf '"launcher":{"type":%s,"version":%s}' \
  "$(jstr "$launcher_type")" "$(jstr "$launcher_version")"
printf '}\n'
