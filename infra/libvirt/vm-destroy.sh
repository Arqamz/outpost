#!/usr/bin/env bash
# ProviderAdapter.Deprovision (libvirt) for a single node.
# Usage: vm-destroy.sh <node-index> [--keep-disk]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
preflight

i="${1:?usage: vm-destroy.sh <node-index> [--keep-disk]}"
keep_disk="${2:-}"
name="$(node_name "$i")"

if ! dom_exists "${name}"; then log "${name} not defined; nothing to do"; exit 0; fi

dom_running "${name}" && { log "[${name}] destroy (force off)"; virsh destroy "${name}" >/dev/null; }
# --remove-all-storage would also drop the shared template; we only manage the overlay/seed ourselves.
virsh undefine "${name}" --nvram >/dev/null 2>&1 || virsh undefine "${name}" >/dev/null

if [[ "${keep_disk}" != "--keep-disk" ]]; then
  rm -f "${CLUSTER_OVERLAY_DIR}/${name}.qcow2" "${CLUSTER_SEED_DIR}/${name}-seed.iso"
  log "[${name}] undefined + overlay/seed removed"
else
  log "[${name}] undefined (disk kept)"
fi
