#!/usr/bin/env bash
# Build the shared base qcow2 template once. Every VM boots a thin overlay of
# this file, so bring-up/teardown is cheap.
#
# Steps: download the Ubuntu cloud image -> (optional) verify SHA256 ->
#        convert/resize into the base template. cloud-init handles the rest at
#        first boot. Nothing job-specific is baked in here.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
require_bin qemu-img
require_bin wget
ensure_dirs

raw="${CLUSTER_TEMPLATE_DIR}/$(basename "${CLUSTER_IMAGE_URL}")"

if [[ -f "${CLUSTER_TEMPLATE_QCOW}" ]]; then
  log "template already present: ${CLUSTER_TEMPLATE_QCOW}"
  log "  (delete it to force a rebuild)"
  exit 0
fi

if [[ ! -f "${raw}" ]]; then
  require_network "download the Ubuntu base image (~600 MB)"
  log "downloading base image -> ${raw}"
  wget -q --show-progress -O "${raw}.part" "${CLUSTER_IMAGE_URL}"
  mv "${raw}.part" "${raw}"
else
  log "base image already downloaded: ${raw}"
fi

if [[ -n "${CLUSTER_IMAGE_SHA256}" ]]; then
  log "verifying SHA256"
  echo "${CLUSTER_IMAGE_SHA256}  ${raw}" | sha256sum -c - || die "checksum mismatch"
fi

log "converting -> ${CLUSTER_TEMPLATE_QCOW}"
qemu-img convert -O qcow2 "${raw}" "${CLUSTER_TEMPLATE_QCOW}"
log "resizing virtual size to ${CLUSTER_DISK_GB}G"
qemu-img resize "${CLUSTER_TEMPLATE_QCOW}" "${CLUSTER_DISK_GB}G" >/dev/null

log "template ready:"
qemu-img info "${CLUSTER_TEMPLATE_QCOW}" | sed 's/^/  /'
