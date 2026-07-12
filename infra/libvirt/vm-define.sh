#!/usr/bin/env bash
# ProviderAdapter.Provision (libvirt) for a single node.
# Usage: vm-define.sh <node-index>
#   - creates a qcow2 overlay backed by the shared template
#   - renders + builds a cloud-init NoCloud seed ISO
#   - renders the domain XML and `virsh define` + `virsh start`s it
# Idempotent-ish: refuses to clobber a running domain.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
preflight
ensure_dirs

i="${1:?usage: vm-define.sh <node-index>}"
name="$(node_name "$i")"; ip="$(node_ip "$i")"; mac="$(node_mac "$i")"

[[ -f "${CLUSTER_TEMPLATE_QCOW}" ]] || die "base template missing — run build-template.sh first"
[[ -r "${CLUSTER_SSH_PUBKEY}" ]]    || die "SSH pubkey not readable: ${CLUSTER_SSH_PUBKEY}"

# provision() means "ready", not "requested" — every exit path below waits
# for SSH before returning, so the caller (the reconciler) never moves on to
# bootstrap against a node that hasn't actually finished coming up.
if dom_running "${name}"; then
  log "${name} already running"
  wait_ssh "${ip}" "${name}" || die "${name} running but SSH never came up (${ip})"
  exit 0
fi
if dom_exists "${name}"; then
  warn "${name} defined but stopped; starting"
  virsh start "${name}" >/dev/null
  wait_ssh "${ip}" "${name}" || die "${name} started but SSH never came up (${ip})"
  exit 0
fi

overlay="${CLUSTER_OVERLAY_DIR}/${name}.qcow2"
seed="${CLUSTER_SEED_DIR}/${name}-seed.iso"

# 1) thin overlay backed by the template
log "[${name}] creating overlay (${CLUSTER_DISK_GB}G, backed by template)"
qemu-img create -q -f qcow2 -F qcow2 -b "${CLUSTER_TEMPLATE_QCOW}" "${overlay}" "${CLUSTER_DISK_GB}G" >/dev/null

# 2) cloud-init seed
pubkey="$(cat "${CLUSTER_SSH_PUBKEY}")"

# Network-dependent package block, gated by CLUSTER_ALLOW_NETWORK.
if network_allowed; then
  net_block=$'package_update: true\npackages:\n  - qemu-guest-agent\n  - python3\nruncmd:\n  - [systemctl, enable, --now, qemu-guest-agent]'
  log "[${name}] cloud-init: network allowed — first boot will apt-update + install guest-agent"
else
  net_block='# (network gated off: no apt on first boot)'
fi

ud="$(mktemp)"; md="$(mktemp)"
# Write the multi-line block to a file and splice it in (avoids sed newline pain).
nb="$(mktemp)"; printf '%s\n' "${net_block}" >"${nb}"
sed -e "s|__HOSTNAME__|${name}|g" -e "s|__USER__|${CLUSTER_SSH_USER}|g" \
    -e "s|__PUBKEY__|${pubkey}|g" -e "/__NET_BLOCK__/{r ${nb}
d}" "${CLUSTER_CLOUDINIT_DIR}/user-data.tpl" >"${ud}"
rm -f "${nb}"
sed -e "s|__HOSTNAME__|${name}|g" "${CLUSTER_CLOUDINIT_DIR}/meta-data.tpl" >"${md}"
log "[${name}] building cloud-init seed"
cloud-localds "${seed}" "${ud}" "${md}"
rm -f "${ud}" "${md}"

# 3) domain XML
domxml="$(mktemp --suffix=.xml)"
sed -e "s|__NAME__|${name}|g" -e "s|__MEM_MB__|${CLUSTER_MEM_MB}|g" \
    -e "s|__VCPUS__|${CLUSTER_VCPUS}|g" -e "s|__DISK__|${overlay}|g" \
    -e "s|__SEED__|${seed}|g" -e "s|__MAC__|${mac}|g" -e "s|__NET__|${CLUSTER_NET_NAME}|g" \
    "${CLUSTER_TEMPLATES_XML}/domain.xml.tpl" >"${domxml}"

virt-xml-validate "${domxml}" >/dev/null 2>&1 || warn "[${name}] domain XML failed schema validation (continuing)"

log "[${name}] virsh define + start  (mac=${mac} -> ip=${ip})"
virsh define "${domxml}" >/dev/null
virsh start  "${name}" >/dev/null
rm -f "${domxml}"

log "[${name}] started, waiting for SSH (cloud-init first boot can take ~1-2 min)…"
wait_ssh "${ip}" "${name}" || die "[${name}] started but SSH never came up (${ip}) — check: virsh console ${name}"
