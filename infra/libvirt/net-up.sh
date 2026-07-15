#!/usr/bin/env bash
# Define + start the cluster0 NAT network with static DHCP leases for every node.
# Idempotent: re-running reconciles the definition.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
preflight
ensure_dirs

# Build the <host> static-lease block from config (deterministic MAC<->IP).
leases=""
for i in $(node_seq); do
  leases+="      <host mac='$(node_mac "$i")' name='$(node_name "$i")' ip='$(node_ip "$i")'/>\n"
done

# DHCP range spans the whole node block so leases are honoured. With 0 nodes
# (a host-only cluster) there is nothing to lease — emit no <dhcp> at all,
# since an inverted range (start .11 > end .10) is invalid network XML.
dhcp_block=""
if [[ "${CLUSTER_NODE_COUNT}" -gt 0 ]]; then
  dhcp_block="$(cat <<EOF
    <dhcp>
      <range start='$(node_ip 1)' end='$(node_ip "${CLUSTER_NODE_COUNT}")'/>
$(echo -e "${leases}")
    </dhcp>
EOF
)"
fi

xml="$(cat <<EOF
<network>
  <name>${CLUSTER_NET_NAME}</name>
  <forward mode='nat'/>
  <bridge name='${CLUSTER_NET_BRIDGE}' stp='on' delay='0'/>
  <ip address='${CLUSTER_HOST_IP}' netmask='255.255.255.0'>
${dhcp_block}
  </ip>
</network>
EOF
)"

tmp="$(mktemp --suffix=.xml)"; printf '%s\n' "${xml}" >"${tmp}"

if net_exists; then
  log "network '${CLUSTER_NET_NAME}' already present; resetting"
  # Always stop-then-undefine, tolerant of state: a transient net is removed by
  # destroy alone (undefine no-ops); a persistent net needs both. `|| true` so a
  # benign failure never aborts under `set -e`, and we never leave it half-defined.
  virsh net-destroy "${CLUSTER_NET_NAME}" >/dev/null 2>&1 || true
  virsh net-undefine "${CLUSTER_NET_NAME}" >/dev/null 2>&1 || true
fi

virsh net-define "${tmp}" >/dev/null
virsh net-autostart "${CLUSTER_NET_NAME}" >/dev/null
virsh net-start "${CLUSTER_NET_NAME}" >/dev/null
rm -f "${tmp}"

log "network '${CLUSTER_NET_NAME}' active on ${CLUSTER_NET_BRIDGE} (${CLUSTER_HOST_IP}/24), ${CLUSTER_NODE_COUNT} static leases"
# `|| true`: with 0 nodes there is no dhcp block for grep to show (pipefail).
virsh net-dumpxml "${CLUSTER_NET_NAME}" | { grep -E 'range|host mac' || true; } | sed 's/^/  /'
