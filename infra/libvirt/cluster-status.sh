#!/usr/bin/env bash
# Observe the cluster: domain state, expected IP, live lease, SSH reachability.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
preflight

printf '%-14s %-10s %-16s %-8s %-6s\n' NODE STATE IP LEASE SSH
printf '%-14s %-10s %-16s %-8s %-6s\n' "----" "-----" "--" "-----" "---"
for i in $(node_seq); do
  name="$(node_name "$i")"; ip="$(node_ip "$i")"
  state="absent"; dom_exists "${name}" && state="$(virsh domstate "${name}" 2>/dev/null)"
  lease="no"; grep -q "${ip}" <<<"$(virsh net-dhcp-leases "${CLUSTER_NET_NAME}" 2>/dev/null)" && lease="yes"
  ssh="-"; [[ "${state}" == "running" ]] && { nc -z -w1 "${ip}" 22 2>/dev/null && ssh="up" || ssh="down"; }
  printf '%-14s %-10s %-16s %-8s %-6s\n' "${name}" "${state}" "${ip}" "${lease}" "${ssh}"
done
echo
log "network: $(net_active && echo active || echo inactive)   template: $([[ -f ${CLUSTER_TEMPLATE_QCOW} ]] && echo present || echo MISSING)"
