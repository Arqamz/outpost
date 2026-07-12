#!/usr/bin/env bash
# Destroy + undefine the cluster0 network. Refuses if VMs are still attached/running.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
preflight

if ! net_exists; then log "network '${CLUSTER_NET_NAME}' not defined; nothing to do"; exit 0; fi

running=""
for i in $(node_seq); do n="$(node_name "$i")"; dom_running "$n" && running+=" $n"; done
[[ -n "${running}" ]] && die "VMs still running:${running} — run cluster-down first"

virsh net-destroy "${CLUSTER_NET_NAME}" >/dev/null 2>&1 || true
virsh net-undefine "${CLUSTER_NET_NAME}" >/dev/null 2>&1 || true
log "network '${CLUSTER_NET_NAME}' destroyed + undefined"
