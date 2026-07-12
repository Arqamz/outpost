#!/usr/bin/env bash
# Tear down every node (overlays + seeds removed). Leaves the network + template.
# Usage: cluster-down.sh [--keep-disk]
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
preflight
here="$(dirname "${BASH_SOURCE[0]}")"

for i in $(node_seq); do "${here}/vm-destroy.sh" "$i" "${1:-}"; done
log "cluster down. (network '${CLUSTER_NET_NAME}' + base template left intact)"
