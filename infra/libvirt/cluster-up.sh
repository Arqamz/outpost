#!/usr/bin/env bash
# Bring up the whole cluster: ensure network + template, then define/start all
# nodes. Writes the ansible inventory once they're up.
# Usage: cluster-up.sh
#
# vm-define.sh itself blocks each node until SSH answers (see lib.sh:wait_ssh),
# so by the time this returns every node is genuinely reachable, not just
# "virsh start"-ed. Run all 8 in parallel (background + wait) rather than one
# at a time — sequentially, 8 nodes x ~60-90s of cloud-init would take 8-12
# minutes; in parallel it's however long the SLOWEST one takes.
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
preflight
here="$(dirname "${BASH_SOURCE[0]}")"

net_active || { log "network down; bringing it up"; "${here}/net-up.sh"; }
[[ -f "${CLUSTER_TEMPLATE_QCOW}" ]] || die "no base template — run: make template"

log "defining + starting all nodes in parallel (each waits for its own SSH)…"
pids=()
for i in $(node_seq); do
  "${here}/vm-define.sh" "$i" &
  pids+=("$!")
done
fail=0
for pid in "${pids[@]}"; do
  wait "${pid}" || fail=1
done
[[ ${fail} -eq 0 ]] || die "one or more nodes failed to come up (see warnings above)"

"${here}/gen-inventory.sh"
log "cluster up. next: make bootstrap"
