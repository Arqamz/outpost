#!/usr/bin/env bash
# Failure-injection harness — hard-kills a node mid-job to exercise quarantine
# and clean job failure in the reconciler. This does NOT clean up — the
# reconciler is expected to observe the dead node and quarantine it.
#
# Usage: inject-failure.sh <node-index> [--mode kill|pause|netcut]
#   kill   : virsh destroy (abrupt power loss)   [default]
#   pause  : virsh suspend  (freeze, mimics a hung host)
#   netcut : detach the NIC   (mimics a fabric/partition failure)
source "$(dirname "${BASH_SOURCE[0]}")/lib.sh"
preflight

i="${1:?usage: inject-failure.sh <node-index> [--mode kill|pause|netcut]}"
mode="kill"; [[ "${2:-}" == "--mode" ]] && mode="${3:-kill}"
name="$(node_name "$i")"; mac="$(node_mac "$i")"
dom_running "${name}" || die "${name} is not running"

case "${mode}" in
  kill)   warn "[${name}] INJECT: virsh destroy (abrupt kill)"; virsh destroy "${name}" >/dev/null ;;
  pause)  warn "[${name}] INJECT: virsh suspend (freeze)";       virsh suspend "${name}" >/dev/null ;;
  netcut) warn "[${name}] INJECT: detaching NIC (partition)";
          virsh detach-interface "${name}" network --mac "${mac}" --live >/dev/null ;;
  *) die "unknown mode '${mode}'" ;;
esac
log "[${name}] failure injected (${mode}). Reconciler should quarantine it."
