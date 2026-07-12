#!/usr/bin/env bash
# Shared helpers for the libvirt ProviderAdapter wrappers.
# Sourced by every script; never executed directly.

set -euo pipefail

# Resolve repo-relative paths regardless of caller's CWD.
_LIB_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export CLUSTER_ROOT="${CLUSTER_ROOT:-$(cd "${_LIB_DIR}/../.." && pwd)}"

# shellcheck source=/dev/null
source "${_LIB_DIR}/config.env"

export LIBVIRT_DEFAULT_URI="${LIBVIRT_DEFAULT_URI:-qemu:///system}"
export CLUSTER_TEMPLATES_XML="${_LIB_DIR}/templates"
export CLUSTER_CLOUDINIT_DIR="${_LIB_DIR}/cloud-init"
export CLUSTER_INVENTORY_DIR="${CLUSTER_ROOT}/infra/ansible/inventory"

# ── logging ───────────────────────────────────────────────────────────────
log()  { printf '\033[1;34m[cluster]\033[0m %s\n' "$*" >&2; }
warn() { printf '\033[1;33m[cluster:warn]\033[0m %s\n' "$*" >&2; }
die()  { printf '\033[1;31m[cluster:err]\033[0m %s\n' "$*" >&2; exit 1; }

# ── node identity helpers (deterministic; no state lookups) ───────────────
# Usage: node_name 1  -> cluster-node-01
node_name() { printf '%s-node-%02d' "${CLUSTER_PREFIX}" "$1"; }
# node_ip 1 -> 192.168.71.11
node_ip()   { printf '%s.%d' "${CLUSTER_NET_SUBNET}" "$(( CLUSTER_IP_BASE + $1 ))"; }
# node_mac 1 -> 52:54:00:71:00:0b   (byte = IP_BASE + i, matches the lease)
node_mac()  { printf '%s:%02x' "${CLUSTER_MAC_PREFIX}" "$(( CLUSTER_IP_BASE + $1 ))"; }

# Iterate 1..CLUSTER_NODE_COUNT
node_seq() { seq 1 "${CLUSTER_NODE_COUNT}"; }

# ── preconditions ─────────────────────────────────────────────────────────
require_bin() { command -v "$1" >/dev/null 2>&1 || die "missing '$1' — are you in the nix dev shell? (nix develop)"; }

preflight() {
  require_bin virsh
  require_bin qemu-img
  virsh version >/dev/null 2>&1 || die "cannot reach libvirt at ${LIBVIRT_DEFAULT_URI} (is libvirtd running / are you in the libvirtd group?)"
  [[ -r /dev/kvm ]] || die "/dev/kvm not accessible"
}

net_exists()    { virsh net-info "${CLUSTER_NET_NAME}" >/dev/null 2>&1; }
net_active()    { virsh net-info "${CLUSTER_NET_NAME}" 2>/dev/null | grep -q 'Active:.*yes'; }
dom_exists()    { virsh dominfo "$1" >/dev/null 2>&1; }
dom_running()   { virsh domstate "$1" 2>/dev/null | grep -q 'running'; }

ensure_dirs() { mkdir -p "${CLUSTER_TEMPLATE_DIR}" "${CLUSTER_OVERLAY_DIR}" "${CLUSTER_SEED_DIR}" "${CLUSTER_INVENTORY_DIR}"; }

# Block until a node's SSH port answers (or timeout_s elapses). A "started"
# domain isn't a USABLE node yet — cloud-init first boot takes ~30-90s — and
# ProviderAdapter.provision() is supposed to mean "ready", not "requested".
# Every provision() call path (fresh boot, restart, already-running) should
# call this before returning, or the very next phase (bootstrap) will race
# a VM that hasn't finished coming up and fail with a misleading SSH error.
# Usage: wait_ssh <ip> <name> [timeout_s=300]
wait_ssh() {
  local ip="$1" name="$2" timeout_s="${3:-300}"
  local deadline=$(( $(date +%s) + timeout_s ))
  until nc -z -w2 "${ip}" 22 2>/dev/null; do
    if [[ $(date +%s) -gt ${deadline} ]]; then
      warn "[${name}] SSH not up before ${timeout_s}s timeout (${ip})"
      return 1
    fi
    sleep 3
  done
  log "[${name}] SSH up on ${ip}"
  return 0
}

# ── network gate ──────────────────────────────────────────────────────────
# Any internet-reaching step must pass through here. Honours CLUSTER_ALLOW_NETWORK=1,
# otherwise prompts on a TTY, otherwise fails with instructions.
# Usage: require_network "download the Ubuntu base image"
require_network() {
  local what="${1:-perform a network operation}"
  if [[ "${CLUSTER_ALLOW_NETWORK:-0}" == "1" ]]; then return 0; fi
  if [[ -t 0 ]]; then
    warn "About to ${what} (reaches the internet)."
    read -r -p "  Proceed? [y/N] " ans
    [[ "${ans}" =~ ^[Yy]$ ]] && return 0
    die "declined by user"
  fi
  die "network op blocked: '${what}'. Set CLUSTER_ALLOW_NETWORK=1 in config.env (or run interactively) to allow."
}

# True when guests may hit the network on first boot (drives cloud-init packages).
network_allowed() { [[ "${CLUSTER_ALLOW_NETWORK:-0}" == "1" ]]; }
