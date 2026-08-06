#!/usr/bin/env bash
# install.sh — stand up the `ssh tashkil` gateway on the control-plane host.
#
# Creates a dedicated, shell-less `tashkil` unix user whose ONLY capability is
# the forced command in infra/gateway/tashkil-gw (the job portal). Authorizes
# one client public key per run. Puts the gateway user and the operator (whoever
# runs the reconciler daemon) in a shared `outpost` group and makes the store +
# drop-zone + logs group-writable, so a submit over ssh and the daemon's ticks
# share one FileStore.
#
# Idempotent: re-run to authorize more keys or fix permissions.
#
# Usage (as root):
#   sudo infra/gateway/install.sh --key ~/keys/laptop.pub
#   sudo infra/gateway/install.sh --key "ssh-ed25519 AAAA... alice@laptop"
#   sudo infra/gateway/install.sh --key laptop.pub --operator arqam --user tashkil
#
# After it runs, on the CLIENT: add the ssh alias + wrapper (see
# infra/gateway/README.md / docs/10-ssh-gateway.md), then `ssh tashkil help`.
set -euo pipefail

GW_USER="tashkil"
OPERATOR=""
GROUP="outpost"
KEY=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --key)      KEY="${2:?--key needs a pubkey file or string}"; shift 2 ;;
    --user)     GW_USER="${2:?}"; shift 2 ;;
    --operator) OPERATOR="${2:?}"; shift 2 ;;
    --group)    GROUP="${2:?}"; shift 2 ;;
    -h|--help)
      grep '^#' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown arg: $1" >&2; exit 2 ;;
  esac
done

[[ $EUID -eq 0 ]] || { echo "run as root (sudo)" >&2; exit 1; }
[[ -n "$KEY" ]]   || { echo "need --key <pubkey file or 'ssh-... ...' string>" >&2; exit 2; }

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GW="${REPO_ROOT}/infra/gateway/tashkil-gw"
[[ -x "$GW" ]] || chmod +x "$GW"

# The operator defaults to whoever owns the repo checkout — that's the account
# the reconciler daemon runs as and that already owns .var/.
if [[ -z "$OPERATOR" ]]; then
  OPERATOR="$(stat -c '%U' "$REPO_ROOT")"
fi

# Read the key: a file path -> its contents; otherwise treat the arg as the key.
if [[ -f "$KEY" ]]; then
  KEY_LINE="$(< "$KEY")"
else
  KEY_LINE="$KEY"
fi
[[ "$KEY_LINE" == ssh-* || "$KEY_LINE" == ecdsa-* || "$KEY_LINE" == sk-* ]] \
  || { echo "that doesn't look like an ssh public key: ${KEY_LINE:0:20}..." >&2; exit 2; }

echo "==> shared group '${GROUP}' (operator=${OPERATOR}, gateway user=${GW_USER})"
getent group "$GROUP" >/dev/null || groupadd "$GROUP"

echo "==> gateway user '${GW_USER}' (no login shell, no password)"
if ! id -u "$GW_USER" >/dev/null 2>&1; then
  useradd --create-home --shell /usr/sbin/nologin --user-group "$GW_USER"
fi
# nologin is fine: sshd runs the forced command via the user's shell, and
# /usr/sbin/nologin still execs a forced command. Put both users in the group.
usermod -aG "$GROUP" "$GW_USER"
usermod -aG "$GROUP" "$OPERATOR" || echo "  (couldn't add ${OPERATOR} to ${GROUP}; add it by hand)"

echo "==> authorized_keys with the forced command + hard restrictions"
GW_HOME="$(getent passwd "$GW_USER" | cut -d: -f6)"
SSH_DIR="${GW_HOME}/.ssh"
AK="${SSH_DIR}/authorized_keys"
install -d -m 700 -o "$GW_USER" -g "$GW_USER" "$SSH_DIR"
touch "$AK"
OPTS='command="'"$GW"'",no-port-forwarding,no-X11-forwarding,no-agent-forwarding,no-user-rc,no-pty'
LINE="${OPTS} ${KEY_LINE}"
# De-dupe on the key body (field 2..3) so re-running with the same key is a no-op.
KEY_BODY="$(awk '{print $(NF-1), $NF}' <<< "$KEY_LINE")"
if grep -qF "$KEY_BODY" "$AK" 2>/dev/null; then
  echo "  key already authorized — refreshing its forced-command line"
  grep -vF "$KEY_BODY" "$AK" > "${AK}.tmp" || true
  mv "${AK}.tmp" "$AK"
fi
printf '%s\n' "$LINE" >> "$AK"
chown "$GW_USER:$GW_USER" "$AK"
chmod 600 "$AK"

echo "==> group-writable store, drop-zone, and logs under .var/"
# The gateway (as ${GW_USER}) and the daemon (as ${OPERATOR}) share one
# FileStore; both must read+write it. setgid dirs so new files inherit ${GROUP},
# and umask 002 on the daemon side keeps them group-writable (see README).
# (tmp is included because env.sh does `mkdir -p .var/tmp` on every gateway call
# — as ${GW_USER} — so that dir must be group-writable too.)
for d in reconciler dropzone logs tmp; do
  p="${REPO_ROOT}/.var/${d}"
  install -d "$p"
  chgrp -R "$GROUP" "$p"
  chmod -R g+rwX "$p"
  find "$p" -type d -exec chmod g+s {} +
done

cat <<EOF

Gateway installed.

  forced command : ${GW}
  gateway user   : ${GW_USER}  (shell-less; forced command only)
  operator       : ${OPERATOR}  (runs the reconciler daemon)
  shared group   : ${GROUP}
  authorized     : ${KEY_BODY}

NEXT:
  1. Make sure the reconciler daemon runs with a group-writable umask so the
     ${GW_USER} user can read/update the store it writes:
         umask 002 && cluster reconcile --execute
     (or set UMask=002 in its systemd unit). ${OPERATOR} may need to re-login
     for '${GROUP}' membership to take effect.
  2. On the CLIENT, add to ~/.ssh/config:
         Host tashkil
             HostName <this-host-ip>
             User ${GW_USER}
             IdentityFile ~/.ssh/<the private key for the pubkey above>
     then:  ssh tashkil help
  3. Optional client sugar: put bin/tashkil on the client's PATH for
     'tashkil submit job.yaml' / 'tashkil fetch <id>'.

Revoke a client: remove its line from ${AK}.
EOF
