#!/usr/bin/env bash
# Build an apptainer SIF, picking the right rootless build mode for the host.
#
# Two worlds:
#   * A host with a SETUID `newuidmap` (Ubuntu's default, or the apptainer-suid
#     package) — apptainer's normal subuid fakeroot works; just `apptainer build`.
#   * This NixOS dev host — the nix apptainer ships a non-setuid `starter` and
#     there is no setuid newuidmap/newgidmap, so subuid fakeroot can't set up its
#     id map. Apptainer falls back to a proot-emulated root build, which needs:
#       - `proot` on PATH (pinned in shell.nix), and
#       - the flags below (skip the subuid map + the in-image fakeroot command), and
#       - the def's %post to disable apt's sandbox (APT::Sandbox::User "root"),
#         because proot's single-uid mapping can't emulate apt dropping to the
#         unprivileged `_apt` uid (see demo/*.def).
#
# Usage: demo/build-sif.sh <out.sif> <def>
set -euo pipefail
sif=${1:?usage: build-sif.sh <out.sif> <def>}
def=${2:?usage: build-sif.sh <out.sif> <def>}

setuid_newuidmap() {
  local p; p=$(command -v newuidmap 2>/dev/null) || return 1
  [ -u "$p" ]   # setuid bit set?
}

flags=()
if setuid_newuidmap; then
  echo "[build-sif] setuid newuidmap present -> native fakeroot build"
elif command -v proot >/dev/null 2>&1; then
  echo "[build-sif] no setuid fakeroot on this host -> proot rootless build"
  flags=(--ignore-subuid --ignore-fakeroot-command)
else
  echo "[build-sif] no setuid fakeroot AND no proot on PATH." >&2
  echo "            Enter the dev shell ('nix develop', which pins proot) or" >&2
  echo "            install proot; then re-run. Trying a native build anyway..." >&2
fi

set -x
exec apptainer build "${flags[@]}" "$sif" "$def"
