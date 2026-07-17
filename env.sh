# Environment for running Outpost WITHOUT the nix dev shell (e.g. a native
# Ubuntu host — full walkthrough in docs/07-ubuntu-setup.md). Mirrors what
# shell.nix's shellHook exports, minus the NixOS-specific GPU binds: on a
# normal distro apptainer --nv detects the driver by itself, so
# CLUSTER_GPU_BINDS deliberately stays unset/empty here.
#
# Usage:  source env.sh
export CLUSTER_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export LIBVIRT_DEFAULT_URI="${LIBVIRT_DEFAULT_URI:-qemu:///system}"
export PATH="$CLUSTER_ROOT/bin:$PATH"
export PYTHONPATH="$CLUSTER_ROOT${PYTHONPATH:+:$PYTHONPATH}"
# apptainer image cache stays inside the repo (.var is gitignored)
export APPTAINER_CACHEDIR="$CLUSTER_ROOT/.var/apptainer-cache"

echo "outpost env ready: CLUSTER_ROOT=$CLUSTER_ROOT  (libvirt: $LIBVIRT_DEFAULT_URI)"
