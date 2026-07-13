#!/usr/bin/env bash
# Simple MPI demo across all live VMs (bare OpenMPI, no reconciler).
# Compiles mpi_demo.c on the head node, distributes the binary to the others,
# and launches it with mpirun over the cluster0 TCP fabric.
#
# Prereqs: `make cluster-up` (VMs running) + `make bootstrap` (openmpi + gcc).
# Run from the repo root inside the nix dev shell.
source "$(cd "$(dirname "${BASH_SOURCE[0]}")/../infra/libvirt" && pwd)/lib.sh"
preflight

SSH_OPTS=(-i "${CLUSTER_SSH_PRIVKEY}" -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -o ConnectTimeout=10)
head_ip="$(node_ip 1)"
np="${1:-${CLUSTER_NODE_COUNT}}"          # ranks (default one per node)

ssh_head() { ssh "${SSH_OPTS[@]}" "${CLUSTER_SSH_USER}@${head_ip}" "$@"; }

log "head node = $(node_name 1) (${head_ip}); ranks = ${np}"

# 1) give the head node the cluster key + ssh config so mpirun can reach workers
log "staging cluster key + ssh config onto the head node"
ssh_head 'mkdir -p ~/.ssh && chmod 700 ~/.ssh'
scp "${SSH_OPTS[@]}" -q "${CLUSTER_SSH_PRIVKEY}" "${CLUSTER_SSH_USER}@${head_ip}:~/.ssh/id_ed25519"
ssh_head "chmod 600 ~/.ssh/id_ed25519; printf 'Host ${CLUSTER_NET_SUBNET}.*\n  User ${CLUSTER_SSH_USER}\n  IdentityFile ~/.ssh/id_ed25519\n  StrictHostKeyChecking no\n  UserKnownHostsFile /dev/null\n' > ~/.ssh/config; chmod 600 ~/.ssh/config"

# 2) compile on the head node
log "compiling mpi_demo.c on the head node (mpicc)"
scp "${SSH_OPTS[@]}" -q "$(dirname "${BASH_SOURCE[0]}")/mpi_demo.c" "${CLUSTER_SSH_USER}@${head_ip}:/tmp/mpi_demo.c"
ssh_head 'mpicc -O3 -o /tmp/mpi_demo /tmp/mpi_demo.c && echo "  built /tmp/mpi_demo"'

# 3) distribute the binary to the other nodes (head reaches them with the key)
log "distributing the binary to workers"
for i in $(seq 2 "${CLUSTER_NODE_COUNT}"); do
  ip="$(node_ip "$i")"
  ssh_head "scp -q -o StrictHostKeyChecking=no /tmp/mpi_demo ${CLUSTER_SSH_USER}@${ip}:/tmp/mpi_demo" \
    && log "  -> $(node_name "$i") (${ip})"
done

# 4) launch across all nodes, one rank per node, over TCP on the cluster0 subnet
log "launching mpirun over ${np} nodes"
echo "────────────────────────────────────────────────────────────"
ssh_head "mpirun -np ${np} --hostfile /etc/cluster/hostfile --map-by node \
  --mca btl tcp,self --mca btl_tcp_if_include ${CLUSTER_NET_SUBNET}.0/24 \
  --mca oob_tcp_if_include ${CLUSTER_NET_SUBNET}.0/24 \
  /tmp/mpi_demo"
echo "────────────────────────────────────────────────────────────"
log "done."
