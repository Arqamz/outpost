#cloud-config
# Rendered per-node by vm-define.sh. Keys-only SSH; passwordless sudo for the
# provisioning user so the ansible plays can `become`.
hostname: __HOSTNAME__
fqdn: __HOSTNAME__.cluster.local
manage_etc_hosts: true
preserve_hostname: false

users:
  - name: __USER__
    groups: [sudo]
    sudo: "ALL=(ALL) NOPASSWD:ALL"
    shell: /bin/bash
    lock_passwd: true
    ssh_authorized_keys:
      - __PUBKEY__

ssh_pwauth: false
disable_root: true

# The network-dependent package block is injected here by vm-define.sh ONLY when
# CLUSTER_ALLOW_NETWORK=1. When gated off, first boot touches the network for nothing
# (Ubuntu cloud images already ship python3, so ansible still works; static DHCP
# leases mean IP discovery does not need qemu-guest-agent).
__NET_BLOCK__
final_message: "cluster node __HOSTNAME__ up after $UPTIME s"
