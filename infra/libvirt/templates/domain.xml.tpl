<!-- Guest domain. Rendered by vm-define.sh (placeholders __X__).
     KVM-accelerated, host-passthrough CPU, virtio disk/net, serial console,
     and a qemu-guest-agent channel so we can read the leased IP reliably. -->
<domain type='kvm'>
  <name>__NAME__</name>
  <memory unit='MiB'>__MEM_MB__</memory>
  <currentMemory unit='MiB'>__MEM_MB__</currentMemory>
  <vcpu placement='static'>__VCPUS__</vcpu>
  <os>
    <type arch='x86_64' machine='q35'>hvm</type>
    <boot dev='hd'/>
  </os>
  <features>
    <acpi/>
    <apic/>
  </features>
  <cpu mode='host-passthrough' check='none' migratable='off'/>
  <clock offset='utc'>
    <timer name='rtc' tickpolicy='catchup'/>
    <timer name='pit' tickpolicy='delay'/>
    <timer name='hpet' present='no'/>
  </clock>
  <on_poweroff>destroy</on_poweroff>
  <on_reboot>restart</on_reboot>
  <on_crash>destroy</on_crash>
  <devices>
    <emulator>/run/current-system/sw/bin/qemu-system-x86_64</emulator>

    <!-- Root disk: qcow2 overlay backed by the shared base template. -->
    <disk type='file' device='disk'>
      <driver name='qemu' type='qcow2' cache='none' discard='unmap'/>
      <source file='__DISK__'/>
      <target dev='vda' bus='virtio'/>
    </disk>

    <!-- cloud-init NoCloud seed (read-only). -->
    <disk type='file' device='cdrom'>
      <driver name='qemu' type='raw'/>
      <source file='__SEED__'/>
      <target dev='sda' bus='sata'/>
      <readonly/>
    </disk>

    <!-- NIC on the cluster0 NAT network; MAC pins the static DHCP lease. -->
    <interface type='network'>
      <mac address='__MAC__'/>
      <source network='__NET__'/>
      <model type='virtio'/>
    </interface>

    <!-- Serial console for `virsh console` debugging. -->
    <serial type='pty'><target type='isa-serial' port='0'/></serial>
    <console type='pty'><target type='serial' port='0'/></console>

    <!-- Guest agent: lets libvirt read the leased IP from inside the guest. -->
    <channel type='unix'>
      <target type='virtio' name='org.qemu.guest_agent.0'/>
    </channel>

    <memballoon model='virtio'/>
    <rng model='virtio'><backend model='random'>/dev/urandom</backend></rng>
  </devices>
</domain>
