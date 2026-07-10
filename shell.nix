# pkgs is normally supplied by flake.nix (which allow-lists terraform/packer).
# The default here is only for a bare `nix-shell` invocation.
{ pkgs ? import <nixpkgs> {
    config.allowUnfreePredicate = pkg:
      builtins.elem (pkg.pname or "") [ "terraform" "packer" ];
  } }:

# Dev shell for local provisioning.
# Everything the libvirt adapter + ansible plays need, pinned via the flake.
# NOTE: libvirtd itself is a *system* service (already enabled on this NixOS
# host via `virtualisation.libvirtd.enable`). This shell only provides the
# client-side tooling; it does not — and cannot — start the daemon.

pkgs.mkShell {
  name = "outpost";

  packages = with pkgs; [
    # --- virtualization / provisioning (libvirt adapter) ---
    qemu # qemu-img, qemu-system-x86_64
    libvirt # virsh, virt-xml-validate
    cloud-utils # cloud-localds -> NoCloud seed ISO
    cdrkit # genisoimage backing cloud-localds
    mkpasswd # password hashes for cloud-init (kept, though we ship keys-only)
    swtpm # in case a guest ever needs an emulated TPM

    # --- container runtime (host GPU node runs jobs via apptainer --nv) ---
    apptainer

    # --- config management (the one bootstrap role) ---
    ansible # ansible-playbook + ansible-galaxy
    openssh
    sshpass # only for first-boot fallback; key auth is the default

    # --- pinned for a future non-local backend, if one gets added ---
    terraform
    packer

    # --- glue / parsing / reconciler ---
    (python3.withPackages (ps: with ps; [ pyyaml pymongo ]))
    jq
    yq-go
    gettext # envsubst for template rendering
    wget
    curl
    gnumake
    coreutils
    util-linux # uuidgen
  ];

  shellHook = ''
    export CLUSTER_ROOT="$(pwd)"
    # libvirt system connection is the default target for every wrapper.
    export LIBVIRT_DEFAULT_URI="qemu:///system"
    # cluster on PATH; reconciler importable.
    export PATH="$CLUSTER_ROOT/bin:$PATH"
    export PYTHONPATH="$CLUSTER_ROOT''${PYTHONPATH:+:$PYTHONPATH}"

    # ── GPU via apptainer --nv (declarative; works on drop-in) ──────────────
    # On NixOS the NVIDIA driver libs live in /run/opengl-driver/lib (populated
    # by hardware.graphics) and need a matching glibc. We pin that glibc here so
    # containers launched with --nv resolve the host driver's userspace. Binding
    # /nix/store lets the driver's own nix-built binaries find their interpreter.
    # Override CLUSTER_GPU_BINDS="" + CLUSTER_GPU_LD_LIBRARY_PATH on a normal distro host.
    export CLUSTER_GPU_BINDS="/nix/store,/run/opengl-driver"
    export CLUSTER_GPU_LD_LIBRARY_PATH="${pkgs.glibc}/lib:/run/opengl-driver/lib"
    # apptainer image cache stays inside the repo (.var is gitignored)
    export APPTAINER_CACHEDIR="$CLUSTER_ROOT/.var/apptainer-cache"

    echo "── outpost dev shell ─────────────────────────────────────────────"
    echo "  libvirt URI : $LIBVIRT_DEFAULT_URI"
    echo "  qemu        : $(qemu-img --version | head -1)"
    echo "  virsh       : $(virsh --version 2>/dev/null)"
    echo "  ansible     : $(ansible --version 2>/dev/null | head -1)"
    echo ""
    echo "  Edit infra/libvirt/config.env, then:"
    echo "    make net-up        # define the cluster0 libvirt network"
    echo "    make template      # build the base qcow2 (downloads Ubuntu img)"
    echo "    make cluster-up    # define + start all VMs"
    echo "    make bootstrap     # run the bootstrap ansible role"
    echo "    make status | make cluster-down | make net-down"
    echo "──────────────────────────────────────────────────────────────────"
  '';
}
