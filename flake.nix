{
  description = "Outpost — local distributed compute cluster (libvirt/KVM)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  # A SECOND pin, only for OpenMPI. nixos-unstable ships OpenMPI 5.x, but the
  # hybrid launch runs the host's mpirun against in-container ranks built on
  # Ubuntu 24.04 (OpenMPI 4.1.6) — mpirun/orted must match the guests' release
  # series or the PMIx/wire handshake fails. nixos-24.05 pins OpenMPI 4.1.6,
  # the exact version Ubuntu 24.04's openmpi-bin (and the demo containers)
  # provide. Only the `openmpi` package is taken from here.
  inputs.nixpkgs-mpi.url = "github:NixOS/nixpkgs/nixos-24.05";

  outputs = { self, nixpkgs, nixpkgs-mpi }:
    let
      system = "x86_64-linux";
      # terraform + packer moved to HashiCorp's BSL license (unfree to Nix).
      # They're only used from T0 onward; allow-list exactly those two rather
      # than a blanket allowUnfree.
      pkgs = import nixpkgs {
        inherit system;
        config.allowUnfreePredicate = pkg:
          builtins.elem (nixpkgs.lib.getName pkg) [ "terraform" "packer" ];
      };
      pkgsMpi = import nixpkgs-mpi { inherit system; };
    in {
      # `nix develop` -> the same environment as `nix-shell` (shell.nix).
      # openmpi is threaded in from the pinned nixos-24.05 (4.1.6), NOT from
      # `pkgs` (which is 5.x on unstable).
      devShells.${system}.default =
        import ./shell.nix { inherit pkgs; openmpi = pkgsMpi.openmpi; };

      # Convenience: `nix flake check` sanity, kept intentionally trivial.
      formatter.${system} = pkgs.nixpkgs-fmt;
    };
}
