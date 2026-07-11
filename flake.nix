{
  description = "Outpost — local distributed compute cluster (libvirt/KVM)";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

  outputs = { self, nixpkgs }:
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
    in {
      # `nix develop` -> the same environment as `nix-shell` (shell.nix).
      devShells.${system}.default = import ./shell.nix { inherit pkgs; };

      # Convenience: `nix flake check` sanity, kept intentionally trivial.
      formatter.${system} = pkgs.nixpkgs-fmt;
    };
}
