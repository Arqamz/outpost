# Getting the GPU into a KinD node on NixOS — the one system change

This is the **only** step of the spike that needs a `nixos-rebuild` (root). It's
yours to apply — I can't rebuild your system config.

## What's already proven (no change needed)

The toolkit is `enable`d in **CDI mode**, and a plain container sees the card
**when GPUs are requested explicitly**:

```
$ docker run --rm --device nvidia.com/gpu=all ubuntu:24.04 nvidia-smi -L
GPU 0: NVIDIA GeForce RTX 5060 Ti     # ✅ exit 0
```

So the driver + toolkit + docker-CDI path all work.

## Why `hardware.nvidia-container-toolkit.enable = true` alone isn't enough

That option (the modern CDI path) only makes `--device nvidia.com/gpu=all` work
— it does **not** change docker's default runtime:

```
$ docker info --format '{{.DefaultRuntime}}'   →  runc      # not nvidia
$ docker run --rm ubuntu:24.04 nvidia-smi -L    →  "nvidia-smi": not found
```

A **KinD node is a docker container** created with no `--device` flag, so it
gets plain `runc` and no GPU — hence the error you saw. KinD has no per-node
device flag, so the fix (what NVIDIA's `nvkind`/klueska recipe does) is two
system-level settings:

1. **nvidia = docker's default runtime** — so every container, kind nodes
   included, goes through the GPU-capable runtime.
2. **`accept-nvidia-visible-devices-as-volume-mounts = true`** — so a container
   can request GPUs via a *volume mount* (`/var/run/nvidia-container-devices/all`),
   which is how `kind-cluster.yaml`'s `extraMount` injects the card into the node
   (pods can't set `NVIDIA_VISIBLE_DEVICES`, so this is the standard workaround).

## The change — drop into your existing docker feature module

Add these to the `config = lib.mkIf cfg.enable { … }` block of your
`my.features.virtualisation.docker` module (guarded by `cfg.nvidia`, next to the
`hardware.nvidia-container-toolkit.enable` line you already have). The paths are
copied from what NixOS itself generates for docker+nvidia, so they resolve to the
right `/nix/store` binaries:

```nix
# nvidia as docker's DEFAULT runtime (needed for KinD GPU nodes).
virtualisation.docker.daemon.settings = lib.mkIf cfg.nvidia {
  default-runtime = "nvidia";
  runtimes.nvidia.path = lib.getExe'
    (lib.getOutput "tools" config.hardware.nvidia-container-toolkit.package)
    "nvidia-container-runtime";
};

# Runtime config with the volume-mounts strategy the kind extraMount relies on.
# (CDI mode is kept, so `--device nvidia.com/gpu=all` keeps working too; the
#  /var/run/nvidia-container-devices/all mount maps to the CDI `all` device.)
environment.etc."nvidia-container-runtime/config.toml" = lib.mkIf cfg.nvidia {
  text = ''
    accept-nvidia-visible-devices-as-volume-mounts = true
    disable-require = true
    [nvidia-container-cli]
    environment = []
    ldconfig = "@${lib.getExe' pkgs.glibc "ldconfig"}"
    load-kmods = true
    no-cgroups = false
    path = "${lib.getExe' pkgs.libnvidia-container "nvidia-container-cli"}"
    [nvidia-container-runtime]
    mode = "cdi"
    runtimes = ["docker-runc", "runc", "crun"]
    [nvidia-container-runtime-hook]
    path = "${lib.getOutput "tools" config.hardware.nvidia-container-toolkit.package}/bin/nvidia-container-runtime-hook"
    skip-mode-detection = false
    [nvidia-ctk]
    path = "${lib.getExe' config.hardware.nvidia-container-toolkit.package "nvidia-ctk"}"
  '';
};
```

`sudo nixos-rebuild switch`, then verify in **two** steps:

```bash
# 1. nvidia is now the default runtime:
docker info --format '{{.DefaultRuntime}}'          # want: nvidia

# 2. THE key test — the exact mechanism kind uses, isolated (no kind needed):
docker run --rm -v /dev/null:/var/run/nvidia-container-devices/all \
  ubuntu:24.04 nvidia-smi -L                          # want: GPU 0: ... RTX 5060 Ti
```

If test 2 lists the GPU, KinD nodes will get it too (same mechanism) — go to
`README.md` step 2. If it errors, paste the output; that pins which knob is off
before you spend a `kind create` cycle on it.

> If Nix complains that `environment.etc."nvidia-container-runtime/config.toml"`
> is defined twice, the toolkit module is also writing it — add `lib.mkForce`
> to the `text` value. (On this box no such file exists today, so it shouldn't.)

## Reverting

Delete the two blocks and `nixos-rebuild switch` — docker returns to
`default-runtime = runc` + CDI-only, exactly as now. Fully reversible.
