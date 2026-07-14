# 5 · GPU jobs & Apptainer

## The host as a GPU node

We have no GPU *cluster*, but the control-plane host can have one real GPU.
So the host **joins the pool as a single GPU node** and GPU jobs are
scheduled onto it — good for exercising the mechanics on one real GPU, not a
performance rig.

Registered by `cluster seed-nodes` from `config.env`:

```
CLUSTER_GPU_HOST=1
CLUSTER_HOST_NODE_NAME="cluster-host"
CLUSTER_HOST_RUNTIME="apptainer"
```

→ a `NodeRecord{name: cluster-host, index: 0, ip: 127.0.0.1, gpu: true, local: true}`.

Properties that fall out of the design:
- **`gpu: true`** → only `gpu:true` jobs land here; CPU jobs go to VMs.
- **`local: true`** → the `LocalHostAdapter` drives it: provision/deprovision are
  no-ops (we never destroy the host), and the container runs in-process.
- **exclusive lock** → only one GPU job at a time (the host is one node).
- **`index: 0`** → `cluster-down`/`inject-failure` (VM indices `1..N`) never touch it.

## Running containers with Apptainer

Apptainer (not Docker) is the primary runtime because it's **rootless**,
**HPC-native**, and — critically for the host GPU — binds the host's NVIDIA
driver/libraries into the container automatically with **`--nv`** (no
nvidia-container-toolkit daemon dance).

A GPU job's run phase becomes:

```
apptainer exec --nv --bind <host_workdir>:/out [--env K=V ...] <image> <command...>
```

built by `container_argv()` in `adapter.py`. `<image>` may be a `.sif` path or a
`docker://` reference — apptainer runs both. Output written to `/out` inside the
container lands in `<host_workdir>` and is then copied to the drop-zone.

Apptainer is pinned in the nix dev shell (`shell.nix`) for the host, and
installed into VMs by the `apptainer.yml` bootstrap task (a pinned, checksummed
`.deb` from the apptainer GitHub releases — see
[02-infrastructure.md](02-infrastructure.md)).

**`CLUSTER_GPU_BINDS`, not `CLUSTER_GPU_LD_LIBRARY_PATH`.** `shell.nix` declaratively
exports `CLUSTER_GPU_BINDS="/nix/store,/run/opengl-driver"` (NixOS keeps the
driver's userspace libs outside the standard `/usr/lib` path `--nv` looks in,
so these extra binds let it find them) — `adapter.py`'s `gpu_launch_extras()`
passes that straight through as extra `--bind` paths. It deliberately does
**not** also force a `LD_LIBRARY_PATH` override into the container, even
though an earlier version of this code did: live-tested, that override shadows
a full-OS container image's own glibc ahead of its normal `ld.so` search path
— a real `nvidia/cuda:*-ubuntu*` image segfaulted / hit "shared library
missing" running `nvidia-smi` with it set, and ran clean without it. `--nv`
already wires the driver's own userspace libraries correctly for a normal OCI
image; only a from-scratch/minimal container with no libc of its own would
need the forced override, and that's not a case this cluster runs.

## Submitting a GPU job

```yaml
# gpu-job.yaml
name: gpu-smoke
gpu: true
node_count: 1                 # only one GPU node exists
runtime: apptainer
image: "/path/to/your-image.sif"    # or docker://…
command: ["python", "run.py", "--out", "/out"]
env: { CUDA_VISIBLE_DEVICES: "0" }
output_dir: "/out"
```

```bash
jid=$(cluster submit --spec gpu-job.yaml)
cluster reconcile --execute            # routes to cluster-host, runs apptainer --nv
cluster result "$jid"                  # -> drop-zone path with the artifacts
```

## Caveats (real, but fine for what this is)

- **Contention:** the host also runs libvirtd, the VMs, and the reconciler —
  GPU timings are noisy. Fine for exercising the mechanics; don't treat
  numbers off a shared consumer GPU as citable results.
- **Isolation:** a container on the control-plane host is less isolated than in
  a VM. Apptainer contains it reasonably; it's still your machine.
- **Single GPU:** multi-GPU jobs can't be scheduled here — they fail cleanly.
