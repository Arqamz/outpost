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

→ a `NodeRecord{name: cluster-host, index: 0, ip: <CLUSTER_HOST_IP>, gpu: true, local: true}`
(the ip is the host's cluster0 bridge address, `192.168.71.1` by default — VM
ranks in a hybrid job must be able to reach the host on it; nothing dials it
for host-only jobs).

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

**Building def files rootless.** `make mpi-sif` / `make cuda-sif` go through
`demo/build-sif.sh`, which picks the build mode per host. The nix apptainer
ships a non-setuid `starter` and this host has no setuid `newuidmap`/`newgidmap`,
so apptainer's normal subuid fakeroot can't set up its id map (`FATAL: /etc/subuid
mapping found but no user namespace available for fakeroot`). The script falls
back to a **proot**-emulated root build (`--ignore-subuid --ignore-fakeroot-command`;
`proot` is pinned in `shell.nix`). Because proot's single-uid mapping can't
emulate apt dropping to the `_apt` uid, each def's `%post` disables apt's
sandbox (`APT::Sandbox::User "root"`). A host with a setuid `newuidmap` (Ubuntu's
default) skips all of that and does a plain native `apptainer build`.

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

## Hybrid jobs: the GPU host + VMs in ONE mpirun

`hybrid: true` (with `launcher: mpi`, `node_count >= 2`) is the one job shape
that mixes the pools: it claims **1 GPU node (the host) + `node_count-1` CPU
VMs** and launches a single `mpirun` **from the host** spanning all of them —
rank 0 forked locally (the host's bridge address is a local interface, so no
sshd is needed on the host), VM ranks launched over ssh with the cluster key
(`--mca plm_rsh_agent`, tree spawn disabled), all ranks talking TCP on the
cluster0 fabric. See `job.hybrid.example.yaml` / `make submit-hybrid`, and the
runbook in [07-ubuntu-setup.md](07-ubuntu-setup.md).

**Per-rank appfile — the ranks differ.** A hybrid job can't use one shared
per-rank command: rank 0 needs `--nv` + the `CLUSTER_GPU_BINDS` driver binds,
but those bind sources (e.g. NixOS's `/nix/store`, `/run/opengl-driver`) don't
exist inside a VM, and a missing bind source is a hard apptainer error. So
`LocalHostAdapter._run_mpi` writes an **OpenMPI appfile** (`write_appfile` /
`mpirun --app`) with a distinct line per rank: rank 0 (the GPU host) gets
`apptainer exec --nv` + the binds; the VM ranks get a plain launch (no `--nv`
warning, no host-only binds). The VM-headed pure-CPU MPI path
(`LibvirtAdapter._run_mpi`) still uses one shared command — every rank there is
an identical VM.

Prerequisite beyond a normal GPU job: `mpirun` on the host **matching the
guests' OpenMPI 4.1.6**. The nix dev shell now pins this — `flake.nix` pulls
`openmpi` from a second input (`nixpkgs-mpi` = `nixos-24.05`, which ships 4.1.6)
because `nixos-unstable` is on 5.x and mpirun/orted must match the guests'
release series. On native Ubuntu it's `apt install openmpi-bin` (also 4.1.6).
The demo containers below apt-install that same 4.1.6 so the in-container
libmpi matches the host `orted`.

**Fabric pinning — OOB and BTL need DIFFERENT `if_include` values.** Running
mpirun on the host (whose cluster0 address is the `virbr-cluster` bridge, on a
box full of `docker0`/`veth*`/`tailscale0`/`virbr0` and the real wifi/LAN nic)
walks straight into two opposite OpenMPI 4.1.x interface-matching bugs, so
`fabric_if_include` returns two values:
- **OOB** (orted control channel) → a list of each node's EXACT `/32`
  (`192.168.71.1/32,192.168.71.11/32`). The `/24` subnet makes OpenMPI's OOB
  matcher mis-resolve among the interface zoo and reject the bridge outright
  ("None of the TCP networks ... could be found").
- **BTL** (rank↔rank data) → the `/24` subnet. The inverse bug: a `/32` in the
  BTL matcher "matches" *every* interface and advertises the host's wifi/docker
  addresses (which a VM can't reach), so the first MPI collective hangs forever.

Both are global to the mpirun and forwarded to every orted, so both forms are
built to be valid on every node. This combination is **live-verified**: the
CUDA demo below ran host-GPU + VM-CPU to `promoted`, `max |error| 0`.

**On NixOS, trust the cluster bridge in the firewall.** The firewall is on by
default and will drop the VM ranks' TCP connections back to the host's `mpirun`
(the job hangs at the mpirun step). Add the cluster0 bridge to
`trustedInterfaces` and rebuild:

```nix
networking.firewall.trustedInterfaces = [ "virbr-cluster" ];   # CLUSTER_NET_BRIDGE
```

(On native Ubuntu the equivalent is a ufw rule — see
[07-ubuntu-setup.md](07-ubuntu-setup.md). Also make sure no other libvirt
network already owns the `192.168.71.0/24` subnet before `make net-up`;
`virsh net-list --all` then `net-destroy`/`net-undefine` the squatter.)

### The CUDA demo: VM CPU rank → host GPU rank

`demo/cuda_axpb.cu` + `demo/cuda_axpb.def` (`make cuda-sif`,
`job.hybrid.cuda.example.yaml`) exercise the full hybrid data path: the VM rank
generates a vector on the CPU and `MPI_Send`s it; rank 0 (the GPU host) copies
it to the real GPU, runs `y = a*x + b` in a CUDA kernel, copies it back, and
validates. The GPU code path is rank-gated, so the VM rank never touches a CUDA
API — it has no driver and runs without `--nv`. The container is built `FROM`
an NVIDIA CUDA devel image and compiles the kernel to `compute_70` PTX so the
driver JITs it forward onto whatever GPU is present (this host's Blackwell
included), without pinning an arch at build time.

## AMD Instinct (ROCm) GPU nodes

A GPU node has a **vendor**: `NodeRecord.gpu_vendor`, `nvidia` (the default —
every node registered before the field existed, and every manifest that does not
mention it) or `amd`. Only the **static-ssh** path runs AMD nodes; the
local-host, libvirt, k8s and slinky backends implement the NVIDIA path only and
**refuse** an AMD GPU node with an explicit error (`require_nvidia` in
`adapter.py`) rather than starting its container with no GPU wired in.

What changes on an AMD node — and nothing changes for an NVIDIA one:

| | NVIDIA (unchanged) | AMD |
|---|---|---|
| apptainer GPU flag | `--nv` | `--rocm` (binds `/dev/kfd`, `/dev/dri`, a few host ROCm libs) |
| per-rank GPU pinning (planned launches) | `CUDA_VISIBLE_DEVICES=<GPU-uuid>` | `ROCR_VISIBLE_DEVICES=<GPU-hex>` (ROCr UUID) |
| GPU inventory / topology | `nvidia-smi --query-gpu`, `nvidia-smi topo -m` | `amd-smi list/static/topology --json` + sysfs (`numa_node`, `local_cpulist`) |
| receipt checks | `CUDA_VISIBLE_DEVICES` | `ROCR_VISIBLE_DEVICES` |
| `runtime: docker` | `--gpus all` | refused (use apptainer) |

**Why `ROCR_VISIBLE_DEVICES`, not `HIP_VISIBLE_DEVICES`.** ROCr filters devices
underneath HIP, OpenCL and OpenMP offload alike, so every ROCm consumer in the
container (rocminfo, PyTorch, rccl, plain HSA code) sees the same restricted set;
`HIP_VISIBLE_DEVICES` narrows only HIP. ROCr also accepts the stable device UUID
(`GPU-` + the ASIC serial in lower-case hex — what `rocminfo` prints as `Uuid:`),
so a plan pins a physical device by identity, exactly as the NVIDIA path pins by
GPU UUID. Outpost sets only this one variable (setting both compounds: HIP
indexes into what ROCr already filtered). A workload that wants to know how many
GPUs it was given should count the comma-separated entries of
`ROCR_VISIBLE_DEVICES` (unset = every GPU on the node). The UUID comes from
`amd-smi static`'s `asic_serial`, **not** amd-smi's own `uuid` field, whose
format differs between amd-smi versions (the host and an in-container amd-smi
on the same MI300X VF print different values). Verified on an MI300X VF:
`ROCR_VISIBLE_DEVICES=GPU-8ac1449593d1d319` → HIP sees 1 device, a bogus UUID →
0.

As on NVIDIA, the **unplanned** single-node path (no `launch` block) sets no
visible-devices variable: the container sees every GPU on the node.

### Preparing an AMD box

The node needs the `amdgpu` kernel driver (so `/dev/kfd` + `/dev/dri` exist),
`amd-smi` on the host (read by the topology probe and by `add-node`'s vendor
detection), and apptainer. On Ubuntu 24.04, apptainer from the official PPA
(this is what the MI300X droplet runs, apptainer 1.5.5-1~noble):

```bash
add-apt-repository -y ppa:apptainer/ppa
apt-get update && apt-get install -y apptainer
apptainer exec --rocm docker://rocm/dev-ubuntu-24.04:10.2.0-full rocminfo | grep -E 'gfx|Uuid: +GPU'
#   Name:  gfx942
#   Uuid:  GPU-8ac1449593d1d319
```

The image brings its own ROCm userspace; pick one recent enough for the host's
driver/firmware. On the DigitalOcean MI300X VF (amdgpu 7.1.3, host ROCm 10.0)
`rocm/dev-ubuntu-24.04:10.2.0-full` works end to end, while the 7.2.4 image's
`rocminfo` fails with `HSA_STATUS_ERROR_INVALID_ARGUMENT` (HIP kernels in that
image still run).

### Adding an AMD static-ssh node

```bash
cp node.amd-mi300x.example.yaml node.fp-mi300x-1.yaml   # node.*.yaml is gitignored
$EDITOR node.fp-mi300x-1.yaml                            # ip, ssh_user, ssh_key
cluster add-node --spec node.fp-mi300x-1.yaml
#   added static-ssh node fp-mi300x-1 at … (gpu, …, gpu_vendor=amd, host_memory=235.9GiB)
cluster nodes
#   fp-mi300x-1  idx=-1 …  available  gpu  static-ssh  apptainer  owner=- vendor=amd
```

`gpu_vendor: amd` in the manifest is authoritative. Leave it out and `add-node`
asks the box (`nvidia-smi -L` first, then `amd-smi list`), falling back to
`nvidia` when neither answers. Bootstrap then checks `apptainer` and, for an AMD
node, `/dev/kfd` + `/dev/dri`; and if a launch-planned job's topology probe finds
a different vendor than the node was registered with, the job fails naming the
fix (re-register it) instead of pinning the wrong variable.

### Submitting to it

```bash
jid=$(cluster submit --spec job.gpu.amd.example.yaml)   # gpu: true, params.node_name: fp-mi300x-1
cluster reconcile --execute --interval 5                 # or run it as the daemon
cluster status "$jid"                                    # -> promoted
ls "$(cluster result "$jid")"                            # stdout.log, hip-axpb.txt, launch-receipt.yaml, …
```

**Live-verified** (2026-11-19, DigitalOcean MI300X VF droplet, local control
plane, static-ssh): `job.gpu.amd.example.yaml` reached `promoted` with exit 0 —
`ROCR_VISIBLE_DEVICES=GPU-8ac1449593d1d319` delivered, rocminfo saw the gfx942
agent, amd-smi ran in the container, the HIP kernel validated
(`max_abs_err=0`), and `launch-plan.yaml`/`launch-receipt.yaml` recorded the GPU
by ROCr UUID with the receipt's GPU check passing against
`ROCR_VISIBLE_DEVICES`. (That receipt read `mismatched` on CPUs only: under
`launcher: single` the preflight probe runs before `exec taskset`, so it sees
the unbound shell — a launcher-path issue independent of the GPU vendor.)

Pin AMD nodes with `params.node_name` (what GTL does with `--target`): an
unpinned `gpu: true` job takes any free GPU node, and there is no vendor
selector in the claim yet, so on a mixed pool a CUDA image could land on an AMD
box (it then fails at run time, not silently).

## Caveats (real, but fine for what this is)

- **Contention:** the host also runs libvirtd, the VMs, and the reconciler —
  GPU timings are noisy. Fine for exercising the mechanics; don't treat
  numbers off a shared consumer GPU as citable results.
- **Isolation:** a container on the control-plane host is less isolated than in
  a VM. Apptainer contains it reasonably; it's still your machine.
- **Single GPU:** multi-GPU jobs can't be scheduled here — they fail cleanly.
