# 9 · Multi-machine clusters — one control plane, many fabrics

> **Status: runbook.** The code that makes this possible (per-node ssh identity,
> per-node GPU binds, per-node kube-context) is implemented and unit/live-tested;
> the end-to-end bring-up across two physical machines is the part you execute on
> your hardware. This doc is that bring-up.

## The model (read this first)

Outpost is **one control plane** — one reconciler + one store — that owns a
**pool of nodes**. It is *not* a federation of control planes. "Different
clusters" here means **different fabrics/backends attached to the one control
plane**, not nested Outposts. A job picks a fabric declaratively:

| Fabric | JobSpec | Runs where |
|--------|---------|------------|
| VM CPU | (default) `gpu: false` | a libvirt VM on the control-plane host |
| host GPU (bare) | `gpu: true` | the control-plane host's own GPU, `apptainer --nv` |
| **remote GPU (bare)** | `gpu: true` | a **static-ssh** GPU worker, `apptainer --nv` over ssh |
| **KAI + HAMi** | `backend: k8s` | a gang of N pods on a k8s cluster's GPU (VRAM-sliced) |

The last two are the ones this doc wires up across machines.

## The two-machine layout used here

- **VM** (server: 32c / 128 GB / L20 48 GB, native Ubuntu) → **control plane +
  drop-zone**, its own CPU VMs, and a **k8s+HAMi cluster on the L20**.
- **PC** (NixOS: 16c / 64 GB / 5060 Ti 16 GB) → attached **two ways**: a
  **static-ssh GPU worker** (bare `apptainer --nv`) *and* its **own k8s+HAMi
  cluster** on the 5060 Ti. "PC is one cluster, VM is another" — both under the
  one control plane on the VM.

Everything below generalizes; swap addresses/names for your boxes.

## Prerequisites

1. **Network.** The control plane must reach every worker over ssh, and (for the
   k8s fabric) `kubectl` must reach each cluster's API server. A flat LAN works;
   otherwise a Tailscale/WireGuard overlay is simplest (use the overlay IPs
   everywhere below). AWS/remote boxes are the same story — this is the EC2
   `static-ssh` path pointed at your own hosts.
2. **ssh.** From the VM: `ssh <user>@<pc>` must work non-interactively with the
   key you'll give the node manifest. Authorize the control plane's public key on
   each worker.
3. **apptainer on the worker's non-login ssh PATH** (bare-GPU fabric only).
   Bootstrap runs `ssh <pc> command -v apptainer`. On NixOS the dev-shell
   apptainer is *not* on that PATH — add it to `environment.systemPackages` (or a
   user profile) so `ssh <pc> apptainer --version` works. On Ubuntu, install the
   pinned `.deb` (see [07-ubuntu-setup.md](07-ubuntu-setup.md)).

## Step 1 — control plane on the VM

Native Ubuntu (`source env.sh` + the apt list in
[07-ubuntu-setup.md](07-ubuntu-setup.md)); Nix works too but isn't required.
Optionally bring up local CPU VMs (`make net-up && make cluster-up && make
bootstrap`) — or run the VM purely as a control plane + scheduler and let the
GPUs live on the k8s clusters. Seeding is configured in Step 4.

## Step 2 — join the PC as a bare-GPU static-ssh worker

Edit [`node.pc-gpu.example.yaml`](../node.pc-gpu.example.yaml) (ip, ssh_user,
ssh_key; keep `gpu_binds: "/nix/store,/run/opengl-driver"` for a NixOS PC, empty
for Ubuntu), then:

```bash
cluster add-node --spec node.pc-gpu.example.yaml
cluster nodes            # pc-gpu shows gpu · static-ssh · binds=/nix/store,...
```

A `gpu: true, launcher: single` job can now land on the PC and run
`apptainer --nv <image>` over ssh, with the PC's own driver binds. (Note: a
plain `gpu: true` job will claim *whichever* GPU node is free — the VM host GPU
or the PC — so if you want to force the PC, keep it the only bare-GPU node, or
add a dedicated selector later.)

## Step 3 — the two k8s+HAMi clusters

- **VM L20** on native Ubuntu: follow
  [`infra/k8s/ubuntu-l20-setup.md`](../infra/k8s/ubuntu-l20-setup.md) (simpler
  than the NixOS recipe — stock `nvidia-container-toolkit`, no `/nix/store` CDI
  work). Gives a kube-context, e.g. `kind-vm`.
- **PC 5060 Ti** on NixOS: already stood up via
  [`infra/k8s/setup.sh`](../infra/k8s/setup.sh) → context `kind-outpost`.

**Making both reachable from the VM's `kubectl`** is the one real gotcha. `kind`
binds its API server to `127.0.0.1` by default, so the VM can't reach the PC's
cluster as-is. Two options:

1. **Expose the API server** — recreate the PC's kind cluster with
   `networking.apiServerAddress: <pc-ip>` + a fixed `apiServerPort`, then copy
   that kubeconfig entry onto the VM and rename its context (e.g. `kind-pc`).
2. **ssh tunnel** — `ssh -L 6444:127.0.0.1:6443 <pc>` and point a `kind-pc`
   context at `https://127.0.0.1:6444` (simplest for a demo; the tunnel must stay
   up while jobs run).

Verify from the VM: `kubectl --context kind-vm get nodes` and
`kubectl --context kind-pc get nodes` both succeed.

## Step 4 — seed both clusters as slot pools, then run

Register both k8s clusters via `CLUSTER_K8S_CLUSTERS` (config.env or inline), so
each becomes a pool of slots the reconciler can target:

```bash
CLUSTER_K8S_BACKEND=1 \
CLUSTER_K8S_CLUSTERS="vm:kind-vm:8,pc:kind-pc:4" \
  cluster seed-nodes
cluster nodes
#   k8s-slot-vm-0..7   gpu · k8s · ctx=kind-vm
#   k8s-slot-pc-0..3   gpu · k8s · ctx=kind-pc
#   pc-gpu             gpu · static-ssh · binds=/nix/store,...
#   + any VM CPU nodes
```

Target a specific cluster per job with `params.k8s_context` (else it lands on any
free slot):

```yaml
# a gang on the big L20 (48 GB -> big slices / many ranks)
name: sim-l20
backend: k8s
image: "nvcr.io/nvidia/pytorch:24.10-py3"
command: ["bash","-lc","python -c 'import torch;print(torch.cuda.get_device_name())'"]
node_count: 4
gpu: true
params: { gpu_memory_mb: 8192, k8s_context: kind-vm }
```

```bash
cluster submit --spec that.yaml
cluster reconcile --once --execute      # loop until terminal
make dashboard                          # gangs on both clusters, live
```

## What each fabric buys — and its limit

- **Bare GPU (static-ssh / host):** the *whole* GPU to one container, over ssh.
  Real throughput, no slicing. One job per GPU at a time.
- **KAI + HAMi:** N ranks sharing one GPU with hard per-rank VRAM caps, gang-
  scheduled, with pod-DNS rendezvous (`MASTER_ADDR`/`RANK`/`WORLD_SIZE`) so real
  torchrun/NCCL/MPI jobs form a communicator. Faithful for
  topology/scheduling/init; **not** for throughput (ranks time-slice one die).
  See [08-kubernetes-backend.md](08-kubernetes-backend.md).

## Not supported (be clear-eyed)

- **Federation / cluster-of-clusters.** One reconciler, one pool. The k8s backend
  is the only "sub-scheduler under Outpost" that exists.
- **A single `mpirun` spanning two machines.** The MPI fabric pinning is specific
  to the single-host libvirt `/24` bridge (see `adapter.fabric_if_include`);
  across a LAN/overlay it's latency-bound and mis-matches. Keep cross-machine
  jobs `launcher: single`, or use a k8s gang (same GPU box) for multi-rank work.
