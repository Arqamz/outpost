# 11 · Slinky backend: Slurm on Kubernetes

> **Status: implemented and live-verified** on tashkil AWS child clusters
> (2026-10-14): kubeadm 1.35, Slinky 1.2.2, Slurm 26.05, pyxis/enroot,
> `/scratch` on FSx for Lustre, 2 × T4 and up to 2 × A10G. GTL's STREAM,
> HPL-MxP and 2-node nccl-tests ran from `gtl submit` through the reconciler
> to fingerprints. See the end of this page for exactly what was run.

`SlinkyAdapter` runs a JobSpec as a Slurm batch job on a
[Slinky](https://github.com/SlinkyProject/slurm-operator) cluster. The JobSpec
and the drop-zone work the same as on every other backend.

## The one assumption: a login pod with a shared filesystem

Outpost never talks to slurmctld or slurmrestd directly. Everything goes through
`kubectl exec` into a running Slinky **login pod**:

| Phase | What it does in the login pod |
|---|---|
| `provision` | `scontrol ping && sinfo`: slurmctld answers |
| `bootstrap` | the workdir root is writable; `srun` has `--container-image` (pyxis loaded); the partition exists, if one is set |
| `run` | imports the image once (below), writes `launcher-slinky.sbatch` into the job's workdir, `sbatch --parsable`, polls `scontrol show job -o` until a terminal state |
| `collect` | `tar` of the job's workdir streamed back into the drop-zone (staging artifacts dropped) |
| `deprovision` | `scancel --name=outpost-<job_id>` (idempotent), then removes the workdir |

The job's workdir, `$CLUSTER_SLINKY_WORKDIR_ROOT/<job_id>` (default
`/scratch/outpost/<job_id>`), must sit on a filesystem that the login pod and
every slurmd pod mount **at the same path**. The batch script binds it at the
job's `output_dir`, and Slurm writes `stdout.log` into it. The egress contract is
the same as every other adapter's: `stdout.log` plus whatever the job wrote to
`output_dir`.

Because of this, outpost needs nothing from the cluster beyond kubectl access:
no LoadBalancer, no slurmrestd token, and no Slurm users. Jobs run as root,
because `kubectl exec` lands as root in the login pod.

## Images: pyxis + enroot, imported once per digest

Pyxis runs the container (`srun --container-image=...`). Before submitting,
`run()` imports the image in the login pod with `enroot import` into
`$WORKDIR_ROOT/.images/<digest>.sqsh`, unless that file already exists. Every
later job on any node starts from that squashfs, and no slurmd pulls anything.
`docker://nvcr.io/x/y@sha256:…` is rewritten to enroot's
`docker://nvcr.io#x/y@sha256:…`. A JobSpec image that is already an absolute
path is used as a squashfs as-is.

Enroot needs a temp dir that both accepts overlay whiteouts and can back an
overlay mount. A login pod has no such dir by default. All three of these were
seen failing on FSx:

- **Lustre:** `failed to mount overlay: Invalid argument`
- **The container's overlay root:** `failed to create ovlfs whiteout: Operation not permitted` (any image with whiteouts, e.g. every NGC image)
- **`/tmp`:** a RAM tmpfs, too small for an NGC image (8.5 GB squashfs, about 11 GB unpacked)

So by default the temp dir (`CLUSTER_SLINKY_ENROOT_TEMP`,
`/var/tmp/outpost-enroot`) is a sparse ext4 image on the shared filesystem
(`CLUSTER_SLINKY_ENROOT_TEMP_IMAGE`, 64G), loop-mounted inside the privileged
login pod under a `flock`. Set `CLUSTER_SLINKY_ENROOT_TEMP_IMAGE=""` if the temp
dir is already a suitable local volume, such as an emptyDir on the node disk.

GPU jobs also export `NVIDIA_VISIBLE_DEVICES=all` and
`NVIDIA_DRIVER_CAPABILITIES=compute,utility`, because enroot's nvidia hook
injects the driver only when the container environment names GPUs. NGC images
set these themselves; plain images do not. Slurm's `--gres=gpu:N` still decides
which GPU the task gets. A job's own `env` overrides both.

## JobSpec mapping

| JobSpec | sbatch / srun |
|---|---|
| `node_count` | `--nodes=N`, one task per node; with per-node slots also `--nodelist`, `--exclusive` |
| `launcher: single` | `srun --ntasks=1` (first node) |
| `launcher: mpi` | `srun --ntasks=N --mpi=$CLUSTER_SLINKY_MPI` (default `pmix`) |
| `gpu: true` | per-node slots: `--gres=gpu:<the node's GPU count>`; cluster-wide: `--gres=gpu:<params.gpu_count, default 1>` |
| `env` | `export`ed in the batch script; `MASTER_ADDR` (first allocated host), `MASTER_PORT`, `WORLD_SIZE` are set first |
| `output_dir` | `--container-mounts=<workdir>:<output_dir>`, `--container-workdir` |
| `params.time_limit` | `--time` |
| `params.partition` | not used; `CLUSTER_SLINKY_PARTITION` (empty = the cluster's default) |
| `launch` (placement intent) | refused: Slurm binds the tasks, not outpost's resolver |

The adapter maps Slurm's terminal states (`COMPLETED`, `FAILED`, `TIMEOUT`,
`NODE_FAIL`, `OUT_OF_MEMORY`, …) to an exit code. Anything other than
`COMPLETED` fails the job. A job pending for a reason that never clears on its
own (`BadConstraints`, `PartitionConfig`, `JobHeldAdmin`, …) is cancelled and
failed straight away instead of waiting out `CLUSTER_SLINKY_RUN_TIMEOUT`
(default 3600s).

## Routing: slots

A Slinky cluster appears in the registry as **slots** (provider `slinky`). A
plain pool claim never takes a backend slot (`BACKEND_PROVIDERS` in
`models.py`). There are two kinds:

- **Per-node slots** (`add-slinky --per-node`): one slot per Slurm node, found
  with `sinfo -N` through the login pod and named `slinky-<name>-<i>` in sinfo
  order. Each carries its Slurm node name and GPU count. A job takes
  `node_count` of them and runs on exactly those nodes: `--nodelist`,
  `--exclusive`, `--gres=gpu:<all of the node's GPUs>`. Rank i runs on the i-th
  slot named (srun `--distribution=arbitrary` over a hostfile in pin order), and
  `MASTER_ADDR` is the first. This matches a bare-metal claim: whole nodes, in
  rank order. If the cluster's nodes are replaced, `remove-node` the slots and
  register again; bootstrap refuses a pinned node Slurm no longer has or
  reports down.
- **Cluster-wide slots** (`--slots N`, the default 1): a concurrency bound, not
  a machine. A job takes one, and Slurm picks `node_count` nodes and gives each
  task `params.gpu_count` GPUs (default 1).

A job reaches slots in one of two ways:

- `backend: slinky`, optionally with `params.slinky_context` (one cluster) or
  `params.node_name` (exact slots);
- **a `params.node_name` pin naming slinky slots, with no `backend` set.** This
  is how GTL gets there: a deployment target pins node names, so a target whose
  `node_names` is `[slinky-a10g-0, slinky-a10g-1]` is a 2-node job on those two
  Slurm nodes, and GTL never learns Slurm is involved. A pin of per-node slots
  must name exactly `node_count` of them, all on one cluster. A cluster-wide slot
  carries a whole job, so a pin names at most one.

```bash
cluster add-slinky --name a10g --context <kube-context> --per-node  # -> slinky-a10g-0, -1, ...
cluster add-slinky --name big --context big --slots 4                # -> slinky-big-0..3 (cluster-wide)
cluster submit --spec job.slinky.smoke.yaml
cluster reconcile --execute
```

The reconciler needs a kubeconfig that reaches the cluster (`KUBECONFIG` plus the
slot's kube-context).

## On a tashkil child cluster

A cluster created with `scheduler: slinky` gets everything this backend assumes:
namespace `slurm`, a pyxis LoginSet `main`, one slurmd per worker, and `/scratch`
(FSx) mounted in both.

**Register the slot only once the cluster is `Delivered`.** In the platform's
shared-GPU mode, slurmd uses the node's GPUs without claiming them from the
device plugin. Nothing stops a Slurm job and a Kubernetes GPU pod from landing
on the same card. One of those pods is the platform's own GTL acceptance run,
which executes between `Provisioned` and `Delivered`. Seen 2026-10-14 on an
A10G: an outpost STREAM job submitted as soon as Slurm answered overlapped
acceptance's HPL-MxP, acceptance failed on performance, and the platform
decommissioned the cluster as its policy says.

**A job can take Slurm's controller down with it.** slurmctld runs as a pod on
one of the GPU workers, and Slurm enforces no memory limit on jobs there. Seen
2026-10-14 on 2 × g5.xlarge (16 GiB): HPL-MxP at `n=45056` (≈17 GB host) drove
its worker `NotReady`, and slurmctld, on that same worker, went down until the
node recovered. The adapter now waits out such an outage
(`CLUSTER_SLINKY_CTLD_GRACE`) instead of failing every job. Size host-memory
hungry jobs for the node (GTL's HPL host-memory rule: `n=36864` fits a
g5.xlarge).

The reconciler reaches the cluster the way cluster-service does: through the
management cluster's chisel door, with a kubeconfig whose server is a local
port-forward.

## Settings

| Variable | Default | |
|---|---|---|
| `CLUSTER_SLINKY_NAMESPACE` | `slurm` | where the login pods live |
| `CLUSTER_SLINKY_LOGIN_SELECTOR` | `app.kubernetes.io/name=login` | label selector for login pods |
| `CLUSTER_SLINKY_LOGIN_CONTAINER` | `login` | |
| `CLUSTER_SLINKY_WORKDIR_ROOT` | `/scratch/outpost` | shared by login and slurmd pods |
| `CLUSTER_SLINKY_PARTITION` | `` | `` = the default partition |
| `CLUSTER_SLINKY_MPI` | `pmix` | `srun --mpi` for `launcher: mpi` |
| `CLUSTER_SLINKY_ENROOT_TEMP` | `/var/tmp/outpost-enroot` | enroot unpack dir |
| `CLUSTER_SLINKY_ENROOT_TEMP_IMAGE` | `<root>/.images/enroot-temp.ext4` | `` = use the temp dir as-is |
| `CLUSTER_SLINKY_ENROOT_TEMP_SIZE` | `64G` | sparse; sized for the largest unpacked image |
| `CLUSTER_SLINKY_RUN_TIMEOUT` | `3600` | seconds, submit to terminal |
| `CLUSTER_SLINKY_POLL_INTERVAL` | `10` | |
| `CLUSTER_SLINKY_KEEP_WORKDIR` | `0` | `1` keeps the workdir after teardown |
| `CLUSTER_SLINKY_CTLD_GRACE` | `600` | seconds slurmctld may be unreachable before poll, provision or teardown give up |
| `CLUSTER_SLINKY_TRANSPORT_RETRIES` | `6` | attempts for a call cut by a kubectl transport error (not a command failure) |
| `CLUSTER_SLINKY_TRANSPORT_BACKOFF` | `5` | seconds between those attempts |
| `CLUSTER_ECR_PROFILE` | `` | AWS profile outpost mints ECR pull tokens with (every backend). A private ECR image with this unset is refused |

## Performance: what running through Slinky costs and gains

Measured 2026-10-14 on 2 × g5.xlarge (A10G, 4 vCPU, 16 GiB, "up to 10 Gbps"),
against the promoted non-Slinky A10G records in GTL staging (same image digest,
same instance type, same parameters).

| | Slinky | baseline | |
|---|---|---|---|
| STREAM Triad | 523.5 GB/s (var 0.65%) | 523.4 GB/s | no loss: GPU-bound kernels are unaffected |
| HPL-MxP, `n=36864` | 13,406 GFLOPS (var 1.6%) | 14,620 GFLOPS | **−8.3%**, reproducible; cause not isolated (below) |
| nccl-tests all_reduce, 2 nodes | 0.80 GB/s avg busbw (run 1) | none on g5.xlarge | 1 GiB: 0.88 GB/s on burst, **0.31 GB/s (2.5 Gbit/s) once burst credits run out** |

Fixed overhead per job, from the reconciler's audit timestamps:

- provision + bootstrap: ~12 s (three or four `kubectl exec` round trips through the chisel door);
- submit + polling: up to `CLUSTER_SLINKY_POLL_INTERVAL` (10 s) after Slurm finishes;
- container start: pyxis extracts the image's squashfs on the node for every job
  (`unsquashfs` at about one core for 30–40 s for the 8.5 GB NGC image). It falls
  inside Slurm's elapsed time, but before the benchmark starts;
- collect ~3–4 s, teardown ~6 s;
- once per cluster: the `enroot import` of each image (minutes; parallel jobs
  wait on its lock).

Where Slinky is better than the bare-metal path:

- multi-node MPI needs nothing on the hosts: `srun --mpi=pmix` starts the
  ranks inside the NGC image. Bare metal needs a matching OpenMPI 4.1.x on every
  host and an ssh fabric key on the head;
- per-node slots run different single-node jobs on different nodes at once.

What the numbers do not yet separate:

- **HPL-MxP's −8%.** HPL-MxP runs at ~290% CPU on the 4-vCPU node outside the
  GPU solve. The worker also runs slurmctld or slurmdbd/mariadb, plus the
  platform's operators, and Slurm binds nothing (`TaskPlugin` is unset).
  Untested: whether binding or a quieter node closes the gap.
- **Pod network vs host network** for NCCL. Slurm runs slurmd in pods on the
  Cilium pod network. No same-node host-network NCCL baseline exists yet.
- **The g5.xlarge network is bursty, on every backend.** "Up to 10 Gbps" is
  a burst. Sustained traffic drains credits within one ~10-minute nccl run, and
  later runs measure the 2.5 Gbit/s baseline. Back-to-back nccl runs on this
  instance cannot be reproducible, and a 10 Gbit/s ceiling overstates what a
  sustained run can reach.

**Placement and affinity on a tashkil child cluster: nothing is enforced.**
The cluster runs Slurm with no `TaskPlugin` and a `cgroup.conf` that
constrains nothing. A probe job (2026-10-14) found that every `--cpu-bind` mode
(`none`, `cores`, `threads`, `map_cpu`, `mask_cpu`) leaves each task on all 4
CPUs. Tasks are not placed in a per-job cgroup (`/proc/self/cgroup` is
`0::/`), and `--mem=1G` leaves `memory.max` at `max`. GPU isolation is only the
`CUDA_VISIBLE_DEVICES` Slurm exports. Slinky's slurm-operator can enforce all
of these (privileged slurmd, Slurm ≥ 25.11), so it is a cluster configuration
gap: `TaskPlugin=task/cgroup,task/affinity` plus `ConstrainCores`,
`ConstrainRAMSpace`, `ConstrainDevices`. Until then, outpost refuses GTL's
`launch` intent here rather than pretend to honour it.

Also seen, outside the backend itself: the reconciler advances a tick only
when every job's phase has returned, so a short job finishing next to a long
one waits for it (STREAM waited ~80 s for HPL to finish its run).

## What was verified, and what was not

Verified live, on T4 (2 × g4dn.xlarge) and A10G (1 and 2 × g5.xlarge) clusters:
- single-node GPU jobs (`launcher: single`), with a plain image and an NGC image;
- import → submit → poll → collect → teardown, and back-to-back jobs on one slot;
- a failure path (an enroot import failure fails the job and cleans up);
- **per-node slots** (2 × A10G): `add-slinky --per-node` found both Slurm
  nodes. A job pinning them in reverse order ran rank 0 on the first pin, with
  `MASTER_ADDR` = that node and one GPU per rank, and PMIx ranks matching Slurm's.
  Two 1-node jobs on the two slots ran at the same time;
- GTL through `gtl submit --target` → `gtl ingest`, all fingerprints recording
  `scheduler: slurm`:
  - STREAM, 3 runs: Triad 523.5 GB/s (87.3% of 600 GB/s), variance 0.65%. This
    matches the non-Slinky A10G Triad in staging (523.4);
  - HPL-MxP at `n=36864`, 3 runs: 13,406 GFLOPS, variance 1.6%. That's 8% below
    the staging g5.xlarge record (14,620). HPL-MxP is CPU-heavy outside the
    solve, and this worker also hosts slurmctld and the platform's operators;
  - nccl-tests `all_reduce` across both nodes (`launcher: mpi`, `srun
    --mpi=pmix` inside the NGC image): real NCCL traffic over the pod network,
    zero wrong results, about 0.9 GB/s bus bandwidth on a 10 Gbit/s link;
- a slurmctld outage while a job runs (see above): the adapter now waits it out.

Not yet exercised live:

- **Multi-GPU nodes.** The tashkil AWS account's on-demand G quota (8 vCPU)
  cannot hold a 4-GPU instance, so `--exclusive` with several GPUs per node and
  more than one rank per node are untested.
- **Private registries (ECR):** `enroot import` needs credentials
  (`$ENROOT_CONFIG_PATH/.credentials`) in the login pod; none are configured.
