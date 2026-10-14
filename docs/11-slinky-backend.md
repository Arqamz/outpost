# 11 · Slinky backend: Slurm on Kubernetes

> **Status: implemented and live-verified** on a tashkil AWS child cluster
> (2026-10-14): kubeadm 1.35, 2 × g4dn.xlarge (T4), Slinky 1.2.2, Slurm 26.05,
> pyxis/enroot, `/scratch` on FSx for Lustre. A GPU smoke job and GTL's STREAM
> suite both ran from `submit` through the reconciler to `promoted`, with
> output in the drop-zone.

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
| `node_count` | `--nodes=N`, one task per node |
| `launcher: single` | `srun --ntasks=1` (first node) |
| `launcher: mpi` | `srun --ntasks=N --mpi=$CLUSTER_SLINKY_MPI` (default `pmix`) |
| `gpu: true` | `--gres=gpu:<params.gpu_count, default 1>` |
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

As with k8s, a Slinky cluster appears in the registry as **slots** (provider
`slinky`). A slot is a concurrency bound on one cluster, not a machine. A plain
pool claim never takes a backend slot (`BACKEND_PROVIDERS` in `models.py`).
A job reaches a slot in one of two ways:

- `backend: slinky`, optionally with `params.slinky_context` (the slot's
  kube-context) or `params.node_name` (one exact slot);
- **a `params.node_name` pin naming the slot, with no `backend` set.** This is
  how GTL gets there: its deployment targets pin node names, so a target whose
  `node_names` is `[slinky-aws-t4]` runs on that cluster without GTL knowing
  Slurm exists. The reconciler sees the pinned node is a slinky slot and
  claims that one slot (`node_count` stays the Slurm node count).

```bash
cluster add-slinky --name aws-t4 --context <kube-context> --gpu   # -> slot slinky-aws-t4
cluster add-slinky --name big --context big --slots 4               # -> slinky-big-0..3
cluster submit --spec job.slinky.smoke.yaml
cluster reconcile --execute
```

The reconciler needs a kubeconfig that reaches the cluster (`KUBECONFIG` plus the
slot's kube-context).

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

## What was verified, and what was not

Verified live: single-node GPU jobs (`launcher: single`) on a T4, a plain image
and an NGC image, import → submit → poll → collect → teardown, three
back-to-back jobs on one slot, failure paths (an enroot import failure fails the
job and cleans up).

Not yet exercised live:

- **`launcher: mpi` across nodes:** the batch script is unit-tested, but
  whether `--mpi=pmix` works in the stock slurmd image is unconfirmed;
  `pmi2` is the fallback (`CLUSTER_SLINKY_MPI`).
- **Multi-GPU nodes:** with `NVIDIA_VISIBLE_DEVICES=all`, it's unconfirmed
  that a task requesting fewer GPUs than the node has sees only its own.
- **Private registries (ECR):** `enroot import` needs credentials
  (`$ENROOT_CONFIG_PATH/.credentials`) in the login pod; none are configured.
