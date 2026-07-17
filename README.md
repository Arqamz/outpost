# Outpost

A small, self-contained compute cluster you can point generic containerized
jobs at. Hand it a JobSpec — a container image, a command, and how much it
needs — and it provisions the resources, runs the job, and hands back
whatever the job produced (stdout, files, exit code). It doesn't know or care
what the job actually does: it spins up local libvirt/KVM VMs for CPU work,
and lets the host machine itself join the pool as a node for GPU work, then
just runs the container it's handed.

The interface is deliberately open — JobSpec in, drop-zone out — so anything
can be the caller: a shell script, a CI pipeline, a bigger scheduler you build
later, or just you poking at the CLI. Nothing here assumes or waits on one
specific consumer.

Apptainer on VMs is the only backend today. Slurm/Slinky, KinD, and
distributing jobs across multiple nodes are planned for later.

**📚 Full walkthrough: [`docs/`](docs/README.md).** Command reference:
[`CLAUDE.md`](CLAUDE.md).

**Status: live-verified end to end** on a real multi-VM + host-GPU cluster —
single-node containers, real multi-node MPI (`mpirun` across claimed nodes),
and real GPU jobs all provision, bootstrap, schedule under contention
(wait-and-retry with a timeout, not immediate failure), run, and land their
output in the drop-zone. A live dashboard (`make dashboard`) shows the node
stats, job queue, and full per-job logs on one page.

## What this is (and isn't)

- **Is:** a generic job-running cluster — a reconciler (state machine + node
  registry with exclusive locks, concurrent job scheduling), provider
  adapters (`libvirt` VMs + `localhost` GPU) that both provision *and* run
  containers via **apptainer** — single-node or real multi-node MPI — a NAT
  network with stable IPs, the `bootstrap` ansible role, a failure-
  injection harness, and a live dashboard.
- **Isn't:** An orchestrator. It pulls **no** jobs itself — a job brings its own container
  image; the cluster just runs it.
- **No separate executor layer** — running a container is what an adapter does
  once it owns a node (see [`docs/01-architecture.md`](docs/01-architecture.md)).

## Launch configuration (all of it lives in `infra/libvirt/config.env`)

None of the numbers below are load-bearing architecture — they're just
today's defaults in one file. Change any of them for your own machine:

| Setting | Default | Why this default |
|---|---|---|
| Guest OS | Ubuntu 24.04 cloud image | boring, well-supported, apt-based |
| GPU | host joins the pool as a GPU node (no passthrough) | one real GPU on the host → GPU jobs run there via apptainer `--nv`; CPU jobs run on VMs |
| Container runtime | apptainer (primary), docker optional | rootless, HPC-native, `--nv` binds host GPU with no daemon |
| Provisioning | `virsh define/start` + domain XML | no `virt-install`, no SLIRP |
| Networking | libvirt NAT `cluster0`, static leases | routable per-VM IPs + VM↔VM traffic for multi-rank MPI (TCP) |
| Node count/size | 8 × (2 vCPU, 3 GB) VMs + 1 host GPU node | fits comfortably on a modest dev box, with headroom |

> Override any of these in `config.env`.

## Prerequisites

- `libvirtd` active + enabled (system service), user in `libvirtd`/`kvm`/`docker`
- `/dev/kvm` present, virtualization (AMD-V/Intel VT-x) enabled
- A recent NVIDIA driver on the host, only if you want the optional host GPU
  node (CPU-only jobs work without one)

All *client* tooling (qemu, virsh, ansible, cloud-utils, terraform, packer, …)
is pinned in the Nix dev shell — you do **not** need it installed globally.

**No Nix?** On a native Ubuntu host, follow
[`docs/07-ubuntu-setup.md`](docs/07-ubuntu-setup.md) instead — apt-based
install of the same tooling plus `source env.sh` in place of the dev shell,
with runbooks for a host-only GPU node (0 VMs) and a hybrid VM + host-GPU job.

## Quickstart

Generate the dedicated cluster SSH key referenced from `config.env` (path
configurable via `CLUSTER_SSH_KEY`, default `~/.ssh/outpost-cluster-ssh`):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/outpost-cluster-ssh -N ""
```

> **Network gate:** internet-reaching steps (the base-image download, guests'
> first-boot `apt`) are blocked unless `CLUSTER_ALLOW_NETWORK=1` in `config.env`
> (or you confirm the interactive prompt). Default is `0`.

```bash
nix develop            # or: nix-shell   (enters the pinned dev shell)
$EDITOR infra/libvirt/config.env   # review; set CLUSTER_ALLOW_NETWORK=1 when ready

make net-up            # define + start the cluster0 network (8 static leases)
make template          # download Ubuntu cloud img -> base qcow2 (needs the gate)
make cluster-up        # define + start 8 VMs, wait for SSH, write inventory
make status            # observe: state / IP / lease / SSH per node

# configure the guests (the one bootstrap role)
ansible-galaxy collection install -r infra/ansible/requirements.yml
make bootstrap         # docker + mpi + hostfile + health (pulls no images)

# teardown
make cluster-down      # destroy all VMs (network + template kept)
make clean             # + undefine the network
```

### Control plane (`cluster`) — dry-run by default, nothing launched

```bash
make seed-nodes                              # register 8 VMs + the cluster-host GPU node
cluster nodes                            # see capabilities (cpu / gpu-local)
cluster submit --spec job.example.yaml   # -> job-id into jobs
make reconcile                               # advance the state machine one tick
make reconcile                               # ... repeat to PROMOTED
cluster status <job_id>                  # full audit trail + run/drop info
cluster result <job_id>                  # where artifacts landed (drop-zone)
cluster logs <job_id>                    # full replay: every command + its output
cluster reconciler-log                   # cross-job chronological timeline
cluster clear-jobs [--force]             # demo reset: wipe history, reset nodes to
                                              # available (also a button on the dashboard)

# fail a node on purpose, to see the cleanup path:
cluster fail-node cluster-node-03            # quarantine + clean job fail (audited)
cluster fail-node cluster-node-03 --inject   # ...and actually virsh-destroy the VM
```

- **dry-run (default):** `NullAdapter` — the whole state machine + scheduling +
  audit runs with nothing launched.
- **`reconcile --execute`:** real adapters — GPU jobs run `apptainer --nv` on
  `cluster-host`, CPU jobs run via ssh+apptainer on the VMs (single-node or, with
  `launcher: mpi`, a real multi-node `mpirun` launch); artifacts → drop-zone.
  Jobs advance **concurrently**; a job that can't get capacity waits and
  retries (with a timeout) instead of failing immediately.

### Live dashboard — node stats + job queue + logs, one page

```bash
make dashboard      # open http://localhost:8087
```

Per-VM CPU/RAM + host GPU utilization, the live job queue (click a row to see
that job's full replay log inline, auto-refreshing while it's still running),
and a cross-job timeline view. Reads the same store and `.var/logs/` files
`cluster` does — no separate state to drift.

### Multi-node MPI jobs

```bash
make mpi-sif         # build demo/mpi_demo.sif once (rootless via build-sif.sh; proot on NixOS)
make submit-mpi       # submit job.mpi.example.yaml (node_count: 4)
make reconcile        # ... repeat to PROMOTED (or `cluster reconcile --execute` to loop)
```

`job.mpi.example.yaml`/`job.mpi.small.example.yaml` set `launcher: mpi`. The
reconciler builds a per-job hostfile from exactly the nodes that job claimed,
stages the image to each, and launches real `mpirun` from the head node — a
hybrid launch where `mpirun` is the host's own OpenMPI and only the per-rank
command runs inside the container (see `demo/mpi_demo.def`).

### See it all end to end: the scheduling demo

```bash
make demo-scheduling
```

Submits 4 jobs at once — two `node_count: 4` MPI jobs that together fill the
8-VM pool, a third `node_count: 2` MPI job that has to **wait** for capacity
and succeeds the instant one of the first two tears down, and a GPU job that
runs independently the whole time — then ticks the reconciler live until
everything reaches a terminal state, narrating each tick.

See [`docs/`](docs/README.md) for the full picture and
[`docs/06-interface-contract.md`](docs/06-interface-contract.md) for the
job-in / output-out contract.

## Layout

```
flake.nix / shell.nix              reproducible dev shell (all tooling pinned)
Makefile                           convenience targets
infra/libvirt/
  config.env                       ← single source of truth (edit here)
  lib.sh                           shared helpers, deterministic node name/ip/mac, wait_ssh
  templates/domain.xml.tpl         KVM guest definition (virsh define)
  cloud-init/*.tpl                 NoCloud seed (SSH key, hostname)
  build-template.sh                base qcow2 build
  net-up.sh / net-down.sh          cluster0 NAT network + static leases
  vm-define.sh / vm-destroy.sh     ProviderAdapter: provision / deprovision one node
                                    (provision blocks until the node is SSH-reachable)
  cluster-up.sh / cluster-down.sh  whole-cluster orchestration (parallel boot)
  cluster-status.sh                observability
  gen-inventory.sh                 libvirt state -> ansible inventory
  inject-failure.sh                kill / pause / netcut a node mid-job
infra/ansible/
  ansible.cfg / site.yml           the one bootstrap entrypoint; waits for
                                    cloud-init's own first-boot apt before the role runs
  group_vars/all.yml               cluster-wide variable overrides
  roles/bootstrap/             driver-check · apptainer (pinned .deb) · docker ·
                                   mpi-fabric · ssh-fabric · runner-image · hostfile ·
                                   healthcheck
reconciler/                        the control plane:
  states.py models.py store.py     state machine · records (incl. JobSpec.launcher) ·
                                    file/Mongo store (fcntl shared/exclusive locked)
  registry.py adapter.py           node pool + exclusive locks · adapters (run/collect,
                                    single-node + multi-node MPI, per-job replay logging)
  reconciler.py cli.py audit.py    the driver (concurrent tick, wait+timeout scheduling)
                                    · cluster CLI (incl. logs/reconciler-log) · audit log
demo/                               mpi_demo.c/.def/.sif · run-mpi-demo.sh (bare OpenMPI) ·
                                     run-scheduling-demo.sh (4-job contention + GPU demo)
viz/                                dashboard.py + cluster.html — node stats, job queue,
                                     per-job/cross-job log viewer, one page
docs/                              full walkthrough (start at docs/README.md)
bin/cluster                    control-plane CLI launcher
job.example.yaml                   sample JobSpec (dry-run)
job.mpi.example.yaml               multi-node MPI JobSpec (node_count: 4)
job.mpi.small.example.yaml         multi-node MPI JobSpec (node_count: 2)
job.gpu.example.yaml               GPU JobSpec (nvidia-smi via apptainer --nv)
job.hybrid.example.yaml            hybrid JobSpec (host GPU node + VM in ONE mpirun)
job.hybrid.cuda.example.yaml       hybrid JobSpec (VM CPU rank -> host GPU rank, y=a*x+b)
```

## Running a real job

Nothing is pulled unless a job asks for it. A job **brings its own
container** — the cluster runs it and returns the output. Three shapes,
all live-verified end to end:

**Single-node, any container:**
```yaml
name: hello-apptainer
node_count: 1
image: "docker://alpine:3.20"
command: ["sh", "-c", "echo hello > /out/result.txt"]
output_dir: "/out"
```

**Multi-node MPI** (`launcher: mpi`, real `mpirun` across the claimed nodes):
```yaml
name: mpi-job
launcher: mpi
node_count: 4
image: "demo/mpi_demo.sif"     # or your own MPI-enabled image
command: ["/opt/your_binary"]
output_dir: "/out"
```

**GPU** (routes to the host node, `apptainer --nv`):
```yaml
name: gpu-smoke
gpu: true
node_count: 1
image: "/path/to/job.sif"      # or docker://…
command: ["python", "run.py", "--out", "/out"]
output_dir: "/out"
```

**Hybrid** (`hybrid: true` — the host GPU node **and** VM CPU node(s) in ONE
`mpirun`, launched from the host; the only shape that mixes the two pools):
```yaml
name: hybrid-job
launcher: mpi
hybrid: true                   # 1 GPU node (the host) + (node_count-1) CPU VMs
node_count: 2
gpu: true
image: "demo/mpi_demo.sif"     # host mpirun must match the VMs' OpenMPI (4.1.x)
command: ["/opt/mpi_demo"]
output_dir: "/out"
```

```bash
jid=$(cluster submit --spec job.yaml)
cluster reconcile --execute       # or: cluster reconcile --execute --interval 3 (loop)
cluster result "$jid"             # -> drop-zone path
cluster logs "$jid"               # full replay: every command that ran + its output
cat "$(cluster result "$jid")/stdout.log"   # the container's captured stdout+stderr
```

`stdout.log` in the drop-zone is guaranteed on every path (single-node, MPI,
GPU) — that's where a caller reads a job's printed output (for tools that
report to stdout, not just files) for parsing into its own result schema.

A different host or fabric just flips `cluster_skip_driver_check`/
`cluster_fabric` in `group_vars/all.yml`; the role body and the reconciler
never change.
