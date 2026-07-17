# 7 · Running Outpost on a native Ubuntu host (no Nix)

Everything on this page was written for a fresh **Ubuntu 24.04** machine and
assumes **no Nix** — all tooling comes from apt plus one pinned upstream
`.deb`. It covers the full lifecycle: install → setup → bringing the cluster
up → the dashboard → submitting jobs through the open interface, with two
concrete runbooks:

- **Scenario 1 — host-only GPU node (0 VMs):** the machine itself is the whole
  pool; a `gpu: true` job runs on it via apptainer `--nv`.
- **Scenario 2 — 1 VM + the host, one job across both:** a `hybrid: true` MPI
  job — a single `mpirun` with rank 0 on the host (GPU) and rank 1 on the VM,
  communicating over the cluster0 network.

Ubuntu 22.04 works too, but the guest VMs run Ubuntu 24.04 with OpenMPI
4.1.x — check `mpirun --version` on the host matches 4.1.x (see
[Troubleshooting](#troubleshooting)) before running hybrid jobs.

---

## 1 · Install

### System packages

```bash
sudo apt update
sudo apt install -y \
  qemu-system-x86 qemu-utils \
  libvirt-daemon-system libvirt-clients \
  cloud-image-utils genisoimage \
  ansible \
  openmpi-bin \
  make git python3 python3-yaml \
  openssh-client netcat-openbsd wget curl jq
```

What each is for: `qemu-*` + `libvirt-*` run the VMs; `cloud-image-utils`
(provides `cloud-localds`) + `genisoimage` build the cloud-init seed ISOs;
`ansible` runs the one bootstrap role; `openmpi-bin` makes the host the
`mpirun` head for hybrid jobs (4.1.x on 24.04 — the exact version the guests
get, which matters: `mpirun` on the host talks to `orted` on the VMs);
`python3-yaml` is the only third-party Python dependency of the reconciler;
`netcat-openbsd` backs the SSH-readiness wait.

### Apptainer

Ubuntu's own archive does **not** package apptainer (universe only has the
unrelated `singularity-container` fork under a different binary name), so
install the upstream project's pinned `.deb` — the **same version + checksum**
the bootstrap role installs into the VMs
(`infra/ansible/roles/bootstrap/defaults/main.yml`), so SIF behavior matches
across the pool:

```bash
APPTAINER_VERSION=1.5.0
APPTAINER_SHA256=fbc27204d0ec0440dfa0ae589089e4b2baf192315b4ac22dfae02b78b28981ea
wget -O /tmp/apptainer.deb \
  "https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer_${APPTAINER_VERSION}_amd64.deb"
echo "${APPTAINER_SHA256}  /tmp/apptainer.deb" | sha256sum -c
sudo apt install -y /tmp/apptainer.deb
apptainer --version
```

### NVIDIA driver (only for GPU jobs)

CPU-only use needs none of this. For GPU jobs the host needs a recent driver
(≥ 550) — `apptainer --nv` binds the driver's userspace libraries into the
container by itself, no nvidia-container-toolkit required:

```bash
sudo ubuntu-drivers install    # or install a specific nvidia-driver-5xx
nvidia-smi                     # must work before any gpu: true job will
```

### Enable virtualization

```bash
sudo systemctl enable --now libvirtd
sudo usermod -aG libvirt,kvm "$USER"
# log out/in (or `newgrp libvirt`) for the group change to take effect, then:
ls -l /dev/kvm                       # must exist (enable VT-x/AMD-V in BIOS if not)
virsh -c qemu:///system version      # must connect without sudo
```

---

## 2 · Setup

```bash
git clone <this-repo> outpost && cd outpost
source env.sh        # the non-nix replacement for the dev shell's env:
                     # CLUSTER_ROOT, PATH (bin/), PYTHONPATH, qemu:///system,
                     # apptainer cache under .var/. Source it in every new shell.
```

Generate the dedicated cluster SSH key (injected into VMs via cloud-init and
used for every host→VM connection, including hybrid `mpirun` launches):

```bash
ssh-keygen -t ed25519 -f ~/.ssh/outpost-cluster-ssh -N ""
```

Install the ansible collections the bootstrap role uses:

```bash
ansible-galaxy collection install -r infra/ansible/requirements.yml
```

Review `infra/libvirt/config.env` — the single source of truth for cluster
shape. For this guide the values that matter:

| Setting | Scenario 1 | Scenario 2 | Notes |
|---|---|---|---|
| `CLUSTER_NODE_COUNT` | `0` | `1` | number of VMs (8 is the default; any value works) |
| `CLUSTER_GPU_HOST` | `1` | `1` | registers the host as the GPU node (default) |
| `CLUSTER_ALLOW_NETWORK` | — | `1` | gates the base-image download + guests' first-boot apt |
| `CLUSTER_GPU_BINDS` | auto | auto | auto-empty on Ubuntu (NixOS-only bind paths); leave it |
| `CLUSTER_QEMU_EMULATOR` | — | auto | auto-detects `/usr/bin/qemu-system-x86_64` on Ubuntu |

---

## 3 · Scenario 1 — host-only GPU node (0 VMs)

Set `CLUSTER_NODE_COUNT=0` in `config.env`. No network, template, or VMs are
needed — the host is the entire pool:

```bash
source env.sh
make seed-nodes            # registers exactly one node: cluster-host (gpu/local)
bin/cluster nodes          # -> cluster-host ... available gpu/local apptainer
```

Submit the GPU smoke job (pulls `docker://nvidia/cuda:...` and runs
`nvidia-smi` inside it — this is the one step that reaches the internet, via
apptainer's own pull) and drive it for real with `--execute`:

```bash
jid=$(bin/cluster submit --spec job.gpu.example.yaml)
bin/cluster reconcile --execute --interval 2   # loop; Ctrl-C once it's promoted
                                               # (or repeat: bin/cluster reconcile --once --execute)
bin/cluster status "$jid"                      # state: promoted
cat "$(bin/cluster result "$jid")/stdout.log"  # the container's nvidia-smi output
bin/cluster logs "$jid"                        # full replay transcript
```

Any container works the same way — bring your own image, set `gpu: true`,
`node_count: 1`, and read the results from the drop-zone
(`.var/dropzone/<job_id>/`). CPU-only single-node jobs on the host are not a
thing in this scheduler (CPU jobs go to VMs), so with 0 VMs every job you
submit should be `gpu: true`.

## 4 · Scenario 2 — 1 VM + the host, one MPI job across both

Set `CLUSTER_NODE_COUNT=1` and `CLUSTER_ALLOW_NETWORK=1` in `config.env`, then
bring up the infrastructure plane:

```bash
source env.sh
make net-up          # cluster0 NAT network (host = 192.168.71.1, VM lease .11)
make template        # download Ubuntu 24.04 cloud image -> base qcow2 (once)
make cluster-up      # define + boot cluster-node-01, wait for SSH (~1-2 min)
make bootstrap       # ansible: apptainer (pinned .deb) + OpenMPI + ssh fabric
make status          # verify: running / .11 / SSH OK
make seed-nodes      # register cluster-node-01 + cluster-host in the pool
```

Build the demo MPI image once (any MPI-enabled image with a libmpi matching
OpenMPI 4.1.x works — see `demo/mpi_demo.def` for the pattern):

```bash
make mpi-sif         # apptainer build demo/mpi_demo.sif (needs --fakeroot or sudo)
```

Optional but recommended the first time — prove the fabric by hand before
involving the reconciler (this is exactly the launch the adapter performs):

```bash
mpirun -np 2 \
  --hostfile <(printf '192.168.71.1 slots=1\n192.168.71.11 slots=1\n') \
  --map-by node --mca btl tcp,self \
  --mca btl_tcp_if_include 192.168.71.0/24 --mca oob_tcp_if_include 192.168.71.0/24 \
  --mca plm_rsh_agent "ssh -i $HOME/.ssh/outpost-cluster-ssh -o IdentitiesOnly=yes -o StrictHostKeyChecking=no -o UserKnownHostsFile=/dev/null -l cluster" \
  --mca plm_rsh_no_tree_spawn 1 \
  hostname
# expected: your host's hostname + cluster-node-01
```

Rank 0 is forked locally (192.168.71.1 is the host's own bridge address — no
sshd needed on the host); the VM rank is launched over ssh with the cluster
key. Now the real thing:

```bash
make submit-hybrid                   # submits job.hybrid.example.yaml
                                     # (launcher: mpi, hybrid: true, node_count: 2)
bin/cluster reconcile --execute --interval 2
bin/cluster list                     # ... promoted
cat "$(bin/cluster result <job_id>)/stdout.log"
# both ranks report in: one from your host (the GPU rank), one from cluster-node-01
```

What `hybrid: true` means: the job claims **1 GPU node (the host) + the
remaining `node_count-1` CPU VMs** and launches a single `mpirun` from the
host across all of them. With more VMs (`CLUSTER_NODE_COUNT=4`,
`node_count: 3`, …) the same spec scales out. The GPU-less VM ranks log an
apptainer `--nv` warning and continue — expected.

Teardown when done:

```bash
make cluster-down    # destroy the VM(s); network + template kept
make clean           # + undefine the network
```

## 5 · Live dashboard

```bash
make dashboard       # -> http://localhost:8087  (port: CLUSTER_DASH_PORT)
```

One page: per-VM CPU/RAM (via `virsh domstats`), host CPU/RAM + GPU (via
`nvidia-smi`), the live job queue (click a row for that job's full replay
log), and a cross-job timeline. Python stdlib only — reads the same store and
`.var/logs/` the CLI does, so there is no separate state to drift.

## 6 · Driving it programmatically (the open interface)

The `cluster` CLI is just a local client of the same interface anything else
can target (full contract: [06-interface-contract.md](06-interface-contract.md)):

- **Intake:** a JobSpec document in the `jobs` collection —
  `bin/cluster submit --spec job.yaml` from any script/CI, or insert the
  document directly into MongoDB if you run the store with `CLUSTER_MONGO_URI`.
- **Egress:** artifacts land in `.var/dropzone/<job_id>/` (override with
  `CLUSTER_DROPZONE`): `stdout.log` (guaranteed on every path) plus whatever
  the job wrote to its `output_dir`.
- **Status:** poll `jobs` (`bin/cluster status <id>` / `list`) until
  `promoted` (success — trust the drop-zone) or `failed` (`error` has the
  reason). The `audit` collection has every transition.

A minimal driver loop:

```bash
jid=$(bin/cluster submit --spec my-job.yaml)
bin/cluster reconcile --execute --interval 2 &   # or run it as a service
until bin/cluster status "$jid" | grep -qE 'state   : (promoted|failed)'; do sleep 5; done
cat "$(bin/cluster result "$jid")/stdout.log"
```

## 7 · Troubleshooting

**qemu permission denied on disk/seed files (AppArmor/DAC).** Ubuntu's
libvirt runs qemu as the `libvirt-qemu` user, which must be able to traverse
into the repo's `.var/` under your home directory (overlays + seed ISOs live
there). If `virsh start` fails with `Permission denied`:

```bash
# let libvirt-qemu traverse the path components (x only, no read):
setfacl -m u:libvirt-qemu:x "$HOME" "$HOME/path/to/outpost" "$HOME/path/to/outpost/.var"
```

or relocate the generated state outside your home by setting
`CLUSTER_VAR_DIR` in `config.env`.

**ufw blocks VM→host MPI traffic.** Default Ubuntu has ufw inactive — nothing
to do. If you've enabled it, hybrid jobs will hang at the mpirun step
(VM ranks can't dial back to the host):

```bash
sudo ufw allow in on virbr-cluster from 192.168.71.0/24
```

**OpenMPI version mismatch.** `mpirun` (host) and `orted` (VMs) must be the
same release series. Ubuntu 24.04 apt gives 4.1.x on both sides. If your host
runs something else, hybrid launches fail with protocol/handshake errors —
check `mpirun --version` on the host against
`ssh -i ~/.ssh/outpost-cluster-ssh cluster@192.168.71.11 mpirun --version`.

**Stale node registry after changing config.** `bin/cluster seed-nodes`
refreshes idle nodes whose config-derived identity changed (e.g. the host
node's IP), but never touches claimed ones. For a hard reset with nothing
running: `bin/cluster clear-jobs` and/or delete
`.var/reconciler/state.json`, then re-seed.

**`missing 'virsh'` / `missing 'qemu-img'`.** You haven't installed the apt
packages above, or you're in a fresh shell without `source env.sh`.

**VM never becomes SSH-reachable.** First boot runs cloud-init (~30-90 s;
with `CLUSTER_ALLOW_NETWORK=1` it also apt-installs the guest agent). Watch it
directly: `virsh console cluster-node-01`.
