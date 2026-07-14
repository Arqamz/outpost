# 2 · Infrastructure (`infra/`)

Two sub-layers: **libvirt** makes nodes exist; **ansible** configures them.

## `infra/libvirt/` — making VMs

Model: **one base image, many cheap copies.**

| File | Role |
|------|------|
| `config.env` | ⭐ single source of truth. Node count/size, network, SSH key, `CLUSTER_ALLOW_NETWORK` gate, **host GPU node** settings, **drop-zone** path. Everything reads from here. |
| `lib.sh` | shared bash. Deterministic helpers: `node_name/ip/mac(i)`. Plus `require_network` (the gate), `preflight`, and `wait_ssh(ip, name, timeout_s)` — polls until SSH answers or times out; the single place "is this node actually up" is decided. |
| `build-template.sh` | download Ubuntu cloud image → base `qcow2`. Network-gated. |
| `net-up.sh` / `net-down.sh` | the `cluster0` NAT network with **static DHCP leases** (MAC→IP) so node-`i` is always `.1{i}`. |
| `vm-define.sh <i>` | provision one VM: thin qcow2 overlay + cloud-init seed (SSH key) + domain XML → `virsh define`+`start` → **blocks until SSH actually answers** (`wait_ssh`, every code path: fresh boot, restart, already-running). `provision()` means "usable," not "requested" — bootstrapping a node before it's really up fails with a misleading SSH error, which is exactly what happened before this was added. |
| `vm-destroy.sh <i>` | inverse. |
| `cluster-up.sh` / `cluster-down.sh` | `cluster-up` starts all N nodes **in parallel** (background + `wait`) — each `vm-define.sh` call blocks on its own SSH readiness internally, so running them sequentially would turn an 8-VM cluster into 8 sequential ~60-90s cloud-init waits; in parallel it's however long the slowest one takes (8 VMs: ~15s in practice). Writes the inventory once all are up. |
| `cluster-status.sh` | per-node state / IP / lease / SSH. |
| `gen-inventory.sh` | live libvirt state → `infra/ansible/inventory/hosts.ini` (bridge to the ansible layer). |
| `inject-failure.sh <i>` | chaos: `kill` / `pause` / `netcut` a node. |

**Lifecycle:** `net-up → template → cluster-up → … → cluster-down → net-down`.

The **host GPU node is NOT here** — it has no VM to build. It's registered
directly in the reconciler (see [05-gpu-and-apptainer.md](05-gpu-and-apptainer.md))
and is `local`, so `cluster-down`/`inject-failure` (which act on VM indices
`1..N`) never touch it.

## `infra/ansible/` — configuring nodes

The `bootstrap` role is *the one bootstrap role*, full stop — no per-host
variants. It is a sequence of **variable-gated** task files
(`roles/bootstrap/tasks/`):

| Task | Default | Notes |
|------|---------|-------|
| `driver-check.yml` | skipped | `cluster_skip_driver_check: true` (VMs have no GPU) |
| `apptainer.yml` | **on** | primary runtime; downloads a **pinned, checksum-verified `.deb`** from the apptainer GitHub releases and installs it via `apt` (deps resolved automatically). **Not** a PPA, and not Ubuntu's own archive — Ubuntu doesn't package `apptainer` under that name at all (only the unrelated `singularity-container` fork, different binary). Version pinned to match the host's own apptainer (nix shell) so SIF/ABI behavior is consistent host vs. VM. |
| `docker.yml` | off | optional; kept for parity |
| `mpi-fabric.yml` | on | OpenMPI + writes fabric env (TCP) |
| `ssh-fabric.yml` | on (with mpi) | stages the cluster's own SSH key + a permissive client config onto **every** node, so whichever node a job's reconciler run picks as head (`nodes[0]`, varies per job) can `ssh` to the others as the `mpirun` launcher — without this, only a fixed node could ever be an MPI head. |
| `runner-image.yml` | off | pulls a runner image only when you set one |
| `hostfile.yml` | on | cluster-wide MPI hostfile from the inventory (legacy/vestigial now — real jobs get a fresh **per-job** hostfile built by the reconciler from just the nodes that job claimed, not the whole cluster) |
| `healthcheck.yml` | on | asserts runtime + mpi + hostfile, else fails cleanly |

Because it's apt-based, the *same* role would run unchanged against any other
apt-based host — a different libvirt image, a real cloud VM, whatever shows up
later. Nothing here pulls anything unless you configure it to; a job's own
container is what actually runs, at run time.

`site.yml` runs the role against the `nodes` group, scoped per invocation
via `--limit <node names>` when called from the reconciler (see
[03-control-plane.md](03-control-plane.md)) so one job's bootstrap never
touches another concurrently-provisioning job's nodes. A `pre_tasks` step waits
for cloud-init's own first-boot `apt-get` (package_update + qemu-guest-agent
install) to finish before the role's own apt tasks run — SSH answers well
before that finishes, and without this wait the two `apt-get` invocations race
for the dpkg lock. `group_vars/all.yml` holds the cluster-wide overrides;
`ansible.cfg` sets the SSH options + the generated inventory path —
**ansible only auto-discovers `ansible.cfg` relative to the current working
directory**, not the playbook's path, so every invocation (including the
reconciler's) must run with `infra/ansible/` as `cwd`.
