"""ProviderAdapter — how the cluster gets a node AND runs a container on it.

There is deliberately no separate "executor" abstraction: running a container
is just what an adapter does once it owns a node. Which adapter is used is
decided by the node itself — a `local` node (the control-plane host,
GPU-capable) uses LocalHostAdapter (apptainer --nv, run in-process); any other
node uses LibvirtAdapter (ssh + apptainer into the VM). NullAdapter is the
dry-run no-op. A future backend (real cloud VMs, KinD, Slinky, whatever) just
implements the same five methods.
"""
from __future__ import annotations
import dataclasses
import fcntl
import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import threading
import time
from abc import ABC, abstractmethod
from pathlib import Path

from .models import NodeRecord, JobSpec, RunResult
from .store import now_iso
from .topology import PROBE_PATH

# Runs ONCE PER RANK inside a k8s pod (KubernetesAdapter, opt-in via
# spec.params.verify_gpu_memory) — see the script for why it must run inside
# the container rather than being read off the pod spec.
GPU_MEMORY_PROBE_PATH = Path(__file__).resolve().parent / "probes" / "gpu-memory-probe.sh"
_GPU_MEMORY_MARKER = "===GPU_MEMORY_PROBE==="


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


def _is_staging_artifact(name: str) -> bool:
    """Staging files that MUST NOT be collected into the drop-zone: the per-node
    container image (a ~5 GB SIF staged into the workdir), the MPI hostfile, and
    (for a planned launch: mpi run) the compiled launcher appfile/rankfile/
    per-rank scripts. collect() ships the whole workdir, so pulling the image
    back duplicated it once per job and filled /home to 100% (2026-07-21) —
    corrupting the state file mid-write. The image is content-addressed +
    cached (.var/sif-cache) and fully regenerable; only stdout.log, the
    output_dir files, and the three launch-*.yaml artifacts are real results."""
    return (name.endswith(".sif") or name == "hostfile" or
           name.startswith("launcher-"))


REPO_ROOT = os.environ.get("CLUSTER_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBVIRT_DIR = os.path.join(REPO_ROOT, "infra", "libvirt")
ANSIBLE_DIR = os.path.join(REPO_ROOT, "infra", "ansible")
RUNS_DIR = os.path.join(REPO_ROOT, ".var", "runs")   # per-job host workdirs
LOGS_DIR = os.path.join(REPO_ROOT, ".var", "logs")   # replay transcripts (gitignored)
SIF_CACHE_DIR = os.path.join(REPO_ROOT, ".var", "sif-cache")  # docker://→SIF, keyed by digest

# SSH params for reaching VMs (match infra/libvirt/config.env defaults).
SSH_USER = os.environ.get("CLUSTER_SSH_USER", "cluster")
SSH_KEY = os.environ.get("CLUSTER_SSH_PRIVKEY", os.path.expanduser("~/.ssh/outpost-cluster-ssh"))

# ── Kubernetes (KAI + HAMi) backend params ──────────────────────────────────
# The whole cluster + KAI + HAMi are stood up out of band by infra/k8s/setup.sh;
# the adapter only templates kubectl against it. All env-overridable so a
# differently-installed cluster (other queue, other CRD version) needs no code
# change. Defaults verified live against the `outpost` kind cluster 2026-07-30.
K8S_NAMESPACE_PREFIX = os.environ.get("CLUSTER_K8S_NS_PREFIX", "outpost-")
K8S_PODGROUP_APIVERSION = os.environ.get("CLUSTER_K8S_PODGROUP_APIVERSION",
                                         "scheduling.run.ai/v2alpha2")
K8S_QUEUE = os.environ.get("CLUSTER_K8S_QUEUE", "default-queue")          # KAI queue
K8S_GPU_MEMORY_MB = int(os.environ.get("CLUSTER_K8S_GPU_MEMORY_MB", "2048"))  # HAMi cap/rank
K8S_SCHEDULER = os.environ.get("CLUSTER_K8S_SCHEDULER", "kai-scheduler")
K8S_CONTEXT = os.environ.get("CLUSTER_K8S_CONTEXT", "")                   # kubectl --context
K8S_RUN_TIMEOUT_S = float(os.environ.get("CLUSTER_K8S_RUN_TIMEOUT", "1800"))
K8S_POLL_INTERVAL_S = float(os.environ.get("CLUSTER_K8S_POLL_INTERVAL", "5"))
# Distributed rendezvous: every gang gets a headless Service so ranks resolve
# each other by DNS, and rank 0's stable name is injected as MASTER_ADDR — the
# env torchrun / c10d / NCCL-over-TCP expect. Named "gang" so a pod's FQDN is
# rank-<i>.gang.<ns>.svc.cluster.local (pod hostname=rank-<i>, subdomain=gang).
K8S_RDZV_SERVICE = "gang"
K8S_MASTER_PORT = os.environ.get("CLUSTER_K8S_MASTER_PORT", "29500")
# N ranks share ONE physical GPU, so intra-device P2P is the known-broken NCCL
# transport here — disable it by default (a real multi-GPU box would not). A
# job's own env wins (it's merged after), so this is only a sane default.
K8S_DEFAULT_ENV = {"NCCL_P2P_DISABLE": "1"}
# Container-waiting reasons that will never resolve on their own — fail the gang
# fast instead of waiting out the whole run timeout on e.g. a bad image ref.
K8S_FATAL_WAIT_REASONS = {"ImagePullBackOff", "ErrImagePull", "InvalidImageName",
                          "CreateContainerConfigError", "CreateContainerError"}


def job_log_path(job_id: str) -> str:
    return os.path.join(LOGS_DIR, f"{job_id}.log")


def append_job_log(job_id: str, text: str) -> None:
    """Append a line to a job's full replay transcript (.var/logs/<job_id>.log):
    every phase transition AND every command's live output, chronological, in
    one file — the point is a single place to review or replay exactly what
    happened to one job, independent of whatever else was on screen at the time."""
    os.makedirs(LOGS_DIR, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    with open(job_log_path(job_id), "a") as f:
        f.write(text)


def run_logged(argv: list[str], job_id: str, cwd: str | None = None, check: bool = True) -> int:
    """Run a subprocess, streaming combined stdout+stderr live to THIS process's
    stdout (so a `reconcile` tick is still watchable in real time) while also
    appending every line to the job's replay log. Returns the exit code; raises
    CalledProcessError on failure unless check=False (matches subprocess.run)."""
    header = f"$ {' '.join(shlex.quote(a) for a in argv)}"
    print(header)
    append_job_log(job_id, f"{now_iso()} {header}")
    proc = subprocess.Popen(argv, cwd=cwd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        sys.stdout.write(line)
        lines.append(line)
    proc.wait()
    if lines:
        append_job_log(job_id, "".join(lines))
    if check and proc.returncode != 0:
        raise subprocess.CalledProcessError(proc.returncode, argv)
    return proc.returncode


def node_ssh_id(node: NodeRecord) -> tuple[str, str]:
    """(ssh_user, ssh_key) for reaching a node — its own credentials if set (a
    static/EC2 node uses `ubuntu` + its keypair), else the cluster-wide default."""
    return node.ssh_user or SSH_USER, node.ssh_key or SSH_KEY


def node_gpu_binds(node: NodeRecord) -> list[str]:
    """`apptainer --nv` driver binds to use when running a GPU job ON THIS node.
    node.gpu_binds (comma-sep) if set — the REMOTE node's own driver paths, which
    must exist there (a NixOS worker: /nix/store,/run/opengl-driver; an Ubuntu
    worker: empty, --nv self-detects). Falls back to the control plane's global
    CLUSTER_GPU_BINDS only for a LOCAL node (same filesystem as the control
    plane), so a remote worker never inherits the control plane's host paths."""
    if node.gpu_binds:
        return [b for b in node.gpu_binds.split(",") if b]
    if node.local:
        return [b for b in os.environ.get("CLUSTER_GPU_BINDS", "").split(",") if b]
    return []


def ssh_base(ip: str, user: str | None = None, key: str | None = None) -> list[str]:
    """argv prefix for reaching a cluster node over ssh. Module-level (not a
    LibvirtAdapter detail) because the hybrid MPI path on the host stages images
    into VMs with the exact same fabric. user/key default to the cluster fabric's;
    a static node passes its own (see node_ssh_id)."""
    return ["ssh", "-i", key or SSH_KEY, "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", f"{user or SSH_USER}@{ip}"]


def scp_to(ip: str, src: str, dst: str, job_id: str,
           user: str | None = None, key: str | None = None) -> None:
    argv = ["scp", "-i", key or SSH_KEY, "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", src, f"{user or SSH_USER}@{ip}:{dst}"]
    run_logged(argv, job_id)


def _with_preflight_probe(files: dict[str, str], probe_script: str,
                          remote_workdir: str) -> dict[str, str]:
    """Prepend the placement-probe.sh TEXT (inlined, same way GTL's own
    probe.wrap_command splices its probe before a benchmark) to every rank
    script's `exec` line, so each rank writes its own observation before
    becoming the workload — the probe runs under the exact rankfile/appfile/
    env the workload will, which is the whole point of a PREFLIGHT check.

    `set -- "{remote_workdir}"` gives the probe its $1 explicitly: the script
    itself defaults $1 to `/out`, which is right INSIDE the container (where
    output_dir is bind-mounted) but this runs BEFORE `exec` swaps the rank
    into the container — found live against real VMs, where the unset default
    silently wrote (or failed to write) under a bare, unrelated `/out` on the
    VM's own filesystem instead of the workdir collect() actually ships."""
    out = {}
    for name, content in files.items():
        if name.startswith("launcher-rank-") and "\nexec " in content:
            preamble = f'(set -- {shlex.quote(remote_workdir)}; {probe_script}) >/dev/null 2>&1\n'
            content = content.replace("\nexec ", "\n" + preamble + "exec ", 1)
        out[name] = content
    return out


def _sif_cache_name(ref: str) -> str:
    """A stable per-image SIF filename — the content digest when the ref is
    digest-pinned (the norm), else a hash of the ref string."""
    m = re.search(r"@sha256:([0-9a-f]{64})", ref)
    key = f"sha256-{m.group(1)}" if m else hashlib.sha256(ref.encode()).hexdigest()[:16]
    return f"{key}.sif"


def ensure_local_sif(image: str, job_id: str) -> str:
    """Resolve a job image to a LOCAL .sif path for MPI staging.

    Single-node runs hand `image` straight to `apptainer exec`, which consumes a
    `docker://…@digest` ref directly. MPI staging can't: it scp's the image file
    onto every rank's node, so a registry ref must be materialized into a real
    SIF on the host first. A path that's already a local file passes through.

    The SIF is built ONCE per image digest into a shared cache and reused across
    jobs — a `--runs 3` batch (three jobs) or repeated suites all share one
    build. Building per-job instead duplicated a ~5G SIF per job AND ran the
    concurrent ~26G rootfs unpacks that filled the disk. The flock serializes a
    cold-cache stampede (the batch's first jobs racing to build the same SIF):
    the winner builds, the rest block then reuse."""
    if os.path.isfile(image):
        return image
    if "://" not in image and not image.startswith(("docker-daemon:", "oci:", "sif:")):
        raise RuntimeError(
            f"job image {image!r} is neither a local file nor a pullable ref "
            "(expected a .sif path or docker://…@sha256:…)")
    os.makedirs(SIF_CACHE_DIR, exist_ok=True)
    cached = os.path.join(SIF_CACHE_DIR, _sif_cache_name(image))
    with open(cached + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not os.path.isfile(cached):
            tmp = f"{cached}.{job_id}.tmp"   # build to tmp, atomic-rename in on success
            run_logged(["apptainer", "build", "--force", tmp, image], job_id)
            os.replace(tmp, cached)
    return cached


def gpu_launch_extras() -> tuple[list[str], dict]:
    """Extra --bind paths for GPU (apptainer --nv), read from the env the nix dev
    shell exports DECLARATIVELY (CLUSTER_GPU_BINDS). Nothing is detected at runtime
    here — the shell owns the host specifics, so a non-NixOS host just exports a
    different value. Unset -> nothing extra; --nv's own driver detection suffices.

    Deliberately NOT forcing CLUSTER_GPU_LD_LIBRARY_PATH as a container --env: verified
    live that overriding it shadows a full-OS image's own glibc ahead of its ld.so
    search path (nvidia/cuda:*-ubuntu* segfaulted/missing-lib'd on nvidia-smi with
    it set, ran clean without it) — --nv already wires the driver's own userspace
    libs correctly for a normal OCI image. Only a from-scratch/minimal container
    with no libc of its own would need it; that's not a case this cluster runs
    today, so there's nothing to wire it through for."""
    binds = [b for b in os.environ.get("CLUSTER_GPU_BINDS", "").split(",") if b]
    return binds, {}


def container_argv(spec: JobSpec, bind_src: str, bind_dst: str,
                   extra_binds: list[str] | None = None,
                   extra_env: dict | None = None) -> list[str]:
    """Build the argv that runs `spec.image` with its command, binding a host
    dir to the job's output_dir. Pure function — the single place that knows how
    apptainer/docker are invoked. `image` may be a SIF path or a docker[://] ref
    (apptainer runs both). extra_binds/extra_env layer on GPU driver paths."""
    extra_binds = extra_binds or []
    env = {**spec.env, **(extra_env or {})}
    if spec.runtime == "docker":
        argv = ["docker", "run", "--rm"]
        if spec.gpu:
            argv += ["--gpus", "all"]
        argv += ["-v", f"{bind_src}:{bind_dst}"]
        for b in extra_binds:
            argv += ["-v", f"{b}:{b}"]
        for k, v in env.items():
            argv += ["-e", f"{k}={v}"]
        return argv + [spec.image] + list(spec.command)
    # default: apptainer
    argv = ["apptainer", "exec"]
    if spec.gpu:
        argv += ["--nv"]                       # bind host NVIDIA driver/devices into the container
    argv += ["--bind", f"{bind_src}:{bind_dst}"]
    for b in extra_binds:
        argv += ["--bind", b]
    for k, v in env.items():
        argv += ["--env", f"{k}={v}"]
    return argv + [spec.image] + list(spec.command)


def fabric_if_include(nodes: list[NodeRecord]) -> tuple[str, str]:
    """Return `(btl_if_include, oob_if_include)` — the MCA values pinning
    OpenMPI's two TCP planes to the cluster0 fabric. They DIFFER, and both forms
    are hard-won hybrid-launch lessons (mpirun runs on the host, whose cluster0
    address is a bridge, virbr-cluster, sharing the box with a zoo of other
    virtual interfaces: docker0, br-*, veth*, tailscale0, virbr0, plus the real
    wifi/LAN nic):

    - OOB (orted control channel) -> a list of each node's EXACT `/32`, e.g.
      `192.168.71.1/32,192.168.71.11/32`. Given the `/24` subnet, OpenMPI 4.1.x's
      OOB matcher mis-resolves among the interface zoo and rejects the bridge
      outright ("None of the TCP networks ... could be found"); an exact /32
      binds cleanly.
    - BTL (rank<->rank data) -> the `/24` subnet. The inverse bug: OpenMPI 4.1.x's
      BTL `/32` matching is broken — it "matches" EVERY interface and ends up
      advertising the host's wifi/docker addresses (192.168.18.x, 172.x) that a
      VM can't reach, so the first MPI collective hangs forever. The `/24` matches
      only the two real cluster addresses (host .1, VM .1x).

    Both are GLOBAL to the mpirun and forwarded to every orted, so both must be
    valid on every node: each node matches its own /32 in the OOB list (and
    ignores the rest — OpenMPI tolerates non-matching list entries), and every
    node's cluster address is inside the /24."""
    oob = ",".join(f"{n.ip}/32" for n in nodes)
    btl = ".".join(nodes[0].ip.split(".")[:3]) + ".0/24"
    return btl, oob


def mpirun_argv(np: int, hostfile: str, btl_if_include: str, oob_if_include: str,
                per_rank_argv: list[str], rsh_agent: str | None = None,
                no_tree_spawn: bool = False) -> list[str]:
    """Wrap a per-rank container argv (from container_argv) in mpirun, one rank
    per claimed node (--map-by node) over the cluster's own TCP fabric — the MCA
    if_include options (see fabric_if_include) keep mpirun off any other interface.

    rsh_agent: how mpirun reaches remote nodes. On a head VM the default plain
    `ssh` works (the fabric key + ~/.ssh/config are staged on every VM by the
    bootstrap role); when mpirun runs on the HOST (hybrid jobs) it must be told
    to use the cluster key + user explicitly. no_tree_spawn forces every remote
    launch to originate from the head — OpenMPI's default tree spawn would make
    one VM launch its sibling re-using the same agent string, whose key path
    only exists on the host."""
    argv = [
        "mpirun", "-np", str(np), "--hostfile", hostfile, "--map-by", "node",
        "--mca", "btl", "tcp,self",
        "--mca", "btl_tcp_if_include", btl_if_include,
        "--mca", "oob_tcp_if_include", oob_if_include,
    ]
    if rsh_agent:
        argv += ["--mca", "plm_rsh_agent", rsh_agent]
    if no_tree_spawn:
        argv += ["--mca", "plm_rsh_no_tree_spawn", "1"]
    return argv + per_rank_argv


def write_appfile(path: str, rank_lines: list[tuple[str, list[str]]]) -> None:
    """Write an OpenMPI appfile: one `-np 1 --host <ip> <argv...>` line per rank.
    Unlike a single shared command, an appfile lets each rank run a DIFFERENT
    per-rank argv — which is exactly what a hybrid job needs: rank 0 (the GPU
    host) gets `apptainer exec --nv` plus the NixOS driver binds, while the VM
    ranks get a plain launch (no --nv warning on a GPU-less guest, and none of
    the host-only `/nix/store` / `/run/opengl-driver` binds whose source paths
    don't exist inside a VM — a missing bind source is a hard apptainer error).

    OpenMPI's appfile parser splits each line on whitespace and does NOT honor
    shell quoting, so every token must be whitespace-free. That holds for our
    argv: apptainer paths live under /tmp/cluster/<job_id> and env is passed as
    `--env KEY=VAL` with space-free values."""
    with open(path, "w") as f:
        for ip, argv in rank_lines:
            f.write(f"-np 1 --host {ip} " + " ".join(argv) + "\n")


def mpirun_appfile_argv(appfile: str, btl_if_include: str, oob_if_include: str,
                        rsh_agent: str | None = None,
                        no_tree_spawn: bool = False) -> list[str]:
    """mpirun driving a per-rank appfile instead of one shared command. Global
    options (fabric pinning via if_include — see fabric_if_include, the rsh agent,
    tree-spawn) stay on the command line; the per-rank command AND host placement
    come from the appfile lines, so no separate --hostfile / -np / --map-by is
    passed here. See write_appfile and mpirun_argv for the rest of the rationale."""
    argv = [
        "mpirun",
        "--mca", "btl", "tcp,self",
        "--mca", "btl_tcp_if_include", btl_if_include,
        "--mca", "oob_tcp_if_include", oob_if_include,
    ]
    if rsh_agent:
        argv += ["--mca", "plm_rsh_agent", rsh_agent]
    if no_tree_spawn:
        argv += ["--mca", "plm_rsh_no_tree_spawn", "1"]
    return argv + ["--app", appfile]


class ProviderAdapter(ABC):
    name: str = "abstract"

    # node lifecycle (job_id: whose phase this is, for the replay log — a node
    # can outlive any one job, but each provision/bootstrap/deprovision call
    # happens on behalf of exactly one)
    @abstractmethod
    def provision(self, node: NodeRecord, job_id: str) -> None: ...
    @abstractmethod
    def deprovision(self, node: NodeRecord, job_id: str) -> None: ...
    @abstractmethod
    def bootstrap(self, nodes: list[NodeRecord], job_id: str) -> None: ...

    # workload lifecycle (head node = nodes[0]). `plan`: the resolved LaunchPlan
    # dict from reconciler.py's _phase_plan, or None for a job with no
    # placement request (the pre-existing case) — only LibvirtAdapter's
    # launcher: mpi path acts on it; every other adapter ignores it.
    @abstractmethod
    def run(self, nodes: list[NodeRecord], job_id: str, spec: JobSpec,
           plan: dict | None = None) -> RunResult: ...
    @abstractmethod
    def collect(self, nodes: list[NodeRecord], job_id: str, spec: JobSpec, dest: str) -> str: ...

    # placement (reconciler.py's _phase_plan; only a launcher: mpi job carrying
    # a launch block ever calls this). NOT abstract: an adapter that never sees
    # one (k8s, and most callers today) inherits this default rather than every
    # adapter needing a stub. The default raises rather than fabricating a
    # topology — a wrong affinity silently binds a rank to the far socket.
    def probe_topology(self, node: NodeRecord, job_id: str) -> dict:
        raise NotImplementedError(f"{self.name} adapter cannot probe topology")


class NullAdapter(ProviderAdapter):
    """Logs intents, touches nothing. Default for dry-run / skeleton runs."""
    name = "null"

    def __init__(self, log=_log_stderr):
        self.log = log

    def provision(self, node, job_id):    self.log(f"[null] provision {node.name} (no-op)")
    def deprovision(self, node, job_id):  self.log(f"[null] deprovision {node.name} (no-op)")
    def bootstrap(self, nodes, job_id):   self.log(f"[null] bootstrap {[n.name for n in nodes]} (no-op)")

    def run(self, nodes, job_id, spec, plan=None):
        self.log(f"[null] would run '{spec.image or '(no image)'}' on {nodes[0].name}")
        return RunResult(job_id, nodes[0].name, None, note="dry-run (null adapter)")

    def collect(self, nodes, job_id, spec, dest):
        self.log(f"[null] would collect -> {dest}")
        os.makedirs(dest, exist_ok=True)
        return dest

    def probe_topology(self, node, job_id):
        # A synthetic single-GPU node, deterministic per node name (not
        # random) — lets a launch block resolve and preview end-to-end in
        # dry-run without real hardware, but it's a MADE-UP machine: never
        # runs a real workload, so nothing downstream depends on it being
        # accurate the way a real probe's output must be.
        self.log(f"[null] would probe topology on {node.name} (synthetic 4-core/1-GPU)")
        return {
            "probe_version": "1", "hostname": node.name, "scope": "host",
            "allowed_cpus": "0-3", "online_cpus": "0-3",
            "cpus": [{"id": i, "core": i, "socket": 0, "numa": 0} for i in range(4)],
            "numa": [{"id": 0, "cpulist": "0-3", "memory_mib": 65536}],
            "gpus": [{"index": 0, "uuid": f"GPU-null-{node.name}", "pci_bus_id": "0000:00:00.0",
                     "memory_mib": 16384, "name": "null-adapter synthetic GPU", "numa": 0}],
            "topo_matrix": "\tGPU0\tCPU Affinity\tNUMA Affinity\nGPU0\t X \t0-3\t0\n",
            "launcher": {"type": "openmpi", "version": "0.0.0-null"},
        }


class LocalHostAdapter(ProviderAdapter):
    """The control-plane host as a GPU node. The host already exists, so
    provision/deprovision are no-ops; the container runs in-process with --nv."""
    name = "localhost"

    def __init__(self, log=_log_stderr):
        self.log = log

    def provision(self, node, job_id):
        self.log(f"[localhost] {node.name} already up — no provisioning")

    def deprovision(self, node, job_id):
        self.log(f"[localhost] {node.name} left running (host is not destroyed)")

    def bootstrap(self, nodes, job_id):
        # Just verify the runtimes exist; the host is already configured.
        if shutil.which("apptainer") is None:
            self.log("[localhost] WARNING: apptainer not found on host "
                     "(install it before running real jobs — nix dev shell, or on "
                     "Ubuntu the pinned .deb, see docs/07-ubuntu-setup.md)")
        else:
            self.log("[localhost] apptainer present")
        # mpirun only matters if this host becomes a hybrid job's MPI head.
        mpirun = shutil.which("mpirun")
        if mpirun:
            out = subprocess.run([mpirun, "--version"], capture_output=True, text=True).stdout
            self.log(f"[localhost] mpirun present ({out.splitlines()[0].strip() if out else 'version unknown'})")
        else:
            self.log("[localhost] WARNING: mpirun not found on host — hybrid MPI jobs "
                     "will fail (on Ubuntu: apt install openmpi-bin, matching the VMs' 4.1.x)")

    def run(self, nodes, job_id, spec, plan=None):
        # `plan` is ignored here: this adapter's launcher: mpi path is the
        # HYBRID one (host GPU + VMs, one appfile), a different shape from
        # the plain multi-VM case LibvirtAdapter's planned path targets.
        node = nodes[0]
        workdir = os.path.join(RUNS_DIR, job_id)
        os.makedirs(workdir, exist_ok=True)
        if spec.is_dry_run:
            return RunResult(job_id, node.name, None, workdir, note="no image -> dry-run")
        if spec.launcher == "mpi" and len(nodes) > 1:
            return self._run_mpi(nodes, job_id, spec)
        # GPU jobs get the driver bind/env the nix shell exported declaratively.
        extra_binds, extra_env = gpu_launch_extras() if spec.gpu else ([], {})
        argv = container_argv(spec, workdir, spec.output_dir, extra_binds, extra_env)
        stdout_path = os.path.join(workdir, "stdout.log")
        header = f"$ {' '.join(shlex.quote(a) for a in argv)}"
        self.log(f"[localhost] {header}")
        append_job_log(job_id, f"{now_iso()} {header}")
        with open(stdout_path, "w") as out:
            rc = subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT).returncode
        # workdir/stdout.log is the job's egress artifact (collect() ships it to
        # the drop-zone); mirror it into the replay log too for a one-stop view.
        with open(stdout_path) as f:
            append_job_log(job_id, f.read())
        return RunResult(job_id, node.name, rc, workdir, stdout_path, f"apptainer exit={rc}")

    def _run_mpi(self, nodes: list[NodeRecord], job_id: str, spec: JobSpec) -> RunResult:
        """Hybrid multi-node launch: mpirun runs ON THE HOST (nodes[0], the GPU
        node) and spans it plus the job's VM nodes in one launch. The host's
        appfile entry is its cluster0 bridge address — a local interface, so
        mpirun forks rank 0 in place and only the VM ranks go over ssh (no sshd
        needed on the host). VM ranks talk TCP back to the host on the bridge.
        The workdir is the SAME absolute path on every node (the image + bind
        source must exist everywhere), and the VM `cluster` user can't mkdir
        under the operator's $HOME — /tmp is the one place both sides can. (If
        mpirun ever fails to detect the bridge address as local, swap the host's
        appfile `--host` value for `localhost` — the btl/oob if_include MCA
        params, not the host string, decide which endpoints rank 0 advertises.)

        Unlike the VM-headed path (LibvirtAdapter._run_mpi, one shared per-rank
        command), this uses a per-rank APPFILE so the ranks can differ: rank 0
        (the GPU host) gets `--nv` + the NixOS driver binds the dev shell
        exported, while the VM ranks get a plain launch. A shared command can't
        express that — --nv's driver binds don't exist on the guests, and a
        missing bind source is a hard apptainer error. See write_appfile."""
        head = nodes[0]
        if shutil.which("mpirun") is None:
            raise RuntimeError(
                "mpirun not found on host — hybrid MPI needs the host's OpenMPI to "
                "match the VMs' 4.1.x (nix dev shell pins it via flake.nix; on "
                "Ubuntu: apt install openmpi-bin)")
        workdir = f"/tmp/cluster/{job_id}"
        os.makedirs(workdir, exist_ok=True)
        # The image must sit at the SAME absolute path on the host and every VM
        # (the appfile references one path for all ranks). Materialize a
        # docker://…@digest ref into a SIF at that shared path; a local .sif is
        # copied there. Either way `image` is workdir/image.sif everywhere.
        image = f"{workdir}/image.sif"
        src = ensure_local_sif(spec.image, job_id)
        if os.path.abspath(src) != os.path.abspath(image):
            shutil.copy2(src, image)
        appfile = f"{workdir}/appfile"
        for n in nodes[1:]:
            run_logged(ssh_base(n.ip) + [f"mkdir -p {workdir}"], job_id)
            scp_to(n.ip, image, image, job_id)

        # rank 0 (host): --nv + the declaratively-exported GPU binds (on NixOS,
        # /nix/store + /run/opengl-driver so the driver's userspace resolves).
        rank_spec = dataclasses.replace(spec, image=image)
        gpu_binds, _ = gpu_launch_extras()
        host_argv = container_argv(dataclasses.replace(rank_spec, gpu=True),
                                   workdir, spec.output_dir, extra_binds=gpu_binds)
        # VM ranks: no GPU on the guest -> plain launch (no --nv, no host binds).
        vm_argv = container_argv(dataclasses.replace(rank_spec, gpu=False),
                                 workdir, spec.output_dir)
        rank_lines = [(head.ip, host_argv)] + [(n.ip, vm_argv) for n in nodes[1:]]
        write_appfile(appfile, rank_lines)

        btl_inc, oob_inc = fabric_if_include(nodes)
        # OpenMPI whitespace-splits the agent string into argv, so this works as
        # long as SSH_KEY contains no spaces (the default path doesn't).
        agent = (f"ssh -i {SSH_KEY} -o IdentitiesOnly=yes -o StrictHostKeyChecking=no "
                 f"-o UserKnownHostsFile=/dev/null -l {SSH_USER}")
        argv = mpirun_appfile_argv(appfile, btl_inc, oob_inc, rsh_agent=agent, no_tree_spawn=True)
        stdout_path = f"{workdir}/stdout.log"
        header = f"$ {' '.join(shlex.quote(a) for a in argv)}"
        self.log(f"[localhost] (mpirun head, {len(nodes)} ranks, per-rank appfile) {header}")
        append_job_log(job_id, f"{now_iso()} {header}")
        with open(stdout_path, "w") as out:
            rc = subprocess.run(argv, stdout=out, stderr=subprocess.STDOUT).returncode
        with open(stdout_path) as f:
            append_job_log(job_id, f.read())
        return RunResult(job_id, head.name, rc, workdir, stdout_path,
                         note=f"hybrid mpirun np={len(nodes)} (appfile) exit={rc}")

    def collect(self, nodes, job_id, spec, dest):
        # Hybrid MPI runs use the shared /tmp workdir (same path as the VM
        # ranks — see _run_mpi); single-node host runs keep the .var/runs scratch.
        if spec.launcher == "mpi" and len(nodes) > 1:
            workdir = f"/tmp/cluster/{job_id}"
        else:
            workdir = os.path.join(RUNS_DIR, job_id)
        os.makedirs(dest, exist_ok=True)
        if os.path.isdir(workdir):
            for name in os.listdir(workdir):
                if _is_staging_artifact(name):
                    continue
                src = os.path.join(workdir, name)
                dst = os.path.join(dest, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        return dest

    def probe_topology(self, node, job_id):
        proc = subprocess.run(["sh", "-c", PROBE_PATH.read_text()],
                              capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"topology probe failed on {node.name}: {proc.stderr.strip()}")
        return json.loads(proc.stdout)


class LibvirtAdapter(ProviderAdapter):
    """VM nodes: virsh define/start for lifecycle, ssh + apptainer for workloads."""
    name = "libvirt"

    # gen-inventory.sh regenerates ONE shared file (infra/ansible/inventory/hosts.ini)
    # from whatever's currently live cluster-wide. With the reconciler now advancing
    # jobs concurrently (see reconciler.tick), two jobs' bootstrap() calls can land
    # in the same instant; without this lock their gen-inventory.sh invocations could
    # interleave writes to that shared file. Only the (fast) regenerate+read window
    # is serialized — the actual ansible-playbook run below is NOT held under it, so
    # concurrent jobs still bootstrap in parallel, just never with a torn inventory.
    _inventory_lock = threading.Lock()

    def __init__(self, log=_log_stderr):
        self.log = log

    def _ssh_base(self, node: NodeRecord) -> list[str]:
        return ssh_base(node.ip, *node_ssh_id(node))

    def provision(self, node, job_id):
        self.log(f"[libvirt] provisioning {node.name}")
        run_logged([os.path.join(LIBVIRT_DIR, "vm-define.sh"), str(node.index)], job_id)

    def deprovision(self, node, job_id):
        self.log(f"[libvirt] deprovisioning {node.name}")
        run_logged([os.path.join(LIBVIRT_DIR, "vm-destroy.sh"), str(node.index)], job_id)

    def bootstrap(self, nodes, job_id):
        # Inventory is regenerated from live libvirt state, then the one role runs
        # scoped to THIS job's nodes only (--limit) — the regenerated inventory
        # lists every currently-running VM cluster-wide (gen-inventory.sh doesn't
        # know about jobs), so without --limit this job's ansible-playbook run
        # would also touch every OTHER job's nodes that happen to be up right now,
        # including ones mid-boot for a concurrently-provisioning job (a real
        # failure seen live: job A's bootstrap failed because job B's freshly
        # -created VMs weren't SSH-ready yet, even though A never touched them).
        #
        # cwd=ANSIBLE_DIR matters too: ansible only auto-discovers ansible.cfg
        # (which sets StrictHostKeyChecking=no / UserKnownHostsFile=/dev/null for
        # the cluster0 fabric) relative to the CURRENT DIRECTORY, not the playbook's
        # path. Without it, connections fall back to ambient SSH defaults and
        # intermittently fail host-key verification with no prompt to answer it.
        self.log(f"[libvirt] bootstrapping {[n.name for n in nodes]}")
        shared = os.path.join(ANSIBLE_DIR, "inventory", "hosts.ini")
        job_inventory = os.path.join(ANSIBLE_DIR, "inventory", f"hosts-{job_id}.ini")
        want = {n.name for n in nodes}
        # gen-inventory.sh rewrites ONE shared hosts.ini listing every running
        # domain cluster-wide, and it only lists domains virsh reports 'running'.
        # Under concurrent jobs that shared file is a race on two axes:
        #   1) a freshly-provisioned VM can miss the 'running' window (domstate
        #      lags the start) -> this job's node is absent from the file;
        #   2) another job's gen-inventory rewrite can be read half-written, so
        #      ansible parses the host line but not the trailing [nodes:vars]
        #      (losing ansible_user=cluster + the key path) and falls back to the
        #      host user -> "Permission denied (publickey)" UNREACHABLE (exit 4).
        # Both silently corrupted runs on 2026-07-21. Fix: under the lock,
        # regenerate until every requested node is present, then snapshot the file
        # to a PER-JOB inventory ansible reads alone — no shared mutable state for
        # a concurrent writer to tear. Fail closed if the nodes never appear.
        deadline = time.monotonic() + 120
        while True:
            with self._inventory_lock:
                run_logged([os.path.join(LIBVIRT_DIR, "gen-inventory.sh")], job_id)
                ready = want <= self._inventory_hosts(shared)
                if ready:
                    shutil.copy2(shared, job_inventory)  # atomic snapshot, still locked
            if ready:
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(
                    f"bootstrap aborted: {sorted(want - self._inventory_hosts(shared))} "
                    f"not in the ansible inventory after 120s (domain not 'running'?) "
                    f"— refusing to run the play on zero hosts and mark it bootstrapped")
            time.sleep(5)
        limit = ",".join(n.name for n in nodes)
        try:
            run_logged(["ansible-playbook", "-i", job_inventory,
                       "--limit", limit, "site.yml"], job_id, cwd=ANSIBLE_DIR)
        finally:
            try:
                os.remove(job_inventory)
            except OSError:
                pass

    @staticmethod
    def _inventory_hosts(inventory: str) -> set[str]:
        """Host names in the generated inventory's [nodes] group (each line is
        `<name> ansible_host=<ip>`; skip comments, blanks, and [section]/vars)."""
        hosts: set[str] = set()
        for line in Path(inventory).read_text().splitlines():
            line = line.strip()
            if not line or line.startswith(("#", "[")) or "ansible_host=" not in line:
                continue
            hosts.add(line.split()[0])
        return hosts

    def run(self, nodes, job_id, spec, plan=None):
        head = nodes[0]
        remote_workdir = f"/tmp/cluster/{job_id}"
        if spec.is_dry_run:
            return RunResult(job_id, head.name, None, remote_workdir, note="no image -> dry-run")
        if spec.launcher == "mpi" and len(nodes) > 1:
            return self._run_mpi(nodes, job_id, spec, remote_workdir, plan)
        # A GPU job on a remote node adds the node's OWN driver binds for
        # `apptainer --nv` (node.gpu_binds), whose source paths must exist ON
        # THAT node — a NixOS worker needs /nix/store,/run/opengl-driver; an
        # Ubuntu worker sets none (--nv self-detects). This is the remote
        # counterpart of the host's CLUSTER_GPU_BINDS (LocalHostAdapter), except
        # it's per-node because it's the REMOTE box's filesystem, not ours.
        extra_binds = node_gpu_binds(head) if spec.gpu else []
        argv = container_argv(spec, remote_workdir, spec.output_dir, extra_binds=extra_binds)
        inner = " ".join(shlex.quote(a) for a in argv)
        stdout_log = f"{remote_workdir}/stdout.log"
        # tee into the workdir (not just captured by run_logged's replay log):
        # collect() scp's remote_workdir wholesale, so this is what makes stdout
        # land in the drop-zone per the documented egress contract (stdout.log +
        # output_dir) — the same place a caller already looks for a job's
        # output_dir files, so container stdout (for tools that print results
        # rather than only writing files) shows up there too.
        # PIPESTATUS preserves the container's real exit code through the pipe
        # (the cluster user's shell is bash — see cloud-init/user-data.tpl).
        remote = (f"mkdir -p {remote_workdir} && ({inner}) 2>&1 | tee {stdout_log}; "
                  f"exit ${{PIPESTATUS[0]}}")
        self.log(f"[libvirt] ssh {head.name}: {inner}")
        rc = run_logged(self._ssh_base(head) + [remote], job_id, check=False)
        return RunResult(job_id, head.name, rc, remote_workdir, stdout_log, note=f"remote exit={rc}")

    def _scp_to(self, node: NodeRecord, src: str, dst: str, job_id: str) -> None:
        scp_to(node.ip, src, dst, job_id, *node_ssh_id(node))

    def _run_mpi(self, nodes: list[NodeRecord], job_id: str, spec: JobSpec,
                 remote_workdir: str, plan: dict | None = None) -> RunResult:
        """Multi-node launch: mpirun runs on the head VM (installed by the
        bootstrap role) and execs `apptainer exec <image> ...` per rank on
        every claimed node over ssh — the container never needs its own MPI
        launcher, only a matching libmpi. Image is per-job (built from the
        nodes THIS job claimed, not the cluster-wide 8) and staged fresh onto
        every claimed node under remote_workdir.

        `plan`: a resolved LaunchPlan dict (reconciler.py's _phase_plan) routes
        to _run_mpi_planned — per-rank rankfile/argv/env compiled by
        launcher.for_plan(), instead of the one shared command below. None
        (no placement request) keeps this EXACT path, unchanged."""
        head = nodes[0]
        local_image = ensure_local_sif(spec.image, job_id)
        remote_image = f"{remote_workdir}/{os.path.basename(local_image)}"
        for n in nodes:
            run_logged(self._ssh_base(n) + [f"mkdir -p {remote_workdir}"], job_id)
            self._scp_to(n, local_image, remote_image, job_id)

        if plan is not None:
            return self._run_mpi_planned(nodes, job_id, spec, remote_workdir, remote_image, plan)

        remote_hostfile = f"{remote_workdir}/hostfile"
        btl_inc, oob_inc = fabric_if_include(nodes)
        local_scratch = os.path.join(RUNS_DIR, job_id)
        os.makedirs(local_scratch, exist_ok=True)
        local_hostfile = os.path.join(local_scratch, "hostfile")
        with open(local_hostfile, "w") as f:
            for n in nodes:
                f.write(f"{n.ip} slots=1\n")
        self._scp_to(head, local_hostfile, remote_hostfile, job_id)

        rank_spec = dataclasses.replace(spec, image=remote_image)
        per_rank = container_argv(rank_spec, remote_workdir, spec.output_dir)
        argv = mpirun_argv(len(nodes), remote_hostfile, btl_inc, oob_inc, per_rank)
        inner = " ".join(shlex.quote(a) for a in argv)
        stdout_log = f"{remote_workdir}/stdout.log"
        # Same reasoning as the single-node path: tee mpirun's combined output
        # (all ranks, since mpirun forwards each rank's stdout to the launching
        # process) into the head's workdir so collect() ships it to the drop-zone.
        remote = f"({inner}) 2>&1 | tee {stdout_log}; exit ${{PIPESTATUS[0]}}"
        self.log(f"[libvirt] ssh {head.name} (mpirun head, {len(nodes)} ranks): {inner}")
        rc = run_logged(self._ssh_base(head) + [remote], job_id, check=False)
        return RunResult(job_id, head.name, rc, remote_workdir, stdout_log,
                         note=f"mpirun np={len(nodes)} exit={rc}")

    def _run_mpi_planned(self, nodes: list[NodeRecord], job_id: str, spec: JobSpec,
                         remote_workdir: str, remote_image: str, plan: dict) -> RunResult:
        """The launcher.for_plan() path: per-rank rankfile/appfile/env compiled
        from the resolved plan, `--report-bindings` captured for the receipt,
        and (when the intent asks for it) a per-rank preflight probe run
        before the workload so the receipt has a real OS-level observation,
        not just the launcher's own claim."""
        from . import launcher as launcher_mod
        from .placement import LaunchPlan
        from .receipt import PLACEMENT_PROBE_PATH, build_receipt

        head = nodes[0]
        launch_plan = LaunchPlan.from_dict(plan)
        node_ips = {n.name: n.ip for n in nodes}
        launcher_adapter = launcher_mod.for_plan(launch_plan)
        # Checked BEFORE compiling: a capability gap should read as itself
        # ("this launcher cannot bind ranks to CPUs"), not as whatever a
        # half-rendered command does next.
        unmet = launcher_adapter.unsupported(launch_plan, spec.launch or {})
        if unmet:
            raise launcher_mod.LauncherUnsupported("; ".join(unmet))
        rendering = launcher_adapter.compile(launch_plan, spec, node_ips=node_ips,
                                             workdir=remote_workdir, image=remote_image)

        require_preflight = bool(((spec.launch or {}).get("validation") or {})
                                 .get("require_preflight"))
        files = rendering.files
        if require_preflight:
            files = _with_preflight_probe(files, PLACEMENT_PROBE_PATH.read_text(), remote_workdir)

        # mpirun (on the head) reads the appfile/rankfile itself, so those two
        # only need to exist THERE. A per-rank script is `/bin/sh`'d via ssh
        # onto the rank's OWN node (the appfile's `--host`) — staging it only
        # on head left every non-head rank unable to find its own script,
        # found live against 2 real VMs: rank 1 on cluster-node-02 failed
        # with "cannot open .../launcher-rank-1.sh: No such file".
        rank_node = {r.global_rank: r.node for r in launch_plan.ranks}
        node_by_name = {n.name: n for n in nodes}
        local_scratch = os.path.join(RUNS_DIR, job_id)
        os.makedirs(local_scratch, exist_ok=True)
        for name, content in files.items():
            local_path = os.path.join(local_scratch, name)
            with open(local_path, "w") as f:
                f.write(content)
            m = re.match(r"launcher-rank-(\d+)\.sh$", name)
            target = node_by_name[rank_node[int(m.group(1))]] if m else head
            self._scp_to(target, local_path, f"{remote_workdir}/{name}", job_id)

        inner = " ".join(shlex.quote(a) for a in rendering.argv)
        stdout_log = f"{remote_workdir}/stdout.log"
        remote = f"({inner}) 2>&1 | tee {stdout_log}; exit ${{PIPESTATUS[0]}}"
        self.log(f"[libvirt] ssh {head.name} (mpirun head, planned, {len(launch_plan.ranks)} "
                f"rank(s)): {inner}")
        rc = run_logged(self._ssh_base(head) + [remote], job_id, check=False)

        # --report-bindings' output is interleaved into the same stream that
        # went to stdout_log; read it back rather than threading a second
        # capture path through run_logged (which only returns an exit code).
        combined = subprocess.run(self._ssh_base(head) + [f"cat {stdout_log}"],
                                  capture_output=True, text=True).stdout
        observations = (self._fetch_preflight_observations(nodes, remote_workdir)
                        if require_preflight else [])
        receipt = build_receipt(launch_plan, observations, binding_report=combined,
                                preflight_ran=require_preflight)
        self._stage_launch_artifacts(head, job_id, remote_workdir, spec.launch, plan, receipt)

        return RunResult(job_id, head.name, rc, remote_workdir, stdout_log,
                         note=f"mpirun np={len(launch_plan.ranks)} (planned) exit={rc}")

    def _fetch_preflight_observations(self, nodes: list[NodeRecord],
                                      remote_workdir: str) -> list[dict]:
        """Every rank writes its own preflight/rank-<N>.json on WHICHEVER node
        it actually landed on, not necessarily the head — collected here (not
        by collect(), which only pulls from head) since receipt.build_receipt
        needs them immediately, before teardown. A node with no such rank, or
        no preflight directory at all, contributes nothing (not an error)."""
        observations: list[dict] = []
        for n in nodes:
            proc = subprocess.run(
                self._ssh_base(n) +
                [f"for f in {remote_workdir}/preflight/*.json; do "
                 f"[ -f \"$f\" ] && echo ===RANK=== && cat \"$f\"; done 2>/dev/null"],
                capture_output=True, text=True)
            for chunk in proc.stdout.split("===RANK==="):
                chunk = chunk.strip()
                if not chunk:
                    continue
                try:
                    observations.append(json.loads(chunk))
                except json.JSONDecodeError:
                    continue
        return observations

    def _stage_launch_artifacts(self, head: NodeRecord, job_id: str, remote_workdir: str,
                                intent: dict | None, plan: dict, receipt) -> None:
        """Write launch-{intent,plan,receipt}.yaml into the SAME remote workdir
        collect() already scp's wholesale — the exact three filenames the
        child-cluster interface contract (and GTL's childcluster.py reader)
        expect beside stdout.log, with no change needed to collect() itself."""
        import yaml
        local_scratch = os.path.join(RUNS_DIR, job_id)
        artifacts = {"launch-intent.yaml": intent, "launch-plan.yaml": plan,
                    "launch-receipt.yaml": receipt.to_dict()}
        for name, doc in artifacts.items():
            local_path = os.path.join(local_scratch, name)
            with open(local_path, "w") as f:
                yaml.safe_dump(doc, f, sort_keys=False)
            self._scp_to(head, local_path, f"{remote_workdir}/{name}", job_id)

    def collect(self, nodes, job_id, spec, dest):
        head = nodes[0]
        remote_workdir = f"/tmp/cluster/{job_id}"
        os.makedirs(dest, exist_ok=True)
        # Drop the staged image (a ~5 GB SIF) + hostfile on the head BEFORE the
        # recursive scp — see _is_staging_artifact. scp -r has no exclude, and
        # the remote /tmp workdir is discarded when the VM is torn down, so
        # removing them here just keeps them out of the drop-zone (they filled
        # /home to 100% otherwise). The image is cached + regenerable.
        run_logged(self._ssh_base(head) +
                   [f"rm -f {remote_workdir}/*.sif {remote_workdir}/hostfile"],
                   job_id, check=False)
        user, key = node_ssh_id(head)
        run_logged(["scp", "-i", key, "-o", "StrictHostKeyChecking=no",
                   "-o", "UserKnownHostsFile=/dev/null", "-r",
                   f"{user}@{head.ip}:{remote_workdir}/.", dest], job_id, check=False)
        return dest

    def probe_topology(self, node, job_id):
        # Same ssh reachability provision()/bootstrap() already proved for
        # this node — no new credential or connectivity path. Piped via stdin
        # (`sh -s`) rather than staged as a file: read-only, and the script is
        # small enough that a round trip to write+chmod+run+clean up a remote
        # file would be pure overhead. StaticSshAdapter inherits this as-is.
        proc = subprocess.run(self._ssh_base(node) + ["sh", "-s"],
                              input=PROBE_PATH.read_text(), capture_output=True, text=True)
        if proc.returncode != 0:
            raise RuntimeError(f"topology probe failed on {node.name}: {proc.stderr.strip()}")
        return json.loads(proc.stdout)


class StaticSshAdapter(LibvirtAdapter):
    """A pre-provisioned, already-running SSH host joined as a plain CPU worker
    (e.g. an EC2 instance). It is reached over ssh + apptainer exactly like a
    libvirt VM, so run()/collect() (and the per-node ssh identity now honored by
    _ssh_base/_scp_to) are inherited UNCHANGED — the whole point is that the
    workload path was never libvirt-specific.

    What differs is lifecycle: the instance's existence is not ours to manage.
    provision/deprovision are no-ops (like LocalHostAdapter for the host — we
    neither boot nor destroy it), and bootstrap can't use the libvirt-derived
    inventory (gen-inventory.sh only knows about virsh domains). Following the
    golden rule that the cluster installs nothing on its own, bootstrap only
    VERIFIES the node is reachable and already has apptainer, failing closed if
    not — bake apptainer into the AMI (the pinned .deb, docs/07-ubuntu-setup.md)
    before joining the node, rather than having a job die later in run()."""
    name = "static-ssh"

    def provision(self, node, job_id):
        self.log(f"[static-ssh] {node.name} already up ({node.ip}) — no provisioning")

    def deprovision(self, node, job_id):
        self.log(f"[static-ssh] {node.name} left running (not ours to destroy)")

    def bootstrap(self, nodes, job_id):
        for n in nodes:
            self.log(f"[static-ssh] verifying {n.name} ({n.ip}): reachable + apptainer present")
            rc = run_logged(self._ssh_base(n) + ["command -v apptainer"], job_id, check=False)
            if rc != 0:
                raise RuntimeError(
                    f"static-ssh node {n.name} ({n.ip}): unreachable over ssh, or no "
                    "apptainer on PATH. Confirm the security group allows ssh from this "
                    "host and the cluster key is authorized, and bake apptainer into the "
                    "image (pinned .deb, docs/07-ubuntu-setup.md) before joining it.")


class KubernetesAdapter(ProviderAdapter):
    """A whole Kubernetes cluster (KAI scheduler + HAMi-core isolation) as a
    backend. Unlike every other adapter, a k8s registry "node" is a SLOT, not a
    machine: the cluster schedules its own pods, so the registry models the
    cluster as a small pool of interchangeable slots and each job claims one.
    The N-way parallelism a job asks for (spec.node_count) is realized INSIDE
    run() as a GANG of N pods sharing the one physical GPU — KAI co-schedules
    them (an explicit PodGroup, minMember=N) and HAMi-core hard-caps each rank's
    VRAM via the `gpu-memory` annotation. This is the "multi-GPU on one GPU"
    simulator; faithful for topology/scheduling/init, not for throughput (the
    ranks time-slice one SM array). See docs/08-kubernetes-backend.md.

    Nothing here installs anything (golden rule): the cluster + KAI + HAMi are
    stood up out of band by infra/k8s/setup.sh. provision/bootstrap only VERIFY
    the pieces exist and fail closed with a pointer if not. run() renders a
    Namespace + PodGroup + N Pods and applies them; collect() ships each rank's
    logs to the drop-zone; deprovision() deletes the job's namespace (leaving
    the cluster standing, exactly like LocalHost/StaticSsh leave their machine).

    The manifest shape was verified live before this landed: a pre-created
    PodGroup + pods carrying the `pod-group-name` annotation bind to that gang
    (KAI's auto-grouper does NOT create per-pod groups instead), and HAMi injects
    CUDA_DEVICE_MEMORY_LIMIT per rank."""
    name = "k8s"

    def __init__(self, log=_log_stderr):
        self.log = log

    # ── kubectl plumbing ─────────────────────────────────────────────────
    @staticmethod
    def _context_of(node: NodeRecord) -> str:
        """Which kube-context this slot targets: its own if set (multi-cluster:
        the L20's cluster vs the 5060 Ti's), else the global CLUSTER_K8S_CONTEXT."""
        return node.kube_context or K8S_CONTEXT

    def _kubectl_base(self, context: str = "") -> list[str]:
        return ["kubectl"] + (["--context", context] if context else [])

    def _kubectl(self, args: list[str], job_id: str, context: str = "", check: bool = True) -> int:
        """A kubectl call whose output streams live into the job's replay log
        (apply/delete/verify) — same treatment every other adapter's commands get."""
        return run_logged(self._kubectl_base(context) + args, job_id, check=check)

    def _kubectl_out(self, args: list[str], context: str = "") -> tuple[int, str]:
        """A kubectl call whose stdout we parse (polling phases, fetching logs) —
        captured, not streamed, so it doesn't spam the replay log every 5s."""
        proc = subprocess.run(self._kubectl_base(context) + args, capture_output=True, text=True)
        return proc.returncode, proc.stdout

    @staticmethod
    def _namespace(job_id: str) -> str:
        # job_id is `job-<12 hex>`, so this is a valid DNS-1123 label (<=63 chars).
        return f"{K8S_NAMESPACE_PREFIX}{job_id}"

    @staticmethod
    def _image_ref(image: str) -> str:
        """A JobSpec image for k8s must be an OCI ref containerd can pull. Accept
        a `docker://` ref (the same form apptainer takes) by stripping the scheme;
        reject a local .sif / path / other scheme with a clear message — the k8s
        backend pulls images, it can't run a host SIF."""
        ref = image[len("docker://"):] if image.startswith("docker://") else image
        if ref.endswith(".sif") or "://" in ref or ref.startswith(("/", "./", "../")):
            raise RuntimeError(
                f"k8s backend needs an OCI image ref (e.g. 'nvidia/cuda:12.6.3-base-ubuntu24.04' "
                f"or 'docker://…'), got {image!r} — it pulls via containerd, not a local SIF.")
        return ref

    # ── node lifecycle (verify only; the cluster is managed out of band) ──
    def provision(self, node, job_id):
        ctx = self._context_of(node)
        self.log(f"[k8s] verifying cluster reachable + KAI up (slot {node.name}"
                 f"{f', context {ctx}' if ctx else ''})")
        if self._kubectl(["get", "nodes"], job_id, context=ctx, check=False) != 0:
            raise RuntimeError(
                f"k8s cluster unreachable (`kubectl {f'--context {ctx} ' if ctx else ''}get nodes` "
                "failed). Stand it up first with infra/k8s/setup.sh, and set the slot's "
                "kube_context / CLUSTER_K8S_CONTEXT / KUBECONFIG if it isn't your current "
                "kube-context.")
        rc, out = self._kubectl_out(["get", "pods", "-n", "kai-scheduler",
                                     "--field-selector=status.phase=Running", "-o", "name"],
                                    context=ctx)
        if rc != 0 or not out.strip():
            raise RuntimeError(
                "KAI scheduler is not Running in namespace 'kai-scheduler' — run "
                "infra/k8s/setup.sh (installs KAI + the HAMi-core isolator).")

    def deprovision(self, node, job_id):
        ctx = self._context_of(node)
        ns = self._namespace(job_id)
        self.log(f"[k8s] deleting namespace {ns} (cascades pods + PodGroup); cluster left standing")
        # --wait=false: teardown shouldn't block the tick on namespace GC; a
        # lingering terminating namespace doesn't affect a later job (each job
        # gets its own uniquely-named namespace).
        self._kubectl(["delete", "namespace", ns, "--ignore-not-found", "--wait=false"],
                      job_id, context=ctx, check=False)

    def bootstrap(self, nodes, job_id):
        # Verify the scheduling prerequisites exist. Installs NOTHING (golden
        # rule) — setup.sh owns the install; a missing piece is a clear failure
        # here, not a mysterious pod-never-schedules later.
        ctx = self._context_of(nodes[0])
        self.log(f"[k8s] verifying scheduling prerequisites (queue={K8S_QUEUE}"
                 f"{f', context {ctx}' if ctx else ''})")
        for args, what in (
            (["get", "queue", K8S_QUEUE], f"KAI queue {K8S_QUEUE!r}"),
            (["get", "runtimeclass", "nvidia"], "RuntimeClass 'nvidia' (isolator injects it)"),
            (["get", "crd", "podgroups.scheduling.run.ai"], "the PodGroup CRD (KAI)"),
        ):
            if self._kubectl(args, job_id, context=ctx, check=False) != 0:
                raise RuntimeError(f"{what} not found — run infra/k8s/setup.sh")

    # ── workload lifecycle ────────────────────────────────────────────────
    def run(self, nodes, job_id, spec, plan=None):
        # `plan` never applies here: backend: k8s jobs are already refused
        # earlier (reconciler.py _phase_provision) if they carry a launch
        # block — the kubelet, not this cluster, owns in-pod CPU/GPU.
        slot = nodes[0]
        ctx = self._context_of(slot)
        workdir = os.path.join(RUNS_DIR, job_id)
        os.makedirs(workdir, exist_ok=True)
        if spec.is_dry_run:
            return RunResult(job_id, slot.name, None, workdir, note="no image -> dry-run")
        n = max(1, spec.node_count)                    # node_count == gang size here
        ns = self._namespace(job_id)
        manifest = self._render_manifests(job_id, spec, n)
        manifest_path = os.path.join(workdir, "manifests.yaml")
        with open(manifest_path, "w") as f:
            f.write(manifest)
        declared_gpu_mem = spec.params.get("gpu_memory_mb", K8S_GPU_MEMORY_MB)
        self.log(f"[k8s] applying gang: ns={ns} minMember={n} image={self._image_ref(spec.image)} "
                 f"gpu-memory={declared_gpu_mem}MiB/rank queue={spec.params.get('queue', K8S_QUEUE)}"
                 f"{f' context={ctx}' if ctx else ''}")
        self._kubectl(["apply", "-f", manifest_path], job_id, context=ctx)
        rc = self._await_gang(ns, n, job_id, ctx)
        stdout_path = self._capture_logs(ns, n, job_id, workdir, ctx)
        if spec.params.get("verify_gpu_memory"):
            self._build_gpu_memory_receipt(workdir, n, spec)
        return RunResult(job_id, slot.name, rc, workdir, stdout_path,
                         note=f"k8s gang np={n} (ns {ns}) exit={rc}")

    @staticmethod
    def _gpu_mem_for_rank(spec: JobSpec, i: int, n: int) -> str:
        """gpu_memory_mb may be one scalar (today's default — every pod gets
        the same value) or a list of exactly N values, one per rank. A
        mismatched list length fails the job before anything is applied,
        rather than silently reusing or truncating the declared policy."""
        declared = spec.params.get("gpu_memory_mb", K8S_GPU_MEMORY_MB)
        if isinstance(declared, list):
            if len(declared) != n:
                raise ValueError(
                    f"gpu_memory_mb is a list of {len(declared)} value(s) but this "
                    f"gang has {n} rank(s) — one entry per rank, or a single shared value")
            return str(declared[i])
        return str(declared)

    def _render_manifests(self, job_id: str, spec: JobSpec, n: int) -> str:
        """Render the Namespace + headless Service + PodGroup + N Pod docs for one
        gang. Pure (no cluster calls) so it's unit-testable. Pods ask for a GPU
        *fraction* via the `gpu-memory` annotation (NOT an `nvidia.com/gpu`
        resource request, which would consume the whole card), attach to the
        explicit PodGroup via `pod-group-name`, and are placed by KAI
        (`schedulerName`). Each pod gets a stable DNS name via the headless
        Service (`hostname`/`subdomain`) and the standard distributed-rendezvous
        env (RANK, WORLD_SIZE, MASTER_ADDR=rank-0's FQDN, MASTER_PORT), so a real
        multi-rank workload (torchrun / c10d / NCCL-over-TCP / MPI-over-TCP) can
        form a communicator across the gang — not just N independent pods.

        `spec.params.verify_gpu_memory` (opt-in, default unset): prepends
        gpu-memory-probe.sh to each pod's command so the VRAM cap HAMi's
        isolator actually enforced is observable from inside the container.
        The pod spec (post-mutation) DOES reference it (`envFrom` a generated
        ConfigMap), but the resolved value — and whether libvgpu.so actually
        intercepted THIS process's CUDA calls with it — is only knowable from
        inside the running container (see the probe's own docstring).
        Requires `spec.command` to be set: there is nothing to prepend a
        probe onto an image's own entrypoint without overriding it."""
        import yaml
        ns = self._namespace(job_id)
        image = self._image_ref(spec.image)
        pg_name = f"pg-{job_id}"
        queue = str(spec.params.get("queue", K8S_QUEUE))
        master_port = str(spec.params.get("master_port", K8S_MASTER_PORT))
        master_addr = f"rank-0.{K8S_RDZV_SERVICE}.{ns}.svc.cluster.local"
        verify_gpu_memory = bool(spec.params.get("verify_gpu_memory"))
        if verify_gpu_memory and not spec.command:
            raise ValueError("verify_gpu_memory needs spec.command set — there is no "
                             "entrypoint to prepend the probe onto otherwise")
        docs: list[dict] = [
            {"apiVersion": "v1", "kind": "Namespace",
             "metadata": {"name": ns, "labels": {"outpost-job": job_id}}},
            # Headless Service: gives each pod (hostname=rank-<i>, subdomain=gang)
            # a DNS A record. publishNotReadyAddresses so ranks resolve during
            # startup rendezvous (the pods carry no readiness probe).
            {"apiVersion": "v1", "kind": "Service",
             "metadata": {"name": K8S_RDZV_SERVICE, "namespace": ns},
             "spec": {"clusterIP": "None", "publishNotReadyAddresses": True,
                      "selector": {"outpost-job": job_id},
                      "ports": [{"name": "rdzv", "port": int(master_port)}]}},
            {"apiVersion": K8S_PODGROUP_APIVERSION, "kind": "PodGroup",
             "metadata": {"name": pg_name, "namespace": ns},
             "spec": {"minMember": n, "queue": queue}},
        ]
        for i in range(n):
            # order: sane defaults, then rendezvous, then the job's own env last
            # so a spec can override any of them.
            merged = {**K8S_DEFAULT_ENV,
                      "RANK": str(i), "WORLD_SIZE": str(n),
                      "MASTER_ADDR": master_addr, "MASTER_PORT": master_port,
                      **spec.env}
            env = [{"name": k, "value": v} for k, v in merged.items()]
            container: dict = {"name": "rank", "image": image,
                               "imagePullPolicy": "IfNotPresent", "env": env}
            if spec.command:                            # else use the image entrypoint
                command = list(spec.command)
                if verify_gpu_memory:
                    probe = GPU_MEMORY_PROBE_PATH.read_text()
                    inner = " ".join(shlex.quote(a) for a in command)
                    command = ["sh", "-c", f"({probe}) 2>&1; exec {inner}"]
                container["command"] = command
            docs.append({
                "apiVersion": "v1", "kind": "Pod",
                "metadata": {
                    "name": f"rank-{i}", "namespace": ns,
                    "labels": {"kai.scheduler/queue": queue,
                               "outpost-job": job_id, "outpost-rank": str(i)},
                    "annotations": {"pod-group-name": pg_name,
                                   "gpu-memory": self._gpu_mem_for_rank(spec, i, n)},
                },
                "spec": {"schedulerName": K8S_SCHEDULER, "restartPolicy": "Never",
                         "hostname": f"rank-{i}", "subdomain": K8S_RDZV_SERVICE,
                         "containers": [container]},
            })
        return yaml.safe_dump_all(docs, default_flow_style=False, sort_keys=False)

    def _await_gang(self, ns: str, n: int, job_id: str, context: str = "") -> int:
        """Poll until all N pods are terminal (Succeeded/Failed) or the run
        timeout elapses. Fail fast on an unrecoverable image/create error rather
        than waiting out the whole timeout. Returns the WORST rank exit code (0
        iff every rank succeeded) so the reconciler fails the job on any nonzero
        rank, exactly like a nonzero apptainer/mpirun exit on the other paths."""
        deadline = time.monotonic() + K8S_RUN_TIMEOUT_S
        while True:
            _, phase_out = self._kubectl_out(["get", "pods", "-n", ns, "-o",
                "jsonpath={range .items[*]}{.metadata.name}={.status.phase};{end}"], context=context)
            phases = dict(p.split("=", 1) for p in phase_out.strip(";").split(";") if "=" in p)
            _, wait_out = self._kubectl_out(["get", "pods", "-n", ns, "-o",
                "jsonpath={range .items[*]}{.metadata.name}="
                "{.status.containerStatuses[0].state.waiting.reason};{end}"], context=context)
            waiting = dict(w.split("=", 1) for w in wait_out.strip(";").split(";") if "=" in w)
            fatal = {p: r for p, r in waiting.items() if r in K8S_FATAL_WAIT_REASONS}
            terminal = {p: v for p, v in phases.items() if v in ("Succeeded", "Failed")}
            msg = f"[k8s] {ns}: {len(terminal)}/{n} pods terminal (phases={phases})"
            self.log(msg)
            append_job_log(job_id, f"{now_iso()} {msg}")
            if fatal:
                raise RuntimeError(f"k8s gang in {ns} has unrecoverable pod(s): {fatal} "
                                   "(check the image ref / pull access)")
            if len(phases) >= n and len(terminal) == len(phases):
                break
            if time.monotonic() >= deadline:
                raise RuntimeError(f"k8s gang in {ns} did not finish within "
                                   f"{K8S_RUN_TIMEOUT_S:.0f}s (phases={phases})")
            time.sleep(K8S_POLL_INTERVAL_S)
        _, code_out = self._kubectl_out(["get", "pods", "-n", ns, "-o",
            "jsonpath={range .items[*]}{.status.containerStatuses[0].state.terminated.exitCode}"
            ";{end}"], context=context)
        codes = [int(c) for c in code_out.strip(";").split(";") if c.strip().lstrip("-").isdigit()]
        return max(codes) if codes else 1              # no exit codes at all -> treat as failure

    def _capture_logs(self, ns: str, n: int, job_id: str, workdir: str, context: str = "") -> str:
        """Fetch each rank's logs into workdir: per-rank rank-<i>.log plus a
        concatenated stdout.log (the guaranteed egress artifact every adapter
        produces). Runs while the namespace still exists (collect precedes
        teardown), so the Succeeded pods' logs are still available."""
        combined = os.path.join(workdir, "stdout.log")
        with open(combined, "w") as agg:
            for i in range(n):
                pod = f"rank-{i}"
                _, out = self._kubectl_out(["logs", "-n", ns, pod], context=context)
                with open(os.path.join(workdir, f"{pod}.log"), "w") as rf:
                    rf.write(out)
                agg.write(f"=== {pod} ===\n{out}\n")
                append_job_log(job_id, f"{now_iso()} [k8s] {ns}/{pod} logs:\n{out}")
        return combined

    def _build_gpu_memory_receipt(self, workdir: str, n: int, spec: JobSpec) -> None:
        """Compare the declared per-rank gpu-memory policy against what
        gpu-memory-probe.sh actually observed inside each pod (already local —
        _capture_logs wrote rank-<i>.log before this runs; no extra kubectl
        round trip needed). Writes gpu-memory-receipt.yaml into workdir, which
        collect() already ships wholesale. A rank with no parseable probe line
        is recorded as unmatched, never silently skipped — same rule as
        receipt.py's build_receipt for the CPU/rankfile path."""
        import yaml
        ranks = []
        for i in range(n):
            declared_mb = int(self._gpu_mem_for_rank(spec, i, n))
            observed = None
            log_path = os.path.join(workdir, f"rank-{i}.log")
            if os.path.isfile(log_path):
                for line in open(log_path):
                    if line.startswith(_GPU_MEMORY_MARKER):
                        try:
                            observed = json.loads(line[len(_GPU_MEMORY_MARKER):].strip())
                        except json.JSONDecodeError:
                            observed = None
                        break
            observed_mb = observed.get("cuda_device_memory_limit_mb") if observed else None
            ranks.append({
                "rank": i, "declared_mb": declared_mb,
                "observed_mb": observed_mb,
                "observed_gpu_uuid": observed.get("gpu_uuid") if observed else None,
                "matched": observed_mb == declared_mb,
            })
        with open(os.path.join(workdir, "gpu-memory-receipt.yaml"), "w") as f:
            yaml.safe_dump({"ranks": ranks}, f, sort_keys=False)

    def collect(self, nodes, job_id, spec, dest):
        # The run already pulled every rank's logs into .var/runs/<job_id>; just
        # ship that workdir to the drop-zone (same copy pattern as LocalHost).
        workdir = os.path.join(RUNS_DIR, job_id)
        os.makedirs(dest, exist_ok=True)
        if os.path.isdir(workdir):
            for name in os.listdir(workdir):
                if _is_staging_artifact(name):
                    continue
                src = os.path.join(workdir, name)
                dst = os.path.join(dest, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        return dest
