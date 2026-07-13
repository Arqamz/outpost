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
import os
import shlex
import shutil
import subprocess
import sys
import threading
from abc import ABC, abstractmethod

from .models import NodeRecord, JobSpec, RunResult
from .store import now_iso


def _log_stderr(msg: str) -> None:
    print(msg, file=sys.stderr)


REPO_ROOT = os.environ.get("CLUSTER_ROOT") or os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LIBVIRT_DIR = os.path.join(REPO_ROOT, "infra", "libvirt")
ANSIBLE_DIR = os.path.join(REPO_ROOT, "infra", "ansible")
RUNS_DIR = os.path.join(REPO_ROOT, ".var", "runs")   # per-job host workdirs
LOGS_DIR = os.path.join(REPO_ROOT, ".var", "logs")   # replay transcripts (gitignored)

# SSH params for reaching VMs (match infra/libvirt/config.env defaults).
SSH_USER = os.environ.get("CLUSTER_SSH_USER", "cluster")
SSH_KEY = os.environ.get("CLUSTER_SSH_PRIVKEY", os.path.expanduser("~/.ssh/outpost-cluster-ssh"))


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


def mpirun_argv(np: int, hostfile: str, cidr: str, per_rank_argv: list[str]) -> list[str]:
    """Wrap a per-rank container argv (from container_argv) in mpirun, one rank
    per claimed node (--map-by node) over the cluster's own TCP fabric — the MCA
    if_include options keep mpirun off any other interface on the box."""
    return [
        "mpirun", "-np", str(np), "--hostfile", hostfile, "--map-by", "node",
        "--mca", "btl", "tcp,self",
        "--mca", "btl_tcp_if_include", cidr,
        "--mca", "oob_tcp_if_include", cidr,
    ] + per_rank_argv


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

    # workload lifecycle (head node = nodes[0])
    @abstractmethod
    def run(self, nodes: list[NodeRecord], job_id: str, spec: JobSpec) -> RunResult: ...
    @abstractmethod
    def collect(self, nodes: list[NodeRecord], job_id: str, spec: JobSpec, dest: str) -> str: ...


class NullAdapter(ProviderAdapter):
    """Logs intents, touches nothing. Default for dry-run / skeleton runs."""
    name = "null"

    def __init__(self, log=_log_stderr):
        self.log = log

    def provision(self, node, job_id):    self.log(f"[null] provision {node.name} (no-op)")
    def deprovision(self, node, job_id):  self.log(f"[null] deprovision {node.name} (no-op)")
    def bootstrap(self, nodes, job_id):   self.log(f"[null] bootstrap {[n.name for n in nodes]} (no-op)")

    def run(self, nodes, job_id, spec):
        self.log(f"[null] would run '{spec.image or '(no image)'}' on {nodes[0].name}")
        return RunResult(job_id, nodes[0].name, None, note="dry-run (null adapter)")

    def collect(self, nodes, job_id, spec, dest):
        self.log(f"[null] would collect -> {dest}")
        os.makedirs(dest, exist_ok=True)
        return dest


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
        # Just verify the runtime exists; the host is already configured.
        rc = subprocess.run(["apptainer", "--version"], capture_output=True).returncode
        if rc != 0:
            self.log("[localhost] WARNING: apptainer not found on host "
                     "(add it via the nix dev shell before running real jobs)")
        else:
            self.log("[localhost] apptainer present")

    def run(self, nodes, job_id, spec):
        node = nodes[0]
        workdir = os.path.join(RUNS_DIR, job_id)
        os.makedirs(workdir, exist_ok=True)
        if spec.is_dry_run:
            return RunResult(job_id, node.name, None, workdir, note="no image -> dry-run")
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

    def collect(self, nodes, job_id, spec, dest):
        workdir = os.path.join(RUNS_DIR, job_id)
        os.makedirs(dest, exist_ok=True)
        if os.path.isdir(workdir):
            for name in os.listdir(workdir):
                src = os.path.join(workdir, name)
                dst = os.path.join(dest, name)
                if os.path.isdir(src):
                    shutil.copytree(src, dst, dirs_exist_ok=True)
                else:
                    shutil.copy2(src, dst)
        return dest


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

    def _ssh_base(self, ip: str) -> list[str]:
        return ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", f"{SSH_USER}@{ip}"]

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
        with self._inventory_lock:
            run_logged([os.path.join(LIBVIRT_DIR, "gen-inventory.sh")], job_id)
        limit = ",".join(n.name for n in nodes)
        run_logged(["ansible-playbook", "-i", os.path.join(ANSIBLE_DIR, "inventory", "hosts.ini"),
                   "--limit", limit, "site.yml"], job_id, cwd=ANSIBLE_DIR)

    def run(self, nodes, job_id, spec):
        head = nodes[0]
        remote_workdir = f"/tmp/cluster/{job_id}"
        if spec.is_dry_run:
            return RunResult(job_id, head.name, None, remote_workdir, note="no image -> dry-run")
        if spec.launcher == "mpi" and len(nodes) > 1:
            return self._run_mpi(nodes, job_id, spec, remote_workdir)
        argv = container_argv(spec, remote_workdir, spec.output_dir)
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
        rc = run_logged(self._ssh_base(head.ip) + [remote], job_id, check=False)
        return RunResult(job_id, head.name, rc, remote_workdir, stdout_log, note=f"remote exit={rc}")

    def _scp_to(self, ip: str, src: str, dst: str, job_id: str) -> None:
        argv = ["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
                "-o", "UserKnownHostsFile=/dev/null", src, f"{SSH_USER}@{ip}:{dst}"]
        run_logged(argv, job_id)

    def _run_mpi(self, nodes: list[NodeRecord], job_id: str, spec: JobSpec,
                 remote_workdir: str) -> RunResult:
        """Multi-node hybrid launch: mpirun runs on the head VM (installed by
        the bootstrap role) and execs `apptainer exec <image> ...` per rank
        on every claimed node over ssh — the container never needs its own MPI
        launcher, only a matching libmpi. Hostfile + image are per-job (built
        from the nodes THIS job claimed, not the cluster-wide 8) and staged
        fresh onto every claimed node under remote_workdir."""
        head = nodes[0]
        remote_image = f"{remote_workdir}/{os.path.basename(spec.image)}"
        remote_hostfile = f"{remote_workdir}/hostfile"
        cidr = ".".join(head.ip.split(".")[:3]) + ".0/24"

        local_scratch = os.path.join(RUNS_DIR, job_id)
        os.makedirs(local_scratch, exist_ok=True)
        local_hostfile = os.path.join(local_scratch, "hostfile")
        with open(local_hostfile, "w") as f:
            for n in nodes:
                f.write(f"{n.ip} slots=1\n")

        for n in nodes:
            run_logged(self._ssh_base(n.ip) + [f"mkdir -p {remote_workdir}"], job_id)
            self._scp_to(n.ip, spec.image, remote_image, job_id)
        self._scp_to(head.ip, local_hostfile, remote_hostfile, job_id)

        rank_spec = dataclasses.replace(spec, image=remote_image)
        per_rank = container_argv(rank_spec, remote_workdir, spec.output_dir)
        argv = mpirun_argv(len(nodes), remote_hostfile, cidr, per_rank)
        inner = " ".join(shlex.quote(a) for a in argv)
        stdout_log = f"{remote_workdir}/stdout.log"
        # Same reasoning as the single-node path: tee mpirun's combined output
        # (all ranks, since mpirun forwards each rank's stdout to the launching
        # process) into the head's workdir so collect() ships it to the drop-zone.
        remote = f"({inner}) 2>&1 | tee {stdout_log}; exit ${{PIPESTATUS[0]}}"
        self.log(f"[libvirt] ssh {head.name} (mpirun head, {len(nodes)} ranks): {inner}")
        rc = run_logged(self._ssh_base(head.ip) + [remote], job_id, check=False)
        return RunResult(job_id, head.name, rc, remote_workdir, stdout_log,
                         note=f"mpirun np={len(nodes)} exit={rc}")

    def collect(self, nodes, job_id, spec, dest):
        head = nodes[0]
        remote_workdir = f"/tmp/cluster/{job_id}"
        os.makedirs(dest, exist_ok=True)
        run_logged(["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
                   "-o", "UserKnownHostsFile=/dev/null", "-r",
                   f"{SSH_USER}@{head.ip}:{remote_workdir}/.", dest], job_id, check=False)
        return dest
