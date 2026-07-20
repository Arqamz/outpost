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
import time
from abc import ABC, abstractmethod
from pathlib import Path

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


def ssh_base(ip: str) -> list[str]:
    """argv prefix for reaching a cluster node over ssh with the cluster key.
    Module-level (not a LibvirtAdapter detail) because the hybrid MPI path on
    the host stages images into VMs with the exact same fabric."""
    return ["ssh", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", f"{SSH_USER}@{ip}"]


def scp_to(ip: str, src: str, dst: str, job_id: str) -> None:
    argv = ["scp", "-i", SSH_KEY, "-o", "StrictHostKeyChecking=no",
            "-o", "UserKnownHostsFile=/dev/null", src, f"{SSH_USER}@{ip}:{dst}"]
    run_logged(argv, job_id)


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

    def run(self, nodes, job_id, spec):
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
        image = f"{workdir}/{os.path.basename(spec.image)}"
        appfile = f"{workdir}/appfile"
        shutil.copy2(spec.image, image)
        for n in nodes[1:]:
            run_logged(ssh_base(n.ip) + [f"mkdir -p {workdir}"], job_id)
            scp_to(n.ip, spec.image, image, job_id)

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
        return ssh_base(ip)

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
        scp_to(ip, src, dst, job_id)

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
        btl_inc, oob_inc = fabric_if_include(nodes)

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
        argv = mpirun_argv(len(nodes), remote_hostfile, btl_inc, oob_inc, per_rank)
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
