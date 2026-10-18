"""Data records + MongoDB collection names.

Plain dataclasses with explicit (de)serialization so the same shapes work for
the Mongo store and the file store.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import re
import uuid

from . import inputs as inputs_mod
from . import launch_models
from .states import JobState, NodeState

# MongoDB collection names (also the JSON keys in the file store).
COL_JOBS = "jobs"
COL_NODES = "nodes"
COL_AUDIT = "audit"


# Providers whose registry nodes are slots on a whole cluster backend rather
# than machines. A plain pool claim never takes one; only a job routed to that
# backend (spec.backend, or a node_name pin naming the slot) does.
BACKEND_PROVIDERS = ("k8s", "slinky")


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


_ORAS_REF_RE = re.compile(r"^oras://[^@\s]+@sha256:[0-9a-f]{64}$")
_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")


def parse_image_sif(sif: Any) -> dict[str, str] | None:
    if sif is None:
        return None
    if not (isinstance(sif, dict) and _ORAS_REF_RE.match(str(sif.get("ref") or ""))
            and _SHA256_RE.match(str(sif.get("sha256") or ""))):
        raise ValueError(
            f"image_sif must be {{ref: oras://<registry>/<repo>@sha256:<64 hex>, sha256: "
            f"sha256:<64 hex>}}, got {sif!r} — fix the image pin that produced this job")
    return {"ref": sif["ref"], "sha256": sif["sha256"]}


@dataclass
class JobSpec:
    """A generic container job — the intake half of the cluster's open interface.

    The cluster runs ANY container (`image` + `command`) under the requested
    resources and returns whatever it writes to `output_dir`. It makes no
    assumptions about what that container does; the meaning of the output is
    entirely the caller's concern.
    """
    name: str
    # what to run
    image: str = ""                    # apptainer SIF path/URI or docker[://] ref; "" = dry-run marker
    command: list[str] = field(default_factory=list)   # argv inside the container
    runtime: str = "apptainer"         # apptainer | docker (how to launch `image`)
    launcher: str = "single"           # single | mpi (mpi -> mpirun, one rank per claimed node)
    # which execution backend runs this job. "" (default) = the VM/host/static
    # pool routed by gpu/hybrid below; "k8s" = the KubernetesAdapter (KAI + HAMi),
    # where the job claims ONE k8s slot and node_count is reinterpreted as the
    # gang size (N pods on the shared GPU). See docs/08-kubernetes-backend.md.
    # "slinky" = the SlinkyAdapter: one slot, node_count = Slurm nodes (docs/11).
    backend: str = ""
    # scheduling constraints
    node_count: int = 1
    gpu: bool = False                  # true -> must land on a GPU-capable node (the host)
    hybrid: bool = False               # true -> claim 1 GPU node (the host) + (node_count-1)
                                       # CPU VMs and launch ONE mpirun across them;
                                       # requires launcher: mpi and node_count >= 2
    # I/O contract
    env: dict[str, str] = field(default_factory=dict)  # env vars set inside the container
    output_dir: str = "/out"           # in-container path the job writes results to
    params: dict[str, Any] = field(default_factory=dict)  # opaque passthrough
    # semantic placement request (contract/launch-intent/v1). None = the caller has
    # no placement opinion, which is the pre-existing behaviour and the default.
    # Unlike `params`, this is NOT opaque passthrough: it is validated on parse and
    # a block we cannot honour refuses the job (see launch_models).
    launch: dict[str, Any] | None = None
    # `image` prebuilt as a SIF: {"ref": "oras://…@sha256:…", "sha256": "sha256:<file>"}.
    # Apptainer hosts download it instead of converting `image`; validated on parse.
    image_sif: dict[str, str] | None = None
    # files outpost downloads, verifies and mounts read-only (reconciler/inputs.py)
    inputs: list[dict[str, Any]] = field(default_factory=list)
    # set by an adapter, never by a caller: `src:dst[:ro]` binds for the container
    mounts: list[str] = field(default_factory=list)
    # HOST RAM the payload needs, in bytes. 0 = the caller states no requirement,
    # which is every job written before this existed and stays the default.
    #
    # A scheduling constraint, not passthrough: a job that needs more host memory
    # than a node has is not slow, it is OOM-killed part-way through a solve, and
    # the measurement is lost along with the GPU time that produced it. HPL-MxP
    # keeps an FP64 copy of the matrix on the host, so the same N that fits a
    # 32 GiB box is killed on a 16 GiB one with the same GPU.
    host_memory_bytes: int = 0
    # legacy label (kept so old specs still parse)
    workload: str = ""

    @staticmethod
    def from_dict(d: dict) -> "JobSpec":
        cmd = d.get("command", []) or []
        if isinstance(cmd, str):
            cmd = cmd.split()
        # Every other field here is best-effort: an unrecognised key is dropped and
        # a missing one defaults, which is right for a deliberately open interface.
        # `launch` is the exception and must stay one — a placement request that is
        # quietly discarded produces a run that succeeds, parses, and measures a
        # machine configuration nobody chose. So it validates, and it raises.
        launch = d.get("launch")
        if launch is not None:
            launch = launch_models.parse(launch)
        return JobSpec(
            name=d["name"],
            image=d.get("image", "") or "",
            command=list(cmd),
            runtime=d.get("runtime", "apptainer"),
            launcher=d.get("launcher", "single") or "single",
            backend=d.get("backend", "") or "",
            node_count=int(d.get("node_count", 1)),
            gpu=bool(d.get("gpu", False)),
            hybrid=bool(d.get("hybrid", False)),
            env={str(k): str(v) for k, v in (d.get("env", {}) or {}).items()},
            output_dir=d.get("output_dir", "/out"),
            params=d.get("params", {}) or {},
            launch=launch,
            image_sif=parse_image_sif(d.get("image_sif")),
            inputs=inputs_mod.parse_inputs(d.get("inputs")),
            # Parsed, not dropped. A requirement silently discarded is worse than
            # one never stated: the job runs, gets OOM-killed, and nothing says
            # why -- the same failure mode `launch` above is parsed to avoid.
            host_memory_bytes=int(d.get("host_memory_bytes", 0) or 0),
            workload=d.get("workload", "") or "",
        )

    @property
    def is_dry_run(self) -> bool:
        """No image -> a control-plane exercise; the run phase does nothing real."""
        return not self.image


@dataclass
class JobRecord:
    job_id: str
    spec: dict                          # serialized JobSpec
    state: str = JobState.SUBMITTED.value
    assigned_nodes: list[str] = field(default_factory=list)
    run: dict | None = None             # serialized RunResult once the run phase completes
    drop_path: str | None = None        # where collected artifacts landed (egress half)
    error: str | None = None
    # Resolved placement, once a planning phase has produced one. Lives on the
    # record rather than only in the drop-zone because a caller has to be able to
    # SEE the plan before execution, and the drop-zone is not written until
    # collect. Both default, so records written before these existed still parse
    # (from_dict is JobRecord(**d): missing keys default, unknown keys raise) —
    # which is what makes this an additive change with no store migration.
    plan: dict | None = None
    plan_status: str = "none"           # none | planning | ready | approved | stale | failed
    created_at: str | None = None
    updated_at: str | None = None

    @staticmethod
    def create(spec: JobSpec) -> "JobRecord":
        return JobRecord(job_id=new_id("job"), spec=asdict(spec))

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "JobRecord":
        d = {k: v for k, v in d.items() if k != "_id"}
        return JobRecord(**d)


@dataclass
class NodeRecord:
    node_id: str                        # stable id, == name today
    name: str                           # cluster-node-01  /  cluster-host
    index: int                          # 1..N drives the libvirt wrappers; 0 = host
    ip: str
    state: str = NodeState.AVAILABLE.value
    owner_job: str | None = None        # exclusive-lock holder
    # capabilities (drive scheduling + which adapter runs the node)
    gpu: bool = False                   # has a usable GPU (the host does)
    local: bool = False                 # the control-plane host itself; never libvirt-provisioned
    runtime: str = "apptainer"          # container runtime available on the node
    # which adapter drives this node's lifecycle + workloads. "" is derived
    # (local -> "local", else "libvirt") so records written before this field
    # keep their old routing. "static-ssh" = a pre-provisioned ssh box (e.g. an
    # EC2 instance) that we don't provision/destroy, only run containers on.
    provider: str = ""
    # per-node ssh identity, for a static node that uses its own credentials
    # (e.g. EC2's `ubuntu` user + its keypair) instead of the cluster fabric's.
    # "" -> fall back to the cluster-wide CLUSTER_SSH_USER / CLUSTER_SSH_PRIVKEY.
    ssh_user: str = ""
    ssh_key: str = ""
    # per-node GPU driver binds for `apptainer --nv` (comma-sep --bind paths),
    # overriding the control plane's global CLUSTER_GPU_BINDS. The point is a
    # REMOTE gpu worker whose driver libs live somewhere non-standard: a NixOS
    # box reached over ssh needs "/nix/store,/run/opengl-driver" (its own paths,
    # which exist on that node), while an Ubuntu worker wants "" (--nv self-
    # detects). Only consulted for gpu runs; empty = plain --nv.
    gpu_binds: str = ""
    # TOTAL host RAM on this node, in bytes, as the node itself reported it at
    # registration (MemTotal). 0 = never probed, which is every node registered
    # before this existed.
    #
    # Nothing is inferred from 0: a job that states a host_memory_bytes
    # requirement is REFUSED on an unprobed node rather than gambled on, because
    # the gamble is paid for in GPU time and a lost measurement. Re-register the
    # node (remove-node + add-node) and it is probed.
    host_memory_bytes: int = 0
    # for a k8s-backend slot: which kube-context (i.e. which cluster) it targets,
    # so several k8s clusters (e.g. an AWS GPU worker + the PC's 5060 Ti) can live
    # under ONE control plane. "" -> the adapter's global CLUSTER_K8S_CONTEXT
    # (current kube-context). Ignored by every non-k8s node.
    kube_context: str = ""
    # the address OTHER ranks use to reach this node for MPI (hostfile/appfile
    # entries, fabric_if_include's BTL/OOB subnet matching, _node_iface's
    # interface lookup) — NOT necessarily the same address the orchestrator
    # uses to SSH in and drive the node (`ip`). For libvirt/local nodes and the
    # AWS static-ssh proof, one address satisfies both (orchestrator and peers
    # share reachability to it), so "" -> fall back to `ip` via the `mpi_ip`
    # property below and nothing changes. It diverges for a cloud that NATs
    # public IPs (confirmed on Nebius): the orchestrator needs the public IP
    # to reach the node at all, but that address is never bound on the guest's
    # own interface, so every MPI-facing lookup against it fails — the guest
    # only answers to its private VPC IP. Two same-subnet nodes already reach
    # each other over that private IP for free; this field just says which one
    # to use for MPI purposes instead of routing around it with a VPN/bastion.
    cluster_ip: str = ""
    # for a slinky slot that stands for ONE Slurm node (`add-slinky --per-node`):
    # that node's Slurm name and GPU count. "" = a cluster-wide slot (a
    # concurrency bound; Slurm picks the nodes).
    slurm_node: str = ""
    gpu_count: int = 0
    updated_at: str | None = None

    @property
    def adapter_key(self) -> str:
        """The adapters-dict key that routes this node (see Reconciler._adapter_for)."""
        return self.provider or ("local" if self.local else "libvirt")

    @property
    def mpi_ip(self) -> str:
        """The address to use for anything a PEER rank (not the orchestrator)
        needs to reach this node at — see `cluster_ip`'s docstring."""
        return self.cluster_ip or self.ip

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "NodeRecord":
        # tolerate records written before capability fields existed
        keep = {"node_id", "name", "index", "ip", "state", "owner_job",
                "gpu", "local", "runtime", "provider", "ssh_user", "ssh_key",
                "gpu_binds", "kube_context", "cluster_ip", "slurm_node", "gpu_count",
                "updated_at", "host_memory_bytes"}
        return NodeRecord(**{k: v for k, v in d.items() if k in keep})


@dataclass
class RunResult:
    """Outcome of executing a job's container on a node."""
    job_id: str
    node: str                           # node the container ran on
    exit_code: int | None               # None if not executed (dry-run)
    workdir: str = ""                   # host dir bound to the job's output_dir
    stdout_path: str = ""               # captured stdout/stderr log
    note: str = ""

    def to_dict(self) -> dict:
        return asdict(self)
