"""Data records + MongoDB collection names.

Plain dataclasses with explicit (de)serialization so the same shapes work for
the Mongo store and the file store.
"""
from __future__ import annotations
from dataclasses import dataclass, field, asdict
from typing import Any
import uuid

from .states import JobState, NodeState

# MongoDB collection names (also the JSON keys in the file store).
COL_JOBS = "jobs"
COL_NODES = "nodes"
COL_AUDIT = "audit"


def new_id(prefix: str) -> str:
    return f"{prefix}-{uuid.uuid4().hex[:12]}"


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
    # scheduling constraints
    node_count: int = 1
    gpu: bool = False                  # true -> must land on a GPU-capable node (the host)
    # I/O contract
    env: dict[str, str] = field(default_factory=dict)  # env vars set inside the container
    output_dir: str = "/out"           # in-container path the job writes results to
    params: dict[str, Any] = field(default_factory=dict)  # opaque passthrough
    # legacy label (kept so old specs still parse)
    workload: str = ""

    @staticmethod
    def from_dict(d: dict) -> "JobSpec":
        cmd = d.get("command", []) or []
        if isinstance(cmd, str):
            cmd = cmd.split()
        return JobSpec(
            name=d["name"],
            image=d.get("image", "") or "",
            command=list(cmd),
            runtime=d.get("runtime", "apptainer"),
            launcher=d.get("launcher", "single") or "single",
            node_count=int(d.get("node_count", 1)),
            gpu=bool(d.get("gpu", False)),
            env={str(k): str(v) for k, v in (d.get("env", {}) or {}).items()},
            output_dir=d.get("output_dir", "/out"),
            params=d.get("params", {}) or {},
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
    updated_at: str | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @staticmethod
    def from_dict(d: dict) -> "NodeRecord":
        # tolerate records written before capability fields existed
        keep = {"node_id", "name", "index", "ip", "state", "owner_job",
                "gpu", "local", "runtime", "updated_at"}
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
