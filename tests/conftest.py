"""Shared fixtures: a sandboxed filesystem, a fake provider, and a tick driver.

Everything here runs the control plane IN-PROCESS against a FileStore in a temp
directory with a fake adapter — no libvirt, no ssh, no ansible, no apptainer.
That is the same shape as the manual smoke described in CLAUDE.md, made
repeatable.

THE IMPORT-ORDER RULE (why the env block is above the imports): REPO_ROOT,
LOGS_DIR, RUNS_DIR and DROPZONE are module-level constants computed when
`reconciler.adapter` / `reconciler.reconciler` are first imported. Setting
CLUSTER_ROOT after the import is too late — the constants already point at the
real repo, and a test run would scribble replay logs and drop-zones into the
working tree. conftest.py is loaded before any test module, so doing it here
covers the whole session.
"""
from __future__ import annotations

import atexit
import os
import shutil
import tempfile

_SANDBOX = tempfile.mkdtemp(prefix="outpost-tests-")
os.environ["CLUSTER_ROOT"] = _SANDBOX
os.environ["CLUSTER_DROPZONE"] = os.path.join(_SANDBOX, "dropzone")
# open_store() prefers Mongo whenever this is set; a developer with it exported
# would otherwise have the whole suite silently run against their real database.
os.environ.pop("CLUSTER_MONGO_URI", None)
atexit.register(shutil.rmtree, _SANDBOX, True)

import random  # noqa: E402
import threading  # noqa: E402
import time  # noqa: E402

import pytest  # noqa: E402

from reconciler.adapter import ProviderAdapter  # noqa: E402
from reconciler.models import JobSpec, NodeRecord, RunResult  # noqa: E402
from reconciler.reconciler import Reconciler  # noqa: E402
from reconciler.states import JOB_TERMINAL, JobState, NodeState  # noqa: E402
from reconciler.store import FileStore  # noqa: E402

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


class FakeAdapter(ProviderAdapter):
    """A provider that records what it was asked to do and does nothing else.

    `jitter` makes each phase take a random (seeded, so reproducible) slice of
    time — that is what turns the concurrency test into a real test: without it
    every thread finishes instantly and the interleavings that broke FileStore
    historically never occur.

    `fail_on` names phases that raise, for the failure paths. `exit_code`
    non-zero makes run() report a failed workload without raising, which is the
    other failure shape (the reconciler is what turns it into a job failure).
    """

    name = "fake"

    def __init__(self, jitter: float = 0.0, seed: int = 1234,
                 fail_on: set[str] | None = None, exit_code: int | None = 0):
        self.jitter = jitter
        self._rand = random.Random(seed)
        self.fail_on = set(fail_on or ())
        self.exit_code = exit_code
        self.calls: list[tuple[str, str, tuple[str, ...]]] = []
        self._lock = threading.Lock()

    def _phase(self, phase: str, job_id: str, names: list[str]) -> None:
        with self._lock:
            self.calls.append((phase, job_id, tuple(names)))
            delay = self._rand.uniform(0, self.jitter) if self.jitter else 0.0
        if delay:
            time.sleep(delay)
        if phase in self.fail_on:
            raise RuntimeError(f"injected {phase} failure")

    def phases_for(self, job_id: str) -> list[str]:
        with self._lock:
            return [p for p, jid, _ in self.calls if jid == job_id]

    def provision(self, node, job_id):
        self._phase("provision", job_id, [node.name])

    def deprovision(self, node, job_id):
        self._phase("deprovision", job_id, [node.name])

    def bootstrap(self, nodes, job_id):
        self._phase("bootstrap", job_id, [n.name for n in nodes])

    def run(self, nodes, job_id, spec, plan=None):
        self._phase("run", job_id, [n.name for n in nodes])
        return RunResult(job_id, nodes[0].name, self.exit_code, note="fake run")

    def collect(self, nodes, job_id, spec, dest):
        self._phase("collect", job_id, [n.name for n in nodes])
        os.makedirs(dest, exist_ok=True)
        return dest

    def probe_topology(self, node, job_id):
        self._phase("probe_topology", job_id, [node.name])
        # A synthetic single-GPU/8-core node, deterministic per node name —
        # enough for placement.resolve() to produce a real plan against a
        # multi-node allocation (incl. the shipped one-rank-per-gpu.json
        # example, cores_per_rank=8) without any real ssh/hardware.
        return {
            "probe_version": "1", "hostname": node.name, "scope": "host",
            "allowed_cpus": "0-7", "online_cpus": "0-7",
            "cpus": [{"id": i, "core": i, "socket": 0, "numa": 0} for i in range(8)],
            "numa": [{"id": 0, "cpulist": "0-7", "memory_mib": 65536}],
            "gpus": [{"index": 0, "uuid": f"GPU-fake-{node.name}", "pci_bus_id": "0000:00:00.0",
                     "memory_mib": 16384, "name": "fake-adapter synthetic GPU", "numa": 0}],
            "topo_matrix": "\tGPU0\tCPU Affinity\tNUMA Affinity\nGPU0\t X \t0-7\t0\n",
            "launcher": {"type": "openmpi", "version": "0.0.0-fake"},
        }


@pytest.fixture()
def store(tmp_path):
    return FileStore(str(tmp_path / "reconciler" / "state.json"))


@pytest.fixture()
def seed(store):
    """Populate the pool: `cpu` VM nodes plus optionally the host GPU node.

    Mirrors what `cluster seed-nodes` produces, without shelling out to
    infra/libvirt/lib.sh — the topology script is not what these tests exercise.
    """
    def _seed(cpu: int = 4, gpu: bool = False) -> list[NodeRecord]:
        nodes = [NodeRecord(node_id=f"cluster-node-{i:02d}", name=f"cluster-node-{i:02d}",
                            index=i, ip=f"192.168.71.{10 + i}",
                            state=NodeState.AVAILABLE.value)
                 for i in range(1, cpu + 1)]
        if gpu:
            nodes.append(NodeRecord(node_id="cluster-host", name="cluster-host", index=0,
                                    ip="192.168.71.1", state=NodeState.AVAILABLE.value,
                                    gpu=True, local=True))
        for n in nodes:
            store.put_node(n)
        return nodes
    return _seed


@pytest.fixture()
def make_reconciler(store):
    """A Reconciler wired to one fake adapter for every provider key.

    Passing `adapters` explicitly keeps `execute` out of it entirely: the real
    adapters are never constructed, so no test can reach libvirt or ssh even by
    accident.
    """
    def _make(adapter: FakeAdapter | None = None, **kw) -> tuple[Reconciler, FakeAdapter]:
        adapter = adapter or FakeAdapter()
        rec = Reconciler(store, log=lambda msg: None,
                         adapters={"local": adapter, "libvirt": adapter,
                                   "static-ssh": adapter}, **kw)
        return rec, adapter
    return _make


@pytest.fixture()
def spec():
    def _spec(**kw) -> JobSpec:
        return JobSpec(**{"name": "smoke", "image": "docker://example/x:1",
                          "command": ["echo", "hi"], **kw})
    return _spec


def active(store) -> list:
    """Jobs that have not reached a terminal state."""
    return [j for j in store.list_jobs() if JobState(j.state) not in JOB_TERMINAL]


def drive(rec: Reconciler, max_ticks: int = 200) -> int:
    """Tick until every job is terminal. Returns the tick count.

    Deliberately NOT "until tick() returns 0". tick() returns how many jobs
    MOVED, and a tick moves nothing whenever every remaining job lost a claim
    race — which run_forever treats as sleep-and-retry, not as done. Stopping on
    the first zero would silently call a job finished while it was still waiting
    its turn, and the assertions after it would be inspecting a half-run cluster.

    The cap is a livelock guard: a job that can never advance surfaces here as a
    named assertion rather than a suite that hangs.
    """
    for i in range(1, max_ticks + 1):
        rec.tick()
        if not active(rec.store):
            return i
    stuck = [(j.spec.get("name"), j.state) for j in active(rec.store)]
    raise AssertionError(f"jobs still active after {max_ticks} ticks: {stuck}")


def states(store) -> dict[str, str]:
    return {j.job_id: j.state for j in store.list_jobs()}


def assert_pool_returned(store) -> None:
    """Every node is back in the pool, unowned. The invariant that catches a
    phase that claimed a node and forgot to release it — which strands capacity
    permanently, with no error anywhere."""
    for n in store.list_nodes():
        assert n.state in (NodeState.AVAILABLE.value, NodeState.QUARANTINED.value), \
            f"{n.name} left in {n.state}"
        if n.state == NodeState.AVAILABLE.value:
            assert n.owner_job is None, f"{n.name} available but still owned by {n.owner_job}"


__all__ = ["FakeAdapter", "JobState", "NodeState", "REPO_DIR",
           "assert_pool_returned", "drive", "states"]
