"""Persistence for jobs / nodes / audit.

Two interchangeable backends behind one interface:
  * FileStore  — JSON file + fcntl lock. Default; zero infra, survives across
                 CLI invocations, gives real cross-process atomic node claims.
  * MongoStore — used when CLUSTER_MONGO_URI is set and pymongo is importable.

The atomic `claim_node` primitive is what makes NodeRegistry's exclusive locks
correct under concurrency.
"""
from __future__ import annotations
import json
import os
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterable

from .models import COL_JOBS, COL_NODES, COL_AUDIT, JobRecord, NodeRecord
from .states import JobState, NodeState


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


class Store:
    """Interface. All mutations stamp updated_at and are individually durable."""

    # jobs -----------------------------------------------------------------
    def put_job(self, job: JobRecord) -> None: raise NotImplementedError
    def get_job(self, job_id: str) -> JobRecord | None: raise NotImplementedError
    def list_jobs(self) -> list[JobRecord]: raise NotImplementedError

    # nodes ----------------------------------------------------------------
    def put_node(self, node: NodeRecord) -> None: raise NotImplementedError
    def get_node(self, node_id: str) -> NodeRecord | None: raise NotImplementedError
    def list_nodes(self) -> list[NodeRecord]: raise NotImplementedError
    def delete_node(self, node_id: str) -> bool: raise NotImplementedError  # True if it existed

    def claim_node(self, job_id: str, require_gpu: bool = False,
                   provider: str | None = None,
                   kube_context: str | None = None) -> NodeRecord | None:
        """Atomically move one AVAILABLE node -> CLAIMED(owner=job_id).

        provider set (e.g. "k8s") -> match ONLY nodes on that backend, ignoring
        require_gpu (a k8s slot is picked by backend, not by the gpu flag). This
        is how a `backend: k8s` job claims a k8s slot instead of a VM/host.

        kube_context set (only meaningful with provider="k8s") -> match ONLY
        slots on that cluster, so a job can target the L20 vs the 5060 Ti when
        several k8s clusters are registered. None -> any slot of the backend.

        provider=None (default, the VM/host/static pool) -> require_gpu=True
        matches only GPU-capable nodes; False matches only non-GPU nodes (so CPU
        jobs never squat the host GPU) AND backend nodes (provider "k8s") are
        excluded, so a normal job never grabs a k8s slot. None if none match.
        """
        raise NotImplementedError

    @staticmethod
    def _node_provider(nd: dict) -> str:
        """Effective provider of a stored node dict — mirrors NodeRecord.adapter_key
        so claim_node can filter by backend without rehydrating a NodeRecord."""
        return nd.get("provider") or ("local" if nd.get("local") else "libvirt")

    # audit ----------------------------------------------------------------
    def append_audit(self, entry: dict) -> None: raise NotImplementedError
    def list_audit(self, entity_id: str | None = None) -> list[dict]: raise NotImplementedError

    # maintenance ------------------------------------------------------------
    def clear_jobs(self) -> None:
        """Wipe job history + audit trail. Nodes are untouched here — callers
        that also want the pool reset to available do that separately (see
        reconciler.clear_jobs), since that's a scheduling decision, not a
        storage primitive."""
        raise NotImplementedError


# ── file-backed store ──────────────────────────────────────────────────────
try:
    import fcntl  # POSIX; present on Linux
except ImportError:  # pragma: no cover
    fcntl = None


class FileStore(Store):
    def __init__(self, path: str):
        self.path = path
        # Lock lives on its OWN stable file, never on the data file: the data
        # file is replaced atomically (see _atomic_write), which would swap the
        # inode a flock is held on. A dedicated lock inode keeps the exclusive
        # read-modify-write correct across the rename and across processes.
        self.lock_path = path + ".lock"
        os.makedirs(os.path.dirname(path), exist_ok=True)
        # Recreate on a missing OR zero-byte file: a full disk (ENOSPC) used to
        # truncate the data file to empty mid-write, which then crashed every
        # reader with a JSONDecodeError. Self-heal instead of wedging the store.
        if not os.path.exists(path) or os.path.getsize(path) == 0:
            self._write({COL_JOBS: {}, COL_NODES: {}, COL_AUDIT: []})

    def _atomic_write(self, data: dict) -> None:
        # Write a temp file in the same dir, fsync, then os.replace() over the
        # target — an atomic rename. If serialization or the write fails (e.g.
        # ENOSPC), the live file is never touched, so it can't be left empty or
        # half-written; the temp is discarded. This is what makes the store
        # crash-safe on a full disk.
        d = os.path.dirname(self.path) or "."
        fd, tmp = tempfile.mkstemp(dir=d, prefix=".state-", suffix=".tmp")
        try:
            with os.fdopen(fd, "w") as f:
                json.dump(data, f, indent=2)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.path)
        except BaseException:
            try:
                os.unlink(tmp)
            except OSError:
                pass
            raise

    def _load(self) -> dict:
        with open(self.path) as f:
            return json.load(f)

    @contextmanager
    def _locked(self):
        # Exclusive advisory lock around a read-modify-write; this is what gives
        # claim_node its atomicity across separate CLI processes/threads.
        lf = open(self.lock_path, "w")
        try:
            if fcntl:
                fcntl.flock(lf, fcntl.LOCK_EX)
            data = self._load()
            yield data, lf
            self._atomic_write(data)
        finally:
            if fcntl:
                fcntl.flock(lf, fcntl.LOCK_UN)
            lf.close()

    def _read(self) -> dict:
        # Shared lock: blocks only while a writer holds _locked()'s exclusive
        # lock, so this never observes the file mid-write (and os.replace is
        # atomic regardless). Concurrent readers don't block each other. Matters
        # now that reconciler.tick() advances jobs from a thread pool.
        lf = open(self.lock_path, "w")
        try:
            if fcntl:
                fcntl.flock(lf, fcntl.LOCK_SH)
            return self._load()
        finally:
            if fcntl:
                fcntl.flock(lf, fcntl.LOCK_UN)
            lf.close()

    def _write(self, data: dict) -> None:
        self._atomic_write(data)

    def put_job(self, job: JobRecord) -> None:
        job.updated_at = now_iso()
        job.created_at = job.created_at or job.updated_at
        with self._locked() as (d, _):
            d[COL_JOBS][job.job_id] = job.to_dict()

    def get_job(self, job_id):
        d = self._read().get(COL_JOBS, {}).get(job_id)
        return JobRecord.from_dict(d) if d else None

    def list_jobs(self):
        return [JobRecord.from_dict(v) for v in self._read().get(COL_JOBS, {}).values()]

    def put_node(self, node: NodeRecord) -> None:
        node.updated_at = now_iso()
        with self._locked() as (d, _):
            d[COL_NODES][node.node_id] = node.to_dict()

    def get_node(self, node_id):
        d = self._read().get(COL_NODES, {}).get(node_id)
        return NodeRecord.from_dict(d) if d else None

    def list_nodes(self):
        return [NodeRecord.from_dict(v) for v in self._read().get(COL_NODES, {}).values()]

    def delete_node(self, node_id: str) -> bool:
        with self._locked() as (d, _):
            return d[COL_NODES].pop(node_id, None) is not None

    def claim_node(self, job_id, require_gpu=False, provider=None, kube_context=None):
        with self._locked() as (d, _):
            for nid, nd in d[COL_NODES].items():
                if nd["state"] != NodeState.AVAILABLE.value:
                    continue
                np = self._node_provider(nd)
                if provider is not None:
                    if np != provider:
                        continue
                    if kube_context is not None and nd.get("kube_context", "") != kube_context:
                        continue
                else:
                    if np == "k8s":            # backend slot, only via explicit provider
                        continue
                    if bool(nd.get("gpu", False)) != bool(require_gpu):
                        continue
                nd["state"] = NodeState.CLAIMED.value
                nd["owner_job"] = job_id
                nd["updated_at"] = now_iso()
                return NodeRecord.from_dict(nd)
            return None

    def append_audit(self, entry):
        with self._locked() as (d, _):
            entry["seq"] = len(d[COL_AUDIT]) + 1
            d[COL_AUDIT].append(entry)

    def list_audit(self, entity_id=None):
        rows = self._read().get(COL_AUDIT, [])
        return [r for r in rows if entity_id is None or r.get("entity_id") == entity_id]

    def clear_jobs(self) -> None:
        with self._locked() as (d, _):
            d[COL_JOBS] = {}
            d[COL_AUDIT] = []


# ── mongo-backed store ─────────────────────────────────────────────────────
class MongoStore(Store):
    def __init__(self, uri: str, db: str = "cluster"):
        from pymongo import MongoClient  # imported lazily; only needed here
        self.cli = MongoClient(uri)
        self.db = self.cli[db]
        self.jobs = self.db[COL_JOBS]
        self.nodes = self.db[COL_NODES]
        self.audit = self.db[COL_AUDIT]

    def put_job(self, job):
        job.updated_at = now_iso(); job.created_at = job.created_at or job.updated_at
        self.jobs.replace_one({"job_id": job.job_id}, job.to_dict(), upsert=True)

    def get_job(self, job_id):
        d = self.jobs.find_one({"job_id": job_id}); return JobRecord.from_dict(d) if d else None

    def list_jobs(self):
        return [JobRecord.from_dict(d) for d in self.jobs.find()]

    def put_node(self, node):
        node.updated_at = now_iso()
        self.nodes.replace_one({"node_id": node.node_id}, node.to_dict(), upsert=True)

    def get_node(self, node_id):
        d = self.nodes.find_one({"node_id": node_id}); return NodeRecord.from_dict(d) if d else None

    def list_nodes(self):
        return [NodeRecord.from_dict(d) for d in self.nodes.find()]

    def delete_node(self, node_id: str) -> bool:
        return self.nodes.delete_one({"node_id": node_id}).deleted_count > 0

    def claim_node(self, job_id, require_gpu=False, provider=None, kube_context=None):
        # find_one_and_update is atomic server-side -> exclusive lock.
        if provider is not None:
            q = {"state": NodeState.AVAILABLE.value, "provider": provider}
            if kube_context is not None:
                q["kube_context"] = kube_context
        else:
            # A normal claim never grabs a backend slot (provider "k8s"); "" and
            # unset both mean the VM/host/static pool, so $nin covers them.
            q = {"state": NodeState.AVAILABLE.value, "gpu": bool(require_gpu),
                 "provider": {"$nin": ["k8s"]}}
        d = self.nodes.find_one_and_update(
            q,
            {"$set": {"state": NodeState.CLAIMED.value, "owner_job": job_id, "updated_at": now_iso()}},
            return_document=True,  # ReturnDocument.AFTER
        )
        return NodeRecord.from_dict(d) if d else None

    def append_audit(self, entry):
        entry["seq"] = self.audit.count_documents({}) + 1
        self.audit.insert_one(entry)

    def list_audit(self, entity_id=None):
        q = {} if entity_id is None else {"entity_id": entity_id}
        return [{k: v for k, v in d.items() if k != "_id"} for d in self.audit.find(q).sort("seq", 1)]

    def clear_jobs(self) -> None:
        self.jobs.delete_many({})
        self.audit.delete_many({})


def open_store(default_file: str) -> Store:
    """Pick a backend: Mongo if CLUSTER_MONGO_URI set + pymongo available, else file."""
    uri = os.environ.get("CLUSTER_MONGO_URI")
    if uri:
        try:
            return MongoStore(uri)
        except Exception as e:  # noqa: BLE001 — fall back loudly
            print(f"[reconciler] CLUSTER_MONGO_URI set but Mongo unavailable ({e}); using file store")
    return FileStore(default_file)
