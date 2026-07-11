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

    def claim_node(self, job_id: str, require_gpu: bool = False) -> NodeRecord | None:
        """Atomically move one AVAILABLE node -> CLAIMED(owner=job_id).

        require_gpu=True matches only GPU-capable nodes; False matches only
        non-GPU nodes (so CPU jobs never squat the host GPU). None if none match.
        """
        raise NotImplementedError

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
        os.makedirs(os.path.dirname(path), exist_ok=True)
        if not os.path.exists(path):
            self._write({COL_JOBS: {}, COL_NODES: {}, COL_AUDIT: []})

    @contextmanager
    def _locked(self):
        # Exclusive advisory lock around a read-modify-write; this is what gives
        # claim_node its atomicity across separate CLI processes/threads.
        f = open(self.path, "r+")
        try:
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_EX)
            f.seek(0)
            data = json.load(f)
            yield data, f
            f.seek(0); f.truncate()
            json.dump(data, f, indent=2)
            f.flush()   # push the write out BEFORE unlocking (see below), else a
            os.fsync(f.fileno())  # waiting reader/writer can acquire the lock the
        finally:                 # instant we unlock and still see stale/truncated
            # content the OS hasn't made visible yet. Unlock (and close, which
            # would implicitly unlock anyway) only after the flush lands, so
            # nothing that was waiting on this lock can observe a half-write.
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_UN)
            f.close()

    def _read(self) -> dict:
        # Shared lock: blocks only while a writer holds _locked()'s exclusive
        # lock, so this can never observe the file mid truncate-then-rewrite.
        # Concurrent readers don't block each other (LOCK_SH is shared). This
        # matters once callers aren't strictly serial — reconciler.tick() now
        # advances jobs from a thread pool, so unlocked reads here used to be
        # able to land exactly between _locked()'s truncate() and its write.
        with open(self.path) as f:
            if fcntl:
                fcntl.flock(f, fcntl.LOCK_SH)
            try:
                return json.load(f)
            finally:
                if fcntl:
                    fcntl.flock(f, fcntl.LOCK_UN)

    def _write(self, data: dict) -> None:
        with open(self.path, "w") as f:
            json.dump(data, f, indent=2)

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

    def claim_node(self, job_id, require_gpu=False):
        with self._locked() as (d, _):
            for nid, nd in d[COL_NODES].items():
                if nd["state"] != NodeState.AVAILABLE.value:
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

    def claim_node(self, job_id, require_gpu=False):
        # find_one_and_update is atomic server-side -> exclusive lock.
        d = self.nodes.find_one_and_update(
            {"state": NodeState.AVAILABLE.value, "gpu": bool(require_gpu)},
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
