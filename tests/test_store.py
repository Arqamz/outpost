"""FileStore — the persistence layer whose atomicity the node locks rest on.

The concurrency case here is the one that matters: two documented races
(unlocked reads, and releasing the write lock before the write was flushed)
were only ever caught under genuine thread contention, never by a sequential
test.
"""
from __future__ import annotations

import json
import threading

from reconciler.models import JobRecord, JobSpec, NodeRecord
from reconciler.states import NodeState
from reconciler.store import FileStore, open_store


def _node(i: int, gpu: bool = False) -> NodeRecord:
    return NodeRecord(node_id=f"n{i}", name=f"n{i}", index=i, ip=f"10.0.0.{i}", gpu=gpu)


class TestRoundTrip:
    def test_job_put_get_list(self, store):
        job = JobRecord.create(JobSpec(name="j"))
        store.put_job(job)
        assert store.get_job(job.job_id).job_id == job.job_id
        assert [j.job_id for j in store.list_jobs()] == [job.job_id]

    def test_get_missing_returns_none(self, store):
        assert store.get_job("job-nope") is None
        assert store.get_node("nope") is None

    def test_put_stamps_timestamps_and_preserves_created_at(self, store):
        job = JobRecord.create(JobSpec(name="j"))
        store.put_job(job)
        created = store.get_job(job.job_id).created_at
        assert created
        job.state = "provisioning"
        store.put_job(job)
        assert store.get_job(job.job_id).created_at == created

    def test_delete_node_reports_whether_it_existed(self, store):
        store.put_node(_node(1))
        assert store.delete_node("n1") is True
        assert store.delete_node("n1") is False


class TestClaim:
    def test_gpu_and_cpu_pools_do_not_cross(self, store):
        store.put_node(_node(1, gpu=True))
        store.put_node(_node(2))
        gpu = store.claim_node("job-a", require_gpu=True)
        cpu = store.claim_node("job-b", require_gpu=False)
        assert (gpu.node_id, cpu.node_id) == ("n1", "n2")

    def test_cpu_request_never_takes_the_gpu_node(self, store):
        # A CPU job squatting the single GPU node would starve every GPU job.
        store.put_node(_node(1, gpu=True))
        assert store.claim_node("job-a", require_gpu=False) is None

    def test_claim_marks_ownership(self, store):
        store.put_node(_node(1))
        claimed = store.claim_node("job-a")
        assert (claimed.state, claimed.owner_job) == (NodeState.CLAIMED.value, "job-a")
        assert store.get_node("n1").owner_job == "job-a"

    def test_exhausted_pool_returns_none(self, store):
        store.put_node(_node(1))
        store.claim_node("job-a")
        assert store.claim_node("job-b") is None

    def test_concurrent_claims_never_double_assign(self, store):
        # 24 threads racing for 8 nodes: exactly 8 win, and no node is handed to
        # two jobs. This is the property the whole scheduler depends on.
        for i in range(1, 9):
            store.put_node(_node(i))
        won: list[tuple[str, str]] = []
        lock = threading.Lock()

        def worker(k: int) -> None:
            node = store.claim_node(f"job-{k}")
            if node is not None:
                with lock:
                    won.append((f"job-{k}", node.node_id))

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(24)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(won) == 8, f"expected 8 winners, got {len(won)}"
        assert len({node_id for _, node_id in won}) == 8, "a node was claimed twice"
        assert all(n.owner_job for n in store.list_nodes())

    def test_concurrent_writes_leave_valid_json(self, store):
        # The atomic-rename path: no reader should ever see a torn file.
        def worker(k: int) -> None:
            for i in range(20):
                store.put_job(JobRecord.create(JobSpec(name=f"j{k}-{i}")))

        threads = [threading.Thread(target=worker, args=(k,)) for k in range(6)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
        json.loads(open(store.path).read())      # parses => never left half-written
        assert len(store.list_jobs()) == 120


class TestResilience:
    def test_zero_byte_state_file_is_rebuilt(self, tmp_path):
        # A full disk once truncated the live file to empty, which then crashed
        # every reader. Self-heal instead of wedging the store.
        path = tmp_path / "reconciler" / "state.json"
        FileStore(str(path))
        path.write_text("")
        store = FileStore(str(path))
        assert store.list_jobs() == [] and store.list_nodes() == []

    def test_lock_lives_on_its_own_inode(self, store):
        # The data file is replaced by rename; a lock held on it would follow the
        # old inode and stop excluding anything.
        assert store.lock_path == store.path + ".lock"


class TestAudit:
    def test_seq_is_monotonic_and_filterable(self, store):
        for i in range(3):
            store.append_audit({"entity_id": "job-1", "ts": "t", "from": "a", "to": f"b{i}"})
        store.append_audit({"entity_id": "job-2", "ts": "t", "from": "a", "to": "b"})
        assert [e["seq"] for e in store.list_audit()] == [1, 2, 3, 4]
        assert len(store.list_audit("job-1")) == 3

    def test_clear_jobs_wipes_history_but_not_the_pool(self, store):
        store.put_node(_node(1))
        store.put_job(JobRecord.create(JobSpec(name="j")))
        store.append_audit({"entity_id": "job-1", "ts": "t", "from": "a", "to": "b"})
        store.clear_jobs()
        assert store.list_jobs() == [] and store.list_audit() == []
        assert len(store.list_nodes()) == 1


class TestBackendSelection:
    def test_open_store_falls_back_to_file_without_mongo(self, tmp_path):
        # conftest clears CLUSTER_MONGO_URI for the session; assert the default.
        assert isinstance(open_store(str(tmp_path / "s" / "state.json")), FileStore)
