"""LibvirtAdapter._ensure_remote_sif — the shared, digest-keyed remote SIF
cache that replaced always-scp-into-the-per-job-workdir.

Three paths: a warm cache (no transfer at all), a cold cache with enough
remote disk headroom (direct remote pull, dodging the slow SSM-tunnelled
scp), and a cold cache without enough headroom (scp fallback, same safety
guarantee as before — just landing in the shared cache instead of a per-job
dir). No real ssh/apptainer here — subprocess.run/run_logged/_scp_to/
ensure_local_sif are all faked.
"""
from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from reconciler.adapter import REMOTE_SIF_CACHE_DIR, LibvirtAdapter, _sif_cache_name
from reconciler.models import NodeRecord

IMAGE = "docker://nvcr.io/nvidia/hpc-benchmarks@sha256:" + "ab" * 32


def node() -> NodeRecord:
    return NodeRecord(node_id="n1", name="n1", index=1, ip="10.0.0.1")


@pytest.fixture()
def adapter():
    return LibvirtAdapter(log=lambda msg: None)


def _cache_name() -> str:
    return _sif_cache_name(IMAGE)


class TestWarmCache:
    def test_cache_hit_does_no_transfer_at_all(self, adapter, monkeypatch):
        probe = MagicMock(returncode=0)
        monkeypatch.setattr("reconciler.adapter.subprocess.run", lambda *a, **kw: probe)
        ensure_calls = []
        monkeypatch.setattr("reconciler.adapter.ensure_local_sif",
                            lambda *a, **kw: ensure_calls.append(True))
        run_logged_calls = []
        monkeypatch.setattr("reconciler.adapter.run_logged",
                            lambda *a, **kw: run_logged_calls.append(a) or 0)
        scp_calls = []
        monkeypatch.setattr(adapter, "_scp_to", lambda *a, **kw: scp_calls.append(True))

        result = adapter._ensure_remote_sif(node(), IMAGE, "job-1")

        assert result == f"{REMOTE_SIF_CACHE_DIR}/{_cache_name()}"
        assert ensure_calls == []  # never touched the local cache — no local build needed
        assert run_logged_calls == []
        assert scp_calls == []


class TestColdCacheWithHeadroom:
    def test_sufficient_headroom_pulls_remotely_not_via_scp(self, adapter, monkeypatch, tmp_path):
        probe = MagicMock(returncode=1)  # cache miss
        local_sif = tmp_path / "cached.sif"
        local_sif.write_bytes(b"x" * 1000)  # a real (tiny) file — only its path matters here
        monkeypatch.setattr("reconciler.adapter.os.path.getsize",
                            lambda p: 5 * 1024 ** 3 if p == str(local_sif) else 1000)  # "5GiB" SIF

        def fake_subprocess_run(argv, **kw):
            if "df --output=avail" in argv[-1]:
                return MagicMock(stdout="999999999999\n")  # ~1TB free — plenty of headroom
            return probe
        monkeypatch.setattr("reconciler.adapter.subprocess.run", fake_subprocess_run)
        monkeypatch.setattr("reconciler.adapter.ensure_local_sif", lambda *a, **kw: str(local_sif))
        run_logged_calls = []
        monkeypatch.setattr("reconciler.adapter.run_logged",
                            lambda argv, *a, **kw: run_logged_calls.append(argv) or 0)
        scp_calls = []
        monkeypatch.setattr(adapter, "_scp_to", lambda *a, **kw: scp_calls.append(True))

        result = adapter._ensure_remote_sif(node(), IMAGE, "job-1")

        assert result == f"{REMOTE_SIF_CACHE_DIR}/{_cache_name()}"
        assert scp_calls == []  # never fell back to scp
        assert len(run_logged_calls) == 1
        remote_cmd = run_logged_calls[0][-1]
        assert "apptainer build" in remote_cmd
        assert "flock" in remote_cmd  # serializes a cold-cache stampede, same as the local cache


class TestColdCacheWithoutHeadroom:
    def test_insufficient_headroom_falls_back_to_scp(self, adapter, monkeypatch, tmp_path):
        probe = MagicMock(returncode=1)  # cache miss
        local_sif = tmp_path / "cached.sif"
        local_sif.write_bytes(b"x" * 1000)  # a real (tiny) file — size is faked below
        monkeypatch.setattr("reconciler.adapter.os.path.getsize",
                            lambda p: 10 * 1024 ** 3 if p == str(local_sif) else 1000)  # "10GiB" SIF

        def fake_subprocess_run(argv, **kw):
            if "df --output=avail" in argv[-1]:
                return MagicMock(stdout="1000000000\n")  # ~1GB free — nowhere near 3x10GiB
            return probe
        monkeypatch.setattr("reconciler.adapter.subprocess.run", fake_subprocess_run)
        monkeypatch.setattr("reconciler.adapter.ensure_local_sif", lambda *a, **kw: str(local_sif))
        run_logged_calls = []
        monkeypatch.setattr("reconciler.adapter.run_logged",
                            lambda argv, *a, **kw: run_logged_calls.append(argv) or 0)
        scp_calls = []
        monkeypatch.setattr(adapter, "_scp_to",
                            lambda node_, src, dst, job_id: scp_calls.append((src, dst)))

        result = adapter._ensure_remote_sif(node(), IMAGE, "job-1")

        assert result == f"{REMOTE_SIF_CACHE_DIR}/{_cache_name()}"
        assert len(scp_calls) == 1
        assert scp_calls[0][0] == str(local_sif)
        assert "apptainer build" not in " ".join(str(c) for c in run_logged_calls)
        # the placement command still lands it under the shared cache, not remote_workdir
        assert any(REMOTE_SIF_CACHE_DIR in c[-1] for c in run_logged_calls)
