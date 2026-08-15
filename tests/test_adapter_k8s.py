"""KubernetesAdapter: manifest rendering + GPU-memory affinity declare/verify.

_render_manifests and _build_gpu_memory_receipt are both pure (no cluster
calls, confirmed in their own docstrings) — every test here runs with no
kubectl, no KinD, no real cluster.
"""
from __future__ import annotations

import json

import pytest
import yaml
from reconciler.adapter import GPU_MEMORY_PROBE_PATH, KubernetesAdapter
from reconciler.models import JobSpec

MARKER = "===GPU_MEMORY_PROBE==="


def spec(**kw) -> JobSpec:
    base = dict(name="gpu-mem-smoke", image="nvidia/cuda:12.6.3-base-ubuntu24.04",
               command=["python3", "train.py"], backend="k8s")
    return JobSpec(**{**base, **kw})


def pods_from(manifest_yaml: str) -> list[dict]:
    return [d for d in yaml.safe_load_all(manifest_yaml) if d and d.get("kind") == "Pod"]


@pytest.fixture()
def adapter():
    return KubernetesAdapter(log=lambda msg: None)


class TestGpuMemForRank:
    def test_default_is_the_shared_constant(self, adapter):
        from reconciler.adapter import K8S_GPU_MEMORY_MB
        assert adapter._gpu_mem_for_rank(spec(), 0, 2) == str(K8S_GPU_MEMORY_MB)

    def test_scalar_override_applies_to_every_rank(self, adapter):
        s = spec(params={"gpu_memory_mb": 4096})
        assert adapter._gpu_mem_for_rank(s, 0, 3) == "4096"
        assert adapter._gpu_mem_for_rank(s, 2, 3) == "4096"

    def test_list_gives_each_rank_its_own_value(self, adapter):
        s = spec(params={"gpu_memory_mb": [1024, 2048]})
        assert adapter._gpu_mem_for_rank(s, 0, 2) == "1024"
        assert adapter._gpu_mem_for_rank(s, 1, 2) == "2048"

    def test_list_length_mismatch_is_refused(self, adapter):
        s = spec(params={"gpu_memory_mb": [1024, 2048]})
        with pytest.raises(ValueError, match="one entry per rank"):
            adapter._gpu_mem_for_rank(s, 0, 3)


class TestRenderManifestsDefaults:
    def test_no_opt_in_leaves_the_command_untouched(self, adapter):
        # No behaviour change without opting in: verify_gpu_memory unset must
        # produce the exact command the job declared, byte-for-byte.
        s = spec()
        pods = pods_from(adapter._render_manifests("job-x", s, 2))
        for pod in pods:
            assert pod["spec"]["containers"][0]["command"] == ["python3", "train.py"]

    def test_per_rank_gpu_memory_annotation(self, adapter):
        s = spec(params={"gpu_memory_mb": [1024, 2048]})
        pods = pods_from(adapter._render_manifests("job-x", s, 2))
        assert pods[0]["metadata"]["annotations"]["gpu-memory"] == "1024"
        assert pods[1]["metadata"]["annotations"]["gpu-memory"] == "2048"


class TestVerifyGpuMemoryProbeInjection:
    def test_opt_in_wraps_the_command_with_the_probe(self, adapter):
        s = spec(params={"verify_gpu_memory": True})
        pods = pods_from(adapter._render_manifests("job-x", s, 1))
        command = pods[0]["spec"]["containers"][0]["command"]
        assert command[:2] == ["sh", "-c"]
        assert MARKER in command[2]
        assert "exec python3 train.py" in command[2]
        assert GPU_MEMORY_PROBE_PATH.read_text().splitlines()[0] in command[2]

    def test_opt_in_without_a_command_is_refused(self, adapter):
        s = spec(command=[], params={"verify_gpu_memory": True})
        with pytest.raises(ValueError, match="verify_gpu_memory needs spec.command"):
            adapter._render_manifests("job-x", s, 1)


class TestGpuMemoryReceipt:
    def test_matched_and_mismatched_ranks(self, adapter, tmp_path):
        workdir = str(tmp_path)
        # rank 0: probe reports exactly what was declared.
        (tmp_path / "rank-0.log").write_text(
            "some workload output\n"
            f'{MARKER}{{"probe_version":"1","rank":0,"hostname":"rank-0",'
            '"cuda_device_memory_limit_mb":1024,"gpu_uuid":"GPU-abc","gpu_total_mb":16384}\n')
        # rank 1: probe reports something OTHER than declared -> mismatch.
        (tmp_path / "rank-1.log").write_text(
            f'{MARKER}{{"probe_version":"1","rank":1,"hostname":"rank-1",'
            '"cuda_device_memory_limit_mb":9999,"gpu_uuid":"GPU-abc","gpu_total_mb":16384}\n')

        s = spec(params={"gpu_memory_mb": [1024, 2048]})
        adapter._build_gpu_memory_receipt(workdir, 2, s)

        receipt = yaml.safe_load((tmp_path / "gpu-memory-receipt.yaml").read_text())
        r0, r1 = receipt["ranks"]
        assert r0 == {"rank": 0, "declared_mb": 1024, "observed_mb": 1024,
                      "observed_gpu_uuid": "GPU-abc", "matched": True}
        assert r1 == {"rank": 1, "declared_mb": 2048, "observed_mb": 9999,
                      "observed_gpu_uuid": "GPU-abc", "matched": False}

    def test_missing_log_is_recorded_as_unmatched_not_skipped(self, adapter, tmp_path):
        # No rank-0.log at all — e.g. the pod never wrote one. Must show up as
        # an explicit gap, never silently absent from the receipt.
        from reconciler.adapter import K8S_GPU_MEMORY_MB
        adapter._build_gpu_memory_receipt(str(tmp_path), 1, spec())
        receipt = yaml.safe_load((tmp_path / "gpu-memory-receipt.yaml").read_text())
        assert receipt["ranks"] == [
            {"rank": 0, "declared_mb": K8S_GPU_MEMORY_MB,
             "observed_mb": None, "observed_gpu_uuid": None, "matched": False}]

    def test_unparseable_marker_line_is_unmatched(self, adapter, tmp_path):
        (tmp_path / "rank-0.log").write_text(f"{MARKER}not valid json at all\n")
        adapter._build_gpu_memory_receipt(str(tmp_path), 1, spec())
        receipt = yaml.safe_load((tmp_path / "gpu-memory-receipt.yaml").read_text())
        assert receipt["ranks"][0]["observed_mb"] is None
        assert receipt["ranks"][0]["matched"] is False


class TestGpuMemoryProbeScript:
    """Real execution of gpu-memory-probe.sh — no mocking, matches the
    convention `tests/test_probe.py`-style probes use in this repo. Guards
    the exact live-hardware bug found this session: kai-resource-isolator's
    injected CUDA_DEVICE_MEMORY_LIMIT carries a trailing unit letter
    ("1141m"), which a naive numeric check rejects outright rather than
    stripping."""

    def _run(self, env):
        import os
        import subprocess
        full_env = {**os.environ, **env}
        proc = subprocess.run(["sh", "-c", GPU_MEMORY_PROBE_PATH.read_text()],
                              capture_output=True, text=True, env=full_env)
        assert proc.returncode == 0, proc.stderr
        line = next(l for l in proc.stdout.splitlines() if l.startswith(MARKER))
        return json.loads(line[len(MARKER):])

    def test_unit_suffixed_value_is_stripped_not_rejected(self):
        doc = self._run({"CUDA_DEVICE_MEMORY_LIMIT": "1141m", "RANK": "0"})
        assert doc["cuda_device_memory_limit_mb"] == 1141

    def test_bare_integer_still_works(self):
        doc = self._run({"CUDA_DEVICE_MEMORY_LIMIT": "2048", "RANK": "1"})
        assert doc["cuda_device_memory_limit_mb"] == 2048

    def test_absent_is_null_not_a_guess(self):
        doc = self._run({"RANK": "0"})
        assert doc["cuda_device_memory_limit_mb"] is None


class TestRenderManifestsDedicatedGpuMode:
    """gpu_mode="dedicated": real per-node GPUs via a plain nvidia.com/gpu
    resource request, no KAI/HAMi at all. The "shared" default (every existing
    test above) must keep working byte-for-byte — this class only covers the
    new opt-in branch."""

    def test_no_podgroup_doc_is_rendered(self, adapter):
        s = spec(params={"gpu_mode": "dedicated"})
        docs = list(yaml.safe_load_all(adapter._render_manifests("job-x", s, 2)))
        assert not any(d and d.get("kind") == "PodGroup" for d in docs)

    def test_no_kai_hami_fields_on_the_pod(self, adapter):
        s = spec(params={"gpu_mode": "dedicated"})
        pods = pods_from(adapter._render_manifests("job-x", s, 1))
        pod = pods[0]
        assert "annotations" not in pod["metadata"]
        assert "kai.scheduler/queue" not in pod["metadata"]["labels"]
        assert "schedulerName" not in pod["spec"]

    def test_default_gpu_count_is_one(self, adapter):
        from reconciler.adapter import K8S_GPU_COUNT
        s = spec(params={"gpu_mode": "dedicated"})
        pods = pods_from(adapter._render_manifests("job-x", s, 1))
        resources = pods[0]["spec"]["containers"][0]["resources"]
        assert resources["requests"]["nvidia.com/gpu"] == str(K8S_GPU_COUNT)
        assert resources["limits"]["nvidia.com/gpu"] == str(K8S_GPU_COUNT)

    def test_gpu_count_override_applies_to_every_rank(self, adapter):
        s = spec(params={"gpu_mode": "dedicated", "gpu_count": 4})
        pods = pods_from(adapter._render_manifests("job-x", s, 2))
        for pod in pods:
            resources = pod["spec"]["containers"][0]["resources"]
            assert resources["requests"]["nvidia.com/gpu"] == "4"
            assert resources["limits"]["nvidia.com/gpu"] == "4"

    def test_no_nccl_p2p_disable_default(self, adapter):
        # "shared" mode disables NCCL P2P by default (ranks time-slice one
        # physical GPU) - meaningless with real distinct per-node GPUs, so
        # dedicated mode must not carry that default forward.
        s = spec(params={"gpu_mode": "dedicated"})
        pods = pods_from(adapter._render_manifests("job-x", s, 1))
        env_names = {e["name"] for e in pods[0]["spec"]["containers"][0]["env"]}
        assert "NCCL_P2P_DISABLE" not in env_names

    def test_rendezvous_env_still_present(self, adapter):
        # The genuinely backend-agnostic mechanism (headless Service +
        # RANK/WORLD_SIZE/MASTER_ADDR/MASTER_PORT) must survive in this mode too.
        s = spec(params={"gpu_mode": "dedicated"})
        docs = list(yaml.safe_load_all(adapter._render_manifests("job-x", s, 2)))
        assert any(d and d.get("kind") == "Service" for d in docs)
        pods = pods_from(adapter._render_manifests("job-x", s, 2))
        env0 = {e["name"]: e["value"] for e in pods[0]["spec"]["containers"][0]["env"]}
        assert env0["RANK"] == "0"
        assert env0["WORLD_SIZE"] == "2"
        assert env0["MASTER_ADDR"].startswith("rank-0.")

    def test_shared_mode_is_unaffected_by_the_new_branch(self, adapter):
        # Regression guard: gpu_mode unset (or explicitly "shared") must still
        # render the exact original shape.
        s = spec(params={"gpu_memory_mb": [1024, 2048]})
        pods = pods_from(adapter._render_manifests("job-x", s, 2))
        assert pods[0]["metadata"]["annotations"]["gpu-memory"] == "1024"
        assert pods[0]["spec"]["schedulerName"] == "kai-scheduler"
        assert "resources" not in pods[0]["spec"]["containers"][0]
