"""SlinkyAdapter: the pure pieces (image refs, the sbatch script, Slurm job
parsing), collect's tar filtering, and how the reconciler routes a job onto a
slinky slot. Nothing here reaches kubectl or a real cluster."""
from __future__ import annotations

import json
import os

import pytest

from reconciler.adapter import (SlinkyAdapter, parse_slurm_job, render_sbatch,
                                slinky_image_ref, slinky_squashfs_name)
from reconciler.models import JobSpec, NodeRecord
from reconciler.states import JobState, NodeState

from conftest import REPO_DIR, FakeAdapter, assert_pool_returned, drive

DIGEST = "sha256:" + "ab" * 32


class TestImageRef:
    def test_registry_is_split_with_a_hash(self):
        assert (slinky_image_ref(f"docker://nvcr.io/nvidia/hpc-benchmarks@{DIGEST}")
                == f"docker://nvcr.io#nvidia/hpc-benchmarks@{DIGEST}")

    def test_registry_with_port(self):
        assert slinky_image_ref("localhost:5000/x:1") == "docker://localhost:5000#x:1"

    def test_docker_hub_ref_has_no_registry(self):
        assert slinky_image_ref("ubuntu:24.04") == "docker://ubuntu:24.04"
        assert slinky_image_ref("docker://library/ubuntu:24.04") == "docker://library/ubuntu:24.04"

    def test_squashfs_path_passes_through(self):
        assert slinky_image_ref("/scratch/img.sqsh") == "/scratch/img.sqsh"

    @pytest.mark.parametrize("bad", ["oras://x/y:1", "x.sif"])
    def test_sif_and_other_schemes_are_refused(self, bad):
        with pytest.raises(RuntimeError, match="pyxis"):
            slinky_image_ref(bad)

    def test_cache_name_is_the_digest(self):
        assert slinky_squashfs_name(f"docker://nvcr.io/a/b@{DIGEST}") == "ab" * 32 + ".sqsh"

    def test_cache_name_without_digest_is_stable(self):
        a = slinky_squashfs_name("docker://ubuntu:24.04")
        assert a == slinky_squashfs_name("docker://ubuntu:24.04") and a.endswith(".sqsh")


def spec(**kw) -> JobSpec:
    return JobSpec(**{"name": "stream", "image": f"docker://nvcr.io/nvidia/x@{DIGEST}",
                      "command": ["bash", "-c", "echo hi > /out/r.txt"], "gpu": True, **kw})


class TestSbatch:
    def test_single_gpu_job(self):
        s = render_sbatch("job-abc", spec(env={"A": "1 2"}), "/scratch/outpost/job-abc",
                          "/scratch/outpost/.images/x.sqsh")
        assert "#SBATCH --job-name=outpost-job-abc" in s
        assert "#SBATCH --nodes=1" in s
        assert "#SBATCH --gres=gpu:1" in s
        assert "#SBATCH --output=/scratch/outpost/job-abc/stdout.log" in s
        assert "#SBATCH --partition" not in s
        assert "export A='1 2'" in s
        assert "export NVIDIA_VISIBLE_DEVICES=all" in s
        srun = s.strip().splitlines()[-1]
        assert srun.startswith("srun --ntasks=1 --container-image=/scratch/outpost/.images/x.sqsh")
        assert "--mpi" not in srun
        assert "--container-mounts=/scratch/outpost/job-abc:/out" in srun
        assert srun.endswith("bash -c 'echo hi > /out/r.txt'")

    def test_cpu_job_asks_for_no_gres(self):
        s = render_sbatch("job-abc", spec(gpu=False), "/w", "/i.sqsh", partition="main")
        assert "--gres" not in s and "NVIDIA_VISIBLE_DEVICES" not in s
        assert "#SBATCH --partition=main" in s

    def test_mpi_job_runs_one_task_per_node(self):
        s = render_sbatch("job-abc", spec(launcher="mpi", node_count=2,
                                          params={"gpu_count": 1, "time_limit": "00:30:00"}),
                          "/w", "/i.sqsh", mpi="pmi2")
        assert "#SBATCH --nodes=2" in s and "#SBATCH --time=00:30:00" in s
        assert "export WORLD_SIZE=2" in s
        assert s.strip().splitlines()[-1].startswith("srun --ntasks=2 --mpi=pmi2 ")


class TestParseSlurmJob:
    def test_completed(self):
        assert parse_slurm_job("JobId=7 JobState=COMPLETED Reason=None ExitCode=0:0") == \
            ("COMPLETED", "None", 0)

    def test_failed_keeps_the_exit_code(self):
        assert parse_slurm_job("JobId=7 JobState=FAILED Reason=NonZeroExitCode ExitCode=2:0")[2] == 2

    def test_signal_never_reads_as_success(self):
        assert parse_slurm_job("JobState=CANCELLED Reason=None ExitCode=0:15")[2] == 143

    def test_pending_has_no_exit_code_yet(self):
        assert parse_slurm_job("JobState=PENDING Reason=Resources") == ("PENDING", "Resources", None)


def test_collect_drops_staging_artifacts(tmp_path, monkeypatch):
    remote = tmp_path / "remote"
    (remote / "sub").mkdir(parents=True)
    for name in ("stdout.log", "probe.json", "sub/r.txt", "launcher-slinky.sbatch"):
        (remote / name).write_text(name)
    a = SlinkyAdapter(log=lambda m: None)
    monkeypatch.setattr(a, "_login_pod", lambda ctx: "login-0")
    monkeypatch.setattr(a, "_exec_argv",
                        lambda ctx, pod, script, stdin=False: ["tar", "-C", str(remote), "-cf", "-", "."])
    node = NodeRecord(node_id="slinky-t4", name="slinky-t4", index=-1, ip="", provider="slinky")
    dest = a.collect([node], "job-abc", spec(), str(tmp_path / "drop"))
    got = sorted(os.path.relpath(os.path.join(d, f), dest)
                 for d, _, fs in os.walk(dest) for f in fs)
    assert got == ["probe.json", "stdout.log", "sub/r.txt"]


# ── routing ─────────────────────────────────────────────────────────────────

@pytest.fixture()
def slinky_pool(store, seed):
    seed(cpu=1, gpu=True)
    slot = NodeRecord(node_id="slinky-t4", name="slinky-t4", index=-1, ip="",
                      state=NodeState.AVAILABLE.value, gpu=True,
                      runtime="slinky", provider="slinky", kube_context="t4")
    store.put_node(slot)
    return slot


@pytest.fixture()
def rec(store):
    pool, slinky = FakeAdapter(), FakeAdapter()
    from reconciler.reconciler import Reconciler
    r = Reconciler(store, log=lambda m: None,
                   adapters={"local": pool, "libvirt": pool, "static-ssh": pool,
                             "slinky": slinky})
    return r, pool, slinky


def test_a_plain_gpu_job_never_claims_a_slinky_slot(store, slinky_pool, rec):
    r, pool, slinky = rec
    r.submit(spec())
    drive(r)
    assert [c[2] for c in pool.calls if c[0] == "run"] == [("cluster-host",)]
    assert slinky.calls == []
    assert_pool_returned(store)


def test_a_node_name_pin_onto_the_slot_runs_on_slinky(store, slinky_pool, rec):
    r, pool, slinky = rec
    job_id = r.submit(spec(node_count=2, launcher="mpi", params={"node_name": "slinky-t4"}))
    drive(r)
    assert store.get_job(job_id).state == JobState.PROMOTED.value
    assert [c[2] for c in slinky.calls if c[0] == "run"] == [("slinky-t4",)]
    assert pool.calls == []
    assert_pool_returned(store)


def test_backend_slinky_claims_by_context(store, slinky_pool, rec):
    r, _, slinky = rec
    job_id = r.submit(spec(backend="slinky", params={"slinky_context": "t4"}))
    drive(r)
    assert store.get_job(job_id).state == JobState.PROMOTED.value
    assert [c[2] for c in slinky.calls if c[0] == "run"] == [("slinky-t4",)]


def test_a_launch_block_is_refused_on_slinky(store, slinky_pool, rec):
    r, _, slinky = rec
    launch = json.loads(open(os.path.join(REPO_DIR, "contract", "launch-intent", "v1",
                                          "examples", "one-rank-per-gpu.json")).read())
    job_id = r.submit(spec(params={"node_name": "slinky-t4"}, launch=launch))
    drive(r)
    job = store.get_job(job_id)
    assert job.state == JobState.FAILED.value and "slinky" in job.error
    assert slinky.calls == []


# ── kubectl transport failures ──────────────────────────────────────────────

class _Proc:
    def __init__(self, rc=0, out="", err=""):
        self.returncode, self.stdout, self.stderr = rc, out, err


@pytest.fixture()
def quick(monkeypatch):
    import reconciler.adapter as ad
    monkeypatch.setattr(ad, "SLINKY_TRANSPORT_BACKOFF_S", 0)
    monkeypatch.setattr(ad.time, "sleep", lambda s: None)
    a = SlinkyAdapter(log=lambda m: None)
    monkeypatch.setattr(a, "_login_pod", lambda ctx: "login-0")
    return a


def _script(monkeypatch, replies):
    """subprocess.run answers from `replies` in order; returns the scripts run."""
    import reconciler.adapter as ad
    seen = []

    def run(argv, **kw):
        seen.append(argv[-1])
        return replies.pop(0)
    monkeypatch.setattr(ad.subprocess, "run", run)
    return seen


def test_a_dropped_stream_is_retried(quick, monkeypatch):
    seen = _script(monkeypatch, [_Proc(1, "", "error: unexpected EOF"), _Proc(0, "ok", "")])
    assert quick._exec("ctx", "login-0", "true") == (0, "ok", "")
    assert len(seen) == 2


def test_the_command_failing_is_not_retried(quick, monkeypatch):
    seen = _script(monkeypatch, [_Proc(1, "", "slurm_load_jobs error: Invalid job id")])
    assert quick._exec("ctx", "login-0", "scontrol show job 9")[0] == 1
    assert len(seen) == 1


def test_a_cut_submit_adopts_the_job_slurm_already_has(quick, monkeypatch):
    seen = _script(monkeypatch, [_Proc(1, "", "error: lost connection to pod"),
                                 _Proc(0, "42\n", "")])
    assert quick._submit("ctx", "login-0", "job-abc", "/w", "#!/bin/bash\n") == "42"
    assert "sbatch" in seen[0] and "squeue" in seen[1] and len(seen) == 2


def test_a_cut_submit_that_never_landed_is_resubmitted(quick, monkeypatch):
    seen = _script(monkeypatch, [_Proc(1, "", "error: unexpected EOF"),
                                 _Proc(0, "", ""), _Proc(0, "43\n", "")])
    assert quick._submit("ctx", "login-0", "job-abc", "/w", "#!/bin/bash\n") == "43"
    assert [("sbatch" in s, "squeue" in s) for s in seen] == [(True, False), (False, True),
                                                             (True, False)]


def test_sbatch_refusing_is_not_retried(quick, monkeypatch):
    _script(monkeypatch, [_Proc(1, "", "sbatch: error: Invalid generic resource (gres) specification")])
    with pytest.raises(RuntimeError, match="sbatch refused"):
        quick._submit("ctx", "login-0", "job-abc", "/w", "#!/bin/bash\n")
