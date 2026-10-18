"""SlinkyAdapter: the pure pieces (image refs, the sbatch script, Slurm job
parsing), collect's tar filtering, and how the reconciler routes a job onto a
slinky slot. Nothing here reaches kubectl or a real cluster."""
from __future__ import annotations

import json
import os

import pytest

from reconciler.adapter import (SlinkyAdapter, parse_sinfo_nodes, parse_slurm_job, render_sbatch,
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

    def test_pinned_nodes_are_whole_and_in_rank_order(self):
        s = render_sbatch("job-abc", spec(launcher="mpi", node_count=2), "/w", "/i.sqsh",
                          nodelist=("node-b", "node-a"), gpus_per_node=4)
        assert "#SBATCH --nodelist=node-b,node-a" in s and "#SBATCH --exclusive" in s
        assert "#SBATCH --gres=gpu:4" in s
        assert "printf '%s\\n' node-b node-a > /w/hostfile" in s
        assert "export SLURM_HOSTFILE=/w/hostfile" in s and "export MASTER_ADDR=node-b" in s
        assert "scontrol show hostnames" not in s
        assert s.strip().splitlines()[-1].startswith("srun --ntasks=2 --distribution=arbitrary --mpi=pmix ")

    def test_a_single_launch_on_pinned_nodes_runs_on_the_first(self):
        s = render_sbatch("job-abc", spec(), "/w", "/i.sqsh", nodelist=("node-b",), gpus_per_node=1)
        assert "printf '%s\\n' node-b > /w/hostfile" in s
        assert s.strip().splitlines()[-1].startswith("srun --ntasks=1 --distribution=arbitrary ")


def test_parse_sinfo_nodes():
    text = ("w-1|gpu:a10g:1(S:0)\nw-0|gpu:a10g:1\nw-0|gpu:a10g:1\n"
            "big|gpu:a10g:2,gpu:t4:2,shard:8\ncpu|(null)\n\n")
    assert parse_sinfo_nodes(text) == [("big", 4), ("cpu", 0), ("w-0", 1), ("w-1", 1)]


def test_slots_must_agree_on_kind_and_gpus():
    a = NodeRecord(node_id="a", name="a", index=-1, ip="", slurm_node="w-0", gpu_count=1)
    b = NodeRecord(node_id="b", name="b", index=-1, ip="", slurm_node="w-1", gpu_count=1)
    wide = NodeRecord(node_id="c", name="c", index=-1, ip="")
    assert SlinkyAdapter._slurm_nodes([b, a]) == (("w-1", "w-0"), 1)
    assert SlinkyAdapter._slurm_nodes([wide]) == ((), 0)
    with pytest.raises(RuntimeError, match="mix"):
        SlinkyAdapter._slurm_nodes([a, wide])
    b.gpu_count = 4
    with pytest.raises(RuntimeError, match="different GPU counts"):
        SlinkyAdapter._slurm_nodes([a, b])


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


@pytest.fixture()
def per_node_pool(store, seed):
    seed(cpu=1, gpu=True)
    for i in range(2):
        store.put_node(NodeRecord(node_id=f"slinky-a10g-{i}", name=f"slinky-a10g-{i}", index=-1,
                                  ip="", state=NodeState.AVAILABLE.value, gpu=True,
                                  runtime="slinky", provider="slinky", kube_context="a10g",
                                  slurm_node=f"w-{i}", gpu_count=1))


def test_pinning_per_node_slots_claims_each_in_rank_order(store, per_node_pool, rec):
    r, pool, slinky = rec
    job_id = r.submit(spec(node_count=2, launcher="mpi",
                           params={"node_name": "slinky-a10g-1,slinky-a10g-0"}))
    drive(r)
    assert store.get_job(job_id).state == JobState.PROMOTED.value
    assert [c[2] for c in slinky.calls if c[0] == "run"] == [("slinky-a10g-1", "slinky-a10g-0")]
    assert pool.calls == []
    assert_pool_returned(store)


def test_a_per_node_pin_must_match_node_count(store, per_node_pool, rec):
    r, _, slinky = rec
    job_id = r.submit(spec(node_count=2, launcher="mpi", params={"node_name": "slinky-a10g-0"}))
    drive(r)
    job = store.get_job(job_id)
    assert job.state == JobState.FAILED.value and "node_count is 2" in job.error
    assert slinky.calls == []


def test_backend_slinky_takes_node_count_per_node_slots(store, per_node_pool, rec):
    r, _, slinky = rec
    job_id = r.submit(spec(backend="slinky", node_count=2, launcher="mpi"))
    drive(r)
    assert store.get_job(job_id).state == JobState.PROMOTED.value
    ran = [c[2] for c in slinky.calls if c[0] == "run"]
    assert len(ran) == 1 and sorted(ran[0]) == ["slinky-a10g-0", "slinky-a10g-1"]


def test_two_cluster_wide_slots_cannot_carry_one_job(store, slinky_pool, rec):
    r, _, slinky = rec
    store.put_node(NodeRecord(node_id="slinky-t4b", name="slinky-t4b", index=-1, ip="",
                              state=NodeState.AVAILABLE.value, gpu=True, runtime="slinky",
                              provider="slinky", kube_context="t4"))
    job_id = r.submit(spec(node_count=2, launcher="mpi",
                           params={"node_name": "slinky-t4,slinky-t4b"}))
    drive(r)
    job = store.get_job(job_id)
    assert job.state == JobState.FAILED.value and "cluster-wide" in job.error


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


# ── slurmctld briefly unreachable ───────────────────────────────────────────
TIMEOUT = "slurm_load_jobs error: Socket timed out on send/recv operation"


def test_polling_rides_out_a_controller_outage(quick, monkeypatch):
    seen = _script(monkeypatch, [_Proc(1, "", TIMEOUT), _Proc(1, "", TIMEOUT),
                                 _Proc(0, "JobId=7 JobState=COMPLETED Reason=None ExitCode=0:0", "")])
    assert quick._await("ctx", "login-0", "7", "job-abc") == ("COMPLETED", 0)
    assert len(seen) == 3


def test_polling_gives_up_after_the_grace(quick, monkeypatch):
    import reconciler.adapter as ad
    monkeypatch.setattr(ad, "SLINKY_CTLD_GRACE_S", 0)
    _script(monkeypatch, [_Proc(1, "", TIMEOUT)])
    with pytest.raises(RuntimeError, match="cannot read Slurm job 7"):
        quick._await("ctx", "login-0", "7", "job-abc")


def test_provision_waits_for_the_controller(quick, monkeypatch):
    seen = _script(monkeypatch, [_Proc(1, "Slurmctld(primary) at c-0 is DOWN", ""), _Proc(0, "UP", "")])
    quick.provision(NodeRecord(node_id="s", name="s", index=-1, ip=""), "job-abc")
    assert seen == ["scontrol ping", "scontrol ping"]


def test_teardown_waits_until_slurm_drops_the_job(quick, monkeypatch):
    seen = _script(monkeypatch, [_Proc(1, "", TIMEOUT), _Proc(0, "3\n", ""), _Proc(0, "", ""),
                                 _Proc(0, "", "")])
    quick.deprovision(NodeRecord(node_id="s", name="s", index=-1, ip=""), "job-abc")
    assert [s.startswith("scancel") for s in seen] == [True, True, True, False]
    assert seen[-1].startswith("rm -rf /scratch/outpost/job-abc")


def test_teardown_refuses_to_release_over_a_live_job(quick, monkeypatch):
    import reconciler.adapter as ad
    monkeypatch.setattr(ad, "SLINKY_CTLD_GRACE_S", 0)
    _script(monkeypatch, [_Proc(0, "3\n", "")])
    with pytest.raises(RuntimeError, match="Slurm still has outpost-job-abc"):
        quick.deprovision(NodeRecord(node_id="s", name="s", index=-1, ip=""), "job-abc")


# ── a finished job slurmctld has already forgotten ──────────────────────────
GONE = "slurm_load_jobs error: Invalid job id specified"


@pytest.mark.parametrize("sacct,expected", [
    ("COMPLETED|0:0\n", ("COMPLETED", 0)),
    ("FAILED|2:0\n", ("FAILED", 2)),
    ("CANCELLED by 0|0:15\n", ("CANCELLED", 143)),
])
def test_a_forgotten_job_is_read_from_accounting(quick, monkeypatch, sacct, expected):
    seen = _script(monkeypatch, [_Proc(1, "", GONE), _Proc(0, sacct, "")])
    assert quick._await("ctx", "login-0", "15", "job-abc") == expected
    assert "sacct -j 15" in seen[1]


@pytest.mark.parametrize("sacct", [_Proc(0, "", ""), _Proc(0, "RUNNING|0:0\n", ""),
                                   _Proc(1, "", "sacct: error: slurmdbd unreachable")])
def test_a_forgotten_job_accounting_cannot_vouch_for_still_fails(quick, monkeypatch, sacct):
    _script(monkeypatch, [_Proc(1, "", GONE), sacct])
    with pytest.raises(RuntimeError, match="cannot read Slurm job 15"):
        quick._await("ctx", "login-0", "15", "job-abc")


# ── our link to the cluster drops while the job runs ────────────────────────
DOWN = "The connection to the server 127.0.0.1:20544 was refused - did you specify the right host or port?"


def test_kubectl_refusing_the_connection_is_a_transport_error():
    from reconciler.adapter import is_kubectl_transport_error
    assert is_kubectl_transport_error(DOWN)


def test_losing_the_link_outlasts_the_controller_grace(quick, monkeypatch):
    import reconciler.adapter as ad
    monkeypatch.setattr(ad, "SLINKY_CTLD_GRACE_S", 0)
    monkeypatch.setattr(ad, "SLINKY_TRANSPORT_RETRIES", 1)
    _script(monkeypatch, [_Proc(1, "", DOWN), _Proc(1, "", DOWN),
                          _Proc(0, "JobId=7 JobState=COMPLETED Reason=None ExitCode=0:0", "")])
    assert quick._await("ctx", "login-0", "7", "job-abc") == ("COMPLETED", 0)


def test_losing_the_link_past_the_run_timeout_fails(quick, monkeypatch):
    import reconciler.adapter as ad
    monkeypatch.setattr(ad, "SLINKY_RUN_TIMEOUT_S", 0)
    monkeypatch.setattr(ad, "SLINKY_TRANSPORT_RETRIES", 1)
    _script(monkeypatch, [_Proc(1, "", DOWN)])
    with pytest.raises(RuntimeError, match="before the run timeout"):
        quick._await("ctx", "login-0", "7", "job-abc")


# ── ranks: Slurm's task ids become the names frameworks read ────────────────
def _srun_tail(script: str) -> list[str]:
    import shlex as _shlex
    srun = next(line for line in script.splitlines() if line.startswith("srun "))
    argv = _shlex.split(srun)
    return argv[next(i for i, a in enumerate(argv) if a.startswith("--container-workdir")) + 1:]


@pytest.mark.parametrize("procid,localid", [("0", "0"), ("1", "0")])
def test_an_mpi_task_runs_with_its_own_rank(procid, localid):
    import subprocess as sp
    s = render_sbatch("job-1", spec(launcher="mpi", node_count=2, command=["env"]), "/w",
                      "/i.sqsh", nodelist=("a", "b"), gpus_per_node=1)
    env = {"PATH": "/usr/bin:/bin", "SLURM_PROCID": procid, "SLURM_LOCALID": localid,
           "SLURM_NTASKS_PER_NODE": "1"}
    out = sp.run(_srun_tail(s), env=env, capture_output=True, text=True).stdout.splitlines()
    assert {f"RANK={procid}", f"LOCAL_RANK={localid}", "LOCAL_WORLD_SIZE=1"} <= set(out)
    assert "export WORLD_SIZE=2" in s


def test_a_single_task_job_gets_no_rank_wrapper_and_world_size_one():
    s = render_sbatch("job-1", spec(command=["python3", "x.py"], node_count=2), "/w", "/i.sqsh")
    assert _srun_tail(s) == ["python3", "x.py"]
    assert "export WORLD_SIZE=1" in s
