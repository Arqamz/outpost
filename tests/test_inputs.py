"""Job inputs (reconciler/inputs.py): the JobSpec contract, S3 URL signing, the
fetch script run under a real `sh` against `file://` URLs, and each backend
mounting what it staged. The aws CLI, ssh and kubectl are faked."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import subprocess
from unittest.mock import MagicMock

import pytest

from reconciler import adapter as A
from reconciler import inputs as I
from reconciler.adapter import (KubernetesAdapter, LibvirtAdapter, SlinkyAdapter,
                                container_argv, render_sbatch)
from reconciler.models import JobSpec, NodeRecord

FILES = {"models/m/config.json": b'{"a": 1}', "models/m/w.safetensors": b"weights",
         "models/big/w.safetensors": b"other weights", "data/d.jsonl": b"rows\n"}


def sha(b: bytes) -> str:
    return hashlib.sha256(b).hexdigest()


MANIFEST = "".join(f"{sha(b)}  {rel}\n" for rel, b in sorted(FILES.items())).encode()
INPUT = {"source": "s3://bucket/qwen-sft/bundle/", "manifest": "ARTIFACTS.sha256",
         "manifest_sha256": "sha256:" + sha(MANIFEST), "path": "/tmp/gtl-artifacts",
         "only": ["models/m/", "data/"]}


class TestParse:
    def test_absent_is_empty(self):
        assert I.parse_inputs(None) == []

    def test_a_valid_input_round_trips_and_gains_a_trailing_slash(self):
        spec = JobSpec.from_dict({"name": "j", "inputs": [{**INPUT, "source": "s3://bucket/x"}]})
        assert spec.inputs == [{**INPUT, "source": "s3://bucket/x/"}]

    def test_a_caller_cannot_set_mounts(self):
        assert JobSpec.from_dict({"name": "j", "mounts": ["/etc:/etc"]}).mounts == []

    @pytest.mark.parametrize("bad,field", [
        ({**INPUT, "source": "https://bucket/x/"}, "source"),
        ({**INPUT, "source": "s3://"}, "source"),
        ({**INPUT, "manifest": "/ARTIFACTS.sha256"}, "manifest"),
        ({**INPUT, "manifest": "../ARTIFACTS.sha256"}, "manifest"),
        ({**INPUT, "manifest": ""}, "manifest"),
        ({**INPUT, "manifest_sha256": "sha256:abc"}, "manifest_sha256"),
        ({**INPUT, "manifest_sha256": None}, "manifest_sha256"),
        ({**INPUT, "path": "tmp/gtl-artifacts"}, "path"),
        ({**INPUT, "path": "/tmp/../etc"}, "path"),
        ({**INPUT, "only": "models/"}, "only"),
        ({**INPUT, "only": ["../x"]}, "only"),
        ({**INPUT, "only": ["/abs"]}, "only"),
    ])
    def test_a_malformed_input_refuses_the_job(self, bad, field):
        with pytest.raises(ValueError, match=rf"inputs\[0\]\.{field}"):
            JobSpec.from_dict({"name": "j", "inputs": [bad]})

    @pytest.mark.parametrize("raw", ["s3://bucket/x/", ["s3://bucket/x/"]])
    def test_a_non_list_or_non_map_refuses(self, raw):
        with pytest.raises(ValueError, match="inputs"):
            I.parse_inputs(raw)


def test_presign_matches_the_aws_published_example():
    """The SigV4 presigned-URL example from the S3 API reference
    ("Authenticating Requests: Using Query Parameters")."""
    url = I.presign("examplebucket", "test.txt", "us-east-1", "AKIAIOSFODNN7EXAMPLE",
                    "wJalrXUtnFEMI/K7MDENG/bPxRfiCYEXAMPLEKEY", None, 86400,
                    dt.datetime(2013, 5, 24, tzinfo=dt.timezone.utc))
    assert url.startswith("https://examplebucket.s3.amazonaws.com/test.txt?")
    assert url.endswith("X-Amz-Signature="
                        "aeeed9bbccd4d02ee5c0109b86d86835f995330da4c265957d157751f604d404")


def test_presign_names_the_region_and_carries_a_session_token():
    url = I.presign("b", "k/a b.txt", "eu-west-1", "AK", "SK", "TOK/=", 60,
                    dt.datetime(2026, 10, 15, tzinfo=dt.timezone.utc))
    assert url.startswith("https://b.s3.eu-west-1.amazonaws.com/k/a%20b.txt?")
    assert "X-Amz-Security-Token=TOK%2F%3D" in url


NOW = dt.datetime(2026, 10, 15, 12, tzinfo=dt.timezone.utc)


def _aws(calls, location=None, expiration=None, rc=0):
    def run(argv, **kw):
        calls.append(argv[3:5])
        if rc:
            return MagicMock(returncode=rc, stdout="", stderr="Token has expired")
        if argv[3] == "configure":
            body = {"AccessKeyId": "AK", "SecretAccessKey": "SK", "SessionToken": "T"}
            if expiration:
                body["Expiration"] = expiration
            return MagicMock(returncode=0, stdout=json.dumps(body))
        return MagicMock(returncode=0, stdout=json.dumps({"LocationConstraint": location}))
    return run


class TestSigner:
    def test_no_profile_refuses_and_names_the_setting(self):
        with pytest.raises(RuntimeError, match="CLUSTER_S3_PROFILE"):
            I.S3Signer(profile="", run=_aws([])).url("s3://bucket/k")

    def test_credentials_and_region_are_fetched_once(self):
        calls = []
        s = I.S3Signer("gtl", run=_aws(calls, expiration="2026-10-15T13:00:00+00:00"),
                       clock=lambda: NOW)
        s.url("s3://bucket/a"), s.url("s3://bucket/b")
        assert calls == [["configure", "export-credentials"], ["s3api", "get-bucket-location"]]

    def test_a_null_location_is_us_east_1_and_others_are_kept(self):
        assert ".s3.amazonaws.com/" in I.S3Signer("gtl", run=_aws([]), clock=lambda: NOW).url(
            "s3://bucket/a")
        assert ".s3.eu-west-1.amazonaws.com/" in I.S3Signer(
            "gtl", run=_aws([], location="eu-west-1"), clock=lambda: NOW).url("s3://bucket/a")

    def test_credentials_near_expiry_are_fetched_again(self):
        calls = []
        s = I.S3Signer("gtl", run=_aws(calls, expiration="2026-10-15T12:05:00+00:00"),
                       clock=lambda: NOW)
        s.url("s3://bucket/a"), s.url("s3://bucket/b")
        assert calls.count(["configure", "export-credentials"]) == 2

    def test_an_aws_failure_refuses_and_names_the_profile(self):
        with pytest.raises(RuntimeError, match="'gtl'.*aws sso login"):
            I.S3Signer("gtl", run=_aws([], rc=255), clock=lambda: NOW).url("s3://bucket/a")


@pytest.fixture()
def bucket(tmp_path):
    """FILES + the manifest on disk, served as file:// URLs by a fake signer."""
    root = tmp_path / "bucket"
    for rel, body in {**FILES, "ARTIFACTS.sha256": MANIFEST}.items():
        (root / rel).parent.mkdir(parents=True, exist_ok=True)
        (root / rel).write_bytes(body)
    sign = MagicMock()
    sign.url = lambda uri: "file://" + str(root / uri.removeprefix(INPUT["source"]))
    get = lambda url: open(url.removeprefix("file://"), "rb").read()  # noqa: E731
    return root, sign, get


def _sh(script, stdin=None):
    return subprocess.run(["sh", "-c", script], input=stdin, capture_output=True,
                          text=True).returncode


class TestPlanFetch:
    def test_lists_the_manifest_first_then_only_the_requested_files(self, bucket):
        _, sign, get = bucket
        rows = [line.split()[:2] for line in I.plan_fetch(INPUT, sign, get).stdin.splitlines()]
        assert rows == [[sha(MANIFEST), "ARTIFACTS.sha256"],
                        [sha(FILES["data/d.jsonl"]), "data/d.jsonl"],
                        [sha(FILES["models/m/config.json"]), "models/m/config.json"],
                        [sha(FILES["models/m/w.safetensors"]), "models/m/w.safetensors"]]

    def test_no_only_means_every_file(self, bucket):
        _, sign, get = bucket
        assert I.plan_fetch({**INPUT, "only": []}, sign, get).files == 1 + len(FILES)

    def test_a_manifest_that_is_not_the_pinned_one_refuses(self, bucket):
        _, sign, get = bucket
        with pytest.raises(RuntimeError, match="re-pin manifest_sha256"):
            I.plan_fetch({**INPUT, "manifest_sha256": "sha256:" + "0" * 64}, sign, get)

    def test_prefixes_that_match_nothing_refuse(self, bucket):
        _, sign, get = bucket
        with pytest.raises(RuntimeError, match="lists nothing"):
            I.plan_fetch({**INPUT, "only": ["models/none/"]}, sign, get)

    @pytest.mark.parametrize("line", ["nothex  data/x", f"{'a' * 64}  ../escape",
                                      f"{'a' * 64}  data/has space"])
    def test_an_unusable_manifest_line_refuses(self, bucket, line):
        root, sign, _ = bucket
        body = MANIFEST + (line + "\n").encode()
        with pytest.raises(RuntimeError, match="unusable line"):
            I.plan_fetch({**INPUT, "only": [], "manifest_sha256": "sha256:" + sha(body)},
                         sign, lambda url: body)


class TestFetchScript:
    def test_stages_verified_files_and_marks_the_subset_complete(self, bucket, tmp_path):
        _, sign, get = bucket
        cache = str(tmp_path / "cache")
        assert _sh(I.fetch_script(cache, INPUT), I.plan_fetch(INPUT, sign, get).stdin) == 0
        d = tmp_path / "cache" / sha(MANIFEST)
        assert (d / "models/m/w.safetensors").read_bytes() == b"weights"
        assert (d / "ARTIFACTS.sha256").read_bytes() == MANIFEST
        assert not (d / "models/big").exists()
        assert _sh(I.check_script(cache, INPUT)) == 0

    def test_a_file_that_does_not_match_is_refused_and_left_out(self, bucket, tmp_path):
        root, sign, get = bucket
        stdin = I.plan_fetch(INPUT, sign, get).stdin
        (root / "data/d.jsonl").write_bytes(b"tampered")
        cache = str(tmp_path / "cache")
        r = subprocess.run(["sh", "-c", I.fetch_script(cache, INPUT)], input=stdin,
                           capture_output=True, text=True)
        assert r.returncode != 0 and "republish the bundle" in r.stderr
        d = tmp_path / "cache" / sha(MANIFEST)
        assert not (d / "data/d.jsonl").exists() and not (d / "data/d.jsonl.part").exists()
        assert _sh(I.check_script(cache, INPUT)) != 0

    def test_a_present_file_is_not_downloaded_again(self, bucket, tmp_path):
        root, sign, get = bucket
        stdin = I.plan_fetch(INPUT, sign, get).stdin
        cache = str(tmp_path / "cache")
        assert _sh(I.fetch_script(cache, INPUT), stdin) == 0
        (tmp_path / "cache" / sha(MANIFEST) / I.marker(INPUT)).unlink()
        for rel in FILES:
            (root / rel).unlink()
        assert _sh(I.fetch_script(cache, INPUT), stdin) == 0

    def test_another_subset_of_the_same_bundle_is_not_complete(self, bucket, tmp_path):
        _, sign, get = bucket
        cache = str(tmp_path / "cache")
        _sh(I.fetch_script(cache, INPUT), I.plan_fetch(INPUT, sign, get).stdin)
        assert _sh(I.check_script(cache, {**INPUT, "only": ["models/big/"]})) != 0


class TestStage:
    def test_a_warm_cache_skips_the_network(self, tmp_path):
        got = I.stage([INPUT], "/c", lambda s, i: 0, lambda m: None,
                      sign=MagicMock(), get=lambda url: pytest.fail("fetched a manifest"))
        assert got == [f"/c/{sha(MANIFEST)}:/tmp/gtl-artifacts:ro"]

    def test_a_cold_cache_fetches_over_stdin(self, bucket):
        _, sign, get = bucket
        runs = []
        I.stage([INPUT], "/c", lambda s, i: runs.append(i) or (1 if i is None else 0),
                lambda m: None, sign=sign, get=get)
        assert runs[0] is None and "ARTIFACTS.sha256" in runs[1]

    def test_a_failed_fetch_refuses(self, bucket):
        _, sign, get = bucket
        with pytest.raises(RuntimeError, match="staging s3://bucket/qwen-sft/bundle/ failed"):
            I.stage([INPUT], "/c", lambda s, i: 1, lambda m: None, sign=sign, get=get)


MOUNT = f"/c/{sha(MANIFEST)}:/tmp/gtl-artifacts:ro"


class TestMounts:
    def test_apptainer_binds_each_mount(self):
        argv = container_argv(JobSpec(name="j", image="/x.sif", mounts=[MOUNT]), "/w", "/out")
        assert argv[argv.index(MOUNT) - 1] == "--bind"

    def test_docker_mounts_each_one(self):
        argv = container_argv(JobSpec(name="j", image="x", runtime="docker", mounts=[MOUNT]),
                              "/w", "/out")
        assert argv[argv.index(MOUNT) - 1] == "-v"

    def test_pyxis_mounts_follow_the_workdir(self):
        s = render_sbatch("job-1", JobSpec(name="j", image="x", mounts=[MOUNT]), "/w", "/i.sqsh")
        assert f"--container-mounts=/w:/out,{MOUNT}" in s


def _patch_stage(monkeypatch):
    roots = []
    monkeypatch.setattr(I, "stage", lambda inputs, root, run_sh, log:
                        roots.append(root) or [f"{root}/x:/tmp/gtl-artifacts:ro"])
    return roots


def test_slinky_stages_on_the_shared_disk_and_mounts_it(monkeypatch):
    roots = _patch_stage(monkeypatch)
    a = SlinkyAdapter(log=lambda m: None)
    submitted = []
    monkeypatch.setattr(a, "_login_pod", lambda ctx: "login-0")
    monkeypatch.setattr(a, "_stage_image", lambda *args: "/scratch/outpost/.images/x.sqsh")
    monkeypatch.setattr(a, "_slurm_nodes", lambda nodes: ((), 0))
    monkeypatch.setattr(a, "_submit", lambda ctx, pod, job_id, wd, script:
                        submitted.append(script) or "7")
    monkeypatch.setattr(a, "_await", lambda *args: ("COMPLETED", 0))
    monkeypatch.setattr(a, "_exec", lambda *args, **kw: (0, "", ""))
    slot = NodeRecord(node_id="s", name="s", index=-1, ip="", provider="slinky")
    a.run([slot], "job-1", JobSpec(name="j", image="docker://x/y:1", inputs=[INPUT]))
    assert roots == [f"{A.SLINKY_WORKDIR_ROOT}/.inputs"]
    assert f"{A.SLINKY_WORKDIR_ROOT}/.inputs/x:/tmp/gtl-artifacts:ro" in submitted[0]


def test_an_apptainer_node_stages_on_each_node_and_binds_it(monkeypatch):
    roots = _patch_stage(monkeypatch)
    ran = []
    monkeypatch.setattr(A, "run_logged", lambda argv, *args, **kw: ran.append(argv) or 0)
    node = NodeRecord(node_id="n1", name="n1", index=1, ip="10.0.0.1")
    LibvirtAdapter(log=lambda m: None).run([node], "job-1",
                                           JobSpec(name="j", image="/x.sif", inputs=[INPUT]))
    assert roots == [A.REMOTE_INPUTS_DIR]
    assert any(f"{A.REMOTE_INPUTS_DIR}/x:/tmp/gtl-artifacts:ro" in " ".join(a) for a in ran)


def test_kubernetes_refuses_a_job_with_inputs(tmp_path, monkeypatch):
    monkeypatch.setattr(A, "RUNS_DIR", str(tmp_path))
    slot = NodeRecord(node_id="k", name="k", index=-1, ip="", provider="k8s")
    with pytest.raises(RuntimeError, match="cannot stage job inputs"):
        KubernetesAdapter(log=lambda m: None).run(
            [slot], "job-1", JobSpec(name="j", image="x/y:1", inputs=[INPUT]))
