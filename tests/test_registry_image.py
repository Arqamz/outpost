"""Images from a private registry: the prebuilt-SIF companion (`image_sif`), the
ECR credentials outpost mints on the control plane, and how they reach a node
(stdin, never argv or a log). The fetch script runs under a real `sh` against a
fake `apptainer`; ssh, kubectl and the aws CLI are faked."""
from __future__ import annotations

import hashlib
import os
import subprocess
from unittest.mock import MagicMock

import pytest

from reconciler import adapter as A
from reconciler.adapter import (REMOTE_SIF_CACHE_DIR, LibvirtAdapter, SlinkyAdapter,
                                _sif_cache_name, _sif_fetch_script, ensure_local_sif,
                                registry_host, registry_login)
from reconciler.models import JobSpec, NodeRecord, parse_image_sif

ECR = "009644552272.dkr.ecr.us-east-1.amazonaws.com"
OCI = f"docker://{ECR}/betacodes/tashkil@sha256:" + "e" * 64
CONTENT = "the prebuilt sif bytes"
SIF_SHA = "sha256:" + hashlib.sha256(CONTENT.encode()).hexdigest()
SIF = {"ref": f"oras://{ECR}/betacodes/tashkil@sha256:" + "c" * 64, "sha256": SIF_SHA}


@pytest.fixture(autouse=True)
def _no_cached_tokens(monkeypatch):
    monkeypatch.setattr(A, "_ecr_tokens", {})


class TestParseImageSif:
    def test_absent_is_none(self):
        assert parse_image_sif(None) is None

    def test_a_valid_companion_round_trips_through_from_dict(self):
        spec = JobSpec.from_dict({"name": "j", "image": OCI, "image_sif": SIF})
        assert spec.image_sif == SIF

    @pytest.mark.parametrize("sif", [
        {**SIF, "ref": f"oras://{ECR}/betacodes/tashkil:runtime-sif"},
        {**SIF, "ref": f"docker://{ECR}/betacodes/tashkil@sha256:" + "c" * 64},
        {**SIF, "ref": f"oras://{ECR}/betacodes/tashkil@sha256:" + "c" * 63},
        {"sha256": SIF_SHA},
        {**SIF, "sha256": "sha256:TBD"},
        {"ref": SIF["ref"]},
        SIF["ref"],
    ])
    def test_a_malformed_companion_refuses_the_job(self, sif):
        with pytest.raises(ValueError, match="fix the image pin"):
            JobSpec.from_dict({"name": "j", "image": OCI, "image_sif": sif})


class TestRegistryHost:
    @pytest.mark.parametrize("image,host", [
        (OCI, ECR),
        (SIF["ref"], ECR),
        ("docker://nvcr.io/nvidia/hpc-benchmarks:26.02", "nvcr.io"),
        ("localhost:5000/x:1", "localhost:5000"),
        ("docker://library/ubuntu:24.04", ""),
        ("ubuntu:24.04", ""),
        ("/scratch/img.sqsh", ""),
    ])
    def test_host_of(self, image, host):
        assert registry_host(image) == host


class TestRegistryLogin:
    def test_a_public_registry_needs_no_login(self, monkeypatch):
        monkeypatch.setattr(A, "ECR_PROFILE", "")
        assert registry_login("docker://nvcr.io/nvidia/hpc-benchmarks:26.02") is None

    def test_ecr_without_a_profile_refuses_and_names_the_setting(self, monkeypatch):
        monkeypatch.setattr(A, "ECR_PROFILE", "")
        with pytest.raises(RuntimeError, match="CLUSTER_ECR_PROFILE"):
            registry_login(OCI)

    def test_ecr_mints_once_with_the_profile_and_region_then_reuses(self, monkeypatch):
        monkeypatch.setattr(A, "ECR_PROFILE", "tashkil-registry")
        calls = []
        monkeypatch.setattr(A.subprocess, "run", lambda argv, **kw: calls.append(argv)
                            or MagicMock(returncode=0, stdout="tok-123\n", stderr=""))
        assert registry_login(OCI) == ("AWS", "tok-123")
        assert registry_login(SIF["ref"]) == ("AWS", "tok-123")
        assert calls == [["aws", "--profile", "tashkil-registry", "--region", "us-east-1",
                          "ecr", "get-login-password"]]

    def test_an_expired_token_is_minted_again(self, monkeypatch):
        monkeypatch.setattr(A, "ECR_PROFILE", "p")
        monkeypatch.setattr(A, "_ecr_tokens", {ECR: (A.time.monotonic() - A.ECR_TOKEN_TTL_S - 1,
                                                     "old")})
        monkeypatch.setattr(A.subprocess, "run",
                            lambda argv, **kw: MagicMock(returncode=0, stdout="new\n", stderr=""))
        assert registry_login(OCI) == ("AWS", "new")

    @pytest.mark.parametrize("rc,out", [(255, ""), (0, "  \n")])
    def test_a_failed_mint_refuses_and_names_the_profile(self, monkeypatch, rc, out):
        monkeypatch.setattr(A, "ECR_PROFILE", "tashkil-registry")
        monkeypatch.setattr(A.subprocess, "run", lambda argv, **kw: MagicMock(
            returncode=rc, stdout=out, stderr="Token has expired"))
        with pytest.raises(RuntimeError, match="'tashkil-registry'.*aws sso login"):
            registry_login(OCI)


@pytest.fixture()
def fake_apptainer(tmp_path, monkeypatch):
    """An `apptainer` on PATH that writes CONTENT to its output path (argv 3) and
    records its argv and the registry credentials it was given."""
    bin_dir, log = tmp_path / "bin", tmp_path / "apptainer.log"
    bin_dir.mkdir()
    exe = bin_dir / "apptainer"
    exe.write_text('#!/bin/sh\necho "$1 $4 user=$APPTAINER_DOCKER_USERNAME '
                   'pw=$APPTAINER_DOCKER_PASSWORD" >> "$FAKE_LOG"\n'
                   'printf %s "$FAKE_CONTENT" > "$3"\n')
    exe.chmod(0o755)
    monkeypatch.setenv("PATH", f"{bin_dir}:{os.environ['PATH']}")
    monkeypatch.setenv("FAKE_LOG", str(log))
    monkeypatch.setenv("FAKE_CONTENT", CONTENT)
    return log


def _sh(script: str, stdin: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(["sh", "-c", script], input=stdin, capture_output=True, text=True)


class TestFetchScript:
    def test_a_prebuilt_sif_with_the_pinned_digest_lands(self, tmp_path, fake_apptainer):
        tmp, final = tmp_path / "x.tmp", tmp_path / "x.sif"
        r = _sh(_sif_fetch_script(OCI, SIF, str(tmp), str(final), None))
        assert r.returncode == 0, r.stderr
        assert final.read_text() == CONTENT and not tmp.exists()
        assert fake_apptainer.read_text().split()[:2] == ["pull", SIF["ref"]]

    def test_a_prebuilt_sif_with_another_digest_is_refused_and_removed(self, tmp_path,
                                                                        fake_apptainer):
        tmp, final = tmp_path / "x.tmp", tmp_path / "x.sif"
        wrong = {**SIF, "sha256": "sha256:" + "0" * 64}
        r = _sh(_sif_fetch_script(OCI, wrong, str(tmp), str(final), None))
        assert r.returncode != 0
        assert "is not the pinned SIF" in r.stderr
        assert not final.exists() and not tmp.exists()

    def test_without_a_sif_the_image_is_converted(self, tmp_path, fake_apptainer):
        final = tmp_path / "x.sif"
        r = _sh(_sif_fetch_script(OCI, None, str(tmp_path / "x.tmp"), str(final), None))
        assert r.returncode == 0, r.stderr
        assert fake_apptainer.read_text().split()[:2] == ["build", OCI]

    def test_the_password_comes_from_stdin_not_the_script(self, tmp_path, fake_apptainer):
        script = _sif_fetch_script(OCI, SIF, str(tmp_path / "x.tmp"), str(tmp_path / "x.sif"),
                                   ("AWS", "s3cret"))
        assert "s3cret" not in script
        r = _sh(script, stdin="s3cret\n")
        assert r.returncode == 0, r.stderr
        assert "user=AWS pw=s3cret" in fake_apptainer.read_text()

    def test_a_cached_sif_runs_nothing(self, tmp_path, fake_apptainer):
        final = tmp_path / "x.sif"
        final.write_text("already here")
        r = _sh(_sif_fetch_script(OCI, SIF, str(tmp_path / "x.tmp"), str(final), None))
        assert r.returncode == 0 and final.read_text() == "already here"
        assert not fake_apptainer.exists()


class TestCacheName:
    def test_a_prebuilt_sif_is_cached_under_its_own_digest(self):
        assert _sif_cache_name(OCI, SIF) == f"sha256-{SIF_SHA[7:]}.sif"

    def test_a_converted_image_is_cached_under_the_image_digest(self):
        assert _sif_cache_name(OCI) == "sha256-" + "e" * 64 + ".sif"


def test_the_local_cache_pulls_the_prebuilt_sif(tmp_path, monkeypatch, fake_apptainer):
    monkeypatch.setattr(A, "SIF_CACHE_DIR", str(tmp_path / "cache"))
    monkeypatch.setattr(A, "registry_login", lambda image: None)
    monkeypatch.setattr(A, "LOGS_DIR", str(tmp_path / "logs"))
    path = ensure_local_sif(OCI, "job-1", SIF)
    assert path == str(tmp_path / "cache" / f"sha256-{SIF_SHA[7:]}.sif")
    assert open(path).read() == CONTENT


class TestRemoteSif:
    def test_a_prebuilt_sif_is_pulled_on_the_node_with_the_token_on_stdin(self, monkeypatch):
        monkeypatch.setattr(A, "registry_login", lambda image: ("AWS", "s3cret"))
        probes = []
        monkeypatch.setattr(A.subprocess, "run",
                            lambda argv, **kw: probes.append(argv[-1]) or MagicMock(returncode=1))
        logged = []
        monkeypatch.setattr(A, "run_logged",
                            lambda argv, job_id, **kw: logged.append((argv, kw)) or 0)
        node = NodeRecord(node_id="n1", name="n1", index=1, ip="10.0.0.1")
        a = LibvirtAdapter(log=lambda m: None)

        final = a._ensure_remote_sif(node, OCI, "job-1", SIF)

        assert final == f"{REMOTE_SIF_CACHE_DIR}/sha256-{SIF_SHA[7:]}.sif"
        assert not any("df --output=avail" in p for p in probes)  # no conversion headroom needed
        (argv, kw), = logged
        assert "apptainer pull" in argv[-1] and SIF["ref"] in argv[-1]
        assert "s3cret" not in " ".join(argv)
        assert kw["input"] == "s3cret\n"


class TestSlinkyStaging:
    @pytest.fixture()
    def slinky(self, monkeypatch):
        a = SlinkyAdapter(log=lambda m: None)
        execs = []

        def fake_exec(ctx, pod, script, job_id=None, input=None):
            execs.append((script, input))
            return (1, "", "") if script.startswith("test -s") else (0, "", "")
        monkeypatch.setattr(a, "_exec", fake_exec)
        return a, execs

    def test_a_cached_image_needs_no_credentials(self, monkeypatch):
        a = SlinkyAdapter(log=lambda m: None)
        monkeypatch.setattr(A, "ECR_PROFILE", "")  # a login attempt would raise
        execs = []
        monkeypatch.setattr(a, "_exec", lambda ctx, pod, script, job_id=None, input=None:
                            execs.append(script) or (0, "", ""))
        assert a._stage_image("c", "login-0", OCI, "job-1").endswith("e" * 64 + ".sqsh")
        assert len(execs) == 1

    def test_an_ecr_import_gets_a_private_credentials_file(self, slinky, monkeypatch):
        a, execs = slinky
        monkeypatch.setattr(A, "ECR_PROFILE", "ecr")
        monkeypatch.setattr(A, "registry_login", lambda image: ("AWS", "s3cret"))
        a._stage_image("c", "login-0", OCI, "job-1")
        script, stdin = execs[-1]
        assert stdin == "s3cret\n" and "s3cret" not in script
        assert "ENROOT_CONFIG_PATH" in script and f"machine %s login %s" in script
        assert f"{ECR}#betacodes/tashkil" in script

    def test_a_public_import_carries_no_credentials(self, slinky, monkeypatch):
        a, execs = slinky
        monkeypatch.setattr(A, "ECR_PROFILE", "")
        a._stage_image("c", "login-0", "docker://nvcr.io/nvidia/hpc-benchmarks@sha256:"
                       + "a" * 64, "job-1")
        script, stdin = execs[-1]
        assert stdin is None and "ENROOT_CONFIG_PATH" not in script

    def test_an_ecr_import_without_a_profile_uses_the_login_pods_credentials(self, slinky,
                                                                               monkeypatch):
        a, execs = slinky
        monkeypatch.setattr(A, "ECR_PROFILE", "")  # registry_login would raise for ECR
        a._stage_image("c", "login-0", OCI, "job-1")
        script, stdin = execs[-1]
        assert stdin is None and "ENROOT_CONFIG_PATH" not in script
        assert f"{ECR}#betacodes/tashkil" in script

    def test_by_default_the_import_uses_the_login_pods_enroot_temp(self, slinky, monkeypatch):
        a, execs = slinky
        monkeypatch.setattr(A, "ECR_PROFILE", "")
        monkeypatch.setattr(A, "SLINKY_ENROOT_TEMP", "")
        monkeypatch.setattr(A, "SLINKY_ENROOT_TEMP_IMAGE", "")
        a._stage_image("c", "login-0", OCI, "job-1")
        script, _ = execs[-1]
        assert "ENROOT_TEMP_PATH" not in script and "mount" not in script
        assert "export ENROOT_CACHE_PATH=/scratch/outpost/.images/.cache;" in script

    def test_a_temp_image_is_loop_mounted_at_the_temp_dir(self, slinky, monkeypatch):
        a, execs = slinky
        monkeypatch.setattr(A, "ECR_PROFILE", "")
        monkeypatch.setattr(A, "SLINKY_ENROOT_TEMP", "/var/tmp/e")
        monkeypatch.setattr(A, "SLINKY_ENROOT_TEMP_IMAGE", "/scratch/outpost/.images/t.ext4")
        a._stage_image("c", "login-0", OCI, "job-1")
        script, _ = execs[-1]
        assert "mount -o loop,noinit_itable /scratch/outpost/.images/t.ext4 /var/tmp/e" in script
        assert "ENROOT_TEMP_PATH=/var/tmp/e;" in script

    def test_a_temp_dir_without_an_image_is_used_as_is(self, slinky, monkeypatch):
        a, execs = slinky
        monkeypatch.setattr(A, "ECR_PROFILE", "")
        monkeypatch.setattr(A, "SLINKY_ENROOT_TEMP", "/enroot-tmp")
        monkeypatch.setattr(A, "SLINKY_ENROOT_TEMP_IMAGE", "")
        a._stage_image("c", "login-0", OCI, "job-1")
        script, _ = execs[-1]
        assert "ENROOT_TEMP_PATH=/enroot-tmp;" in script and "mount" not in script

    def test_a_failed_import_refuses(self, monkeypatch):
        a = SlinkyAdapter(log=lambda m: None)
        monkeypatch.setattr(A, "ECR_PROFILE", "ecr")
        monkeypatch.setattr(A, "registry_login", lambda image: ("AWS", "s3cret"))
        monkeypatch.setattr(a, "_exec", lambda *args, **kw: (1, "", ""))
        with pytest.raises(RuntimeError, match="enroot import") as e:
            a._stage_image("c", "login-0", OCI, "job-1")
        assert "CLUSTER_ECR_PROFILE" not in str(e.value)

    def test_a_failed_ecr_import_without_a_profile_names_the_setting(self, monkeypatch):
        a = SlinkyAdapter(log=lambda m: None)
        monkeypatch.setattr(A, "ECR_PROFILE", "")
        monkeypatch.setattr(a, "_exec", lambda *args, **kw: (1, "", ""))
        with pytest.raises(RuntimeError, match="set CLUSTER_ECR_PROFILE"):
            a._stage_image("c", "login-0", OCI, "job-1")

    def test_a_failed_public_import_does_not_name_the_ecr_setting(self, monkeypatch):
        a = SlinkyAdapter(log=lambda m: None)
        monkeypatch.setattr(A, "ECR_PROFILE", "")
        monkeypatch.setattr(a, "_exec", lambda *args, **kw: (1, "", ""))
        with pytest.raises(RuntimeError, match="enroot import") as e:
            a._stage_image("c", "login-0", "docker://nvcr.io/nvidia/hpc-benchmarks@sha256:"
                           + "a" * 64, "job-1")
        assert "CLUSTER_ECR_PROFILE" not in str(e.value)
