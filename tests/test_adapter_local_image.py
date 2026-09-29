"""An already-local image is run, not re-converted — on EVERY launch path.

`_ensure_remote_sif` exists to FETCH a ref with a scheme (docker://, oras://)
and convert it to a SIF. Handing it a path that is already a SIF does not fail
— apptainer will happily build a SIF out of a SIF — it just re-creates a
multi-GB image under a hashed cache name for nothing. On the gtl-qwen runtime
that is 4.6 GB and ~30 seconds per node, per new digest, on top of a transfer
that had already put the exact bytes in place.

The single-node path (`run`) has guarded this for a long time. The multi-node
path did not, so the same locally-staged image took the cheap route at
node_count 1 and the expensive one above it — a divergence that only shows up
on the shape which is hardest to get hardware for. These tests pin BOTH paths
to the same rule so they cannot drift apart again.
"""
from __future__ import annotations

import pytest
from reconciler.adapter import LibvirtAdapter
from reconciler.models import JobSpec, NodeRecord

LOCAL_SIF = "/opt/gtl-images/gtl-qwen-sft-runtime-1.2.0.sif"


def node(name, ip, cluster_ip=None) -> NodeRecord:
    return NodeRecord(node_id=name, name=name, index=1, ip=ip, cluster_ip=cluster_ip)


def spec(**kw) -> JobSpec:
    base = dict(name="local-image", image=LOCAL_SIF, command=["true"],
                launcher="mpi", node_count=2, output_dir="/out")
    return JobSpec(**{**base, **kw})


@pytest.fixture()
def staged(monkeypatch):
    """Record every image _ensure_remote_sif was asked to stage, and what the
    per-rank argv ended up pointing at."""
    calls: list[str] = []
    seen: dict = {}
    real = __import__("reconciler.adapter", fromlist=["container_argv"]).container_argv

    def spy_argv(job_spec, bind_src, bind_dst, extra_binds=None, extra_env=None):
        seen["image"] = job_spec.image
        return real(job_spec, bind_src, bind_dst, extra_binds, extra_env)

    def spy_sif(self, n, image, job_id):
        calls.append(image)
        return "/tmp/outpost-sif-cache/converted.sif"

    monkeypatch.setattr("reconciler.adapter.container_argv", spy_argv)
    monkeypatch.setattr("reconciler.adapter.run_logged", lambda *a, **kw: 0)
    monkeypatch.setattr(LibvirtAdapter, "_ensure_remote_sif", spy_sif)
    monkeypatch.setattr(LibvirtAdapter, "_scp_to", lambda self, *a, **kw: None)
    monkeypatch.setattr(LibvirtAdapter, "_node_iface", lambda self, n: "eth0")
    seen["staged"] = calls
    return seen


class TestMultiNodeLaunch:
    def test_a_local_sif_is_never_staged(self, staged):
        adapter = LibvirtAdapter(log=lambda msg: None)
        nodes = [node("n1", "10.0.0.11"), node("n2", "10.0.0.12")]
        adapter._run_launch(nodes, "job-1", spec(), "/tmp/cluster/job-1", None)

        assert staged["staged"] == [], (
            "a node-local .sif was sent through _ensure_remote_sif — that "
            "re-converts a SIF into a SIF, once per node, for no benefit")

    def test_the_ranks_run_the_path_that_was_given(self, staged):
        adapter = LibvirtAdapter(log=lambda msg: None)
        nodes = [node("n1", "10.0.0.11"), node("n2", "10.0.0.12")]
        adapter._run_launch(nodes, "job-1", spec(), "/tmp/cluster/job-1", None)

        assert staged["image"] == LOCAL_SIF

    def test_a_ref_with_a_scheme_is_still_staged_on_every_node(self, staged):
        """The guard must not go too far: a docker:// ref genuinely has to be
        fetched and converted, once per node (the cache is per node)."""
        adapter = LibvirtAdapter(log=lambda msg: None)
        nodes = [node("n1", "10.0.0.11"), node("n2", "10.0.0.12")]
        adapter._run_launch(nodes, "job-2", spec(image="docker://busybox:latest"),
                            "/tmp/cluster/job-2", None)

        assert staged["staged"] == ["docker://busybox:latest"] * 2
        assert staged["image"] == "/tmp/outpost-sif-cache/converted.sif"


class TestBothPathsAgree:
    def test_the_single_and_multi_node_paths_apply_the_same_rule(self):
        """The rule lives in two functions; this is what stops them drifting.

        Both must decide to stage on exactly the same condition — a ref with a
        scheme, and a runtime that is not docker.
        """
        import inspect

        from reconciler import adapter

        launch = inspect.getsource(adapter.LibvirtAdapter._run_launch)
        run = inspect.getsource(adapter.LibvirtAdapter.run)
        condition = 'spec.runtime != "docker" and "://" in spec.image'
        assert condition in launch, "_run_launch no longer guards on the scheme"
        assert condition in run, "run() no longer guards on the scheme"
