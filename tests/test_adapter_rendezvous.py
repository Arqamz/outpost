"""The multi-node launch path publishes MASTER_ADDR.

A framework that rendezvouses over TCP (torch.distributed/NCCL, JAX,
DeepSpeed) needs one address — rank 0's — and cannot derive it from inside the
container: it can see its own MPI rank but not whose node is rank 0, and the
hostfile that would say so is scp'd to the HEAD only, so a non-head rank
cannot read it. Only the launcher knows, because the launcher is what ordered
the nodes.
"""
from __future__ import annotations

import pytest
from reconciler.adapter import LibvirtAdapter
from reconciler.models import JobSpec, NodeRecord


def node(name, ip, cluster_ip=None) -> NodeRecord:
    return NodeRecord(node_id=name, name=name, index=1, ip=ip, cluster_ip=cluster_ip)


def spec(**kw) -> JobSpec:
    base = dict(name="rendezvous", image="docker://x", command=["true"],
                launcher="mpi", node_count=2, output_dir="/out")
    return JobSpec(**{**base, **kw})


@pytest.fixture()
def captured(monkeypatch):
    """Run _run_launch far enough to build the per-rank argv, capturing it."""
    seen: dict = {}
    real = __import__("reconciler.adapter", fromlist=["container_argv"]).container_argv

    def spy(job_spec, bind_src, bind_dst, extra_binds=None, extra_env=None):
        seen["extra_env"] = extra_env or {}
        return real(job_spec, bind_src, bind_dst, extra_binds, extra_env)

    monkeypatch.setattr("reconciler.adapter.container_argv", spy)
    monkeypatch.setattr("reconciler.adapter.run_logged", lambda *a, **kw: 0)
    monkeypatch.setattr(LibvirtAdapter, "_ensure_remote_sif",
                        lambda self, n, image, job_id, sif=None: "/tmp/cached.sif")
    monkeypatch.setattr(LibvirtAdapter, "_scp_to", lambda self, *a, **kw: None)
    monkeypatch.setattr(LibvirtAdapter, "_node_iface", lambda self, n: "eth0")
    return seen


class TestRendezvousAddress:
    def test_the_head_is_published_as_master_addr(self, captured, monkeypatch):
        adapter = LibvirtAdapter(log=lambda msg: None)
        nodes = [node("n1", "10.0.0.11"), node("n2", "10.0.0.12")]
        adapter._run_launch(nodes, "job-1", spec(), "/tmp/cluster/job-1", None)
        assert captured["extra_env"]["MASTER_ADDR"] == "10.0.0.11"

    def test_it_is_the_peer_address_not_the_ssh_target(self, captured):
        """mpi_ip, not ip. On a NAT'd cloud the orchestrator's ssh target is a
        public address that is bound on no interface of the guest, so a peer
        dialling it never connects — the same distinction the hostfile makes."""
        adapter = LibvirtAdapter(log=lambda msg: None)
        nodes = [node("n1", "203.0.113.7", cluster_ip="10.0.0.11"),
                 node("n2", "203.0.113.8", cluster_ip="10.0.0.12")]
        adapter._run_launch(nodes, "job-1", spec(), "/tmp/cluster/job-1", None)
        assert captured["extra_env"]["MASTER_ADDR"] == nodes[0].mpi_ip == "10.0.0.11"

    def test_the_port_is_left_to_the_payload(self, captured):
        """Every distributed framework has a default port and its own opinion;
        the address is the only thing it cannot work out for itself."""
        adapter = LibvirtAdapter(log=lambda msg: None)
        adapter._run_launch([node("n1", "10.0.0.11"), node("n2", "10.0.0.12")],
                            "job-1", spec(), "/tmp/cluster/job-1", None)
        assert "MASTER_PORT" not in captured["extra_env"]
