"""Normalizing a raw topology probe into something a plan can be resolved against.

Every case here runs off a fixture, never off this machine — unit tests must not
need a GPU, a NUMA node, or Linux.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import REPO_DIR

from reconciler.topology import (
    NodeTopology,
    digest_of,
    expand_cpulist,
    normalize,
    parse_topo_matrix,
    probe_source,
)

FIXTURES = Path(REPO_DIR) / "tests" / "fixtures"


def raw(name: str) -> dict:
    return json.loads((FIXTURES / f"topology_{name}.json").read_text())


def topo(name: str, node: str = "n1") -> NodeTopology:
    return normalize(raw(name), node)


class TestCpuList:
    @pytest.mark.parametrize(("spec", "expected"), [
        ("0-3", (0, 1, 2, 3)),
        ("0,2,4", (0, 2, 4)),
        ("0-3,8,12-13", (0, 1, 2, 3, 8, 12, 13)),
        ("5", (5,)),
        ("", ()),
        (None, ()),
        ("garbage", ()),
        ("0-3,,7", (0, 1, 2, 3, 7)),
    ])
    def test_expansion(self, spec, expected):
        assert expand_cpulist(spec) == expected

    def test_overlapping_ranges_are_deduplicated_and_sorted(self):
        assert expand_cpulist("4-6,0-1,5") == (0, 1, 4, 5, 6)


class TestTopoMatrix:
    """`nvidia-smi topo -m` is the driver's own affinity view — the authoritative
    source for which cores are local to a device, and the only source for the
    GPU-to-GPU link class an NCCL collective actually rides."""

    def test_rows_and_affinity(self):
        parsed = parse_topo_matrix(raw("2socket_8gpu")["topo_matrix"])
        assert set(parsed) == {f"GPU{i}" for i in range(8)} | {"NIC0"}
        assert parsed["GPU0"]["numa"] == 0 and parsed["GPU7"]["numa"] == 1
        # socket 0 owns 0-23 and 48-71 (thread 0 of every core comes first)
        assert parsed["GPU0"]["cpu_affinity"][:3] == (0, 1, 2)
        assert 80 in parsed["GPU0"]["cpu_affinity"]
        assert parsed["GPU7"]["cpu_affinity"][:3] == (40, 41, 42)

    def test_link_classes_are_captured(self):
        parsed = parse_topo_matrix(raw("2socket_8gpu")["topo_matrix"])
        links = parsed["GPU0"]["links"]
        assert links["GPU1"] == "NV18"          # same socket, NVLink
        assert links["GPU4"] == "SYS"           # across the SMP interconnect
        assert links["GPU0"] == "X"             # self

    def test_a_matrix_without_nic_rows_parses(self):
        parsed = parse_topo_matrix(raw("dev_1gpu")["topo_matrix"])
        assert parsed["GPU0"]["cpu_affinity"] == (0, 1, 2, 3, 4, 5, 6, 7)
        assert parsed["GPU0"]["numa"] == 0

    def test_two_token_column_headers_are_one_column(self):
        # "CPU Affinity" is a single column whose name contains a space; a naive
        # whitespace split shifts every field after it by one.
        text = ("\tGPU0\tCPU Affinity\tNUMA Affinity\n"
                "GPU0\t X \t0-3\t1\n")
        assert parse_topo_matrix(text)["GPU0"] == {
            "cpu_affinity": (0, 1, 2, 3), "numa": 1, "links": {"GPU0": "X"}}

    def test_tab_and_space_separated_forms_agree(self):
        # The probe escapes the blob into JSON and converts tabs; both forms must
        # parse the same, or the meaning would depend on the transport.
        tabbed = "\tGPU0\tCPU Affinity\tNUMA Affinity\nGPU0\t X \t0-3\t1\n"
        spaced = tabbed.replace("\t", "    ")
        assert parse_topo_matrix(tabbed) == parse_topo_matrix(spaced)

    def test_unknown_numa_is_none_not_minus_one(self):
        # -1 means "the platform exposes no affinity", which must never become a
        # NUMA node id the resolver then tries to place memory on.
        text = "\tGPU0\tCPU Affinity\tNUMA Affinity\nGPU0\t X \t0-3\t-1\n"
        assert parse_topo_matrix(text)["GPU0"]["numa"] is None

    def test_missing_affinity_columns_are_tolerated(self):
        text = "\tGPU0\tGPU1\nGPU0\t X \tNV12\nGPU1\tNV12\t X \n"
        parsed = parse_topo_matrix(text)
        assert parsed["GPU0"]["cpu_affinity"] == () and parsed["GPU0"]["numa"] is None
        assert parsed["GPU0"]["links"]["GPU1"] == "NV12"

    def test_legend_is_not_parsed_as_data(self):
        parsed = parse_topo_matrix(raw("2socket_8gpu")["topo_matrix"])
        assert not any(k.startswith("Legend") or k == "X" for k in parsed)

    @pytest.mark.parametrize("text", [None, "", "no table here", "\n\n"])
    def test_unusable_input_yields_nothing(self, text):
        assert parse_topo_matrix(text) == {}


class TestNormalize:
    def test_physical_cores_group_smt_siblings(self):
        t = topo("2socket_8gpu")
        assert len(t.cpus) == 160 and len(t.cores) == 80
        assert t.cores[0].cpu_ids == (0, 80)      # thread 0 and thread 1 of one core
        assert all(len(c.cpu_ids) == 2 for c in t.cores)

    def test_numa_and_gpus_are_carried(self):
        t = topo("2socket_8gpu")
        assert [n.id for n in t.numa_nodes] == [0, 1]
        assert len(t.gpus) == 8
        assert t.gpus[0].uuid.startswith("GPU-")

    def test_driver_affinity_beats_sysfs_for_gpu_numa(self):
        # Both sources agree in the fixture; what is asserted is WHICH was used,
        # because the driver's view does not depend on inferring locality through
        # a NUMA id a VM may have invented.
        t = topo("2socket_8gpu")
        assert t.gpus[0].numa_source == "topo_matrix"
        assert t.gpus[0].cpu_affinity[:2] == (0, 1)

    def test_sysfs_is_the_fallback_when_no_matrix(self):
        doc = raw("2socket_8gpu")
        doc["topo_matrix"] = None
        t = normalize(doc, "n1")
        assert t.gpus[0].numa_source == "sysfs" and t.gpus[0].numa == 0
        assert t.gpus[0].cpu_affinity == ()

    def test_a_full_host_is_high_confidence(self):
        t = topo("2socket_8gpu")
        assert t.confidence == "high" and t.warnings == ()

    def test_a_guest_is_low_confidence_and_says_why(self):
        t = topo("vm_guest")
        assert t.scope == "guest" and t.confidence == "low"
        assert any("guest" in w for w in t.warnings)
        assert any("NUMA" in w for w in t.warnings)

    def test_a_container_is_low_confidence(self):
        assert topo("cgroup_restricted").confidence == "low"

    def test_an_opaque_node_reports_every_gap(self):
        t = topo("no_sources")
        assert t.confidence == "low"
        assert t.cores == () and t.numa_nodes == () and t.gpus == ()
        assert any("socket/core" in w for w in t.warnings)

    def test_missing_cpuset_is_assumed_wide_and_flagged(self):
        # "these exist" is not "the job may use these"; assuming the former
        # silently is how a plan binds outside its allocation.
        doc = raw("dev_1gpu")
        doc["allowed_cpus"] = None
        doc["online_cpus"] = None
        t = normalize(doc, "n1")
        assert t.allowed_cpu_ids == tuple(range(8))
        assert any("wider than the allocation" in w for w in t.warnings)

    def test_gpus_without_any_affinity_source_are_flagged(self):
        doc = raw("dev_1gpu")
        doc["topo_matrix"] = None
        doc["gpus"][0]["numa"] = None
        t = normalize(doc, "n1")
        assert any("nothing to work from" in w for w in t.warnings)


class TestAllowedCores:
    def test_a_cgroup_narrows_the_candidate_set(self):
        # The whole point of an allocation-aware probe: 160 CPUs exist, 16 are ours.
        t = topo("cgroup_restricted")
        assert t.allowed_cpu_ids == tuple(list(range(0, 8)) + list(range(80, 88)))
        cores = t.allowed_cores()
        assert len(cores) == 8
        assert all(set(c.cpu_ids) <= set(t.allowed_cpu_ids) for c in cores)

    def test_a_partially_allowed_core_keeps_only_the_allowed_units(self):
        doc = raw("dev_1gpu")
        doc["allowed_cpus"] = "0-3"          # thread 0 of cores 0-3, no siblings
        cores = normalize(doc, "n1").allowed_cores()
        assert [c.cpu_ids for c in cores] == [(0,), (1,), (2,), (3,)]

    def test_ordering_is_deterministic(self):
        first = [c.cpu_ids for c in topo("2socket_8gpu").allowed_cores()]
        second = [c.cpu_ids for c in topo("2socket_8gpu").allowed_cores()]
        assert first == second == sorted(first)


class TestDigest:
    def test_identical_snapshots_digest_identically(self):
        assert topo("2socket_8gpu").digest() == topo("2socket_8gpu").digest()

    def test_a_changed_machine_changes_the_digest(self):
        # This is what makes an approved plan detectably stale rather than
        # silently re-resolved against different hardware.
        doc = raw("2socket_8gpu")
        doc["gpus"][0]["uuid"] = "GPU-swapped-card"
        assert normalize(doc, "n1").digest() != topo("2socket_8gpu").digest()

    def test_key_order_does_not_affect_the_digest(self):
        assert digest_of({"a": 1, "b": 2}) == digest_of({"b": 2, "a": 1})


class TestProbeShipping:
    def test_the_probe_travels_with_the_code(self):
        source = probe_source()
        assert "probe_version" in source and "nvidia-smi" in source

    def test_the_probe_is_posix_sh(self):
        # It runs on whatever the node has; bashisms would fail on a minimal image.
        import shutil
        import subprocess
        sh = shutil.which("sh")
        from reconciler.topology import PROBE_PATH
        assert subprocess.run([sh, "-n", str(PROBE_PATH)]).returncode == 0
