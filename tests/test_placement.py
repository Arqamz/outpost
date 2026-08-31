"""Resolving an intent against real topology into an exact per-rank plan.

The resolver is the one place a semantic request meets actual hardware, so these
cover both halves: that a satisfiable request resolves to the right cores and
devices, and that an unsatisfiable one is REFUSED rather than approximated. A
benchmark that quietly ran on the far socket produces a number nothing
downstream can tell is wrong.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import REPO_DIR

from reconciler.placement import RESOLVER_VERSION, LaunchPlan, PlacementError, resolve
from reconciler.topology import normalize

FIXTURES = Path(REPO_DIR) / "tests" / "fixtures"
EXAMPLES = Path(REPO_DIR) / "contract" / "launch-intent" / "v1" / "examples"


def topo(name: str, node: str = "n1"):
    return normalize(json.loads((FIXTURES / f"topology_{name}.json").read_text()), node)


def dev_node(name: str, **over):
    """The dev box as a DISTINCT machine.

    GPU UUIDs are globally unique, so two nodes can never report the same one —
    the resolver refuses that outright, and reusing one fixture for two nodes
    would be testing against a topology that cannot exist.
    """
    doc = json.loads((FIXTURES / "topology_dev_1gpu.json").read_text())
    doc["hostname"] = name
    doc["gpus"][0]["uuid"] = f"GPU-{name}0001-0000-0000-0000-000000000001"
    doc.update(over)
    return normalize(doc, name)


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and \
            isinstance(out.get(key), dict) else value
    return out


def intent(**over) -> dict:
    """The canonical one-rank-per-GPU intent, with targeted overrides."""
    base = json.loads((EXAMPLES / "one-rank-per-gpu.json").read_text())
    return _merge(base, over)


class TestOneRankPerGpu:
    def test_eight_gpus_yield_eight_ranks(self):
        plan = resolve(intent(), [topo("2socket_8gpu")])
        assert len(plan.ranks) == 8
        assert [r.global_rank for r in plan.ranks] == list(range(8))
        assert [r.local_rank for r in plan.ranks] == list(range(8))

    def test_each_rank_gets_a_distinct_gpu(self):
        plan = resolve(intent(), [topo("2socket_8gpu")])
        uuids = [r.gpu_uuids[0] for r in plan.ranks]
        assert len(set(uuids)) == 8 and all(uuids)

    def test_cores_are_local_to_the_assigned_gpu(self):
        # The headline behaviour: socket 0's GPUs get socket 0's cores.
        plan = resolve(intent(), [topo("2socket_8gpu")])
        for rank in plan.ranks[:4]:
            assert all(cpu < 40 or 80 <= cpu < 120 for cpu in rank.cpu_ids), rank
            assert rank.numa_nodes == (0,)
        for rank in plan.ranks[4:]:
            assert all(40 <= cpu < 80 or cpu >= 120 for cpu in rank.cpu_ids), rank
            assert rank.numa_nodes == (1,)

    def test_each_rank_gets_the_requested_core_count(self):
        plan = resolve(intent(), [topo("2socket_8gpu")])
        assert all(len(r.cpu_ids) == 8 for r in plan.ranks)

    def test_no_two_ranks_share_a_cpu(self):
        plan = resolve(intent(), [topo("2socket_8gpu")])
        allocated = [cpu for r in plan.ranks for cpu in r.cpu_ids]
        assert len(allocated) == len(set(allocated))

    def test_visible_index_is_zero_for_a_single_device(self):
        # One device made visible per rank means the application sees it at 0,
        # whatever its physical enumeration — that is the point of pinning
        # visibility rather than trusting device order.
        plan = resolve(intent(), [topo("2socket_8gpu")])
        assert {r.visible_gpu_indices for r in plan.ranks} == {(0,)}

    def test_pci_addresses_are_carried_for_verification(self):
        plan = resolve(intent(), [topo("2socket_8gpu")])
        assert all(r.gpu_pci_bus_ids for r in plan.ranks)


class TestSmt:
    def test_physical_only_takes_one_unit_per_core(self):
        # 8 physical cores per rank, one processing unit each — and the sibling
        # is reserved, so no other rank can land on the same physical core.
        plan = resolve(intent(), [topo("2socket_8gpu")])
        first = plan.ranks[0]
        assert len(first.cpu_ids) == 8
        assert all(cpu < 80 for cpu in first.cpu_ids)     # thread 0 of each core
        siblings = {cpu + 80 for cpu in first.cpu_ids}
        others = {cpu for r in plan.ranks[1:] for cpu in r.cpu_ids}
        assert not siblings & others, "a sibling of a reserved core was reused"

    def test_allow_siblings_hands_out_both_units(self):
        plan = resolve(intent(cpu={"smt": "allow_siblings", "cores_per_rank": 2}),
                       [topo("2socket_8gpu")])
        first = plan.ranks[0]
        assert len(first.cpu_ids) == 4                    # 2 cores x 2 threads
        assert first.cpu_ids == (0, 1, 80, 81)


class TestDeterminism:
    def test_the_same_inputs_produce_the_same_plan(self):
        # §7.3: an approved plan must be the plan that runs.
        a = resolve(intent(), [topo("2socket_8gpu")])
        b = resolve(intent(), [topo("2socket_8gpu")])
        assert a.plan_id == b.plan_id
        assert a.to_dict() == b.to_dict()

    def test_a_changed_intent_changes_the_plan_id(self):
        a = resolve(intent(), [topo("2socket_8gpu")])
        b = resolve(intent(cpu={"cores_per_rank": 4}), [topo("2socket_8gpu")])
        assert a.plan_id != b.plan_id

    def test_the_plan_records_what_it_was_resolved_from(self):
        plan = resolve(intent(), [topo("2socket_8gpu")], allocation_id="alloc-1")
        assert set(plan.digests) == {"intent", "topology", "allocation", "resolver"}
        assert plan.digests["resolver"] == RESOLVER_VERSION
        assert plan.launcher == {"type": "openmpi", "version": "4.1.6"}

    def test_node_order_does_not_depend_on_input_order(self):
        nodes = [dev_node("b"), dev_node("a")]
        plan = resolve(intent(cpu={"cores_per_rank": 2}), nodes)
        assert [r.node for r in plan.ranks] == ["a", "b"]


class TestMultiNode:
    def test_global_ranks_are_node_major(self):
        plan = resolve(intent(cpu={"cores_per_rank": 2}),
                       [dev_node("a"), dev_node("b")])
        assert [(r.global_rank, r.node, r.local_rank) for r in plan.ranks] == [
            (0, "a", 0), (1, "b", 0)]

    def test_launcher_disagreement_is_refused(self):
        a = dev_node("a")
        b = dev_node("b", launcher={"type": "mpich", "version": "4.2"})
        with pytest.raises(PlacementError, match="disagree on the launcher"):
            resolve(intent(cpu={"cores_per_rank": 2}), [a, b])

    def test_a_version_mismatch_is_only_a_warning(self):
        a = dev_node("a")
        b = dev_node("b", launcher={"type": "openmpi", "version": "5.0.1"})
        plan = resolve(intent(cpu={"cores_per_rank": 2}), [a, b])
        assert any("different launcher versions" in w
                   for w in plan.validation["warnings"])


class TestDerivedRankCount:
    def test_null_ranks_per_node_follows_the_gpu_count(self):
        assert len(resolve(intent(cpu={"cores_per_rank": 2}),
                           [topo("dev_1gpu")]).ranks) == 1
        assert len(resolve(intent(), [topo("2socket_8gpu")]).ranks) == 8

    def test_null_ranks_with_no_gpus_is_refused(self):
        with pytest.raises(PlacementError, match="no GPUs were discovered"):
            resolve(intent(), [topo("vm_guest")])

    def test_an_explicit_count_beyond_the_gpus_is_refused(self):
        # GPUs are assigned before cores, so this is the GPU shortfall, not
        # the (separately tested) core shortfall.
        with pytest.raises(PlacementError, match="only 1 GPU"):
            resolve(intent(process={"ranks_per_node": 2},
                           cpu={"cores_per_rank": 2}), [topo("dev_1gpu")])


class TestCapacityRefusals:
    def test_not_enough_cores_is_refused(self):
        # 4 physical cores on the dev box; asking for 8 cannot be satisfied.
        with pytest.raises(PlacementError, match="needs 8 core"):
            resolve(intent(), [topo("dev_1gpu")])

    def test_a_cgroup_limits_what_can_be_promised(self):
        # 160 CPUs exist; 8 physical cores are ours, so 8 ranks x 8 cores cannot fit.
        with pytest.raises(PlacementError, match="core"):
            resolve(intent(), [topo("cgroup_restricted")])

    def test_a_narrower_request_fits_inside_the_cgroup(self):
        plan = resolve(intent(process={"ranks_per_node": 2},
                              cpu={"cores_per_rank": 4, "strategy": "none",
                                   "binding_unit": "none"},
                              memory={"strategy": "none"}),
                       [topo("cgroup_restricted")])
        allowed = set(topo("cgroup_restricted").allowed_cpu_ids)
        assert all(set(r.cpu_ids) <= allowed for r in plan.ranks)

    def test_an_opaque_node_cannot_be_planned_against(self):
        with pytest.raises(PlacementError):
            resolve(intent(), [topo("no_sources")])

    def test_no_nodes_at_all_is_refused(self):
        with pytest.raises(PlacementError, match="no allocated nodes"):
            resolve(intent(), [])


class TestGpuLocalityRefusals:
    def test_closest_to_gpu_without_locality_is_refused_under_strict(self):
        doc = json.loads((FIXTURES / "topology_dev_1gpu.json").read_text())
        doc["topo_matrix"] = None
        doc["gpus"][0]["numa"] = None
        with pytest.raises(PlacementError, match="no cores local to GPU"):
            resolve(intent(cpu={"cores_per_rank": 2}), [normalize(doc, "n1")])

    def test_advisory_mode_falls_back_and_says_so(self):
        doc = json.loads((FIXTURES / "topology_dev_1gpu.json").read_text())
        doc["topo_matrix"] = None
        doc["gpus"][0]["numa"] = None
        plan = resolve(intent(cpu={"cores_per_rank": 2},
                              validation={"mode": "advisory"}), [normalize(doc, "n1")])
        assert len(plan.ranks) == 1 and len(plan.ranks[0].cpu_ids) == 2
        assert any("falling back" in w for w in plan.validation["warnings"])

    def test_memory_locality_without_numa_is_refused_under_strict(self):
        with pytest.raises(PlacementError, match="memory.strategy"):
            resolve(intent(process={"ranks_per_node": 1},
                           cpu={"cores_per_rank": 1, "strategy": "none"},
                           gpu={"strategy": "none", "gpus_per_rank": 0}),
                    [topo("vm_guest")])

    def test_memory_locality_without_numa_is_a_warning_under_advisory(self):
        plan = resolve(intent(process={"ranks_per_node": 1},
                              cpu={"cores_per_rank": 1, "strategy": "none"},
                              gpu={"strategy": "none", "gpus_per_rank": 0},
                              validation={"mode": "advisory"}),
                       [topo("vm_guest")])
        assert any("cannot be expressed" in w for w in plan.validation["warnings"])

    def test_guest_warnings_reach_the_plan(self):
        # Low-confidence topology is not hidden behind a valid-looking plan.
        plan = resolve(intent(process={"ranks_per_node": 1},
                              cpu={"cores_per_rank": 1, "strategy": "none"},
                              memory={"strategy": "none"},
                              gpu={"strategy": "none", "gpus_per_rank": 0}),
                       [topo("vm_guest")])
        assert any("guest" in w for w in plan.validation["warnings"])


class TestExplicitPlacement:
    def _explicit(self, **over) -> dict:
        base = json.loads((EXAMPLES / "explicit-placement.json").read_text())
        return _merge(base, over)

    def test_explicit_cpus_and_gpus_are_honoured_verbatim(self):
        t = topo("2socket_8gpu")
        doc = self._explicit(gpu={"explicit_gpu_uuids": [[t.gpus[0].uuid], [t.gpus[1].uuid]]})
        plan = resolve(doc, [t])
        assert [r.cpu_ids for r in plan.ranks] == [(0, 1, 2, 3), (8, 9, 10, 11)]
        assert [r.gpu_uuids[0] for r in plan.ranks] == [t.gpus[0].uuid, t.gpus[1].uuid]

    def test_cpus_outside_the_allocation_are_refused(self):
        t = topo("cgroup_restricted")
        doc = self._explicit(
            cpu={"explicit_cpu_ids": [[0, 1, 2, 3], [90, 91, 92, 93]]},
            gpu={"explicit_gpu_uuids": [[t.gpus[0].uuid], [t.gpus[1].uuid]]})
        with pytest.raises(PlacementError, match="not in this allocation"):
            resolve(doc, [t])

    def test_overlapping_explicit_cpus_are_refused(self):
        t = topo("2socket_8gpu")
        doc = self._explicit(
            cpu={"explicit_cpu_ids": [[0, 1, 2, 3], [3, 4, 5, 6]]},
            gpu={"explicit_gpu_uuids": [[t.gpus[0].uuid], [t.gpus[1].uuid]]})
        with pytest.raises(PlacementError, match="already assigned"):
            resolve(doc, [t])

    def test_overlap_is_allowed_when_asked_for(self):
        t = topo("2socket_8gpu")
        doc = self._explicit(
            cpu={"explicit_cpu_ids": [[0, 1, 2, 3], [3, 4, 5, 6]], "allow_overlap": True},
            gpu={"explicit_gpu_uuids": [[t.gpus[0].uuid], [t.gpus[1].uuid]]})
        assert len(resolve(doc, [t]).ranks) == 2

    def test_an_unknown_gpu_uuid_is_refused(self):
        t = topo("2socket_8gpu")
        doc = self._explicit(gpu={"explicit_gpu_uuids": [[t.gpus[0].uuid], ["GPU-not-here"]]})
        with pytest.raises(PlacementError, match="not on this node"):
            resolve(doc, [t])

    def test_a_duplicate_gpu_is_refused(self):
        t = topo("2socket_8gpu")
        doc = self._explicit(gpu={"explicit_gpu_uuids": [[t.gpus[0].uuid], [t.gpus[0].uuid]]})
        with pytest.raises(PlacementError, match="more than one rank"):
            resolve(doc, [t])

    def test_sharing_is_allowed_when_asked_for(self):
        t = topo("2socket_8gpu")
        doc = self._explicit(gpu={"explicit_gpu_uuids": [[t.gpus[0].uuid], [t.gpus[0].uuid]],
                                  "allow_sharing": True})
        assert {r.gpu_uuids[0] for r in resolve(doc, [t]).ranks} == {t.gpus[0].uuid}

    def test_too_few_explicit_entries_for_the_rank_count_is_refused(self):
        t = topo("2socket_8gpu")
        doc = self._explicit(process={"ranks_per_node": 3},
                             gpu={"explicit_gpu_uuids": [[t.gpus[0].uuid], [t.gpus[1].uuid]]})
        with pytest.raises(PlacementError, match="entr"):
            resolve(doc, [t])


class TestMultiGpuPerRank:
    """gpus_per_rank > 1: one rank owns several GPUs (e.g. HPL's own internal
    mpirun fanning out across every GPU its one container is given). No real
    hardware has ≥2 GPUs on one node today, so this is proven in-process only,
    the same way the existing intra-node multi-GPU gap is already documented
    as untestable on real hardware elsewhere."""

    def test_one_per_rank_takes_a_contiguous_slice_per_rank(self):
        t = topo("2socket_8gpu")
        doc = intent(process={"ranks_per_node": 2}, cpu={"cores_per_rank": 2},
                    gpu={"gpus_per_rank": 2, "strategy": "one_per_rank"})
        plan = resolve(doc, [t])
        assert len(plan.ranks) == 2
        assert [r.gpu_uuids for r in plan.ranks] == [
            tuple(g.uuid for g in t.gpus[0:2]),
            tuple(g.uuid for g in t.gpus[2:4]),
        ]
        assert plan.ranks[0].visible_gpu_indices == (0, 1)
        assert plan.ranks[1].visible_gpu_indices == (0, 1)

    def test_explicit_strategy_honours_a_multi_gpu_list_per_rank(self):
        t = topo("2socket_8gpu")
        doc = json.loads((EXAMPLES / "explicit-placement.json").read_text())
        doc = _merge(doc, {
            "process": {"ranks_per_node": 1},
            "cpu": {"explicit_cpu_ids": [[0, 1, 2, 3]]},
            "gpu": {"gpus_per_rank": 2, "explicit_gpu_uuids": [[t.gpus[3].uuid, t.gpus[0].uuid]]},
        })
        plan = resolve(doc, [t])
        assert plan.ranks[0].gpu_uuids == (t.gpus[3].uuid, t.gpus[0].uuid)


class TestDeviceOrdering:
    def test_pci_bus_order_is_stable(self):
        plan = resolve(intent(), [topo("2socket_8gpu")])
        pci = [r.gpu_pci_bus_ids[0] for r in plan.ranks]
        assert pci == sorted(pci)

    def test_fastest_first_prefers_the_larger_device(self):
        doc = json.loads((FIXTURES / "topology_2socket_8gpu.json").read_text())
        doc["gpus"][3]["memory_mib"] = 200000          # the odd big card
        t = normalize(doc, "n1")
        plan = resolve(intent(gpu={"ordering": "fastest_first"},
                              cpu={"cores_per_rank": 2}), [t])
        assert plan.ranks[0].gpu_uuids[0] == doc["gpus"][3]["uuid"]


class TestNoOpinion:
    def test_a_no_placement_intent_resolves_to_bare_ranks(self):
        plan = resolve(intent(process={"ranks_per_node": 2},
                              cpu={"strategy": "none", "cores_per_rank": None,
                                   "binding_unit": "none"},
                              memory={"strategy": "none"},
                              gpu={"strategy": "none", "gpus_per_rank": 0}),
                       [topo("vm_guest")])
        assert len(plan.ranks) == 2
        assert all(r.cpu_ids == () and r.gpu_uuids == () for r in plan.ranks)


class TestGlobalConsistency:
    def test_two_nodes_claiming_the_same_gpu_is_refused(self):
        # GPU UUIDs are globally unique. Two nodes reporting one means the probe
        # or the allocation is lying, and a plan built on it would place two
        # ranks on hardware that cannot hold both.
        doc = json.loads((FIXTURES / "topology_dev_1gpu.json").read_text())
        clone_a, clone_b = normalize(doc, "a"), normalize(doc, "b")
        with pytest.raises(PlacementError, match="assigned to both rank"):
            resolve(intent(cpu={"cores_per_rank": 2}), [clone_a, clone_b])


class TestSerialization:
    def test_from_dict_survives_a_json_round_trip(self):
        # job.plan is stored as JSON (FileStore/Mongo): every tuple field comes
        # back as a list. from_dict must reconstruct the ACTUAL field types
        # (RankPlacement.cpu_ids etc. as tuples), not hand a launcher adapter
        # list-shaped-like-tuples — the exact bug this guards.
        plan = resolve(intent(), [topo("2socket_8gpu")])

        # json.dumps/loads is the FileStore's exact round trip (store.py).
        roundtripped = json.loads(json.dumps(plan.to_dict()))
        rebuilt = LaunchPlan.from_dict(roundtripped)

        assert rebuilt == plan
        for rank in rebuilt.ranks:
            assert isinstance(rank.cpu_ids, tuple)
            assert isinstance(rank.cpu_slots, tuple)
            assert all(isinstance(slot, tuple) for slot in rank.cpu_slots)
            assert isinstance(rank.numa_nodes, tuple)
