"""Reconciling a plan against what the ranks actually reported.

The rule: a receipt may say `verified` only when it really checked. Every gap —
a missing observation, an unparseable report, a device that could not be
confirmed — has to come out as `mismatched` or `unverified`, because a run whose
placement silently did not hold produces a number that looks entirely normal.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import REPO_DIR

from reconciler.placement import resolve
from reconciler.receipt import build_receipt, parse_binding_report
from reconciler.topology import normalize

FIXTURES = Path(REPO_DIR) / "tests" / "fixtures"
EXAMPLES = Path(REPO_DIR) / "contract" / "launch-intent" / "v1" / "examples"


def _merge(base: dict, over: dict) -> dict:
    out = dict(base)
    for key, value in over.items():
        out[key] = _merge(out[key], value) if isinstance(value, dict) and \
            isinstance(out.get(key), dict) else value
    return out


def intent(**over) -> dict:
    return _merge(json.loads((EXAMPLES / "one-rank-per-gpu.json").read_text()), over)


def plan_for(name: str = "dev_1gpu", **over):
    topo = normalize(json.loads((FIXTURES / f"topology_{name}.json").read_text()), "n1")
    return resolve(intent(**_merge({"cpu": {"cores_per_rank": 2}}, over)), [topo])


def observation(plan, rank_index: int = 0, **over) -> dict:
    """What a well-behaved rank would report for its own planned placement."""
    rank = plan.ranks[rank_index]
    doc = {
        "probe_version": "1", "hostname": "n1",
        "global_rank": rank.global_rank, "local_rank": rank.local_rank,
        "world_size": len(plan.ranks), "pid": 4501,
        "allowed_cpus": ",".join(str(c) for c in rank.cpu_ids),
        "allowed_mems": "0", "current_cpu": rank.cpu_ids[0] if rank.cpu_ids else 0,
        "online_cpus": "0-7",
        "cuda_visible_devices": rank.gpu_uuid or "",
        "driver_gpu_uuids": rank.gpu_uuid or "",
        "driver_gpu_count": 1,
    }
    return {**doc, **over}


class TestBindingReport:
    def test_a_bound_rank_is_parsed(self):
        text = ("[lima:04469] MCW rank 0 bound to socket 0[core 0[hwt 0]]: [B/././.]\n"
                "[lima:04469] MCW rank 1 bound to socket 0[core 1[hwt 0]]: [./B/./.]\n")
        parsed = parse_binding_report(text)
        assert parsed[0]["bound"] is True and parsed[0]["cores"] == [(0, 0)]
        assert parsed[1]["cores"] == [(0, 1)]

    def test_a_multi_core_binding_is_parsed(self):
        text = "[h:1] MCW rank 0 bound to socket 0[core 0[hwt 0]], socket 0[core 1[hwt 0]]: [BB/..]"
        assert parse_binding_report(text)[0]["cores"] == [(0, 0), (0, 1)]

    def test_an_unbound_rank_is_recognised(self):
        # The launcher stating outright that it bound nothing. No amount of
        # comparing masks afterwards establishes this as clearly.
        text = "[h:1] MCW rank 2 is not bound (or bound to all available processors)"
        parsed = parse_binding_report(text)
        assert parsed[2]["bound"] is False and "not bound" in parsed[2]["claim"]

    @pytest.mark.parametrize("text", [None, "", "unrelated mpirun chatter\n"])
    def test_nothing_parseable_yields_nothing(self, text):
        assert parse_binding_report(text) == {}


class TestVerified:
    def test_a_matching_run_verifies(self):
        plan = plan_for()
        receipt = build_receipt(plan, [observation(plan)])
        assert receipt.status == "verified"
        assert receipt.mismatches == ()
        assert receipt.plan_id == plan.plan_id

    def test_the_launcher_claim_is_carried_onto_the_rank(self):
        plan = plan_for()
        report = "[h:1] MCW rank 0 bound to socket 0[core 0[hwt 0]]: [B/./././././.]"
        receipt = build_receipt(plan, [observation(plan)], binding_report=report)
        assert receipt.status == "verified"
        assert "bound to socket 0" in receipt.ranks[0].launcher_claim

    def test_what_the_receipt_can_prove_about_gpus_is_stated(self):
        # CUDA_VISIBLE_DEVICES proves delivery, the driver list proves existence;
        # neither proves the application enumerated it. Saying so in the artifact
        # stops it being read as more than it is.
        receipt = build_receipt(plan_for(), [observation(plan_for())])
        assert "not CUDA" in receipt.sources["gpu_visibility_verified_by"]


class TestMismatch:
    def test_a_rank_that_cannot_run_where_planned(self):
        plan = plan_for()
        receipt = build_receipt(plan, [observation(plan, allowed_cpus="4,5")])
        assert receipt.status == "mismatched"
        assert any("cannot run where the plan says" in m for m in receipt.mismatches)

    def test_an_unbound_rank_is_caught_despite_being_a_superset(self):
        # THE case a naive subset check misses: allowed = every CPU contains the
        # planned CPUs, so "is the plan satisfiable" says yes while no binding
        # happened at all.
        plan = plan_for()
        receipt = build_receipt(plan, [observation(plan, allowed_cpus="0-7",
                                                   online_cpus="0-7")])
        assert receipt.status == "mismatched"
        assert any("binding was not applied" in m for m in receipt.mismatches)

    def test_the_launcher_admitting_it_did_not_bind_is_a_mismatch(self):
        plan = plan_for()
        receipt = build_receipt(
            plan, [observation(plan)],
            binding_report="[h:1] MCW rank 0 is not bound (or bound to all available)")
        assert receipt.status == "mismatched"
        assert any("reported this rank as not bound" in m for m in receipt.mismatches)

    def test_the_wrong_device_is_caught(self):
        plan = plan_for()
        receipt = build_receipt(plan, [observation(plan, cuda_visible_devices="GPU-someone-else")])
        assert receipt.status == "mismatched"
        assert any("CUDA_VISIBLE_DEVICES" in m for m in receipt.mismatches)

    def test_a_device_absent_from_the_node_is_caught(self):
        plan = plan_for()
        receipt = build_receipt(plan, [observation(plan, driver_gpu_uuids="GPU-different")])
        assert any("not among the devices the driver exposes" in m
                   for m in receipt.mismatches)

    def test_a_missing_observation_is_a_mismatch_not_a_pass(self):
        # Absence of evidence recorded as evidence of absence, deliberately.
        plan = plan_for("2socket_8gpu", cpu={"cores_per_rank": 8})
        receipt = build_receipt(plan, [observation(plan, i) for i in range(7)])
        assert receipt.status == "mismatched"
        assert any("placement is unknown" in m for m in receipt.mismatches)
        assert receipt.ranks[7].observed is False

    def test_more_processes_than_planned_is_a_mismatch(self):
        plan = plan_for()
        stray = observation(plan) | {"global_rank": 99}
        receipt = build_receipt(plan, [observation(plan), stray])
        assert any("more processes ran than were planned" in m for m in receipt.mismatches)


class TestUnverified:
    def test_expected_observations_that_never_arrive_are_a_mismatch(self):
        # Preflight was supposed to run and did not report: that contradicts the
        # expectation, so it is a failure rather than a shrug.
        receipt = build_receipt(plan_for(), [])
        assert receipt.status == "mismatched"
        assert all(not r.observed for r in receipt.ranks)

    def test_a_skipped_preflight_is_unverified_not_mismatched(self):
        # `require_preflight: false`. Nothing contradicted the plan and nothing
        # confirmed it. Calling that a mismatch would cry wolf; calling it
        # verified would be a lie. This is the case the third status exists for.
        receipt = build_receipt(plan_for(), [], preflight_ran=False)
        assert receipt.status == "unverified"
        assert receipt.mismatches == ()
        assert any("not confirmed" in n for n in receipt.notes)
        assert receipt.sources["preflight_ran"] is False

    def test_a_skipped_preflight_still_carries_the_launcher_claim(self):
        plan = plan_for()
        receipt = build_receipt(
            plan, [], preflight_ran=False,
            binding_report="[h:1] MCW rank 0 bound to socket 0[core 0[hwt 0]]: [B/.]")
        assert receipt.status == "unverified"
        assert "bound to socket 0" in receipt.ranks[0].launcher_claim

    def test_an_empty_plan_cannot_be_verified(self):
        from reconciler.placement import LaunchPlan
        empty = LaunchPlan("lp-x", {}, (), {"warnings": ()}, {})
        assert build_receipt(empty, []).status == "unverified"


class TestNotesRatherThanFailures:
    def test_smt_siblings_are_noted_not_failed(self):
        # Binding to a whole core admits both its SMT siblings while the plan
        # named one. Real, expected, and not a failure — but recorded, because
        # the plan does not carry sibling identity and guessing would be inventing.
        plan = plan_for()
        planned = plan.ranks[0].cpu_ids
        wider = ",".join(str(c) for c in list(planned) + [c + 4 for c in planned])
        receipt = build_receipt(plan, [observation(plan, allowed_cpus=wider)])
        assert receipt.status == "verified"
        assert any("SMT siblings" in n for n in receipt.notes)

    def test_a_differing_hostname_is_a_note(self):
        plan = plan_for()
        receipt = build_receipt(plan, [observation(plan, hostname="ip-10-0-0-1")])
        assert receipt.status == "verified"
        assert any("reported hostname" in n for n in receipt.notes)

    def test_running_outside_the_allowed_set_is_noted(self):
        plan = plan_for()
        receipt = build_receipt(plan, [observation(plan, current_cpu=99)])
        assert any("outside its own allowed set" in n for n in receipt.notes)


class TestSerialisation:
    def test_the_receipt_round_trips_to_a_plain_document(self):
        plan = plan_for()
        # What matters is the artifact: dataclass tuples must survive to JSON
        # (and to YAML) as plain arrays, since this is written to the drop zone
        # and read back by whoever is auditing the run.
        doc = json.loads(json.dumps(build_receipt(plan, [observation(plan)]).to_dict()))
        assert doc["status"] == "verified"
        assert doc["ranks"][0]["planned_cpu_ids"] == list(plan.ranks[0].cpu_ids)
        import yaml
        assert yaml.safe_load(yaml.safe_dump(doc))["plan_id"] == plan.plan_id


class TestProbeShipping:
    def test_the_probe_is_posix_sh(self):
        import shutil
        import subprocess
        path = Path(REPO_DIR) / "reconciler" / "probes" / "placement-probe.sh"
        assert path.is_file()
        assert subprocess.run([shutil.which("sh"), "-n", str(path)]).returncode == 0

    def test_the_probe_emits_valid_json_with_no_sources(self, tmp_path):
        # It runs wherever the workload runs; on a machine with none of these
        # sources it must still produce a document rather than half a line.
        import shutil
        import subprocess
        path = Path(REPO_DIR) / "reconciler" / "probes" / "placement-probe.sh"
        result = subprocess.run([shutil.which("sh"), str(path), str(tmp_path)],
                                capture_output=True, text=True)
        assert result.returncode == 0
        doc = json.loads(result.stdout)
        assert doc["probe_version"] == "1"
        written = list((tmp_path / "preflight").glob("rank-*.json"))
        assert len(written) == 1
        assert json.loads(written[0].read_text()) == doc

    def test_the_rank_is_taken_from_the_launcher_environment(self, tmp_path):
        import os
        import shutil
        import subprocess
        path = Path(REPO_DIR) / "reconciler" / "probes" / "placement-probe.sh"
        env = {**os.environ, "OMPI_COMM_WORLD_RANK": "3",
               "OMPI_COMM_WORLD_LOCAL_RANK": "1", "OMPI_COMM_WORLD_SIZE": "8"}
        subprocess.run([shutil.which("sh"), str(path), str(tmp_path)],
                       capture_output=True, text=True, env=env, check=True)
        doc = json.loads((tmp_path / "preflight" / "rank-3.json").read_text())
        assert (doc["global_rank"], doc["local_rank"], doc["world_size"]) == (3, 1, 8)
