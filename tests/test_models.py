"""Record shapes — the intake contract's parsing half.

`JobSpec.from_dict` is what every external caller's document passes through, so
its tolerance rules ARE the contract, whether or not they were chosen
deliberately. They are pinned here, including the two asymmetries that are
easy to trip over.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from conftest import REPO_DIR

from reconciler.models import JobRecord, JobSpec, NodeRecord, RunResult, new_id

EXAMPLE = Path(REPO_DIR) / "contract" / "launch-intent" / "v1" / "examples" / \
    "one-rank-per-gpu.json"


class TestJobSpec:
    def test_minimal_document_gets_documented_defaults(self):
        spec = JobSpec.from_dict({"name": "j"})
        assert (spec.image, spec.command, spec.runtime, spec.launcher) == ("", [], "apptainer",
                                                                          "single")
        assert (spec.node_count, spec.gpu, spec.hybrid) == (1, False, False)
        assert (spec.output_dir, spec.env, spec.params, spec.workload) == ("/out", {}, {}, "")

    def test_command_may_arrive_as_a_string(self):
        assert JobSpec.from_dict({"name": "j", "command": "echo hi"}).command == ["echo", "hi"]

    def test_env_values_are_coerced_to_strings(self):
        # YAML happily yields ints/bools here; the container env cannot take them.
        spec = JobSpec.from_dict({"name": "j", "env": {"RANKS": 8, "DEBUG": True}})
        assert spec.env == {"RANKS": "8", "DEBUG": "True"}

    def test_null_fields_fall_back_to_defaults(self):
        # `image: ` in YAML parses as None, not "" — the `or` guards exist for this.
        spec = JobSpec.from_dict({"name": "j", "image": None, "launcher": None, "params": None})
        assert (spec.image, spec.launcher, spec.params) == ("", "single", {})

    def test_empty_image_is_the_dry_run_marker(self):
        assert JobSpec.from_dict({"name": "j"}).is_dry_run
        assert not JobSpec.from_dict({"name": "j", "image": "docker://x"}).is_dry_run

    def test_unknown_spec_key_is_still_silently_dropped(self):
        # Deliberate for the OPEN part of the interface: the cluster runs any
        # container, and a caller's extra bookkeeping key is not its business.
        # `launch` is carved out of this rule — see test_launch_intent.py.
        spec = JobSpec.from_dict({"name": "j", "seasoning": "paprika"})
        assert not hasattr(spec, "seasoning")

    def test_launch_block_is_carried_not_dropped(self):
        # The carve-out. A placement request that vanished would produce a run
        # that succeeds, parses, and measures a configuration nobody chose.
        intent = json.loads(EXAMPLE.read_text())
        spec = JobSpec.from_dict({"name": "j", "launch": intent})
        assert spec.launch == intent

    def test_absent_launch_is_none(self):
        assert JobSpec.from_dict({"name": "j"}).launch is None

    def test_missing_name_is_the_one_hard_error(self):
        with pytest.raises(KeyError):
            JobSpec.from_dict({"image": "docker://x"})


class TestJobRecord:
    def test_roundtrip(self):
        rec = JobRecord.create(JobSpec(name="j", image="docker://x"))
        assert JobRecord.from_dict(rec.to_dict()) == rec
        assert rec.job_id.startswith("job-")

    def test_unknown_top_level_key_raises(self):
        # from_dict is JobRecord(**d): unlike JobSpec, an unrecognised key here is
        # fatal and breaks EVERY read of the store, not just this record. Anything
        # new therefore has to be a declared field, never a passenger.
        with pytest.raises(TypeError):
            JobRecord.from_dict({"job_id": "job-1", "spec": {}, "surprise": 1})

    def test_missing_optional_keys_are_tolerated(self):
        # The other half of the asymmetry, and what makes adding a field to
        # JobRecord safe for records written before it existed: absent keys fall
        # back to their defaults, so no store migration is needed.
        rec = JobRecord.from_dict({"job_id": "job-1", "spec": {"name": "j"}})
        assert rec.state == "submitted"
        assert (rec.run, rec.drop_path, rec.error) == (None, None, None)
        assert rec.assigned_nodes == []

    def test_mongo_object_id_is_stripped(self):
        rec = JobRecord.from_dict({"_id": "abc", "job_id": "job-1", "spec": {}})
        assert rec.job_id == "job-1"


class TestNodeRecord:
    def test_unknown_keys_are_filtered_not_fatal(self):
        # NodeRecord.from_dict whitelists, so a record written by a newer version
        # still loads on an older one — deliberate, and the opposite of JobRecord.
        node = NodeRecord.from_dict({"node_id": "n1", "name": "n1", "index": 1,
                                     "ip": "10.0.0.1", "future_field": "x"})
        assert node.name == "n1"

    @pytest.mark.parametrize(("provider", "local", "expected"), [
        ("", False, "libvirt"),          # default routing for a plain VM
        ("", True, "local"),             # the control-plane host
        ("static-ssh", False, "static-ssh"),
        ("static-ssh", True, "static-ssh"),   # an explicit provider always wins
    ])
    def test_adapter_key_routing(self, provider, local, expected):
        node = NodeRecord(node_id="n", name="n", index=1, ip="10.0.0.1",
                          provider=provider, local=local)
        assert node.adapter_key == expected


class TestMisc:
    def test_new_id_is_prefixed_and_unique(self):
        ids = {new_id("job") for _ in range(100)}
        assert len(ids) == 100
        assert all(i.startswith("job-") for i in ids)

    def test_run_result_serialises(self):
        assert RunResult("job-1", "n1", 0).to_dict()["exit_code"] == 0
