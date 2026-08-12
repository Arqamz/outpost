"""Parsing and refusing the `launch` block.

Two distinct refusals, deliberately not merged:
  * the document is wrong (bad version, malformed) -> the caller's bug;
  * the document is fine but this backend cannot resolve it yet -> ours.
Both stop the job. Only the message differs, and the message is the whole point:
an operator has to be able to tell "fix your spec" from "this cluster is too old
for it" without reading the source.
"""
from __future__ import annotations

import json
from copy import deepcopy
from pathlib import Path

import pytest
from conftest import REPO_DIR

from reconciler import launch_models
from reconciler.launch_models import SCHEMA_VERSION, UnsupportedLaunchIntent, parse
from reconciler.models import JobSpec

EXAMPLES = Path(REPO_DIR) / "contract" / "launch-intent" / "v1" / "examples"


def intent(name: str = "one-rank-per-gpu.json") -> dict:
    return json.loads((EXAMPLES / name).read_text())


class TestAcceptance:
    @pytest.mark.parametrize("name", ["one-rank-per-gpu.json", "explicit-placement.json"])
    def test_shipped_examples_parse(self, name):
        # The examples are what the schema documents; if the validator disagreed
        # with them, one of the two is wrong and the contract is not a contract.
        assert parse(intent(name)) == intent(name)

    def test_parse_returns_the_block_unchanged(self):
        doc = intent()
        assert parse(doc) is doc

    def test_validator_reads_the_shipped_schema(self):
        # Not a restated copy: the vocabulary lives in one file, and this is what
        # keeps it that way.
        assert launch_models.SCHEMA_PATH.is_file()
        assert launch_models.schema()["properties"]["schema_version"]["const"] == SCHEMA_VERSION


class TestVersionBoundary:
    def test_future_version_is_refused_by_name(self):
        doc = intent() | {"schema_version": "launch-intent/v99"}
        with pytest.raises(UnsupportedLaunchIntent) as e:
            parse(doc)
        # The operator needs the version they sent AND the one we speak.
        assert "launch-intent/v99" in str(e.value)
        assert SCHEMA_VERSION in str(e.value)

    def test_missing_version_is_refused(self):
        doc = intent()
        del doc["schema_version"]
        with pytest.raises(UnsupportedLaunchIntent, match="schema_version"):
            parse(doc)

    def test_version_is_reported_before_anything_else(self):
        # A v2 document will also trip every unknown-key rule. Reporting those
        # first would bury the one fact that matters, so the version check runs
        # alone and returns immediately.
        doc = {"schema_version": "launch-intent/v2", "totally": "different"}
        message = str(pytest.raises(UnsupportedLaunchIntent, parse, doc).value)
        assert "launch-intent/v2" in message
        assert "unknown key" not in message


class TestMalformed:
    def test_unknown_top_level_key_is_refused(self):
        doc = intent() | {"nic": {"strategy": "closest"}}
        with pytest.raises(UnsupportedLaunchIntent, match="nic"):
            parse(doc)

    def test_unknown_nested_key_is_refused(self):
        # A launcher flag has no place in a semantic intent; catching the typo
        # here is the difference between "rejected" and "silently not applied".
        doc = deepcopy(intent())
        doc["cpu"]["bind_to"] = "socket"
        with pytest.raises(UnsupportedLaunchIntent, match="bind_to"):
            parse(doc)

    def test_missing_block_is_refused(self):
        doc = intent()
        del doc["gpu"]
        with pytest.raises(UnsupportedLaunchIntent, match="gpu"):
            parse(doc)

    def test_unknown_enum_value_is_refused(self):
        doc = deepcopy(intent())
        doc["cpu"]["strategy"] = "closest_to_nic"        # a planned policy, not v1
        with pytest.raises(UnsupportedLaunchIntent, match="closest_to_nic"):
            parse(doc)

    def test_wrong_type_is_refused(self):
        doc = deepcopy(intent())
        doc["process"]["threads_per_rank"] = "eight"
        with pytest.raises(UnsupportedLaunchIntent, match="threads_per_rank"):
            parse(doc)

    def test_boolean_is_not_an_integer(self):
        # bool subclasses int in Python but not in JSON Schema; True would
        # otherwise sail through as a rank count.
        doc = deepcopy(intent())
        doc["process"]["threads_per_rank"] = True
        with pytest.raises(UnsupportedLaunchIntent, match="threads_per_rank"):
            parse(doc)

    def test_non_mapping_is_refused(self):
        with pytest.raises(UnsupportedLaunchIntent, match="mapping"):
            parse(["cpu", "gpu"])

    def test_every_problem_is_reported_at_once(self):
        # One round trip per fix is a bad way to correct a spec.
        doc = deepcopy(intent())
        doc["cpu"]["strategy"] = "nonsense"
        doc["memory"]["strategy"] = "nonsense"
        del doc["validation"]
        message = str(pytest.raises(UnsupportedLaunchIntent, parse, doc).value)
        assert "cpu.strategy" in message and "memory.strategy" in message
        assert "validation" in message


class TestPlanningSupportGate:
    def test_a_valid_intent_is_refused_while_planning_is_unimplemented(self, monkeypatch):
        # Proves the gate is the flag and nothing else — a backend with no
        # resolver at all (or one that has since regressed) refuses cleanly.
        monkeypatch.setattr(launch_models, "PLANNING_SUPPORTED", False)
        reason = launch_models.unsupported_reason(intent())
        assert reason and "cannot resolve a launch plan yet" in reason

    def test_no_intent_is_always_supported(self):
        assert launch_models.unsupported_reason(None) is None

    def test_a_valid_intent_is_accepted_now_that_planning_has_landed(self):
        # PLANNING_SUPPORTED is True by default — reconciler.py's _phase_plan
        # is the real resolver, not a monkeypatched stand-in.
        assert launch_models.PLANNING_SUPPORTED is True
        assert launch_models.unsupported_reason(intent()) is None


class TestThroughJobSpec:
    def test_valid_intent_survives_the_document_round_trip(self):
        doc = {"name": "j", "image": "docker://x", "launch": intent()}
        assert JobSpec.from_dict(doc).launch == intent()

    def test_malformed_intent_raises_out_of_from_dict(self):
        doc = {"name": "j", "launch": {"schema_version": "launch-intent/v1"}}
        with pytest.raises(UnsupportedLaunchIntent):
            JobSpec.from_dict(doc)

    def test_unsupported_is_a_value_error(self):
        # The reconciler's phase handler catches Exception and fails the job
        # cleanly; being a ValueError keeps it in the same family as the hybrid
        # shape check rather than something callers must special-case.
        assert issubclass(UnsupportedLaunchIntent, ValueError)
