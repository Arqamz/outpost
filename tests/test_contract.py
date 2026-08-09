"""The intake contract's machine-readable half: contract/launch-intent/v1/.

Skipped when jsonschema is unavailable — it is not a runtime dependency of the
control plane, only of validating what a caller sent.
"""
from __future__ import annotations

import glob
import json
import os

import pytest
from conftest import REPO_DIR

jsonschema = pytest.importorskip("jsonschema", reason="jsonschema is a test-only dependency")
from jsonschema.validators import Draft202012Validator  # noqa: E402

CONTRACT_DIR = os.path.join(REPO_DIR, "contract", "launch-intent", "v1")
SCHEMA_PATH = os.path.join(CONTRACT_DIR, "launch-intent.schema.json")
EXAMPLES = sorted(glob.glob(os.path.join(CONTRACT_DIR, "examples", "*.json")))


def _schema() -> dict:
    with open(SCHEMA_PATH) as f:
        return json.load(f)


def test_schema_is_a_legal_draft_2020_12_document():
    Draft202012Validator.check_schema(_schema())


def test_schema_version_matches_the_directory():
    # The version segment is the compatibility boundary; a copy whose declared
    # version disagreed with where it sits would validate the wrong documents.
    schema = _schema()
    assert schema["properties"]["schema_version"]["const"] == "launch-intent/v1"
    assert schema["$id"].endswith("/launch-intent/v1/launch-intent.schema.json")


def test_examples_exist():
    assert EXAMPLES, f"no examples under {CONTRACT_DIR}/examples"


@pytest.mark.parametrize("path", EXAMPLES, ids=lambda p: os.path.basename(p))
def test_example_validates(path):
    with open(path) as f:
        doc = json.load(f)
    errors = sorted(Draft202012Validator(_schema()).iter_errors(doc), key=str)
    assert not errors, "\n".join(
        f"{'/'.join(str(p) for p in e.absolute_path) or '<root>'}: {e.message}" for e in errors)


def test_an_unrecognised_version_does_not_validate():
    # The rule the control plane leans on: a document from a future version must
    # never look acceptable, or it would be executed under placement rules this
    # cluster does not implement.
    with open(EXAMPLES[0]) as f:
        doc = json.load(f)
    doc["schema_version"] = "launch-intent/v99"
    assert list(Draft202012Validator(_schema()).iter_errors(doc))


def test_an_unknown_placement_key_does_not_validate():
    with open(EXAMPLES[0]) as f:
        doc = json.load(f)
    doc["cpu"]["bind_to"] = "socket"
    assert list(Draft202012Validator(_schema()).iter_errors(doc))
