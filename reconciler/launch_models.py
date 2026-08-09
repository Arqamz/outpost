"""Launch Intent — parsing and structural validation for the `launch` block.

The schema under contract/launch-intent/v1/ is the source of truth. This module
LOADS it and applies it; it does not restate the field list or the vocabularies,
so adding an enum value or a field is a one-file change there rather than a
change that has to be mirrored here and will eventually not be.

Why a hand-rolled checker instead of `jsonschema`: the control plane has no
third-party runtime dependencies (only optional pymongo, and pyyaml in the CLI),
and validating what a caller sent is not worth making the reconciler
undeployable without a new package. What is implemented is `const`, `enum`,
`required`, `additionalProperties: false`, `properties` recursion, and `type`.

WHAT IS DELIBERATELY NOT CHECKED HERE: the `allOf`/`if-then` conditionals (the
explicit-list-requires-explicit-strategy pair, and the null-ranks-requires-
one-per-rank rule) and the numeric/array bounds. Those need the allocated
topology to produce a message worth reading — "explicit_cpu_ids has 2 entries
but the plan resolved 4 ranks" beats "does not match schema" — so they belong to
the placement resolver, which fails the job just as loudly. This layer's job is
narrower: catch a document this cluster must not act on at all.
"""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

# The version this backend implements. A document declaring anything else is
# refused outright — see parse().
SCHEMA_VERSION = "launch-intent/v1"

# Derived from THIS FILE, not CLUSTER_ROOT: the schema ships with the code, and
# CLUSTER_ROOT can point at a scratch directory that has no contract/ tree.
SCHEMA_PATH = (Path(__file__).resolve().parent.parent
               / "contract" / "launch-intent" / "v1" / "launch-intent.schema.json")

# Set when the planning phases land (topology discovery -> placement resolver ->
# launcher compilation -> preflight). Until then a launch block is parsed and
# validated but the job is still REFUSED, because executing it would run the
# benchmark under whatever placement the launcher defaults to while the caller
# believes their request was applied — the exact failure the contract exists to
# prevent, and one nothing downstream can detect from the output.
PLANNING_SUPPORTED = False

_PY_TYPES = {
    "object": dict, "array": list, "string": str, "boolean": bool,
    "integer": int, "number": (int, float), "null": type(None),
}


class UnsupportedLaunchIntent(ValueError):
    """A `launch` block this cluster must not act on: unknown version, or malformed."""


@lru_cache(maxsize=1)
def schema() -> dict:
    return json.loads(SCHEMA_PATH.read_text())


def _type_ok(value: object, spec: dict) -> bool:
    declared = spec.get("type")
    if declared is None:
        return True
    for name in ([declared] if isinstance(declared, str) else declared):
        py = _PY_TYPES.get(name)
        if py is None:                      # a type we do not model -> do not reject
            return True
        # bool is a subclass of int in Python but not an integer in JSON Schema.
        if name in ("integer", "number") and isinstance(value, bool):
            continue
        if isinstance(value, py):
            return True
    return False


def _errors(value: object, spec: dict, path: str) -> list[str]:
    where = path or "<root>"
    if not _type_ok(value, spec):
        return [f"{where}: expected {spec['type']}, got {type(value).__name__}"]

    found: list[str] = []
    if "const" in spec and value != spec["const"]:
        found.append(f"{where}: must be {spec['const']!r}, got {value!r}")
    if "enum" in spec and value not in spec["enum"]:
        found.append(f"{where}: {value!r} is not one of {spec['enum']}")

    if isinstance(value, dict):
        properties = spec.get("properties", {})
        for key in spec.get("required", []):
            if key not in value:
                found.append(f"{where}: missing required key {key!r}")
        if spec.get("additionalProperties") is False:
            for key in value:
                if key not in properties:
                    found.append(f"{where}: unknown key {key!r} "
                                 f"(this version accepts {sorted(properties)})")
        for key, sub in properties.items():
            if key in value:
                found += _errors(value[key], sub, f"{path}.{key}" if path else key)
    return found


def parse(block: object) -> dict:
    """Validate a `launch` block and return it, or raise UnsupportedLaunchIntent.

    The version is checked FIRST and on its own: a document from a later version
    would otherwise produce a wall of unknown-key errors that buries the one fact
    the operator needs, which is that this cluster is too old to run it.
    """
    if not isinstance(block, dict):
        raise UnsupportedLaunchIntent(
            f"spec.launch must be a mapping, got {type(block).__name__}")

    declared = block.get("schema_version")
    if declared != SCHEMA_VERSION:
        raise UnsupportedLaunchIntent(
            f"spec.launch declares schema_version {declared!r}, which this cluster does "
            f"not implement (it supports {SCHEMA_VERSION!r}). Refusing the job: running "
            "it would apply a placement the caller did not ask for.")

    problems = _errors(block, schema(), "")
    if problems:
        raise UnsupportedLaunchIntent(
            f"spec.launch is not a valid {SCHEMA_VERSION} document "
            f"({len(problems)} problem(s)):\n  " + "\n  ".join(problems))
    return block


def unsupported_reason(block: dict | None) -> str | None:
    """Why this cluster cannot run a job carrying `block`, or None if it can.

    Separate from parse() because the two refusals are different: a malformed
    intent is the caller's bug, while an intent this backend simply cannot
    resolve yet is ours. Both refuse; only the message differs.
    """
    if block is None:
        return None
    if not PLANNING_SUPPORTED:
        return ("spec.launch was supplied, but this backend cannot resolve a launch plan "
                "yet (no topology discovery or placement resolver). Refusing rather than "
                "running the benchmark with the requested placement silently ignored.")
    return None
