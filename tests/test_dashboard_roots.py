"""The dashboard's CODE root and its STATE root are different things.

CLUSTER_ROOT relocates a cluster's state (.var/reconciler/state.json, runs,
logs, dropzone) so a second outpost instance can run beside the first. It must
NOT relocate the Python import path: the reconciler package ships with the
dashboard and lives next to it in the repo, not under whatever var directory a
second cluster was pointed at.

Conflating the two is a real regression, not a hypothetical: running a second
dashboard with CLUSTER_ROOT set to a sibling state directory died at import
with `ModuleNotFoundError: No module named 'reconciler'`, because sys.path had
been pointed at the state root. adapter.py and launch_models.py already draw
this line explicitly; these tests hold the dashboard to it.

Run out-of-process on purpose. conftest sets CLUSTER_ROOT for the whole suite,
and the thing under test is what happens at interpreter start under a DIFFERENT
value, which an in-process import cannot observe once the module is cached.
"""
import json
import os
import subprocess
import sys

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_PROBE = (
    "import json, viz.dashboard as d; "
    "print(json.dumps({'state': d.STATE_PATH, 'code': d.CODE_ROOT, 'repo': d.REPO_ROOT}))"
)


def _probe(cluster_root: str | None) -> dict:
    env = dict(os.environ)
    env.pop("CLUSTER_ROOT", None)
    if cluster_root is not None:
        env["CLUSTER_ROOT"] = cluster_root
    env["PYTHONPATH"] = REPO
    out = subprocess.run([sys.executable, "-c", _PROBE], cwd=REPO, env=env,
                         capture_output=True, text=True)
    assert out.returncode == 0, f"dashboard failed to import: {out.stderr}"
    return json.loads(out.stdout.strip().splitlines()[-1])


def test_imports_with_state_root_elsewhere(tmp_path):
    """A relocated CLUSTER_ROOT must not break the reconciler import."""
    got = _probe(str(tmp_path))
    assert got["code"] == REPO
    assert got["state"].startswith(str(tmp_path))


def test_state_follows_cluster_root(tmp_path):
    """...and the state path must actually move, or the second dashboard would
    quietly render the FIRST cluster's jobs."""
    a = _probe(str(tmp_path / "a"))
    b = _probe(str(tmp_path / "b"))
    assert a["state"] != b["state"]
    assert a["code"] == b["code"] == REPO


def test_defaults_to_repo_when_unset():
    """Unset CLUSTER_ROOT keeps the single-cluster behaviour: state in the repo."""
    got = _probe(None)
    assert got["repo"] == got["code"] == REPO
    assert got["state"].startswith(REPO)
