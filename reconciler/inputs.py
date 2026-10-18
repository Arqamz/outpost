"""Job inputs: files a job declares (`JobSpec.inputs`) that outpost downloads,
verifies and mounts read-only, so no node has to be prepared by hand.

An input is a manifest of `<sha256>  <relative path>` lines in an S3 prefix,
pinned by the manifest's own sha256, plus the path the container reads it at.
The control plane holds the S3 credentials: it checks the manifest against its
pin and signs a short-lived URL per file. The node (or the Slinky login pod)
downloads, checks each file against the manifest and keeps it in a cache keyed
by the manifest digest. The URLs reach the node over stdin, never argv or a log.
"""
from __future__ import annotations

import datetime as dt
import hashlib
import hmac
import json
import os
import re
import shlex
import subprocess
import urllib.parse
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable

S3_PROFILE = os.environ.get("CLUSTER_S3_PROFILE", "")
URL_TTL_S = 3600

_SHA256_RE = re.compile(r"^sha256:[0-9a-f]{64}$")
_S3_RE = re.compile(r"^s3://([a-z0-9][a-z0-9.-]{1,61}[a-z0-9])/(.*)$")


def _relative(p: str) -> bool:
    return bool(p) and not p.startswith("/") and ".." not in p.split("/")


def parse_inputs(raw: Any) -> list[dict]:
    """Validate `JobSpec.inputs`. A malformed entry refuses the job: an input
    quietly dropped runs a job without the files it was promised."""
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError(f"inputs must be a list, got {raw!r}")
    out = []
    for i, inp in enumerate(raw):
        where = f"inputs[{i}]"
        if not isinstance(inp, dict):
            raise ValueError(f"{where} must be a map, got {inp!r}")
        source, manifest = str(inp.get("source") or ""), str(inp.get("manifest") or "")
        digest, path = str(inp.get("manifest_sha256") or ""), str(inp.get("path") or "")
        only = inp.get("only") or []
        if not _S3_RE.match(source):
            raise ValueError(f"{where}.source must be s3://<bucket>/<prefix>, got {source!r}")
        if not _relative(manifest):
            raise ValueError(f"{where}.manifest must be a path relative to source, got {manifest!r}")
        if not _SHA256_RE.match(digest):
            raise ValueError(f"{where}.manifest_sha256 must be sha256:<64 hex>, got {digest!r}")
        if not path.startswith("/") or ".." in path.split("/"):
            raise ValueError(f"{where}.path must be an absolute container path, got {path!r}")
        if not (isinstance(only, list) and all(isinstance(p, str) and _relative(p) for p in only)):
            raise ValueError(f"{where}.only must be a list of relative path prefixes, got {only!r}")
        out.append({"source": source.rstrip("/") + "/", "manifest": manifest,
                    "manifest_sha256": digest, "path": path, "only": list(only)})
    return out


def cache_dir(root: str, inp: dict) -> str:
    return f"{root}/{inp['manifest_sha256'].removeprefix('sha256:')}"


def marker(inp: dict) -> str:
    """Names the complete, verified subset `inp` asks for, so a warm cache is
    recognised without the network."""
    key = "\n".join([inp["manifest_sha256"], *sorted(inp["only"])])
    return f".complete-{hashlib.sha256(key.encode()).hexdigest()[:16]}"


def mount(root: str, inp: dict) -> str:
    return f"{cache_dir(root, inp)}:{inp['path']}:ro"


def check_script(root: str, inp: dict) -> str:
    return f"test -f {shlex.quote(cache_dir(root, inp) + '/' + marker(inp))}"


def fetch_script(root: str, inp: dict) -> str:
    """Reads `<sha256> <relative path> <url>` lines from stdin. A file is renamed
    into the cache only after its hash matches, so anything present is verified."""
    d, q = cache_dir(root, inp), shlex.quote
    refusal = (f"does not match the sha256 in {inp['source']}{inp['manifest']} — the bucket "
               "changed under a pinned manifest; republish the bundle or re-pin manifest_sha256")
    return (
        f"set -e; d={q(d)}; mkdir -p \"$d\"; exec 9>\"$d.lock\"; flock 9; "
        f"[ -f \"$d\"/{q(marker(inp))} ] && exit 0; "
        "while read -r sha rel url; do "
        "[ -n \"$sha\" ] || continue; f=\"$d/$rel\"; [ -f \"$f\" ] && continue; "
        "mkdir -p \"$(dirname \"$f\")\"; "
        "curl -sSfL --retry 3 -o \"$f.part\" \"$url\"; "
        "if [ \"$(sha256sum \"$f.part\" | cut -d' ' -f1)\" != \"$sha\" ]; then "
        f"echo \"input $rel \"{q(refusal)} >&2; rm -f \"$f.part\"; exit 1; fi; "
        "mv \"$f.part\" \"$f\"; done; "
        f"touch \"$d\"/{q(marker(inp))}")


# ── S3 signing ────────────────────────────────────────────────────────────

def presign(bucket: str, key: str, region: str, access_key: str, secret_key: str,
            token: str | None, expires: int, now: dt.datetime) -> str:
    """A SigV4 query-string presigned GET URL (the form `aws s3 presign` emits)."""
    host = "s3.amazonaws.com" if region == "us-east-1" else f"s3.{region}.amazonaws.com"
    host = f"{bucket}.{host}"
    amz_date, day = now.strftime("%Y%m%dT%H%M%SZ"), now.strftime("%Y%m%d")
    scope = f"{day}/{region}/s3/aws4_request"
    params = {"X-Amz-Algorithm": "AWS4-HMAC-SHA256", "X-Amz-Credential": f"{access_key}/{scope}",
              "X-Amz-Date": amz_date, "X-Amz-Expires": str(expires), "X-Amz-SignedHeaders": "host"}
    if token:
        params["X-Amz-Security-Token"] = token
    enc = lambda s: urllib.parse.quote(s, safe="-_.~")  # noqa: E731
    query = "&".join(f"{enc(k)}={enc(v)}" for k, v in sorted(params.items()))
    uri = "/" + urllib.parse.quote(key, safe="-_.~/")
    request = "\n".join(["GET", uri, query, f"host:{host}\n", "host", "UNSIGNED-PAYLOAD"])
    to_sign = "\n".join(["AWS4-HMAC-SHA256", amz_date, scope,
                         hashlib.sha256(request.encode()).hexdigest()])
    k = f"AWS4{secret_key}".encode()
    for part in (day, region, "s3", "aws4_request"):
        k = hmac.new(k, part.encode(), hashlib.sha256).digest()
    sig = hmac.new(k, to_sign.encode(), hashlib.sha256).hexdigest()
    return f"https://{host}{uri}?{query}&X-Amz-Signature={sig}"


class S3Signer:
    """Signs with the credentials of CLUSTER_S3_PROFILE, fetched once from the
    aws CLI and refreshed when they near expiry."""

    def __init__(self, profile: str = "", run: Callable = subprocess.run,
                 clock: Callable[[], dt.datetime] = lambda: dt.datetime.now(dt.timezone.utc)):
        self.profile, self._run, self._clock = profile or S3_PROFILE, run, clock
        self._creds: dict | None = None
        self._regions: dict[str, str] = {}

    def _aws(self, *args: str) -> str:
        out = self._run(["aws", "--profile", self.profile, *args], capture_output=True, text=True)
        if out.returncode != 0:
            raise RuntimeError(f"aws {' '.join(args[:2])} with profile {self.profile!r} failed: "
                               f"{out.stderr.strip()[-300:]} — refresh that profile's credentials "
                               "(e.g. `aws sso login`)")
        return out.stdout

    def _credentials(self) -> dict:
        exp = self._creds and self._creds.get("Expiration")
        if self._creds and (not exp or dt.datetime.fromisoformat(exp)
                            > self._clock() + dt.timedelta(minutes=10)):
            return self._creds
        self._creds = json.loads(self._aws("configure", "export-credentials", "--format", "process"))
        return self._creds

    def _region(self, bucket: str) -> str:
        if bucket not in self._regions:
            loc = json.loads(self._aws("s3api", "get-bucket-location", "--bucket", bucket))
            self._regions[bucket] = loc.get("LocationConstraint") or "us-east-1"
        return self._regions[bucket]

    def url(self, s3_uri: str) -> str:
        if not self.profile:
            raise RuntimeError(f"{s3_uri} needs S3 credentials and outpost has none — set "
                               "CLUSTER_S3_PROFILE to an AWS profile that can read it")
        bucket, key = _S3_RE.match(s3_uri).groups()
        c = self._credentials()
        return presign(bucket, key, self._region(bucket), c["AccessKeyId"], c["SecretAccessKey"],
                       c.get("SessionToken"), URL_TTL_S, self._clock())


_signer: S3Signer | None = None


def signer() -> S3Signer:
    global _signer
    if _signer is None:
        _signer = S3Signer()
    return _signer


def _get(url: str) -> bytes:
    with urllib.request.urlopen(url, timeout=60) as r:
        return r.read()


@dataclass
class Fetch:
    stdin: str
    files: int


def plan_fetch(inp: dict, sign: S3Signer, get: Callable[[str], bytes] = _get) -> Fetch:
    """The stdin for fetch_script: the manifest itself and every file under
    `only`. The manifest is fetched here and refused unless it is the pinned one."""
    manifest_uri = inp["source"] + inp["manifest"]
    body = get(sign.url(manifest_uri))
    got = "sha256:" + hashlib.sha256(body).hexdigest()
    if got != inp["manifest_sha256"]:
        raise RuntimeError(f"{manifest_uri} is {got}, not the pinned {inp['manifest_sha256']} — "
                           "the bundle was republished; re-pin manifest_sha256 in the job's suite")
    rows = [(inp["manifest_sha256"].removeprefix("sha256:"), inp["manifest"])]
    for line in body.decode().splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        sha, rel = line.split(None, 1)
        if not inp["only"] or rel.startswith(tuple(inp["only"])):
            if not (re.fullmatch(r"[0-9a-f]{64}", sha) and _relative(rel) and " " not in rel):
                raise RuntimeError(f"{manifest_uri} has an unusable line {line!r}")
            rows.append((sha, rel))
    if len(rows) == 1:
        raise RuntimeError(f"{manifest_uri} lists nothing under {inp['only']} — fix the input's "
                           "`only` prefixes")
    lines = [f"{sha} {rel} {sign.url(inp['source'] + rel)}" for sha, rel in rows]
    return Fetch(stdin="\n".join(lines) + "\n", files=len(rows))


def stage(spec_inputs: list[dict], root: str,
          run_sh: Callable[[str, str | None], int], log: Callable[[str], None],
          sign: S3Signer | None = None, get: Callable[[str], bytes] = _get) -> list[str]:
    """Stage every input under `root` through `run_sh(script, stdin) -> rc`, the
    backend's own way of running a shell on the target. Returns the mounts."""
    mounts = []
    for inp in spec_inputs:
        if run_sh(check_script(root, inp), None) == 0:
            log(f"[inputs] {inp['source']} ({inp['manifest_sha256'][:19]}…) already staged")
        else:
            fetch = plan_fetch(inp, sign or signer(), get)
            log(f"[inputs] staging {fetch.files} file(s) from {inp['source']} into "
                f"{cache_dir(root, inp)}")
            rc = run_sh(fetch_script(root, inp), fetch.stdin)
            if rc != 0:
                raise RuntimeError(f"staging {inp['source']} failed (exit {rc}); the output is "
                                   "in the job's replay log")
        mounts.append(mount(root, inp))
    return mounts
