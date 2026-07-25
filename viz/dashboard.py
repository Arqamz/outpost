#!/usr/bin/env python3
"""Live cluster dashboard — per-VM CPU/RAM from libvirt + host GPU, the job
queue, and per-job/cross-job replay logs, all in one page. Stdlib only.
Run: `make dashboard` then open the URL.

  /                -> the dashboard page (viz/cluster.html)
  /api/stats.json  -> node stats (CPU% computed from cpu.time deltas)
  /api/jobs.json   -> the job queue (reads the same store cluster does)
  /api/logs?job=<job_id>        -> that job's full replay transcript
  /api/logs?job=__reconciler__  -> the cross-job chronological narration log
  POST /api/clear[?force=1]     -> wipe job history for a blank-slate demo view
"""
from __future__ import annotations
import json, os, re, subprocess, sys, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

_JOB_ID_RE = re.compile(r"^(job-[0-9a-f]+|__reconciler__)$")  # matches models.new_id("job")

HERE = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.environ.get("CLUSTER_ROOT") or os.path.dirname(HERE)
sys.path.insert(0, REPO_ROOT)  # viz/ is a sibling of reconciler/, not a package under it

from reconciler.adapter import job_log_path, ssh_base, node_ssh_id  # noqa: E402
from reconciler.reconciler import GLOBAL_LOG, clear_jobs  # noqa: E402
from reconciler.store import open_store                 # noqa: E402

STATE_PATH = os.path.join(REPO_ROOT, ".var", "reconciler", "state.json")
LOG_TAIL_LINES = int(os.environ.get("CLUSTER_DASH_LOG_TAIL", "1000"))

URI = os.environ.get("LIBVIRT_DEFAULT_URI", "qemu:///system")
PORT = int(os.environ.get("CLUSTER_DASH_PORT", "8087"))
_prev: dict = {}   # sampler state for delta-based CPU%
# static-ssh nodes are probed over ssh (slow, ~seconds); cache the probe so the
# 2s stats poll doesn't pay an ssh round-trip every time (which stacked requests
# and tore down mid-write on refresh -> BrokenPipe). Refresh at most every TTL.
_static_cache: dict = {"ts": 0.0, "data": []}
STATIC_TTL = float(os.environ.get("CLUSTER_DASH_STATIC_TTL", "5"))


def _virsh(*a) -> str:
    return subprocess.run(["virsh", "-c", URI, *a], capture_output=True, text=True).stdout


def _domains() -> list[str]:
    return [d for d in _virsh("list", "--name", "--state-running").split() if d.startswith("cluster")]


def _domstats(dom: str) -> dict:
    kv = {}
    for line in _virsh("domstats", dom, "--cpu-total", "--balloon", "--vcpu").splitlines():
        line = line.strip()
        if "=" in line:
            k, v = line.split("=", 1)
            kv[k] = v
    return kv


def _cpu_pct(key: str, cpu_ns: int, vcpus: int, now: float):
    prev = _prev.get(key)
    _prev[key] = (cpu_ns, now)
    if not prev:
        return None
    p_ns, p_ts = prev
    dt = now - p_ts
    if dt <= 0:
        return None
    return max(0.0, min(100.0, (cpu_ns - p_ns) / (dt * 1e9) / max(vcpus, 1) * 100))


def _vm_nodes(now: float) -> list[dict]:
    out = []
    for d in _domains():
        kv = _domstats(d)
        vcpus = int(kv.get("vcpu.current") or kv.get("vcpu.maximum") or 1)
        assigned = int(kv.get("balloon.current", 0))          # KiB configured
        rss = int(kv.get("balloon.rss", 0))                   # KiB host RSS = real usage
        out.append({
            "name": d, "kind": "vm", "vcpus": vcpus,
            "cpu_pct": _cpu_pct(d, int(kv.get("cpu.time", 0)), vcpus, now),
            "mem_used_mib": round(rss / 1024) if rss else None,
            "mem_total_mib": round(assigned / 1024) if assigned else None,
        })
    return sorted(out, key=lambda x: x["name"])


def _host_node(now: float) -> dict:
    # host CPU% from /proc/stat delta
    with open("/proc/stat") as f:
        parts = f.readline().split()[1:]
    vals = list(map(int, parts))
    idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
    total = sum(vals)
    cpu_pct = None
    prev = _prev.get("__host__")
    _prev["__host__"] = (total, idle)
    if prev:
        dt, di = total - prev[0], idle - prev[1]
        if dt > 0:
            cpu_pct = max(0.0, min(100.0, (1 - di / dt) * 100))
    # host RAM from /proc/meminfo
    mi = {}
    with open("/proc/meminfo") as f:
        for line in f:
            k, v = line.split(":", 1)
            mi[k] = int(v.strip().split()[0])   # KiB
    mem_total = mi.get("MemTotal", 0)
    mem_used = mem_total - mi.get("MemAvailable", 0)
    # GPU via nvidia-smi
    gpu = None
    try:
        q = subprocess.run(
            ["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu",
             "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5).stdout.strip()
        if q:
            u, mu, mt, temp = [x.strip() for x in q.split(",")]
            gpu = {"util_pct": float(u), "mem_used_mib": float(mu),
                   "mem_total_mib": float(mt), "temp_c": float(temp)}
    except Exception:
        pass
    return {"name": "cluster-host", "kind": "host", "vcpus": os.cpu_count(),
            "cpu_pct": cpu_pct, "mem_used_mib": round(mem_used / 1024),
            "mem_total_mib": round(mem_total / 1024), "gpu": gpu}


def _static_nodes(now: float) -> list[dict]:
    """Registry nodes reached over ssh (provider=static-ssh, e.g. EC2) — they
    have no libvirt domain, so _vm_nodes never sees them. One short ssh round-trip
    per node reads /proc/stat + /proc/meminfo + nproc (same figures as _host_node,
    just remote); CPU% is a delta between polls keyed per node. Best-effort: an
    unreachable node (spot reclaimed, sg change) still shows up with reachable:
    false and null stats rather than vanishing from the grid.

    Cached for STATIC_TTL seconds: the ssh probe is slow relative to the 2s poll,
    so without this every poll paid an ssh round-trip, stacking requests that got
    torn down mid-write on a page refresh (BrokenPipe)."""
    if now - _static_cache["ts"] < STATIC_TTL:
        return _static_cache["data"]
    out = []
    try:
        nodes = [n for n in open_store(STATE_PATH).list_nodes() if n.provider == "static-ssh"]
    except Exception:
        return out
    for n in nodes:
        argv = ssh_base(n.ip, *node_ssh_id(n))
        argv[1:1] = ["-o", "ConnectTimeout=2", "-o", "BatchMode=yes"]
        argv += ["head -n1 /proc/stat; "
                 "grep -E '^(MemTotal|MemAvailable):' /proc/meminfo; nproc"]
        card = {"name": n.name, "kind": "remote", "vcpus": None, "cpu_pct": None,
                "mem_used_mib": None, "mem_total_mib": None, "reachable": False}
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=4)
            lines = r.stdout.split("\n")
            vals = list(map(int, lines[0].split()[1:]))       # cpu aggregate line
            mi = {}
            for ln in lines[1:]:
                if ":" in ln:
                    k, v = ln.split(":", 1)
                    mi[k.strip()] = int(v.strip().split()[0])  # KiB
            vcpus = int([ln for ln in lines if ln.strip().isdigit()][-1])
        except Exception:
            out.append(card)
            continue
        card["reachable"] = True
        card["vcpus"] = vcpus
        idle = vals[3] + (vals[4] if len(vals) > 4 else 0)
        total = sum(vals)
        key = f"__static__{n.name}"
        prev = _prev.get(key)
        _prev[key] = (total, idle)
        if prev:
            dt, di = total - prev[0], idle - prev[1]
            if dt > 0:
                card["cpu_pct"] = max(0.0, min(100.0, (1 - di / dt) * 100))
        mem_total = mi.get("MemTotal", 0)
        card["mem_total_mib"] = round(mem_total / 1024) if mem_total else None
        card["mem_used_mib"] = round((mem_total - mi.get("MemAvailable", 0)) / 1024) if mem_total else None
        out.append(card)
    out.sort(key=lambda x: x["name"])
    _static_cache["ts"], _static_cache["data"] = now, out
    return out


def collect() -> dict:
    now = time.time()
    return {"ts": now, "nodes": [_host_node(now)] + _vm_nodes(now) + _static_nodes(now)}


def _jobs() -> list[dict]:
    """The job queue, straight from the same store `cluster list/status`
    reads — the dashboard has no state of its own, so it can never drift from
    what the reconciler actually sees."""
    store = open_store(STATE_PATH)
    out = []
    for j in store.list_jobs():
        spec = j.spec or {}
        out.append({
            "job_id": j.job_id,
            "name": spec.get("name", ""),
            "state": j.state,
            "launcher": spec.get("launcher", "single"),
            "gpu": bool(spec.get("gpu", False)),
            "node_count": spec.get("node_count", 0),
            "assigned_nodes": j.assigned_nodes,
            "image": spec.get("image", "") or "(dry-run)",
            "run": j.run,
            "drop_path": j.drop_path,
            "error": j.error,
            "created_at": j.created_at,
            "updated_at": j.updated_at,
        })
    out.sort(key=lambda x: x["created_at"] or "", reverse=True)
    return out


def _log_tail(path: str, max_lines: int = LOG_TAIL_LINES) -> str:
    if not os.path.exists(path):
        return "(no log yet)"
    with open(path) as f:
        lines = f.readlines()
    return "".join(lines[-max_lines:])


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _write(self, body: bytes) -> None:
        """Write a response body, tolerating a client that already hung up. The
        2s poll + page refreshes routinely close a connection mid-response, so a
        BrokenPipe/reset here is normal, not a crash worth a traceback."""
        try:
            self.wfile.write(body)
        except (BrokenPipeError, ConnectionResetError):
            pass

    def _json(self, obj) -> None:
        body = json.dumps(obj).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self._write(body)

    def do_GET(self):
        path = urlparse(self.path).path
        if path.startswith("/api/stats"):
            self._json(collect())
        elif path.startswith("/api/jobs"):
            self._json(_jobs())
        elif path.startswith("/api/logs"):
            job_id = (parse_qs(urlparse(self.path).query).get("job") or [""])[0]
            # job_id lands in a filesystem path (job_log_path) and this server
            # binds 0.0.0.0 — reject anything that isn't a real job-id shape
            # before it ever reaches a path join (no path traversal via ../).
            if not _JOB_ID_RE.match(job_id):
                self.send_error(400, "invalid job id")
                return
            log_path = GLOBAL_LOG if job_id == "__reconciler__" else job_log_path(job_id)
            body = _log_tail(log_path).encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self._write(body)
        else:
            try:
                with open(os.path.join(HERE, "cluster.html"), "rb") as f:
                    body = f.read()
                self.send_response(200)
                self.send_header("Content-Type", "text/html")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self._write(body)
            except FileNotFoundError:
                self.send_error(404)

    def do_POST(self):
        path = urlparse(self.path).path
        if path == "/api/clear":
            force = (parse_qs(urlparse(self.path).query).get("force") or ["0"])[0] == "1"
            self._json(clear_jobs(open_store(STATE_PATH), force=force))
        else:
            self.send_error(404)


if __name__ == "__main__":
    print(f"Outpost dashboard → http://localhost:{PORT}  (Ctrl-C to stop)")
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
