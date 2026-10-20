#!/usr/bin/env python3
"""Live cluster dashboard — per-VM CPU/RAM from libvirt + host GPU, the job
queue, and per-job/cross-job replay logs, all in one page. Stdlib only.
Run: `make dashboard` then open the URL.

  /                -> the dashboard page (viz/cluster.html)
  /api/stats.json  -> node stats (CPU% computed from cpu.time deltas)
  /api/jobs.json   -> the job queue (reads the same store cluster does)
  /api/slinky.json -> per Slinky cluster: its slots, Slurm nodes and queue
  /api/logs?job=<job_id>        -> that job's full replay transcript
  /api/logs?job=__reconciler__  -> the cross-job chronological narration log
  POST /api/clear[?force=1]     -> wipe job history for a blank-slate demo view
"""
from __future__ import annotations
import json, os, re, subprocess, sys, threading, time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import urlparse, parse_qs

_JOB_ID_RE = re.compile(r"^(job-[0-9a-f]+|__reconciler__)$")  # matches models.new_id("job")

HERE = os.path.dirname(os.path.abspath(__file__))
# Two different roots, and conflating them broke the second dashboard: the
# reconciler PACKAGE ships with this file and is found relative to it, while
# CLUSTER_ROOT says only where the STATE lives. Pointing CLUSTER_ROOT at a
# second cluster's var directory used to move the import path with it and
# raise ModuleNotFoundError. adapter.py and launch_models.py already draw this
# line ("derived from THIS FILE, not CLUSTER_ROOT"); this now matches them.
CODE_ROOT = os.path.dirname(HERE)  # viz/ is a sibling of reconciler/, not a package under it
REPO_ROOT = os.environ.get("CLUSTER_ROOT") or CODE_ROOT
sys.path.insert(0, CODE_ROOT)

from reconciler.adapter import (job_log_path, ssh_base, node_ssh_id,  # noqa: E402
                                SLINKY_NAMESPACE, SLINKY_LOGIN_SELECTOR, SLINKY_LOGIN_CONTAINER)
from reconciler.reconciler import GLOBAL_LOG, clear_jobs  # noqa: E402
from reconciler.models import BACKEND_PROVIDERS        # noqa: E402
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
        # ConnectTimeout=2 undercounted a real ProxyCommand-tunneled node (e.g.
        # AWS SSM session start): observed 3.2s to the SSH banner, well past
        # 2s, so every poll reported reachable:false for a genuinely-up node.
        # Cached for STATIC_TTL (5s default) regardless, so the slower timeout
        # doesn't cost extra polling.
        argv[1:1] = ["-o", "ConnectTimeout=8", "-o", "BatchMode=yes"]
        argv += ["head -n1 /proc/stat; "
                 "grep -E '^(MemTotal|MemAvailable):' /proc/meminfo; nproc"]
        card = {"name": n.name, "kind": "remote", "vcpus": None, "cpu_pct": None,
                "mem_used_mib": None, "mem_total_mib": None, "reachable": False}
        try:
            r = subprocess.run(argv, capture_output=True, text=True, timeout=10)
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


_K8S_CTX = os.environ.get("CLUSTER_K8S_CONTEXT", "")
_k8s_cache: dict = {"ts": 0.0, "data": None}
K8S_TTL = float(os.environ.get("CLUSTER_DASH_K8S_TTL", "3"))


def _kubectl(args: list[str]) -> subprocess.CompletedProcess:
    base = ["kubectl"] + (["--context", _K8S_CTX] if _K8S_CTX else [])
    return subprocess.run(base + args, capture_output=True, text=True, timeout=6)


def _k8s_gangs() -> dict:
    """Live KAI + HAMi gangs for the k8s backend — the dashboard counterpart of
    `kubectl get podgroups,pods -A -l outpost-job`. Per outpost-<job> namespace:
    the gang's PodGroup minMember + each rank pod's phase and HAMi VRAM cap. Only
    meaningful when the k8s backend is in use; degrades to {enabled:false} when
    kubectl can't reach a cluster, so it never blocks or errors the rest of the
    dashboard. Cached briefly (kubectl is ~0.3s vs the 2s poll)."""
    now = time.time()
    if _k8s_cache["data"] is not None and now - _k8s_cache["ts"] < K8S_TTL:
        return _k8s_cache["data"]
    result: dict = {"enabled": False, "gangs": []}
    try:
        r = _kubectl(["get", "pods", "-A", "-l", "outpost-job", "-o",
                      'jsonpath={range .items[*]}{.metadata.namespace},'
                      '{.metadata.labels.outpost-job},{.metadata.name},'
                      '{.status.phase},{.metadata.annotations.gpu-memory}{"\\n"}{end}'])
        if r.returncode != 0:
            _k8s_cache.update(ts=now, data=result)
            return result
        result["enabled"] = True
        gangs: dict[str, dict] = {}
        for line in r.stdout.splitlines():
            parts = line.split(",")
            if len(parts) < 5:
                continue
            ns, job, pod, phase, mem = parts[:5]
            g = gangs.setdefault(ns, {"namespace": ns, "job": job, "pods": []})
            g["pods"].append({"name": pod, "phase": phase or "Pending", "gpu_memory_mb": mem})
        pg = _kubectl(["get", "podgroups", "-A", "-o",
                       'jsonpath={range .items[*]}{.metadata.namespace},'
                       '{.metadata.name},{.spec.minMember}{"\\n"}{end}'])
        if pg.returncode == 0:
            for line in pg.stdout.splitlines():
                parts = line.split(",")
                if len(parts) < 3 or parts[0] not in gangs:
                    continue
                gangs[parts[0]]["podgroup"] = parts[1]
                gangs[parts[0]]["min_member"] = parts[2]
        for g in gangs.values():
            g["pods"].sort(key=lambda p: p["name"])
        result["gangs"] = sorted(gangs.values(), key=lambda x: x["namespace"])
    except Exception:
        pass  # kubectl absent / cluster down / timeout -> enabled:false, silently
    _k8s_cache.update(ts=now, data=result)
    return result


_slinky_cache: dict = {"ts": 0.0, "data": None}
_slinky_lock = threading.Lock()  # one snapshot at a time; other requests get the last one
SLINKY_TTL = float(os.environ.get("CLUSTER_DASH_SLINKY_TTL", "10"))
SLINKY_HISTORY = os.environ.get("CLUSTER_DASH_SLINKY_HISTORY", "now-12hours")
# Per kubectl call. A tashkil child is reached through the management cluster's
# chisel door, where one call takes 4-6 s (2026-10-20); 8 s marked a healthy
# cluster unreachable. The snapshot exec runs seven Slurm commands, so it gets twice this.
SLINKY_TIMEOUT = float(os.environ.get("CLUSTER_DASH_SLINKY_TIMEOUT", "30"))
_SLURM_CONFIG_KEYS = ("SLURM_VERSION", "SelectType", "SelectTypeParameters", "TaskPlugin",
                      "TaskPluginParam", "ProctrackType", "CgroupPlugin", "JobAcctGatherType",
                      "GresTypes", "MpiDefault", "SchedulerType", "PriorityType",
                      "AccountingStorageType", "MinJobAge", "SlurmctldHost[0]")
# One exec, one JSON document per section. `|| true` so one failing command
# (slurmctld down) still leaves the others.
_SLURM_SNAPSHOT = "; ".join(
    f"echo @@{k}; {cmd} 2>&1 || true" for k, cmd in (
        ("ping", "scontrol ping --json"),
        ("nodes", "scontrol show node --json"),
        ("queue", "squeue --json"),
        ("history", f"sacct -X -S {SLINKY_HISTORY} --json"),
        ("diag", "sdiag --json"),
        ("partitions", "scontrol show partition --json"),
        ("config", "scontrol show config"),
    ))


def _num(v):
    """Slurm JSON wraps numbers as {set, infinite, number}."""
    if isinstance(v, dict):
        if v.get("infinite"):
            return "inf"
        return v.get("number") if v.get("set") else None
    return v


def _sections(text: str) -> dict[str, str]:
    out, key = {}, None
    for line in text.splitlines(keepends=True):
        if line.startswith("@@"):
            key = line[2:].strip()
            out[key] = ""
        elif key:
            out[key] += line
    return out


def _json_or_none(text: str):
    try:
        return json.loads(text[text.index("{"):]) if "{" in text else None
    except ValueError:
        return None


def _slurm_view(sec: dict[str, str], job_names: dict[str, str]) -> dict:
    """The parts of each Slurm section the card shows."""
    v: dict = {"controller": None, "nodes": [], "queue": [], "history": [], "diag": None,
               "partitions": [], "config": {}, "version": ""}
    ping = _json_or_none(sec.get("ping", ""))
    if ping and ping.get("pings"):
        p = ping["pings"][0]
        v["controller"] = {"host": p.get("hostname"), "up": bool(p.get("responding")),
                           "latency_us": p.get("latency"), "status": p.get("status")}
        v["version"] = ping.get("meta", {}).get("slurm", {}).get("release", "")
    elif sec.get("ping"):
        v["controller"] = {"host": "", "up": False, "status": sec["ping"].strip()[-200:]}
    nodes = _json_or_none(sec.get("nodes", ""))
    for n in (nodes or {}).get("nodes", []):
        try:
            pod = json.loads(n.get("comment") or "{}")
        except ValueError:
            pod = {}
        v["nodes"].append({
            "name": n.get("name"), "state": "+".join(n.get("state") or []),
            "reason": n.get("reason") or "", "address": n.get("address"),
            "cpus": n.get("cpus"), "alloc_cpus": n.get("alloc_cpus"),
            "cpu_load": (n.get("cpu_load") or 0) / 100,
            "sockets": n.get("sockets"), "cores": n.get("cores"), "threads": n.get("threads"),
            "real_memory_mb": n.get("real_memory"), "alloc_memory_mb": n.get("alloc_memory"),
            "free_memory_mb": _num(n.get("free_mem")),
            "gres": n.get("gres") or "", "gres_used": n.get("gres_used") or "",
            "tres_used": n.get("tres_used") or "",
            "boot_time": _num(n.get("boot_time")), "slurmd_start": _num(n.get("slurmd_start_time")),
            "last_busy": _num(n.get("last_busy")), "version": n.get("version"),
            "pod": pod.get("podName", ""), "k8s_node": pod.get("node", ""),
        })
    queue = _json_or_none(sec.get("queue", ""))
    for j in (queue or {}).get("jobs", []):
        name = j.get("name") or ""
        v["queue"].append({
            "id": j.get("job_id"), "name": name, "outpost": job_names.get(name, ""),
            "state": "+".join(j.get("job_state") or []), "reason": j.get("state_reason"),
            "nodes": j.get("nodes") or "", "node_count": _num(j.get("node_count")),
            "cpus": _num(j.get("cpus")), "tres": j.get("tres_alloc_str") or j.get("tres_req_str") or "",
            "tres_per_node": j.get("tres_per_node") or "",
            "submit": _num(j.get("submit_time")), "start": _num(j.get("start_time")),
            "time_limit": _num(j.get("time_limit")), "priority": _num(j.get("priority")),
            "workdir": j.get("current_working_directory") or "",
        })
    hist = _json_or_none(sec.get("history", ""))
    for j in reversed((hist or {}).get("jobs", [])[-15:]):
        name = j.get("name") or ""
        t = j.get("time") or {}
        ec = j.get("exit_code") or {}
        v["history"].append({
            "id": j.get("job_id"), "name": name, "outpost": job_names.get(name, ""),
            "state": "+".join((j.get("state") or {}).get("current") or []),
            "elapsed_s": t.get("elapsed"), "start": t.get("start"), "end": t.get("end"),
            "nodes": j.get("nodes") or "",
            "exit": _num(ec.get("return_code")), "signal": (ec.get("signal") or {}).get("name") or "",
            "tres": ",".join(f"{x['type']}={x['count']}" for x in
                             ((j.get("tres") or {}).get("allocated") or []) if x.get("count", 0) > 0),
        })
    diag = _json_or_none(sec.get("diag", ""))
    if diag and diag.get("statistics"):
        s = diag["statistics"]
        v["diag"] = {k: _num(s.get(k)) for k in (
            "jobs_submitted", "jobs_started", "jobs_completed", "jobs_canceled", "jobs_failed",
            "jobs_pending", "jobs_running", "server_thread_count", "agent_queue_size",
            "schedule_cycle_last", "schedule_cycle_max", "schedule_cycle_mean",
            "bf_backfilled_jobs", "req_time_start")}
    parts = _json_or_none(sec.get("partitions", ""))
    for p in (parts or {}).get("partitions", []):
        v["partitions"].append({
            "name": p.get("name"), "nodes": (p.get("nodes") or {}).get("configured", ""),
            "cpus": (p.get("cpus") or {}).get("total"),
            "state": "+".join((p.get("partition") or {}).get("state") or []),
            "default": "DEFAULT" in ((p.get("flags") or []) + ((p.get("partition") or {}).get("flags") or [])),
            "max_time": _num((p.get("maximums") or {}).get("time")),
        })
    for line in sec.get("config", "").splitlines():
        k, _, val = line.partition("=")
        if k.strip() in _SLURM_CONFIG_KEYS:
            v["config"][k.strip()] = val.strip()
    return v


def _slurm_pods(base: list[str]) -> list[dict]:
    r = subprocess.run(base + ["get", "pods", "-n", SLINKY_NAMESPACE, "-o", "json"],
                       capture_output=True, text=True, timeout=SLINKY_TIMEOUT)
    if r.returncode != 0:
        return []
    out = []
    for p in json.loads(r.stdout).get("items", []):
        cs = (p.get("status") or {}).get("containerStatuses") or []
        out.append({"name": p["metadata"]["name"],
                    "phase": (p.get("status") or {}).get("phase", ""),
                    "ready": f"{sum(1 for c in cs if c.get('ready'))}/{len(cs)}",
                    "restarts": sum(c.get("restartCount", 0) for c in cs),
                    "node": (p.get("spec") or {}).get("nodeName", "")})
    return sorted(out, key=lambda x: x["name"])


def _slinky_cluster(ctx: str, job_names: dict[str, str]) -> dict:
    """One kubectl exec into the login pod for the whole Slurm snapshot, plus the
    slurm namespace's pods. No transport retries (they would stall the poll)."""
    base = ["kubectl"] + (["--context", ctx] if ctx else [])
    out: dict = {"reachable": False, "error": "", "login_pod": "", "pods": []}
    try:
        out["pods"] = _slurm_pods(base)
        logins = sorted(p["name"] for p in out["pods"]
                        if p["phase"] == "Running" and "login" in p["name"])
        r = subprocess.run(base + ["get", "pods", "-n", SLINKY_NAMESPACE, "-l", SLINKY_LOGIN_SELECTOR,
                                   "--field-selector=status.phase=Running", "-o", "name"],
                           capture_output=True, text=True, timeout=SLINKY_TIMEOUT)
        logins = sorted(p.split("/", 1)[-1] for p in r.stdout.split()) or logins
        if not logins:
            out["error"] = (r.stderr.strip() or "no Running login pod")[:200]
            return out
        out["login_pod"] = logins[0]
        r = subprocess.run(base + ["exec", "-n", SLINKY_NAMESPACE, logins[0], "-c", SLINKY_LOGIN_CONTAINER,
                                   "--", "bash", "-lc", _SLURM_SNAPSHOT],
                           capture_output=True, text=True, timeout=2 * SLINKY_TIMEOUT)
    except Exception as e:
        out["error"] = str(e)[:200]
        return out
    if r.returncode != 0 and "@@" not in r.stdout:
        out["error"] = r.stderr.strip()[:200]
        return out
    out["reachable"] = True
    out.update(_slurm_view(_sections(r.stdout), job_names))
    return out


def _slinky() -> dict:
    """Slinky slots from the registry, grouped by cluster (kube-context), each
    with a live Slurm + pod snapshot. {enabled:false} with no slinky slots."""
    now = time.time()
    if _slinky_cache["data"] is not None and now - _slinky_cache["ts"] < SLINKY_TTL:
        return _slinky_cache["data"]
    if not _slinky_lock.acquire(blocking=_slinky_cache["data"] is None):
        return _slinky_cache["data"]
    try:
        return _slinky_refresh(now)
    finally:
        _slinky_lock.release()


def _slinky_refresh(now: float) -> dict:
    try:
        store = open_store(STATE_PATH)
        slots = [n for n in store.list_nodes() if n.provider == "slinky"]
        job_names = {f"outpost-{j.job_id}": f"{j.job_id}|{(j.spec or {}).get('name', '')}"
                     for j in store.list_jobs()}
    except Exception:
        slots, job_names = [], {}
    by_ctx: dict[str, list] = {}
    for n in slots:
        by_ctx.setdefault(n.kube_context or _K8S_CTX, []).append(n)
    clusters = []
    for ctx, ns in sorted(by_ctx.items()):
        c = {"context": ctx or "(current)", "gpu": any(n.gpu for n in ns),
             "slots": [{"name": n.name, "state": n.state, "owner_job": n.owner_job,
                        "slurm_node": n.slurm_node, "gpu_count": n.gpu_count,
                        "owner_name": job_names.get(f"outpost-{n.owner_job}", "").partition("|")[2]}
                       for n in sorted(ns, key=lambda n: n.name)]}
        c.update(_slinky_cluster(ctx, job_names))
        clusters.append(c)
    result = {"enabled": bool(clusters), "clusters": clusters, "ts": now}
    _slinky_cache.update(ts=now, data=result)
    return result


def _jobs() -> list[dict]:
    """The job queue, straight from the same store `cluster list/status`
    reads — the dashboard has no state of its own, so it can never drift from
    what the reconciler actually sees."""
    store = open_store(STATE_PATH)
    slots = {n.name for n in store.list_nodes() if n.provider in BACKEND_PROVIDERS}
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
            "backend_slot": next((n for n in j.assigned_nodes or [] if n in slots), ""),
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
        elif path.startswith("/api/k8s"):
            self._json(_k8s_gangs())
        elif path.startswith("/api/slinky"):
            self._json(_slinky())
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
