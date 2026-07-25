"""cluster — operator CLI for the control plane.

Talks to the store (file by default, Mongo via CLUSTER_MONGO_URI); it does NOT run on
the nodes. Commands:
    submit --spec job.yaml     write a JobSpec into jobs
    reconcile [--once] [--execute]   drive the state machine
    list                       jobs + states
    status <job_id>            one job + its audit trail
    result <job_id>            print where the job's artifacts landed (egress)
    logs <job_id> [--tail N]   full replay transcript for one job (every command + output)
    nodes                      the NodeRegistry (with gpu/local capabilities)
    seed-nodes                 populate nodes from config (VMs + host GPU node)
    add-node --spec node.yaml  join a pre-provisioned ssh host (e.g. EC2) as a worker
    remove-node <name>         deregister a node (does not touch the machine)
    fail-node <name> [--inject]   quarantine a node + fail its job (failure-injection test)
    clear-jobs [--force]       wipe job history + audit + replay logs (demo reset)
"""
from __future__ import annotations
import argparse
import os
import subprocess
import sys

from .adapter import REPO_ROOT, LIBVIRT_DIR, job_log_path
from .models import JobSpec, NodeRecord
from .reconciler import Reconciler, GLOBAL_LOG, clear_jobs
from .states import NodeState
from .store import open_store

DEFAULT_STATE = os.path.join(REPO_ROOT, ".var", "reconciler", "state.json")


def _store():
    return open_store(DEFAULT_STATE)


def _topology() -> list[tuple[int, str, str]]:
    """(index, name, ip) per node, reusing infra/libvirt as the single source of truth."""
    out = subprocess.run(
        ["bash", "-c", 'source "$0"/infra/libvirt/lib.sh; '
         'for i in $(node_seq); do echo "$i $(node_name $i) $(node_ip $i)"; done', REPO_ROOT],
        capture_output=True, text=True, check=True,
    ).stdout.strip().splitlines()
    rows = []
    for line in out:
        idx, name, ip = line.split()
        rows.append((int(idx), name, ip))
    return rows


def _host_node() -> NodeRecord | None:
    """The control-plane host as a GPU node, read from config.env (or None if
    CLUSTER_GPU_HOST != 1). Registered alongside the VMs so GPU jobs land here.
    Its ip is the host's cluster0 bridge address (CLUSTER_HOST_IP), not
    127.0.0.1 — hybrid MPI hostfiles hand it to VM ranks, which must be able to
    reach the host on it. LocalHostAdapter itself never dials the ip."""
    out = subprocess.run(
        ["bash", "-c", 'source "$0"/infra/libvirt/lib.sh; '
         'echo "${CLUSTER_GPU_HOST:-0} ${CLUSTER_HOST_NODE_NAME:-cluster-host} '
         '${CLUSTER_HOST_RUNTIME:-apptainer} ${CLUSTER_HOST_IP:-127.0.0.1}"', REPO_ROOT],
        capture_output=True, text=True, check=True,
    ).stdout.split()
    if not out or out[0] != "1":
        return None
    _, name, runtime, ip = out
    return NodeRecord(node_id=name, name=name, index=0, ip=ip,
                      state=NodeState.AVAILABLE.value, gpu=True, local=True, runtime=runtime)


def cmd_submit(args):
    import yaml
    with open(args.spec) as f:
        spec = JobSpec.from_dict(yaml.safe_load(f))
    r = Reconciler(_store())
    print(r.submit(spec))


def cmd_reconcile(args):
    store = _store()
    # execute=True wires the real adapters (local host + libvirt); default dry-run.
    r = Reconciler(store, execute=args.execute)
    if args.once:
        n = r.tick()
        print(f"[reconcile] moved {n} job(s)")
    else:
        r.run_forever(interval=args.interval)


def cmd_list(args):
    for j in _store().list_jobs():
        img = j.spec.get("image") or "(dry-run)"
        gpu = "gpu" if j.spec.get("gpu") else "cpu"
        print(f"{j.job_id}  {j.state:<13} {gpu} {img:<22} "
              f"nodes={len(j.assigned_nodes)}  {'ERR:'+j.error if j.error else ''}")


def cmd_status(args):
    store = _store()
    j = store.get_job(args.job_id)
    if not j:
        sys.exit(f"no such job: {args.job_id}")
    print(f"job     : {j.job_id}\nstate   : {j.state}\nspec    : {j.spec}\n"
          f"nodes   : {j.assigned_nodes}\nrun     : {j.run}\n"
          f"drop    : {j.drop_path}\nerror   : {j.error}\n"
          f"log     : {job_log_path(j.job_id)}  (cluster logs {j.job_id})")
    print("audit   :")
    for e in store.list_audit(args.job_id):
        print(f"  #{e['seq']:>3} {e['ts']}  {e['from']} -> {e['to']}  {e.get('reason','')}")


def cmd_result(args):
    """Egress side of the open interface: where a job's artifacts landed."""
    j = _store().get_job(args.job_id)
    if not j:
        sys.exit(f"no such job: {args.job_id}")
    print(j.drop_path or "(no artifacts yet)")


def cmd_logs(args):
    """Full replay transcript for one job: every command + its live output,
    chronological, written by adapter.run_logged / reconciler._set_job."""
    path = job_log_path(args.job_id)
    if not os.path.exists(path):
        sys.exit(f"no log yet for {args.job_id} (expected {path})")
    with open(path) as f:
        lines = f.readlines()
    if args.tail:
        lines = lines[-args.tail:]
    sys.stdout.writelines(lines)


def cmd_reconciler_log(args):
    """The cross-job, chronological narration log (every job's state
    transitions, interleaved) — the cross-job counterpart to `logs <job_id>`."""
    if not os.path.exists(GLOBAL_LOG):
        sys.exit(f"no reconciler log yet (expected {GLOBAL_LOG})")
    with open(GLOBAL_LOG) as f:
        lines = f.readlines()
    if args.tail:
        lines = lines[-args.tail:]
    sys.stdout.writelines(lines)


def cmd_nodes(args):
    for n in _store().list_nodes():
        caps = ("gpu" if n.gpu else "cpu") + ("/local" if n.local else "")
        print(f"{n.name:<14} idx={n.index} {n.ip:<15} {n.state:<12} "
              f"{caps:<10} {n.adapter_key:<11} {n.runtime:<10} owner={n.owner_job or '-'}")


def cmd_add_node(args):
    """Register a pre-provisioned, already-running ssh host (e.g. an EC2
    instance) as a worker, from a YAML manifest (see node.ec2.example.yaml).
    Unlike seed-nodes (which derives libvirt VMs + the host from config), this
    joins ONE static node the cluster doesn't provision or destroy — it only
    runs containers on it (StaticSshAdapter). index=-1 marks it as not driven by
    the libvirt index-based wrappers."""
    import yaml
    with open(args.spec) as f:
        d = yaml.safe_load(f) or {}
    missing = [k for k in ("name", "ip") if not d.get(k)]
    if missing:
        sys.exit(f"node spec {args.spec} missing required field(s): {', '.join(missing)}")
    store = _store()
    if store.get_node(d["name"]):
        sys.exit(f"node already registered: {d['name']} (remove it first, or pick another name)")
    key = d.get("ssh_key", "") or ""
    node = NodeRecord(
        node_id=d["name"], name=d["name"], index=-1, ip=str(d["ip"]),
        state=NodeState.AVAILABLE.value, gpu=bool(d.get("gpu", False)), local=False,
        runtime=d.get("runtime", "apptainer"), provider="static-ssh",
        ssh_user=d.get("ssh_user", "") or "",
        ssh_key=os.path.expanduser(key) if key else "",
    )
    store.put_node(node)
    print(f"added static-ssh node {node.name} at {node.ip} "
          f"({'gpu' if node.gpu else 'cpu'}, ssh {node.ssh_user or '<cluster-default>'}"
          f"@{node.ip} key={node.ssh_key or '<cluster-default>'}, runtime {node.runtime}). "
          f"Bake apptainer into the image before running real jobs.")


def cmd_seed_nodes(args):
    store = _store()
    existing = {n.node_id: n for n in store.list_nodes()}
    added = updated = 0
    nodes = [NodeRecord(node_id=name, name=name, index=idx, ip=ip,
                        state=NodeState.AVAILABLE.value)
             for idx, name, ip in _topology()]
    host = _host_node()
    if host:
        nodes.append(host)
    for n in nodes:
        old = existing.get(n.node_id)
        if old is None:
            store.put_node(n)
            added += 1
            continue
        # Refresh a stale record whose config-derived identity changed (e.g.
        # cluster-host migrating off 127.0.0.1 to its bridge address) — but only
        # while it's idle; never rewrite a node some job currently owns.
        if (old.state == NodeState.AVAILABLE.value and not old.owner_job and
                (old.ip, old.gpu, old.local, old.runtime) != (n.ip, n.gpu, n.local, n.runtime)):
            store.put_node(n)
            updated += 1
    print(f"seeded {added} node(s) into the registry ({len(existing)} already present"
          + (f", {updated} refreshed" if updated else "") + ")"
          + (f"; host GPU node = {host.name}" if host else "; no host GPU node (CLUSTER_GPU_HOST != 1)"))


def cmd_remove_node(args):
    """Remove a node from the registry. Does NOT touch the underlying machine
    (destroying a libvirt VM is `make cluster-down`; terminating an EC2 instance
    is done in AWS) — this just deregisters it so the scheduler stops considering
    it. Refuses a node a job currently owns unless --force."""
    store = _store()
    node = store.get_node(args.name)
    if not node:
        sys.exit(f"no such node in registry: {args.name}")
    if node.owner_job and not args.force:
        sys.exit(f"{args.name} is in use by {node.owner_job} (state {node.state}); "
                 f"re-run with --force to deregister anyway")
    store.delete_node(args.name)
    print(f"removed node {args.name} ({node.adapter_key}, was {node.state})")


def cmd_fail_node(args):
    store = _store()
    node = store.get_node(args.name)
    if not node:
        sys.exit(f"no such node in registry: {args.name} (run: cluster seed-nodes)")
    if args.inject:
        print(f"[fail-node] injecting real failure on {args.name} (idx {node.index})")
        subprocess.run([os.path.join(LIBVIRT_DIR, "inject-failure.sh"), str(node.index)], check=False)
    Reconciler(store).handle_node_failure(args.name, reason=args.reason)


def cmd_clear_jobs(args):
    """Demo/dev reset: wipe job history so the dashboard starts from a blank
    slate. Not part of the open interface -- purely an operator convenience."""
    result = clear_jobs(_store(), force=args.force)
    if not result["cleared"]:
        print(f"refusing: {len(result['active'])} job(s) still active (not terminal):")
        for a in result["active"]:
            print(f"  {a['job_id']}  {a['state']}")
        sys.exit("wait for them to finish, or re-run with --force to clear anyway "
                 "(releases their nodes even though real infrastructure may still be running)")
    if result["forced_active"]:
        print(f"--force: cleared {result['forced_active']} still-active job(s) too")
    print(f"cleared {result['jobs_removed']} job(s) + audit history + "
          f"{result['logs_removed']} replay log(s); {result['nodes_reset']} node(s) reset to available")


def main(argv=None):
    p = argparse.ArgumentParser(prog="cluster", description="Outpost control-plane CLI")
    sub = p.add_subparsers(dest="cmd", required=True)

    s = sub.add_parser("submit"); s.add_argument("--spec", required=True); s.set_defaults(fn=cmd_submit)
    s = sub.add_parser("reconcile")
    s.add_argument("--once", action="store_true"); s.add_argument("--execute", action="store_true")
    s.add_argument("--interval", type=float, default=2.0); s.set_defaults(fn=cmd_reconcile)
    sub.add_parser("list").set_defaults(fn=cmd_list)
    s = sub.add_parser("status"); s.add_argument("job_id"); s.set_defaults(fn=cmd_status)
    s = sub.add_parser("result"); s.add_argument("job_id"); s.set_defaults(fn=cmd_result)
    s = sub.add_parser("logs"); s.add_argument("job_id")
    s.add_argument("--tail", type=int, default=0, help="show only the last N lines")
    s.set_defaults(fn=cmd_logs)
    s = sub.add_parser("reconciler-log", help="cross-job chronological narration log")
    s.add_argument("--tail", type=int, default=0, help="show only the last N lines")
    s.set_defaults(fn=cmd_reconciler_log)
    sub.add_parser("nodes").set_defaults(fn=cmd_nodes)
    sub.add_parser("seed-nodes").set_defaults(fn=cmd_seed_nodes)
    s = sub.add_parser("add-node", help="join a pre-provisioned ssh host (e.g. EC2) as a worker")
    s.add_argument("--spec", required=True, help="node manifest YAML (see node.ec2.example.yaml)")
    s.set_defaults(fn=cmd_add_node)
    s = sub.add_parser("remove-node", help="deregister a node from the registry (leaves the machine alone)")
    s.add_argument("name")
    s.add_argument("--force", action="store_true", help="remove even if a job currently owns it")
    s.set_defaults(fn=cmd_remove_node)
    s = sub.add_parser("fail-node"); s.add_argument("name")
    s.add_argument("--inject", action="store_true", help="also hard-kill the VM via inject-failure.sh")
    s.add_argument("--reason", default="injected failure"); s.set_defaults(fn=cmd_fail_node)
    s = sub.add_parser("clear-jobs", help="wipe job history + audit + replay logs (demo reset)")
    s.add_argument("--force", action="store_true",
                   help="also clear active (non-terminal) jobs, releasing their nodes")
    s.set_defaults(fn=cmd_clear_jobs)

    args = p.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
