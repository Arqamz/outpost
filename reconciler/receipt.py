"""Compare what was planned against what actually happened.

Three sources, not two, and the difference between them is the point:

  the PLAN            what was asked for
  --report-bindings   what the launcher says it did
  the rank's probe    what the OS actually enforced, read from inside the
                      container the workload runs in

Plan vs observation alone tells you something went wrong. Adding the launcher's
own report tells you WHERE: a rank the launcher reports as "not bound" is a
launcher or syntax problem, while a rank the launcher bound but whose observed
mask is the whole machine is the boundary between them losing it. Those need
different fixes, and a receipt that could not tell them apart would send someone
looking in the wrong place.

Nothing here is silent. A rank with no observation is a mismatch, not an
absence; a plan that could not be checked is `unverified`, never `verified`.
"""
from __future__ import annotations

import re
from dataclasses import asdict, dataclass, field

from .placement import LaunchPlan
from .topology import expand_cpulist

# `[host:04469] MCW rank 0 bound to socket 0[core 0[hwt 0]]: [B/././.]`
_BOUND = re.compile(
    r"MCW rank (?P<rank>\d+) bound to (?P<where>.+?):\s*\[(?P<mask>[B./ ]+)\]")
# `[host:04469] MCW rank 0 is not bound (or bound to all available processors)`
_UNBOUND = re.compile(r"MCW rank (?P<rank>\d+) is not bound")
_CORE = re.compile(r"socket (?P<socket>\d+)\[core (?P<core>\d+)")


@dataclass(frozen=True)
class RankVerification:
    global_rank: int
    node: str
    observed: bool
    planned_cpu_ids: tuple[int, ...]
    observed_cpu_ids: tuple[int, ...]
    planned_gpu_uuid: str | None
    observed_gpu_uuid: str | None
    launcher_claim: str | None            # what --report-bindings said, verbatim
    mismatches: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()


@dataclass(frozen=True)
class LaunchReceipt:
    plan_id: str
    status: str                           # verified | mismatched | unverified
    ranks: tuple[RankVerification, ...]
    mismatches: tuple[str, ...] = ()
    notes: tuple[str, ...] = ()
    sources: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return asdict(self)


def parse_binding_report(text: str | None) -> dict[int, dict]:
    """`--report-bindings` stderr -> {rank: {"claim": str, "bound": bool, "cores": [...]}}.

    The `is not bound` form is the valuable one: it is the launcher stating
    outright that it applied no binding, which no amount of comparing masks
    afterwards can establish as clearly.
    """
    out: dict[int, dict] = {}
    for line in (text or "").splitlines():
        unbound = _UNBOUND.search(line)
        if unbound:
            out[int(unbound.group("rank"))] = {
                "claim": line.strip(), "bound": False, "cores": []}
            continue
        bound = _BOUND.search(line)
        if bound:
            where = bound.group("where")
            out[int(bound.group("rank"))] = {
                "claim": line.strip(), "bound": True,
                "cores": [(int(m.group("socket")), int(m.group("core")))
                          for m in _CORE.finditer(where)],
            }
    return out


def _observations_by_rank(observations: list[dict]) -> dict[int, dict]:
    keyed: dict[int, dict] = {}
    for doc in observations:
        rank = doc.get("global_rank")
        if rank is not None:
            keyed[int(rank)] = doc
    return keyed


def _verify_rank(planned, observation: dict | None,
                 report: dict | None) -> RankVerification:
    mismatches: list[str] = []
    notes: list[str] = []

    if observation is None:
        return RankVerification(
            global_rank=planned.global_rank, node=planned.node, observed=False,
            planned_cpu_ids=planned.cpu_ids, observed_cpu_ids=(),
            planned_gpu_uuid=planned.gpu_uuid, observed_gpu_uuid=None,
            launcher_claim=(report or {}).get("claim"),
            mismatches=("no preflight observation was returned for this rank; its placement "
                        "is unknown, which is not the same as correct",))

    observed_cpus = expand_cpulist(observation.get("allowed_cpus"))
    online = expand_cpulist(observation.get("online_cpus"))

    if planned.cpu_ids:
        missing = sorted(set(planned.cpu_ids) - set(observed_cpus))
        if missing:
            mismatches.append(
                f"planned CPU(s) {missing} are not in the rank's allowed set "
                f"{list(observed_cpus)} — it cannot run where the plan says it does")
        # An unbound rank is allowed everything, which is a SUPERSET of its plan
        # and would otherwise pass a subset check while no binding happened at all.
        elif online and set(observed_cpus) == set(online) and \
                set(planned.cpu_ids) != set(online):
            mismatches.append(
                f"the rank is allowed every CPU on the node ({observation.get('online_cpus')}) "
                f"but was planned onto {list(planned.cpu_ids)} — the binding was not applied")
        else:
            extra = sorted(set(observed_cpus) - set(planned.cpu_ids))
            if extra:
                # Expected under `physical_only`: binding to a core admits both of
                # its SMT siblings, and only one was named in the plan. Recorded
                # rather than judged, because the plan does not carry sibling
                # identity and guessing which case this is would be inventing.
                notes.append(
                    f"the rank may also run on {extra}, beyond the {list(planned.cpu_ids)} "
                    "planned — expected when binding to whole cores whose SMT siblings were "
                    "reserved but not individually named")

    if report is not None and report.get("bound") is False and planned.cpu_ids:
        mismatches.append(f"the launcher reported this rank as not bound: {report['claim']}")

    observed_gpu = (observation.get("cuda_visible_devices") or "").strip() or None
    if planned.gpu_uuid:
        if observed_gpu != planned.gpu_uuid:
            mismatches.append(
                f"planned GPU {planned.gpu_uuid} but the rank was given "
                f"CUDA_VISIBLE_DEVICES={observed_gpu!r}")
        driver = (observation.get("driver_gpu_uuids") or "").split(",")
        if planned.gpu_uuid not in [d.strip() for d in driver if d.strip()]:
            mismatches.append(
                f"planned GPU {planned.gpu_uuid} is not among the devices the driver exposes "
                f"on {observation.get('hostname')}")

    if observation.get("hostname") and planned.node not in str(observation["hostname"]):
        notes.append(f"planned node {planned.node!r} but the rank reported hostname "
                     f"{observation['hostname']!r}; these may legitimately differ")

    current = observation.get("current_cpu")
    if current is not None and observed_cpus and current not in observed_cpus:
        notes.append(f"the rank was observed running on CPU {current}, outside its own "
                     "allowed set — a point sample, but an odd one")

    return RankVerification(
        global_rank=planned.global_rank, node=planned.node, observed=True,
        planned_cpu_ids=planned.cpu_ids, observed_cpu_ids=observed_cpus,
        planned_gpu_uuid=planned.gpu_uuid, observed_gpu_uuid=observed_gpu,
        launcher_claim=(report or {}).get("claim"),
        mismatches=tuple(mismatches), notes=tuple(notes),
    )


def build_receipt(plan: LaunchPlan, observations: list[dict], *,
                  binding_report: str | None = None,
                  preflight_ran: bool = True) -> LaunchReceipt:
    """Reconcile a plan against what its ranks reported. Never raises.

    Failure is DATA here, not an exception: a mismatched receipt is exactly the
    evidence a run needs to carry, and raising would lose it at the point it
    became worth keeping.

    `preflight_ran=False` is the `require_preflight: false` case and is the whole
    reason `unverified` exists as a status distinct from `mismatched`. Nothing
    contradicted the plan, but nothing confirmed it either — reporting that as a
    mismatch would cry wolf, and reporting it as verified would be a lie.
    """
    keyed = _observations_by_rank(observations)
    report = parse_binding_report(binding_report)

    if not preflight_ran:
        # Every rank is recorded as unobserved, but WITHOUT the "no observation"
        # mismatch: none was expected. The launcher's own claim still rides along
        # when it made one — that evidence never depended on the probe running.
        unchecked = tuple(
            RankVerification(
                global_rank=p.global_rank, node=p.node, observed=False,
                planned_cpu_ids=p.cpu_ids, observed_cpu_ids=(),
                planned_gpu_uuid=p.gpu_uuid, observed_gpu_uuid=None,
                launcher_claim=(report.get(p.global_rank) or {}).get("claim"))
            for p in plan.ranks)
        return LaunchReceipt(
            plan_id=plan.plan_id, status="unverified", ranks=unchecked,
            notes=("no placement preflight was run (validation.require_preflight was false), "
                   "so this plan is recorded as issued but not confirmed",),
            sources={"observations": 0, "planned_ranks": len(plan.ranks),
                     "binding_report": bool(report), "preflight_ran": False})

    ranks = tuple(_verify_rank(p, keyed.get(p.global_rank), report.get(p.global_rank))
                  for p in plan.ranks)

    mismatches = [f"rank {r.global_rank}: {m}" for r in ranks for m in r.mismatches]
    notes = [f"rank {r.global_rank}: {n}" for r in ranks for n in r.notes]

    extra = sorted(set(keyed) - {p.global_rank for p in plan.ranks})
    if extra:
        mismatches.append(
            f"observations arrived for rank(s) {extra}, which the plan does not contain — "
            "more processes ran than were planned")

    if mismatches:
        status = "mismatched"
    elif not plan.ranks or all(not r.observed for r in ranks):
        status = "unverified"
    else:
        status = "verified"

    return LaunchReceipt(
        plan_id=plan.plan_id, status=status, ranks=ranks,
        mismatches=tuple(mismatches), notes=tuple(notes),
        sources={
            "observations": len(keyed),
            "planned_ranks": len(plan.ranks),
            "binding_report": bool(report),
            "preflight_ran": True,
            # What this receipt can and cannot establish about GPUs, stated in
            # the artifact so nobody reads more into it than it proves:
            # CUDA_VISIBLE_DEVICES confirms the launcher DELIVERED the device to
            # the rank, and the driver list confirms the device EXISTS there.
            # Neither confirms the application enumerated it at a given index —
            # that needs a CUDA runtime in the container, which is not assumed.
            "gpu_visibility_verified_by": "env + driver enumeration, not CUDA",
        },
    )
