"""Resolve a Launch Intent against real topology into an exact Launch Plan.

PURE AND DETERMINISTIC. Same intent + same topology + same resolver version =
the same plan, byte for byte. Nothing here reads the clock, the filesystem or a
random source, and every selection walks a sorted candidate list, because a plan
an engineer approved must be the plan that runs — and "the resolver picked
differently this time" would make approval meaningless.

The intent says what placement behaviour is wanted; the topology says what the
allocation actually has. This module is the only place the two meet, and it
refuses rather than approximates: a requirement that cannot be satisfied under
`strict` fails the job naming the requirement, because a benchmark that ran on
the wrong cores produces a number that is wrong in a way nothing downstream can
detect.

RANK NUMBERING is node-major: node 0 takes global ranks 0..k-1, node 1 takes
k..2k-1. That is the natural reading of `ranks_per_node`, and the launcher is
handed an explicit per-rank mapping, so nothing depends on a launcher's own
round-robin default.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from .topology import Core, Gpu, NodeTopology, digest_of

# Bump when a change would produce a DIFFERENT plan for inputs that previously
# resolved — the plan records it so a stale plan is detectable rather than
# silently re-resolved into something else.
RESOLVER_VERSION = "placement/1"


class PlacementError(ValueError):
    """A requirement the allocation cannot satisfy. Carries what and why."""


@dataclass(frozen=True)
class RankPlacement:
    global_rank: int
    local_rank: int
    node: str
    cpu_ids: tuple[int, ...]
    # The same cores as (socket, core) pairs. Carried so a launcher adapter can
    # compile the plan WITHOUT the topology it was resolved from: OpenMPI's
    # rankfile addresses cores as socket:core, and an approved plan should be
    # compilable on its own rather than only alongside the machine snapshot.
    cpu_slots: tuple[tuple[int, int], ...]
    numa_nodes: tuple[int, ...]
    gpu_uuid: str | None
    gpu_pci_bus_id: str | None
    visible_gpu_index: int | None


@dataclass(frozen=True)
class LaunchPlan:
    plan_id: str
    launcher: dict
    ranks: tuple[RankPlacement, ...]
    validation: dict
    digests: dict

    def to_dict(self) -> dict:
        return asdict(self)


def _sorted_gpus(gpus: tuple[Gpu, ...], ordering: str) -> list[Gpu]:
    """Deterministic device order, so a rank's visible device is reproducible
    rather than whatever the driver enumerated first this boot."""
    if ordering == "fastest_first":
        # No FLOPS in the probe; memory size is the available proxy. PCI address
        # breaks ties so identical cards still order deterministically.
        return sorted(gpus, key=lambda g: (-(g.memory_mib or 0), g.pci_bus_id, g.index))
    return sorted(gpus, key=lambda g: (g.pci_bus_id, g.index))


def _core_is_local_to(core: Core, gpu: Gpu) -> bool:
    """Does this core sit next to that device?

    The driver's own CPU-affinity list is preferred: it is what nvidia-smi
    reports as local, and it does not depend on inferring locality through a
    NUMA id that a VM may have invented. NUMA equality is the fallback.
    """
    if gpu.cpu_affinity:
        return bool(set(core.cpu_ids) & set(gpu.cpu_affinity))
    if gpu.numa is not None and core.numa is not None:
        return core.numa == gpu.numa
    return False


def _take_cores(candidates: list[Core], used: set[int], count: int,
                allow_overlap: bool) -> list[Core] | None:
    """First `count` candidate cores with no processing unit already spoken for."""
    taken: list[Core] = []
    for core in candidates:
        if len(taken) == count:
            break
        if not allow_overlap and set(core.cpu_ids) & used:
            continue
        taken.append(core)
    return taken if len(taken) == count else None


def _select_cpu_ids(core_list: list[Core], smt: str) -> tuple[int, ...]:
    """Which processing units a rank actually gets from its cores.

    `physical_only` hands out ONE unit per core and reserves the whole core, so
    two ranks can never land on siblings of the same physical core and quietly
    halve each other's throughput. `allow_siblings` treats every unit as its own
    assignable resource.
    """
    if smt == "physical_only":
        return tuple(sorted(core.cpu_ids[0] for core in core_list))
    return tuple(sorted(cpu for core in core_list for cpu in core.cpu_ids))


def _ranks_per_node(intent: dict, topo: NodeTopology, mode: str,
                    warnings: list[str]) -> int:
    declared = intent["process"]["ranks_per_node"]
    if declared is not None:
        return int(declared)
    # None means "one rank per discovered GPU" — the schema already guarantees
    # gpu.strategy is one_per_rank here, so there IS something to derive from.
    if not topo.gpus:
        raise PlacementError(
            f"{topo.node}: process.ranks_per_node is null, which means one rank per "
            "discovered GPU, but no GPUs were discovered on this node")
    return len(topo.gpus)


def _assign_gpu(intent: dict, topo: NodeTopology, local_rank: int,
                ordered: list[Gpu], claimed: dict[str, str],
                mode: str) -> Gpu | None:
    gpu_spec = intent["gpu"]
    strategy = gpu_spec["strategy"]
    if strategy == "none" or gpu_spec["gpus_per_rank"] == 0:
        return None

    if strategy == "explicit":
        uuids = gpu_spec["explicit_gpu_uuids"] or []
        if local_rank >= len(uuids):
            raise PlacementError(
                f"{topo.node}: gpu.explicit_gpu_uuids has {len(uuids)} entr(ies) but this "
                f"node was resolved to at least {local_rank + 1} ranks")
        wanted = uuids[local_rank]
        found = next((g for g in ordered if g.uuid == wanted), None)
        if found is None:
            raise PlacementError(
                f"{topo.node}: gpu.explicit_gpu_uuids names {wanted!r}, which is not on "
                f"this node (it has {[g.uuid for g in ordered]})")
        gpu = found
    else:                                    # one_per_rank
        if local_rank >= len(ordered):
            raise PlacementError(
                f"{topo.node}: gpu.strategy is one_per_rank and this node was resolved to "
                f"{local_rank + 1} rank(s), but it has only {len(ordered)} GPU(s)")
        gpu = ordered[local_rank]

    if not gpu_spec["allow_sharing"] and gpu.uuid in claimed:
        raise PlacementError(
            f"{topo.node}: GPU {gpu.uuid} would be assigned to more than one rank "
            f"(already held by {claimed[gpu.uuid]}), and gpu.allow_sharing is false")
    return gpu


def _assign_cores(intent: dict, topo: NodeTopology, local_rank: int, gpu: Gpu | None,
                  used: set[int], mode: str,
                  warnings: list[str]) -> tuple[tuple[int, ...], list[Core]]:
    cpu = intent["cpu"]
    strategy, count = cpu["strategy"], cpu["cores_per_rank"]

    if strategy == "explicit":
        lists = cpu["explicit_cpu_ids"] or []
        if local_rank >= len(lists):
            raise PlacementError(
                f"{topo.node}: cpu.explicit_cpu_ids has {len(lists)} entr(ies) but this "
                f"node was resolved to at least {local_rank + 1} ranks")
        wanted = tuple(sorted(lists[local_rank]))
        outside = sorted(set(wanted) - set(topo.allowed_cpu_ids))
        if outside:
            raise PlacementError(
                f"{topo.node}: cpu.explicit_cpu_ids names CPU(s) {outside} that are not in "
                f"this allocation's allowed cpuset {list(topo.allowed_cpu_ids)}")
        clash = sorted(set(wanted) & used)
        if clash and not cpu["allow_overlap"]:
            raise PlacementError(
                f"{topo.node}: rank {local_rank}'s explicit CPUs {clash} are already "
                "assigned to another rank, and cpu.allow_overlap is false")
        cores = [c for c in topo.allowed_cores() if set(c.cpu_ids) & set(wanted)]
        return wanted, cores

    if strategy == "none" or count is None:
        return (), []

    candidates = topo.allowed_cores()
    if not candidates:
        raise PlacementError(
            f"{topo.node}: cpu.cores_per_rank is {count} but no usable cores were "
            "discovered in the allocation's cpuset")

    chosen = None
    if strategy == "closest_to_gpu":
        if gpu is None:
            raise PlacementError(
                f"{topo.node}: cpu.strategy is closest_to_gpu but rank {local_rank} was "
                "assigned no GPU to be close to")
        local = [c for c in candidates if _core_is_local_to(c, gpu)]
        if not local:
            message = (f"{topo.node}: cpu.strategy is closest_to_gpu but the topology "
                       f"exposes no cores local to GPU {gpu.uuid or gpu.index} "
                       f"(numa={gpu.numa}, source={gpu.numa_source}, "
                       f"confidence={topo.confidence})")
            if mode == "strict":
                raise PlacementError(message)
            warnings.append(message + " — falling back to any available core")
        else:
            chosen = _take_cores(local, used, count, cpu["allow_overlap"])
            if chosen is None and mode == "strict":
                raise PlacementError(
                    f"{topo.node}: rank {local_rank} needs {count} core(s) local to GPU "
                    f"{gpu.uuid or gpu.index}, but only {len(local)} such core(s) exist "
                    f"and {len([c for c in local if set(c.cpu_ids) & used])} are taken")
            if chosen is None:
                warnings.append(
                    f"{topo.node}: rank {local_rank} could not get {count} GPU-local "
                    "core(s); falling back to any available core")

    if chosen is None:
        chosen = _take_cores(candidates, used, count, cpu["allow_overlap"])
    if chosen is None:
        raise PlacementError(
            f"{topo.node}: rank {local_rank} needs {count} core(s) but only "
            f"{len([c for c in candidates if not set(c.cpu_ids) & used])} of "
            f"{len(candidates)} are still free (cpu.allow_overlap is "
            f"{cpu['allow_overlap']})")
    return _select_cpu_ids(chosen, cpu["smt"]), chosen


def resolve(intent: dict, topologies: list[NodeTopology], *,
            allocation_id: str = "") -> LaunchPlan:
    """Intent + allocated topology -> an exact per-rank Launch Plan.

    Raises PlacementError when a requirement cannot be met and the intent is
    strict. Advisory mode records a warning on the plan instead, but only ever
    for requirements whose loss does not change what is being measured.
    """
    if not topologies:
        raise PlacementError("no allocated nodes to resolve against")

    mode = intent["validation"]["mode"]
    warnings: list[str] = []
    ordered_nodes = sorted(topologies, key=lambda t: t.node)

    launchers = {(t.launcher or {}).get("type") for t in ordered_nodes}
    launchers.discard(None)
    launchers.discard("")
    if len(launchers) > 1:
        raise PlacementError(
            f"allocated nodes disagree on the launcher ({sorted(launchers)}); one plan "
            "cannot be compiled for two")
    versions = {(t.launcher or {}).get("version") for t in ordered_nodes} - {None, ""}
    if len(versions) > 1:
        warnings.append(f"allocated nodes run different launcher versions {sorted(versions)}; "
                        "rank placement syntax may not be interpreted identically")

    ranks: list[RankPlacement] = []
    global_rank = 0
    for topo in ordered_nodes:
        for w in topo.warnings:
            warnings.append(f"{topo.node}: {w}")
        per_node = _ranks_per_node(intent, topo, mode, warnings)
        if per_node < 1:
            raise PlacementError(f"{topo.node}: resolved to {per_node} ranks")

        # `physical_only` means "never count two SMT siblings as two cores". On a
        # node with no core identity, siblings are indistinguishable from separate
        # cores, so the count would be wrong — and wrong in a direction nobody can
        # see from the output. Refuse rather than pick a plausible number.
        wants_cores = intent["cpu"]["strategy"] != "none" and \
            intent["cpu"]["cores_per_rank"] is not None
        if (wants_cores and intent["cpu"]["smt"] == "physical_only"
                and topo.cpus and not any(c.core is not None for c in topo.cpus)):
            message = (f"{topo.node}: cpu.smt is physical_only but this node exposes no "
                       "per-CPU core identity, so SMT siblings cannot be told apart from "
                       "distinct cores and a physical-core count cannot be honoured")
            if mode == "strict":
                raise PlacementError(message)
            warnings.append(message + " — treating every processing unit as a core")

        ordered_gpus = _sorted_gpus(topo.gpus, intent["gpu"]["ordering"])
        used_cpus: set[int] = set()
        claimed_gpus: dict[str, str] = {}

        for local_rank in range(per_node):
            gpu = _assign_gpu(intent, topo, local_rank, ordered_gpus, claimed_gpus, mode)
            cpu_ids, cores = _assign_cores(intent, topo, local_rank, gpu, used_cpus,
                                           mode, warnings)
            if gpu is not None:
                claimed_gpus[gpu.uuid] = f"rank {global_rank}"
            used_cpus.update(cpu for core in cores for cpu in core.cpu_ids)

            numa = tuple(sorted({c.numa for c in cores if c.numa is not None}))
            if intent["memory"]["strategy"] != "none" and not numa:
                message = (f"{topo.node}: memory.strategy is "
                           f"{intent['memory']['strategy']!r} but rank {local_rank} has no "
                           "known NUMA node, so the policy cannot be expressed")
                if mode == "strict":
                    raise PlacementError(message)
                warnings.append(message)

            # Only cores whose units this rank actually got — an explicit CPU
            # list may cover part of a core, and claiming the whole one would
            # overstate what was reserved.
            slots = tuple(sorted(
                (c.socket, c.core) for c in cores
                if c.socket is not None and c.core is not None
                and set(c.cpu_ids) & set(cpu_ids)))

            ranks.append(RankPlacement(
                global_rank=global_rank, local_rank=local_rank, node=topo.node,
                cpu_ids=cpu_ids, cpu_slots=slots, numa_nodes=numa,
                gpu_uuid=gpu.uuid if gpu else None,
                gpu_pci_bus_id=gpu.pci_bus_id if gpu else None,
                # With one device made visible per rank, the application sees it
                # at index 0 regardless of its physical enumeration — that is the
                # point of pinning visibility rather than trusting device order.
                visible_gpu_index=0 if gpu else None,
            ))
            global_rank += 1

    _assert_globally_consistent(intent, ranks)

    digests = {
        "intent": digest_of(intent),
        "topology": digest_of([t.to_dict() for t in ordered_nodes]),
        "allocation": digest_of(allocation_id or [t.node for t in ordered_nodes]),
        "resolver": RESOLVER_VERSION,
    }
    plan_body = {"ranks": [asdict(r) for r in ranks], "digests": digests}
    return LaunchPlan(
        plan_id=f"lp-{digest_of(plan_body)}",
        launcher=dict(ordered_nodes[0].launcher or {}),
        ranks=tuple(ranks),
        validation={"status": "valid", "warnings": tuple(warnings)},
        digests=digests,
    )


def _assert_globally_consistent(intent: dict, ranks: list[RankPlacement]) -> None:
    """The checks that only make sense once every rank is placed.

    Per-rank assignment already refuses to reuse a CPU or a GPU within a node.
    This re-derives both across the whole plan, so a future change to the
    assignment path cannot quietly reintroduce a conflict the plan then presents
    as valid.
    """
    if not intent["gpu"]["allow_sharing"]:
        seen: dict[str, int] = {}
        for rank in ranks:
            if rank.gpu_uuid is None:
                continue
            if rank.gpu_uuid in seen:
                raise PlacementError(
                    f"GPU {rank.gpu_uuid} is assigned to both rank {seen[rank.gpu_uuid]} "
                    f"and rank {rank.global_rank}, and gpu.allow_sharing is false")
            seen[rank.gpu_uuid] = rank.global_rank

    if not intent["cpu"]["allow_overlap"]:
        owner: dict[tuple[str, int], int] = {}
        for rank in ranks:
            for cpu in rank.cpu_ids:
                key = (rank.node, cpu)
                if key in owner:
                    raise PlacementError(
                        f"{rank.node}: CPU {cpu} is assigned to both rank {owner[key]} and "
                        f"rank {rank.global_rank}, and cpu.allow_overlap is false")
                owner[key] = rank.global_rank
