"""Normalize a node's raw topology probe into the model the resolver plans against.

The probe collects; this interprets. Everything here is a pure function of the
probe's JSON, so the awkward part — `nvidia-smi topo -m`, whose columns move with
the GPU and NIC count — is parsed where it can be held against fixtures from real
machines instead of guessed at in shell.

SCOPE AND CONFIDENCE ARE PART OF THE DATA. A VM shows a synthetic, flat topology
and a container may see the host's /sys while confined to a slice of it. A plan
resolved against either must not claim host-level precision, so what could be
seen travels with what was seen, and the resolver refuses `closest_to_gpu` under
`strict` when the topology cannot actually support the claim.
"""
from __future__ import annotations

import hashlib
import json
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path

PROBE_PATH = Path(__file__).resolve().parent / "probes" / "topology.sh"

# Separator inside `nvidia-smi topo -m`: two or more spaces. Single spaces are
# data ("CPU Affinity" is one column, " X " is one cell), which is exactly why
# a naive whitespace split mangles this table.
_COLUMNS = re.compile(r"\s{2,}")
# Link classes worth distinguishing: NV# is NVLink (count = bandwidth), PIX/PXB
# stay on one PCIe host bridge, NODE/SYS cross one or more CPU sockets.
_LINK = re.compile(r"^(NV\d+|PIX|PXB|NODE|SYS|X)$")


@dataclass(frozen=True)
class Cpu:
    """One processing unit. Two sharing (socket, core) are SMT siblings."""
    id: int
    core: int | None
    socket: int | None
    numa: int | None


@dataclass(frozen=True)
class Core:
    """A physical core and the processing units on it."""
    socket: int | None
    core: int | None
    numa: int | None
    cpu_ids: tuple[int, ...]

    @property
    def key(self) -> tuple:
        return (self.socket if self.socket is not None else -1,
                self.core if self.core is not None else -1)


@dataclass(frozen=True)
class NumaNode:
    id: int
    cpu_ids: tuple[int, ...]
    memory_mib: int | None


@dataclass(frozen=True)
class Gpu:
    index: int
    uuid: str
    pci_bus_id: str
    memory_mib: int | None
    name: str
    numa: int | None                      # driver's view wins over sysfs
    cpu_affinity: tuple[int, ...]         # cores the driver calls local; () = unknown
    links: dict = field(default_factory=dict)   # peer label -> link class
    numa_source: str = "unknown"          # topo_matrix | sysfs | unknown


@dataclass(frozen=True)
class NodeTopology:
    node: str
    hostname: str
    scope: str                            # host | guest | container
    confidence: str                       # high | low
    allowed_cpu_ids: tuple[int, ...]
    cpus: tuple[Cpu, ...]
    cores: tuple[Core, ...]
    numa_nodes: tuple[NumaNode, ...]
    gpus: tuple[Gpu, ...]
    launcher: dict
    warnings: tuple[str, ...] = ()

    def to_dict(self) -> dict:
        return asdict(self)

    def digest(self) -> str:
        """Stable identity for this snapshot, so a plan can detect that the
        machine it was resolved against is no longer the machine in front of it."""
        return digest_of(self.to_dict())

    def allowed_cores(self) -> list[Core]:
        """Cores with at least one processing unit inside the allocation, in a
        deterministic order — the resolver's candidate set."""
        allowed = set(self.allowed_cpu_ids)
        usable = [Core(c.socket, c.core, c.numa,
                       tuple(sorted(set(c.cpu_ids) & allowed)))
                  for c in self.cores]
        return sorted([c for c in usable if c.cpu_ids], key=lambda c: (c.key, c.cpu_ids))


def digest_of(payload: object) -> str:
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode()).hexdigest()[:12]


def expand_cpulist(spec: str | None) -> tuple[int, ...]:
    """`0-3,8,12-13` -> (0,1,2,3,8,12,13). Empty or unparseable -> ()."""
    if not spec:
        return ()
    out: set[int] = set()
    for chunk in str(spec).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            lo, _, hi = chunk.partition("-")
            try:
                out.update(range(int(lo), int(hi) + 1))
            except ValueError:
                continue
        else:
            try:
                out.add(int(chunk))
            except ValueError:
                continue
    return tuple(sorted(out))


def parse_topo_matrix(text: str | None) -> dict[str, dict]:
    """`nvidia-smi topo -m` -> {"GPU0": {"cpu_affinity": (...), "numa": 0, "links": {...}}}.

    Parsed here rather than in the probe because the column set varies by
    machine: NIC rows appear only when there are NICs, and `NUMA Affinity` and
    `GPU NUMA ID` are newer additions. Anything unrecognised is skipped rather
    than guessed at — a wrong affinity silently binds a rank to the far socket,
    which is the exact failure this whole feature exists to prevent.
    """
    if not text:
        return {}
    header: list[str] | None = None
    rows: dict[str, dict] = {}
    for raw_line in text.replace("\t", "    ").splitlines():
        if not raw_line.strip():
            continue
        if raw_line.lstrip().startswith("Legend"):
            break
        cells = [c.strip() for c in _COLUMNS.split(raw_line.strip())]
        if header is None:
            # The header's first cell is the (unnamed) row-label column; some
            # versions emit it as leading whitespace, which strip() already ate.
            if cells and cells[0].startswith(("GPU", "NIC")):
                header = cells
            continue
        label, values = cells[0], cells[1:]
        if not label.startswith(("GPU", "NIC")):
            continue
        # strict=False on purpose: a NIC row is shorter than the header (it has
        # no CPU/NUMA affinity columns), and a GPU row may be longer on versions
        # that append columns we do not read. Pairing what lines up is correct;
        # refusing the row because the widths differ would drop real data.
        named = dict(zip(header, values, strict=False))
        links = {peer: value for peer, value in named.items()
                 if peer.startswith(("GPU", "NIC")) and _LINK.match(value)}
        numa_raw = (named.get("NUMA Affinity") or "").strip()
        rows[label] = {
            "cpu_affinity": expand_cpulist(named.get("CPU Affinity")),
            "numa": int(numa_raw) if numa_raw.lstrip("-").isdigit() and numa_raw != "-1"
            else None,
            "links": links,
        }
    return rows


def _cores_from_cpus(cpus: list[Cpu]) -> tuple[Core, ...]:
    """Group processing units into physical cores.

    With no socket/core identity there is nothing to group BY. Folding every unit
    into one "core" would make the node look like it has a single core, so each
    unit stands alone instead — and its socket/core stay None rather than being
    given a made-up value, because a fabricated identity would be written into a
    rankfile as if it addressed real hardware. SMT siblings cannot be identified
    on such a node, which is why the resolver refuses physical-core counting
    there rather than miscounting in either direction.
    """
    grouped: dict[tuple, list[Cpu]] = {}
    identity: dict[tuple, tuple[int | None, int | None]] = {}
    for cpu in cpus:
        known = cpu.socket is not None and cpu.core is not None
        key = (cpu.socket, cpu.core) if known else ("\0unknown", cpu.id)
        grouped.setdefault(key, []).append(cpu)
        identity[key] = (cpu.socket, cpu.core) if known else (None, None)
    cores = []
    for key, members in grouped.items():
        socket, core = identity[key]
        numa = next((m.numa for m in members if m.numa is not None), None)
        cores.append(Core(socket, core, numa, tuple(sorted(m.id for m in members))))
    return tuple(sorted(cores, key=lambda c: (c.key, c.cpu_ids)))


def normalize(raw: dict, node_name: str) -> NodeTopology:
    """Probe JSON -> NodeTopology. Records what was missing rather than filling it."""
    warnings: list[str] = []

    cpus = tuple(Cpu(id=int(c["id"]), core=c.get("core"), socket=c.get("socket"),
                     numa=c.get("numa"))
                 for c in raw.get("cpus") or [] if c.get("id") is not None)
    cores = _cores_from_cpus(list(cpus))

    numa_nodes = tuple(NumaNode(id=int(n["id"]), cpu_ids=expand_cpulist(n.get("cpulist")),
                                memory_mib=n.get("memory_mib"))
                       for n in raw.get("numa") or [] if n.get("id") is not None)

    allowed = expand_cpulist(raw.get("allowed_cpus")) or expand_cpulist(raw.get("online_cpus"))
    if not allowed and cpus:
        # No cpuset source at all: assume every processing unit we can see, and
        # say so — this is the difference between "the job may use these" and
        # "these exist", and only the first is safe to bind against.
        allowed = tuple(sorted(c.id for c in cpus))
        warnings.append("no cgroup cpuset or affinity mask was readable; assuming every "
                        "online CPU is usable, which may be wider than the allocation")

    matrix = parse_topo_matrix(raw.get("topo_matrix"))
    gpus = []
    for g in raw.get("gpus") or []:
        label = f"GPU{g.get('index')}"
        from_matrix = matrix.get(label, {})
        numa = from_matrix.get("numa")
        source = "topo_matrix"
        if numa is None:
            numa, source = g.get("numa"), ("sysfs" if g.get("numa") is not None
                                           else "unknown")
        gpus.append(Gpu(
            index=int(g["index"]), uuid=g.get("uuid") or "",
            pci_bus_id=g.get("pci_bus_id") or "", memory_mib=g.get("memory_mib"),
            name=g.get("name") or "", numa=numa,
            cpu_affinity=from_matrix.get("cpu_affinity", ()),
            links=from_matrix.get("links", {}), numa_source=source,
        ))
    gpus_t = tuple(sorted(gpus, key=lambda x: x.index))

    scope = raw.get("scope") or "host"
    has_structure = any(c.socket is not None and c.core is not None for c in cpus)
    confidence = "high"
    if scope != "host":
        confidence = "low"
        warnings.append(f"topology observed from inside a {scope}; socket, core and NUMA "
                        "structure is what the guest was shown, not necessarily the "
                        "hardware's")
    if not has_structure:
        confidence = "low"
        warnings.append("no per-CPU socket/core topology was readable; SMT siblings cannot "
                        "be identified, so physical-core counting is not possible")
    if not numa_nodes:
        confidence = "low"
        warnings.append("no NUMA nodes were readable; memory locality cannot be verified")
    if gpus_t and all(g.numa is None for g in gpus_t):
        warnings.append("no GPU-to-NUMA affinity from either nvidia-smi topo or sysfs; "
                        "GPU-local core selection has nothing to work from")

    return NodeTopology(
        node=node_name, hostname=raw.get("hostname") or "", scope=scope,
        confidence=confidence, allowed_cpu_ids=allowed, cpus=cpus, cores=cores,
        numa_nodes=numa_nodes, gpus=gpus_t, launcher=raw.get("launcher") or {},
        warnings=tuple(warnings),
    )


def probe_source() -> str:
    """The probe script's text, for shipping to a node over the existing fabric."""
    return PROBE_PATH.read_text()
