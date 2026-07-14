# reconciler — Outpost's control plane

Receives generic container jobs, schedules them (GPU→host, CPU→VMs) with
exclusive locks, and drives each through the state machine, running the
container via an adapter. **Dry-run by default** (nothing launched). Full prose:
[`../docs/03-control-plane.md`](../docs/03-control-plane.md).

## Pieces

| File | Role |
|---|---|
| `states.py` | `JobState`/`NodeState` + legal transition tables (the source of truth) |
| `models.py` | `JobSpec` (generic container job), `JobRecord`, `NodeRecord` (capabilities), `RunResult`; collection names |
| `store.py` | `Store` iface; `FileStore` (default, fcntl) + `MongoStore` (`CLUSTER_MONGO_URI`); **capability-aware atomic `claim_node`** |
| `registry.py` | `NodeRegistry` — `claim(require_gpu)`, `release`, `quarantine` |
| `adapter.py` | `ProviderAdapter` ABC (`provision/deprovision/bootstrap/run/collect`) + `LibvirtAdapter`, `LocalHostAdapter`, `NullAdapter`; `container_argv()` |
| `reconciler.py` | the driver: `submit`, `tick`, phases, `_adapter_for(node)`, `handle_node_failure` |
| `audit.py` | append-only transition log |
| `cli.py` | `cluster` CLI (local client of the open interface) |

## Flow driven

```
submit -> provision -> bootstrap -> run -> collect -> teardown -> validate -> promote
                                     └─ node failure ─> quarantine node + FAIL job (clean)
```

Each phase calls `adapter.{provision,bootstrap,run,collect,deprovision}`; the
adapter is chosen per node by `node.local` (host → `LocalHostAdapter`, else
`LibvirtAdapter`). `run` executes the container (`apptainer --nv` on the host, or
ssh+apptainer on a VM); `collect` drops artifacts into `${CLUSTER_DROPZONE}/<job_id>/`.

## Try it (no infrastructure, nothing launched)

```bash
cluster seed-nodes                       # register 8 VMs + cluster-host GPU node
cluster submit --spec job.example.yaml   # -> job-xxxxxxxx
cluster reconcile --once                 # advance one phase per actionable job
cluster reconcile --once                 # ... repeat until PROMOTED
cluster list ; cluster status <id> ; cluster result <id>

# failure-injection exit criterion (control-plane only, no VM killed):
cluster fail-node cluster-node-02            # add --inject to also virsh-destroy it
```

`reconcile --execute` swaps `NullAdapter` for the real adapters (`LocalHost` +
`Libvirt`). Leave it off for a pure control-plane exercise.

## Store backends

- **Default:** `FileStore` at `.var/reconciler/state.json` — no dependencies,
  persists across CLI calls, real cross-process locking via `fcntl`.
- **MongoDB:** `export CLUSTER_MONGO_URI=mongodb://localhost:27017` — same
  `jobs`/`nodes`/`audit` collections, atomic claims via
  `find_one_and_update`.
