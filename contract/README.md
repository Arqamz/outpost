# Intake contract — schemas

Machine-readable definitions for optional parts of the `JobSpec` a caller writes into `jobs`.
The prose contract lives in [`docs/06-interface-contract.md`](../docs/06-interface-contract.md);
this directory holds the parts that are worth validating rather than describing.

| Path | What it defines |
|---|---|
| `launch-intent/v1/launch-intent.schema.json` | the optional `launch` block: a semantic, launcher-independent statement of the process placement a job needs |

## Launch Intent

A caller that cares where its ranks land attaches a `launch` block to the spec. It says how many
ranks per node, how many cores and threads each rank gets, how those cores should relate to the
rank's GPU and NUMA node, and how strictly the request must be honoured.

It contains **no CPU ids, no device indices, and no launcher flags** — by design. The caller
cannot know those: which node the job lands on, which CPUs the cgroup actually grants, which GPU
UUIDs exist, and which launcher runs it are all decided here, after the job is submitted. So the
caller states the requirement and stops, and **this cluster owns the exact mapping** — it is the
only party that knows the allocation.

That split is the same one the rest of the interface already makes: the caller says *what to run*,
the cluster decides *how and where*. Nothing benchmark-specific enters this repo, exactly as
today — a placement policy like "one rank per GPU" is a statement about processes and hardware,
not about what the container computes.

```
JobSpec.launch          →  allocate  →  discover topology  →  resolve exact plan
   (semantic request)       (nodes)      (cpuset, NUMA, GPUs)   (per-rank cpu ids + GPU UUIDs)
                                                                        ↓
                                              compile for the launcher → preflight → verify → run
```

## The two rules that matter

**Absent is normal.** A spec with no `launch` block runs exactly as it does today: no planning
phase, no topology discovery, no behaviour change. Every existing caller and every example spec in
this repo is unaffected. This is the compatibility guarantee.

**Present but unhonourable is a failure, never a shrug.** If the block carries a `schema_version`
this cluster does not recognise, or asks for something the allocated topology or the selected
launcher cannot express and marks it strict, the job **fails and says which requirement and which
capability**. It does not run the workload with the placement quietly dropped.

That second rule is the whole reason the block is validated rather than passed through. A caller
that asked for one rank per GPU with cores local to that GPU, and got round-robin placement
instead, gets numbers that are wrong in a way nothing downstream can detect — the job succeeds,
the output parses, and the measurement is of a machine configuration nobody chose. Failing loudly
is cheaper than that every time.

## Keep the file verbatim

`launch-intent.schema.json` is a copy of the published `launch-intent/v1` definition, identified by
its `$id`. **Do not edit it in place.** A local edit means this cluster validates against rules the
caller never saw, which reintroduces exactly the silent-mismatch failure above — only now the
mismatch is in the contract itself rather than in the placement.

To adopt a change: take the new published version, and if its `$id` version segment moved, treat
it as a new version this cluster must explicitly learn to accept.

## Validating by hand

Both examples under `v1/examples/` exercise both branches of every conditional rule in the schema.
Until the test harness lands, check the schema and the examples directly:

```bash
python3 -c "
import json, glob
from jsonschema.validators import Draft202012Validator as D
schema = json.load(open('contract/launch-intent/v1/launch-intent.schema.json'))
D.check_schema(schema)
for path in sorted(glob.glob('contract/launch-intent/v1/examples/*.json')):
    errs = sorted(D(schema).iter_errors(json.load(open(path))), key=str)
    print(('FAIL ' if errs else 'ok   ') + path)
    for e in errs:
        print('       ', '/'.join(str(p) for p in e.absolute_path) or '<root>', e.message)
"
```

`jsonschema` is not currently a dependency of this repo; it arrives with the test harness.
