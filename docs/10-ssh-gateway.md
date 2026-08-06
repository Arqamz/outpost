# 10 · The `ssh tashkil` gateway — a clean front door

> **Status: implemented + tested.** The gateway (`infra/gateway/tashkil-gw`),
> the installer (`infra/gateway/install.sh`), the stdin-submit + `fetch` CLI
> paths, and the client wrapper (`bin/tashkil`) are all in the repo and
> unit-tested (dispatch, denials, injection-safety, stdout purity). Bringing it
> up on your control-plane host is the part you run.

## What it is

Outpost's contract is already open: **a JobSpec goes in, artifacts come out of
the drop-zone.** The gateway is nothing more than a *remote transport for that
same contract* — it does not add a new interface, it puts the existing one
behind ssh:

```
ssh tashkil submit < job.yaml     # JobSpec in  (stdin)   -> prints a job-id
ssh tashkil status <job-id>       # poll until it reads: promoted
ssh tashkil fetch  <job-id> | tar x   # artifacts out (stdout tar) -> land locally
```

The reconciler daemon already running on the control plane picks the job up and
distributes it across your nodes (VMs, host GPU, static-ssh GPU, k8s gangs)
exactly as it does for a locally-submitted job. The gateway touches none of
that — it just writes the JobSpec into the same store and streams the same
drop-zone back.

## Why it's safe to expose

`ssh tashkil` is a **forced command**. The gateway user has no shell; its
`authorized_keys` pins `command="…/tashkil-gw"` plus `no-pty` and all the
`no-*-forwarding` restrictions, so a key holder can *only* run the whitelist
below — never an interactive shell, a file read, a port-forward, or an admin
command.

| Exposed (job portal) | Denied (operator-only, stays local) |
|----------------------|-------------------------------------|
| `submit` (spec on stdin) | `reconcile` |
| `fetch` (artifacts to stdout) | `seed-nodes`, `add-node`, `remove-node` |
| `status`, `result`, `list`, `nodes` | `fail-node` |
| `logs [--tail N]`, `reconciler-log [--tail N]` | `clear-jobs` |

The gateway never evaluates the client's string through a shell: it word-splits
it once, hard-matches the subcommand, validates job-ids/`--tail` against a
strict charset, and `exec`s `cluster` with a fixed argv per command. A request
like `status job;rm -rf /` is rejected as a bad job-id, not run.

## Install (on the control-plane host)

One command, as root, with a client's **public** key:

```bash
sudo infra/gateway/install.sh --key ~/keys/laptop.pub
#   or:  make gateway-install KEY=~/keys/laptop.pub
#   or paste the key inline:
sudo infra/gateway/install.sh --key "ssh-ed25519 AAAA... alice@laptop"
```

It is idempotent — re-run to authorize more client keys or repair permissions.
What it sets up:

- a dedicated, shell-less **`tashkil`** unix user;
- that user's `authorized_keys` line: the forced command + `no-pty` +
  `no-*-forwarding`, carrying the client key you passed;
- a shared **`outpost`** group containing both `tashkil` and the **operator**
  (whoever runs the reconciler daemon — auto-detected as the repo owner), with
  the store, drop-zone, and logs under `.var/` made group-writable + setgid.

That last point is the one real requirement: the ssh submit (running as
`tashkil`) and the daemon's ticks (running as the operator) **share one
FileStore**, so both must be able to read and write it. After installing, run
the daemon with a group-writable umask so files it creates stay group-writable:

```bash
umask 002 && cluster reconcile --execute
#   (or set UMask=002 in its systemd unit; re-login once for group membership)
```

> Choosing the ForceCommand-on-an-existing-user model instead (no dedicated
> user)? Add to `sshd_config`:
> `Match User <you>` → `ForceCommand /path/to/infra/gateway/tashkil-gw`, then
> `systemctl reload sshd`. The store is already yours, so the group dance above
> is unnecessary — but every one of your keys then hits the portal, which is why
> the dedicated user is the default.

## Client setup

Add an alias to `~/.ssh/config` on the machine you submit from:

```sshconfig
Host tashkil
    HostName <control-plane-ip>      # LAN or Tailscale/WireGuard address
    User tashkil
    IdentityFile ~/.ssh/tashkil      # the private key for the authorized pubkey
```

Now raw ssh already works:

```bash
ssh tashkil help
ssh tashkil submit < job.yaml
ssh tashkil fetch <job-id> | tar x
```

Optionally drop `bin/tashkil` on your PATH for sugar that wires the pipes for
you:

```bash
tashkil submit job.yaml            # -> job-id
tashkil status <job-id>
tashkil fetch  <job-id> [dir]      # untars artifacts into dir (default: .)
tashkil run    job.yaml [dir]      # submit, wait for terminal state, then fetch
```

(`TASHKIL_HOST=other-alias tashkil …` overrides the ssh alias.)

## End-to-end from a laptop

```bash
$ tashkil run job.gpu.example.yaml results/
submitted job-1837…; waiting for it to finish...
  [job-1837…] provisioning
  [job-1837…] running
  [job-1837…] promoted
results/job-1837…/stdout.log
results/job-1837…/…
artifacts -> results/job-1837…/
```

Same JobSpec, same drop-zone, same scheduler — now reachable over one ssh alias
from anywhere that can dial the control plane.

## Revoking access

Delete the client's line from `~tashkil/.ssh/authorized_keys`. To retire the
whole front door, `userdel -r tashkil`. Neither touches the cluster, the store,
or any running job.
```
