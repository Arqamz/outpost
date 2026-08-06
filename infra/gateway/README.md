# infra/gateway — the `ssh tashkil` job portal

A forced-command ssh front door that exposes Outpost's open interface (JobSpec
in, artifacts out) as `ssh tashkil …`, without giving the key holder a shell.
Full walkthrough: [`../../docs/10-ssh-gateway.md`](../../docs/10-ssh-gateway.md).

## Files

| File | Runs on | What it is |
|------|---------|------------|
| `tashkil-gw` | control plane | the forced command pinned in the gateway user's `authorized_keys`. Word-splits `$SSH_ORIGINAL_COMMAND`, whitelists the subcommand, validates args, `exec`s `cluster`. Job portal only — no infra mutation. |
| `install.sh` | control plane (root) | idempotent installer: dedicated shell-less `tashkil` user, forced-command `authorized_keys`, shared `outpost` group + group-writable `.var/`. |
| `../../bin/tashkil` | **client** | optional sugar over `ssh tashkil …` (wires stdin for `submit`, untars `fetch`, adds `run`). |

## Quick start

```bash
# control plane, as root:
sudo infra/gateway/install.sh --key ~/keys/laptop.pub   # (or make gateway-install KEY=…)
umask 002 && cluster reconcile --execute                # daemon, group-writable store

# client ~/.ssh/config:
#   Host tashkil
#       HostName <control-plane-ip>
#       User tashkil
#       IdentityFile ~/.ssh/tashkil
ssh tashkil help
```

## Exposed vs denied

Exposed: `submit` (stdin), `fetch` (stdout tar), `status`, `result`, `list`,
`nodes`, `logs [--tail N]`, `reconciler-log [--tail N]`.
Denied (operator-only, local): `reconcile`, `seed-nodes`, `add-node`,
`remove-node`, `fail-node`, `clear-jobs`.

Revoke one client: remove its line from `~tashkil/.ssh/authorized_keys`.
