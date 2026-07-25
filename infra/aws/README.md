# Joining an EC2 instance as an Outpost CPU worker

An EC2 box is paired the same way any pre-provisioned host is: the control plane
never boots or destroys it, it just runs containers on it over `ssh + apptainer`
(`StaticSshAdapter`). Reconcile, scheduling, locks, egress are all unchanged.

## 1. SSH key — generate locally, import the public half

Generate on the **control-plane host** (the machine that will ssh into the
instance). The private key must live here; AWS only ever needs the public half.

```bash
ssh-keygen -t ed25519 -f ~/.ssh/outpost-ec2 -C outpost-ec2 -N ''
aws ec2 import-key-pair --key-name outpost-ec2 \
    --public-key-material fileb://~/.ssh/outpost-ec2.pub
```

Prefer this over "create key pair" in the console: that generates the private
key server-side and makes you download it — you'd have to move it to the control
plane anyway, and AWS briefly held your private key. With import, it never
leaves this host. Put the key name (`outpost-ec2`) in `KeyName` in
`launch-template.yaml`, and the private path (`~/.ssh/outpost-ec2`) in the node
manifest's `ssh_key`.

## 2. Network interface — no *second* NIC, just configure the primary one

You do **not** add an extra network interface (m5.xlarge supports one card —
that's what the console's "does not support multiple network cards" note means).
But the primary interface (device index 0) is where SSH reachability is decided,
so it's configured in `launch-template.yaml`:

- `AssociatePublicIpAddress: true` — so you get a public IP even if the subnet
  doesn't auto-assign one (required to ssh in from outside AWS).
- `Groups: [sg-...]` — a security group allowing **inbound tcp/22 from the
  control plane's public IP**.
- `SubnetId` — a subnet in a VPC with an internet gateway.

Note: because that block carries `Groups`, do **not** also set a top-level
`SecurityGroupIds` — the two conflict and the launch fails. (If your control
plane reaches AWS over a Tailscale/WireGuard tunnel instead, you can skip the
public IP and just allow 22 from the tunnel address.)

**The default SG (`sg-096de9e999b9d459d`) does NOT allow SSH.** Its only inbound
rule permits traffic from itself — nothing from the internet. Add a tcp/22 rule
for the control plane's public IP before launching, or you'll get connection
timeouts:

```bash
aws ec2 authorize-security-group-ingress --region eu-north-1 \
    --group-id sg-096de9e999b9d459d \
    --protocol tcp --port 22 --cidr <CONTROL_PLANE_IP>/32
```

(The control plane's current public IP looked like `154.192.10.43` — confirm
with `curl -s https://checkip.amazonaws.com` on that host, and widen the CIDR or
re-run if it's dynamic.)

## 3. Create the launch template + launch

Fill the four `<PLACEHOLDER>`s in `launch-template.yaml` (AMI, KeyName,
SubnetId, security group), then:

```bash
# optional: bake apptainer in on first boot
#   edit launch-template.yaml, set  UserData: $(base64 -w0 infra/aws/user-data.sh)
aws ec2 create-launch-template --cli-input-yaml file://infra/aws/launch-template.yaml
aws ec2 run-instances --launch-template LaunchTemplateName=gtl-t0-cpu-preflight \
    --query 'Instances[0].InstanceId' --output text
# grab its public IP once running:
aws ec2 describe-instances --instance-ids <id> \
    --query 'Reservations[0].Instances[0].PublicIpAddress' --output text
```

The AMI is a **plain Ubuntu 22.04 LTS**, not the SQL-Server image from the old
`gtl-t0-cpu-preflight` template — that one bills a SQL Server Standard license
this worker never uses.

## 4. Join it to Outpost

Put the public IP into a node manifest (copy `node.ec2.example.yaml`) and:

```bash
cluster add-node --spec node.ec2.example.yaml
cluster nodes                         # confirm it shows provider `static-ssh`, state available
```

apptainer must already be on the instance (user-data above, or a custom AMI).
`StaticSshAdapter.bootstrap` verifies reachability + apptainer and fails closed
otherwise, so a job never dies half-way for a missing runtime.

Then submit a `gpu: false, launcher: single` job and
`cluster reconcile --execute` — it'll claim the EC2 node like any VM.
