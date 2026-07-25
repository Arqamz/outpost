#!/bin/bash
# EC2 first-boot user-data: install apptainer so the instance is ready to be
# joined as an Outpost static CPU worker. Pinned to the SAME version + checksum
# the VMs use (infra/ansible/roles/bootstrap/defaults/main.yml) — upstream's own
# .deb, deps resolved via apt, no PPA. Ubuntu's archive has no `apptainer`
# (only the unrelated `singularity-container` fork), hence the direct release.
#
# ssh access itself comes from the launch template's KeyName (authorizes the
# `ubuntu` user), so there's nothing to do here for keys.
set -euxo pipefail

APPTAINER_VERSION="1.5.0"
APPTAINER_DEB_SHA256="fbc27204d0ec0440dfa0ae589089e4b2baf192315b4ac22dfae02b78b28981ea"
DEB="/tmp/apptainer_${APPTAINER_VERSION}_amd64.deb"
URL="https://github.com/apptainer/apptainer/releases/download/v${APPTAINER_VERSION}/apptainer_${APPTAINER_VERSION}_amd64.deb"

export DEBIAN_FRONTEND=noninteractive
# wait out any first-boot apt lock, then install
for _ in $(seq 1 30); do
  fuser /var/lib/dpkg/lock-frontend >/dev/null 2>&1 || break
  sleep 5
done

curl -fsSL "$URL" -o "$DEB"
echo "${APPTAINER_DEB_SHA256}  ${DEB}" | sha256sum -c -
apt-get update -y
apt-get install -y "$DEB"          # apt resolves the .deb's deps

apptainer --version                # smoke: surfaces in the instance's cloud-init log
