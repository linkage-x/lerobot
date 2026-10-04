#!/usr/bin/env bash
# Explicit one-time host scheduling permission. Does not move/connect FR3.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
account="$(id -un)"
if [[ "$account" == root ]]; then
  echo "Run as the normal host operator; this script invokes sudo itself." >&2
  exit 1
fi
if [[ "${1:-}" == --performance ]]; then
  # Explicit profile tuning, reapplied after reboot by start_host_fr3.sh.
  sudo -n sh -c 'for governor in /sys/devices/system/cpu/cpufreq/policy*/scaling_governor; do
    [ -f "$governor" ] || exit 1
    echo performance > "$governor" || exit 1
  done'
  echo "Host CPU governors set to performance for FR3 timing."
  exit 0
fi
if [[ "${1:-}" != --install ]]; then
  echo "Current shell FIFO ceiling: $(ulimit -r)"
  echo "To permit this account's native FCI thread: bash run/setup_host_fr3_permissions.sh --install"
  echo "To disable frequency scaling for FR3: bash run/setup_host_fr3_permissions.sh --performance"
  exit 0
fi
# This permits priority requests; it does not run Python or sensor capture FIFO.
printf '%s soft rtprio 99\n%s hard rtprio 99\n' "$account" "$account" | \
  sudo tee /etc/security/limits.d/90-lerobot-host-fr3.conf >/dev/null
sudo chmod 644 /etc/security/limits.d/90-lerobot-host-fr3.conf
mkdir -p outputs/secrets
chmod 700 outputs/secrets
# Opt-in marker for refreshing only this repository's newly spawned host service
# from an older desktop login. No sudoers rule or ambient capability is added.
touch outputs/secrets/host_fifo_approved
chmod 600 outputs/secrets/host_fifo_approved
echo "FIFO ceiling configured for future logins. deploy.sh can refresh its host service using sudo -n prlimit."
