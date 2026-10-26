#!/usr/bin/env bash
# Start a training run on this GPU machine as a systemd user unit (run/user_job.sh) and have the
# data-collection GUI follow it, so the Training page shows the run that is actually going.
#
# Usage:
#   bash run/train_job.sh start <name> [user_job.sh start options] -- <trainer cmd...>
#   bash run/train_job.sh announce          # (re)attach the page to the trainer running here
#
# The gateway (on the robot host) only knew runs it had launched itself, so a run started here
# left the page showing an old one. `announce` is the Training page's "Re-attach" button from
# this side: POST /api/training/reattach, and the gateway asks this machine over ssh which
# trainer is running in its repo, writes that trainer's pid file (Stop signals through it) and
# relays its log. Use either after anything that made the page lose the run (a network drop
# ends the relay, and the page then reports the run as ended).
set -euo pipefail

repo_root="$(cd "$(dirname "$0")/.." && pwd)"
gateway="${GUI_GATEWAY:-http://192.168.1.156:8765}"

usage() { sed -n '5,7p' "$0" >&2; exit 2; }

announce() {
  python3 - "$gateway" <<'EOF'
import json, subprocess, sys, urllib.request, urllib.error
gateway = sys.argv[1]
body = {}
try:
    # One machine can be registered more than once (Dante: wired and Wi-Fi); ask through the
    # entry whose ssh target is an address this machine has right now.
    with urllib.request.urlopen(f"{gateway}/api/training/hosts", timeout=10) as response:
        hosts = json.load(response).get("hosts", [])
    mine = set(subprocess.run(["hostname", "-I"], capture_output=True, text=True).stdout.split())
    ours = [h for h in hosts if h.get("kind") == "remote" and h.get("sshTarget", "").rsplit("@", 1)[-1] in mine]
    if len(ours) == 1:
        body["hostId"] = ours[0]["id"]
except OSError:
    pass  # the call below reports an unreachable gateway
request = urllib.request.Request(f"{gateway}/api/training/reattach", data=json.dumps(body).encode(),
                                 headers={"Content-Type": "application/json"}, method="POST")
try:
    with urllib.request.urlopen(request, timeout=60) as response:
        result = json.load(response)
except urllib.error.HTTPError as exc:
    result = json.load(exc)
except OSError as exc:
    sys.exit(f"gateway {gateway} unreachable: {exc}")
if not result.get("ok"):
    sys.exit(f"gateway refused: {result.get('error')}")
print(result.get("message") or result["training"]["message"])
EOF
}

action="${1:-}"
case "$action" in
  announce) announce ;;
  start)
    name="${2:-}"
    [[ -n "$name" ]] || usage
    shift 2
    bash "$repo_root/run/user_job.sh" start "$name" "$@"
    # The trainer must be running before the gateway can find it; MainPID is set as soon as it forks.
    for _ in $(seq 20); do
      [[ "$(systemctl --user show "job-$name" -p MainPID --value)" != 0 ]] && break
      sleep 0.5
    done
    # The run is up either way; a page that cannot be told is a display problem, not a reason
    # to fail the start.
    announce || echo "warning: the Training page was not told; run: bash $0 announce (or press Re-attach)" >&2
    ;;
  *) usage ;;
esac
