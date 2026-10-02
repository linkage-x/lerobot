#!/usr/bin/env bash
# Prepare the separate FCI owner. Starting this listener never moves the robot.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
arm_remote="hph@192.168.100.155"
arm_dir="/home/hph/Code/lerobot"
thor_remote="nvidia@192.168.111.122"
thor_dir="/home/nvidia/lerobot"
token_file="$repo_root/outputs/.fr3_bridge_token"

bash run/sync_to_target.sh workstation
mkdir -p outputs
if [[ ! -s "$token_file" ]]; then
  (umask 077; python3 -c 'import secrets; print(secrets.token_hex(32))' > "$token_file")
fi
for destination in "$arm_remote:$arm_dir" "$thor_remote:$thor_dir"; do
  peer="${destination%%:*}"
  directory="${destination#*:}"
  ssh -o ConnectTimeout=5 "$peer" "mkdir -p '$directory/outputs'"
  rsync -a --chmod=F600 "$token_file" "$peer:$directory/outputs/.fr3_bridge_token"
done

ssh -o ConnectTimeout=5 "$arm_remote" "bash -s -- '$arm_dir'" <<'REMOTE'
set -euo pipefail
cd "$1"
python_bin=.venv-fr3/bin/python
if [[ ! -x "$python_bin" ]]; then
  echo "ERROR: FR3 workstation needs .venv-fr3 (see docs/thor_fr3_teleoperation.md)" >&2
  exit 1
fi
export PYTHONPATH="src:.:${PYTHONPATH:-}"
export PYTHONUNBUFFERED=1
native_libs="$($python_bin -c 'import site; from pathlib import Path; print(":".join(str(Path(p)/"cmeel.prefix/lib") for p in site.getsitepackages()))')"
export LD_LIBRARY_PATH="$native_libs:${LD_LIBRARY_PATH:-}"
# No robot connection or communication_test motion is performed by this check.
"$python_bin" -m tools.fr3.box_arm_server --check
mkdir -p outputs/logs
pid_file=outputs/fr3_box_arm_server.pid
if [[ -f "$pid_file" ]]; then
  previous_pid="$(cat "$pid_file")"
  if [[ "$previous_pid" =~ ^[0-9]+$ ]] && [[ -r "/proc/$previous_pid/cmdline" ]] \
     && tr '\0' ' ' < "/proc/$previous_pid/cmdline" | grep -q 'tools.fr3.box_arm_server'; then
    kill "$previous_pid"
    for attempt in {1..50}; do
      kill -0 "$previous_pid" 2>/dev/null || break
      sleep 0.1
    done
    if kill -0 "$previous_pid" 2>/dev/null; then
      echo "ERROR: previous FR3 bridge did not stop; inspect it before deploying" >&2
      exit 1
    fi
  fi
fi
nohup "$python_bin" -m tools.fr3.box_arm_server > outputs/logs/fr3_box_arm_server.log 2>&1 < /dev/null &
bridge_pid=$!
printf '%s\n' "$bridge_pid" > "$pid_file"
sleep 0.5
if ! kill -0 "$bridge_pid" 2>/dev/null; then
  tail -30 outputs/logs/fr3_box_arm_server.log >&2
  exit 1
fi
REMOTE
