#!/usr/bin/env bash
# Incremental deploy: replace changed files, restart the selected gateway, then
# start the local frontend against that gateway.
#
# Usage:
#   bash run/deploy.sh                         # defaults to Thor
#   bash run/deploy.sh thor
#   bash run/deploy.sh workstation
#   bash run/deploy.sh workstation --sync-only
#   bash run/deploy.sh thor --no-frontend
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

target="thor"
target_explicit=false
sync_only=false
no_frontend=false

usage() {
  sed -n '2,10p' "$0" | sed 's/^# \{0,1\}//'
}

for arg in "$@"; do
  case "$arg" in
    thor|workstation)
      if $target_explicit; then
        echo "ERROR: specify exactly one deployment target" >&2
        exit 2
      fi
      target="$arg"
      target_explicit=true
      ;;
    --sync-only) sync_only=true ;;
    --no-frontend) no_frontend=true ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "ERROR: unknown argument '$arg'" >&2
      usage >&2
      exit 2
      ;;
  esac
done

case "$target" in
  thor)
    remote="nvidia@192.168.111.122"
    remote_dir="/home/nvidia/lerobot"
    profile="thor"
    config_path="tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml"
    gateway_target="http://192.168.111.122:8765"
    ;;
  workstation)
    remote="hph@192.168.100.155"
    remote_dir="/home/hph/Code/lerobot"
    profile="workstation"
    config_path="tools/fr3/fr3_record_config.yaml"
    gateway_target="http://192.168.100.155:8765"
    ;;
esac

# Always complete incremental replacement before touching a running service.
bash "$script_dir/sync_to_target.sh" "$target"

if [[ "$target" == "thor" ]]; then
  calibration_paths=(
    "outputs/calibration/thor_gmsl2_intrinisics_dict_0720"
    "outputs/calibration/thor_gmsl2_extrinisics_robot_base_0720"
  )
  require_ee_calibration=false
  case "${REQUIRE_EE_CALIBRATION:-}" in
    1|true|TRUE|yes|YES) require_ee_calibration=true ;;
  esac

  calibration_sync_incomplete=false
  warn_or_fail_calibration_sync() {
    local message="$1"
    if $require_ee_calibration; then
      echo "ERROR: ${message}" >&2
      exit 1
    fi
    calibration_sync_incomplete=true
    echo "WARN: ${message}" >&2
  }

  echo "==> Syncing Thor EE-trajectory calibration inputs..."
  for calibration_path in "${calibration_paths[@]}"; do
    local_calibration_dir="${repo_root}/${calibration_path}"
    if [[ ! -f "${local_calibration_dir}/summary.json" ]]; then
      warn_or_fail_calibration_sync "calibration summary not found: ${local_calibration_dir}/summary.json"
      continue
    fi
    if ! ssh -o ConnectTimeout=5 "$remote" "mkdir -p '${remote_dir}/${calibration_path}'"; then
      warn_or_fail_calibration_sync "failed to create ${remote_dir}/${calibration_path}"
      continue
    fi
    if ! rsync -avz --itemize-changes "${local_calibration_dir}/" "${remote}:${remote_dir}/${calibration_path}/"; then
      warn_or_fail_calibration_sync "failed to sync ${local_calibration_dir}"
    fi
  done
  if $calibration_sync_incomplete; then
    echo "WARN: EE-trajectory calibration inputs are incomplete; deployment will continue." >&2
  fi
fi

if $sync_only; then
  echo "==> ${target} incremental sync complete (--sync-only)."
  exit 0
fi

echo "==> Restarting ${profile} gateway on ${remote}..."
ssh -o ConnectTimeout=5 "$remote" \
  "flock -n -o /tmp/lerobot_gateway_deploy.lock bash -s -- '$remote_dir' '$profile' '$config_path'" \
  <<'REMOTE'
set -euo pipefail

repo_dir="$1"
profile="$2"
config_path="$3"

matching_pids() {
  local pattern="$1"
  python3 - "$pattern" <<'PY'
import os
import sys

pattern = sys.argv[1]
for name in os.listdir("/proc"):
    if not name.isdigit():
        continue
    if int(name) == os.getpid():
        continue
    try:
        args = [
            item.decode("utf-8", "ignore")
            for item in open(f"/proc/{name}/cmdline", "rb").read().split(b"\0")
            if item
        ]
    except OSError:
        continue
    if pattern in " ".join(args):
        print(name)
PY
}

stop_matching() {
  local pattern="$1"
  local pids
  pids="$(matching_pids "$pattern" || true)"
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs -r kill 2>/dev/null || true
    sleep 1
  fi
  pids="$(matching_pids "$pattern" || true)"
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs -r kill -9 2>/dev/null || true
  fi
}

stop_matching "tools.data_collection_gui.gateway"
if [[ "$profile" == "thor" ]]; then
  stop_matching "tools/thor/gmsl2/thor_record.py"
else
  stop_matching "tools/fr3/fr3_mujoco_teleop.py"
fi

cd "$repo_dir"
if [[ "$profile" == "thor" ]]; then
  bash tools/thor/box_sdk/ensure_box_net.sh >/dev/null 2>&1 || true
  . tools/thor/box_sdk/setup_env.sh
fi

python_bin="python3"
if [[ "$profile" == "workstation" && -x .venv-fr3/bin/python ]]; then
  python_bin=".venv-fr3/bin/python"
elif [[ -x .venv/bin/python ]]; then
  python_bin=".venv/bin/python"
fi

if [[ "$profile" == "workstation" && "$python_bin" == "python3" ]]; then
  echo "ERROR: workstation environment is missing: ${repo_dir}/.venv-fr3" >&2
  echo "Run: bash tools/fr3/setup_workstation_teleop_env.sh" >&2
  exit 1
fi

display="${DISPLAY:-}"
xauthority="${XAUTHORITY:-}"
xdg_runtime_dir="${XDG_RUNTIME_DIR:-}"
if [[ "$profile" == "workstation" ]]; then
  if [[ -z "$display" ]]; then
    display="$(who | awk -v user="$(id -un)" '$1 == user && $2 ~ /^:/ { print $2; exit }')"
  fi
  if [[ -z "$xdg_runtime_dir" ]]; then
    xdg_runtime_dir="/run/user/$(id -u)"
  fi
  if [[ -z "$xauthority" && -r "$xdg_runtime_dir/gdm/Xauthority" ]]; then
    xauthority="$xdg_runtime_dir/gdm/Xauthority"
  fi
  if [[ -z "$display" ]]; then
    echo "ERROR: no active workstation X display found for MuJoCo viewer" >&2
    exit 1
  fi
fi

mkdir -p run/logs
setsid env \
  PYTHONPATH=src:. \
  PYTHONUNBUFFERED=1 \
  DISPLAY="$display" \
  XAUTHORITY="$xauthority" \
  XDG_RUNTIME_DIR="$xdg_runtime_dir" \
  "$python_bin" -m tools.data_collection_gui.gateway \
    --profile "$profile" \
    --config-path "$config_path" \
    --datasets-root outputs/datasets \
    --port 8765 \
    --host 0.0.0.0 \
    --repo-root "$repo_dir" \
  </dev/null >run/logs/gateway.log 2>&1 &
disown

sleep 3
new_pids="$(matching_pids "tools.data_collection_gui.gateway" || true)"
if [[ -z "$new_pids" ]]; then
  echo "ERROR: ${profile} gateway failed to start" >&2
  tail -30 run/logs/gateway.log
  exit 1
fi
echo "${profile} gateway started (pid $(echo "$new_pids" | head -1))"
tail -5 run/logs/gateway.log
REMOTE

if $no_frontend; then
  echo "==> ${target} gateway ready at ${gateway_target} (--no-frontend)."
  exit 0
fi

echo "==> Starting local frontend for ${target}..."
cd "$repo_root/tools/data_collection_gui/frontend"

if ! command -v npm >/dev/null 2>&1; then
  export NVM_DIR="${NVM_DIR:-$HOME/.nvm}"
  [[ -s "$NVM_DIR/nvm.sh" ]] && . "$NVM_DIR/nvm.sh"
fi

if [[ ! -x node_modules/.bin/vite ]]; then
  echo "    frontend dependencies missing; restoring from lockfile..."
  if [[ -f package-lock.json ]]; then
    npm ci --no-audit
  else
    npm install --no-audit
  fi
fi

echo "    http://localhost:5173/ -> ${gateway_target}"
export GUI_API_TARGET="$gateway_target"
exec npm run dev -- --host 0.0.0.0 --port 5173
