#!/usr/bin/env bash
# One-shot: sync code → restart Thor gateway → start local frontend.
#
# Usage:
#   bash run/deploy.sh              # full deploy
#   bash run/deploy.sh --sync-only  # only sync, skip restart & frontend
#   bash run/deploy.sh --no-frontend # sync + restart, skip frontend
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
THOR="nvidia@192.168.111.122"
THOR_DIR="~/lerobot"
GATEWAY_LOG="/tmp/gateway.log"
CALIBRATION_PATHS=(
  "outputs/calibration/thor_gmsl2_intrinisics_dict_0720"
  "outputs/calibration/thor_gmsl2_extrinisics_robot_base_0720"
)

sync_only=false
no_frontend=false
for arg in "$@"; do
  case "$arg" in
    --sync-only)  sync_only=true ;;
    --no-frontend) no_frontend=true ;;
  esac
done

# ---- 1. Sync ----
echo "==> Syncing to ${THOR}..."
bash "$SCRIPT_DIR/sync_to_thor.sh"

# The general repo sync deliberately excludes outputs/. The gateway's EE
# trajectory job needs these exact calibration folders, so stage them
# explicitly before either --sync-only exits or the gateway restarts.
echo "==> Syncing EE-trajectory calibration inputs to ${THOR}..."
for calibration_path in "${CALIBRATION_PATHS[@]}"; do
  local_calibration_dir="${REPO_ROOT}/${calibration_path}"
  if [[ ! -f "${local_calibration_dir}/summary.json" ]]; then
    echo "ERROR: calibration summary not found: ${local_calibration_dir}/summary.json" >&2
    exit 1
  fi
  remote_calibration_dir="${THOR_DIR}/${calibration_path}"
  remote_mkdir_path="${remote_calibration_dir}"
  [[ "${remote_mkdir_path}" == "~/"* ]] && remote_mkdir_path="\$HOME/${remote_mkdir_path#\~/}"
  ssh -o ConnectTimeout=5 "${THOR}" "mkdir -p \"${remote_mkdir_path}\""
  rsync -avz "${local_calibration_dir}/" "${THOR}:${THOR_DIR}/${calibration_path}/"
done

if $sync_only; then
  exit 0
fi

# ---- 2. Restart gateway on Thor ----
echo "==> Restarting gateway on Thor..."
ssh -o ConnectTimeout=5 "$THOR" 'flock -n /tmp/lerobot_gateway_deploy.lock bash -s || { echo "ERROR: another deploy is already restarting the Thor gateway" >&2; exit 75; }' <<'REMOTE'
set -e

_gateway_pids() {
  python3 - <<'PY'
import os
for name in os.listdir('/proc'):
    if not name.isdigit():
        continue
    try:
        raw = open(f'/proc/{name}/cmdline', 'rb').read().split(b'\0')
    except OSError:
        continue
    args = [x.decode('utf-8', 'ignore') for x in raw if x]
    if len(args) >= 3 and args[0].endswith('python3') and args[1] == '-m' and args[2] == 'tools.data_collection_gui.gateway':
        print(name)
PY
}

old_pids="$(_gateway_pids || true)"
if [[ -n "$old_pids" ]]; then
  echo "$old_pids" | xargs -r kill 2>/dev/null || true
  sleep 1
fi
left_pids="$(_gateway_pids || true)"
if [[ -n "$left_pids" ]]; then
  echo "$left_pids" | xargs -r kill -9 2>/dev/null || true
  sleep 1
fi

# The gateway spawns the recorder (thor_record.py) as a child; killing the
# gateway can orphan it, and it holds box_client.py + the box/camera (Argus)
# sessions in memory. Stop it too so the next Connect respawns a fresh recorder
# with the just-synced code instead of a stale orphan clashing over the hardware.
_recorder_pids() {
  python3 - <<'PY'
import os
for name in os.listdir('/proc'):
    if not name.isdigit():
        continue
    try:
        raw = open(f'/proc/{name}/cmdline', 'rb').read().split(b'\0')
    except OSError:
        continue
    args = [x.decode('utf-8', 'ignore') for x in raw if x]
    if any(a.endswith('thor_record.py') for a in args):
        print(name)
PY
}
rec_pids="$(_recorder_pids || true)"
if [[ -n "$rec_pids" ]]; then
  echo "$rec_pids" | xargs -r kill 2>/dev/null || true
  sleep 1
  echo "$(_recorder_pids || true)" | xargs -r kill -9 2>/dev/null || true
fi

cd ~/lerobot
bash tools/thor/box_sdk/ensure_box_net.sh >/dev/null 2>&1 || true
. tools/thor/box_sdk/setup_env.sh

mkdir -p ~/lerobot/run/logs
setsid bash -c '''exec 3>&-; exec env PYTHONPATH=src:. PYTHONUNBUFFERED=1 \
  python3 -m tools.data_collection_gui.gateway \
  --config-path tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml \
  --datasets-root outputs/datasets \
  --port 8765 --host 0.0.0.0 \
  --repo-root /home/nvidia/lerobot''' \
  </dev/null >~/lerobot/run/logs/gateway.log 2>&1 &
disown

sleep 3
new_pids="$(_gateway_pids || true)"
if [[ -n "$new_pids" ]]; then
  echo "gateway started (pid $(echo "$new_pids" | head -1))"
  tail -3 ~/lerobot/run/logs/gateway.log
else
  echo "ERROR: gateway failed to start"
  tail -20 ~/lerobot/run/logs/gateway.log
  exit 1
fi
REMOTE

if $no_frontend; then
  echo "==> Done (--no-frontend). Open http://192.168.111.122:5173/ from a browser."
  exit 0
fi

# ---- 3. Start local frontend ----
echo "==> Starting local frontend (vite)..."
cd "$REPO_ROOT/tools/data_collection_gui/frontend"

if ! command -v npm &>/dev/null; then
  export NVM_DIR="$HOME/.nvm"
  [ -s "$NVM_DIR/nvm.sh" ] && . "$NVM_DIR/nvm.sh"
fi

if [ ! -d node_modules ]; then
  echo "    npm install..."
  npm install --silent
fi

echo "    http://localhost:5173/ -> gateway@192.168.111.122:8765"
exec npm run dev -- --host 0.0.0.0 --port 5173
