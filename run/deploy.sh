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
# trajectory job can use these calibration folders when available, so stage
# them explicitly. Missing calibration assets should not block gateway/frontend
# deploys; set REQUIRE_EE_CALIBRATION=1 to make this section strict.
echo "==> Syncing EE-trajectory calibration inputs to ${THOR}..."
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

for calibration_path in "${CALIBRATION_PATHS[@]}"; do
  local_calibration_dir="${REPO_ROOT}/${calibration_path}"
  if [[ ! -f "${local_calibration_dir}/summary.json" ]]; then
    warn_or_fail_calibration_sync "calibration summary not found: ${local_calibration_dir}/summary.json"
    continue
  fi
  remote_calibration_dir="${THOR_DIR}/${calibration_path}"
  remote_mkdir_path="${remote_calibration_dir}"
  [[ "${remote_mkdir_path}" == "~/"* ]] && remote_mkdir_path="\$HOME/${remote_mkdir_path#\~/}"
  if ! ssh -o ConnectTimeout=5 "${THOR}" "mkdir -p \"${remote_mkdir_path}\""; then
    warn_or_fail_calibration_sync "failed to create remote calibration directory: ${THOR}:${THOR_DIR}/${calibration_path}"
    continue
  fi
  if ! rsync -avz "${local_calibration_dir}/" "${THOR}:${THOR_DIR}/${calibration_path}/"; then
    warn_or_fail_calibration_sync "failed to sync calibration directory: ${local_calibration_dir}"
    continue
  fi
done
if $calibration_sync_incomplete; then
  echo "WARN: EE-trajectory calibration inputs were not fully synced; continuing deploy. EE trajectory features may be unavailable until calibration outputs are generated or synced." >&2
fi

if $sync_only; then
  exit 0
fi

# ---- 2. Restart gateway on Thor ----
echo "==> Restarting gateway on Thor..."
# flock -n reports a lock conflict with -E's exit code, so a failure inside the
# remote script stays distinguishable from "someone else is deploying".
restart_rc=0
ssh -o ConnectTimeout=5 "$THOR" 'flock -n -E 75 /tmp/lerobot_gateway_deploy.lock bash -s' <<'REMOTE' || restart_rc=$?
set -e

GATEWAY_LOG_DIR=~/lerobot/outputs/logs/data_collection_gui

_gateway_pids() {
  python3 - <<'PY'
import os

MODULE = 'tools.data_collection_gui.gateway'
for name in os.listdir('/proc'):
    if not name.isdigit():
        continue
    try:
        raw = open(f'/proc/{name}/cmdline', 'rb').read().split(b'\0')
    except OSError:
        continue
    args = [x.decode('utf-8', 'ignore') for x in raw if x]
    # Match on the interpreter + `-m <module>` pair rather than on argv[0]
    # being exactly python3: other branches launch the gateway through
    # .venv/bin/python (a symlink to python3) and may pass flags before -m.
    # A plain substring test would instead match greps, editors and pkill.
    if not args or not os.path.basename(args[0]).startswith('python'):
        continue
    if any(flag == '-m' and mod == MODULE for flag, mod in zip(args, args[1:])):
        print(name)
PY
}

# gateway.py redirects stdout/stderr into its own gateway_<ts>_<pid>.log inside
# main(), so run/logs/gateway.log only ever captures pre-main crashes. Pick the
# newest log this restart created, keyed off the timestamp in the file name:
# mtime is useless here because a gateway that survived the kill above keeps
# appending to its own log and would always look like the freshest file.
_gateway_log_since() {
  local since="$1" newest="" path name stamp
  for path in "$GATEWAY_LOG_DIR"/gateway_*.log; do
    [[ -e "$path" ]] || continue
    name="${path##*/}"
    name="${name#gateway_}"
    stamp="${name%_*}"
    [[ "$stamp" < "$since" ]] || newest="$path"
  done
  printf '%s' "$newest"
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
spawn_ts="$(date +%Y%m%d_%H%M%S)"
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
started_log="$(_gateway_log_since "$spawn_ts" || true)"
if [[ -n "$new_pids" ]]; then
  echo "gateway started (pid $(echo "$new_pids" | head -1))"
  if [[ -n "$started_log" ]]; then
    echo "--- ${started_log}"
    tail -3 "$started_log"
  fi
else
  echo "ERROR: gateway failed to start" >&2
  # Failures before _setup_gateway_log() (bad interpreter, import errors) are
  # all run/logs/gateway.log ever sees; everything later is in the gateway log.
  if [[ -s ~/lerobot/run/logs/gateway.log ]]; then
    echo "--- run/logs/gateway.log"
    tail -20 ~/lerobot/run/logs/gateway.log
  fi
  if [[ -n "$started_log" ]]; then
    echo "--- ${started_log}"
    tail -30 "$started_log"
  else
    echo "(no new ${GATEWAY_LOG_DIR}/gateway_*.log was created)"
  fi
  exit 1
fi
REMOTE

if [[ $restart_rc -eq 75 ]]; then
  echo "ERROR: another deploy is already restarting the Thor gateway" >&2
  exit 75
elif [[ $restart_rc -ne 0 ]]; then
  echo "ERROR: Thor gateway restart failed (exit ${restart_rc})" >&2
  exit "$restart_rc"
fi

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

if [ ! -x node_modules/.bin/vite ]; then
  echo "    frontend dependencies missing or incomplete; restoring from lockfile..."
  if [ -f package-lock.json ]; then
    npm ci --no-audit
  else
    npm install --no-audit
  fi
fi

echo "    http://localhost:5173/ -> gateway@192.168.111.122:8765"
exec npm run dev -- --host 0.0.0.0 --port 5173
