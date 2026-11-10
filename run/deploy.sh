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
#   bash run/deploy.sh thor --sandbox NAME     # your own sandbox: own directory + port, no hardware
#
# Production only accepts a clean checkout whose HEAD is on origin/box (see run/sync_to_target.sh);
# a maintainer adds --maintainer to deploy a clean commit that is not merged yet. Students deploy
# to their sandbox; run/sandbox.sh copies datasets into it and shows/stops/removes it.
set -euo pipefail

script_dir="$(cd "$(dirname "$0")" && pwd)"
repo_root="$(cd "$script_dir/.." && pwd)"

target="thor"
target_explicit=false
sync_only=false
no_frontend=false

usage() {
  sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'
}

# shellcheck source=run/sandbox_common.sh
. "$script_dir/sandbox_common.sh"

sandbox=""
maintainer=false
expect_sandbox_name=false
for arg in "$@"; do
  if $expect_sandbox_name; then
    sandbox="$arg"
    expect_sandbox_name=false
    continue
  fi
  case "$arg" in
    --sandbox) expect_sandbox_name=true ;;
    --sandbox=*) sandbox="${arg#--sandbox=}" ;;
    --maintainer) maintainer=true ;;
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
if $expect_sandbox_name; then
  echo "ERROR: --sandbox needs a name" >&2
  exit 2
fi

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

sync_flags=()
gateway_port=8765
if [[ -n "$sandbox" ]]; then
  if [[ "$target" != "thor" ]]; then
    echo "ERROR: sandboxes exist only on Thor" >&2
    exit 2
  fi
  if $maintainer; then
    echo "ERROR: --maintainer is for production deploys; a sandbox takes any tree" >&2
    exit 2
  fi
  validate_sandbox_name "$sandbox"
  remote_dir="$(sandbox_dir "$sandbox")"
  sync_flags+=(--sandbox "$sandbox")
elif $maintainer; then
  sync_flags+=(--maintainer)
fi

# Always complete incremental replacement before touching a running service.
bash "$script_dir/sync_to_target.sh" "$target" ${sync_flags[@]+"${sync_flags[@]}"}

# A sandbox starts from production's calibration (copied by the remote script below), never from
# this checkout's outputs/.
if [[ "$target" == "thor" && -z "$sandbox" ]]; then
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
# flock -n reports a lock conflict with -E's exit code, so a failure inside the
# remote script stays distinguishable from "someone else is deploying".
if [[ -n "$sandbox" ]]; then
  deploy_lock="/tmp/lerobot_gateway_deploy.sandbox_${sandbox}.lock"
else
  deploy_lock="/tmp/lerobot_gateway_deploy.lock"
fi
restart_log="$(mktemp)"
trap 'rm -f "$restart_log"' EXIT
restart_rc=0
ssh -o ConnectTimeout=5 "$remote" \
  "flock -n -o -E 75 '$deploy_lock' bash -s -- '$remote_dir' '$profile' '$config_path' '$sandbox' '$THOR_PRODUCTION_DIR' '$SANDBOX_PORT_MIN' '$SANDBOX_PORT_MAX' '$THOR_SANDBOX_BASE'" \
  <<'REMOTE' 2>&1 | tee "$restart_log" || restart_rc=${PIPESTATUS[0]}
set -euo pipefail

repo_dir="$1"
profile="$2"
config_path="$3"
sandbox="$4"
production_dir="$5"
port_min="$6"
port_max="$7"
sandbox_base="$8"
gateway_log_dir="$repo_dir/outputs/logs/data_collection_gui"
port=8765

if [[ -n "$sandbox" ]]; then
  # One port per sandbox, kept across deploys in .sandbox_port (excluded from the sync).
  if [[ -s "$repo_dir/.sandbox_port" ]]; then
    port="$(cat "$repo_dir/.sandbox_port")"
  else
    taken="$(cat "$sandbox_base"/*/.sandbox_port 2>/dev/null || true)"
    listening="$(ss -Htln 2>/dev/null | awk '{print $4}' | sed -E 's/.*://' || true)"
    port=""
    for candidate in $(seq "$port_min" "$port_max"); do
      if ! grep -qx "$candidate" <<<"$taken"$'\n'"$listening"; then
        port="$candidate"
        break
      fi
    done
    if [[ -z "$port" ]]; then
      echo "ERROR: no free sandbox port in ${port_min}-${port_max}; remove an unused sandbox (run/sandbox.sh)" >&2
      exit 1
    fi
    echo "$port" >"$repo_dir/.sandbox_port"
  fi
  # What the sync leaves out but the gateway needs: the Python environments (shared, read-only use),
  # Thor-only world files and production's calibration (copied once, so nothing here writes back).
  for venv in .venv .venv-fr3 third_party/opencv_kalibr/.venv; do
    if [[ ! -e "$repo_dir/$venv" && -e "$production_dir/$venv" ]]; then
      ln -s "$(readlink -f "$production_dir/$venv")" "$repo_dir/$venv"
    fi
  done
  for state_file in tools/thor/gmsl2/world/world_registration.json tools/thor/gmsl2/world/world_stable_source.json outputs/box_six_d_origin.json; do
    if [[ ! -e "$repo_dir/$state_file" && -e "$production_dir/$state_file" ]]; then
      mkdir -p "$(dirname "$repo_dir/$state_file")"
      cp -a "$production_dir/$state_file" "$repo_dir/$state_file"
    fi
  done
  mkdir -p "$repo_dir/outputs/datasets" "$repo_dir/outputs/exports"
  if [[ ! -e "$repo_dir/outputs/calibration" ]]; then
    echo "    first deploy: copying production calibration into the sandbox..."
    cp -a "$production_dir/outputs/calibration" "$repo_dir/outputs/calibration"
  fi
fi
echo "GATEWAY_PORT=${port}"

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

# The gateway gets a scan of its own instead of a cmdline substring match: the
# interpreter differs per profile (.venv-fr3/bin/python, .venv/bin/python which
# is a symlink to python3, or plain python3) and flags may precede -m, while a
# substring test would also match greps, editors and pkill on the module name.
#
# Only the gateway of THIS deployment: the one started for this tree or on this port. Thor also
# runs sandbox gateways (other trees, other ports), and a deploy must never take those down -- nor
# a sandbox deploy the production gateway.
gateway_pids() {
  python3 - "$repo_dir" "$port" <<'PY'
import os
import sys

MODULE = "tools.data_collection_gui.gateway"
repo_dir, port = sys.argv[1], sys.argv[2]


def option(args, flag):
    for i, item in enumerate(args):
        if item == flag and i + 1 < len(args):
            return args[i + 1]
        if item.startswith(flag + "="):
            return item.split("=", 1)[1]
    return None


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
    if not args or not os.path.basename(args[0]).startswith("python"):
        continue
    if not any(flag == "-m" and mod == MODULE for flag, mod in zip(args, args[1:])):
        continue
    if option(args, "--repo-root") == repo_dir or option(args, "--port") == port:
        print(name)
PY
}

# Kill whatever a pid-listing command reports: `stop_pids gateway_pids` for the
# gateway, `stop_pids matching_pids <pattern>` for the recorder/teleop scripts.
stop_pids() {
  local pids
  pids="$("$@" || true)"
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs -r kill 2>/dev/null || true
    sleep 1
  fi
  pids="$("$@" || true)"
  if [[ -n "$pids" ]]; then
    echo "$pids" | xargs -r kill -9 2>/dev/null || true
  fi
}

# gateway.py redirects stdout/stderr into its own gateway_<ts>_<pid>.log inside
# main(), so run/logs/gateway.log only ever captures pre-main crashes. Pick the
# newest log this restart created, keyed off the timestamp in the file name:
# mtime is useless here because a gateway that survived the kill above keeps
# appending to its own log and would always look like the freshest file.
gateway_log_since() {
  local since="$1" newest="" path name stamp
  for path in "$gateway_log_dir"/gateway_*.log; do
    [[ -e "$path" ]] || continue
    name="${path##*/}"
    name="${name#gateway_}"
    stamp="${name%_*}"
    [[ "$stamp" < "$since" ]] || newest="$path"
  done
  printf '%s' "$newest"
}

stop_pids gateway_pids
if [[ -n "$sandbox" ]]; then
  : # a sandbox never starts a recorder or teleop process (no hardware), so there is none to stop
# The gateway spawns the recorder (thor_record.py) as a child; killing the
# gateway can orphan it, and it holds box_client.py + the box/camera (Argus)
# sessions in memory. Stop it too so the next Connect respawns a fresh recorder
# with the just-synced code instead of a stale orphan clashing over the hardware.
elif [[ "$profile" == "thor" ]]; then
  stop_pids matching_pids "tools/thor/gmsl2/thor_record.py"
else
  stop_pids matching_pids "tools/fr3/fr3_mujoco_teleop.py"
fi

cd "$repo_dir"
if [[ "$profile" == "thor" ]]; then
  if [[ -z "$sandbox" ]]; then
    bash tools/thor/box_sdk/ensure_box_net.sh >/dev/null 2>&1 || true
  fi
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

# Only export the X variables we actually resolved. Handing Thor an empty
# DISPLAY is worse than handing it none: the recorder's EGL setup then tries
# the X11 platform against "" and every Argus camera fails preflight with
# "Could not get EGL display connection" / NvBufSurfaceMapEglImage failed,
# while an unset DISPLAY takes the headless path and captures fine.
gateway_env=(PYTHONPATH=src:. PYTHONUNBUFFERED=1)
if [[ -n "$display" ]]; then
  gateway_env+=(DISPLAY="$display")
fi
if [[ -n "$xauthority" ]]; then
  gateway_env+=(XAUTHORITY="$xauthority")
fi
if [[ -n "$xdg_runtime_dir" ]]; then
  gateway_env+=(XDG_RUNTIME_DIR="$xdg_runtime_dir")
fi

sandbox_args=()
if [[ -n "$sandbox" ]]; then
  sandbox_args=(--sandbox "$sandbox")
fi

mkdir -p run/logs
spawn_ts="$(date +%Y%m%d_%H%M%S)"
setsid env \
  "${gateway_env[@]}" \
  "$python_bin" -m tools.data_collection_gui.gateway \
    --profile "$profile" \
    --config-path "$config_path" \
    --datasets-root outputs/datasets \
    --port "$port" \
    --host 0.0.0.0 \
    --repo-root "$repo_dir" \
    ${sandbox_args[@]+"${sandbox_args[@]}"} \
  </dev/null >run/logs/gateway.log 2>&1 &
disown

sleep 3
new_pids="$(gateway_pids || true)"
started_log="$(gateway_log_since "$spawn_ts" || true)"
if [[ -z "$new_pids" ]]; then
  echo "ERROR: ${profile} gateway failed to start" >&2
  # Failures before _setup_gateway_log() (bad interpreter, import errors) are
  # all run/logs/gateway.log ever sees; everything later is in the gateway log.
  if [[ -s run/logs/gateway.log ]]; then
    echo "--- run/logs/gateway.log"
    tail -20 run/logs/gateway.log
  fi
  if [[ -n "$started_log" ]]; then
    echo "--- ${started_log}"
    tail -30 "$started_log"
  else
    echo "(no new ${gateway_log_dir}/gateway_*.log was created)"
  fi
  exit 1
fi
echo "${profile} gateway started (pid $(echo "$new_pids" | head -1), port ${port}${sandbox:+, sandbox ${sandbox}})"
if [[ -n "$started_log" ]]; then
  echo "--- ${started_log}"
  tail -5 "$started_log"
fi
REMOTE

gateway_port="$(sed -n 's/^GATEWAY_PORT=\([0-9]*\)$/\1/p' "$restart_log" | tail -1)"
gateway_port="${gateway_port:-8765}"
if [[ -n "$sandbox" ]]; then
  gateway_target="http://${THOR_HOST}:${gateway_port}"
fi

if [[ $restart_rc -eq 75 ]]; then
  echo "ERROR: another deploy is already restarting the ${profile} gateway on ${remote}" >&2
  exit 75
elif [[ $restart_rc -ne 0 ]]; then
  echo "ERROR: ${profile} gateway restart failed (exit ${restart_rc})" >&2
  exit "$restart_rc"
fi

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

echo "    http://localhost:5173/ -> ${gateway_target} (vite moves to the next free port if 5173 is taken)"
export GUI_API_TARGET="$gateway_target"
exec npm run dev -- --host 0.0.0.0 --port 5173
