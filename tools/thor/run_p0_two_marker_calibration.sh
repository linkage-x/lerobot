#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
THOR="${THOR:-nvidia@192.168.111.122}"
THOR_DIR="${THOR_DIR:-/home/nvidia/lerobot}"

sync_repo=true
remote_args=()

usage() {
  cat <<'EOF'
Usage:
  bash tools/thor/run_p0_two_marker_calibration.sh [launcher options] [-- calibration options]

Launcher options:
  --no-sync       Do not rsync the repository before starting.
  --sync-only     Sync the repository and exit.
  -h, --help      Show this help.

Common calibration options:
  --existing reuse
      Keep the passing active calibration and exit without connecting hardware.
  --existing recalibrate --execute --confirmation P0_TWO_MARKER_TEACHING
      Start a new manual teaching-mode capture.

Environment:
  THOR       SSH target (default: nvidia@192.168.111.122)
  THOR_DIR   repository on Thor (default: /home/nvidia/lerobot)
  PYTHON_BIN Python on Thor (default: /home/nvidia/Code/infer/.venv-fr3/bin/python3)

The viewer uses trusted X11 forwarding. The host needs a working X server and
Thor sshd must permit X11Forwarding.
EOF
}

sync_only=false
while [[ $# -gt 0 ]]; do
  case "$1" in
    --no-sync)
      sync_repo=false
      shift
      ;;
    --sync-only)
      sync_only=true
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    --)
      shift
      remote_args+=("$@")
      break
      ;;
    *)
      remote_args+=("$1")
      shift
      ;;
  esac
done

if ${sync_repo}; then
  echo "==> Syncing repository to ${THOR}:${THOR_DIR}"
  bash "${REPO_ROOT}/run/sync_to_thor.sh"
fi
if ${sync_only}; then
  exit 0
fi

quoted_args=""
for arg in "${remote_args[@]}"; do
  printf -v quoted_arg '%q' "${arg}"
  quoted_args+=" ${quoted_arg}"
done
remote_command=$(printf \
  'cd %q && PYTHONPATH=src:. LD_LIBRARY_PATH=%q %q tools/thor/p0_two_marker_calibration.py%s' \
  "${THOR_DIR}" \
  "/home/nvidia/lerobot/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/usr/local/lib" \
  "${PYTHON_BIN:-/home/nvidia/Code/infer/.venv-fr3/bin/python3}" \
  "${quoted_args}")

echo "==> Starting standalone calibration on ${THOR} (camera and robot remain on Thor1)"
exec ssh -Y -t -o ConnectTimeout=5 -o ForwardX11Timeout=0 "${THOR}" "${remote_command}"
