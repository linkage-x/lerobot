#!/usr/bin/env bash
set -euo pipefail

# Run this entry point from a terminal on Thor itself.  It avoids SSH/X11
# forwarding; cameras and the FR3 still use the same local Thor runtime.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-/home/nvidia/Code/infer/.venv-fr3/bin/python3}"
CMEEL_LIB="/home/nvidia/lerobot/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/usr/local/lib"

if [[ -z "${DISPLAY:-}" ]]; then
  if [[ -S /tmp/.X11-unix/X0 ]]; then
    export DISPLAY=:0
  else
    echo "WARNING: no local DISPLAY; capture mode will fail, but offline --solve-run can continue." >&2
  fi
fi

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
export OPENBLAS_NUM_THREADS="${OPENBLAS_NUM_THREADS:-1}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-1}"
export PYTHONPATH="${REPO_ROOT}/src:${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}"
export LD_LIBRARY_PATH="${CMEEL_LIB}${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"

cd "${REPO_ROOT}"
echo "==> Starting local Thor calibration (DISPLAY=${DISPLAY:-unset})"
exec "${PYTHON_BIN}" tools/thor/p0_two_marker_calibration.py "$@"
