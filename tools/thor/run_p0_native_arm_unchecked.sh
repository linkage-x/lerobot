#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../.." && pwd)"
BUNDLE_ROOT="/home/nvidia/box_api/replay_p0_native_arm_only_20260908"

export PYTHONPATH="${BUNDLE_ROOT}/python:/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="/home/nvidia/lerobot/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/usr/local/lib"

exec /home/nvidia/Code/infer/.venv-fr3/bin/python \
  "${REPO_ROOT}/tools/thor/p0_native_arm_unchecked.py" "$@"
