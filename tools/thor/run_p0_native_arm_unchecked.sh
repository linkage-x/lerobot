#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
BUNDLE_ROOT="/home/nvidia/box_api/replay_p0_native_arm_only_20260908"
DATASET_RUNNER="/home/nvidia/lerobot/tools/thor/p0_native_arm_dataset_replay.py"

export PYTHONPATH="${BUNDLE_ROOT}/python:/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib/python3.12/site-packages"
export LD_LIBRARY_PATH="/home/nvidia/lerobot/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:/usr/local/lib"

for arg in "$@"; do
  if [[ "${arg}" == "--dataset-root" || "${arg}" == --dataset-root=* \
    || "${arg}" == "--episode-index" || "${arg}" == --episode-index=* \
    || "${arg}" == "--side" || "${arg}" == --side=* ]]; then
    exec /home/nvidia/Code/infer/.venv-fr3/bin/python \
      "${DATASET_RUNNER}" --unchecked-execution --gripper-mode dataset "$@"
  fi
done

# Preserve the original sealed frozen-P0 behavior when no dataset-mode option is present.
exec /home/nvidia/Code/infer/.venv-fr3/bin/python "${BUNDLE_ROOT}/native_arm.py" "$@"
