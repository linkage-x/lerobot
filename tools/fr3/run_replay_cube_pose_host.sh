#!/usr/bin/env bash
set -euo pipefail

cd /home/corenetic/Code/zyx/lerobot

select_python() {
  if [[ -n "${FR3_HOST_PYTHON:-}" ]]; then
    printf '%s\n' "${FR3_HOST_PYTHON}"
    return
  fi

  if [[ -x ".venv-fr3/bin/python" ]]; then
    printf '%s\n' ".venv-fr3/bin/python"
    return
  fi

  printf 'Could not find .venv-fr3/bin/python. Run: UV_PROJECT_ENVIRONMENT=.venv-fr3 uv sync --extra fr3-host --extra cv2-gui\n' >&2
  exit 2
}

FR3_HOST_PYTHON="$(select_python)"
venv_root="$(cd "$(dirname "${FR3_HOST_PYTHON}")/.." && pwd)"
cmeel_prefix="$(find "${venv_root}/lib" -path '*/site-packages/cmeel.prefix' -type d | head -n 1 || true)"

export PYTHONPATH="$PWD/src:/opt/MVS/Samples/64/Python:/opt/MVS/Samples/32/Python${PYTHONPATH:+:$PYTHONPATH}"
export HIKROBOT_MVS_HOME=/opt/MVS
export MVCAM_COMMON_RUNENV=/opt/MVS/lib
export LD_LIBRARY_PATH="${cmeel_prefix:+${cmeel_prefix}/lib:}/usr/local/lib:/opt/MVS/lib/64:/opt/MVS/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

echo "[INFO] host_python=${FR3_HOST_PYTHON}"
echo "[INFO] LD_LIBRARY_PATH=${LD_LIBRARY_PATH}"

exec "${FR3_HOST_PYTHON}" \
  third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.host.yaml \
  "$@"
