#!/usr/bin/env bash
set -euo pipefail

if [[ ${EUID} -ne 0 ]]; then
  echo "Run this launcher with sudo." >&2
  exit 2
fi
if [[ "${FRANKA_PHYSICAL_WATCHER_CONFIRMED:-}" != "YES" ]]; then
  echo "A person with immediate E-stop access must be beside the robot." >&2
  exit 3
fi

ROOT="/home/nvidia/box_api"
PYTHON="/home/nvidia/Code/infer/.venv-fr3/bin/python"
CSV="/home/nvidia/lerobot/outputs/datasets/thor_gmsl2_10ch_v1_20260826_164543/derived/april_cube_tracking_in_robot_base/ik_solutions_p0_v2_box_api_start_with_gripper_v2_retimed_8x_20260902/ik_solution.left.csv"

cleanup() {
  "${ROOT}/scripts/restore_franka_non_rt.sh" || true
}
trap cleanup EXIT INT TERM

"${ROOT}/scripts/prepare_franka_non_rt.sh"
export LD_LIBRARY_PATH="/home/nvidia/Code/infer/.venv-fr3/lib/python3.12/site-packages/cmeel.prefix/lib:${LD_LIBRARY_PATH:-}"

timeout --signal=INT --kill-after=3s 75s \
  taskset -c 13 "${PYTHON}" "${ROOT}/scripts/move_to_episode_start.py" \
  --csv "${CSV}" \
  --episode 0 \
  --robot-ip 192.168.11.102 \
  --speed-factor 0.03 \
  --success-tolerance-rad 0.03
