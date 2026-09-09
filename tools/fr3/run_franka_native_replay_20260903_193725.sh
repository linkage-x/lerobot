#!/usr/bin/env bash
set -euo pipefail

echo "BLOCKED: the episode start pose was reported incorrect on 2026-09-03. Recalibrate/confirm the BOX-to-TCP transform and IK branch before hardware replay." >&2
exit 9

# Two single-arm episodes captured on 2026-09-03 with BOX 1819152274.
# The native panda-py backend performs its own low-speed trajectory generation,
# so it consumes the 991 original IK waypoints instead of the 8x interpolation.
export FRANKA_IK_DIR="/home/nvidia/lerobot/outputs/datasets/thor_gmsl2_10ch_v1_20260903_193725/derived/april_cube_tracking_in_robot_base/ik_solutions_p0_v2_right_box_with_gripper_retimed_8x_20260903"
export FRANKA_URDF_PATH="/home/nvidia/box_api/models/fr3_p0_v2/fr3_corenetic_gripper_v2_p0.urdf"
export FRANKA_CUBE="right"
export FRANKA_REPLAY_BACKEND="native"
export FRANKA_NATIVE_CHUNK_SIZE="25"
export FRANKA_SUPERVISOR_TIMEOUT_S="360"
export FRANKA_REPLAY_FPS="60"

exec /home/nvidia/box_api/scripts/run_franka_non_rt_replay_otg.sh "$@"
