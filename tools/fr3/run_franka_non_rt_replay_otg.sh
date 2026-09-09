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

EPISODE="${1:?Usage: $0 EPISODE_INDEX start-only|replay}"
MODE="${2:-start-only}"
if [[ "${MODE}" != "start-only" && "${MODE}" != "replay" ]]; then
  echo "Mode must be start-only or replay." >&2
  exit 4
fi

ROOT="/home/nvidia/box_api"
PYTHON="/home/nvidia/Code/infer/.venv-fr3/bin/python"
CONTROLLER_BACKEND="${FRANKA_REPLAY_BACKEND:-native}"
if [[ "${CONTROLLER_BACKEND}" != "native" && "${CONTROLLER_BACKEND}" != "python-otg" ]]; then
  echo "FRANKA_REPLAY_BACKEND must be native or python-otg." >&2
  exit 5
fi
IK_DIR="${FRANKA_IK_DIR:-/home/nvidia/lerobot/outputs/datasets/thor_gmsl2_10ch_v1_20260826_164543/derived/april_cube_tracking_in_robot_base/ik_solutions_p0_v2_box_api_start_with_gripper_v2_retimed_8x_20260902}"
URDF="${FRANKA_URDF_PATH:-${ROOT}/models/fr3_p0_v2/fr3_corenetic_gripper_v2_p0.urdf}"
CUBE="${FRANKA_CUBE:-left}"
if [[ "${CUBE}" != "left" && "${CUBE}" != "right" ]]; then
  echo "FRANKA_CUBE must be left or right." >&2
  exit 8
fi
SUMMARY="${IK_DIR}/hardware_${CONTROLLER_BACKEND}_${CUBE}_episode_${EPISODE}_${MODE}.json"
SUPERVISOR_TIMEOUT_S="${FRANKA_SUPERVISOR_TIMEOUT_S:-240}"
if [[ "${MODE}" == "start-only" ]]; then
  SUPERVISOR_TIMEOUT_S="${FRANKA_START_ONLY_TIMEOUT_S:-90}"
fi

extra_args=()
if [[ "${MODE}" == "start-only" ]]; then
  extra_args+=(--start-only)
else
  extra_args+=(
    --enable-gripper
    --gripper-backend legacy
    --gripper-sdk-dir /home/nvidia/lerobot/tools/thor/box_sdk
    --gripper-remote-ip 192.168.2.60
  )
fi

runner=("${PYTHON}")
backend_args=(
  --controller-backend "${CONTROLLER_BACKEND}"
)
if [[ "${CONTROLLER_BACKEND}" == "native" ]]; then
  ROBOT_NIC="${FRANKA_NIC:-enP2p1s0}"
  route_line="$(ip route get 192.168.11.102 | head -n 1)"
  if [[ "${route_line}" != *"dev ${ROBOT_NIC}"* ]]; then
    echo "Robot route is not using ${ROBOT_NIC}: ${route_line}" >&2
    exit 6
  fi
  ping_output="$(ping -q -c 10000 -i 0.001 -W 1 -s 1200 -I "${ROBOT_NIC}" 192.168.11.102)"
  printf '%s\n' "${ping_output}"
  packet_loss="$(grep -oE '[0-9.]+% packet loss' <<<"${ping_output}" | awk '{print $1}' | tr -d '%')"
  read -r _min_rtt _avg_rtt max_rtt mdev_rtt < <(
    awk -F'=' '/min\/avg\/max/ {gsub(/ /, "", $2); split($2, a, "/"); print a[1], a[2], a[3], a[4]}' <<<"${ping_output}"
  )
  if [[ "${packet_loss}" != "0" ]] \
    || ! awk -v value="${max_rtt}" 'BEGIN {exit !(value <= 0.90)}' \
    || ! awk -v value="${mdev_rtt}" 'BEGIN {exit !(value <= 0.10)}'; then
    echo "FCI network gate failed: loss=${packet_loss}% max=${max_rtt}ms mdev=${mdev_rtt}ms." >&2
    exit 7
  fi
  echo "FCI network gate passed: max=${max_rtt}ms mdev=${mdev_rtt}ms."
  backend_args+=(
    --native-speed-factor 0.01
    --native-start-speed-factor 0.03
    --native-chunk-size "${FRANKA_NATIVE_CHUNK_SIZE:-200}"
    --native-max-deviation-rad 0.02
    --native-endpoint-error-rad 0.03
  )
else
  cleanup() {
    "${ROOT}/scripts/restore_franka_non_rt.sh" || true
  }
  trap cleanup EXIT INT TERM
  "${ROOT}/scripts/prepare_franka_non_rt.sh"
  runner=(taskset -c 13 "${PYTHON}")
  backend_args+=(
    --otg-control-frequency 800
    --producer-cpu 11
    --otg-max-velocity 0.12,0.12,0.12,0.12,0.12,0.12,0.12
    --otg-max-acceleration 0.4,0.4,0.4,0.4,0.4,0.4,0.4
    --otg-max-jerk 5,5,5,5,5,5,5
    --controller-watchdog-s 0.15
    --controller-stop-timeout-s 2
  )
fi

timeout --signal=INT --kill-after=3s "${SUPERVISOR_TIMEOUT_S}s" \
  "${runner[@]}" \
  "${ROOT}/scripts/replay_ik_trajectory_otg.py" \
  --ik-dir "${IK_DIR}" \
  --urdf-path "${URDF}" \
  --cube "${CUBE}" \
  --episode "${EPISODE}" \
  --robot-ip 192.168.11.102 \
  --fps "${FRANKA_REPLAY_FPS:-20}" \
  --execute \
  --allow-non-realtime \
  --move-to-first-frame \
  --start-timeout-s 60 \
  --summary-json "${SUMMARY}" \
  "${backend_args[@]}" \
  "${extra_args[@]}"
