#!/usr/bin/env bash

_box_setup_sourced=0
if [[ "${BASH_SOURCE[0]}" != "$0" ]]; then
  _box_setup_sourced=1
else
  set -euo pipefail
fi

THIS_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROS_LIB_PATH=""
if [[ -n "${ROS_DISTRO:-}" && -d "/opt/ros/${ROS_DISTRO}/lib" ]]; then
  ROS_LIB_PATH="/opt/ros/${ROS_DISTRO}/lib"
elif [[ -d "/opt/ros/humble/lib" ]]; then
  ROS_LIB_PATH="/opt/ros/humble/lib"
fi

if [[ -n "$ROS_LIB_PATH" ]]; then
  export LD_LIBRARY_PATH="$THIS_DIR/lib:$ROS_LIB_PATH:${LD_LIBRARY_PATH:-}"
else
  export LD_LIBRARY_PATH="$THIS_DIR/lib:${LD_LIBRARY_PATH:-}"
fi
export BOX_SDK_URDF="${BOX_SDK_URDF:-$THIS_DIR/share/monte_gripper.urdf}"
# v3 SDK reads $BOX_SDK_CONFIG (box_sdk.conf) for sensor_csv_enabled; pointing
# at the repo copy (sensor_csv_enabled=0) disables the high-rate CSV dump at the
# source, superseding box_client.py's cleanup_box_csv workaround.
export BOX_SDK_CONFIG="${BOX_SDK_CONFIG:-$THIS_DIR/box_sdk.conf}"

echo "LD_LIBRARY_PATH=$LD_LIBRARY_PATH"
echo "BOX_SDK_URDF=$BOX_SDK_URDF"
echo "BOX_SDK_CONFIG=$BOX_SDK_CONFIG"

if [[ $# -gt 0 && $_box_setup_sourced -eq 0 ]]; then
  exec "$@"
fi
