#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

python_version="${PYTHON_VERSION:-3.12}"
venv_path="${VENV_PATH:-.venv-fr3}"
install_system_deps="${INSTALL_SYSTEM_DEPS:-1}"
configure_device_access="${CONFIGURE_DEVICE_ACCESS:-1}"
install_real_robot_deps="${INSTALL_REAL_ROBOT_DEPS:-1}"

if [[ -n "${UV_BIN:-}" ]]; then
  uv_bin="$UV_BIN"
elif command -v uv >/dev/null 2>&1; then
  uv_bin="$(command -v uv)"
elif [[ -x "$HOME/.local/bin/uv" ]]; then
  uv_bin="$HOME/.local/bin/uv"
else
  echo "ERROR: uv is required. Install it before running this script." >&2
  exit 1
fi

uv_version="$($uv_bin --version | awk '{print $2}')"
if [[ "$(printf '%s\n' "0.10.0" "$uv_version" | sort -V | head -n1)" != "0.10.0" ]]; then
  echo "ERROR: uv>=0.10.0 is required for package-specific Ruckig build dependencies (found $uv_version)." >&2
  echo "Run: $uv_bin self update" >&2
  exit 1
fi

apt_packages=(
  libegl1
  libgl1
  libglfw3
  libhidapi-dev
  libhidapi-hidraw0
  libhidapi-libusb0
  libudev-dev
  libusb-1.0-0
  build-essential
  cmake
  libeigen3-dev
  libpoco-dev
  liburdfdom-dev
  ninja-build
)

if [[ "$install_system_deps" == "1" ]]; then
  missing_packages=()
  for package in "${apt_packages[@]}"; do
    if ! dpkg-query -W -f='${Status}' "$package" 2>/dev/null | grep -q "install ok installed"; then
      missing_packages+=("$package")
    fi
  done
  if ((${#missing_packages[@]})); then
    echo "==> Installing system dependencies: ${missing_packages[*]}"
    sudo apt-get update
    sudo apt-get install -y --no-install-recommends "${missing_packages[@]}"
  else
    echo "==> System dependencies already installed"
  fi
fi

if [[ "$configure_device_access" == "1" ]]; then
  login_user="${SUDO_USER:-$USER}"
  for device_group in plugdev dialout; do
    if getent group "$device_group" >/dev/null \
      && ! id -nG "$login_user" | tr ' ' '\n' | grep -qx "$device_group"; then
      echo "==> Adding ${login_user} to ${device_group} for workstation device access"
      sudo usermod -aG "$device_group" "$login_user"
    fi
  done
fi

echo "==> Syncing ${venv_path} with Python ${python_version}"
sync_extras=(--extra fr3-workstation-teleop)
if [[ "$install_real_robot_deps" == "1" ]]; then
  sync_extras+=(--extra fr3-host)
fi
UV_PROJECT_ENVIRONMENT="$venv_path" "$uv_bin" sync \
  --python "$python_version" \
  "${sync_extras[@]}" \
  --no-dev

echo "==> Running FR3 workstation dependency smoke checks"
PYTHONPATH=src:. "$venv_path/bin/python" - <<'PY'
from pathlib import Path

import cv2
import mujoco
import pyrealsense2 as rs
import pyspacemouse

model_path = Path(
    "src/lerobot/robots/franka_research3/assets/franka_fr3/"
    "fr3_pika_gripper_scene.xml"
)
model = mujoco.MjModel.from_xml_path(str(model_path))
realsense_serials = [
    device.get_info(rs.camera_info.serial_number)
    for device in rs.context().query_devices()
]
if hasattr(pyspacemouse, "list_devices"):
    spacemouse_devices = list(pyspacemouse.list_devices())
else:
    spacemouse_devices = list(pyspacemouse.get_connected_devices())

print("python=OK")
print(f"opencv={cv2.__version__}")
print(f"mujoco={mujoco.__version__} nq={model.nq} nu={model.nu}")
print(f"realsense_devices={realsense_serials}")
print(f"spacemouse_devices={len(spacemouse_devices)}")
PY

if [[ "$install_real_robot_deps" == "1" ]]; then
  PYTHONPATH=src:. "$venv_path/bin/python" - <<'PY'
import panda_py
import pika
import pinocchio
import placo
import ruckig

print(f"panda_py={panda_py.__file__}")
print(f"pika={pika.__file__}")
print(f"pinocchio={pinocchio.__version__}")
print(f"placo={placo.__version__ if hasattr(placo, '__version__') else placo.__file__}")
print(f"ruckig={ruckig.__version__ if hasattr(ruckig, '__version__') else ruckig.__file__}")
PY
fi

echo "==> FR3 workstation environment ready: ${repo_root}/${venv_path}"
