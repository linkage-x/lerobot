#!/usr/bin/env bash
# Run on Thor. This explicit one-time setup touches only USB access and the
# collection environment's input dependencies; it never connects to FR3.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"
python_bin="${THOR_PYTHON:-.venv/bin/python}"

if [[ "${1:-}" == "--help" || "${1:-}" == "-h" ]]; then
  echo "Usage: bash run/setup_thor_spacemouse.sh [--check]"
  echo "Default installs USB libraries, input Python dependencies and udev rules on Thor."
  echo "--check only opens and reads the SpaceMouse; THOR_PYTHON overrides .venv/bin/python."
  exit 0
fi
if [[ $# -gt 1 || ( $# -eq 1 && "$1" != "--check" ) ]]; then
  echo "ERROR: expected no arguments or --check" >&2
  exit 2
fi
if [[ ! -x "$python_bin" ]]; then
  echo "ERROR: collection Python not found: $python_bin" >&2
  exit 1
fi

if [[ "${1:-}" != "--check" ]]; then
  sudo apt-get update
  sudo apt-get install -y --no-install-recommends libhidapi-hidraw0 libhidapi-libusb0 libusb-1.0-0
  if [[ -n "${UV_BIN:-}" ]]; then
    uv_bin="$UV_BIN"
  elif command -v uv >/dev/null 2>&1; then
    uv_bin="$(command -v uv)"
  elif [[ -x "$HOME/.local/bin/uv" ]]; then
    uv_bin="$HOME/.local/bin/uv"
  else
    echo "ERROR: uv is required to install SpaceMouse Python dependencies." >&2
    exit 1
  fi
  "$uv_bin" pip install --python "$python_bin" \
    'pyspacemouse==2.1.0' 'draccus==0.10.0' 'huggingface-hub>=1,<2' numpy scipy
  getent group plugdev >/dev/null || sudo groupadd plugdev
  login_user="${SUDO_USER:-$(id -un)}"
  sudo usermod -aG plugdev "$login_user"
  sudo tee /etc/udev/rules.d/70-lerobot-spacemouse.rules >/dev/null <<'RULES'
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0660", GROUP="plugdev"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c626", MODE="0660", GROUP="plugdev"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c628", MODE="0660", GROUP="plugdev"
SUBSYSTEM=="hidraw", ATTRS{idVendor}=="046d", ATTRS{idProduct}=="c62b", MODE="0660", GROUP="plugdev"
RULES
  sudo udevadm control --reload-rules
  sudo udevadm trigger --subsystem-match=hidraw
  echo "SpaceMouse setup complete. Replug it and log out/in so plugdev applies to the gateway."
  echo "Then run: bash run/setup_thor_spacemouse.sh --check"
  exit 0
fi

"$python_bin" - <<'PY'
import pyspacemouse

enumerate_devices = getattr(pyspacemouse, 'get_connected_devices', None) or getattr(pyspacemouse, 'list_devices', None)
if not callable(enumerate_devices):
    raise SystemExit('SpaceMouse package has no supported device enumeration API.')
devices = list(enumerate_devices())
print(f'SpaceMouse devices: {devices}')
if not devices:
    raise SystemExit('No SpaceMouse found. Check USB, hidapi libraries and udev permissions.')
device = None
closer = None
try:
    device = pyspacemouse.open()
    if not device:
        raise SystemExit('Cannot open SpaceMouse. Replug and log out/in after udev/group setup.')
    # pyspacemouse 2.1.0 returns an owning SpaceMouseDevice. Older releases
    # return a bool and expose module-level read/close instead.
    reader = getattr(device, 'read', None) or getattr(pyspacemouse, 'read', None)
    closer = getattr(device, 'close', None) or getattr(pyspacemouse, 'close', None)
    if not callable(reader) or not callable(closer):
        raise SystemExit('SpaceMouse package has no supported read/close API.')
    state = reader()
    if state is None:
        raise SystemExit('SpaceMouse opened but did not return a state.')
    print({key: getattr(state, key) for key in ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'buttons')})
finally:
    if callable(closer):
        closer()
PY
