#!/usr/bin/env bash
# Run on Thor once. Does not install the FR3 backend or start robot motion.
set -euo pipefail
repo_root="$(cd "$(dirname "$0")/.." && pwd)"
cd "$repo_root"
python_bin="${SPACEMOUSE_PYTHON:-.venv/bin/python}"
sudo apt-get install -y libhidapi-hidraw0 libhidapi-libusb0
uv pip install --python "$python_bin" 'pyspacemouse==2.1.0' draccus huggingface_hub numpy
sudo tee /etc/udev/rules.d/70-lerobot-spacemouse.rules >/dev/null <<'RULE'
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="256f", MODE="0660", GROUP="plugdev", TAG+="uaccess"
KERNEL=="hidraw*", SUBSYSTEM=="hidraw", ATTRS{idVendor}=="046d", MODE="0660", GROUP="plugdev", TAG+="uaccess"
RULE
sudo usermod -aG plugdev "$(id -un)"
sudo udevadm control --reload-rules
sudo udevadm trigger --subsystem-match=hidraw
echo "Log out/in to refresh group membership, replug SpaceMouse, then run the spacemouse component check."
