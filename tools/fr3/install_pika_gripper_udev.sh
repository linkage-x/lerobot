#!/usr/bin/env bash
# Give the Pika gripper a host-independent name, /dev/pika_gripper, so configs need not carry a
# robot host's USB topology.
#
# Usage:
#   bash tools/fr3/install_pika_gripper_udev.sh                 # find the gripper with --scan
#   bash tools/fr3/install_pika_gripper_udev.sh --port /dev/ttyUSB0
#
# The rule matches the gripper by the USB socket it is plugged into (udev's ID_PATH, the same key
# /dev/serial/by-path uses), not by the CH340's vendor/product id: that id is shared by every
# CH340 adapter, the Pika Sense among them, and two matches would race for the one name. So the
# rule is per host and per socket -- re-run this after moving the gripper to another socket or
# moving the rig to another robot host.
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$repo_root"

rule_path="/etc/udev/rules.d/99-pika-gripper.rules"
link_name="pika_gripper"
python_bin="${PYTHON_BIN:-.venv-fr3/bin/python}"

port=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --port) port="$2"; shift 2 ;;
    --port=*) port="${1#--port=}"; shift ;;
    *) sed -n '2,14p' "$0" | sed 's/^# \{0,1\}//' >&2; exit 2 ;;
  esac
done

if [[ -z "$port" ]]; then
  echo "==> Scanning serial ports for Pika gripper frames (read-only)"
  scan_output="$(PYTHONPATH=src:. "$python_bin" tools/fr3/fr3_pika_gripper_verify.py --scan 2>&1 || true)"
  echo "$scan_output" | sed 's/^/    /'
  port="$(echo "$scan_output" | sed -n 's/^Pika gripper is on \([^ ]*\) .*/\1/p' | head -1)"
  if [[ -z "$port" ]]; then
    echo "ERROR: no port streamed Pika gripper frames; power and cable the gripper, or pass --port" >&2
    exit 1
  fi
fi

id_path="$(udevadm info -q property -n "$port" | sed -n 's/^ID_PATH=//p')"
if [[ -z "$id_path" ]]; then
  echo "ERROR: udev knows no ID_PATH for ${port}; cannot pin it to a USB socket" >&2
  exit 1
fi

rule="SUBSYSTEM==\"tty\", ENV{ID_PATH}==\"${id_path}\", SYMLINK+=\"${link_name}\", GROUP=\"dialout\", MODE=\"0660\""
contents="# Written by tools/fr3/install_pika_gripper_udev.sh: the Pika gripper on USB socket ${id_path}.
${rule}"

if [[ -f "$rule_path" ]] && [[ "$(cat "$rule_path")" == "$contents" ]]; then
  echo "==> ${rule_path} already up to date"
else
  echo "==> Writing ${rule_path}"
  printf '%s\n' "$contents" | sudo tee "$rule_path" >/dev/null
fi
sudo udevadm control --reload
sudo udevadm trigger --subsystem-match=tty --action=change
sudo udevadm settle

if [[ ! -e "/dev/${link_name}" ]]; then
  echo "ERROR: /dev/${link_name} did not appear after reloading udev" >&2
  exit 1
fi
echo "==> /dev/${link_name} -> $(readlink -f "/dev/${link_name}")"
