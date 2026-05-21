#!/usr/bin/env bash
# Read MAX96726 link-lock status and map locked links to /dev/video IDs.
#
# Run on the Jetson/Thor:
#   ./tools/gmsl2/check_max96726_locks.sh
#
# MAX96726 REG0x0008:
#   bit0 = LOCKED_A
#   bit1 = LOCKED_B
#   bit2 = LOCKED_C
#   bit3 = LOCKED_D

set -euo pipefail

I2C_ADDR="${I2C_ADDR:-0x33}"
REG_HI=0x00
REG_LO=0x08

BUS_PORTS=(17 18 19 20)
BUS_NAMES=("port1" "port2" "port3" "port4")
VIDEO_BASES=(0 4 8 12)
LINK_NAMES=("A" "B" "C" "D")

if ! command -v i2ctransfer >/dev/null 2>&1; then
  echo "ERROR: i2ctransfer not found. Install i2c-tools on the Thor." >&2
  exit 127
fi

run_i2c() {
  local bus="$1"
  if [[ "${EUID}" -eq 0 ]]; then
    i2ctransfer -f -y "${bus}" "w2@${I2C_ADDR}" "${REG_HI}" "${REG_LO}" r1
  elif sudo -n true >/dev/null 2>&1; then
    sudo -n i2ctransfer -f -y "${bus}" "w2@${I2C_ADDR}" "${REG_HI}" "${REG_LO}" r1
  else
    echo "ERROR: need root or passwordless sudo to read I2C bus ${bus}" >&2
    exit 1
  fi
}

locked_ids=()
unlocked_ids=()

printf "MAX96726 link-lock register: addr=%s reg=0x0008\n" "${I2C_ADDR}"
printf "%-6s %-6s %-8s %-7s %-8s %-8s\n" "bus" "port" "video" "link" "locked" "reg0x08"

for idx in "${!BUS_PORTS[@]}"; do
  bus="${BUS_PORTS[$idx]}"
  port="${BUS_NAMES[$idx]}"
  base="${VIDEO_BASES[$idx]}"
  raw="$(run_i2c "${bus}")"
  value="$((raw))"

  for bit in 0 1 2 3; do
    video_id=$((base + bit))
    link="${LINK_NAMES[$bit]}"
    if (( (value >> bit) & 1 )); then
      locked="yes"
      locked_ids+=("${video_id}")
    else
      locked="no"
      unlocked_ids+=("${video_id}")
    fi
    printf "%-6s %-6s %-8s %-7s %-8s %-8s\n" \
      "${bus}" "${port}" "video${video_id}" "${link}" "${locked}" "${raw}"
  done
done

echo
printf "LOCKED_VIDEO_IDS="
(IFS=,; echo "${locked_ids[*]}")
printf "UNLOCKED_VIDEO_IDS="
(IFS=,; echo "${unlocked_ids[*]}")
