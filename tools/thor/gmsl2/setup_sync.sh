#!/usr/bin/env bash
# Cold-boot bring-up for the SG16A_AGTH_G3Y_A1 GMSL2 board.
#
# This is the runtime equivalent of the SDK's `load_modules.sh` + `pwm.sh`
# combined: it (1) unloads stale camera modules, (2) insmods the three
# kernel objects we vendor under `tools/thor/gmsl2/sdk/ko/`, (3) runs
# `boost_clock.sh` to lock VI/ISP/NVCSI/EMC at max, (4) arms `pwmchip4/pwm0`
# at the requested frequency, and (5) puts every `/dev/videoN` into slave
# trigger mode (`trig_mode=1`).
#
# Once this has run, each subsequent session only needs `sudo sh
# tools/thor/gmsl2/sdk/pwm.sh` (which `gmsl2_record.py` invokes automatically).
#
# Usage (from the repo root):
#   sudo ./tools/thor/gmsl2/setup_sync.sh [--sdk DIR] [--fps 60] [--num 16]
#                                    [--trig-pin 0x00020007] [--master-id N]
#                                    [--dry-run]
#
# `--sdk` defaults to the vendored copy at `tools/thor/gmsl2/sdk/`. Pass it
# explicitly if you want to point at a different driver pack.
#
# Notes:
#   * Must run as root (use `sudo`).
#   * `--master-id N` keeps camera N free-running (trig_mode=0) -- useful
#     when no PWM wire is hooked up to the SG16A trigger pin.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SDK_DIR="${SDK_DIR:-${SCRIPT_DIR}/sdk}"
FPS=60
NUM=16
TRIG_PIN="0x00020007"
MASTER_ID="-1"
SENSOR_MODE=0
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --sdk) SDK_DIR="$2"; shift 2 ;;
    --fps) FPS="$2"; shift 2 ;;
    --num) NUM="$2"; shift 2 ;;
    --trig-pin) TRIG_PIN="$2"; shift 2 ;;
    --master-id) MASTER_ID="$2"; shift 2 ;;
    --sensor-mode) SENSOR_MODE="$2"; shift 2 ;;
    --dry-run) DRY_RUN=1; shift ;;
    -h|--help)
      sed -n '2,25p' "$0"
      exit 0
      ;;
    *) echo "Unknown argument: $1" >&2; exit 2 ;;
  esac
done

if [[ $EUID -ne 0 && $DRY_RUN -eq 0 ]]; then
  echo "This script must be run as root (sudo)." >&2
  exit 1
fi

if [[ ! -d "$SDK_DIR" ]]; then
  echo "SDK directory not found: $SDK_DIR" >&2
  echo "Pass --sdk /path/to/SG16A_AGTH_G3Y_A1 or set SDK_DIR." >&2
  exit 1
fi

run() {
  if [[ $DRY_RUN -eq 1 ]]; then
    echo "+ $*"
  else
    echo "+ $*"
    "$@"
  fi
}

echo "==> Unloading any previous camera modules"
for mod in sg8-imx715c-g3a sg12-imx577c-g3a sg17-imx735c-g3a sgx-yuv-gmsl2 sg2-ar0234c-g2f max96726; do
  run rmmod "$mod" 2>/dev/null || true
done

echo "==> Loading SG16A modules (max96726 + AR0234 driver)"
run insmod "$SDK_DIR/ko/max96726.ko"
run insmod "$SDK_DIR/ko/pwm-gpio.ko" 2>/dev/null || true
run insmod "$SDK_DIR/ko/sg2-ar0234c-g2f.ko"

echo "==> Boosting clocks (VI/ISP/NVCSI/EMC -> max)"
if [[ -x "$SDK_DIR/boost_clock.sh" ]]; then
  run bash "$SDK_DIR/boost_clock.sh" || true
fi

echo "==> Programming PWM @ ${FPS}Hz on pwmchip4/pwm0 (sync clock)"
PERIOD_NS=$(awk -v fps="$FPS" 'BEGIN { printf "%d", 1000000000 / fps }')
DUTY_NS=$(( PERIOD_NS / 2 ))
PWM=/sys/class/pwm/pwmchip4
if [[ -d "$PWM" ]]; then
  if [[ -w "$PWM/unexport" ]]; then
    echo 0 > "$PWM/unexport" 2>/dev/null || true
  fi
  if [[ ! -d "$PWM/pwm0" ]]; then
    echo 0 > "$PWM/export"
  fi
  if [[ $DRY_RUN -eq 0 ]]; then
    echo "$PERIOD_NS" > "$PWM/pwm0/period"
    echo "$DUTY_NS"  > "$PWM/pwm0/duty_cycle"
    echo 1 > "$PWM/pwm0/enable"
  else
    echo "+ would set period=$PERIOD_NS duty=$DUTY_NS on $PWM/pwm0"
  fi
else
  echo "WARNING: $PWM not available -- check that pwm-gpio.ko loaded successfully" >&2
fi

echo "==> Configuring per-camera trigger controls (NUM=$NUM, master=$MASTER_ID)"
for i in $(seq 0 $((NUM - 1))); do
  DEV="/dev/video$i"
  if [[ ! -e "$DEV" && $DRY_RUN -eq 0 ]]; then
    echo "WARNING: $DEV does not exist (driver may not have enumerated this channel)" >&2
    continue
  fi
  if [[ "$i" -eq "$MASTER_ID" ]]; then
    TRIG_MODE=0
  else
    TRIG_MODE=1
  fi
  run v4l2-ctl -d "$DEV" -c \
    "sensor_mode=${SENSOR_MODE},trig_pin=${TRIG_PIN},trig_mode=${TRIG_MODE}"
done

echo "==> Done. Cameras are configured for hardware-synced capture at ${FPS}Hz."
