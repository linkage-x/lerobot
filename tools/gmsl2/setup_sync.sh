#!/usr/bin/env bash
# Configure the SG16A_AGTH_G3Y_A1 GMSL2 board for hardware-synchronous capture.
#
# This script is the runtime equivalent of `load_modules.sh` + `pwm.sh` from the
# SDK, with one important difference: every connected camera is put into SLAVE
# trigger mode (`trig_mode=1`) and the Jetson's own PWM is used as the trigger
# source for all of them. AR0234 in 1920x1080 is locked to 60 fps in the dtbo
# (see `dtb/SG2_AR0234C_G2F/...`), so the default PWM period below matches that
# (16.666 ms = 60 Hz, 50% duty).
#
# Usage (on the Jetson, with the SDK directory available):
#   sudo ./tools/gmsl2/setup_sync.sh [--sdk DIR] [--fps 60] [--num 11]
#                                    [--trig-pin 0x00020007] [--master-id N]
#
# Notes:
#   * The script must run as root.
#   * Re-runs are safe -- existing modules are unloaded first.
#   * `--master-id N` makes camera N free-running (trig_mode=0) and useful when
#     a PWM signal is not available; the remaining cameras are still slaves.

set -euo pipefail

SDK_DIR="${SDK_DIR:-$HOME/Desktop/SG16A_AGTH_G3Y_A1}"
FPS=60
NUM=11
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
