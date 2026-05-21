#!/usr/bin/env bash
# Arm the SG16A_AGTH_G3Y_A1 hardware-sync trigger source.
#
# The board's per-camera trigger input is gated by `trig_mode=1` and driven
# by the Jetson PWM controller `pwmchip4/pwm0`. This script exports that
# PWM channel, programs it to 60 Hz with 50% duty (period = 16666666 ns,
# duty_cycle = 8333333 ns) -- the AR0234 dtbo's only allowed framerate --
# and enables the output.
#
# Re-runs are idempotent: the channel is unexported then re-exported so a
# different period/duty can be applied without rebooting.
#
# Provenance:
#   Derived from SENSING's SDK pwm.sh (which ships at 25 Hz, period=40 ms).
#   The frequency was bumped to 60 Hz here to match the AR0234 dtbo, where
#   min/max/default_framerate are hard-locked to 60000000 (* 1/1e6 = 60 fps).
#
# Usage: sudo sh tools/gmsl2/sdk/pwm.sh
#        FPS=30 sudo -E sh tools/gmsl2/sdk/pwm.sh   # override

set -eu

PWM_CHIP=${PWM_CHIP:-pwmchip4}
PWM_ID=${PWM_ID:-0}
FPS=${FPS:-60}

PWM_DIR=/sys/class/pwm/$PWM_CHIP
PWM_CH=$PWM_DIR/pwm$PWM_ID

if [ ! -d "$PWM_DIR" ]; then
    echo "ERROR: $PWM_DIR not present -- is the pwm-gpio kernel module loaded?" >&2
    exit 1
fi

# period = round(1e9 / fps); duty = period / 2 (50%)
PERIOD_NS=$(awk -v fps="$FPS" 'BEGIN { printf "%d", 1000000000 / fps }')
DUTY_NS=$(( PERIOD_NS / 2 ))

# Ensure the channel is exported. If a previous run left it running,
# disable it first so we can rewrite period / duty_cycle.
if [ ! -d "$PWM_CH" ]; then
    echo "$PWM_ID" > "$PWM_DIR/export"
fi
if [ -f "$PWM_CH/enable" ]; then
    # Disable while reprogramming (the kernel rejects period writes when
    # enable=1, and rejects duty>period when scaling up).
    echo 0 > "$PWM_CH/enable" 2>/dev/null || true
fi

# Order matters: shrink duty before period when going to a smaller value,
# expand period before duty when going larger. Setting duty=0 first works
# both ways.
echo 0            > "$PWM_CH/duty_cycle" 2>/dev/null || true
echo "$PERIOD_NS" > "$PWM_CH/period"
echo "$DUTY_NS"   > "$PWM_CH/duty_cycle"
echo 1            > "$PWM_CH/enable"

printf "pwm armed: %s/pwm%s period=%s ns duty=%s ns (%s Hz)\n" \
    "$PWM_CHIP" "$PWM_ID" "$PERIOD_NS" "$DUTY_NS" "$FPS"
