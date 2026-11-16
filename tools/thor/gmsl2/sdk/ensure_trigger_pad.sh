#!/bin/sh
# Make sure the Jetson Thor pad that carries the hardware-sync trigger is
# driven by the GPIO, so the pwm-gpio PWM actually reaches the SG16A board.
#
# Why: the PWM (`pwmchip6/pwm0`, a pwm-gpio on gpio-537 = PDD.07) toggles the
# GPIO controller's output. The pad SOC_GPIO25_PDD7 only follows it when its
# pinmux register selects GPIO. That register is set by the pinmux table
# flashed into QSPI (MB1 BCT), not by the rootfs, so a cloned or re-flashed
# Thor can boot with the pad in SFIO mode (function tsc_edge_out0). Then the
# PWM looks armed, `trig_mode=1` is set, and every camera silently free-runs
# at its own 16.660 ms frame time instead of the 16.6667 ms PWM period: frames
# from cameras started at different times never line up and the online-sync
# recorder cannot form SOF clusters. Seen on thor2 (module 3834-0008-500
# C.2, 2026-11-12): register 0x00203401 at boot; thor1 (3834-0008-400 G.5)
# boots with 0x00203000 and syncs.
#
# Bit 10 of the pad register is the SFIO/GPIO select (debugfs pinconf shows it
# as `gpio-mode`; 1 = SFIO here); bits 1:0 select the SFIO function. Clearing
# both gives thor1's value. The register address was located on L4T R38.2.1
# by changing the pad's function through debugfs pinmux-select and diffing
# the AON pinmux block; before writing, the script checks that debugfs, which
# reads the same register, agrees with it.
#
# Usage (root): sh ensure_trigger_pad.sh [fix|check]
#   fix   (default) clear SFIO select if set, verify, report
#   check report only; exit 1 if the pad is not in GPIO mode
# Exit: 0 pad is GPIO (or not a Thor / not applicable), 1 wrong and not fixed.

set -u

MODE=${1:-fix}
PINMUX_COMPAT=/proc/device-tree/bus@0/pinmux@c7a2000/compatible
PINCONF=/sys/kernel/debug/pinctrl/c7a2000.pinmux/pinconf-groups
GPIO_DEBUG=/sys/kernel/debug/gpio
GROUP=soc_gpio25_pdd7
REG=0x0c7a3078
SFIO_SEL=0x400
FUNC_MASK=0x3

say() { echo "trigger pad: $*"; }

case "$MODE" in fix|check) ;; *) echo "usage: $0 [fix|check]" >&2; exit 2 ;; esac

if ! tr -d '\0' < "$PINMUX_COMPAT" 2>/dev/null | grep -q 'nvidia,tegra264-pinmux-aon'; then
    say "not a Jetson Thor AON pinmux, nothing to do"
    exit 0
fi
if ! command -v busybox >/dev/null 2>&1; then
    say "ERROR busybox (devmem) not installed, cannot read $REG" >&2
    exit 1
fi
if [ ! -r "$PINCONF" ]; then
    say "ERROR $PINCONF not readable (run as root, debugfs mounted?)" >&2
    exit 1
fi

# The PWM must be the pwm-gpio on PDD.07; a different wiring is not ours to touch.
if [ -r "$GPIO_DEBUG" ] && grep -q 'pwm-gpio' "$GPIO_DEBUG" \
        && ! grep 'pwm-gpio' "$GPIO_DEBUG" | grep -q 'PDD.07'; then
    say "pwm-gpio is not on PDD.07 here, leaving the pinmux alone"
    exit 0
fi

debugfs_gpio_mode() {
    awk -v g="($GROUP):" '
        $2 == g { f = 1; next }
        f && /^[0-9]+ \(/ { exit }
        f && /gpio-mode=/ { sub(/.*gpio-mode=/, ""); print; exit }
    ' "$PINCONF"
}

val=$(busybox devmem "$REG" 32)
dbg=$(debugfs_gpio_mode)
sel=$(( (val & SFIO_SEL) != 0 ))
if [ "$dbg" != "$sel" ]; then
    say "ERROR $REG=$val does not match debugfs $GROUP gpio-mode=$dbg; wrong register for this L4T, not touching it" >&2
    exit 1
fi

if [ $(( val & (SFIO_SEL | FUNC_MASK) )) -eq 0 ]; then
    say "OK ($GROUP driven by GPIO, $REG=$val)"
    exit 0
fi

if [ "$MODE" = check ]; then
    say "WRONG $GROUP is in SFIO mode ($REG=$val): the PWM does not reach the cameras" >&2
    exit 1
fi

new=$(printf '0x%08x' $(( val & ~(SFIO_SEL | FUNC_MASK) )))
busybox devmem "$REG" 32 "$new"
back=$(busybox devmem "$REG" 32)
if [ $(( back )) -ne $(( new )) ] || [ "$(debugfs_gpio_mode)" != 0 ]; then
    say "ERROR wrote $new to $REG but read back $back" >&2
    exit 1
fi
say "FIXED $GROUP was SFIO ($val), now GPIO ($back): the PWM reaches the SG16A trigger"
exit 0
