#!/usr/bin/env bash
set -euo pipefail

STATE_DIR="/run/franka_non_rt_tuning"

if [[ ${EUID} -ne 0 ]]; then
  echo "Run as root." >&2
  exit 2
fi
if [[ ! -f "${STATE_DIR}/state.env" ]]; then
  echo "No saved non-RT tuning state; nothing to restore."
  exit 0
fi

# shellcheck disable=SC1091
source "${STATE_DIR}/state.env"

if [[ -n "${IRQ}" && -n "${IRQ_AFFINITY_WAS}" && -w "/proc/irq/${IRQ}/smp_affinity_list" ]]; then
  printf '%s\n' "${IRQ_AFFINITY_WAS}" >"/proc/irq/${IRQ}/smp_affinity_list"
fi
if [[ "${EEE_WAS}" == enabled* ]]; then
  ethtool --set-eee "${NIC}" eee on || true
else
  ethtool --set-eee "${NIC}" eee off || true
fi
ethtool -K "${NIC}" \
  tso "${TSO_WAS%% *}" \
  gso "${GSO_WAS%% *}" \
  gro "${GRO_WAS%% *}" || true
jetson_clocks --restore "${STATE_DIR}/jetson_clocks.conf" || true

rm -f "${STATE_DIR}/active"
echo "Restored CPU, NIC offload, EEE and IRQ settings saved before non-RT replay."
