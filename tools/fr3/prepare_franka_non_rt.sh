#!/usr/bin/env bash
set -euo pipefail

# Best-effort low-latency setup for Thor's single shared Ethernet interface.
# This does not replace a PREEMPT_RT kernel or switch QoS. It only removes
# latency sources that can be controlled safely on the host.

NIC="${FRANKA_NIC:-enP2p1s0}"
ROBOT_IP="${FRANKA_ROBOT_IP:-192.168.11.102}"
CONTROL_CPU="${FRANKA_CONTROL_CPU:-13}"
IRQ_CPU="${FRANKA_IRQ_CPU:-12}"
PING_COUNT="${FRANKA_PING_COUNT:-10000}"
MAX_RTT_LIMIT_MS="${FRANKA_MAX_RTT_MS:-0.90}"
MAX_MDEV_LIMIT_MS="${FRANKA_MAX_MDEV_MS:-0.10}"
TUNING_SETTLE_S="${FRANKA_TUNING_SETTLE_S:-5}"
STATE_DIR="/run/franka_non_rt_tuning"

if [[ ${EUID} -ne 0 ]]; then
  echo "Run as root." >&2
  exit 2
fi

apply_tuning() {
  mkdir -p "${STATE_DIR}"
  rm -f "${STATE_DIR}/state.env" "${STATE_DIR}/jetson_clocks.conf"

  local irq irq_affinity_was eee_was tso_was gso_was gro_was
  irq="$(awk -v nic="${NIC}-0" '$0 ~ nic {gsub(":", "", $1); print $1; exit}' /proc/interrupts)"
  irq_affinity_was=""
  if [[ -n "${irq}" && -r "/proc/irq/${irq}/smp_affinity_list" ]]; then
    irq_affinity_was="$(<"/proc/irq/${irq}/smp_affinity_list")"
  fi
  eee_was="$(ethtool --show-eee "${NIC}" 2>/dev/null | awk -F': ' '/EEE status/ {print $2; exit}')"
  tso_was="$(ethtool -k "${NIC}" | awk -F': ' '/^tcp-segmentation-offload:/ {print $2}')"
  gso_was="$(ethtool -k "${NIC}" | awk -F': ' '/^generic-segmentation-offload:/ {print $2}')"
  gro_was="$(ethtool -k "${NIC}" | awk -F': ' '/^generic-receive-offload:/ {print $2}')"

  jetson_clocks --store "${STATE_DIR}/jetson_clocks.conf"
  {
    printf 'IRQ=%q\n' "${irq}"
    printf 'IRQ_AFFINITY_WAS=%q\n' "${irq_affinity_was}"
    printf 'EEE_WAS=%q\n' "${eee_was}"
    printf 'TSO_WAS=%q\n' "${tso_was}"
    printf 'GSO_WAS=%q\n' "${gso_was}"
    printf 'GRO_WAS=%q\n' "${gro_was}"
    printf 'NIC=%q\n' "${NIC}"
  } >"${STATE_DIR}/state.env"

  jetson_clocks
  ethtool --set-eee "${NIC}" eee off
  ethtool -K "${NIC}" tso off gso off gro off
  if [[ -n "${irq}" && -w "/proc/irq/${irq}/smp_affinity_list" ]]; then
    # Keep NAPI/softirq work off the FIFO control CPU. A 10-second stress
    # probe exposed a long stall with IRQ and the control process both on 13.
    printf '%s\n' "${IRQ_CPU}" >"/proc/irq/${irq}/smp_affinity_list"
  fi
  touch "${STATE_DIR}/active"
}

if [[ ! -e "${STATE_DIR}/active" ]]; then
  apply_tuning
  # jetson_clocks and NIC offload changes caused a one-off latency spike when
  # the network gate started immediately. Let the platform settle first.
  sleep "${TUNING_SETTLE_S}"
else
  echo "Single-NIC FCI tuning is already active; rerunning the network gate."
fi

route_line="$(ip route get "${ROBOT_IP}" | head -n 1)"
if [[ "${route_line}" != *"dev ${NIC}"* ]]; then
  echo "Robot route is not using ${NIC}: ${route_line}" >&2
  exit 3
fi

# Franka's documented FCI-level preflight: 1200-byte packets at 1 kHz.
ping_output="$(taskset -c "${CONTROL_CPU}" ping -q -c "${PING_COUNT}" -i 0.001 -W 1 -s 1200 -I "${NIC}" "${ROBOT_IP}")"
printf '%s\n' "${ping_output}"
packet_loss="$(grep -oE '[0-9.]+% packet loss' <<<"${ping_output}" | awk '{print $1}' | tr -d '%')"
read -r min_rtt avg_rtt max_rtt mdev_rtt < <(
  awk -F'=' '/min\/avg\/max/ {gsub(/ /, "", $2); split($2, a, "/"); print a[1], a[2], a[3], a[4]}' <<<"${ping_output}"
)

if [[ "${packet_loss}" != "0" ]]; then
  echo "FCI network gate failed: packet loss=${packet_loss}%." >&2
  exit 4
fi
if ! awk -v value="${max_rtt}" -v limit="${MAX_RTT_LIMIT_MS}" 'BEGIN {exit !(value <= limit)}'; then
  echo "FCI network gate failed: max RTT ${max_rtt} ms > ${MAX_RTT_LIMIT_MS} ms." >&2
  exit 5
fi
if ! awk -v value="${mdev_rtt}" -v limit="${MAX_MDEV_LIMIT_MS}" 'BEGIN {exit !(value <= limit)}'; then
  echo "FCI network gate failed: RTT mdev ${mdev_rtt} ms > ${MAX_MDEV_LIMIT_MS} ms." >&2
  exit 6
fi

echo "FCI network gate passed: avg=${avg_rtt} ms max=${max_rtt} ms mdev=${mdev_rtt} ms."
echo "Non-RT tuning active: NIC=${NIC}, IRQ CPU=${IRQ_CPU}, control CPU=${CONTROL_CPU}."
