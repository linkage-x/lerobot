#!/usr/bin/env bash
# Ensure Thor exposes the BOX upstream target address after boot.
# The BOX firmware is configured to push sensor UDP to 192.168.2.45:15000.
set -euo pipefail

IFACE="${BOX_NET_IFACE:-enP2p1s0}"
ADDR="${BOX_NET_ADDR:-192.168.2.45/24}"
BOX_IP="${BOX_REMOTE_IP:-192.168.2.60}"
DISABLE_RPFILTER="${BOX_NET_DISABLE_RPFILTER:-0}"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  exec sudo BOX_NET_IFACE="$IFACE" BOX_NET_ADDR="$ADDR" BOX_REMOTE_IP="$BOX_IP" BOX_NET_DISABLE_RPFILTER="$DISABLE_RPFILTER" "$0" "$@"
fi

if ! ip link show dev "$IFACE" >/dev/null 2>&1; then
  echo "BOX net error: interface $IFACE not found" >&2
  exit 1
fi

ip link set dev "$IFACE" up
if ! ip -4 addr show dev "$IFACE" | grep -Fq "${ADDR%/*}/"; then
  ip addr add "$ADDR" dev "$IFACE"
fi

if [[ "$DISABLE_RPFILTER" == "1" ]]; then
  sysctl -w "net.ipv4.conf.${IFACE}.rp_filter=0" >/dev/null
  sysctl -w net.ipv4.conf.all.rp_filter=0 >/dev/null
fi

ip -br addr show "$IFACE"
ip route get "$BOX_IP" || true
