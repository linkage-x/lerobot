#!/usr/bin/env bash
# Install a systemd unit that restores the BOX UDP target IP after Thor boots.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENSURE_SCRIPT="$REPO_ROOT/tools/thor/box_sdk/ensure_box_net.sh"
UNIT=/etc/systemd/system/thor-box-net.service
IFACE="${BOX_NET_IFACE:-enP2p1s0}"
ADDR="${BOX_NET_ADDR:-192.168.2.45/24}"
BOX_IP="${BOX_REMOTE_IP:-192.168.2.60}"
DISABLE_RPFILTER="${BOX_NET_DISABLE_RPFILTER:-0}"

if [[ "${EUID:-$(id -u)}" -ne 0 ]]; then
  exec sudo BOX_NET_IFACE="$IFACE" BOX_NET_ADDR="$ADDR" BOX_REMOTE_IP="$BOX_IP" BOX_NET_DISABLE_RPFILTER="$DISABLE_RPFILTER" "$0" "$@"
fi

if [[ ! -x "$ENSURE_SCRIPT" ]]; then
  echo "BOX net error: missing executable $ENSURE_SCRIPT" >&2
  exit 1
fi

cat > "$UNIT" <<UNIT_EOF
[Unit]
Description=Restore Thor BOX UDP target address
After=network-online.target
Wants=network-online.target

[Service]
Type=oneshot
Environment=BOX_NET_IFACE=$IFACE
Environment=BOX_NET_ADDR=$ADDR
Environment=BOX_REMOTE_IP=$BOX_IP
Environment=BOX_NET_DISABLE_RPFILTER=$DISABLE_RPFILTER
ExecStart=$ENSURE_SCRIPT
RemainAfterExit=yes

[Install]
WantedBy=multi-user.target
UNIT_EOF

systemctl daemon-reload
systemctl enable --now thor-box-net.service
systemctl status --no-pager --lines=20 thor-box-net.service
