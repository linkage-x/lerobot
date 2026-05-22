# BOX 采集板 SDK — Jetson Troubleshooting

Bring-up notes for the vendored BOX collection SDK on the Thor host. This
file captures (a) the verification path that proves the LeRobot wrapper is
correctly wired and (b) the diagnostic ladder for the open sensor-stream
issue that the supplier is currently investigating.

## On-Thor verification status (2026-05-22)

Hardware / network topology validated by the operator:

- Jetson, router, and 采集板 all share one PoE switch.
- Jetson Ethernet `enP2p1s0` reconfigured to `192.168.2.44/24` (same
  subnet as the 采集板 at `192.168.2.60`, MAC `00:80:e1:00:00:00`).
- `tcpdump -i enP2p1s0 -nn udp` confirms outbound packets
  `192.168.2.44:15000 -> 192.168.2.60:15000`.
- TX-checksum offload disabled; UDP checksums are valid on the wire.
- The 采集板 ACKs commands (`set_mode`, `set_clamp_pos` return rc=0 and
  the gripper actually moves) — proving the downlink is intact.
- No inbound UDP from `192.168.2.60`/`00:80:e1:00:00:00` reaches the host.
  Supplier confirmed downlink is healthy and suggested the ARM-side
  gateway/host stack is the next thing to inspect.

LeRobot wrapper verification on the same host:

| Step | Result |
| --- | --- |
| `pip install tools/thor/box_sdk/python/box_collection_sdk-*.whl` | ok |
| `bash tools/thor/box_sdk/install_compat_links.sh` | links `libtinyxml2.so.9 -> .so.10`, `liburdfdom_model.so.3.0 -> .so.4.0` |
| `source tools/thor/box_sdk/setup_env.sh` | exports `LD_LIBRARY_PATH`, `BOX_SDK_URDF` |
| `from box_sdk import Box; Box(); Box.close()` | ok |
| `BoxClient(cfg).start()` | returns `True`; SDK logs `build_time=May 21 2026 11:24:48, commit=e2ea1a3` |
| `BoxClient.read()` after 2 s | `{valid: False, sensors: {}}` |
| `BoxClient.detect()` | `[]` (same upstream issue) |
| `BoxClient.stop()` | clean teardown |

So the integration is correctly built end-to-end; the empty sensor cache
is purely the upstream "no inbound UDP" symptom.

## Diagnostic ladder for the missing sensor stream

Walk these in order on Thor with the gripper powered and `demo.py` (or
`BoxClient`) actively polling. Stop at the first one that produces signal.

1. **Capture everything from the 采集板, not just UDP/15000.**

    ```bash
    sudo tcpdump -i enP2p1s0 -nn -X \
      'host 192.168.2.60 or ether host 00:80:e1:00:00:00'
    ```

    The sensor stream may use a different port or even multicast. If
    *anything* arrives from the 采集板, note the dst port / IP.

2. **Check multicast group membership.** If `tcpdump` shows multicast (`224.x`
   or `239.x`) traffic but the SDK doesn't see it, the kernel hasn't joined
   the group:

    ```bash
    ip maddr show enP2p1s0
    cat /proc/net/igmp
    ```

3. **Reverse-path filter.** With `192.168.2.0/24` on `enP2p1s0` and any
   other interface (Wi-Fi, USB-Eth) carrying a default route, `rp_filter`
   can silently drop inbound from 192.168.2.60:

    ```bash
    sudo sysctl net.ipv4.conf.enP2p1s0.rp_filter
    sudo sysctl -w net.ipv4.conf.enP2p1s0.rp_filter=0
    sudo sysctl -w net.ipv4.conf.all.rp_filter=0
    ```

4. **Disable RX-side offload too**, not just TX. JetPack's `nv_eqos`
   firmware revs have been known to drop fragmented or LRO-aggregated
   inbound UDP:

    ```bash
    sudo ethtool -K enP2p1s0 rx-checksum off gro off gso off tso off lro off
    sudo ethtool -K enP2p1s0 ntuple off
    ```

5. **Firewall.** Even a default-deny `ufw` lets outbound ACKs through but
   silently swallows inbound UDP/15000:

    ```bash
    sudo nft list ruleset
    sudo iptables -L INPUT -nv
    sudo ufw status verbose
    ```

6. **Bind-address mismatch.** SDK binds `0.0.0.0:15000`. If the host IP
   changed (e.g., `192.168.1.44/16` -> `192.168.2.44/24`) the 采集板 may
   have cached the prior reply address. Power-cycle the gripper and reissue
   `set_mode` to force re-learning, then re-capture.

7. **Strace at the syscall layer.** Definitive answer for "is the kernel
   even delivering packets to the SDK socket?":

    ```bash
    sudo strace -f -e trace=network -p $(pgrep -f box_sdk) 2>&1 \
      | grep -E 'recvfrom|setsockopt'
    ```

    - `recvfrom` returning packets => SDK parser rejected them (firmware /
      TLV-type mismatch — escalate to vendor with the dump).
    - `recvfrom` returning zero bytes / `EAGAIN` only => packets are not
      reaching the socket; root cause is in steps 1–5.

8. **Install a raw packet observer** to peek at what the SDK actually
   ingests:

    ```python
    def obs(user, pkt):
        print(f"rx idx={pkt.index} type=0x{pkt.type:x} len={pkt.length}")
    box.set_packet_observer(obs)
    ```

    If `set_packet_observer` fires but `get_sensor_cache` keeps returning
    rc=4, the cache path rejects the packet `type`. That's the cleanest
    evidence to send back: vendor's `e2ea1a3` build may need a matching
    firmware on the 采集板 MCU that emits the sensor TLV it's looking for.

## What the wrapper handles already

- `BoxClient.start()` always calls `box.set_mode(<startup_mode>)` so the
  command path goes through every session, exposing any firmware-side
  rejection in logs.
- `BoxClient.detect()` only reports sensors that have published at least
  once *and* whose last sample is within `stale_threshold_s`. Once the
  upstream issue is resolved, the gateway's `Box devices:` line will
  reflect the real subset attached (the rig may have 0, 1, or 2 Paxini
  pads on at a time).
- Module imports without the wheel installed: the gateway can still come
  up on dev hosts; only `BoxClient.start()` is a no-op.

## Quick re-run cookbook

```bash
ssh nvidia@192.168.1.44
cd ~/lerobot

# One-time per host:
sudo apt install -y libeigen3-dev liburdfdom-dev
bash tools/thor/box_sdk/install_compat_links.sh

# Per shell:
. ~/box_collection_sdk/release_bundle_v2_arm/release_bundle/.venv/bin/activate
. tools/thor/box_sdk/setup_env.sh
pip install --quiet --force-reinstall \
  tools/thor/box_sdk/python/box_collection_sdk-*.whl

# Live wrapper smoke test:
PYTHONPATH=src:. python -m tools.thor.box_sdk.demo 15000 15000 192.168.2.60

# Gateway smoke test (without launching the actual recorder process):
PYTHONPATH=src:. python -c "
from pathlib import Path
from tools.data_collection_gui import gateway
state = gateway.make_state(Path.cwd(), gateway.DEFAULT_CONFIG_PATH)
snap = gateway._snapshot(state)
print('repoId:', snap['configSummary']['repoId'])
print('box:', [d['id'] for d in snap['devices'] if d['kind']=='box_collection'])
"
```
