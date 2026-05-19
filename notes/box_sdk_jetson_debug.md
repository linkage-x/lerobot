# BOX SDK Jetson Debug Notes

Date: 2026-05-22

## Deployment

- Jetson host: `nvidia@192.168.1.44`
- SDK path: `/home/nvidia/box_collection_sdk/release_bundle_v2_arm/release_bundle`
- Source bundle: `/home/hanyu/下载/release_bundle_v2_arm.tar.gz`
- Python environment: `/home/nvidia/box_collection_sdk/release_bundle_v2_arm/release_bundle/.venv`

## Confirmed Network Facts

- Physical topology: Jetson Ethernet, LAN router, and gripper Ethernet are connected to the same PoE switch.
- Jetson physical Ethernet interface connected to the gripper: `enP2p1s0`
- Jetson interface address: `192.168.1.44/16`
- Link state: up, `1000Mb/s Full`
- Gripper IP confirmed by ARP: `192.168.2.60`
- Gripper MAC confirmed by ARP: `00:80:e1:00:00:00`
- Route to gripper: `192.168.2.60 dev enP2p1s0 src 192.168.1.44`
- ICMP ping to `192.168.2.60` does not reply, but ARP resolves successfully. Treat ping failure as non-conclusive.

## SDK Validation So Far

The SDK loads and can be instantiated on Jetson after installing dependencies and adding local compatibility links in the SDK `lib/` directory.

Passed:

- `from box_sdk import Box`
- `Box()` construction
- `start(bind_ip="0.0.0.0", bind_port=15000, remote_ip="192.168.2.60", remote_port=15000)` returns `0 ok`
- `set_mode(1)` returns `0 ok`
- `set_clamp_pos(0.004)` returns `0 ok`
- `set_mode(0)` returns `0 ok`
- `stop()` / `close()`

Not passing yet:

- `get_mode()` returns timeout
- `get_sensor_cache()` returns `4 no cached sensor data`

Current hypothesis:

- The gripper IP is correct.
- The SDK can send commands.
- Because Jetson, router, and gripper share one PoE switch and `enP2p1s0` has a `/16` address, traffic to `192.168.2.60` should be direct L2 traffic on `enP2p1s0`, not routed through the LAN router.
- Jetson is not receiving valid UDP sensor/status responses on port `15000`, or the board is not sending them to Jetson's expected IP/port.

## Packet Capture Finding

`tcpdump` on `enP2p1s0` and `any` while running SDK calls showed ARP traffic from the shared LAN/PoE-switch segment, but no UDP packets matching port `15000`. `ss` showed the Python process bound to `0.0.0.0:15000` during `box.start(...)`.

This means the next root-cause step is to confirm at syscall level whether the SDK actually calls `sendto()` and what destination address/port it uses.

## Re-run Commands

```bash
ssh nvidia@192.168.1.44
cd /home/nvidia/box_collection_sdk/release_bundle_v2_arm/release_bundle
. .venv/bin/activate
. ./setup_env.sh
python -u demo.py 15000 15000
```

