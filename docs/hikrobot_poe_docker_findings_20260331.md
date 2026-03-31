# Hikrobot PoE Docker Findings 2026-03-31

## Summary

- Host-side MVS enumeration works for Hikrobot PoE cameras on the physical GigE NIC.
- The helper script `tools/hikrobot/list_hikrobot_gige_cameras.py` can enumerate GigE cameras without opening them and can filter by `net_export` or local interface name.
- Direct Hikrobot MVS access from the current Docker runtime does not work for PoE cameras, even though the container can start and load the SDK.
- The current Docker daemon is running in `rootless` mode. In this mode, the container does not actually see the host PoE NIC/IP that the cameras are attached to, so MVS GigE discovery inside the container returns zero devices.

## Evidence

### Host works

Host-side enumeration with the physical NIC IP succeeds:

```bash
python tools/hikrobot/list_hikrobot_gige_cameras.py --net-export 192.168.0.23
```

Observed result:

- `DA9342673` at `192.168.0.2`
- `DA9342611` at `192.168.0.3`
- both reported `net_export: 192.168.0.23`

### Container does not see the host NIC

Inside the container launched by:

```bash
docker compose -f docker/docker-compose.yml run --rm -T lerobot-user ...
```

the visible IPs were:

- `10.0.2.100`
- `172.18.0.1`
- `172.17.0.1`

The host PoE NIC IP `192.168.0.23` was not visible inside that container.

As a result:

- `lerobot-find-cameras hikrobot` returned zero Hikrobot cameras
- `python tools/hikrobot/list_hikrobot_gige_cameras.py --net-export 192.168.0.23` also returned zero devices inside the container

## Root Cause

The failure is not just a LeRobot enumeration bug.

The main blocker is runtime topology:

- Hikrobot GigE/PoE discovery depends on the real host NIC and broadcast/subnet visibility.
- In the current `rootless Docker` setup, the container is attached to a virtualized network stack instead of the physical PoE NIC.
- Therefore MVS inside the container cannot enumerate or open the GigE cameras that are reachable from the host.

## Code Changes in This Round

### 1. Host-side GigE enumeration helper

Added:

- `tools/hikrobot/list_hikrobot_gige_cameras.py`

This script:

- enumerates GigE devices without opening them
- prints `serial`, `current_ip`, and `net_export`
- supports `--net-export <ip>`
- supports `--interface <ifname>`

### 2. Hikrobot camera discovery fixes

Updated:

- `src/lerobot/cameras/hikrobot/camera_hikrobot.py`

Changes:

- ensure MVS SDK initialize/finalize is handled centrally
- include `MV_GENTL_GIGE_DEVICE` in GigE discovery
- expose `net_export` in extracted Hikrobot metadata

### 3. `lerobot-find-cameras` Hikrobot config fix

Updated:

- `src/lerobot/scripts/lerobot_find_cameras.py`

Change:

- when a Hikrobot camera is instantiated from discovered metadata, preserve the discovered `transport_layer` instead of falling back to the config default (`usb`)

## Practical Guidance

### What works now

- Host-side enumeration
- Host-side Hikrobot recording scripts, for example:

```bash
uv run python tools/hikrobot/hikrobot_record_test.py --serial <serial> --transport-layer gige
```

### What does not work in the current environment

- direct Hikrobot MVS discovery/open inside the current rootless Docker runtime

### Recommended paths

Choose one of these:

1. Run Hikrobot discovery and recording on the host.
2. If containerized recording is required, move to a Docker runtime that really exposes the physical NIC to the container.
3. Keep Hikrobot/MVS on the host and stream frames into the container using a transport such as ZMQ, then record in-container via `ZMQCamera`.
