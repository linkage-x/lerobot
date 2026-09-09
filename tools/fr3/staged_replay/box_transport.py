"""Addressed BOX transport. Opening/closing never changes mode or position.

Read-only means discovery, mode query and sensor reads, not a passive network
tap. This replay adapter never changes modes, zeros, calibrates or homes.
Mode 0 -> 1 physically closed the installed gripper during a previous test.
Consequently replay requires an already-enabled gripper and fails otherwise.
"""
import dataclasses
import importlib
import ipaddress
import os
from pathlib import Path
import socket
import sys
import time

from replay_timed_candidate import FreshBoxMeasurement, WIDTH_MAX


def select_physical_device(found, expected_id):
    """Deduplicate messages, not identities or network endpoints.

    The SDK can return both msg_type=2 and msg_type=3 for one physical board.
    Ignore only message kind, uptime and derived capability names. A changed
    IP/port or conflicting stable metadata for the expected ID still fails.
    """
    matches = [d for d in found if d.device_id == expected_id]
    observed = [dataclasses.asdict(d) for d in found]
    if not matches:
        raise RuntimeError(f'Expected physical BOX {expected_id} not uniquely discovered (no matching response); observed {observed}')
    signatures = set()
    for d in matches:
        try:
            address = ipaddress.IPv4Address(d.ip)
        except ipaddress.AddressValueError as exc:
            raise RuntimeError('Invalid BOX discovery address') from exc
        if address.is_multicast or address.is_unspecified or address.is_loopback:
            raise RuntimeError('Invalid BOX discovery endpoint')
        if d.data_port != 15000 or not d.capabilities & 1:
            raise RuntimeError('Unexpected BOX data port / missing gripper capability')
        signatures.add((int(d.device_id), str(address), int(d.data_port),
                        d.sn, d.device_type, d.fw_version, d.capabilities, d.proto_ver))
    if len(signatures) != 1:
        raise RuntimeError(f'Ambiguous physical BOX {expected_id}: conflicting endpoint or identity metadata; observed {observed}')
    # Prefer a query reply when present; neither message type changes the
    # selected device identity, endpoint or any actuator state.
    selected = max(matches, key=lambda d: (d.msg_type == 2, d.uptime_ms))
    details = dict(expected_device_id=expected_id, endpoint_ip=selected.ip,
                   raw_matching_messages=len(matches), unique_devices=1,
                   duplicate_messages_merged=len(matches)-1,
                   message_types=sorted(set(d.msg_type for d in matches)),
                   rule='Identical device ID, IP, port and stable metadata only')
    return selected, details


class BoxTransport:
    def __init__(self, sdk_dir, device_id, remote_ip, *, allow_commands=False):
        self.sdk_dir = Path(sdk_dir)
        self.device_id, self.ip = int(device_id), remote_ip
        self.allow_commands = allow_commands
        self.box = None
        self.keepalive = None
        self.mode = None
        self.discovery = []
        self.discovery_resolution = {}
        self.decoder = FreshBoxMeasurement(device_id)

    def open(self):
        # Refuse an existing SDK/recorder rather than sharing its UDP endpoint.
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as probe:
            probe.bind(("0.0.0.0", 15000))
        wheels = list((self.sdk_dir / 'python').glob('box_collection_sdk-*.whl'))
        if len(wheels) != 1:
            raise RuntimeError('Exactly one audited BOX wheel is required')
        os.environ['BOX_SDK_URDF'] = str(self.sdk_dir / 'share/monte_gripper.urdf')
        os.environ['BOX_SDK_CONFIG'] = str(self.sdk_dir / 'box_sdk.conf')
        sys.path.insert(0, str(wheels[0]))
        sdk = importlib.import_module('box_sdk')
        so = str(self.sdk_dir / 'lib/libbox_controller.so')
        found = sdk.discover(timeout=2., broadcast_addr=self.ip, so_path=so)
        # Firmware may acquire a DHCP address; the configured IP is only a hint.
        # Discover again, but select solely the explicitly configured identity,
        # never the first available / recording BOX.
        if not any(x.device_id == self.device_id for x in found):
            found = sdk.discover(timeout=3., broadcast_addr='255.255.255.255', so_path=so)
        self.discovery = [dataclasses.asdict(x) for x in found]
        selected, self.discovery_resolution = select_physical_device(found, self.device_id)
        self.ip = selected.ip
        self.box = sdk.Box(so_path=so)
        try:
            if self.box.start('0.0.0.0', 15000, self.ip, 15000) != 0:
                raise RuntimeError('BOX receive socket failed')
            if self.box.register_device(self.device_id, self.ip, 15000) != 0:
                raise RuntimeError('BOX address registration failed')
            rc, self.mode = self.box.get_mode(self.device_id, timeout_ms=500)
            if rc != 0:
                raise RuntimeError(f'Cannot read BOX mode, rc={rc}')
            self.keepalive = sdk.DiscoveryKeepAlive(broadcast_addr=self.ip, so_path=so)
        except BaseException:
            self.close()
            raise
        return self

    def snapshot(self):
        rc, cache = self.box.get_sensor_cache(self.device_id)
        if rc != 0 or not cache.valid:
            return None
        g = cache.data.gripper_data
        # A mode/discovery response can mark the aggregate cache valid before
        # the first gripper sensor packet. Wait for that first timestamp, but
        # reject a reset-to-zero after an established measurement stream.
        if int(g.timestamp) == 0 and self.decoder.last_timestamp is None:
            return None
        return dict(status=dict(active=True, device_id=int(cache.device_id)),
                    sensors=dict(box_gripper=dict(timestamp=int(g.timestamp), distance_m=float(g.distance))))

    def fresh(self):
        sample = self.snapshot()
        return None if sample is None else self.decoder.decode(sample)

    def wait_fresh(self, timeout=3.):
        end = time.monotonic() + timeout
        while time.monotonic() < end:
            sample = self.fresh()
            if sample is not None:
                return sample
            time.sleep(.005)
        raise TimeoutError('No advancing gripper measurement; cached data is not accepted')

    def send(self, width):
        if not self.allow_commands or self.mode != 1:
            raise RuntimeError('Metric-width command denied: not explicitly armed / BOX not in control mode')
        if not 0 <= width <= WIDTH_MAX:
            raise ValueError('Invalid gripper target')
        # rc=0 confirms SDK submission, not physical arrival. The independently
        # timestamped measured width feeds the native tracking watchdog.
        return self.box.set_clamp_pos(float(width), self.device_id) == 0

    def require_control_mode(self):
        """Read-only gate. Never energize the actuator as part of replay."""
        rc, mode = self.box.get_mode(self.device_id, timeout_ms=500)
        if rc != 0 or mode != 1:
            raise RuntimeError(f'BOX must already be in control mode (readback rc={rc}, mode={mode}); automatic mode switching is forbidden')
        self.mode = mode

    def close(self):
        if self.keepalive is not None:
            self.keepalive.close()
            self.keepalive = None
        if self.box is not None:
            try:
                self.box.stop()
            finally:
                self.box.close()
                self.box = None
