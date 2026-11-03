"""Unit tests for the LeRobot-side wrapper around the vendored BOX SDK.

The real wheel is ARM-only and not importable on the dev host, so we
inject a tiny in-memory stub that mimics the C structures (`SensorCache`,
`AllSensor`, ...) the wrapper expects. The wrapper code under test never
touches `libbox_controller.so` directly -- it goes through the wheel's
`box_sdk.Box` class -- so a faithful Python stand-in is enough.
"""

from __future__ import annotations

import logging
import sys
import time
import types
from dataclasses import dataclass, field
from typing import Any

import pytest

from tools.thor.box_sdk import box_client


# --- helpers ----------------------------------------------------------------


@dataclass
class _Gripper:
    timestamp: int = 0
    distance: float = 0.0


@dataclass
class _Imu:
    timestamp: int = 0
    acc: tuple = (0.0, 0.0, 0.0)
    gyr: tuple = (0.0, 0.0, 0.0)
    roll: float = 0.0
    pitch: float = 0.0
    yaw: float = 0.0
    quat: tuple = (1.0, 0.0, 0.0, 0.0)


@dataclass
class _Trigger:
    timestamp: int = 0
    depth_percent: float = 0.0  # v4.1: effective (zeroed) squeeze depth, 0..100
    raw_depth_percent: float = 0.0
    locked: int = 0


@dataclass
class _SixD:
    timestamp: int = 0
    data: tuple = (0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class _TouchForce:
    fx: int = 0
    fy: int = 0
    fz: int = 0


@dataclass
class _Touch:
    """Stand-in for v4.1 ``TouchSensorSample``: ``forces`` is exactly point_count long."""
    timestamp: int = 0
    total_force: _TouchForce = field(default_factory=_TouchForce)
    forces: tuple = field(default_factory=lambda: tuple(_TouchForce() for _ in range(9)))

    @property
    def point_count(self) -> int:
        return len(self.forces)


def _terminals(left: _Touch | None = None, right: _Touch | None = None, touch_device_id: int = 1):
    """Build v4.1 ``SensorCache.touch``: per terminal ``{touch_device_id: sample}``."""
    return (
        {touch_device_id: left} if left is not None else {},
        {touch_device_id: right} if right is not None else {},
    )


@dataclass
class _LinkStatsRecord:
    """Stand-in for box_sdk's LinkStats; only the fields the host reads."""
    tlv_type: int = 0
    samples_total: int = 0
    device_id: int = 0


@dataclass
class _AllSensor:
    gripper_data: _Gripper = field(default_factory=_Gripper)
    imu_data: _Imu = field(default_factory=_Imu)
    trigger_data: _Trigger = field(default_factory=_Trigger)
    six_d_force_data: _SixD = field(default_factory=_SixD)
    filtered: _SixD = field(default_factory=_SixD)
    filtered_gravity: _SixD = field(default_factory=_SixD)
    filtered_no_gravity: _SixD = field(default_factory=_SixD)
    six_d_force_data_filter: _SixD = field(default_factory=_SixD)
    gripper_speed: float | None = None


@dataclass
class _SensorCache:
    valid: int = 1
    liwp_index: int = 0
    liwp_timestemp: int = 0  # SDK typo preserved
    device_id: int = 0  # the device that produced this frame (v4 attribution)
    data: _AllSensor = field(default_factory=_AllSensor)
    # v4.1: one dict per touch terminal (0 = left, 1 = right), keyed by
    # touch_device_id. Default: nothing cached on either terminal.
    touch: tuple = field(default_factory=lambda: ({}, {}))


@dataclass
class _FakeDiscovered:
    """Stand-in for ``box_sdk.DiscoveredDevice`` (v3 broadcast enumeration)."""

    device_id: int
    sn: str = ""
    ip: str = "192.168.2.60"
    data_port: int = 15000
    fw_version: int = 0
    capabilities: int = 0


class _FakeKeepAlive:
    """Stand-in for ``box_sdk.DiscoveryKeepAlive``."""

    instances: list["_FakeKeepAlive"] = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        self.closed = False
        self.instances.append(self)

    def send_req(self, *_, **__):
        return 0

    def close(self):
        self.closed = True


class _FakeBox:
    """In-memory stand-in for the v3 multi-device ``box_sdk.Box``."""

    def __init__(self, *_, **__):
        # snaps -> the most-recent / default-device cache (device_id=None path);
        # snaps_by_id -> per-device caches keyed by device_id.
        self.snaps: list[_SensorCache] = []
        self.snaps_by_id: dict[int, _SensorCache] = {}
        self.mode: dict[int | None, int] = {}
        self.clamp_cmds: list[tuple] = []
        self.trigger_zeroed: list = []
        self.started = False
        self.stopped = False
        self.registered: list[tuple] = []
        self.btn_key_action_observer = None
        self.error_observer = None
        self.device_state_observer = None
        self.stats_observer = None
        self.link_stats_observer = None

    # --- protocol the v3 Box exposes to BoxClient ----------------------
    def start(self, bind_ip, bind_port, remote_ip, remote_port):
        self.started = True
        self.bind = (bind_ip, bind_port)
        self.remote = (remote_ip, remote_port)
        return 0

    def stop(self):
        self.stopped = True

    def close(self):
        pass

    def set_mode(self, mode, device_id=None):
        self.mode[device_id] = mode
        return 0

    def set_clamp_pos(self, pos_m, device_id=None):
        self.clamp_cmds.append((float(pos_m), device_id))
        return 0

    def set_trigger_zero(self, device_id=None):
        self.trigger_zeroed.append(device_id)
        return 0

    def register_device(self, device_id, ip, port=15000):
        self.registered.append((int(device_id), ip, port))
        return 0

    def get_device_ids(self, *_):
        ids = set(self.snaps_by_id) | {r[0] for r in self.registered}
        if self.snaps:
            ids.add(1)  # synthetic id for the most-recent-device path
        return sorted(ids)

    def get_known_device_ids(self, *_):
        return sorted({r[0] for r in self.registered})

    def get_sensor_cache(self, device_id=None):
        if device_id is not None and device_id in self.snaps_by_id:
            return 0, self.snaps_by_id[device_id]
        if self.snaps:
            return 0, self.snaps[0]
        return 4, None  # v4.1: a failed read hands back no snapshot

    def get_system_status(self, device_id, timeout_ms=200):
        self.system_status_calls = getattr(self, "system_status_calls", []) + [(device_id, timeout_ms)]
        return 0, types.SimpleNamespace(status_code=0b100011, error_code=(1 << 6) | (1 << 12))

    def set_btn_key_action_observer(self, callback):
        self.btn_key_action_observer = callback

    def set_error_observer(self, callback):
        self.error_observer = callback

    def set_device_state_observer(self, callback):
        self.device_state_observer = callback

    def set_stats_observer(self, callback):
        self.stats_observer = callback

    def set_link_stats_observer(self, callback):
        self.link_stats_observer = callback

    def err_str(self, rc):
        return "no cached sensor data" if rc == 4 else f"rc={rc}"


@pytest.fixture
def fake_box_module(monkeypatch):
    _FakeKeepAlive.instances = []
    module = types.ModuleType("box_sdk")
    module.Box = _FakeBox
    module.discover = lambda **kw: []  # default: no broadcast discovery
    module.DiscoveryKeepAlive = _FakeKeepAlive
    monkeypatch.setitem(sys.modules, "box_sdk", module)
    return module


# --- tests ------------------------------------------------------------------


def test_from_yaml_dict_applies_defaults_and_rejects_unknown_devices():
    raw = {
        "enabled": True,
        "remote_ip": "10.0.0.1",
        "expected_devices": ["box_gripper", "box_imu"],
    }
    cfg = box_client.from_yaml_dict(raw)
    assert cfg.enabled
    assert cfg.bind_port == 15000  # fixed by vendor spec
    assert cfg.remote_ip == "10.0.0.1"
    assert cfg.expected_devices == ["box_gripper", "box_imu"]

    with pytest.raises(ValueError):
        box_client.from_yaml_dict({"expected_devices": ["box_unknown"]})


def test_from_yaml_dict_handles_disabled_and_missing():
    assert box_client.from_yaml_dict(None).enabled is False
    assert box_client.from_yaml_dict({"enabled": False}).enabled is False


def test_decode_sensor_cache_filters_zero_timestamp_sensors():
    snap = _SensorCache(
        valid=1,
        liwp_index=42,
        liwp_timestemp=1234,
        data=_AllSensor(
            gripper_data=_Gripper(timestamp=10, distance=0.04),
            gripper_speed=0.0125,
            imu_data=_Imu(),  # timestamp=0 -> dropped
            trigger_data=_Trigger(
                timestamp=20, depth_percent=42.0, raw_depth_percent=45.5, locked=1
            ),
            six_d_force_data=_SixD(timestamp=30, data=(1, 2, 3, 4, 5, 6)),
            six_d_force_data_filter=_SixD(timestamp=31, data=(10, 20, 30, 40, 50, 60)),
        ),
        touch=_terminals(
            _Touch(
                timestamp=40,
                total_force=_TouchForce(fx=7, fy=-8, fz=9),
                forces=tuple(
                    _TouchForce(fx=i % 7, fy=-(i % 5), fz=(i % 11))
                    for i in range(239)
                ),
            ),
            _Touch(timestamp=0),  # cached but never stamped -> dropped
        ),
    )
    out = box_client.decode_sensor_cache(snap)
    assert out["valid"] is True
    assert out["liwp_index"] == 42
    assert out["liwp_timestamp"] == 1234
    assert set(out["sensors"]) == {
        "box_gripper", "box_trigger", "box_six_d_force", "box_touch_left",
    }
    assert out["sensors"]["box_gripper"]["distance_m"] == pytest.approx(0.04)
    assert out["sensors"]["box_gripper"]["velocity_m_s"] == pytest.approx(0.0125)
    force = out["sensors"]["box_six_d_force"]
    assert force["timestamp"] == 31
    assert force["source"] == "filtered"
    assert force["fxyz_mxyz"] == [10, 20, 30, 40, 50, 60]
    assert force["fxyz_mxyz_raw"] == [1, 2, 3, 4, 5, 6]
    trigger = out["sensors"]["box_trigger"]
    assert trigger == {
        "timestamp": 20, "travel_pct": 42.0, "raw_travel_pct": 45.5, "locked": True,
    }
    touch = out["sensors"]["box_touch_left"]
    assert touch["total_force_0p1N"] == [7, -8, 9]
    assert len(touch["fx_0p1N"]) == 239 == len(touch["fy_0p1N"]) == len(touch["fz_0p1N"])
    assert touch["model"] == box_client.TOUCH_MODEL_PAXINI_L5325
    assert touch["touch_device_id"] == 1


def test_decode_sensor_cache_falls_back_to_raw_six_d_when_filter_absent():
    snap = _SensorCache(
        data=_AllSensor(
            six_d_force_data=_SixD(timestamp=30, data=(1, 2, 3, 4, 5, 6)),
        ),
    )

    out = box_client.decode_sensor_cache(snap)
    timestamps = box_client._decode_sensor_timestamps(snap)

    assert out["sensors"]["box_six_d_force"]["source"] == "raw"
    assert out["sensors"]["box_six_d_force"]["timestamp"] == 30
    assert timestamps["box_six_d_force"] == 30


def test_decode_sensor_cache_exposes_v4_gravity_compensated_force():
    snap = _SensorCache(
        data=_AllSensor(
            six_d_force_data=_SixD(timestamp=30, data=(1, 2, 3, 4, 5, 6)),
            filtered=_SixD(timestamp=31, data=(10, 20, 30, 40, 50, 60)),
            filtered_gravity=_SixD(timestamp=31, data=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6)),
            filtered_no_gravity=_SixD(timestamp=31, data=(9, 18, 27, 36, 45, 54)),
        ),
    )

    out = box_client.decode_sensor_cache(snap)
    force = out["sensors"]["box_six_d_force"]

    assert force["source"] == "filtered"
    assert force["fxyz_mxyz"] == [10, 20, 30, 40, 50, 60]
    assert force["fxyz_mxyz_raw"] == [1, 2, 3, 4, 5, 6]
    assert force["fxyz_mxyz_filtered"] == [10, 20, 30, 40, 50, 60]
    assert force["fxyz_mxyz_gravity"] == pytest.approx(
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6]
    )
    assert force["fxyz_mxyz_no_gravity"] == [9, 18, 27, 36, 45, 54]
    assert force["fxyz_mxyz_gravity_compensated"] == [9, 18, 27, 36, 45, 54]


def test_decode_changed_sensors_skips_unchanged_touch_payload(monkeypatch):
    snap = _SensorCache(
        data=_AllSensor(
            six_d_force_data=_SixD(timestamp=30, data=(1, 2, 3, 4, 5, 6)),
            six_d_force_data_filter=_SixD(timestamp=31, data=(10, 20, 30, 40, 50, 60)),
        ),
        touch=_terminals(_Touch(timestamp=40), _Touch(timestamp=41)),
    )

    def fail_touch_decode(*_args):
        raise AssertionError("unchanged touch payload should not be decoded")

    monkeypatch.setattr(box_client, "_decode_touch", fail_touch_decode)
    out = box_client._decode_changed_sensors(snap, {"box_six_d_force"})

    assert set(out["sensors"]) == {"box_six_d_force"}
    assert out["sensors"]["box_six_d_force"]["timestamp"] == 31


def test_decode_sensor_cache_reads_touch_per_terminal():
    # v4.1 SensorCache.touch[0]/[1] are the left/right terminals.
    array_left = _Touch(
        timestamp=101,
        forces=tuple(_TouchForce(fx=1, fy=2, fz=3) for _ in range(239)),
    )
    array_right = _Touch(
        timestamp=202,
        forces=tuple(_TouchForce(fx=-1, fy=-2, fz=4) for _ in range(239)),
    )
    snap = _SensorCache(touch=_terminals(array_left, array_right))

    out = box_client.decode_sensor_cache(snap)
    timestamps = box_client._decode_sensor_timestamps(snap)

    assert out["sensors"]["box_touch_left"]["timestamp"] == 101
    assert out["sensors"]["box_touch_left"]["fz_0p1N"][0] == 3
    assert out["sensors"]["box_touch_right"]["timestamp"] == 202
    assert out["sensors"]["box_touch_right"]["fz_0p1N"][0] == 4
    assert timestamps["box_touch_left"] == 101
    assert timestamps["box_touch_right"] == 202


def test_decode_sensor_cache_keeps_gripper_distance_when_timestamp_is_zero():
    snap = _SensorCache(
        valid=1,
        data=_AllSensor(gripper_data=_Gripper(timestamp=0, distance=0.09785686433315277)),
    )

    out = box_client.decode_sensor_cache(snap)

    assert out["sensors"]["box_gripper"] == {
        "timestamp": 0,
        "distance_m": pytest.approx(0.09785686433315277),
    }


def test_box_client_start_stop_pulls_snapshot_and_marks_detected(fake_box_module):
    cfg = box_client.BoxClientConfig(
        enabled=True, poll_interval_s=0.01, stale_threshold_s=5.0,
    )
    client = box_client.BoxClient(cfg)

    # Pre-load the fake Box with one good snapshot before start.
    fake_module = fake_box_module
    fake_module.Box.__init__ = _FakeBox.__init__  # keep reference for typing

    def _factory(*a, **kw):
        b = _FakeBox()
        b.snaps.append(
            _SensorCache(
                valid=1, liwp_index=1, liwp_timestemp=1,
                data=_AllSensor(
                    gripper_data=_Gripper(timestamp=1, distance=0.05),
                    imu_data=_Imu(timestamp=1, acc=(0.0, 0.0, 9.8)),
                ),
            ),
        )
        return b

    fake_module.Box = _factory  # type: ignore[assignment]

    assert client.start() is True
    # Wait briefly for the poll thread to pick up the snapshot.
    import time as _t
    _t.sleep(0.05)
    snap = client.read()
    assert snap["valid"]
    assert "box_gripper" in snap["sensors"]
    assert "box_imu" in snap["sensors"]
    detected = client.detect()
    assert "box_gripper" in detected
    assert "box_imu" in detected
    # Sensors absent from the snapshot should not appear in detect().
    assert "box_six_d_force" not in detected
    client.stop()
    assert client.is_active() is False


def test_box_client_forwards_control_calls_to_the_addressed_device(fake_box_module):
    # The FR3 Corenetic gripper driver drives set_mode()/set_clamp_pos() on the
    # BoxClient itself, so they must exist here -- not only on the internal
    # _DeviceHandle -- and must carry the resolved device_id (v4 SDK addresses
    # every command by id).
    created: list[_FakeBox] = []

    def _factory(*a, **kw):
        b = _FakeBox()
        b.snaps.append(
            _SensorCache(
                valid=1,
                data=_AllSensor(gripper_data=_Gripper(timestamp=1, distance=0.05)),
            ),
        )
        created.append(b)
        return b

    fake_box_module.Box = _factory  # type: ignore[assignment]

    cfg = box_client.BoxClientConfig(enabled=True, poll_interval_s=0.01)
    client = box_client.BoxClient(cfg)
    assert client.start() is True
    try:
        assert client.set_mode(1) == 0
        assert client.set_clamp_pos(0.042) == 0
        assert client.set_trigger_zero() == 0
    finally:
        client.stop()

    box = created[-1]
    # device_id 1 is the synthetic id _FakeBox reports for the most-recent path.
    assert box.mode == {1: 1}
    assert box.clamp_cmds == [(0.042, 1)]
    assert box.trigger_zeroed == [1]


def test_box_client_control_calls_raise_before_start(fake_box_module):
    client = box_client.BoxClient(box_client.BoxClientConfig(enabled=True))
    with pytest.raises(RuntimeError, match="not started"):
        client.set_clamp_pos(0.01)


def _force_box_factory(
    force=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0),
    *,
    with_cali,
    with_origin=False,
):
    """Fake Box that publishes a 6D force sample; optionally exposes cali()."""
    calls: list[str] = []

    class _Box(_FakeBox):
        if with_cali:
            def cali_6d_force_sensor(self, device_id=None):  # mirrors the v3 SDK
                calls.append("software")
                return 0

        if with_origin:
            def cali_6d_force_sensor_origin(self, device_id=None):
                calls.append("origin")
                return 0

    def _factory(*a, **kw):
        b = _Box()
        b.snaps.append(
            _SensorCache(
                valid=1,
                data=_AllSensor(six_d_force_data=_SixD(timestamp=7, data=tuple(force))),
            ),
        )
        return b

    return _factory, calls


def test_box_client_gets_v4_gravity_compensated_force(fake_box_module):
    def _factory(*a, **kw):
        b = _FakeBox()
        b.snaps.append(
            _SensorCache(
                valid=1,
                data=_AllSensor(
                    six_d_force_data=_SixD(timestamp=30, data=(1, 2, 3, 4, 5, 6)),
                    filtered=_SixD(timestamp=31, data=(10, 20, 30, 40, 50, 60)),
                    filtered_no_gravity=_SixD(timestamp=31, data=(9, 18, 27, 36, 45, 54)),
                ),
            ),
        )
        return b

    fake_box_module.Box = _factory  # type: ignore[assignment]
    cfg = box_client.BoxClientConfig(enabled=True, poll_interval_s=0.01)
    client = box_client.BoxClient(cfg)
    assert client.start() is True
    import time as _t
    _t.sleep(0.05)
    try:
        assert client.get_six_d_force() == [10, 20, 30, 40, 50, 60]
        assert client.get_six_d_force(gravity_compensated=True) == [
            9, 18, 27, 36, 45, 54,
        ]
        assert client.get_six_d_force_gravity_compensated() == [9, 18, 27, 36, 45, 54]
    finally:
        client.stop()


def test_box_client_registers_v4_observers_and_reports_monitor(fake_box_module):
    boxes: list[_FakeBox] = []

    def _factory(*a, **kw):
        b = _FakeBox()
        boxes.append(b)
        return b

    fake_box_module.Box = _factory  # type: ignore[assignment]
    cfg = box_client.BoxClientConfig(enabled=True, poll_interval_s=0.01)
    client = box_client.BoxClient(cfg)
    seen_buttons: list[dict[str, Any]] = []
    client.set_button_callback(seen_buttons.append)
    assert client.start() is True
    try:
        box = boxes[-1]
        assert callable(box.btn_key_action_observer)
        assert callable(box.stats_observer)
        assert callable(box.link_stats_observer)

        box.btn_key_action_observer(7, types.SimpleNamespace(timestamp=123, event=2))
        box.device_state_observer(7, True)
        box.stats_observer(
            types.SimpleNamespace(
                rx_packets=11, rx_bytes=22, decode_errors=1, tlv_errors=2,
                events_enqueued=33, events_dropped=4, queue_depth=5,
                online_devices=1, lost_packets=6, reordered_packets=7,
            )
        )
        box.link_stats_observer([
            types.SimpleNamespace(
                device_id=7, tlv_type=0x0007, measured_hz=480.0, nominal_hz=480.0,
                host_jitter_ewma_ms=0.1, host_jitter_max_ms=0.2, device_hz=479.5,
                jitter_ewma_ms=0.3, jitter_max_ms=0.4, samples_total=100,
                lost_total=1, lost_window=0, reordered_total=0, stalled=0, online=1,
            )
        ])

        monitor = client.read()["status"]["monitor"]
        assert seen_buttons and seen_buttons[0]["event"] == 2
        assert monitor["latest_button_event"]["timestamp"] == 123
        assert monitor["device_online"] == {"7": True}
        assert monitor["latest_stats"]["rx_packets"] == 11
        assert monitor["latest_stats"]["events_dropped"] == 4
        assert monitor["latest_link_stats"][0]["device_hz"] == pytest.approx(479.5)
        assert monitor["latest_link_stats"][0]["online"] is True
    finally:
        client.stop()


def test_calibrate_six_d_force_gracefully_handles_missing_sdk_method(fake_box_module):
    # The currently shipped wheel (0.1.0) has no cali_6d_force_sensor(); the
    # wrapper must report ok=False with an explanatory error, never raise.
    fake_box_module.Box, _ = _force_box_factory(with_cali=False)
    cfg = box_client.BoxClientConfig(enabled=True, poll_interval_s=0.01)
    client = box_client.BoxClient(cfg)
    assert client.start() is True
    try:
        result = client.calibrate_six_d_force()
    finally:
        client.stop()

    assert result["ok"] is False
    assert result["rc"] is None
    assert "cali_6d_force_sensor" in result["error"]


def test_calibrate_six_d_force_invokes_sdk_and_reports_before_after(fake_box_module):
    fake_box_module.Box, calls = _force_box_factory(with_cali=True)
    cfg = box_client.BoxClientConfig(enabled=True, poll_interval_s=0.01)
    client = box_client.BoxClient(cfg)
    assert client.start() is True
    import time as _t
    _t.sleep(0.05)  # let the poll loop populate _latest for the before-read
    try:
        result = client.calibrate_six_d_force()
    finally:
        client.stop()

    assert calls == ["software"]
    assert result["ok"] is True
    assert result["rc"] == 0
    assert result["method"] == "cali_6d_force_sensor"
    assert result["error"] is None
    assert result["before"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert result["after"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


def test_calibrate_six_d_force_origin_invokes_origin_sdk_method(fake_box_module):
    fake_box_module.Box, calls = _force_box_factory(with_cali=True, with_origin=True)
    cfg = box_client.BoxClientConfig(enabled=True, poll_interval_s=0.01)
    client = box_client.BoxClient(cfg)
    assert client.start() is True
    import time as _t
    _t.sleep(0.05)
    try:
        result = client.calibrate_six_d_force(origin=True)
    finally:
        client.stop()

    assert calls == ["origin"]
    assert result["ok"] is True
    assert result["method"] == "cali_6d_force_sensor_origin"


def test_box_pool_calibrate_six_d_force_targets_force_box(fake_box_module):
    fake_box_module.Box, _ = _force_box_factory(with_cali=True)
    fleet = box_client.fleet_from_yaml_dict({"enabled": True, "poll_interval_s": 0.01})
    pool = box_client.BoxPool(fleet)
    assert pool.start() is True
    import time as _t
    _t.sleep(0.05)
    try:
        results = pool.calibrate_six_d_force()
    finally:
        pool.stop()

    assert len(results) == 1
    assert results[0]["box_id"] == ""
    assert results[0]["ok"] is True
    assert results[0]["rc"] == 0


def test_box_client_marks_gripper_seen_from_distance_without_timestamp(fake_box_module):
    cfg = box_client.BoxClientConfig(
        enabled=True,
        poll_interval_s=0.01,
        stale_threshold_s=5.0,
        expected_devices=["box_gripper"],
    )
    client = box_client.BoxClient(cfg)

    fake_module = fake_box_module

    def _factory(*a, **kw):
        b = _FakeBox()
        b.snaps.append(
            _SensorCache(
                valid=1,
                data=_AllSensor(
                    gripper_data=_Gripper(timestamp=0, distance=0.09785686433315277),
                ),
            ),
        )
        return b

    fake_module.Box = _factory  # type: ignore[assignment]

    assert client.start() is True
    import time as _t
    _t.sleep(0.05)

    snap = client.read()
    assert snap["sensors"]["box_gripper"]["distance_m"] == pytest.approx(0.09785686433315277)
    assert snap["status"]["sensor_status"]["box_gripper"]["seen"] is True
    assert snap["status"]["sensor_status"]["box_gripper"]["last_timestamp"] == 0
    assert client.detect() == ["box_gripper"]
    client.stop()



def test_box_client_reports_connected_session_and_no_cache_status(fake_box_module):
    cfg = box_client.BoxClientConfig(
        enabled=True,
        poll_interval_s=0.01,
        stale_threshold_s=5.0,
        expected_devices=["box_gripper", "box_imu"],
    )
    client = box_client.BoxClient(cfg)

    assert client.start() is True
    import time as _t
    _t.sleep(0.05)

    assert client.connected_devices() == ["box_gripper", "box_imu"]
    snap = client.read()
    assert snap["valid"] is False
    assert snap["status"]["active"] is True
    assert snap["status"]["poll_count"] > 0
    assert snap["status"]["valid_poll_count"] == 0
    assert snap["status"]["last_rc"] == 4
    assert snap["status"]["last_error"] == "no cached sensor data"
    assert snap["status"]["sensor_status"]["box_gripper"]["seen"] is False
    client.stop()

def test_box_client_start_is_noop_when_wheel_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "box_sdk", raising=False)
    monkeypatch.setattr(box_client, "available", lambda: False)

    # Force `import box_sdk` to fail inside .start().
    real_import = __import__

    def _import(name, *args, **kw):
        if name == "box_sdk":
            raise ImportError("synthetic-missing")
        return real_import(name, *args, **kw)

    monkeypatch.setattr("builtins.__import__", _import)
    cfg = box_client.BoxClientConfig(enabled=True)
    client = box_client.BoxClient(cfg)
    assert client.start() is False
    client.stop()  # idempotent / safe


def test_cleanup_session_csv_removes_only_session_files(tmp_path):
    # The SDK .so dumps box_sensor_data_*.csv to CWD; stop() must delete only
    # the file(s) that appeared this session, leaving older/unrelated ones.
    cfg = box_client.BoxClientConfig()
    assert cfg.cleanup_box_csv is True
    client = box_client.BoxClient(cfg)
    pre = tmp_path / "box_sensor_data_20260101_000000.csv"
    pre.write_text("old")
    client._csv_dir = tmp_path
    client._pre_session_csv = {pre}
    new = tmp_path / "box_sensor_data_20260615_120000.csv"
    new.write_text("session")
    unrelated = tmp_path / "keep.txt"
    unrelated.write_text("z")

    client._cleanup_session_csv()

    assert pre.exists()        # pre-existing dump untouched
    assert not new.exists()    # this session's CSV removed
    assert unrelated.exists()  # non-matching file untouched


def test_cleanup_session_csv_respects_disable_flag(tmp_path):
    cfg = box_client.BoxClientConfig(cleanup_box_csv=False)
    client = box_client.BoxClient(cfg)
    f = tmp_path / "box_sensor_data_20260615_120000.csv"
    f.write_text("session")
    client._csv_dir = tmp_path
    client._pre_session_csv = set()

    client._cleanup_session_csv()

    assert f.exists()  # cleanup disabled -> file kept


def _write_sdk_conf(path, **values) -> None:
    lines = ["# box_collection_sdk 运行时配置", "sensor_csv_enabled=1"]
    lines += [f"{k}={v}  # trailing comment" for k, v in values.items()]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def test_cleanup_removes_v41_session_dirs_created_this_session(tmp_path, monkeypatch):
    # v4.1 writes <cwd>/<sensor_csv_output_dir>/session_<ts>/*.csv per Box.start().
    conf = tmp_path / "box_sdk.conf"
    _write_sdk_conf(conf, sensor_csv_output_dir="sensor_data")
    monkeypatch.setenv("BOX_SDK_CONFIG", str(conf))
    cwd = tmp_path / "cwd"
    cwd.mkdir()
    root = cwd / "sensor_data"
    old = root / "session_20261101_100000"
    old.mkdir(parents=True)
    (old / "imu.csv").write_text("old")

    client = box_client.BoxClient(box_client.BoxClientConfig())
    client._csv_dir = cwd
    client._pre_session_csv = box_client._snapshot_session_csv(cwd)
    new = root / "session_20261102_120000"
    new.mkdir()
    (new / "touch_points_0.csv").write_text("session")
    (root / "notes.txt").write_text("not ours")

    client._cleanup_session_csv()

    assert not new.exists()
    assert (old / "imu.csv").exists()  # an earlier session's output is left alone
    assert (root / "notes.txt").exists()


def test_cleanup_drops_output_root_it_created_and_honours_absolute_dir(tmp_path, monkeypatch):
    out_root = tmp_path / "abs_out"
    conf = tmp_path / "box_sdk.conf"
    _write_sdk_conf(conf, sensor_csv_output_dir=str(out_root))
    monkeypatch.setenv("BOX_SDK_CONFIG", str(conf))
    cwd = tmp_path / "cwd"
    cwd.mkdir()

    client = box_client.BoxClient(box_client.BoxClientConfig())
    client._csv_dir = cwd
    client._pre_session_csv = box_client._snapshot_session_csv(cwd)
    session = out_root / "session_20261102_120000"
    session.mkdir(parents=True)
    (session / "gripper.csv").write_text("x")

    client._cleanup_session_csv()

    assert not out_root.exists()  # created by this session and empty again
    assert not (cwd / "sensor_data").exists()


def test_sdk_csv_root_defaults_when_config_missing(tmp_path, monkeypatch):
    monkeypatch.setenv("BOX_SDK_CONFIG", str(tmp_path / "missing.conf"))
    assert box_client._sdk_csv_session_root(tmp_path) == tmp_path / "sensor_data"
    monkeypatch.delenv("BOX_SDK_CONFIG")
    assert box_client._sdk_csv_session_root(tmp_path) == tmp_path / "sensor_data"


def test_vendored_conf_keeps_sensor_csv_off():
    conf = box_client.Path(box_client.__file__).resolve().parent / "box_sdk.conf"
    values = box_client._sdk_config_values(conf)
    assert values["sensor_csv_enabled"] == "0"
    assert values["sensor_csv_output_dir"] == "sensor_data"


# --- multi-box scaffolding --------------------------------------------------


def test_namespace_sid_only_prefixes_when_box_id_present():
    assert box_client.namespace_sid("", "box_gripper") == "box_gripper"
    assert box_client.namespace_sid("box1", "box_gripper") == "box1/box_gripper"


def test_fleet_from_yaml_legacy_flat_block_is_single_unnamespaced_box():
    raw = {"enabled": True, "remote_ip": "10.0.0.1", "expected_devices": ["box_gripper"]}
    fleet = box_client.fleet_from_yaml_dict(raw)
    assert fleet.enabled is True
    assert fleet.discovery == "static"
    assert len(fleet.boxes) == 1
    assert fleet.boxes[0].box_id == ""  # legacy -> bare sensor ids
    assert fleet.boxes[0].remote_ip == "10.0.0.1"


def test_fleet_from_yaml_disabled_and_missing():
    assert box_client.fleet_from_yaml_dict(None).enabled is False
    assert box_client.fleet_from_yaml_dict(None).boxes == []
    assert box_client.fleet_from_yaml_dict({"enabled": False}).enabled is False


def test_fleet_from_yaml_boxes_list_inherits_enabled_and_keeps_ids():
    raw = {
        "enabled": True,
        "discovery": "sdk",
        "boxes": [
            {"box_id": "box0", "remote_ip": "192.168.2.60", "expected_devices": ["box_gripper"]},
            {"box_id": "box1", "remote_ip": "192.168.2.61", "expected_devices": ["box_gripper"]},
        ],
    }
    fleet = box_client.fleet_from_yaml_dict(raw)
    assert fleet.discovery == "sdk"
    assert [b.box_id for b in fleet.boxes] == ["box0", "box1"]
    assert [b.remote_ip for b in fleet.boxes] == ["192.168.2.60", "192.168.2.61"]
    assert all(b.enabled for b in fleet.boxes)  # inherited from fleet


def test_fleet_config_rejects_missing_or_duplicate_box_ids():
    with pytest.raises(ValueError):
        box_client.BoxFleetConfig(
            boxes=[box_client.BoxClientConfig(box_id="box0"), box_client.BoxClientConfig()]
        )
    with pytest.raises(ValueError):
        box_client.BoxFleetConfig(
            boxes=[box_client.BoxClientConfig(box_id="dup"), box_client.BoxClientConfig(box_id="dup")]
        )
    with pytest.raises(ValueError):
        box_client.BoxFleetConfig(discovery="bogus")


def test_make_discovery_is_the_static_fallback_for_both_config_values():
    # Live enumeration is unconditional now (BoxPool.start -> discover_boxes), so
    # this strategy only supplies the fallback list; discovery: sdk is accepted
    # for deployed YAML but no longer selects a different code path.
    boxes = [box_client.BoxClientConfig(box_id="box0")]
    for mode in ("static", "sdk"):
        fleet = box_client.BoxFleetConfig(discovery=mode, boxes=boxes)
        disco = box_client.make_discovery(fleet)
        assert isinstance(disco, box_client.StaticBoxDiscovery)
        assert [c.box_id for c in disco.discover()] == ["box0"]


def test_box_config_record_poll_default_keeps_margin_for_480hz_force():
    cfg = box_client.from_yaml_dict({"enabled": True})
    assert cfg.record_poll_interval_s == 0.0005


def _preloaded_box_factory(distance: float):
    def _factory(*a, **kw):
        b = _FakeBox()
        b.snaps.append(
            _SensorCache(
                valid=1,
                data=_AllSensor(gripper_data=_Gripper(timestamp=1, distance=distance)),
            ),
        )
        return b

    return _factory


def test_box_pool_single_empty_id_delegates_without_namespacing(fake_box_module):
    fake_box_module.Box = _preloaded_box_factory(0.05)  # type: ignore[assignment]
    fleet = box_client.BoxFleetConfig(
        boxes=[box_client.BoxClientConfig(box_id="", poll_interval_s=0.01, stale_threshold_s=5.0)]
    )
    pool = box_client.BoxPool(fleet)
    assert pool.start() is True
    import time as _t
    _t.sleep(0.05)

    snap = pool.read()
    assert "box_gripper" in snap["sensors"]  # bare, no prefix
    assert "boxes" not in snap  # passthrough returns the lone client's dict
    assert pool.detect() == ["box_gripper"]
    assert pool.is_active() is True
    assert _FakeKeepAlive.instances
    assert "so_path" in _FakeKeepAlive.instances[-1].kwargs
    pool.stop()
    assert pool.is_active() is False


class _MultiDeviceBox(_FakeBox):
    """Shared v3 Box that serves per-device gripper snaps whose MCU timestamp
    increments each poll so the recorder's timestamp-dedup captures samples."""

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._ts: dict[int, int] = {}

    def get_sensor_cache(self, device_id=None):
        if device_id is None:
            return 4, None
        self._ts[device_id] = self._ts.get(device_id, 0) + 1
        return 0, _SensorCache(
            valid=1,
            data=_AllSensor(
                gripper_data=_Gripper(timestamp=self._ts[device_id], distance=0.05)
            ),
        )


def test_box_pool_two_boxes_namespace_sensors(fake_box_module):
    # 方案A: one shared Box, two devices enumerated by broadcast discovery.
    fake_box_module.Box = _MultiDeviceBox
    fake_box_module.discover = lambda **kw: [
        _FakeDiscovered(device_id=1, sn="box0", ip="192.168.2.61",
                        capabilities=box_client.CAP_GRIPPER),
        _FakeDiscovered(device_id=2, sn="box1", ip="192.168.2.62",
                        capabilities=box_client.CAP_GRIPPER),
    ]
    fleet = box_client.BoxFleetConfig(
        boxes=[
            box_client.BoxClientConfig(box_id="box0", poll_interval_s=0.01, stale_threshold_s=5.0),
            box_client.BoxClientConfig(box_id="box1", poll_interval_s=0.01, stale_threshold_s=5.0),
        ]
    )
    pool = box_client.BoxPool(fleet)
    assert pool.start() is True
    import time as _t
    _t.sleep(0.05)

    # Discovery roster surfaces both devices for the GUI/recorder.
    roster = pool.discovered_devices()
    assert {d["box_id"] for d in roster} == {"box0", "box1"}
    assert {d["device_id"] for d in roster} == {1, 2}
    assert all(d["capability_names"] == ["box_gripper"] for d in roster)

    snap = pool.read()
    assert set(snap["sensors"]) == {"box0/box_gripper", "box1/box_gripper"}
    assert set(snap["boxes"]) == {"box0", "box1"}
    assert set(pool.detect()) == {"box0/box_gripper", "box1/box_gripper"}
    # observed_rates() reports every known sensor (0.0 when absent); just check
    # the keys are namespaced per box and never bare.
    rates = pool.observed_rates()
    assert {"box0/box_gripper", "box1/box_gripper"} <= set(rates)
    assert all("/" in sid for sid in rates)

    pool.start_recording(t0_wall_s=100.0)
    _t.sleep(0.05)
    samples = pool.stop_recording()
    # Keys cover every known sensor per box (empty lists for absent ones); the
    # grippers actually produced samples and every key is namespaced.
    assert all("/" in sid for sid in samples)
    assert samples["box0/box_gripper"] and samples["box1/box_gripper"]
    # SensorSample.sensor_id is re-tagged so serialization emits namespaced ids.
    for nsid, sample_list in samples.items():
        assert all(s.sensor_id == nsid for s in sample_list)
    pool.stop()


def test_box_pool_leaves_the_mode_of_a_kept_box_alone(fake_box_module):
    # The FR3's gripper BOX: mode 0 hands its jaw to its own trigger and drops what it holds,
    # so neither the recorder's Connect nor its rearm may set_mode() it.
    fake_box_module.Box = _MultiDeviceBox
    fake_box_module.discover = lambda **kw: [
        _FakeDiscovered(device_id=7, ip="192.168.2.61", capabilities=box_client.CAP_GRIPPER),
        _FakeDiscovered(device_id=9, ip="192.168.2.62", capabilities=box_client.CAP_GRIPPER),
    ]
    fleet = box_client.fleet_from_yaml_dict(
        {"enabled": True, "keep_mode_device_ids": [9], "rearm_interval_s": 0.01, "poll_interval_s": 0.005}
    )
    assert fleet.keep_mode_device_ids == [9]
    pool = box_client.BoxPool(fleet)
    assert pool.start() is True
    import time as _t
    _t.sleep(0.08)
    box = pool._shared_box
    assert box.mode.get(7) == 0
    assert 9 not in box.mode
    assert pool.client_for_device(9) is not None and pool.client_for_device(3) is None
    assert pool.client_for_device(9).set_clamp_pos(0.03) == 0  # still commandable
    assert box.clamp_cmds[-1] == (0.03, 9)
    pool.stop()


class _TwoBoxSharedSocket(_FakeBox):
    """Two BOXes streaming on one socket while broadcast discovery answers nothing.

    ``get_sensor_cache(None)`` alternates between the two devices, exactly like the
    SDK's most-recent-device path; an addressed read returns only that device's
    frames. The gripper distance identifies the source box.
    """

    DISTANCE = {7: 0.01, 9: 0.09}

    def __init__(self, *a, report_ids: bool = False, **kw):
        super().__init__(*a, **kw)
        self.report_ids = report_ids
        self.addressed: list[int | None] = []
        self._turn = 0
        self._ts = 0

    def _frame(self, device_id: int) -> _SensorCache:
        self._ts += 1
        return _SensorCache(
            valid=1,
            device_id=device_id,
            data=_AllSensor(
                gripper_data=_Gripper(timestamp=self._ts, distance=self.DISTANCE[device_id])
            ),
        )

    def get_device_ids(self, *_):
        return [7, 9] if self.report_ids else []

    def get_known_device_ids(self, *_):
        return [7, 9] if self.report_ids else []

    def get_sensor_cache(self, device_id=None):
        self.addressed.append(device_id)
        if device_id is None:
            self._turn += 1
            return 0, self._frame(7 if self._turn % 2 else 9)
        return 0, self._frame(int(device_id))


def test_unaddressed_device_handle_pins_from_the_cache_device_id():
    # Discovery failed, two boxes push to the same socket, and this SDK exposes no
    # id list. Every SensorCache still names its source device, so the handle must
    # pin to one box rather than let the most-recent-device path interleave both
    # streams into one recorded view.
    box = _TwoBoxSharedSocket()
    handle = box_client._DeviceHandle(box, None)

    rc, first = handle.get_sensor_cache()
    assert rc == 0
    pinned = first.device_id
    assert pinned in (7, 9)
    assert handle.device_id == pinned

    for _ in range(6):
        _rc, snap = handle.get_sensor_cache()
        assert snap.device_id == pinned  # never the other box's frames

    assert box.addressed[0] is None  # only the first read is ambiguous
    assert set(box.addressed[1:]) == {pinned}
    # Commands go to the same device the samples came from.
    assert handle.set_mode(1) == 0
    assert box.mode == {pinned: 1}


def test_unaddressed_device_handle_pins_from_reporting_ids_and_warns(caplog):
    box = _TwoBoxSharedSocket(report_ids=True)
    handle = box_client._DeviceHandle(box, None)

    with caplog.at_level(logging.WARNING, logger="box_client"):
        rc, snap = handle.get_sensor_cache()

    assert rc == 0
    assert handle.device_id == 7  # lowest reporting id -- deterministic per session
    assert snap.device_id == 7  # already an addressed read, never the merged path
    assert box.addressed == [7]
    warning = " ".join(r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING)
    assert "device_id=7" in warning
    assert "[9]" in warning  # the box this view is NOT recording is named


def test_box_pool_fallback_view_records_a_single_device(fake_box_module):
    # Broadcast discovery finds nothing (blocked/mid-cycle) but two boxes stream on
    # :15000. The lone fallback view must record one device's samples, never a
    # blend of both, and say which device that was.
    fake_box_module.Box = _TwoBoxSharedSocket
    fake_box_module.discover = lambda **kw: []
    fleet = box_client.BoxFleetConfig(
        boxes=[
            box_client.BoxClientConfig(
                box_id="", poll_interval_s=0.005, stale_threshold_s=5.0
            )
        ]
    )
    pool = box_client.BoxPool(fleet)
    assert pool.start() is True
    import time as _t

    pool.start_recording(t0_wall_s=100.0)
    _t.sleep(0.1)
    samples = pool.stop_recording()
    status = pool.read()["status"]
    pool.stop()

    pinned = status["device_id"]
    assert pinned in (7, 9)
    assert samples["box_gripper"]
    distances = {s.data["distance_m"] for s in samples["box_gripper"]}
    assert distances == {_TwoBoxSharedSocket.DISTANCE[pinned]}


# --- v3 discovery helpers ---------------------------------------------------


def test_caps_to_sensor_ids_expands_touch_and_keeps_canonical_order():
    assert box_client.caps_to_sensor_ids(0) == []
    assert box_client.caps_to_sensor_ids(box_client.CAP_GRIPPER) == ["box_gripper"]
    # CAP_TOUCH is a single vendor bit covering both Paxini pads.
    assert box_client.caps_to_sensor_ids(box_client.CAP_TOUCH) == [
        "box_touch_left", "box_touch_right",
    ]
    # All caps -> KNOWN_SENSOR_IDS in canonical order, less the second pads: one
    # CAP_TOUCH bit cannot say how many pads a finger has, so those surface once they stream.
    all_caps = (
        box_client.CAP_GRIPPER | box_client.CAP_IMU | box_client.CAP_TRIGGER
        | box_client.CAP_6D_FORCE | box_client.CAP_TOUCH
    )
    assert box_client.caps_to_sensor_ids(all_caps) == [
        sid for sid in box_client.KNOWN_SENSOR_IDS if sid not in ("box_touch_left_1", "box_touch_right_1")
    ]


def test_discover_boxes_returns_empty_when_wheel_missing(monkeypatch):
    # No box_sdk on dev hosts -> graceful [] (never raises) so the gateway can
    # still render the Device Manager.
    monkeypatch.delitem(sys.modules, "box_sdk", raising=False)
    real_import = __import__

    def _import(name, *args, **kw):
        if name == "box_sdk":
            raise ImportError("synthetic-missing")
        return real_import(name, *args, **kw)

    monkeypatch.setattr("builtins.__import__", _import)
    assert box_client.discover_boxes() == []


def _two_boxes_on_the_bench(fake_box_module):
    created: list[_FakeBox] = []

    def _factory(*a, **kw):
        b = _FakeBox()
        created.append(b)
        return b

    fake_box_module.Box = _factory  # type: ignore[assignment]
    # 2026-10-09: the handheld BOX that recorded and the BOX on the FR3, both on Thor.
    # IP order puts the arm's first today; DHCP can swap that tomorrow.
    fake_box_module.discover = lambda **kw: [
        _FakeDiscovered(device_id=596523097, sn="box", ip="192.168.2.134", capabilities=box_client.CAP_GRIPPER),
        _FakeDiscovered(device_id=1819152274, sn="box", ip="192.168.216.196", capabilities=box_client.CAP_GRIPPER),
    ]
    return created


def test_box_client_drives_the_pinned_device_not_the_first_discovered(fake_box_module):
    created = _two_boxes_on_the_bench(fake_box_module)
    client = box_client.BoxClient(box_client.BoxClientConfig(enabled=True, poll_interval_s=0.01), device_id=1819152274)
    assert client.start() is True
    try:
        assert client.set_clamp_pos(0.03) == 0
    finally:
        client.stop()
    assert created[-1].clamp_cmds == [(0.03, 1819152274)]


def test_box_client_refuses_to_stand_in_for_a_pinned_device_that_is_absent(fake_box_module):
    created = _two_boxes_on_the_bench(fake_box_module)
    client = box_client.BoxClient(box_client.BoxClientConfig(enabled=True, poll_interval_s=0.01), device_id=42)
    assert client.start() is False
    assert created[-1].stopped
    assert created[-1].clamp_cmds == [] and created[-1].mode == {}


def test_discover_boxes_maps_capabilities_to_expected_devices(fake_box_module):
    fake_box_module.discover = lambda **kw: [
        _FakeDiscovered(device_id=107, sn="SN-7", ip="192.168.2.61",
                        capabilities=box_client.CAP_GRIPPER | box_client.CAP_IMU),
    ]
    found = box_client.discover_boxes()
    assert len(found) == 1
    d = found[0]
    assert d.device_id == 107
    assert d.expected_devices == ["box_gripper", "box_imu"]
    pub = d.to_public_dict()
    assert pub["device_id"] == 107 and pub["sn"] == "SN-7"
    assert pub["capability_names"] == ["box_gripper", "box_imu"]


def test_discover_boxes_drops_a_box_answering_twice(fake_box_module):
    # v4.1 on Thor (2026-11-02) listed device 3100715645 @ 192.168.1.207 twice,
    # which built two views polling the same box under one namespace.
    fake_box_module.discover = lambda **kw: [
        _FakeDiscovered(device_id=3100715645, ip="192.168.1.207"),
        _FakeDiscovered(device_id=3100715645, ip="192.168.1.207"),
        _FakeDiscovered(device_id=42, ip="192.168.1.208"),
    ]
    found = box_client.discover_boxes()
    assert [d.device_id for d in found] == [3100715645, 42]


def test_discover_boxes_unions_rounds_that_each_see_one_box(fake_box_module):
    # Thor 2026-11-02 (v4.1, collection + execution box): every broadcast
    # answered with exactly one of the two boxes, alternating.
    rounds = iter([
        [_FakeDiscovered(device_id=3100715645, ip="192.168.1.207")],
        [_FakeDiscovered(device_id=3906437858, ip="192.168.1.150", fw_version=3),
         _FakeDiscovered(device_id=3906437858, ip="192.168.1.150", fw_version=2)],
        [_FakeDiscovered(device_id=3100715645, ip="192.168.1.207")],
        [_FakeDiscovered(device_id=3906437858, ip="192.168.1.150")],
        [],
    ])
    calls = []

    def _discover(**kw):
        calls.append(kw)
        return next(rounds)

    fake_box_module.discover = _discover
    found = box_client.discover_boxes()
    assert [(d.device_id, d.ip) for d in found] == [
        (3906437858, "192.168.1.150"), (3100715645, "192.168.1.207"),
    ]
    assert found[0].fw_version == 3  # first answer for a device wins
    assert len(calls) == 4  # rounds 3 and 4 added nothing new -> stop


def test_discover_boxes_keeps_trying_while_nothing_answers(fake_box_module):
    calls = []

    def _discover(**kw):
        calls.append(kw)
        return []

    fake_box_module.discover = _discover
    assert box_client.discover_boxes(retries=3) == []
    assert len(calls) == 3


def test_discovery_forwards_so_path_when_sdk_lives_in_the_wheel(fake_box_module):
    # Deployed on Thor, box_sdk is imported straight out of the vendored .whl, so
    # its own library resolver looks inside the zip (<...>.whl/box_sdk/lib) and
    # raises NotADirectoryError. Every broadcast must therefore hand the SDK the
    # extracted so_path -- including the one inside BoxClient.start(), which used
    # to call box_sdk.discover() bare and lose the device_id (and with it set_mode
    # and every calibration command) on the standalone single-box path.
    seen: list[str | None] = []

    def _discover(**kw):
        seen.append(kw.get("so_path"))
        if not kw.get("so_path"):
            raise NotADirectoryError(20, "Not a directory", "sdk.whl/box_sdk/lib")
        return [_FakeDiscovered(device_id=7, ip="192.168.2.60")]

    fake_box_module.discover = _discover
    cfg = box_client.BoxClientConfig(enabled=True, poll_interval_s=0.01)
    client = box_client.BoxClient(cfg, so_path="/lib/libbox_controller.so")
    assert client.start() is True
    try:
        # every broadcast round (answers are unioned over several) carries it
        assert seen and set(seen) == {"/lib/libbox_controller.so"}
        assert client._device_id == 7
        # device_id learned -> address registered and startup_mode actually applied
        assert client._raw_box.registered == [(7, "192.168.2.60", 15000)]
        assert client._raw_box.mode == {7: 0}
    finally:
        client.stop()


def test_box_pool_duplicate_serials_namespace_by_device_id(fake_box_module):
    # Un-personalized firmware ships every unit with the same box_serial; the
    # pool must still give each a UNIQUE namespace (by device_id) instead of
    # merging both physical boxes under one colliding serial.
    class _Multi(_FakeBox):
        def __init__(self, *a, **kw):
            super().__init__(*a, **kw)
            self._ts: dict[int, int] = {}

        def get_sensor_cache(self, device_id=None):
            if device_id is None:
                return 4, None
            self._ts[device_id] = self._ts.get(device_id, 0) + 1
            return 0, _SensorCache(
                valid=1,
                data=_AllSensor(gripper_data=_Gripper(timestamp=self._ts[device_id], distance=0.05)),
            )

    fake_box_module.Box = _Multi
    fake_box_module.discover = lambda **kw: [
        _FakeDiscovered(device_id=111, sn="dup", ip="192.168.122.75",
                        capabilities=box_client.CAP_GRIPPER),
        _FakeDiscovered(device_id=222, sn="dup", ip="192.168.216.196",
                        capabilities=box_client.CAP_GRIPPER),
    ]
    fleet = box_client.fleet_from_yaml_dict({"enabled": True, "poll_interval_s": 0.01})
    pool = box_client.BoxPool(fleet)
    assert pool.start() is True
    import time as _t
    _t.sleep(0.05)
    snap = pool.read()
    # Two distinct device_id namespaces despite identical serials -> no merge.
    assert set(snap["sensors"]) == {"box111/box_gripper", "box222/box_gripper"}
    assert {d["box_id"] for d in pool.discovered_devices()} == {"box111", "box222"}
    pool.stop()


def test_box_pool_single_discovered_device_stays_unnamespaced(fake_box_module):
    # A lone discovered device on a legacy single-box fleet keeps bare ids so
    # existing datasets/rows are byte-compatible.
    fake_box_module.Box = _preloaded_box_factory(0.05)  # type: ignore[assignment]

    def _shared(*a, **kw):
        b = _FakeBox()
        b.snaps_by_id[55] = _SensorCache(
            valid=1, data=_AllSensor(gripper_data=_Gripper(timestamp=3, distance=0.05))
        )
        return b

    fake_box_module.Box = _shared  # type: ignore[assignment]
    fake_box_module.discover = lambda **kw: [
        _FakeDiscovered(device_id=55, sn="SN-1", ip="192.168.2.60",
                        capabilities=box_client.CAP_GRIPPER),
    ]
    fleet = box_client.BoxFleetConfig(
        boxes=[box_client.BoxClientConfig(box_id="", poll_interval_s=0.01, stale_threshold_s=5.0)]
    )
    pool = box_client.BoxPool(fleet)
    assert pool.start() is True
    import time as _t
    _t.sleep(0.05)
    snap = pool.read()
    assert "box_gripper" in snap["sensors"]  # bare, no prefix
    assert "boxes" not in snap  # passthrough delegates to the lone client
    # roster still reports the real device_id/sn for the GUI.
    assert pool.discovered_devices()[0]["device_id"] == 55
    pool.stop()


def test_ensure_box_sdk_importable_prefers_vendored_wheel(tmp_path, monkeypatch) -> None:
    wheel_dir = tmp_path / "box_collection_sdk-9.9.9-py3-none-any.whl"
    package_dir = wheel_dir / "box_sdk"
    package_dir.mkdir(parents=True)
    (package_dir / "__init__.py").write_text("MARKER = 'vendored'\n")

    system_box_sdk = types.ModuleType("box_sdk")
    system_box_sdk.__file__ = "/usr/lib/python3/dist-packages/box_sdk/__init__.py"
    submodule = types.ModuleType("box_sdk.old_submodule")
    monkeypatch.setitem(sys.modules, "box_sdk", system_box_sdk)
    monkeypatch.setitem(sys.modules, "box_sdk.old_submodule", submodule)
    monkeypatch.setattr(box_client, "_vendored_wheel_path", lambda: wheel_dir)
    old_path = list(sys.path)
    try:
        box_client._ensure_box_sdk_importable()
        imported = __import__("box_sdk")
        assert imported.MARKER == "vendored"
        assert str(wheel_dir) == sys.path[0]
        assert "box_sdk.old_submodule" not in sys.modules
    finally:
        sys.path[:] = old_path
        sys.modules.pop("box_sdk", None)
        sys.modules.pop("box_sdk.old_submodule", None)


# --------------------------------------------------------------- touch pads ---
#
# SDK v4.1 frames are self-describing: each TouchSensorSample carries exactly
# the taxels its pad reported, and both pad families share link streams
# 0x0002/0x0003, so the frame length is the only evidence of the pad.


def test_decode_touch_tags_the_pad_its_frame_length_identifies():
    m2020_frame = _Touch(
        timestamp=40,
        total_force=_TouchForce(fx=7, fy=-8, fz=9),
        forces=tuple(_TouchForce(fx=i + 1, fy=-(i + 1), fz=(i + 1) * 3) for i in range(9)),
    )
    odd_frame = _Touch(timestamp=41, forces=tuple(_TouchForce() for _ in range(64)))
    snap = _SensorCache(touch=_terminals(m2020_frame, odd_frame, touch_device_id=3))

    sensors = box_client.decode_sensor_cache(snap)["sensors"]
    m2020 = sensors["box_touch_left"]
    assert m2020["model"] == box_client.TOUCH_MODEL_M2020
    assert m2020["points"] == 9
    assert m2020["fz_0p1N"] == [3, 6, 9, 12, 15, 18, 21, 24, 27]
    # total_force is its own MCU-side aggregate, not a sum over the taxels.
    assert m2020["total_force_0p1N"] == [7, -8, 9]
    assert m2020["touch_device_id"] == 3

    # A length no known pad has is passed through whole and tagged unknown.
    odd = sensors["box_touch_right"]
    assert odd["model"] == "unknown"
    assert odd["points"] == 64 == len(odd["fz_0p1N"])


def test_two_pads_per_finger_record_as_their_own_sensors():
    # 11-02 rig: each finger carries two pads, touch_device_id 0 and 1. The first keeps the
    # pre-v4.1 id (a one-pad rig records what it always did), the second gets ``_1``.
    box_client._warned_multi_touch.clear()
    snap = _SensorCache(
        touch=(
            {1: _Touch(timestamp=11), 0: _Touch(timestamp=10)},
            {0: _Touch(timestamp=20), 1: _Touch(timestamp=21)},
        )
    )
    sensors = box_client.decode_sensor_cache(snap)["sensors"]
    assert {sid: (d["touch_device_id"], d["timestamp"]) for sid, d in sensors.items() if "touch" in sid} == {
        "box_touch_left": (0, 10),
        "box_touch_left_1": (1, 11),
        "box_touch_right": (0, 20),
        "box_touch_right_1": (1, 21),
    }
    stamps = box_client._decode_sensor_timestamps(snap)
    assert [stamps[sid] for sid in box_client.TOUCH_SENSOR_IDS] == [10, 11, 20, 21]
    # The recording path decodes only what changed: one pad's new frame, not its neighbour's.
    changed = box_client._decode_changed_sensors(snap, {"box_touch_right_1"})["sensors"]
    assert list(changed) == ["box_touch_right_1"] and changed["box_touch_right_1"]["timestamp"] == 21
    assert not box_client._warned_multi_touch


def test_a_third_pad_on_a_finger_is_dropped_with_one_warning(caplog):
    box_client._warned_multi_touch.clear()
    snap = _SensorCache(
        touch=(
            {7: _Touch(timestamp=70), 2: _Touch(timestamp=20), 5: _Touch(timestamp=50)},
            {},
        )
    )
    with caplog.at_level(logging.WARNING, logger="box_client"):
        for _ in range(3):
            out = box_client.decode_sensor_cache(snap)
    assert out["sensors"]["box_touch_left"]["touch_device_id"] == 2
    assert out["sensors"]["box_touch_left_1"]["touch_device_id"] == 5
    assert not any(d.get("touch_device_id") == 7 for d in out["sensors"].values())
    assert "box_touch_right" not in out["sensors"]
    assert box_client._decode_sensor_timestamps(snap)["box_touch_right_1"] == 0
    warnings = [r for r in caplog.records if "reports 3 devices" in r.getMessage()]
    assert len(warnings) == 1


def test_a_second_pad_gets_a_status_row_once_it_streams():
    # Discovery's one CAP_TOUCH bit cannot announce it, so it is not in expected_devices.
    cfg = box_client.BoxClientConfig(enabled=True, box_id="box0",
                                     expected_devices=list(box_client.caps_to_sensor_ids(box_client.CAP_TOUCH)))
    client = box_client.BoxClient(cfg)
    with client._lock:
        assert "box_touch_left_1" not in client._status_locked()["sensor_status"]
        client._last_seen_at_s["box_touch_left_1"] = time.monotonic()
        assert client._status_locked()["sensor_status"]["box_touch_left_1"]["fresh"]


def test_touch_point_count_round_trips_through_the_model_table():
    assert box_client.touch_point_count(box_client.TOUCH_MODEL_M2020) == 9
    assert box_client.touch_point_count(box_client.TOUCH_MODEL_PAXINI_L5325) == 239
    assert box_client.touch_point_count(None) is None
    assert box_client.touch_point_count("nope") is None
    assert box_client.touch_model_for_point_count(9) == box_client.TOUCH_MODEL_M2020
    assert box_client.touch_model_for_point_count(239) == box_client.TOUCH_MODEL_PAXINI_L5325
    assert box_client.touch_model_for_point_count(64) is None


def test_touch_model_latches_on_first_frame_and_link_stats_do_not_move_it(caplog):
    cfg = box_client.BoxClientConfig(enabled=True, box_id="box0")
    client = box_client.BoxClient(cfg)
    assert client._touch_model is None

    # v4.1 publishes both pads on 0x0002/0x0003: link stats are not evidence.
    client._handle_link_stats([_LinkStatsRecord(tlv_type=0x0002, samples_total=10)])
    assert client._touch_model is None

    with client._lock:
        client._note_touch_models_locked(
            {"box_touch_left": {"model": box_client.TOUCH_MODEL_M2020, "points": 9}}
        )
    assert client._touch_model == box_client.TOUCH_MODEL_M2020

    with caplog.at_level(logging.WARNING, logger="box_client"), client._lock:
        for _ in range(2):
            client._note_touch_models_locked(
                {"box_touch_right": {"model": box_client.TOUCH_MODEL_PAXINI_L5325, "points": 239}}
            )
    assert client._touch_model == box_client.TOUCH_MODEL_M2020
    assert len([r for r in caplog.records if "touch_model is m2020" in r.getMessage()]) == 1


def test_pinned_touch_model_is_reported_before_any_frame():
    cfg = box_client.BoxClientConfig(enabled=True, touch_model=box_client.TOUCH_MODEL_M2020)
    client = box_client.BoxClient(cfg)
    assert client._touch_model == box_client.TOUCH_MODEL_M2020

    with pytest.raises(ValueError, match="unknown touch_model"):
        box_client.BoxClientConfig(enabled=True, touch_model="l5325")


def test_system_status_decodes_named_bits():
    out = box_client.system_status_to_dict(
        0, types.SimpleNamespace(status_code=0b100011, error_code=(1 << 6) | (1 << 12))
    )
    assert out["rc"] == 0
    assert out["status"] == ["system_initialized", "six_d_force_ready", "trigger_ready"]
    assert out["errors"] == ["six_d_force_collision", "unknown_bits_0x00001000"]
    assert box_client.system_status_to_dict(5, None) == {
        "rc": 5, "status_code": None, "error_code": None, "status": [], "errors": [],
    }
