"""Unit tests for the LeRobot-side wrapper around the vendored BOX SDK.

The real wheel is ARM-only and not importable on the dev host, so we
inject a tiny in-memory stub that mimics the C structures (`SensorCache`,
`AllSensor`, ...) the wrapper expects. The wrapper code under test never
touches `libbox_controller.so` directly -- it goes through the wheel's
`box_sdk.Box` class -- so a faithful Python stand-in is enough.
"""

from __future__ import annotations

import sys
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
    distance: float = 0.0  # SDK calls this "distance" but it's travel pct


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
    timestamp: int = 0
    forces: tuple = field(default_factory=lambda: tuple(_TouchForce() for _ in range(239)))


@dataclass
class _AllSensor:
    gripper_data: _Gripper = field(default_factory=_Gripper)
    imu_data: _Imu = field(default_factory=_Imu)
    trigger_data: _Trigger = field(default_factory=_Trigger)
    six_d_force_data: _SixD = field(default_factory=_SixD)
    touch_sensor_data_first: _Touch = field(default_factory=_Touch)
    touch_sensor_data_sec: _Touch = field(default_factory=_Touch)


@dataclass
class _SensorCache:
    valid: int = 1
    liwp_index: int = 0
    liwp_timestemp: int = 0  # SDK typo preserved
    data: _AllSensor = field(default_factory=_AllSensor)
    touch_sensor_data: tuple = field(default_factory=tuple)


class _FakeBox:
    """In-memory stand-in for ``box_sdk.Box``."""

    def __init__(self, *_, **__):
        self.snaps: list[_SensorCache] = []
        self.mode = 0
        self.started = False
        self.stopped = False

    # --- protocol Box exposes to BoxClient -----------------------------
    def start(self, bind_ip, bind_port, remote_ip, remote_port):
        self.started = True
        self.bind = (bind_ip, bind_port)
        self.remote = (remote_ip, remote_port)
        return 0

    def stop(self):
        self.stopped = True

    def close(self):
        pass

    def set_mode(self, mode):
        self.mode = mode
        return 0

    def get_sensor_cache(self):
        if self.snaps:
            return 0, self.snaps[0]
        return 4, _SensorCache(valid=0)

    def err_str(self, rc):
        return "no cached sensor data" if rc == 4 else f"rc={rc}"


@pytest.fixture
def fake_box_module(monkeypatch):
    module = types.ModuleType("box_sdk")
    module.Box = _FakeBox
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
            imu_data=_Imu(),  # timestamp=0 -> dropped
            trigger_data=_Trigger(timestamp=20, distance=42.0),
            six_d_force_data=_SixD(timestamp=30, data=(1, 2, 3, 4, 5, 6)),
            touch_sensor_data_first=_Touch(
                timestamp=40,
                forces=tuple(
                    _TouchForce(fx=i % 7, fy=-(i % 5), fz=(i % 11))
                    for i in range(239)
                ),
            ),
            touch_sensor_data_sec=_Touch(timestamp=0),  # dropped
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
    assert out["sensors"]["box_six_d_force"]["fxyz_mxyz"] == [1, 2, 3, 4, 5, 6]
    touch = out["sensors"]["box_touch_left"]
    assert len(touch["fx_0p1N"]) == 239 == len(touch["fy_0p1N"]) == len(touch["fz_0p1N"])


def test_decode_sensor_cache_prefers_top_level_touch_array_over_legacy_fields():
    legacy_left = _Touch(
        timestamp=999,
        forces=tuple(_TouchForce(fx=9, fy=9, fz=9) for _ in range(239)),
    )
    legacy_right = _Touch(timestamp=998)
    array_left = _Touch(
        timestamp=101,
        forces=tuple(_TouchForce(fx=1, fy=2, fz=3) for _ in range(239)),
    )
    array_right = _Touch(
        timestamp=202,
        forces=tuple(_TouchForce(fx=-1, fy=-2, fz=4) for _ in range(239)),
    )
    snap = _SensorCache(
        data=_AllSensor(
            touch_sensor_data_first=legacy_left,
            touch_sensor_data_sec=legacy_right,
        ),
        touch_sensor_data=(array_left, array_right),
    )

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


def _force_box_factory(force=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0), *, with_cali):
    """Fake Box that publishes a 6D force sample; optionally exposes cali()."""
    calls: list[bool] = []

    class _Box(_FakeBox):
        if with_cali:
            def cali_6d_force_sensor(self):  # mirrors the newer vendor SDK
                calls.append(True)
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

    assert calls == [True]
    assert result["ok"] is True
    assert result["rc"] == 0
    assert result["error"] is None
    assert result["before"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
    assert result["after"] == [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]


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


def test_sdk_discovery_falls_back_to_static_boxes():
    boxes = [box_client.BoxClientConfig(box_id="box0")]
    # No enumerate_fn -> box_discover() unavailable -> fall back to static.
    assert box_client.SdkBoxDiscovery(boxes).discover()[0].box_id == "box0"
    fleet = box_client.BoxFleetConfig(discovery="sdk", boxes=boxes)
    assert isinstance(box_client.make_discovery(fleet), box_client.SdkBoxDiscovery)


def test_decode_sensor_mask_follows_known_sensor_bit_order():
    assert box_client.decode_sensor_mask(0) == []
    # bit 0 -> box_gripper, bit 1 -> box_imu (KNOWN_SENSOR_IDS order)
    assert box_client.decode_sensor_mask(0b1) == ["box_gripper"]
    assert box_client.decode_sensor_mask(0b11) == ["box_gripper", "box_imu"]
    assert box_client.decode_sensor_mask((1 << len(box_client.KNOWN_SENSOR_IDS)) - 1) == list(
        box_client.KNOWN_SENSOR_IDS
    )


def test_box_info_to_config_maps_identity_ip_and_sensor_mask():
    template = box_client.BoxClientConfig(
        bind_ip="192.168.2.45", remote_ip="0.0.0.0", poll_interval_s=0.002
    )
    info = box_client.BoxInfo(box_serial="SN-AAA", ip="192.168.2.61", sensor_mask=0b11)
    cfg = box_client.box_info_to_config(info, template)
    assert cfg.box_id == "SN-AAA"  # serial is the default namespace id
    assert cfg.remote_ip == "192.168.2.61"
    assert cfg.bind_ip == "192.168.2.45"  # shared default inherited from template
    assert cfg.poll_interval_s == 0.002
    assert cfg.expected_devices == ["box_gripper", "box_imu"]

    # aliases pin a friendly id; empty sensor_mask inherits the template list.
    template2 = box_client.BoxClientConfig(expected_devices=["box_gripper"])
    info2 = box_client.BoxInfo(box_serial="SN-BBB", ip="192.168.2.62", sensor_mask=0)
    cfg2 = box_client.box_info_to_config(info2, template2, aliases={"SN-BBB": "box1"})
    assert cfg2.box_id == "box1"
    assert cfg2.expected_devices == ["box_gripper"]


def test_sdk_discovery_with_injected_enumerate_maps_to_configs():
    template = box_client.BoxClientConfig(bind_ip="192.168.2.45")

    def _fake_box_discover():
        return [
            box_client.BoxInfo(box_serial="SN-0", ip="192.168.2.60", sensor_mask=0b1),
            box_client.BoxInfo(box_serial="SN-1", ip="192.168.2.61", sensor_mask=0b11),
        ]

    disco = box_client.SdkBoxDiscovery(
        fallback=[], enumerate_fn=_fake_box_discover, template=template,
        aliases={"SN-0": "box0", "SN-1": "box1"},
    )
    configs = disco.discover()
    assert [c.box_id for c in configs] == ["box0", "box1"]
    assert [c.remote_ip for c in configs] == ["192.168.2.60", "192.168.2.61"]
    assert configs[0].expected_devices == ["box_gripper"]
    assert configs[1].expected_devices == ["box_gripper", "box_imu"]
    assert all(c.bind_ip == "192.168.2.45" for c in configs)  # template default


def test_sdk_discovery_empty_or_failing_enumerate_falls_back():
    fallback = [box_client.BoxClientConfig(box_id="static0")]

    # enumerate returns nothing -> fall back.
    empty = box_client.SdkBoxDiscovery(fallback, enumerate_fn=lambda: [])
    assert [c.box_id for c in empty.discover()] == ["static0"]

    # enumerate raises -> fall back (SDK present but the call failed).
    def _boom():
        raise RuntimeError("sdk discovery error")

    failing = box_client.SdkBoxDiscovery(fallback, enumerate_fn=_boom)
    assert [c.box_id for c in failing.discover()] == ["static0"]


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
    pool.stop()
    assert pool.is_active() is False


def _counting_box_factory(distance: float):
    """Fake whose gripper MCU timestamp increments each poll so the recorder's
    timestamp-dedup actually captures samples."""

    def _factory(*a, **kw):
        b = _FakeBox()
        state = {"ts": 0}

        def _get():
            state["ts"] += 1
            return 0, _SensorCache(
                valid=1,
                data=_AllSensor(gripper_data=_Gripper(timestamp=state["ts"], distance=distance)),
            )

        b.get_sensor_cache = _get
        return b

    return _factory


def test_box_pool_two_boxes_namespace_sensors(fake_box_module):
    fake_box_module.Box = _counting_box_factory(0.05)  # type: ignore[assignment]
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
