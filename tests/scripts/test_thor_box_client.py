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
