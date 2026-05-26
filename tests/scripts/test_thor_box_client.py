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
