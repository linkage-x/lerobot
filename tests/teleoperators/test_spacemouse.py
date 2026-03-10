#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pytest
import sys
import types

from lerobot.teleoperators.spacemouse import SpaceMouseTeleop, SpaceMouseTeleopConfig
from lerobot.teleoperators.spacemouse.backend import PySpaceMouseDriver, SpaceMouseReading
from lerobot.teleoperators.spacemouse.configuration_spacemouse import (
    SpaceMouseEnableButton,
    SpaceMouseToolMode,
)


class DummySpaceMouseDriver:
    instances: list["DummySpaceMouseDriver"] = []
    queued_readings: list[SpaceMouseReading | None] = []

    def __init__(self, *args, **kwargs):
        del args, kwargs
        type(self).instances.append(self)
        self.connected = False
        self.readings: list[SpaceMouseReading | None] = list(type(self).queued_readings)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def poll(self) -> SpaceMouseReading | None:
        if not self.readings:
            return None
        return self.readings.pop(0)


@pytest.fixture
def teleop(monkeypatch):
    DummySpaceMouseDriver.instances = []
    DummySpaceMouseDriver.queued_readings = []
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    cfg = SpaceMouseTeleopConfig(
        tool_mode=SpaceMouseToolMode.INCREMENTAL,
        move_time=0.0,
        bias_sample_count=0,
    )
    device = SpaceMouseTeleop(cfg)
    yield device
    if device.is_connected:
        device.disconnect()


def test_connect_disconnect(teleop):
    assert not teleop.is_connected
    teleop.connect()
    assert teleop.is_connected
    teleop.disconnect()
    assert not teleop.is_connected


def test_get_action_returns_zero_when_no_data(teleop):
    teleop.connect()
    action = teleop.get_action()
    assert action["enabled"] is False
    assert action["target_x"] == pytest.approx(0.0)
    assert action["gripper"] == pytest.approx(teleop.config.initial_gripper)


def test_get_action_maps_axes_and_scales(teleop):
    teleop.connect()
    teleop._driver.readings.append(
        SpaceMouseReading(
            translation=[0.2, -0.3, 0.4],
            rotation=[0.5, -0.6, 0.7],
            buttons=(False, False),
        )
    )
    action = teleop.get_action()

    assert action["enabled"] is True
    assert action["target_x"] == pytest.approx(0.3 * teleop.config.scale_x)
    assert action["target_y"] == pytest.approx(0.2 * teleop.config.scale_y)
    assert action["target_z"] == pytest.approx(0.4 * teleop.config.scale_z)
    assert action["target_wx"] == pytest.approx(0.5 * teleop.config.scale_wx)
    assert action["target_wy"] == pytest.approx(-0.6 * teleop.config.scale_wy)
    assert action["target_wz"] == pytest.approx(0.7 * teleop.config.scale_wz)


def test_connect_estimates_idle_bias_and_cancels_idle_reading(monkeypatch):
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    bias_reading = SpaceMouseReading(
        translation=[0.05, -0.04, 0.03],
        rotation=[0.02, -0.01, 0.015],
        buttons=(False, False),
    )
    DummySpaceMouseDriver.queued_readings = [bias_reading, bias_reading, bias_reading, bias_reading]
    teleop = SpaceMouseTeleop(
        SpaceMouseTeleopConfig(
            bias_sample_count=3,
            bias_sample_sleep_s=0.0,
            move_time=0.0,
        )
    )
    teleop.connect()
    action = teleop.get_action()

    assert action["enabled"] is False
    assert action["target_x"] == pytest.approx(0.0)
    assert action["target_y"] == pytest.approx(0.0)
    assert action["target_z"] == pytest.approx(0.0)
    teleop.disconnect()


def test_get_action_zeroes_subthreshold_axes_even_when_other_axis_is_active(monkeypatch):
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    DummySpaceMouseDriver.queued_readings = []
    teleop = SpaceMouseTeleop(
        SpaceMouseTeleopConfig(
            threshold_x=0.02,
            threshold_y=0.02,
            threshold_z=0.02,
            threshold_wx=0.04,
            threshold_wy=0.04,
            threshold_wz=0.04,
            bias_sample_count=0,
            move_time=0.0,
        )
    )
    teleop.connect()
    teleop._driver.readings.append(
        SpaceMouseReading(
            translation=[0.03, -0.01, 0.015],
            rotation=[0.05, 0.03, 0.01],
            buttons=(False, False),
        )
    )

    action = teleop.get_action()

    assert action["enabled"] is True
    assert action["target_x"] == pytest.approx(0.0)
    assert action["target_y"] == pytest.approx(0.03 * teleop.config.scale_y)
    assert action["target_z"] == pytest.approx(0.0)
    assert action["target_wx"] == pytest.approx(0.05 * teleop.config.scale_wx)
    assert action["target_wy"] == pytest.approx(0.0)
    assert action["target_wz"] == pytest.approx(0.0)
    teleop.disconnect()


def test_get_action_requires_deadman_button(monkeypatch):
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    teleop = SpaceMouseTeleop(
        SpaceMouseTeleopConfig(
            tool_mode=SpaceMouseToolMode.INCREMENTAL,
            motion_enable_button=SpaceMouseEnableButton.LEFT,
            move_time=0.0,
        )
    )
    teleop.connect()
    teleop._driver.readings.extend(
        [
            SpaceMouseReading(
                translation=[0.2, -0.3, 0.4],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.2, -0.3, 0.4],
                rotation=[0.0, 0.0, 0.0],
                buttons=(True, False),
            ),
        ]
    )

    disabled = teleop.get_action()
    enabled = teleop.get_action()

    assert disabled["enabled"] is False
    assert enabled["enabled"] is True
    teleop.disconnect()


def test_incremental_gripper_updates(teleop):
    teleop.connect()
    teleop._last_gripper = 0.5
    start = teleop._last_gripper
    teleop._driver.readings.append(
        SpaceMouseReading(
            translation=[0.03, 0.0, 0.0],
            rotation=[0.0, 0.0, 0.0],
            buttons=(True, False),
        )
    )
    action = teleop.get_action()
    assert action["gripper"] == pytest.approx(start + teleop.config.incremental_step)


def test_binary_gripper_uses_left_open_right_close(monkeypatch):
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    teleop = SpaceMouseTeleop(
        SpaceMouseTeleopConfig(tool_mode=SpaceMouseToolMode.BINARY, initial_gripper=0.5)
    )
    teleop.connect()
    teleop._driver.readings.extend(
        [
            SpaceMouseReading(
                translation=[0.03, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(True, False),
            ),
            SpaceMouseReading(
                translation=[0.03, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, True),
            ),
        ]
    )
    first = teleop.get_action()
    second = teleop.get_action()
    assert first["gripper"] == pytest.approx(1.0)
    assert second["gripper"] == pytest.approx(0.0)
    teleop.disconnect()


def test_connect_cleans_up_failed_backend(monkeypatch):
    class FailingSpaceMouseDriver(DummySpaceMouseDriver):
        def connect(self) -> None:
            self.connected = True
            raise RuntimeError("spacemouse connect failed")

    FailingSpaceMouseDriver.instances = []
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", FailingSpaceMouseDriver)
    teleop = SpaceMouseTeleop(SpaceMouseTeleopConfig())

    with pytest.raises(RuntimeError, match="spacemouse connect failed"):
        teleop.connect()

    assert not teleop.is_connected
    assert teleop._driver is None
    assert FailingSpaceMouseDriver.instances[-1].connected is False


def test_pyspacemouse_driver_supports_get_connected_devices_api(monkeypatch):
    class DummyDevice:
        def __init__(self):
            self.closed = False

        def read(self):
            return types.SimpleNamespace(
                x=0.1,
                y=0.2,
                z=0.3,
                roll=0.4,
                pitch=0.5,
                yaw=0.6,
                buttons=(1, 0),
            )

        def close(self):
            self.closed = True

    opened = DummyDevice()
    fake_module = types.SimpleNamespace(
        get_connected_devices=lambda: ["/dev/hidraw7"],
        open=lambda **kwargs: opened if kwargs == {"device": "/dev/hidraw7"} else None,
    )
    monkeypatch.setitem(sys.modules, "pyspacemouse", fake_module)

    driver = PySpaceMouseDriver(device_id=0)
    driver.connect()
    reading = driver.poll()
    driver.disconnect()

    assert reading.translation.tolist() == pytest.approx([0.1, 0.2, 0.3])
    assert reading.rotation.tolist() == pytest.approx([0.4, 0.5, 0.6])
    assert reading.buttons == (True, False)
    assert opened.closed is True


def test_pyspacemouse_driver_raises_when_device_index_out_of_range(monkeypatch):
    fake_module = types.SimpleNamespace(
        get_connected_devices=lambda: ["/dev/hidraw7"],
        open=lambda **kwargs: None,
    )
    monkeypatch.setitem(sys.modules, "pyspacemouse", fake_module)

    driver = PySpaceMouseDriver(device_id=1)

    with pytest.raises(ConnectionError, match="out of range"):
        driver.connect()
