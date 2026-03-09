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

from lerobot.teleoperators.spacemouse import SpaceMouseTeleop, SpaceMouseTeleopConfig
from lerobot.teleoperators.spacemouse.backend import SpaceMouseReading
from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseToolMode


class DummySpaceMouseDriver:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.connected = False
        self.readings: list[SpaceMouseReading | None] = []

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
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    cfg = SpaceMouseTeleopConfig(
        tool_mode=SpaceMouseToolMode.INCREMENTAL,
        move_time=0.0,
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


def test_incremental_gripper_updates(teleop):
    teleop.connect()
    start = teleop.config.initial_gripper
    teleop._driver.readings.append(
        SpaceMouseReading(
            translation=[0.03, 0.0, 0.0],
            rotation=[0.0, 0.0, 0.0],
            buttons=(True, False),
        )
    )
    action = teleop.get_action()
    assert action["gripper"] == pytest.approx(start + teleop.config.incremental_step)


def test_binary_gripper_toggles_on_rising_edge(monkeypatch):
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    teleop = SpaceMouseTeleop(
        SpaceMouseTeleopConfig(tool_mode=SpaceMouseToolMode.BINARY, initial_gripper=1.0)
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
                buttons=(True, False),
            ),
        ]
    )
    first = teleop.get_action()
    second = teleop.get_action()
    assert first["gripper"] == pytest.approx(0.0)
    assert second["gripper"] == pytest.approx(0.0)
    teleop.disconnect()
