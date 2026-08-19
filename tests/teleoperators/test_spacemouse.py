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
    assert action["target_x"] == pytest.approx(0.3 * teleop.translation_scale_vector[0])
    assert action["target_y"] == pytest.approx(0.2 * teleop.translation_scale_vector[1])
    assert action["target_z"] == pytest.approx(0.4 * teleop.translation_scale_vector[2])
    # rotation_axis_map negates roll: +raw_wx is a negative rotation about the tool x axis.
    assert action["target_wx"] == pytest.approx(-0.5 * teleop.rotation_scale_vector[0])
    assert action["target_wy"] == pytest.approx(-0.6 * teleop.rotation_scale_vector[1])
    assert action["target_wz"] == pytest.approx(0.7 * teleop.rotation_scale_vector[2])


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
    assert action["target_y"] == pytest.approx(0.03 * teleop.translation_scale_vector[1])
    assert action["target_z"] == pytest.approx(0.0)
    # rotation_axis_map negates roll: +raw_wx is a negative rotation about the tool x axis.
    assert action["target_wx"] == pytest.approx(-0.05 * teleop.rotation_scale_vector[0])
    assert action["target_wy"] == pytest.approx(0.0)
    assert action["target_wz"] == pytest.approx(0.0)
    teleop.disconnect()


def test_get_action_uses_explicit_axis_map_instead_of_hidden_legacy_remap(monkeypatch):
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    teleop = SpaceMouseTeleop(
        SpaceMouseTeleopConfig(
            translation_axis_map=((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0)),
            bias_sample_count=0,
            move_time=0.0,
        )
    )
    teleop.connect()
    teleop._driver.readings.append(
        SpaceMouseReading(
            translation=[0.2, -0.3, 0.4],
            rotation=[0.0, 0.0, 0.0],
            buttons=(False, False),
        )
    )

    action = teleop.get_action()

    assert action["target_x"] == pytest.approx(0.2 * teleop.translation_scale_vector[0])
    assert action["target_y"] == pytest.approx(-0.3 * teleop.translation_scale_vector[1])
    assert action["target_z"] == pytest.approx(0.4 * teleop.translation_scale_vector[2])
    teleop.disconnect()


def test_get_action_enables_motion_as_soon_as_an_axis_clears_its_threshold(monkeypatch):
    # Motion enable comes straight from the per-axis deadband: there is no extra
    # enter margin on top of `threshold_*`, so a reading just above it already moves.
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
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
    teleop._driver.readings.extend(
        [
            SpaceMouseReading(
                translation=[0.028, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.019, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
        ]
    )

    above_threshold = teleop.get_action()
    below_threshold = teleop.get_action()

    assert above_threshold["enabled"] is True
    # translation_axis_map maps raw x onto the command frame's -y.
    assert above_threshold["target_y"] == pytest.approx(0.028 * teleop.translation_scale_vector[1])
    assert below_threshold["enabled"] is False
    assert below_threshold["target_y"] == pytest.approx(0.0)
    teleop.disconnect()


def test_get_action_keeps_motion_enabled_until_the_axis_drops_below_its_threshold(monkeypatch):
    # Releasing the puck only stops motion once the axis re-enters the deadband;
    # values that merely shrink but stay above `threshold_*` keep commanding.
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
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
    teleop._driver.readings.extend(
        [
            SpaceMouseReading(
                translation=[0.03, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.022, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.012, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
        ]
    )

    entered = teleop.get_action()
    still_above = teleop.get_action()
    released = teleop.get_action()

    assert entered["enabled"] is True
    assert still_above["enabled"] is True
    assert released["enabled"] is False
    teleop.disconnect()


def test_get_action_follows_high_amplitude_release_decay_down_to_the_deadband(monkeypatch):
    # pyspacemouse emits a decaying tail after the puck is let go. Motion stays
    # enabled through the tail and stops on the first sample inside the deadband.
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
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
    teleop._driver.readings.extend(
        [
            SpaceMouseReading(
                translation=[0.0, -0.30, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.0, -0.29, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.0, -0.08, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.0, -0.015, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
        ]
    )

    first = teleop.get_action()
    plateau = teleop.get_action()
    decaying = teleop.get_action()
    released = teleop.get_action()

    assert first["enabled"] is True
    assert plateau["enabled"] is True
    assert decaying["enabled"] is True
    assert released["enabled"] is False
    teleop.disconnect()


def test_get_action_keeps_gradual_same_direction_reduction_enabled(monkeypatch):
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
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
    teleop._driver.readings.extend(
        [
            SpaceMouseReading(
                translation=[0.0, -0.30, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.0, -0.27, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.0, -0.22, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
        ]
    )

    teleop.get_action()
    second = teleop.get_action()
    third = teleop.get_action()

    assert second["enabled"] is True
    assert third["enabled"] is True
    teleop.disconnect()


def test_get_action_ignores_cross_axis_jitter_below_its_own_threshold(monkeypatch):
    # A jittering x that never clears threshold_x contributes nothing on its own;
    # enable state tracks the dominant y axis, and stops when y enters the deadband.
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
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
    # translation_axis_map is [-raw_y, raw_x, raw_z], so thresholds apply in the
    # command frame: raw_y drives target_x, and the raw_x jitter lands on target_y.
    teleop._driver.readings.extend(
        [
            SpaceMouseReading(
                translation=[0.010, -0.30, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.010, -0.29, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.015, -0.012, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
        ]
    )

    first = teleop.get_action()
    second = teleop.get_action()
    released = teleop.get_action()

    assert first["enabled"] is True
    assert second["enabled"] is True
    # The sub-threshold jitter is deadbanded away instead of leaking into the command.
    assert first["target_y"] == pytest.approx(0.0)
    assert second["target_y"] == pytest.approx(0.0)
    # Once the dominant axis also re-enters its deadband, nothing is left to enable.
    assert released["enabled"] is False
    teleop.disconnect()


def test_per_dof_scale_overrides_can_isolate_wz_rotation(monkeypatch):
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    teleop = SpaceMouseTeleop(
        SpaceMouseTeleopConfig(
            translation_scale=0.0,
            scale_wx=0.0,
            scale_wy=0.0,
            bias_sample_count=0,
            move_time=0.0,
        )
    )
    teleop.connect()
    teleop._driver.readings.append(
        SpaceMouseReading(
            translation=[0.2, -0.3, 0.4],
            rotation=[0.5, -0.6, 0.7],
            buttons=(False, False),
        )
    )

    action = teleop.get_action()

    assert action["target_x"] == pytest.approx(0.0)
    assert action["target_y"] == pytest.approx(0.0)
    assert action["target_z"] == pytest.approx(0.0)
    assert action["target_wx"] == pytest.approx(0.0)
    assert action["target_wy"] == pytest.approx(0.0)
    assert action["target_wz"] == pytest.approx(0.7 * teleop.rotation_scale_vector[2])
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


def test_wait_until_idle_drains_stale_motion_before_returning(monkeypatch):
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    teleop = SpaceMouseTeleop(
        SpaceMouseTeleopConfig(
            bias_sample_count=0,
            move_time=0.0,
            threshold_x=0.02,
            threshold_y=0.02,
            threshold_z=0.02,
            threshold_wx=0.04,
            threshold_wy=0.04,
            threshold_wz=0.04,
        )
    )
    teleop.connect()
    teleop._driver.readings.extend(
        [
            SpaceMouseReading(
                translation=[0.0, -0.3, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.0, -0.3, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            SpaceMouseReading(
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
            None,
        ]
    )

    assert teleop.wait_until_idle(consecutive_samples=2, poll_interval_s=0.0) is True

    teleop.disconnect()


def test_wait_until_idle_times_out_when_motion_persists(monkeypatch):
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    teleop = SpaceMouseTeleop(
        SpaceMouseTeleopConfig(
            bias_sample_count=0,
            move_time=0.0,
        )
    )
    teleop.connect()
    teleop._driver.readings.append(
        SpaceMouseReading(
            translation=[0.0, -0.3, 0.0],
            rotation=[0.0, 0.0, 0.0],
            buttons=(False, False),
        )
    )

    assert teleop.wait_until_idle(consecutive_samples=1, timeout_s=0.0, poll_interval_s=0.0) is False

    teleop.disconnect()


def test_incremental_gripper_updates(teleop):
    teleop.connect()
    teleop._last_gripper = 0.5
    teleop.sync_gripper_baseline(0.5)
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


def test_sync_gripper_baseline_updates_internal_and_output_state(monkeypatch):
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    teleop = SpaceMouseTeleop(
        SpaceMouseTeleopConfig(
            bias_sample_count=0,
            gripper_cmd_ema_alpha=0.9,
            gripper_cmd_max_rate=12.0,
        )
    )
    teleop.connect()

    synced = teleop.sync_gripper_baseline(0.25)
    action = teleop.get_action()

    assert synced == pytest.approx(0.25)
    assert teleop._last_gripper == pytest.approx(0.25)
    assert action["gripper"] == pytest.approx(0.25)
    teleop.disconnect()


def test_gripper_filter_applies_rate_limit_and_ema(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    monkeypatch.setattr("lerobot.teleoperators.spacemouse.teleop_spacemouse.time.perf_counter", lambda: clock["now"])
    teleop = SpaceMouseTeleop(
        SpaceMouseTeleopConfig(
            bias_sample_count=0,
            tool_mode=SpaceMouseToolMode.BINARY,
            initial_gripper=0.0,
            gripper_cmd_ema_alpha=0.9,
            gripper_cmd_max_rate=12.0,
        )
    )
    teleop.connect()
    teleop.sync_gripper_baseline(0.0)
    teleop._driver.readings.append(
        SpaceMouseReading(
            translation=[0.0, 0.0, 0.0],
            rotation=[0.0, 0.0, 0.0],
            buttons=(True, False),
        )
    )

    clock["now"] = 0.005
    action = teleop.get_action()

    assert action["enabled"] is False
    assert action["gripper"] == pytest.approx(0.054, abs=1e-9)
    assert teleop._last_gripper == pytest.approx(1.0)
    teleop.disconnect()


def test_button_release_grace_keeps_incremental_gripper_active_briefly(monkeypatch):
    clock = {"now": 0.0}
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    monkeypatch.setattr("lerobot.teleoperators.spacemouse.teleop_spacemouse.time.perf_counter", lambda: clock["now"])
    teleop = SpaceMouseTeleop(
        SpaceMouseTeleopConfig(
            bias_sample_count=0,
            tool_mode=SpaceMouseToolMode.INCREMENTAL,
            initial_gripper=0.5,
            incremental_step=0.02,
            move_time=0.0,
            button_release_grace_s=0.01,
        )
    )
    teleop.connect()
    teleop.sync_gripper_baseline(0.5)
    teleop._driver.readings.extend(
        [
            SpaceMouseReading(
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(True, False),
            ),
            SpaceMouseReading(
                translation=[0.0, 0.0, 0.0],
                rotation=[0.0, 0.0, 0.0],
                buttons=(False, False),
            ),
        ]
    )

    clock["now"] = 0.001
    first = teleop.get_action()
    clock["now"] = 0.006
    second = teleop.get_action()

    assert first["gripper"] == pytest.approx(0.52)
    assert second["gripper"] == pytest.approx(0.54)
    teleop.disconnect()


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


# ------------------------------------------------------------------- rotation axis convention ---
#
# Roll and pitch were pinned off (`scale_wx: 0.0` / `scale_wy: 0.0` in fr3_record_config.yaml) for
# as long as the rig recorded against pika_task_tcp, because rotating about a frame 411 mm behind
# the fingers swung them through an arc. Nothing exercised those two axes in that whole period, so
# an inverted roll sat in the default map unnoticed until the switch to pika_gripper_ee turned them
# back on and an operator felt the tool roll the wrong way. These pin the convention so it cannot
# drift back silently -- there is no way to notice it from reading the code.


def test_roll_is_inverted_relative_to_the_raw_device_axis(monkeypatch):
    """+raw_wx must command a *negative* rotation about the tool x axis."""
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    teleop = SpaceMouseTeleop(SpaceMouseTeleopConfig(bias_sample_count=0, move_time=0.0))
    teleop.connect()
    teleop._driver.readings.append(
        SpaceMouseReading(translation=[0.0, 0.0, 0.0], rotation=[0.5, 0.0, 0.0], buttons=(False, False))
    )

    action = teleop.get_action()

    assert action["target_wx"] == pytest.approx(-0.5 * teleop.rotation_scale_vector[0])
    assert action["target_wy"] == pytest.approx(0.0)
    assert action["target_wz"] == pytest.approx(0.0)
    teleop.disconnect()


def test_pitch_and_yaw_are_not_inverted(monkeypatch):
    """Only roll is flipped. Yaw was live throughout and pitch was measured with it."""
    monkeypatch.setattr(SpaceMouseTeleop, "driver_cls", DummySpaceMouseDriver)
    teleop = SpaceMouseTeleop(SpaceMouseTeleopConfig(bias_sample_count=0, move_time=0.0))
    teleop.connect()
    teleop._driver.readings.append(
        SpaceMouseReading(translation=[0.0, 0.0, 0.0], rotation=[0.0, 0.5, 0.4], buttons=(False, False))
    )

    action = teleop.get_action()

    assert action["target_wy"] == pytest.approx(0.5 * teleop.rotation_scale_vector[1])
    assert action["target_wz"] == pytest.approx(0.4 * teleop.rotation_scale_vector[2])
    assert action["target_wx"] == pytest.approx(0.0)
    teleop.disconnect()


def test_the_rotation_map_does_not_borrow_the_translation_map(monkeypatch):
    """The two maps reach different frames, so they must not be kept in step with each other.

    `target_{x,y,z}` is added to the reference position in the *base* frame; `target_{wx,wy,wz}` is
    right-multiplied onto the reference orientation, i.e. applied about the *tool* axes. A future
    reader tidying these two into one matrix would silently re-aim every rotation command.
    """
    config = SpaceMouseTeleopConfig()

    assert config.rotation_axis_map != config.translation_axis_map
    # A pure sign flip on roll: no axis swapping, which would be a different (and wrong) fix.
    assert config.rotation_axis_map == ((-1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))

