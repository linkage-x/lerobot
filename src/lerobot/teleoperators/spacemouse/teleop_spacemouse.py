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

from __future__ import annotations

import time
from functools import cached_property
from typing import TYPE_CHECKING, Any

import numpy as np

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
if TYPE_CHECKING:
    from lerobot.processor import RobotAction
else:
    RobotAction = dict[str, Any]

from .backend import PySpaceMouseDriver
from .configuration_spacemouse import SpaceMouseEnableButton, SpaceMouseTeleopConfig, SpaceMouseToolMode


class SpaceMouseTeleop(Teleoperator):
    config_class = SpaceMouseTeleopConfig
    name = "spacemouse"

    driver_cls = PySpaceMouseDriver
    TRANSLATION_AXIS_CALIBRATION = np.array(
        [1.0, 0.9414634146341463, 0.5902439024390244],
        dtype=np.float64,
    )
    ROTATION_AXIS_CALIBRATION = np.array(
        [1.0, 0.9490740740740741, 0.9259259259259259],
        dtype=np.float64,
    )

    def __init__(self, config: SpaceMouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self._driver = None
        self._is_connected = False
        self._motion_active = False
        self._last_gripper = float(np.clip(config.initial_gripper, 0.0, 1.0))
        self._last_gripper_update = 0.0
        self._filtered_gripper = self._last_gripper
        self._last_filtered_gripper_time = float("-inf")
        self._last_button_raw: np.ndarray | None = None
        self._debounced_buttons = np.zeros(2, dtype=np.float64)
        self._button_change_time = 0.0
        self._button_press_times = np.full(2, float("-inf"), dtype=np.float64)
        self._translation_bias = np.zeros(3, dtype=np.float64)
        self._rotation_bias = np.zeros(3, dtype=np.float64)
        # The timestamp of the report the last action was computed from. None until one arrives.
        self._last_report_timestamp: float | None = None
        # State-change tracking for targeted debug logging
        self._prev_motion_detected = False
        self._prev_motion_enabled = False
        self._prev_enabled_out = False
        self._prev_motion_active = False
        self._log_count = 0

    @cached_property
    def action_features(self) -> dict[str, type]:
        return {
            "enabled": bool,
            "target_x": float,
            "target_y": float,
            "target_z": float,
            "target_wx": float,
            "target_wy": float,
            "target_wz": float,
            "gripper": float,
        }

    @cached_property
    def feedback_features(self) -> dict[str, type]:
        return {}

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def last_report_timestamp(self) -> float | None:
        """When the device last said anything, as the driver dates it. None if it never has.

        The driver returns its cached state rather than nothing when no report arrived, which is
        the right default for teleoperation -- a puck held off centre is a rate command that
        should keep applying between reports. A caller that has to know whether the operator is
        still there cannot read that from the axes: this value stops advancing when the device
        goes quiet, and is the only thing that does.
        """
        return self._last_report_timestamp

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        driver = self.driver_cls(device_id=self.config.device_id)
        try:
            driver.connect()
        except Exception:
            try:
                driver.disconnect()
            except Exception:
                pass
            raise
        self._driver = driver
        self._translation_bias, self._rotation_bias = self._estimate_idle_bias()
        self._is_connected = True

    def _estimate_idle_bias(self) -> tuple[np.ndarray, np.ndarray]:
        if self._driver is None or self.config.bias_sample_count <= 0:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

        translations: list[np.ndarray] = []
        rotations: list[np.ndarray] = []
        for _ in range(self.config.bias_sample_count):
            reading = self._driver.poll()
            if reading is not None:
                translations.append(np.asarray(reading.translation, dtype=np.float64))
                rotations.append(np.asarray(reading.rotation, dtype=np.float64))
            if self.config.bias_sample_sleep_s > 0.0:
                time.sleep(self.config.bias_sample_sleep_s)

        if not translations:
            return np.zeros(3, dtype=np.float64), np.zeros(3, dtype=np.float64)

        return (
            np.median(np.stack(translations, axis=0), axis=0),
            np.median(np.stack(rotations, axis=0), axis=0),
        )

    @property
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    def _zero_action(self) -> RobotAction:
        return {
            "enabled": False,
            "target_x": 0.0,
            "target_y": 0.0,
            "target_z": 0.0,
            "target_wx": 0.0,
            "target_wy": 0.0,
            "target_wz": 0.0,
            "gripper": self._filtered_gripper,
        }

    def _motion_threshold_vector(self) -> np.ndarray:
        return np.array(
            [
                self.config.threshold_x,
                self.config.threshold_y,
                self.config.threshold_z,
                self.config.threshold_wx,
                self.config.threshold_wy,
                self.config.threshold_wz,
            ],
            dtype=np.float64,
        )

    def _reading_motion_data(self, reading) -> np.ndarray:
        translation = np.asarray(reading.translation, dtype=np.float64) - self._translation_bias
        rotation = np.asarray(reading.rotation, dtype=np.float64) - self._rotation_bias
        translation = self.translation_axis_map @ translation
        rotation = self.rotation_axis_map @ rotation
        return np.array(
            [
                translation[0],
                translation[1],
                translation[2],
                rotation[0],
                rotation[1],
                rotation[2],
            ],
            dtype=np.float64,
        )

    def _reading_has_motion(self, reading) -> bool:
        if reading is None:
            return False
        return bool(np.any(np.abs(self._reading_motion_data(reading)) >= self._motion_threshold_vector()))

    def _update_gripper(self, button_0: bool, button_1: bool) -> None:
        if self.config.tool_mode == SpaceMouseToolMode.BINARY:
            if button_0 and not button_1:
                self._last_gripper = 1.0
            elif button_1 and not button_0:
                self._last_gripper = 0.0
        else:
            now = time.perf_counter()
            if now - self._last_gripper_update >= self.config.move_time:
                if button_0 and not button_1:
                    self._last_gripper = min(1.0, self._last_gripper + self.config.incremental_step)
                    self._last_gripper_update = now
                elif button_1 and not button_0:
                    self._last_gripper = max(0.0, self._last_gripper - self.config.incremental_step)
                    self._last_gripper_update = now

    def _apply_button_filters(self, buttons: tuple[bool, bool] | np.ndarray) -> tuple[bool, bool]:
        filtered = np.asarray(buttons, dtype=np.float64).copy()
        now = time.perf_counter()

        if self.config.button_debounce_s > 0.0:
            if self._last_button_raw is None:
                self._last_button_raw = filtered.copy()
                self._debounced_buttons = filtered.copy()
                self._button_change_time = now
            else:
                if not np.array_equal(filtered, self._last_button_raw):
                    self._last_button_raw = filtered.copy()
                    self._button_change_time = now
                if (now - self._button_change_time) >= self.config.button_debounce_s:
                    self._debounced_buttons = self._last_button_raw.copy()
            filtered = self._debounced_buttons.copy()

        if self.config.button_release_grace_s > 0.0:
            for idx in range(len(filtered)):
                if filtered[idx]:
                    self._button_press_times[idx] = now
                elif (now - self._button_press_times[idx]) <= self.config.button_release_grace_s:
                    filtered[idx] = 1.0

        return bool(filtered[0]), bool(filtered[1])

    def _filter_gripper_command(self, value: float) -> float:
        raw_value = float(np.clip(value, 0.0, 1.0))
        now = time.perf_counter()
        last_value = float(self._filtered_gripper)
        last_time = float(self._last_filtered_gripper_time)

        if not np.isfinite(last_time):
            filtered = raw_value
        else:
            filtered = raw_value
            delta = abs(filtered - last_value)
            if self.config.gripper_cmd_min_delta > 0.0 and delta < self.config.gripper_cmd_min_delta:
                if self.config.gripper_cmd_min_interval_s > 0.0 and (now - last_time) < self.config.gripper_cmd_min_interval_s:
                    return last_value
                filtered = last_value

            if self.config.gripper_cmd_max_rate > 0.0:
                step_dt = 1.0 / max(float(self.config.frequency), 1.0)
                max_delta = self.config.gripper_cmd_max_rate * step_dt
                delta = filtered - last_value
                if abs(delta) > max_delta:
                    filtered = last_value + np.sign(delta) * max_delta

            if self.config.gripper_cmd_ema_alpha > 0.0:
                alpha = float(np.clip(self.config.gripper_cmd_ema_alpha, 0.0, 1.0))
                filtered = alpha * filtered + (1.0 - alpha) * last_value

        self._filtered_gripper = float(np.clip(filtered, 0.0, 1.0))
        self._last_filtered_gripper_time = now
        return self._filtered_gripper

    @property
    def translation_scale_vector(self) -> np.ndarray:
        default_vector = (
            float(self.config.translation_scale)
            * self.TRANSLATION_AXIS_CALIBRATION
        )
        overrides = np.array(
            [
                self.config.scale_x,
                self.config.scale_y,
                self.config.scale_z,
            ],
            dtype=np.float64,
        )
        return np.where(np.isnan(overrides), default_vector, overrides)

    @cached_property
    def translation_axis_map(self) -> np.ndarray:
        matrix = np.asarray(self.config.translation_axis_map, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"translation_axis_map must be 3x3, got {matrix.shape}")
        return matrix

    @cached_property
    def rotation_axis_map(self) -> np.ndarray:
        matrix = np.asarray(self.config.rotation_axis_map, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"rotation_axis_map must be 3x3, got {matrix.shape}")
        return matrix

    @property
    def rotation_scale_vector(self) -> np.ndarray:
        default_vector = (
            float(self.config.rotation_scale)
            * self.ROTATION_AXIS_CALIBRATION
        )
        overrides = np.array(
            [
                self.config.scale_wx,
                self.config.scale_wy,
                self.config.scale_wz,
            ],
            dtype=np.float64,
        )
        return np.where(np.isnan(overrides), default_vector, overrides)

    @check_if_not_connected
    def wait_until_idle(
        self,
        *,
        consecutive_samples: int = 3,
        timeout_s: float | None = None,
        poll_interval_s: float | None = None,
    ) -> bool:
        if consecutive_samples < 1:
            raise ValueError("consecutive_samples must be >= 1")

        interval_s = poll_interval_s if poll_interval_s is not None else 1.0 / max(self.config.frequency, 1)
        deadline = None if timeout_s is None else time.perf_counter() + timeout_s
        idle_samples = 0

        while True:
            reading = self._driver.poll()
            if self._reading_has_motion(reading):
                idle_samples = 0
            else:
                idle_samples += 1
                if idle_samples >= consecutive_samples:
                    return True

            if deadline is not None and time.perf_counter() >= deadline:
                return False

            if interval_s > 0.0:
                time.sleep(interval_s)

    def set_gripper(self, normalized_position: float) -> None:
        self.sync_gripper_baseline(normalized_position)

    def sync_gripper_baseline(self, normalized_position: float) -> float:
        value = float(np.clip(normalized_position, 0.0, 1.0))
        now = time.perf_counter()
        self._last_gripper = value
        self._filtered_gripper = value
        self._last_gripper_update = now
        self._last_filtered_gripper_time = now
        return value

    def sync_tool_activity_baseline(self, normalized_command: float) -> float:
        return self.sync_gripper_baseline(normalized_command)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        reading = self._driver.poll()
        # Left alone when a poll brings nothing back, because "no report" is exactly what an
        # unchanged timestamp already means -- overwriting it with None would erase the last
        # thing the device did say.
        report_timestamp = getattr(reading, "timestamp", None)
        if report_timestamp is not None:
            self._last_report_timestamp = float(report_timestamp)
        if reading is None:
            self._motion_active = False
            # Log if enabled output just went True->False (transition to stop)
            if self._prev_enabled_out and not self._prev_enabled_out:  # was True, now False
                pass  # transition already logged below
            enabled_out = False
            self._prev_enabled_out = enabled_out
            self._prev_motion_detected = False
            self._prev_motion_enabled = False
            self._prev_motion_active = False
            return self._zero_action()

        data = self._reading_motion_data(reading)
        button_0, button_1 = self._apply_button_filters(reading.buttons)
        threshold = self._motion_threshold_vector()
        scale = np.concatenate((self.translation_scale_vector, self.rotation_scale_vector))
        abs_data = np.abs(data)
        active_mask = abs_data >= threshold
        data = np.where(active_mask, data, 0.0)
        # Match the HIROL 3D-mouse teleop behavior more closely: once each axis is
        # deadbanded, motion enable/disable is determined directly from the zeroed
        # delta instead of a separate hysteresis/release-decay state machine.
        motion_detected = bool(np.any(active_mask))
        if self.config.motion_enable_button == SpaceMouseEnableButton.LEFT:
            motion_enabled = motion_detected and button_0
        elif self.config.motion_enable_button == SpaceMouseEnableButton.RIGHT:
            motion_enabled = motion_detected and button_1
        else:
            motion_enabled = motion_detected
        self._motion_active = motion_detected

        if not motion_enabled:
            self._update_gripper(button_0, button_1)
            self._filter_gripper_command(self._last_gripper)
            enabled_out = False
            self._prev_motion_detected = motion_detected
            self._prev_motion_enabled = motion_enabled
            self._prev_enabled_out = enabled_out
            self._prev_motion_active = motion_detected
            return self._zero_action()

        target = data * scale
        if not self.config.enable_rotation:
            target[3:] = 0.0

        self._update_gripper(button_0, button_1)
        filtered_gripper = self._filter_gripper_command(self._last_gripper)
        enabled_out = True
        self._prev_motion_detected = motion_detected
        self._prev_motion_enabled = motion_enabled
        self._prev_enabled_out = enabled_out
        self._prev_motion_active = motion_detected
        return {
            "enabled": True,
            "target_x": float(target[0]),
            "target_y": float(target[1]),
            "target_z": float(target[2]),
            "target_wx": float(target[3]),
            "target_wy": float(target[4]),
            "target_wz": float(target[5]),
            "gripper": filtered_gripper,
        }

    def send_feedback(self, feedback: dict[str, Any]) -> None:
        del feedback

    @check_if_not_connected
    def disconnect(self) -> None:
        try:
            if self._driver is not None:
                try:
                    self._driver.disconnect()
                except Exception:
                    pass
        finally:
            self._driver = None
            self._is_connected = False
            self._motion_active = False
            self._last_report_timestamp = None
            self._last_button_raw = None
            self._debounced_buttons.fill(0.0)
            self._button_change_time = 0.0
            self._button_press_times.fill(float("-inf"))
            self._translation_bias = np.zeros(3, dtype=np.float64)
            self._rotation_bias = np.zeros(3, dtype=np.float64)
