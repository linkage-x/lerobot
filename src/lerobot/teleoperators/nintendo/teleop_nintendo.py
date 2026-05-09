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
from typing import Any

import numpy as np

from lerobot.processor import RobotAction
from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected

from ..teleoperator import Teleoperator
from .backend import NintendoControllerReading, NintendoHIDDriver
from .configuration_nintendo import NintendoGripperMode, NintendoTeleopConfig


class NintendoTeleop(Teleoperator):
    config_class = NintendoTeleopConfig
    name = "nintendo"

    driver_cls = NintendoHIDDriver

    def __init__(self, config: NintendoTeleopConfig):
        super().__init__(config)
        self.config = config
        self._driver = None
        self._is_connected = False
        self._last_reading: NintendoControllerReading | None = None
        self._last_update_s = float("-inf")
        self._last_gripper = float(np.clip(config.initial_gripper, 0.0, 1.0))
        self._filtered_gripper = self._last_gripper
        self._last_gripper_update_s = float("-inf")
        self._last_filtered_gripper_time = float("-inf")

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
    def is_calibrated(self) -> bool:
        return True

    def calibrate(self) -> None:
        pass

    def configure(self) -> None:
        pass

    @check_if_already_connected
    def connect(self, calibrate: bool = True) -> None:
        del calibrate
        driver = self.driver_cls(
            controller=str(self.config.controller),
            side=str(self.config.side),
            device_id=self.config.device_id,
            read_timeout_ms=self.config.read_timeout_ms,
        )
        try:
            driver.connect()
        except Exception:
            try:
                driver.disconnect()
            except Exception:
                pass
            raise
        self._driver = driver
        self._is_connected = True

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

    def _latest_reading(self) -> NintendoControllerReading | None:
        if self._driver is None:
            return None
        reading = self._driver.poll()
        now = time.perf_counter()
        if reading is not None:
            self._last_reading = reading
            self._last_update_s = now
            return reading
        if self._last_reading is not None and now - self._last_update_s <= float(self.config.stale_timeout_s):
            return self._last_reading
        return None

    @staticmethod
    def _deadband(value: float, threshold: float) -> float:
        value = float(value)
        threshold = float(abs(threshold))
        if abs(value) < threshold:
            return 0.0
        if threshold >= 1.0:
            return float(np.clip(value, -1.0, 1.0))
        scaled = (abs(value) - threshold) / (1.0 - threshold)
        return float(np.sign(value) * np.clip(scaled, 0.0, 1.0))

    @staticmethod
    def _button_axis(buttons: frozenset[str], positive: tuple[str, ...], negative: tuple[str, ...]) -> float:
        pos = any(button in buttons for button in positive)
        neg = any(button in buttons for button in negative)
        if pos and not neg:
            return 1.0
        if neg and not pos:
            return -1.0
        return 0.0

    def _primary_stick(self, reading: NintendoControllerReading) -> tuple[float, float]:
        if reading.controller_type == "right":
            return reading.right_stick
        return reading.left_stick

    def _motion_inputs(self, reading: NintendoControllerReading) -> np.ndarray:
        primary_x, primary_y = self._primary_stick(reading)
        right_x, right_y = reading.right_stick
        buttons = reading.buttons

        x = primary_y
        y = primary_x
        z = right_y if reading.controller_type == "pro" else 0.0
        wz = right_x if reading.controller_type == "pro" else 0.0

        z_button_axis = self._button_axis(buttons, self.config.z_up_buttons, self.config.z_down_buttons)
        yaw_button_axis = self._button_axis(
            buttons,
            self.config.yaw_positive_buttons,
            self.config.yaw_negative_buttons,
        )
        if z_button_axis != 0.0:
            z = z_button_axis
        if yaw_button_axis != 0.0:
            wz = yaw_button_axis

        values = np.array([x, y, z, 0.0, 0.0, wz], dtype=np.float64)
        threshold = float(self.config.stick_deadband)
        values = np.array([self._deadband(value, threshold) for value in values], dtype=np.float64)
        if self.config.invert_x:
            values[0] *= -1.0
        if self.config.invert_y:
            values[1] *= -1.0
        if self.config.invert_z:
            values[2] *= -1.0
        if self.config.invert_wz:
            values[5] *= -1.0
        return values

    def _scale_vector(self) -> np.ndarray:
        defaults = np.array(
            [
                self.config.translation_scale,
                self.config.translation_scale,
                self.config.vertical_scale,
                self.config.rotation_scale,
                self.config.rotation_scale,
                self.config.rotation_scale,
            ],
            dtype=np.float64,
        )

        def _optional_float(value: float | None) -> float:
            return np.nan if value is None else float(value)

        overrides = np.array(
            [
                _optional_float(self.config.scale_x),
                _optional_float(self.config.scale_y),
                _optional_float(self.config.scale_z),
                _optional_float(self.config.scale_wx),
                _optional_float(self.config.scale_wy),
                _optional_float(self.config.scale_wz),
            ],
            dtype=np.float64,
        )
        return np.where(np.isnan(overrides), defaults, overrides)

    def _clutch_active(self, buttons: frozenset[str]) -> bool:
        if not self.config.clutch_buttons:
            return True
        return any(button in buttons for button in self.config.clutch_buttons)

    def _update_gripper(self, buttons: frozenset[str]) -> None:
        close_pressed = any(button in buttons for button in self.config.gripper_close_buttons)
        open_pressed = any(button in buttons for button in self.config.gripper_open_buttons)
        if close_pressed == open_pressed:
            return

        now = time.perf_counter()
        if self.config.gripper_mode == NintendoGripperMode.BINARY:
            self._last_gripper = 0.0 if close_pressed else 1.0
            self._last_gripper_update_s = now
            return

        if now - self._last_gripper_update_s < float(self.config.gripper_move_time):
            return
        step = float(self.config.gripper_step)
        if close_pressed:
            self._last_gripper = max(0.0, self._last_gripper - step)
        else:
            self._last_gripper = min(1.0, self._last_gripper + step)
        self._last_gripper_update_s = now

    def _filter_gripper_command(self, value: float) -> float:
        raw_value = float(np.clip(value, 0.0, 1.0))
        now = time.perf_counter()
        last_value = float(self._filtered_gripper)
        last_time = float(self._last_filtered_gripper_time)
        filtered = raw_value
        if np.isfinite(last_time):
            if self.config.gripper_cmd_max_rate > 0.0:
                step_dt = 1.0 / max(float(self.config.frequency), 1.0)
                max_delta = float(self.config.gripper_cmd_max_rate) * step_dt
                delta = filtered - last_value
                if abs(delta) > max_delta:
                    filtered = last_value + np.sign(delta) * max_delta
            if self.config.gripper_cmd_ema_alpha > 0.0:
                alpha = float(np.clip(self.config.gripper_cmd_ema_alpha, 0.0, 1.0))
                filtered = alpha * filtered + (1.0 - alpha) * last_value
        self._filtered_gripper = float(np.clip(filtered, 0.0, 1.0))
        self._last_filtered_gripper_time = now
        return self._filtered_gripper

    def sync_gripper_baseline(self, normalized_position: float) -> float:
        value = float(np.clip(normalized_position, 0.0, 1.0))
        now = time.perf_counter()
        self._last_gripper = value
        self._filtered_gripper = value
        self._last_gripper_update_s = now
        self._last_filtered_gripper_time = now
        return value

    def set_gripper(self, normalized_position: float) -> None:
        self.sync_gripper_baseline(normalized_position)

    def latest_debug_state(self) -> dict[str, Any]:
        reading = self._last_reading
        return {
            "connected": self.is_connected,
            "controller_type": None if reading is None else reading.controller_type,
            "buttons": () if reading is None else tuple(sorted(reading.buttons)),
            "left_stick": (0.0, 0.0) if reading is None else reading.left_stick,
            "right_stick": (0.0, 0.0) if reading is None else reading.right_stick,
            "last_update_age_s": (
                float("inf")
                if not np.isfinite(self._last_update_s)
                else time.perf_counter() - self._last_update_s
            ),
            "gripper": self._filtered_gripper,
        }

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
        interval_s = (
            poll_interval_s
            if poll_interval_s is not None
            else 1.0 / max(int(self.config.frequency), 1)
        )
        deadline = None if timeout_s is None else time.perf_counter() + timeout_s
        idle_samples = 0
        while True:
            reading = self._latest_reading()
            motion = False
            if reading is not None:
                motion = bool(np.any(np.abs(self._motion_inputs(reading)) > 0.0))
            if motion:
                idle_samples = 0
            else:
                idle_samples += 1
                if idle_samples >= consecutive_samples:
                    return True

            if deadline is not None and time.perf_counter() >= deadline:
                return False
            if interval_s > 0.0:
                time.sleep(interval_s)

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        reading = self._latest_reading()
        if reading is None:
            return self._zero_action()

        self._update_gripper(reading.buttons)
        filtered_gripper = self._filter_gripper_command(self._last_gripper)
        inputs = self._motion_inputs(reading)
        if not self.config.enable_rotation:
            inputs[3:] = 0.0
        target = inputs * self._scale_vector()
        enabled = self._clutch_active(reading.buttons) and bool(np.any(np.abs(target) > 0.0))
        if not enabled:
            action = self._zero_action()
            action["gripper"] = filtered_gripper
            return action

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
            self._last_reading = None
            self._last_update_s = float("-inf")
