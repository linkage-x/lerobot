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
        self._last_gripper = float(np.clip(config.initial_gripper, 0.0, 1.0))
        self._last_gripper_update = 0.0
        self._translation_bias = np.zeros(3, dtype=np.float64)
        self._rotation_bias = np.zeros(3, dtype=np.float64)

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
            "gripper": self._last_gripper,
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
        return np.array(
            [
                -(reading.translation[1] - self._translation_bias[1]),
                reading.translation[0] - self._translation_bias[0],
                reading.translation[2] - self._translation_bias[2],
                reading.rotation[0] - self._rotation_bias[0],
                reading.rotation[1] - self._rotation_bias[1],
                reading.rotation[2] - self._rotation_bias[2],
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

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        reading = self._driver.poll()
        if reading is None:
            return self._zero_action()

        data = self._reading_motion_data(reading)
        threshold = self._motion_threshold_vector()
        scale = np.concatenate((self.translation_scale_vector, self.rotation_scale_vector))
        active_mask = np.abs(data) >= threshold
        data = np.where(active_mask, data, 0.0)
        motion_detected = bool(np.any(active_mask))
        if self.config.motion_enable_button == SpaceMouseEnableButton.LEFT:
            motion_enabled = motion_detected and bool(reading.buttons[0])
        elif self.config.motion_enable_button == SpaceMouseEnableButton.RIGHT:
            motion_enabled = motion_detected and bool(reading.buttons[1])
        else:
            motion_enabled = motion_detected

        if not motion_enabled:
            self._update_gripper(*reading.buttons)
            return self._zero_action()

        target = data * scale
        if not self.config.enable_rotation:
            target[3:] = 0.0

        self._update_gripper(*reading.buttons)
        return {
            "enabled": True,
            "target_x": float(target[0]),
            "target_y": float(target[1]),
            "target_z": float(target[2]),
            "target_wx": float(target[3]),
            "target_wy": float(target[4]),
            "target_wz": float(target[5]),
            "gripper": self._last_gripper,
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
            self._translation_bias = np.zeros(3, dtype=np.float64)
            self._rotation_bias = np.zeros(3, dtype=np.float64)
