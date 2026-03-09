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
from .configuration_spacemouse import SpaceMouseTeleopConfig, SpaceMouseToolMode


class SpaceMouseTeleop(Teleoperator):
    config_class = SpaceMouseTeleopConfig
    name = "spacemouse"

    driver_cls = PySpaceMouseDriver

    def __init__(self, config: SpaceMouseTeleopConfig):
        super().__init__(config)
        self.config = config
        self._driver = None
        self._is_connected = False
        self._last_button_0 = False
        self._last_gripper = float(np.clip(config.initial_gripper, 0.0, 1.0))
        self._last_gripper_update = 0.0

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
        self._is_connected = True

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

    def _update_gripper(self, button_0: bool, button_1: bool) -> None:
        if self.config.tool_mode == SpaceMouseToolMode.BINARY:
            if button_0 and not self._last_button_0:
                self._last_gripper = 1.0 - self._last_gripper
        else:
            now = time.perf_counter()
            if now - self._last_gripper_update >= self.config.move_time:
                if button_0 and not button_1:
                    self._last_gripper = min(1.0, self._last_gripper + self.config.incremental_step)
                    self._last_gripper_update = now
                elif button_1 and not button_0:
                    self._last_gripper = max(0.0, self._last_gripper - self.config.incremental_step)
                    self._last_gripper_update = now
        self._last_button_0 = button_0

    @check_if_not_connected
    def get_action(self) -> RobotAction:
        reading = self._driver.poll()
        if reading is None:
            return self._zero_action()

        data = np.array(
            [
                -reading.translation[1],
                reading.translation[0],
                reading.translation[2],
                reading.rotation[0],
                reading.rotation[1],
                reading.rotation[2],
            ],
            dtype=np.float64,
        )
        threshold = np.array(
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
        scale = np.array(
            [
                self.config.scale_x,
                self.config.scale_y,
                self.config.scale_z,
                self.config.scale_wx,
                self.config.scale_wy,
                self.config.scale_wz,
            ],
            dtype=np.float64,
        )

        if not np.any(np.abs(data) >= threshold):
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
