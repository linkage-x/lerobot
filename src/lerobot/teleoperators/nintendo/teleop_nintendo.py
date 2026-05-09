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
    GRAVITY_MPS2 = 9.80665

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
        self._last_clutch_active = False
        self._imu_baseline_accel_g: np.ndarray | None = None
        self._imu_baseline_gyro_dps: np.ndarray | None = None
        self._imu_position_m = np.zeros(3, dtype=np.float64)
        self._imu_velocity_mps = np.zeros(3, dtype=np.float64)
        self._imu_rotvec_rad = np.zeros(3, dtype=np.float64)
        self._last_imu_time_s = float("-inf")

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

    def _latest_reading(self, now: float | None = None) -> tuple[NintendoControllerReading | None, bool]:
        if self._driver is None:
            return None, False
        reading = self._driver.poll()
        now = time.perf_counter() if now is None else float(now)
        if reading is not None:
            self._last_reading = reading
            self._last_update_s = now
            return reading, True
        if self._last_reading is not None and now - self._last_update_s <= float(self.config.stale_timeout_s):
            return self._last_reading, False
        return None, False

    @staticmethod
    def _deadband_vector(values: np.ndarray, threshold: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        threshold = float(abs(threshold))
        return np.where(np.abs(values) >= threshold, values, 0.0)

    @staticmethod
    def _deadband_scalar(value: float, threshold: float) -> float:
        value = float(value)
        threshold = float(abs(threshold))
        if abs(value) < threshold:
            return 0.0
        return value

    @staticmethod
    def _clamp_norm(values: np.ndarray, max_norm: float) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        max_norm = float(max_norm)
        if max_norm <= 0.0:
            return values
        norm = float(np.linalg.norm(values))
        if norm > max_norm:
            return values / norm * max_norm
        return values

    @cached_property
    def accel_axis_map(self) -> np.ndarray:
        matrix = np.asarray(self.config.accel_axis_map, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"accel_axis_map must be 3x3, got {matrix.shape}")
        return matrix

    @cached_property
    def gyro_axis_map(self) -> np.ndarray:
        matrix = np.asarray(self.config.gyro_axis_map, dtype=np.float64)
        if matrix.shape != (3, 3):
            raise ValueError(f"gyro_axis_map must be 3x3, got {matrix.shape}")
        return matrix

    def _reading_accel_g(self, reading: NintendoControllerReading) -> np.ndarray:
        accel = self.accel_axis_map @ np.asarray(reading.accel_g, dtype=np.float64)
        if self.config.invert_x:
            accel[0] *= -1.0
        if self.config.invert_y:
            accel[1] *= -1.0
        if self.config.invert_z:
            accel[2] *= -1.0
        return accel

    def _reading_gyro_dps(self, reading: NintendoControllerReading) -> np.ndarray:
        gyro = self.gyro_axis_map @ np.asarray(reading.gyro_dps, dtype=np.float64)
        if self.config.invert_wx:
            gyro[0] *= -1.0
        if self.config.invert_wy:
            gyro[1] *= -1.0
        if self.config.invert_wz:
            gyro[2] *= -1.0
        return gyro

    def _reset_imu_clutch_state(self) -> None:
        self._last_clutch_active = False
        self._imu_baseline_accel_g = None
        self._imu_baseline_gyro_dps = None
        self._imu_position_m.fill(0.0)
        self._imu_velocity_mps.fill(0.0)
        self._imu_rotvec_rad.fill(0.0)
        self._last_imu_time_s = float("-inf")

    def _imu_relative_target(
        self,
        reading: NintendoControllerReading,
        now: float,
        *,
        is_fresh: bool,
    ) -> tuple[np.ndarray, bool]:
        clutch_active = self._clutch_active(reading.buttons)
        if not clutch_active:
            self._reset_imu_clutch_state()
            return np.zeros(6, dtype=np.float64), False

        accel_g = self._reading_accel_g(reading)
        gyro_dps = self._reading_gyro_dps(reading)
        if self._imu_baseline_accel_g is None or self._imu_baseline_gyro_dps is None or not self._last_clutch_active:
            self._imu_baseline_accel_g = accel_g.copy()
            self._imu_baseline_gyro_dps = gyro_dps.copy()
            self._imu_position_m.fill(0.0)
            self._imu_velocity_mps.fill(0.0)
            self._imu_rotvec_rad.fill(0.0)
            self._last_imu_time_s = now
            self._last_clutch_active = True
            return np.zeros(6, dtype=np.float64), True

        if not is_fresh:
            return np.zeros(6, dtype=np.float64), True

        dt = now - self._last_imu_time_s
        if not np.isfinite(dt) or dt <= 0.0:
            dt = 1.0 / max(float(self.config.frequency), 1.0)
        dt = min(float(dt), float(self.config.max_imu_dt_s))
        self._last_imu_time_s = now

        rel_accel_g = self._deadband_vector(
            accel_g - self._imu_baseline_accel_g,
            self.config.imu_accel_deadband_g,
        )
        rel_gyro_dps = self._deadband_vector(
            gyro_dps - self._imu_baseline_gyro_dps,
            self.config.imu_gyro_deadband_dps,
        )

        accel_norm_delta_g = abs(float(np.linalg.norm(accel_g)) - float(np.linalg.norm(self._imu_baseline_accel_g)))
        stationary_gyro_threshold_dps = float(self.config.imu_stationary_gyro_dps)
        stationary_imu = (
            stationary_gyro_threshold_dps > 0.0
            and min(float(np.linalg.norm(rel_gyro_dps)), float(np.linalg.norm(gyro_dps)))
            <= stationary_gyro_threshold_dps
            and accel_norm_delta_g <= float(self.config.imu_stationary_accel_norm_tolerance_g)
        )
        if stationary_imu:
            rel_accel_g = np.zeros(3, dtype=np.float64)
            rel_gyro_dps = np.zeros(3, dtype=np.float64)
            self._imu_baseline_accel_g = accel_g.copy()
            self._imu_baseline_gyro_dps = gyro_dps.copy()
            self._imu_velocity_mps.fill(0.0)

        step_pos_m = np.zeros(3, dtype=np.float64)
        if self.config.experimental_imu_translation:
            self._imu_velocity_mps += rel_accel_g * self.GRAVITY_MPS2 * dt
            self._imu_velocity_mps *= float(np.clip(self.config.imu_velocity_decay, 0.0, 1.0))
            step_pos_m = self._imu_velocity_mps * dt
            self._imu_position_m += step_pos_m
            self._imu_position_m = self._clamp_norm(self._imu_position_m, self.config.max_step_pos_m)
            step_pos_m = self._clamp_norm(step_pos_m, self.config.max_step_pos_m)
        else:
            self._imu_velocity_mps.fill(0.0)

        step_rotvec_rad = np.deg2rad(rel_gyro_dps) * dt

        self._imu_rotvec_rad += step_rotvec_rad
        self._imu_rotvec_rad = self._clamp_norm(self._imu_rotvec_rad, self.config.max_step_rot_rad)

        step_rotvec_rad = self._clamp_norm(step_rotvec_rad, self.config.max_step_rot_rad)
        return np.concatenate((step_pos_m, step_rotvec_rad)), True

    def _stick_translation_target(self, reading: NintendoControllerReading) -> tuple[np.ndarray, bool]:
        left_x, left_y = reading.left_stick
        _right_x, right_y = reading.right_stick
        deadband = float(self.config.stick_deadband)
        translation = np.array(
            [
                self._deadband_scalar(left_y, deadband),
                -self._deadband_scalar(left_x, deadband),
                self._deadband_scalar(right_y, deadband),
            ],
            dtype=np.float64,
        )
        if self.config.invert_x:
            translation[0] *= -1.0
        if self.config.invert_y:
            translation[1] *= -1.0
        if self.config.invert_z:
            translation[2] *= -1.0
        return translation, bool(np.any(translation))

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
        if close_pressed and open_pressed:
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
            "accel_g": (0.0, 0.0, 0.0) if reading is None else reading.accel_g,
            "gyro_dps": (0.0, 0.0, 0.0) if reading is None else reading.gyro_dps,
            "clutch_active": self._last_clutch_active,
            "imu_position_m": tuple(float(v) for v in self._imu_position_m),
            "imu_rotvec_rad": tuple(float(v) for v in self._imu_rotvec_rad),
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
            now = time.perf_counter()
            reading, _is_fresh = self._latest_reading(now)
            motion = False
            if reading is not None:
                _stick_translation, translation_enabled = self._stick_translation_target(reading)
                motion = self._clutch_active(reading.buttons) or translation_enabled
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
        now = time.perf_counter()
        reading, is_fresh = self._latest_reading(now)
        if reading is None:
            self._reset_imu_clutch_state()
            return self._zero_action()

        self._update_gripper(reading.buttons)
        filtered_gripper = self._filter_gripper_command(self._last_gripper)
        stick_translation, translation_enabled = self._stick_translation_target(reading)
        inputs, rotation_enabled = self._imu_relative_target(reading, now, is_fresh=is_fresh)
        if not self.config.experimental_imu_translation:
            inputs[:3] = stick_translation
        else:
            inputs[:3] += stick_translation
        if not self.config.enable_rotation:
            inputs[3:] = 0.0
            rotation_enabled = False
        target = inputs * self._scale_vector()
        enabled = translation_enabled or rotation_enabled
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
            self._reset_imu_clutch_state()
