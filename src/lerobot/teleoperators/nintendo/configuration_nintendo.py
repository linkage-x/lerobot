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

from dataclasses import dataclass
from enum import StrEnum

from ..config import TeleoperatorConfig


class NintendoController(StrEnum):
    ANY = "any"
    LEFT = "left"
    RIGHT = "right"
    PRO = "pro"


class NintendoGripperMode(StrEnum):
    BINARY = "binary"
    INCREMENTAL = "incremental"


@TeleoperatorConfig.register_subclass("nintendo")
@dataclass
class NintendoTeleopConfig(TeleoperatorConfig):
    """Nintendo Joy-Con / Pro Controller teleop configuration for FR3-style actions."""

    device_id: int | None = None
    controller: NintendoController = NintendoController.ANY
    # Compatibility alias for users thinking in Joy-Con side selection.
    side: NintendoController = NintendoController.ANY
    frequency: int = 200

    translation_scale: float = 0.001
    vertical_scale: float = 0.001
    rotation_scale: float = 1.0
    scale_x: float | None = None
    scale_y: float | None = None
    scale_z: float | None = None
    scale_wx: float | None = None
    scale_wy: float | None = None
    scale_wz: float | None = None
    enable_rotation: bool = True
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    invert_wx: bool = False
    invert_wy: bool = True
    invert_wz: bool = True

    # Left stick X/Y and right stick Z are used for translation by default. IMU
    # translation is kept as an opt-in experimental mode because accelerometer
    # double integration drifts without external tracking.
    stick_deadband: float = 0.12
    experimental_imu_translation: bool = False

    # Hold one of these buttons to set the IMU origin and enable relative rotation.
    clutch_buttons: tuple[str, ...] = ("ZL",)
    imu_accel_deadband_g: float = 0.03
    imu_gyro_deadband_dps: float = 1.5
    imu_stationary_gyro_dps: float = 3.0
    imu_stationary_accel_norm_tolerance_g: float = 0.08
    imu_velocity_decay: float = 0.98
    max_imu_dt_s: float = 0.05
    max_step_pos_m: float = 0.18
    max_step_rot_rad: float = 0.6
    accel_axis_map: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    gyro_axis_map: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )

    gripper_mode: NintendoGripperMode = NintendoGripperMode.INCREMENTAL
    initial_gripper: float = 1.0
    gripper_step: float = 0.03
    gripper_move_time: float = 0.006
    gripper_close_buttons: tuple[str, ...] = ("R",)
    gripper_open_buttons: tuple[str, ...] = ()
    gripper_cmd_ema_alpha: float = 0.8
    gripper_cmd_max_rate: float = 12.0

    read_timeout_ms: int = 1
    stale_timeout_s: float = 0.25
