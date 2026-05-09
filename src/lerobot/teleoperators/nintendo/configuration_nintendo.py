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

    translation_scale: float = 0.0015
    vertical_scale: float = 0.0015
    rotation_scale: float = 0.006
    scale_x: float | None = None
    scale_y: float | None = None
    scale_z: float | None = None
    scale_wx: float | None = 0.0
    scale_wy: float | None = 0.0
    scale_wz: float | None = None
    stick_deadband: float = 0.12
    enable_rotation: bool = True
    invert_x: bool = False
    invert_y: bool = False
    invert_z: bool = False
    invert_wz: bool = False

    # Hold one of these buttons to enable robot motion. Empty tuple means always enabled.
    clutch_buttons: tuple[str, ...] = ("R", "L")
    z_up_buttons: tuple[str, ...] = ("X", "UP")
    z_down_buttons: tuple[str, ...] = ("B", "DOWN")
    yaw_positive_buttons: tuple[str, ...] = ("Y", "LEFT")
    yaw_negative_buttons: tuple[str, ...] = ("A", "RIGHT")

    gripper_mode: NintendoGripperMode = NintendoGripperMode.INCREMENTAL
    initial_gripper: float = 1.0
    gripper_step: float = 0.03
    gripper_move_time: float = 0.006
    gripper_close_buttons: tuple[str, ...] = ("ZR", "A")
    gripper_open_buttons: tuple[str, ...] = ("ZL", "B")
    gripper_cmd_ema_alpha: float = 0.8
    gripper_cmd_max_rate: float = 12.0

    read_timeout_ms: int = 1
    stale_timeout_s: float = 0.25
