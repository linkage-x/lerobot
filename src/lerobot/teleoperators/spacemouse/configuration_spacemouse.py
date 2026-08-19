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

from dataclasses import dataclass
from enum import StrEnum

from ..config import TeleoperatorConfig


class SpaceMouseToolMode(StrEnum):
    BINARY = "binary"
    INCREMENTAL = "incremental"


class SpaceMouseEnableButton(StrEnum):
    NONE = "none"
    LEFT = "left"
    RIGHT = "right"


@TeleoperatorConfig.register_subclass("space_mouse")
@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseTeleopConfig(TeleoperatorConfig):
    """SpaceMouse teleop configuration.

    `translation_axis_map` and `rotation_axis_map` are 3x3 matrices that map
    raw debiased device axes into the teleop command frame. The default
    translation map preserves the FR3 SpaceMouse convention validated in
    teleop: `[-raw_y, raw_x, raw_z]`.

    The two maps are deliberately *not* the same matrix, because they do not
    target the same frame. `target_{x,y,z}` is added to the reference position
    in the robot base frame, while `target_{wx,wy,wz}` is right-multiplied onto
    the reference orientation (`desired_R = reference_R @ delta_R` in
    franka_research3.py), i.e. applied about the tool's own axes. A device axis
    therefore needs one alignment to reach the base and a different one to reach
    the tool; copying the translation map onto rotation would be wrong.

    The rotation default is `[-raw_wx, raw_wy, raw_wz]`: the device's roll axis
    opposes the tool's x axis on this rig, measured in teleop. It read as
    identity for a long time only because `fr3_record_config.yaml` pinned
    `scale_wx`/`scale_wy` to `0.0` while the rig recorded against
    `pika_task_tcp` -- roll and pitch were switched off, so nothing exercised
    them. Yaw was live throughout and is not negated.
    """

    device_id: int = 0
    frequency: int = 200
    translation_scale: float = 0.000615
    rotation_scale: float = 0.000648
    translation_axis_map: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (0.0, -1.0, 0.0),
        (1.0, 0.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    rotation_axis_map: tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float]] = (
        (-1.0, 0.0, 0.0),
        (0.0, 1.0, 0.0),
        (0.0, 0.0, 1.0),
    )
    scale_x: float | None = None
    scale_y: float | None = None
    scale_z: float | None = None
    scale_wx: float | None = None
    scale_wy: float | None = None
    scale_wz: float | None = None
    threshold_x: float = 0.02
    threshold_y: float = 0.02
    threshold_z: float = 0.02
    threshold_wx: float = 0.04
    threshold_wy: float = 0.04
    threshold_wz: float = 0.04
    bias_sample_count: int = 30
    bias_sample_sleep_s: float = 0.005
    enable_rotation: bool = True
    motion_enable_button: SpaceMouseEnableButton = SpaceMouseEnableButton.NONE
    tool_mode: SpaceMouseToolMode = SpaceMouseToolMode.INCREMENTAL
    initial_gripper: float = 1.0
    incremental_step: float = 0.02
    move_time: float = 0.006
    button_debounce_s: float = 0.0
    button_release_grace_s: float = 0.0
    gripper_cmd_min_delta: float = 0.0
    gripper_cmd_min_interval_s: float = 0.0
    gripper_cmd_ema_alpha: float = 0.0
    gripper_cmd_max_rate: float = 0.0
