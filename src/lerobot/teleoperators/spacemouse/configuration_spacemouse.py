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


@TeleoperatorConfig.register_subclass("space_mouse")
@TeleoperatorConfig.register_subclass("spacemouse")
@dataclass
class SpaceMouseTeleopConfig(TeleoperatorConfig):
    device_id: int = 0
    frequency: int = 200
    scale_x: float = 0.0006
    scale_y: float = 0.0006
    scale_z: float = 0.0006
    scale_wx: float = 0.0004
    scale_wy: float = 0.0004
    scale_wz: float = 0.0004
    threshold_x: float = 0.02
    threshold_y: float = 0.02
    threshold_z: float = 0.02
    threshold_wx: float = 0.04
    threshold_wy: float = 0.04
    threshold_wz: float = 0.04
    enable_rotation: bool = True
    tool_mode: SpaceMouseToolMode = SpaceMouseToolMode.INCREMENTAL
    initial_gripper: float = 1.0
    incremental_step: float = 0.02
    move_time: float = 0.006
