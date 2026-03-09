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

from dataclasses import dataclass, field

from lerobot.cameras import CameraConfig

from ..config import RobotConfig


@RobotConfig.register_subclass("franka_research3")
@dataclass
class FrankaResearch3Config(RobotConfig):
    robot_ip: str = "127.0.0.1"
    gripper_port: str = "/dev/ttyUSB0"
    urdf_path: str = ""
    target_frame_name: str = "pika_gripper_ee"
    joint_names: list[str] = field(
        default_factory=lambda: [
            "fr3_joint1",
            "fr3_joint2",
            "fr3_joint3",
            "fr3_joint4",
            "fr3_joint5",
            "fr3_joint6",
            "fr3_joint7",
        ]
    )
    workspace_min: tuple[float, float, float] = (0.2, -0.6, 0.05)
    workspace_max: tuple[float, float, float] = (0.9, 0.6, 0.8)
    gripper_max_width_mm: float = 90.0
    disable_torque_on_disconnect: bool = True
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    damping: list[float] | None = None
    stiffness: list[float] | None = None
    filter_coeff: float | None = None

    def __post_init__(self):
        super().__post_init__()
        if len(self.workspace_min) != 3 or len(self.workspace_max) != 3:
            raise ValueError("workspace_min and workspace_max must be 3D tuples.")
        if any(mn >= mx for mn, mx in zip(self.workspace_min, self.workspace_max, strict=True)):
            raise ValueError("workspace_min must be strictly smaller than workspace_max.")
        if self.gripper_max_width_mm <= 0:
            raise ValueError("gripper_max_width_mm must be positive.")
