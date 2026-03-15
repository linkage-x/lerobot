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
    gripper_port: str = "/dev/ttyUSB80"
    allow_mock_gripper: bool = True
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
    max_target_delta_pos: tuple[float, float, float] | None = None
    max_target_delta_rot: tuple[float, float, float] | None = None
    gripper_max_width_mm: float = 90.0
    gripper_command_rate_limit_hz: float | None = 15.0
    gripper_command_deadband_mm: float = 0.5
    disable_torque_on_disconnect: bool = True
    cameras: dict[str, CameraConfig] = field(default_factory=dict)
    damping: list[float] | None = None
    stiffness: list[float] | None = None
    filter_coeff: float | None = None
    use_otg: bool = True
    otg_control_frequency: float = 800.0
    otg_async_control_frequency: float = 1000.0
    otg_max_velocity: tuple[float, ...] = (2.096, 2.096, 2.096, 2.096, 4.208, 3.344, 4.208)
    otg_max_acceleration: tuple[float, ...] = (8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0)
    otg_max_jerk: tuple[float, ...] = (4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0)
    otg_min_position: tuple[float, ...] = (-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159)
    otg_max_position: tuple[float, ...] = (2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159)
    otg_synchronization: bool = True
    otg_sync_mode: str = "time"

    def __post_init__(self):
        super().__post_init__()
        if len(self.workspace_min) != 3 or len(self.workspace_max) != 3:
            raise ValueError("workspace_min and workspace_max must be 3D tuples.")
        if self.max_target_delta_pos is not None and len(self.max_target_delta_pos) != 3:
            raise ValueError("max_target_delta_pos must be a 3D tuple when provided.")
        if self.max_target_delta_rot is not None and len(self.max_target_delta_rot) != 3:
            raise ValueError("max_target_delta_rot must be a 3D tuple when provided.")
        if any(mn >= mx for mn, mx in zip(self.workspace_min, self.workspace_max, strict=True)):
            raise ValueError("workspace_min must be strictly smaller than workspace_max.")
        if self.gripper_max_width_mm <= 0:
            raise ValueError("gripper_max_width_mm must be positive.")
        if self.gripper_command_rate_limit_hz is not None and self.gripper_command_rate_limit_hz <= 0:
            raise ValueError("gripper_command_rate_limit_hz must be positive when provided.")
        if self.gripper_command_deadband_mm < 0:
            raise ValueError("gripper_command_deadband_mm must be non-negative.")
        if self.otg_control_frequency <= 0:
            raise ValueError("otg_control_frequency must be positive.")
        if self.otg_async_control_frequency <= 0:
            raise ValueError("otg_async_control_frequency must be positive.")
        otg_limits = (
            self.otg_max_velocity,
            self.otg_max_acceleration,
            self.otg_max_jerk,
            self.otg_min_position,
            self.otg_max_position,
        )
        if any(len(values) != len(self.joint_names) for values in otg_limits):
            raise ValueError("All OTG limit arrays must match the number of FR3 joints.")

    @property
    def otg_dt(self) -> float:
        return 1.0 / self.otg_control_frequency

    @property
    def otg_async_dt(self) -> float:
        return 1.0 / self.otg_async_control_frequency
