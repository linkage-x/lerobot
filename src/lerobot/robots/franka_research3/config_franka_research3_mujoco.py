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

from ..config import RobotConfig


@RobotConfig.register_subclass("franka_research3_mujoco")
@dataclass
class FrankaResearch3MujocoConfig(RobotConfig):
    """Simulation twin of :class:`FrankaResearch3Config`.

    Only the fields the ee2ee record pipeline reads off ``cfg.robot`` (workspace bounds and
    per-step delta clamps) are shared with the hardware config; everything else configures the
    MuJoCo scene. Keeping this a separate registry entry rather than a flag on the hardware
    config means a sim run can never accidentally inherit an FCI ip or a serial gripper port.
    """

    urdf_path: str = ""
    sim_xml_path: str = ""
    target_frame_name: str = "pika_task_tcp"
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
    # Names must match the hardware camera keys so sim and real datasets share one schema.
    camera_names: tuple[str, ...] = ("external", "wrist")
    # dataset camera key -> MuJoCo camera in the scene XML. Lets a hardware rig keep its own
    # names (e.g. "ee"/"side") while still pointing at the right simulated viewpoint.
    camera_name_mapping: dict[str, str] = field(
        default_factory=lambda: {"external": "external_cam", "wrist": "ee_cam"}
    )
    camera_width: int = 640
    camera_height: int = 480
    workspace_min: tuple[float, float, float] = (0.2, -0.6, 0.05)
    workspace_max: tuple[float, float, float] = (0.9, 0.6, 0.8)
    max_target_delta_pos: tuple[float, float, float] | None = None
    max_target_delta_rot: tuple[float, float, float] | None = None
    initial_joint_positions: tuple[float, ...] = (
        -0.09534768,
        0.39794592,
        -0.23614631,
        -2.39008542,
        1.72320219,
        0.61671872,
        2.14073504,
    )
    initial_gripper: float = 1.0
    use_otg: bool = False
    continuous_physics: bool = False
    continuous_physics_frequency: float | None = None
    teleop_control_frequency: float = 200.0
    arm_actuator_kp: float | None = None
    enable_arm_gravity_compensation: bool = True
    arm_gravity_compensation_scale: float = 0.5
    ik_solver: str = "mujoco"
    ik_tolerance: float = 1e-6
    ik_max_iterations: int = 200
    # Same guard as the hardware robot, kept numerically identical so a sim episode is held to
    # the envelope the hardware one is: a render pass that straggles this far behind its siblings
    # makes the frame unusable for training, so it fails the episode loudly.
    #
    # This side is what sets the number. Under software EGL the renderer was measured straggling
    # 19 ms, which aborted a sim episode against the 15 ms this used to be. Hardware, where
    # camera anchoring holds skew to 7.8 ms p95, would have been fine at 15.
    camera_max_skew_ms: float = 20.0
    # Re-randomize the workspace object at each episode reset so sim episodes are not identical.
    randomize_workspace_object: bool = True
    workspace_object_random_radius_m: float = 0.10

    def __post_init__(self):
        super().__post_init__()
        if len(self.workspace_min) != 3 or len(self.workspace_max) != 3:
            raise ValueError("workspace_min and workspace_max must be 3D tuples.")
        if any(mn >= mx for mn, mx in zip(self.workspace_min, self.workspace_max, strict=True)):
            raise ValueError("workspace_min must be strictly smaller than workspace_max.")
        if self.max_target_delta_pos is not None and len(self.max_target_delta_pos) != 3:
            raise ValueError("max_target_delta_pos must be a 3D tuple when provided.")
        if self.max_target_delta_rot is not None and len(self.max_target_delta_rot) != 3:
            raise ValueError("max_target_delta_rot must be a 3D tuple when provided.")
        if not self.camera_names:
            raise ValueError("camera_names must not be empty.")
        missing = [name for name in self.camera_names if name not in self.camera_name_mapping]
        if missing:
            raise ValueError(f"camera_name_mapping is missing entries for cameras: {missing}")
        if self.camera_width <= 0 or self.camera_height <= 0:
            raise ValueError("camera_width and camera_height must be positive.")
        if self.camera_max_skew_ms < 0:
            raise ValueError("camera_max_skew_ms must be non-negative.")
