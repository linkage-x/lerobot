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
from pathlib import Path

from ..config import TeleoperatorConfig


_REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_QUEST3_CERT_FILE = _REPO_ROOT / "tools/fr3/quest3_certifications/cert.pem"
DEFAULT_QUEST3_KEY_FILE = _REPO_ROOT / "tools/fr3/quest3_certifications/key.pem"
DEFAULT_QUEST3_CALIBRATION_DIR = _REPO_ROOT / "outputs/fr3_quest3_calibration/teleoperators/quest3"


class Quest3TeleopMode(StrEnum):
    WEARABLE = "wearable"
    NECK = "neck"


class Quest3Hand(StrEnum):
    LEFT = "left"
    RIGHT = "right"


class Quest3GripperMapping(StrEnum):
    PINCH_VALUE = "pinch_value"
    FINGERTIP_DISTANCE = "fingertip_distance"


@TeleoperatorConfig.register_subclass("quest3")
@dataclass
class Quest3TeleopConfig(TeleoperatorConfig):
    calibration_dir: Path | None = DEFAULT_QUEST3_CALIBRATION_DIR
    # Compatibility with existing FR3 configs that define teleop.device_id for SpaceMouse.
    device_id: int | None = None
    mode: Quest3TeleopMode = Quest3TeleopMode.WEARABLE
    hand: Quest3Hand = Quest3Hand.RIGHT
    host: str = "0.0.0.0"
    port: int = 8012
    cert_file: Path | None = DEFAULT_QUEST3_CERT_FILE
    key_file: Path | None = DEFAULT_QUEST3_KEY_FILE
    frequency: int = 200
    use_hand_tracking: bool = True
    translation_scale: float = 1.0
    rotation_scale: float = 1.0
    translation_deadband_m: float = 0.002
    rotation_deadband_rad: float = 0.02
    enable_rotation: bool = True
    clutch_source: str = "squeeze"
    clutch_threshold: float = 0.5
    gripper_mapping: Quest3GripperMapping = Quest3GripperMapping.PINCH_VALUE
    initial_gripper: float = 1.0
    open_pinch_value: float = 0.111
    closed_pinch_value: float = 0.004
    open_fingertip_distance_m: float = 0.085
    closed_fingertip_distance_m: float = 0.018
    gripper_cmd_ema_alpha: float = 0.8
    gripper_cmd_max_rate: float = 12.0
    lost_tracking_timeout_s: float = 0.25
    # --- Phase 1: incremental control (mocap direct-drive) ---
    pos_scale: float = 1.0
    rot_scale: float = 1.0
    delta_deadband_m: float = 0.001
    delta_deadband_rad: float = 0.01
    max_step_pos_m: float = 0.25
    max_step_rot_rad: float = 0.3
    # --- Phase 2: controller mode ---
    grip_threshold: float = 0.5
    controller_gripper_close_threshold: float = 0.8
    controller_gripper_open_threshold: float = 0.2
    # --- Phase 3: stability ---
    filter_alpha_pos: float = 0.3
    filter_alpha_rot: float = 0.3
