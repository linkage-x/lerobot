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

"""What an FR3 dataset's ``action`` column means.

Kept in its own dependency-free module because the robot *config* modules are imported eagerly
(draccus resolves ``robot.type`` against the registry at config-parse time) and must stay cheap,
while the processors that implement these modes pull in numpy and scipy rotations.
"""

from __future__ import annotations

# Absolute EE target pose. Rotation as a sign-continuous quaternion, because an absolute
# orientation covers all of SO(3) and a rotvec would alias at theta = pi.
ACTION_MODE_ABSOLUTE_EE = "absolute_ee"
# Delta against the pose commanded on the previous frame. The observation already carries
# `prev_cmd.ee.*`, so no extra deployment state is needed; arm tracking lag stays out of the
# action, and a motion-disabled frame records as an exact zero delta.
ACTION_MODE_DELTA_EE_FROM_PREV_CMD = "delta_ee_from_prev_cmd"
# Delta against the measured EE pose. Purely reactive at deployment, but it bakes the recording
# rig's tracking lag into the action, so a policy learns to compensate for that specific lag.
ACTION_MODE_DELTA_EE_FROM_CURRENT = "delta_ee_from_current"

ACTION_MODES = (
    ACTION_MODE_ABSOLUTE_EE,
    ACTION_MODE_DELTA_EE_FROM_PREV_CMD,
    ACTION_MODE_DELTA_EE_FROM_CURRENT,
)

DELTA_ACTION_MODES = (
    ACTION_MODE_DELTA_EE_FROM_PREV_CMD,
    ACTION_MODE_DELTA_EE_FROM_CURRENT,
)


def validate_action_mode(action_mode: str) -> str:
    if action_mode not in ACTION_MODES:
        raise ValueError(f"action_mode must be one of {ACTION_MODES}, got {action_mode!r}")
    return action_mode


def is_delta_action_mode(action_mode: str) -> bool:
    return validate_action_mode(action_mode) in DELTA_ACTION_MODES


def delta_reference_for_action_mode(action_mode: str) -> str:
    """Which pose a delta mode measures against: ``"prev_cmd"`` or ``"current"``."""
    if validate_action_mode(action_mode) == ACTION_MODE_DELTA_EE_FROM_PREV_CMD:
        return "prev_cmd"
    if action_mode == ACTION_MODE_DELTA_EE_FROM_CURRENT:
        return "current"
    raise ValueError(f"{action_mode!r} is not a delta action mode")
