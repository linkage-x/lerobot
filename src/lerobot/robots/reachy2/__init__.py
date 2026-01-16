#!/usr/bin/env python

# Copyright 2025 The HuggingFace Inc. team. All rights reserved.
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

from .configuration_reachy2 import Reachy2RobotConfig

try:
    from .robot_reachy2 import (
        REACHY2_ANTENNAS_JOINTS,
        REACHY2_L_ARM_JOINTS,
        REACHY2_NECK_JOINTS,
        REACHY2_R_ARM_JOINTS,
        REACHY2_VEL,
        Reachy2Robot,
    )
except ModuleNotFoundError as exc:
    if exc.name and exc.name.split(".")[0] != "reachy2_sdk":
        raise

    _reachy2_robot_import_error = exc

    REACHY2_ANTENNAS_JOINTS: dict[str, str] = {}
    REACHY2_L_ARM_JOINTS: dict[str, str] = {}
    REACHY2_NECK_JOINTS: dict[str, str] = {}
    REACHY2_R_ARM_JOINTS: dict[str, str] = {}
    REACHY2_VEL: dict[str, str] = {}

    class Reachy2Robot:  # type: ignore[no-redef]
        def __init__(self, *args: object, **kwargs: object) -> None:
            raise ModuleNotFoundError(
                "reachy2_sdk is required to use Reachy2Robot. Install it to enable Reachy 2 hardware."
            ) from _reachy2_robot_import_error


__all__ = [
    "Reachy2RobotConfig",
    "REACHY2_ANTENNAS_JOINTS",
    "REACHY2_L_ARM_JOINTS",
    "REACHY2_NECK_JOINTS",
    "REACHY2_R_ARM_JOINTS",
    "REACHY2_VEL",
    "Reachy2Robot",
]
