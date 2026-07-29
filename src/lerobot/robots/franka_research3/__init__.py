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

# Imported eagerly so that `@RobotConfig.register_subclass("franka_research3")` runs as
# soon as this package is imported: draccus resolves `robot.type` against the registry at
# config-parse time, so a lazy config import makes `--config_path=...` fail with
# "Couldn't find a choice class for 'franka_research3'". This module only pulls in
# dataclasses plus the camera/robot config bases, so it stays cheap; the driver and
# processor modules (panda_py, placo, numpy) stay lazy below.
from .config_franka_research3 import FrankaResearch3Config

__all__ = [
    "AbsoluteEEActionToRobotAction",
    "DeltaActionToAbsoluteEEAction",
    "FrankaResearch3",
    "FrankaResearch3Config",
    "KeepAbsoluteEEObservation",
]


def __getattr__(name: str):
    if name == "FrankaResearch3":
        from .franka_research3 import FrankaResearch3

        return FrankaResearch3
    if name in {"AbsoluteEEActionToRobotAction", "DeltaActionToAbsoluteEEAction", "KeepAbsoluteEEObservation"}:
        from . import processor_franka_research3

        return getattr(processor_franka_research3, name)
    raise AttributeError(name)
