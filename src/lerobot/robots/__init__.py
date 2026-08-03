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

import logging

try:
    from .config import RobotConfig
    from .robot import Robot
    from .utils import make_robot_from_config
except ImportError as exc:  # pragma: no cover - optional aggregate import for direct robot submodules
    # Importing a robot submodule directly must keep working even when the aggregate
    # re-exports are unavailable (e.g. a missing optional dependency). Anything other than
    # an ImportError is a real bug and is left to propagate.
    logging.getLogger(__name__).warning(
        "lerobot.robots aggregate imports unavailable (%s); "
        "`RobotConfig`, `Robot` and `make_robot_from_config` will not be importable from "
        "`lerobot.robots`. Import the robot submodule directly, or install the missing dependency.",
        exc,
    )
