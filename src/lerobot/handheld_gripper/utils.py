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

from typing import cast

from lerobot.utils.import_utils import make_device_from_device_class

from .configs import HandheldGripperConfig
from .handheld_gripper import HandheldGripper


def make_handheld_grippers_from_configs(
    handheld_gripper_configs: dict[str, HandheldGripperConfig],
) -> dict[str, HandheldGripper]:
    handheld_grippers: dict[str, HandheldGripper] = {}

    for key, cfg in handheld_gripper_configs.items():
        if cfg.type == "pika_sense":
            from .pika_sense import PikaSense

            handheld_grippers[key] = PikaSense(cfg)
        else:
            try:
                handheld_grippers[key] = cast(HandheldGripper, make_device_from_device_class(cfg))
            except Exception as e:
                raise ValueError(f"Error creating handheld gripper {key} with config {cfg}: {e}") from e

    return handheld_grippers
