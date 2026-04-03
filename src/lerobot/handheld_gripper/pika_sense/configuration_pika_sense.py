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

from ..configs import HandheldGripperConfig

__all__ = ["PikaSenseConfig"]


@HandheldGripperConfig.register_subclass("pika_sense")
@dataclass(kw_only=True)
class PikaSenseConfig(HandheldGripperConfig):
    port: str
    fps: int = 120
    warmup_s: float = 1.0

    def __post_init__(self) -> None:
        if not self.port:
            raise ValueError("`port` must be a non-empty serial device path.")

        if self.fps <= 0:
            raise ValueError(f"`fps` must be > 0, but {self.fps} is provided.")
        if self.warmup_s < 0:
            raise ValueError(f"`warmup_s` must be >= 0, but {self.warmup_s} is provided.")
