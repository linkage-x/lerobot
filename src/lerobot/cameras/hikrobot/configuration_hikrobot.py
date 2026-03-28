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

from ..configs import CameraConfig, ColorMode, Cv2Rotation

__all__ = ["HikrobotCameraConfig", "ColorMode", "Cv2Rotation"]


@CameraConfig.register_subclass("hikrobot")
@dataclass
class HikrobotCameraConfig(CameraConfig):
    serial: str | None = None
    device_index: int | None = None
    transport_layer: str = "usb"
    color_mode: ColorMode = ColorMode.RGB
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = 1
    exposure_us: float | None = None
    gain_db: float | None = None
    gamma: float | None = None
    white_balance_auto: str = "continuous"
    white_balance_red: int | None = None
    white_balance_green: int | None = None
    white_balance_blue: int | None = None
    lock_white_balance_after_warmup: bool = False
    timeout_ms: int = 1000

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.transport_layer = self.transport_layer.lower()
        self.white_balance_auto = self.white_balance_auto.lower()

        if self.transport_layer not in {"usb", "gige", "all"}:
            raise ValueError(
                f"`transport_layer` must be one of ['usb', 'gige', 'all'], but {self.transport_layer!r} is provided."
            )
        if self.white_balance_auto not in {"off", "once", "continuous"}:
            raise ValueError(
                "`white_balance_auto` must be one of ['off', 'once', 'continuous'], "
                f"but {self.white_balance_auto!r} is provided."
            )
        if self.device_index is not None and self.device_index < 0:
            raise ValueError(f"`device_index` must be >= 0, but {self.device_index} is provided.")
        if self.gamma is not None and self.gamma <= 0:
            raise ValueError(f"`gamma` must be > 0 when provided, but {self.gamma} is provided.")
        for name, value in (
            ("white_balance_red", self.white_balance_red),
            ("white_balance_green", self.white_balance_green),
            ("white_balance_blue", self.white_balance_blue),
        ):
            if value is not None and value <= 0:
                raise ValueError(f"`{name}` must be > 0 when provided, but {value} is provided.")
        if self.timeout_ms <= 0:
            raise ValueError(f"`timeout_ms` must be > 0, but {self.timeout_ms} is provided.")
