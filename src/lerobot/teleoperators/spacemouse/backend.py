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

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np


@dataclass
class SpaceMouseReading:
    translation: np.ndarray
    rotation: np.ndarray
    buttons: tuple[bool, bool]


class SpaceMouseDriver(Protocol):
    def connect(self) -> None: ...

    def disconnect(self) -> None: ...

    def poll(self) -> SpaceMouseReading | None: ...


@dataclass
class PySpaceMouseDriver:
    device_id: int

    def __post_init__(self):
        try:
            import pyspacemouse
        except Exception as e:  # pragma: no cover - exercised with real hardware only
            raise ImportError(
                "spacemouse teleoperator requires pyspacemouse in the runtime environment."
            ) from e

        self._pyspacemouse = pyspacemouse
        self._device = None

    def connect(self) -> None:
        devices = self._pyspacemouse.list_devices()
        if not devices:
            raise ConnectionError("No SpaceMouse devices detected.")
        self._device = self._pyspacemouse.open(device=devices[0], DeviceNumber=self.device_id)
        if self._device is None:
            raise ConnectionError(f"Could not open SpaceMouse device {self.device_id}.")

    def disconnect(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None

    def poll(self) -> SpaceMouseReading | None:
        if self._device is None:
            raise RuntimeError("SpaceMouse backend is not connected.")
        state = self._device.read()
        if state is None:
            return None
        translation = np.array([state.x, state.y, state.z], dtype=np.float64)
        rotation = np.array([state.roll, state.pitch, state.yaw], dtype=np.float64)
        buttons = (bool(state.buttons[0]), bool(state.buttons[1]))
        return SpaceMouseReading(translation=translation, rotation=rotation, buttons=buttons)
