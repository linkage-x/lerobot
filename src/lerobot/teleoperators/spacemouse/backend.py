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
    # When the device produced this report, on whatever clock the backend keeps. `poll()` hands
    # back the last state it saw whenever no new report arrived, so the six axes alone cannot
    # say whether the operator is still pushing the puck or stopped touching it a minute ago --
    # an unchanged timestamp says nothing has been heard since. None when a backend does not
    # date its reports.
    timestamp: float | None = None


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

    def _list_devices(self) -> list[object]:
        if hasattr(self._pyspacemouse, "list_devices"):
            return list(self._pyspacemouse.list_devices())
        if hasattr(self._pyspacemouse, "get_connected_devices"):
            return list(self._pyspacemouse.get_connected_devices())
        raise AttributeError("pyspacemouse does not expose a supported device enumeration API.")

    def _open_device(self, device: object) -> object | None:
        open_candidates = (
            {"device": device},
            {"device_index": self.device_id},
            {"device": device, "DeviceNumber": self.device_id},
        )
        # A TypeError only means this pyspacemouse version has a different `open()`
        # signature, so keep trying the other forms. Any other failure is the real
        # reason the device would not open; remember it for the connect() message
        # instead of letting a raw easyhid traceback escape.
        self._open_error: Exception | None = None
        for kwargs in open_candidates:
            try:
                opened = self._pyspacemouse.open(**kwargs)
            except TypeError:
                continue
            except Exception as exc:  # noqa: BLE001 - reported by connect()
                self._open_error = exc
                continue
            if opened is not None:
                return opened
        return None

    def connect(self) -> None:
        devices = self._list_devices()
        if not devices:
            raise ConnectionError("No SpaceMouse devices detected.")
        if self.device_id >= len(devices):
            raise ConnectionError(f"SpaceMouse device index {self.device_id} out of range for {len(devices)} devices.")
        self._device = self._open_device(devices[self.device_id])
        if self._device is None:
            hint = (
                "The HID device is enumerated but could not be opened. It is usually already "
                "held by another process (stop any running teleop first), or the hidraw node is "
                "not readable by this user."
            )
            reason = f" Last open error: {self._open_error!r}." if self._open_error is not None else ""
            raise ConnectionError(f"Could not open SpaceMouse device {self.device_id}. {hint}{reason}")

    def disconnect(self) -> None:
        if self._device is not None:
            self._device.close()
            self._device = None

    def describe(self) -> str:
        """Report the detected model and its button map.

        `poll()` reads the gripper buttons as `state.buttons[0]` and `[1]`, but which
        physical button lands at those indices comes from the pyspacemouse device
        profile, not from the puck. A model whose profile declares fewer than two
        buttons, or declares something other than LEFT/RIGHT first, silently breaks
        gripper control while motion keeps working, so surface the map explicitly.
        """
        if self._device is None:
            return "SpaceMouse backend is not connected."
        info = getattr(self._device, "info", None)
        model = getattr(info, "name", None) or getattr(self._device, "name", "<unknown>")
        button_names = tuple(getattr(info, "button_names", ()) or ())
        if not button_names:
            button_count = len(getattr(info, "button_specs", ()) or ())
            button_names = tuple(f"BUTTON_{index}" for index in range(button_count))
        mapping = ", ".join(f"[{index}]={name}" for index, name in enumerate(button_names)) or "<none>"
        summary = f"model={model} buttons={len(button_names)} map={mapping}"
        if len(button_names) < 2:
            summary += (
                "  WARNING: gripper control needs buttons [0] and [1]; "
                "this profile cannot drive the gripper."
            )
        return summary

    def poll(self) -> SpaceMouseReading | None:
        if self._device is None:
            raise RuntimeError("SpaceMouse backend is not connected.")
        state = self._device.read()
        if state is None:
            return None
        translation = np.array([state.x, state.y, state.z], dtype=np.float64)
        rotation = np.array([state.roll, state.pitch, state.yaw], dtype=np.float64)
        buttons = (bool(state.buttons[0]), bool(state.buttons[1]))
        # pyspacemouse writes `t` only while processing a HID report, and `read()` returns the
        # previous tuple untouched when the queue is empty. That makes this field the one part
        # of the reading that distinguishes a new report from a copy of the last one.
        report_time = getattr(state, "t", None)
        return SpaceMouseReading(
            translation=translation,
            rotation=rotation,
            buttons=buttons,
            timestamp=None if report_time is None else float(report_time),
        )
