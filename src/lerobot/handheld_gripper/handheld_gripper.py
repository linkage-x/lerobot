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

import abc
import warnings
from typing import Any

from .configs import HandheldGripperConfig


class HandheldGripper(abc.ABC):
    def __init__(self, config: HandheldGripperConfig):
        self.fps: int | None = config.fps

    def __enter__(self):
        self.connect()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.disconnect()

    def __del__(self) -> None:
        try:
            if self.is_connected:
                self.disconnect()
        except Exception:  # nosec B110
            pass

    @property
    @abc.abstractmethod
    def is_connected(self) -> bool:
        pass

    @staticmethod
    @abc.abstractmethod
    def find_handheld_grippers() -> list[dict[str, Any]]:
        pass

    @abc.abstractmethod
    def connect(self, warmup: bool = True) -> None:
        pass

    @abc.abstractmethod
    def read(self) -> float:
        pass

    @abc.abstractmethod
    def async_read(self, timeout_ms: float = ...) -> float:
        pass

    def read_latest(self, max_age_ms: int = 500) -> float:
        warnings.warn(
            f"{self.__class__.__name__}.read_latest() is not implemented. "
            "Please override read_latest(); it will be required in future releases.",
            FutureWarning,
            stacklevel=2,
        )
        return self.async_read()

    @abc.abstractmethod
    def disconnect(self) -> None:
        pass
