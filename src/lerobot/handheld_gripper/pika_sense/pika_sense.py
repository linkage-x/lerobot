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

import importlib
import logging
import sys
import time, math
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import serial.tools.list_ports

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError
from pika.sense import Sense

from ..handheld_gripper import HandheldGripper
from .configuration_pika_sense import PikaSenseConfig

logger = logging.getLogger(__name__)

class PikaSense(HandheldGripper):
    def __init__(self, config: PikaSenseConfig):
        super().__init__(config)

        self.config = config
        self.port = config.port
        self.warmup_s = config.warmup_s

        self._sense: Any | None = None
        self._connected = False

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.read_lock: Lock = Lock()
        self.latest_width_mm: float | None = None
        self.latest_timestamp: float | None = None
        self.last_error: Exception | None = None
        self.new_reading_event: Event = Event()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.port})"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_handheld_grippers() -> list[dict[str, Any]]:
        grippers: list[dict[str, Any]] = []
        for port in serial.tools.list_ports.comports():
            grippers.append(
                {
                    "name": f"Pika Sense @ {port.device}",
                    "type": "PikaSense",
                    "id": port.device,
                    "port": port.device,
                    "description": port.description,
                    "manufacturer": port.manufacturer,
                    "serial_number": port.serial_number,
                    "hwid": port.hwid,
                }
            )
        return grippers

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        self._sense = Sense(port=self.port)

        connected = self._sense.connect()
        if not connected:
            self._sense = None
            raise ConnectionError(f"Failed to connect {self}.")

        try:
            self._connected = True
            self._start_read_thread()

            if warmup and self.warmup_s > 0:
                timeout_ms = max(int(self.warmup_s * 1000), 200)
                self.async_read(timeout_ms=timeout_ms)
        except Exception:
            self.disconnect()
            raise

        logger.info("%s connected.", self)

    def _start_read_thread(self) -> None:
        self.stop_event = Event()
        self.new_reading_event.clear()
        self.thread = Thread(target=self._read_loop, daemon=True, name=f"{self.__class__.__name__}-{self.port}")
        self.thread.start()

    def _read_loop(self) -> None:
        poll_interval_s = 1.0 / self.fps if self.fps else 0.0
        while self.stop_event is not None and not self.stop_event.is_set():
            try:
                width_mm = self._read_from_hardware()
                timestamp = time.perf_counter()
                with self.read_lock:
                    self.latest_width_mm = width_mm
                    self.latest_timestamp = timestamp
                    self.last_error = None
                self.new_reading_event.set()
            except Exception as exc:  # noqa: BLE001
                logger.exception("%s background read failed.", self)
                with self.read_lock:
                    self.last_error = exc
                self.new_reading_event.set()

            if self.stop_event is None:
                break
            self.stop_event.wait(poll_interval_s if poll_interval_s > 0 else 0.01)

    def _read_from_hardware(self) -> float:
        if self._sense is None:
            raise DeviceNotConnectedError(f"{self} Sense instance is not initialized.")

        width_mm = self._sense.get_gripper_distance()
        return float(width_mm)

    @check_if_not_connected
    def read(self) -> float:
        width_mm = self._read_from_hardware()
        timestamp = time.perf_counter()
        with self.read_lock:
            self.latest_width_mm = width_mm
            self.latest_timestamp = timestamp
            self.last_error = None
        self.new_reading_event.set()
        return width_mm

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> float:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if timeout_ms < 0:
            raise ValueError(f"`timeout_ms` must be >= 0, but {timeout_ms} is provided.")

        if not self.new_reading_event.wait(timeout_ms / 1000):
            raise TimeoutError(f"{self} timed out waiting for a new gripper-width sample.")

        self.new_reading_event.clear()
        with self.read_lock:
            if self.last_error is not None:
                raise RuntimeError(f"{self} background read failed.") from self.last_error
            if self.latest_width_mm is None:
                raise RuntimeError(f"{self} has not produced a gripper-width sample yet.")
            return float(self.latest_width_mm)

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> float:
        if max_age_ms < 0:
            raise ValueError(f"`max_age_ms` must be >= 0, but {max_age_ms} is provided.")

        with self.read_lock:
            if self.last_error is not None:
                raise RuntimeError(f"{self} background read failed.") from self.last_error
            if self.latest_width_mm is None or self.latest_timestamp is None:
                raise RuntimeError(f"{self} has not produced a gripper-width sample yet.")

            age_ms = (time.perf_counter() - self.latest_timestamp) * 1000
            if age_ms > max_age_ms:
                raise TimeoutError(
                    f"{self} latest gripper-width sample is stale ({age_ms:.1f} ms > {max_age_ms} ms)."
                )

            return float(self.latest_width_mm)

    @check_if_not_connected
    def disconnect(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None:
            self.thread.join(timeout=1.0)
            self.thread = None

        if self._sense is not None:
            self._sense.disconnect()
            self._sense = None

        self.stop_event = None
        self.new_reading_event.clear()
        self._connected = False

        with self.read_lock:
            self.last_error = None

        logger.info("%s disconnected.", self)
