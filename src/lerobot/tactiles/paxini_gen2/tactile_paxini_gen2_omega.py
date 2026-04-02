# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
import time
from threading import Event, Lock, Thread
from typing import Any

import serial.tools.list_ports
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..tactile import Tactile
from .configuration_paxini_gen2_omega import PaxiniGen2OmegaTactileConfig
from .serial_wrapper import PaxiniSerialWrapper

logger = logging.getLogger(__name__)


class PaxiniGen2OmegaTactile(Tactile):
    def __init__(self, config: PaxiniGen2OmegaTactileConfig):
        super().__init__(config)

        self.config = config
        self.serial_port = config.serial_port
        self.connect_id = config.connect_id
        self.control_mode = config.control_mode

        self.wrapper: PaxiniSerialWrapper | None = None
        self._is_connected = False

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event = Event()

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_port}, connect_id={self.connect_id})"

    @property
    def is_connected(self) -> bool:
        return self._is_connected and self.wrapper is not None and self.wrapper.is_open

    @staticmethod
    def find_tactiles() -> list[dict[str, Any]]:
        found_tactiles: list[dict[str, Any]] = []
        for port in serial.tools.list_ports.comports():
            found_tactiles.append(
                {
                    "name": f"Paxini tactile @ {port.device}",
                    "type": "PaxiniGen2Omega",
                    "id": port.device,
                    "serial_port": port.device,
                    "description": port.description,
                    "hwid": port.hwid,
                }
            )
        return found_tactiles

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        wrapper = PaxiniSerialWrapper.acquire(
            serial_port=self.config.serial_port,
            baudrate=self.config.baudrate,
            timeout=self.config.timeout,
            control_mode=self.config.control_mode,
            provided_serial=self.config.serial,
        )

        try:
            wrapper.ensure_sensor_ready(self.connect_id, recalibrate=True)
            self.wrapper = wrapper
            self._start_read_thread()
            self._is_connected = True

            if warmup:
                timeout_ms = max(1000.0, (1000.0 / self.fps) * 2.0) if self.fps else 1000.0
                self.async_read(timeout_ms=timeout_ms)
        except Exception:
            self._is_connected = False
            self._stop_read_thread()
            self.wrapper = None
            wrapper.release()
            raise

        logger.info("%s connected.", self)

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()
        timeout_ms = max(1000.0, (1000.0 / self.fps) * 2.0) if self.fps else 1000.0
        return self.async_read(timeout_ms=timeout_ms)

    def _read_loop(self) -> None:
        stop_event = self.stop_event
        if stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        target_period_s = 1.0 / self.fps if self.fps and self.fps > 0 else None

        while not stop_event.is_set():
            loop_started_at = time.perf_counter()
            try:
                wrapper = self.wrapper
                if wrapper is None:
                    raise DeviceNotConnectedError(f"{self} wrapper is not initialized.")

                tactile_frame = wrapper.read_module_sensing_data(self.connect_id)
                capture_time = time.perf_counter()

                with self.frame_lock:
                    self.latest_frame = tactile_frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0
            except DeviceNotConnectedError:
                break
            except Exception as exc:
                failure_count += 1
                if failure_count <= 10:
                    logger.warning("Error reading tactile frame for %s: %s", self, exc)
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from exc

            if target_period_s is None:
                time.sleep(0.001)
            else:
                elapsed_s = time.perf_counter() - loop_started_at
                remaining_s = target_period_s - elapsed_s
                if remaining_s > 0:
                    time.sleep(remaining_s)

    def _start_read_thread(self) -> None:
        self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self.thread.start()
        time.sleep(0.05)

    def _stop_read_thread(self) -> None:
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for tactile frame from {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no tactile frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any tactile frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest tactile frame is too old: {age_ms:.1f} ms "
                f"(max allowed: {max_age_ms} ms)."
            )

        return frame

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        wrapper = self.wrapper
        self._is_connected = False
        self._stop_read_thread()
        self.wrapper = None

        if wrapper is not None:
            wrapper.release()

        logger.info("%s disconnected.", self)
