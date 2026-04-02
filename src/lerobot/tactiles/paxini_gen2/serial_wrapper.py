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

import time
from threading import Lock, RLock

import numpy as np
import serial
from numpy.typing import NDArray
from serial import Serial

from .configuration_paxini_gen2_omega import PAXINI_NUM_DIMENSIONS, PAXINI_NUM_TAXELS

PAXINI_PACKET_HEAD = bytes([0x55, 0xAA, 0x7B, 0x7B])
PAXINI_PACKET_TAIL = bytes([0x55, 0xAA, 0x7D, 0x7D])
PAXINI_PROTOCOL_MIN_RESPONSE_BYTES = 17
PAXINI_ADDR_BEGIN = 1038
PAXINI_ADDR_END = 1397
PAXINI_SENSING_BYTES = PAXINI_ADDR_END - PAXINI_ADDR_BEGIN + 1
PAXINI_SENSING_RESPONSE_BYTES = PAXINI_PROTOCOL_MIN_RESPONSE_BYTES + 6 + PAXINI_SENSING_BYTES
PAXINI_RECALIBRATION_SUCCESS_CODES = {0, 6}


def extract_low_high_byte(value: int) -> list[int]:
    high_byte = (value >> 8) & 0xFF
    low_byte = value & 0xFF
    return [low_byte, high_byte]


def decimal_to_hex(value: int) -> int:
    return value & 0xFF


class PaxiniSerialWrapper:
    _shared_wrappers: dict[str, "PaxiniSerialWrapper"] = {}
    _shared_wrappers_lock = Lock()

    def __init__(
        self,
        serial_port: str,
        baudrate: int,
        timeout: float,
        control_mode: int,
        provided_serial: Serial | None = None,
    ):
        self.serial_port = serial_port
        self.baudrate = baudrate
        self.timeout = timeout
        self.control_mode = control_mode
        self._provided_serial = provided_serial

        self._serial: Serial | None = None
        self._io_lock = RLock()
        self._state_lock = Lock()
        self._reference_count = 0
        self._calibrated_connect_ids: set[int] = set()
        self._box_version: str | None = None

    @classmethod
    def acquire(
        cls,
        serial_port: str,
        baudrate: int,
        timeout: float,
        control_mode: int,
        provided_serial: Serial | None = None,
    ) -> "PaxiniSerialWrapper":
        with cls._shared_wrappers_lock:
            wrapper = cls._shared_wrappers.get(serial_port)
            if wrapper is None:
                wrapper = cls(
                    serial_port=serial_port,
                    baudrate=baudrate,
                    timeout=timeout,
                    control_mode=control_mode,
                    provided_serial=provided_serial,
                )
                cls._shared_wrappers[serial_port] = wrapper
            else:
                wrapper._validate_shared_configuration(
                    baudrate=baudrate,
                    timeout=timeout,
                    control_mode=control_mode,
                    provided_serial=provided_serial,
                )

            wrapper._reference_count += 1
            return wrapper

    @property
    def is_open(self) -> bool:
        return isinstance(self._serial, serial.Serial) and self._serial.is_open

    @property
    def box_version(self) -> str | None:
        return self._box_version

    def release(self) -> None:
        should_close = False
        with self._shared_wrappers_lock:
            if self._reference_count <= 0:
                return

            self._reference_count -= 1
            if self._reference_count == 0:
                self._shared_wrappers.pop(self.serial_port, None)
                should_close = True

        if should_close:
            self.close()

    def open(self) -> None:
        with self._state_lock:
            if self.is_open:
                return

            try:
                if self._provided_serial is not None:
                    self._serial = self._provided_serial
                    if not self._serial.is_open:
                        self._serial.open()
                else:
                    self._serial = serial.Serial(
                        port=self.serial_port,
                        baudrate=self.baudrate,
                        timeout=self.timeout,
                    )
            except serial.SerialException as exc:
                raise ConnectionError(
                    f"Failed to open serial port {self.serial_port}: {exc}. "
                    f"Try running `sudo chmod 666 {self.serial_port}`."
                ) from exc

            self._calibrated_connect_ids.clear()

        try:
            with self._io_lock:
                self._box_version = self._get_control_box_version()
                self._set_control_box_mode(self.control_mode)
                actual_mode = self._get_control_box_mode()
                if actual_mode != self.control_mode:
                    raise RuntimeError(
                        f"Control box on {self.serial_port} is in mode {actual_mode}, "
                        f"expected {self.control_mode}."
                    )
        except Exception:
            self.close()
            raise

    def close(self) -> None:
        with self._state_lock:
            if self._serial is not None:
                try:
                    if self._serial.is_open:
                        self._serial.close()
                finally:
                    self._serial = None

            self._box_version = None
            self._calibrated_connect_ids.clear()

    def ensure_sensor_ready(self, connect_id: int, recalibrate: bool = True) -> None:
        self.open()

        if not recalibrate:
            return

        with self._state_lock:
            if connect_id in self._calibrated_connect_ids:
                return

            with self._io_lock:
                success = self._recalibrate_sensor_module(connect_id)
            if not success:
                raise RuntimeError(
                    f"Failed to recalibrate Paxini sensor connect_id={connect_id} on {self.serial_port}."
                )

            self._calibrated_connect_ids.add(connect_id)

    def read_module_sensing_data(self, connect_id: int) -> NDArray[np.int16]:
        self.open()
        with self._io_lock:
            return self._get_module_sensing_data(connect_id)

    def _validate_shared_configuration(
        self,
        baudrate: int,
        timeout: float,
        control_mode: int,
        provided_serial: Serial | None = None,
    ) -> None:
        mismatches: list[str] = []
        if self.baudrate != baudrate:
            mismatches.append(f"baudrate: existing={self.baudrate}, requested={baudrate}")
        if self.timeout != timeout:
            mismatches.append(f"timeout: existing={self.timeout}, requested={timeout}")
        if self.control_mode != control_mode:
            mismatches.append(f"control_mode: existing={self.control_mode}, requested={control_mode}")
        if provided_serial is not None and self._provided_serial not in (None, provided_serial):
            mismatches.append("provided_serial: existing wrapper already uses a different serial object")

        if mismatches:
            raise ValueError(
                f"Serial port {self.serial_port} is already managed by another Paxini wrapper with different "
                f"settings ({'; '.join(mismatches)}). Reuse the same control box settings for all sensors "
                "sharing this port."
            )

        if self._provided_serial is None and provided_serial is not None and not self.is_open:
            self._provided_serial = provided_serial

    def _require_serial(self) -> Serial:
        if not self.is_open or self._serial is None:
            raise ConnectionError(f"Paxini serial port {self.serial_port} is not open.")
        return self._serial

    def _calculate_lrc(self, data: list[int]) -> int:
        checksum = sum(data) & 0xFF
        return (~checksum + 1) & 0xFF

    def _build_protocol(
        self,
        fix_id: int,
        index: int,
        main_cmd: int,
        sub_cmd: list[int],
        length: list[int],
        data: list[int],
    ) -> bytes:
        lrc_packet = [fix_id, index, main_cmd] + sub_cmd + length + data
        lrc = self._calculate_lrc(lrc_packet)
        packet = list(PAXINI_PACKET_HEAD) + [fix_id, index, main_cmd] + sub_cmd + length + data + [lrc]
        packet += list(PAXINI_PACKET_TAIL)
        return bytes(packet)

    def _read_response(self, expected_bytes: int) -> bytes:
        ser = self._require_serial()
        response = ser.read(expected_bytes)
        if len(response) == 0:
            raise TimeoutError(
                f"Expected up to {expected_bytes} bytes from Paxini device on {self.serial_port}, "
                "received 0."
            )
        return response

    def _transact(
        self,
        packet: bytes,
        expected_response_bytes: int | None = None,
        settle_s: float = 0.0,
        use_in_waiting: bool = False,
    ) -> bytes:
        ser = self._require_serial()
        ser.reset_input_buffer()
        ser.write(packet)
        ser.flush()

        if settle_s > 0:
            time.sleep(settle_s)

        if use_in_waiting:
            response_length = ser.in_waiting
            if response_length <= 0:
                raise TimeoutError(f"No response available from Paxini device on {self.serial_port}.")
            return ser.read(response_length)

        if expected_response_bytes is None:
            raise ValueError("`expected_response_bytes` must be provided when `use_in_waiting` is False.")

        return self._read_response(expected_response_bytes)

    def _check_response(self, response: bytes, allowed_error_codes: set[int] | None = None) -> int:
        if len(response) < PAXINI_PROTOCOL_MIN_RESPONSE_BYTES:
            raise ValueError(f"Paxini response is too short: {len(response)} bytes.")
        if response[:4] != PAXINI_PACKET_HEAD:
            raise ValueError(f"Invalid Paxini response header: {response[:4].hex()}.")
        if response[-4:] != PAXINI_PACKET_TAIL:
            raise ValueError(f"Invalid Paxini response tail: {response[-4:].hex()}.")

        payload = list(response[4:-5])
        expected_lrc = self._calculate_lrc(payload)
        actual_lrc = response[-5]
        if actual_lrc != expected_lrc:
            raise ValueError(
                f"Invalid Paxini response checksum on {self.serial_port}: "
                f"expected={expected_lrc:#x}, actual={actual_lrc:#x}."
            )

        error_code = response[9]
        allowed = allowed_error_codes or {0}
        if error_code not in allowed:
            raise ValueError(f"Paxini device returned error code {error_code:#x}.")

        return error_code

    def _extract_data(self, response: bytes) -> bytes:
        return response[12:-5]

    def _get_control_box_version(self) -> str:
        packet = self._build_protocol(
            fix_id=0x0E,
            index=0x00,
            main_cmd=0x60,
            sub_cmd=[0xA0, 0x01],
            length=[0x00, 0x00],
            data=[],
        )
        response = self._transact(packet=packet, settle_s=0.01, use_in_waiting=True)
        self._check_response(response)
        return self._extract_data(response).decode("utf-8", errors="ignore")

    def _set_control_box_mode(self, mode: int) -> None:
        packet = self._build_protocol(
            fix_id=0x0E,
            index=0x00,
            main_cmd=0x70,
            sub_cmd=[0xC0, 0x0C],
            length=[0x01, 0x00],
            data=[mode],
        )
        response = self._transact(packet=packet, expected_response_bytes=17)
        self._check_response(response)

    def _get_control_box_mode(self) -> int:
        packet = self._build_protocol(
            fix_id=0x0E,
            index=0x00,
            main_cmd=0x70,
            sub_cmd=[0xC0, 0x0D],
            length=[0x00, 0x00],
            data=[],
        )
        response = self._transact(packet=packet, expected_response_bytes=18)
        self._check_response(response)
        response_data = self._extract_data(response)
        if not response_data:
            raise ValueError("Control box mode response does not contain any payload.")
        return int(response_data[0])

    def _set_module_port(self, connect_id: int) -> None:
        packet = self._build_protocol(
            fix_id=0x0E,
            index=0x00,
            main_cmd=0x70,
            sub_cmd=[0xB1, 0x0A],
            length=[0x01, 0x00],
            data=[decimal_to_hex((connect_id - 1) * 3)],
        )
        response = self._transact(packet=packet, expected_response_bytes=17)
        self._check_response(response)

    def _recalibrate_sensor_module(self, connect_id: int) -> bool:
        self._set_module_port(connect_id)
        packet = self._build_protocol(
            fix_id=0x0E,
            index=0x00,
            main_cmd=0x70,
            sub_cmd=[0xB0, 0x02],
            length=[0x02, 0x00],
            data=[0x03, 0x01],
        )
        response = self._transact(packet=packet, expected_response_bytes=18)
        if len(response) <= 9:
            raise ValueError(
                f"Recalibration response for connect_id={connect_id} on {self.serial_port} is too short: "
                f"{len(response)} bytes."
            )

        if len(response) >= PAXINI_PROTOCOL_MIN_RESPONSE_BYTES:
            error_code = self._check_response(response, allowed_error_codes=PAXINI_RECALIBRATION_SUCCESS_CODES)
            return error_code in PAXINI_RECALIBRATION_SUCCESS_CODES

        return response[9] == 6

    def _get_module_sensing_data(self, connect_id: int) -> NDArray[np.int16]:
        self._set_module_port(connect_id)
        packet = self._build_protocol(
            fix_id=0x0E,
            index=0x00,
            main_cmd=0x70,
            sub_cmd=[0xC0, 0x06],
            length=[0x05, 0x00],
            data=[0x7B] + extract_low_high_byte(PAXINI_ADDR_BEGIN) + extract_low_high_byte(PAXINI_SENSING_BYTES),
        )
        response = self._transact(packet=packet, expected_response_bytes=PAXINI_SENSING_RESPONSE_BYTES)
        self._check_response(response)

        response_data = self._extract_data(response)
        if len(response_data) < 6 + PAXINI_SENSING_BYTES:
            raise ValueError(
                f"Paxini sensing response payload is too short: expected at least {6 + PAXINI_SENSING_BYTES} bytes, "
                f"received {len(response_data)}."
            )

        sensing_data = response_data[6 : 6 + PAXINI_SENSING_BYTES]
        sensing_data_array = np.frombuffer(sensing_data, dtype=np.uint8).astype(np.int16)
        sensing_data_array = sensing_data_array.reshape(PAXINI_NUM_TAXELS, PAXINI_NUM_DIMENSIONS)

        xy = sensing_data_array[:, 0:2]
        sensing_data_array[:, 0:2] = np.where(xy >= 128, xy - 256, xy)
        return sensing_data_array


__all__ = [
    "PAXINI_NUM_DIMENSIONS",
    "PAXINI_NUM_TAXELS",
    "PaxiniSerialWrapper",
]
