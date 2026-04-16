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

import time
from dataclasses import dataclass
from typing import Iterable


NINTENDO_VENDOR_ID = 0x057E
JOYCON_L_PRODUCT_ID = 0x2006
JOYCON_R_PRODUCT_ID = 0x2007
PRO_CONTROLLER_PRODUCT_ID = 0x2009
SUPPORTED_PRODUCT_IDS = {
    JOYCON_L_PRODUCT_ID,
    JOYCON_R_PRODUCT_ID,
    PRO_CONTROLLER_PRODUCT_ID,
}
NEUTRAL_RUMBLE = [0x00, 0x01, 0x40, 0x40, 0x00, 0x01, 0x40, 0x40]


@dataclass(frozen=True)
class NintendoDeviceInfo:
    index: int
    path: bytes
    vendor_id: int
    product_id: int
    product_string: str
    manufacturer_string: str
    serial_number: str
    interface_number: int | None

    @property
    def controller_type(self) -> str:
        if self.product_id == JOYCON_L_PRODUCT_ID:
            return "left"
        if self.product_id == JOYCON_R_PRODUCT_ID:
            return "right"
        if self.product_id == PRO_CONTROLLER_PRODUCT_ID:
            return "pro"
        return "unknown"


@dataclass(frozen=True)
class NintendoControllerReading:
    controller_type: str
    left_stick: tuple[float, float]
    right_stick: tuple[float, float]
    buttons: frozenset[str]
    accel_g: tuple[float, float, float]
    gyro_dps: tuple[float, float, float]


def _text(value) -> str:
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return str(value)


def _path_as_bytes(value) -> bytes:
    if isinstance(value, bytes):
        return value
    return str(value).encode()


def normalize_report(data: Iterable[int]) -> list[int]:
    report = list(data)
    if report and report[0] == 0x00 and len(report) > 1 and report[1] in (0x21, 0x30, 0x3F):
        return report[1:]
    return report


def parse_stick_at(report: list[int], offset: int) -> tuple[float, float]:
    raw_x = report[offset] | ((report[offset + 1] & 0x0F) << 8)
    raw_y = (report[offset + 1] >> 4) | (report[offset + 2] << 4)
    norm_x = max(-1.0, min(1.0, (raw_x - 2048) / 2048.0))
    norm_y = max(-1.0, min(1.0, (raw_y - 2048) / 2048.0))
    return norm_x, norm_y


def parse_buttons(report: list[int], controller_type: str) -> frozenset[str]:
    right = report[3]
    shared = report[4]
    left = report[5]
    buttons: list[str] = []

    right_mapping = [
        ("Y", 0x01),
        ("X", 0x02),
        ("B", 0x04),
        ("A", 0x08),
        ("SR_R", 0x10),
        ("SL_R", 0x20),
        ("R", 0x40),
        ("ZR", 0x80),
    ]
    left_mapping = [
        ("DOWN", 0x01),
        ("UP", 0x02),
        ("RIGHT", 0x04),
        ("LEFT", 0x08),
        ("SR_L", 0x10),
        ("SL_L", 0x20),
        ("L", 0x40),
        ("ZL", 0x80),
    ]

    if controller_type in ("right", "pro"):
        buttons.extend(name for name, mask in right_mapping if right & mask)
        if shared & 0x02:
            buttons.append("+")
        if shared & 0x04:
            buttons.append("R_STICK")
        if shared & 0x10:
            buttons.append("HOME")

    if controller_type in ("left", "pro"):
        buttons.extend(name for name, mask in left_mapping if left & mask)
        if shared & 0x01:
            buttons.append("-")
        if shared & 0x08:
            buttons.append("L_STICK")
        if shared & 0x20:
            buttons.append("CAPTURE")
    return frozenset(buttons)


def parse_imu_sample(
    report: list[int],
    sample_index: int = 0,
) -> tuple[tuple[float, float, float], tuple[float, float, float]]:
    offset = 13 + sample_index * 12
    accel_raw = [
        int.from_bytes(bytes(report[offset + i : offset + i + 2]), "little", signed=True)
        for i in (0, 2, 4)
    ]
    gyro_raw = [
        int.from_bytes(bytes(report[offset + i : offset + i + 2]), "little", signed=True)
        for i in (6, 8, 10)
    ]
    accel_g = tuple(value / 4096.0 for value in accel_raw)
    gyro_dps = tuple(value / 13.371 for value in gyro_raw)
    return accel_g, gyro_dps


def parse_input_report(report: list[int], controller_type: str) -> NintendoControllerReading | None:
    if len(report) < 49 or report[0] != 0x30:
        return None
    left_stick = parse_stick_at(report, 6) if controller_type in ("left", "pro") else (0.0, 0.0)
    right_stick = parse_stick_at(report, 9) if controller_type in ("right", "pro") else (0.0, 0.0)
    accel_g, gyro_dps = parse_imu_sample(report, 0)
    return NintendoControllerReading(
        controller_type=controller_type,
        left_stick=left_stick,
        right_stick=right_stick,
        buttons=parse_buttons(report, controller_type),
        accel_g=accel_g,
        gyro_dps=gyro_dps,
    )


def enumerate_nintendo_devices() -> list[NintendoDeviceInfo]:
    try:
        import hid
    except ImportError as exc:
        raise ImportError(
            "Nintendo teleop requires the optional 'hidapi' package. Install with "
            "`pip install 'lerobot[nintendo]'` or run the joycon-robotics Ubuntu setup script "
            "in this environment."
        ) from exc

    devices: list[NintendoDeviceInfo] = []
    for raw in hid.enumerate(0, 0):
        vendor_id = int(raw.get("vendor_id") or 0)
        product_id = int(raw.get("product_id") or 0)
        if vendor_id != NINTENDO_VENDOR_ID or product_id not in SUPPORTED_PRODUCT_IDS:
            continue
        path = raw.get("path")
        if not path:
            continue
        devices.append(
            NintendoDeviceInfo(
                index=len(devices),
                path=_path_as_bytes(path),
                vendor_id=vendor_id,
                product_id=product_id,
                product_string=_text(raw.get("product_string")),
                manufacturer_string=_text(raw.get("manufacturer_string")),
                serial_number=_text(raw.get("serial_number") or raw.get("serial")),
                interface_number=raw.get("interface_number"),
            )
        )
    return devices


def choose_device(
    devices: list[NintendoDeviceInfo],
    *,
    controller: str = "any",
    side: str = "any",
    index: int | None = None,
) -> NintendoDeviceInfo:
    if index is not None:
        for device in devices:
            if device.index == index:
                return device
        raise ConnectionError(f"No supported Nintendo controller with index {index}.")

    if controller != "any":
        matched = [device for device in devices if device.controller_type == controller]
        if not matched:
            raise ConnectionError(f"No {controller} Nintendo controller found.")
        return matched[0]

    if side != "any":
        matched = [device for device in devices if device.controller_type == side]
        if not matched:
            raise ConnectionError(f"No {side} Nintendo controller found.")
        return matched[0]

    if not devices:
        raise ConnectionError("No supported Nintendo controller found.")
    return devices[0]


def write_report(device, packet: list[int]) -> None:
    sizes = [49, 64, len(packet)]
    last_error: Exception | None = None
    for size in dict.fromkeys(sizes):
        payload = packet[:size] + [0x00] * max(0, size - len(packet))
        try:
            device.write(bytes(payload))
            return
        except OSError as exc:
            last_error = exc
    raise OSError(f"write failed for report sizes {sizes}: {last_error}")


class NintendoHIDDriver:
    def __init__(
        self,
        *,
        controller: str = "any",
        side: str = "any",
        device_id: int | None = None,
        read_timeout_ms: int = 1,
    ) -> None:
        self.controller = controller
        self.side = side
        self.device_id = device_id
        self.read_timeout_ms = int(read_timeout_ms)
        self.info: NintendoDeviceInfo | None = None
        self.device = None
        self.packet_number = 0
        self._last_reading: NintendoControllerReading | None = None

    def connect(self) -> None:
        try:
            import hid
        except ImportError as exc:
            raise ImportError(
                "Nintendo teleop requires the optional 'hidapi' package. Install with "
                "`pip install 'lerobot[nintendo]'` or run the joycon-robotics Ubuntu setup script "
                "in this environment."
            ) from exc

        devices = enumerate_nintendo_devices()
        self.info = choose_device(devices, controller=self.controller, side=self.side, index=self.device_id)
        self.device = hid.device()
        self.device.open_path(self.info.path)
        self.device.set_nonblocking(0)
        self.initialize()
        self.device.set_nonblocking(1)

    def disconnect(self) -> None:
        if self.device is not None:
            try:
                self.device.close()
            except OSError:
                pass
        self.device = None
        self.info = None
        self._last_reading = None

    def read(self, timeout_ms: int | None = None) -> list[int]:
        if self.device is None:
            raise RuntimeError("Nintendo HID device is not connected.")
        resolved_timeout = self.read_timeout_ms if timeout_ms is None else int(timeout_ms)
        return normalize_report(self.device.read(64, timeout_ms=resolved_timeout))

    def send_subcommand(self, subcommand: int, argument: list[int], wait_ack: bool = True) -> bool:
        if self.device is None:
            raise RuntimeError("Nintendo HID device is not connected.")
        packet = [0x01, self.packet_number & 0x0F]
        packet.extend(NEUTRAL_RUMBLE)
        packet.append(subcommand)
        packet.extend(argument)
        write_report(self.device, packet)
        self.packet_number = (self.packet_number + 1) & 0x0F

        if not wait_ack:
            time.sleep(0.03)
            return True

        deadline = time.monotonic() + 1.0
        while time.monotonic() < deadline:
            report = self.read(timeout_ms=80)
            if len(report) >= 15 and report[0] == 0x21 and report[14] == subcommand:
                return bool(report[13] & 0x80)
        return False

    def initialize(self) -> None:
        for subcommand, argument in (
            (0x03, [0x30]),
            (0x40, [0x01]),
            (0x48, [0x01]),
        ):
            self.send_subcommand(subcommand, argument, wait_ack=True)
            time.sleep(0.08)

    def poll(self) -> NintendoControllerReading | None:
        if self.info is None:
            raise RuntimeError("Nintendo HID device is not connected.")

        latest = None
        while True:
            report = self.read()
            if not report:
                break
            reading = parse_input_report(report, self.info.controller_type)
            if reading is not None:
                latest = reading

        if latest is not None:
            self._last_reading = latest
        return latest
