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

import ctypes
import logging
import sys
import threading
import time
from threading import Event, Lock, Thread
from typing import Any

import cv2  # type: ignore
import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..utils import get_cv2_rotation
from .configuration_hikrobot import ColorMode, HikrobotCameraConfig

logger = logging.getLogger(__name__)

_MVS_PYTHON_PATHS = [
    "/opt/MVS/Samples/64/Python",
    "/opt/MVS/Samples/32/Python",
]
_PIXEL_FORMAT_RGB8 = 0x02180014
_PIXEL_FORMAT_BGR8 = 0x02180015
_MVS_TIMEOUT_ERROR = 0x80000006
_MVS_OPEN_LOCK = threading.Lock()
_AUTO_MODE_MAP = {
    "off": 0,
    "once": 1,
    "continuous": 2,
}
_BALANCE_RATIO_SELECTOR_MAP = {
    "red": 0,
    "green": 1,
    "blue": 2,
}


def _load_mvs_sdk():
    try:
        import MvCameraControl_class as mvs  # noqa: PLC0415

        return mvs
    except ImportError:
        pass

    for python_path in reversed(_MVS_PYTHON_PATHS):
        if python_path not in sys.path:
            sys.path.insert(0, python_path)

    try:
        from MvImport import MvCameraControl_class as mvs  # noqa: PLC0415

        return mvs
    except ImportError:
        pass

    try:
        import MvCameraControl_class as mvs  # noqa: PLC0415

        return mvs
    except ImportError as exc:
        raise RuntimeError(
            "Hikrobot MVS SDK not found. "
            "Install the Linux runtime under /opt/MVS and ensure the Python bindings are available under "
            "/opt/MVS/Samples/64/Python or /opt/MVS/Samples/32/Python, or rebuild the Docker image with "
            "INSTALL_HIKROBOT_SDK=true."
        ) from exc


def _decode_char_buffer(field: Any) -> str:
    raw = bytes(field) if not isinstance(field, (bytes, bytearray)) else field
    return raw.decode("utf-8", errors="ignore").rstrip("\x00")


def _extract_device_info(entry: Any, mvs_module: Any) -> Any:
    if hasattr(entry, "contents"):
        return entry.contents
    if hasattr(entry, "SpecialInfo"):
        return entry
    return ctypes.cast(entry, ctypes.POINTER(mvs_module.MV_CC_DEVICE_INFO)).contents


class HikrobotCamera(Camera):
    def __init__(self, config: HikrobotCameraConfig, mvs_module: Any | None = None):
        super().__init__(config)

        self.config = config
        self.serial = config.serial
        self.device_index = config.device_index
        self.color_mode = config.color_mode
        self.warmup_s = config.warmup_s
        self.timeout_ms = config.timeout_ms
        self._white_balance_auto_mode = config.white_balance_auto

        self.capture_width = self.width
        self.capture_height = self.height
        self.rotation = get_cv2_rotation(config.rotation)
        if (
            self.capture_width is not None
            and self.capture_height is not None
            and self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]
        ):
            self.capture_width, self.capture_height = self.height, self.width

        self._mvs = mvs_module or _load_mvs_sdk()
        self._cam = self._mvs.MvCamera()
        self._connected = False
        self._started = False

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.new_frame_event: Event = Event()

    def __str__(self) -> str:
        identifier = self.serial if self.serial is not None else self.device_index
        return f"{self.__class__.__name__}({identifier})"

    @property
    def is_connected(self) -> bool:
        return self._connected

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        mvs = _load_mvs_sdk()
        transport_flag = getattr(mvs, "MV_USB_DEVICE", 0) | getattr(mvs, "MV_GIGE_DEVICE", 0)
        device_list = mvs.MV_CC_DEVICE_INFO_LIST()
        ret = mvs.MvCamera.MV_CC_EnumDevices(transport_flag, device_list)
        if ret != 0:
            raise RuntimeError(f"MVS EnumDevices failed: 0x{ret:08x}")

        cameras = []
        for idx in range(device_list.nDeviceNum):
            device_info = _extract_device_info(device_list.pDeviceInfo[idx], mvs)
            usb_info = getattr(device_info.SpecialInfo, "stUsb3VInfo", None)
            serial = _decode_char_buffer(usb_info.chSerialNumber) if usb_info is not None else str(idx)
            model = _decode_char_buffer(usb_info.chModelName) if usb_info is not None else ""
            manufacturer = _decode_char_buffer(usb_info.chVendorName) if usb_info is not None else ""
            cameras.append(
                {
                    "name": f"Hikrobot Camera @ {serial}",
                    "type": "Hikrobot",
                    "id": serial,
                    "device_index": idx,
                    "manufacturer": manufacturer,
                    "model": model,
                }
            )

        return cameras

    def _transport_layer_flag(self) -> int:
        if self.config.transport_layer == "usb":
            return getattr(self._mvs, "MV_USB_DEVICE", 0)
        if self.config.transport_layer == "gige":
            return getattr(self._mvs, "MV_GIGE_DEVICE", 0)
        return getattr(self._mvs, "MV_USB_DEVICE", 0) | getattr(self._mvs, "MV_GIGE_DEVICE", 0)

    def _set_value(self, method_name: str, key: str, value: Any) -> None:
        method = getattr(self._cam, method_name, None)
        if method is None:
            return
        ret = method(key, value)
        if ret not in (0, None):
            raise RuntimeError(f"{self} failed to set {key}: 0x{ret:08x}")

    def _set_white_balance_auto_mode(self, mode: str) -> None:
        self._set_value("MV_CC_SetEnumValue", "BalanceWhiteAuto", _AUTO_MODE_MAP[mode])
        self._white_balance_auto_mode = mode

    def _get_enum_values(self, key: str) -> tuple[int, list[int]]:
        method = getattr(self._cam, "MV_CC_GetEnumValue", None)
        if method is None:
            raise RuntimeError(f"{self} camera handle does not expose MV_CC_GetEnumValue for {key}.")

        enum_value = self._mvs.MVCC_ENUMVALUE()
        ret = method(key, enum_value)
        if ret != 0:
            raise RuntimeError(f"{self} failed to query {key}: 0x{ret:08x}")

        supported_num = int(getattr(enum_value, "nSupportedNum", 0))
        supported_values = [int(enum_value.nSupportValue[idx]) for idx in range(supported_num)]
        return int(enum_value.nCurValue), supported_values

    def _device_pixel_format(self) -> int:
        if self.color_mode == ColorMode.BGR:
            return _PIXEL_FORMAT_BGR8
        return _PIXEL_FORMAT_RGB8

    def _require_device_color_stream(self) -> None:
        requested_pixel_format = self._device_pixel_format()
        current_value, supported_values = self._get_enum_values("PixelFormat")
        if requested_pixel_format not in supported_values:
            requested_name = "BGR888" if requested_pixel_format == _PIXEL_FORMAT_BGR8 else "RGB888"
            raise RuntimeError(
                f"{self} does not support device-side {requested_name} output. "
                f"Current PixelFormat=0x{current_value:08x}."
            )

        self._set_value("MV_CC_SetEnumValue", "PixelFormat", requested_pixel_format)
        applied_value, _ = self._get_enum_values("PixelFormat")
        if applied_value != requested_pixel_format:
            requested_name = "BGR888" if requested_pixel_format == _PIXEL_FORMAT_BGR8 else "RGB888"
            raise RuntimeError(
                f"{self} failed to switch PixelFormat to {requested_name}. "
                f"Current PixelFormat=0x{applied_value:08x}."
            )

    def _configure_gamma(self) -> None:
        if self.config.gamma is None:
            disable_gamma = getattr(self._cam, "MV_CC_SetBoolValue", None)
            if disable_gamma is not None:
                disable_gamma("GammaEnable", False)
            return

        try:
            self._set_value("MV_CC_SetBoolValue", "GammaEnable", True)
        except Exception:
            pass

        try:
            self._set_value("MV_CC_SetFloatValue", "Gamma", float(self.config.gamma))
            return
        except Exception:
            pass

        gamma_method = getattr(self._cam, "MV_CC_SetGammaValue", None)
        if gamma_method is None:
            raise RuntimeError(f"{self} camera handle does not expose a gamma configuration API.")

        ret = gamma_method(self._device_pixel_format(), float(self.config.gamma))
        if ret not in (0, None):
            raise RuntimeError(f"{self} failed to set gamma: 0x{ret:08x}")

    def _set_balance_ratio(self, selector: str, value: int) -> None:
        selector = selector.lower()
        string_setter = getattr(self._cam, "MV_CC_SetEnumValueByString", None)
        ret = None
        if string_setter is not None:
            try:
                ret = string_setter("BalanceRatioSelector", selector.capitalize())
            except Exception:
                ret = None
        if ret not in (0, None):
            ret = None
        if ret is None:
            self._set_value("MV_CC_SetEnumValue", "BalanceRatioSelector", _BALANCE_RATIO_SELECTOR_MAP[selector])
        self._set_value("MV_CC_SetIntValueEx", "BalanceRatio", int(value))

    def get_balance_ratio(self, selector: str) -> int:
        selector = selector.lower()
        string_setter = getattr(self._cam, "MV_CC_SetEnumValueByString", None)
        ret = None
        if string_setter is not None:
            try:
                ret = string_setter("BalanceRatioSelector", selector.capitalize())
            except Exception:
                ret = None
        if ret not in (0, None):
            ret = None
        if ret is None:
            self._set_value("MV_CC_SetEnumValue", "BalanceRatioSelector", _BALANCE_RATIO_SELECTOR_MAP[selector])
        int_value = self._mvs.MVCC_INTVALUE_EX()
        ret = self._cam.MV_CC_GetIntValueEx("BalanceRatio", int_value)
        if ret != 0:
            raise RuntimeError(f"{self} failed to query BalanceRatio for {selector}: 0x{ret:08x}")
        return int(int_value.nCurValue)

    def get_white_balance_ratios(self) -> dict[str, int]:
        previous_mode = self._white_balance_auto_mode
        if previous_mode != "off":
            self._set_white_balance_auto_mode("off")
        ratios = {
            "red": self.get_balance_ratio("red"),
            "green": self.get_balance_ratio("green"),
            "blue": self.get_balance_ratio("blue"),
        }
        if previous_mode != "off":
            self._set_white_balance_auto_mode(previous_mode)
        return ratios

    def _configure_white_balance(self) -> None:
        manual_ratios = {
            "red": self.config.white_balance_red,
            "green": self.config.white_balance_green,
            "blue": self.config.white_balance_blue,
        }
        if any(value is not None for value in manual_ratios.values()):
            self._set_white_balance_auto_mode("off")
            for selector, value in manual_ratios.items():
                if value is not None:
                    self._set_balance_ratio(selector, value)
            return

        self._set_white_balance_auto_mode(self.config.white_balance_auto)

    def _enumerate_devices(self) -> Any:
        device_list = self._mvs.MV_CC_DEVICE_INFO_LIST()
        ret = self._mvs.MvCamera.MV_CC_EnumDevices(self._transport_layer_flag(), device_list)
        if ret != 0:
            raise RuntimeError(f"MVS EnumDevices failed: 0x{ret:08x}")
        if device_list.nDeviceNum == 0:
            raise RuntimeError("No Hikrobot devices found.")
        return device_list

    def _find_device_index(self, device_list: Any) -> int:
        if self.serial is not None:
            for idx in range(device_list.nDeviceNum):
                device_info = _extract_device_info(device_list.pDeviceInfo[idx], self._mvs)
                usb_info = getattr(device_info.SpecialInfo, "stUsb3VInfo", None)
                serial = _decode_char_buffer(usb_info.chSerialNumber) if usb_info is not None else ""
                if serial == self.serial:
                    return idx
            raise RuntimeError(f"Hikrobot device with serial {self.serial!r} not found.")

        if self.device_index is not None:
            if self.device_index >= device_list.nDeviceNum:
                raise RuntimeError(
                    f"Hikrobot device_index {self.device_index} is out of range for {device_list.nDeviceNum} devices."
                )
            return self.device_index

        return 0

    def _open(self) -> None:
        device_list = self._enumerate_devices()
        idx = self._find_device_index(device_list)
        device_info = _extract_device_info(device_list.pDeviceInfo[idx], self._mvs)

        ret = self._cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            raise RuntimeError(f"MVS CreateHandle failed: 0x{ret:08x}")
        ret = self._cam.MV_CC_OpenDevice(getattr(self._mvs, "MV_ACCESS_Exclusive", 1), 0)
        if ret != 0:
            raise RuntimeError(f"MVS OpenDevice failed: 0x{ret:08x}")

        self._require_device_color_stream()

        if self.capture_width is not None:
            self._set_value("MV_CC_SetIntValue", "Width", int(self.capture_width))
        if self.capture_height is not None:
            self._set_value("MV_CC_SetIntValue", "Height", int(self.capture_height))
        if self.fps is not None:
            self._set_value("MV_CC_SetBoolValue", "AcquisitionFrameRateEnable", True)
            self._set_value("MV_CC_SetFloatValue", "AcquisitionFrameRate", float(self.fps))
        if self.config.exposure_us is not None:
            self._set_value("MV_CC_SetEnumValue", "ExposureAuto", 0)
            self._set_value("MV_CC_SetFloatValue", "ExposureTime", float(self.config.exposure_us))
        else:
            self._set_value("MV_CC_SetEnumValue", "ExposureAuto", 2)
        if self.config.gain_db is not None:
            self._set_value("MV_CC_SetEnumValue", "GainAuto", 0)
            self._set_value("MV_CC_SetFloatValue", "Gain", float(self.config.gain_db))
        else:
            self._set_value("MV_CC_SetEnumValue", "GainAuto", 2)
        self._configure_gamma()
        self._configure_white_balance()

    def _lock_white_balance(self) -> None:
        if self._white_balance_auto_mode != "continuous" or not self.config.lock_white_balance_after_warmup:
            return
        self._set_white_balance_auto_mode("off")

    def _start_grabbing(self) -> None:
        ret = self._cam.MV_CC_StartGrabbing()
        if ret != 0:
            raise RuntimeError(f"MVS StartGrabbing failed: 0x{ret:08x}")
        self._started = True

    def _stop_grabbing(self) -> None:
        if not self._started:
            return
        try:
            self._cam.MV_CC_StopGrabbing()
        finally:
            self._started = False

    def _close_device(self) -> None:
        try:
            self._cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            self._cam.MV_CC_DestroyHandle()
        except Exception:
            pass

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        with _MVS_OPEN_LOCK:
            self._open()
            self._start_grabbing()
        self._connected = True
        self._start_read_thread()

        if warmup and self.warmup_s > 0:
            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.async_read(timeout_ms=self.timeout_ms)
                time.sleep(0.05)
            self._lock_white_balance()

        logger.info("%s connected.", self)

    def _postprocess_image(self, image: NDArray[Any]) -> NDArray[Any]:
        processed = image

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed = cv2.rotate(processed, self.rotation)

        if self.width is None or self.height is None:
            self.height, self.width = processed.shape[:2]

        return processed

    def _read_from_hardware(self) -> NDArray[Any]:
        st_out_frame = self._mvs.MV_FRAME_OUT()
        ret = self._cam.MV_CC_GetImageBuffer(st_out_frame, self.timeout_ms)
        if ret != 0:
            if ret == _MVS_TIMEOUT_ERROR:
                raise TimeoutError(f"{self} timed out waiting for frame.")
            raise RuntimeError(f"MVS GetImageBuffer failed: 0x{ret:08x}")

        try:
            frame_info = st_out_frame.stFrameInfo
            payload_size = int(frame_info.nFrameLen)
            frame_width = int(getattr(frame_info, "nWidth", self.capture_width or self.width or 0))
            frame_height = int(getattr(frame_info, "nHeight", self.capture_height or self.height or 0))
            expected_size = frame_width * frame_height * 3
            if payload_size != expected_size:
                raise RuntimeError(
                    f"{self} returned unexpected payload size {payload_size}, expected {expected_size} for RGB8."
                )
            frame_bytes = ctypes.string_at(st_out_frame.pBufAddr, payload_size)
        finally:
            self._cam.MV_CC_FreeImageBuffer(st_out_frame)

        image = np.frombuffer(frame_bytes, dtype=np.uint8).reshape(frame_height, frame_width, 3)
        return self._postprocess_image(image)

    def _read_loop(self) -> None:
        stop_event = self.stop_event
        if stop_event is None:
            raise RuntimeError(f"{self} stop_event is not initialized.")

        failure_count = 0
        while not stop_event.is_set():
            try:
                frame = self._read_from_hardware()
                capture_time = time.perf_counter()
                with self.frame_lock:
                    self.latest_frame = frame
                    self.latest_timestamp = capture_time
                self.new_frame_event.set()
                failure_count = 0
            except TimeoutError:
                continue
            except DeviceNotConnectedError:
                break
            except Exception as exc:
                failure_count += 1
                if failure_count <= 10:
                    logger.warning("Error reading frame in background thread for %s: %s", self, exc)
                    continue
                raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from exc

    def _start_read_thread(self) -> None:
        self._stop_read_thread()
        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, name=f"{self}_read_loop")
        self.thread.daemon = True
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
    def read(self) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        self.new_frame_event.clear()
        return self.async_read(timeout_ms=max(self.timeout_ms, 10_000))

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(f"Timed out waiting for frame from camera {self} after {timeout_ms} ms.")

        with self.frame_lock:
            frame = self.latest_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )

        return frame

    def disconnect(self) -> None:
        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self.thread is not None:
            self._stop_read_thread()

        self._stop_grabbing()
        self._close_device()
        self._cam = self._mvs.MvCamera()
        self._connected = False

        with self.frame_lock:
            self.latest_frame = None
            self.latest_timestamp = None
            self.new_frame_event.clear()

        logger.info("%s disconnected.", self)
