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

import ctypes

import numpy as np

from lerobot.cameras.hikrobot import HikrobotCamera, HikrobotCameraConfig
from lerobot.cameras.hikrobot.configuration_hikrobot import (
    HIKROBOT_DEFAULT_COLOR_MODE,
    HIKROBOT_DEFAULT_EXPOSURE_US,
    HIKROBOT_DEFAULT_GAIN_DB,
    HIKROBOT_DEFAULT_GAMMA,
    HIKROBOT_DEFAULT_LOCK_WHITE_BALANCE_AFTER_WARMUP,
    HIKROBOT_DEFAULT_WARMUP_S,
)


class _FakeIntValue:
    def __init__(self) -> None:
        self.nCurValue = 0


class _FakeFloatValue:
    def __init__(self) -> None:
        self.fCurValue = 0.0
        self.fMin = 0.0
        self.fMax = 24.0


class _FakeEnumValue:
    def __init__(self) -> None:
        self.nCurValue = 0
        self.nSupportedNum = 0
        self.nSupportValue = [0] * 64


class _FakeFrameInfo:
    def __init__(self) -> None:
        self.nFrameLen = 0
        self.nWidth = 0
        self.nHeight = 0
        self.nFrameNum = 0


class _FakeFrameOut:
    def __init__(self) -> None:
        self.stFrameInfo = _FakeFrameInfo()
        self.pBufAddr = None


class _FakeUsb3Info:
    def __init__(self, serial: str, model: str = "MV-CE060") -> None:
        self.chSerialNumber = serial.encode("utf-8") + b"\x00"
        self.chModelName = model.encode("utf-8") + b"\x00"
        self.chVendorName = b"HIKROBOT\x00"
        self.chDeviceVersion = b"1.0\x00"
        self.chUserDefinedName = b"test\x00"


class _FakeSpecialInfo:
    def __init__(self, serial: str) -> None:
        self.stUsb3VInfo = _FakeUsb3Info(serial)


class _FakeDeviceInfo:
    def __init__(self, serial: str) -> None:
        self.nTLayerType = 1
        self.SpecialInfo = _FakeSpecialInfo(serial)


class _FakeGigEInfo:
    def __init__(self, serial: str, ip: str, model: str = "MV-CE060-10GC", net_export: str = "192.168.1.10") -> None:
        self.chSerialNumber = serial.encode("utf-8") + b"\x00"
        self.chModelName = model.encode("utf-8") + b"\x00"
        self.chManufacturerName = b"HIKROBOT\x00"
        octets = [int(part) for part in ip.split(".")]
        self.nCurrentIp = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]
        export_octets = [int(part) for part in net_export.split(".")]
        self.nNetExport = (
            (export_octets[0] << 24)
            | (export_octets[1] << 16)
            | (export_octets[2] << 8)
            | export_octets[3]
        )


class _FakeGigESpecialInfo:
    def __init__(self, serial: str, ip: str, net_export: str = "192.168.1.10") -> None:
        self.stGigEInfo = _FakeGigEInfo(serial, ip, net_export=net_export)


class _FakeGigEDeviceInfo:
    def __init__(self, serial: str, ip: str, net_export: str = "192.168.1.10", tlayer_type: int = 2) -> None:
        self.nTLayerType = tlayer_type
        self.SpecialInfo = _FakeGigESpecialInfo(serial, ip, net_export=net_export)


class _FakeDeviceInfoList:
    def __init__(self) -> None:
        self.nDeviceNum = 0
        self.pDeviceInfo: list[_FakeDeviceInfo] = []


class _FakeMvCamera:
    _devices = [_FakeDeviceInfo("LEFT123"), _FakeDeviceInfo("RIGHT456")]
    _pixel_format_rgb8 = 0x02180014
    _pixel_format_bgr8 = 0x02180015

    @staticmethod
    def MV_CC_EnumDevices(_transport_flag, device_list) -> int:
        device_list.nDeviceNum = len(_FakeMvCamera._devices)
        device_list.pDeviceInfo = list(_FakeMvCamera._devices)
        return 0

    def __init__(self) -> None:
        self._frame_num = 0
        self._buffer = None
        self.started = False
        self.pixel_format = self._pixel_format_bgr8
        self.balance_white_auto = 2
        self.balance_ratio_selector = "red"
        self.balance_ratios = {"red": 900, "green": 900, "blue": 900}
        self.calls: list[tuple[str, object | None, object | None]] = []

    def MV_CC_CreateHandle(self, _device_info) -> int:
        self.calls.append(("create_handle", None, None))
        return 0

    def MV_CC_OpenDevice(self, *_args) -> int:
        self.calls.append(("open_device", None, None))
        return 0

    def MV_CC_GetEnumValue(self, key, enum_value) -> int:
        if key != "PixelFormat":
            return -1
        enum_value.nCurValue = self.pixel_format
        enum_value.nSupportedNum = 2
        enum_value.nSupportValue[0] = self._pixel_format_rgb8
        enum_value.nSupportValue[1] = self._pixel_format_bgr8
        return 0

    def MV_CC_SetEnumValue(self, key, value) -> int:
        self.calls.append(("set_enum", key, value))
        if key == "PixelFormat":
            self.pixel_format = value
        if key == "BalanceWhiteAuto":
            self.balance_white_auto = value
        if key == "BalanceRatioSelector":
            selector_map = {0: "red", 1: "green", 2: "blue"}
            self.balance_ratio_selector = selector_map[value]
        return 0

    def MV_CC_SetEnumValueByString(self, key, value) -> int:
        self.calls.append(("set_enum_str", key, value))
        if key == "BalanceRatioSelector":
            self.balance_ratio_selector = value.lower()
        return 0

    def MV_CC_SetBoolValue(self, key, value) -> int:
        self.calls.append(("set_bool", key, value))
        return 0

    def MV_CC_SetFloatValue(self, key, value) -> int:
        self.calls.append(("set_float", key, value))
        return 0

    def MV_CC_SetIntValue(self, key, value) -> int:
        self.calls.append(("set_int", key, value))
        return 0

    def MV_CC_SetIntValueEx(self, key, value) -> int:
        self.calls.append(("set_int_ex", key, value))
        if key == "BalanceRatio":
            self.balance_ratios[self.balance_ratio_selector] = value
        return 0

    def MV_CC_GetFloatValue(self, _key, float_value) -> int:
        float_value.fCurValue = 12.0
        float_value.fMin = 0.0
        float_value.fMax = 24.0
        return 0

    def MV_CC_GetIntValueEx(self, key, int_value) -> int:
        if key == "BalanceRatio":
            value = self.balance_ratios[self.balance_ratio_selector]
            int_value.nCurValue = value
            int_value.nMin = 1
            int_value.nMax = 16376
            return 0
        return -1

    def MV_CC_StartGrabbing(self) -> int:
        self.calls.append(("start_grabbing", None, None))
        self.started = True
        return 0

    def MV_CC_StopGrabbing(self) -> int:
        self.started = False
        return 0

    def MV_CC_CloseDevice(self) -> int:
        return 0

    def MV_CC_DestroyHandle(self) -> int:
        return 0

    def MV_CC_FreeImageBuffer(self, *_args) -> int:
        return 0

    def MV_CC_GetImageBuffer(self, frame_out, _timeout_ms) -> int:
        image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
        self._buffer = (ctypes.c_ubyte * image.size)(*image.flatten())
        frame_out.stFrameInfo.nFrameLen = image.size
        frame_out.stFrameInfo.nWidth = 2
        frame_out.stFrameInfo.nHeight = 2
        frame_out.stFrameInfo.nFrameNum = self._frame_num
        self._frame_num += 1
        frame_out.pBufAddr = ctypes.addressof(self._buffer)
        return 0


class _FakeMVS:
    MV_USB_DEVICE = 1
    MV_GIGE_DEVICE = 2
    MV_GENTL_GIGE_DEVICE = 4
    MV_ACCESS_Exclusive = 1
    MV_CC_DEVICE_INFO_LIST = _FakeDeviceInfoList
    MV_CC_DEVICE_INFO = _FakeDeviceInfo
    MV_FRAME_OUT = _FakeFrameOut
    MVCC_INTVALUE_EX = _FakeIntValue
    MVCC_FLOATVALUE = _FakeFloatValue
    MVCC_ENUMVALUE = _FakeEnumValue
    MvCamera = _FakeMvCamera


def test_find_cameras(monkeypatch):
    monkeypatch.setattr("lerobot.cameras.hikrobot.camera_hikrobot._load_mvs_sdk", lambda: _FakeMVS)
    cameras = HikrobotCamera.find_cameras()
    assert [camera["id"] for camera in cameras] == ["LEFT123", "RIGHT456"]
    assert cameras[0]["serial"] == "LEFT123"
    assert cameras[0]["transport_layer"] == "usb"


def test_find_cameras_reports_gige_serial_and_current_ip(monkeypatch):
    class _FakeMvCameraGigE(_FakeMvCamera):
        _devices = [_FakeGigEDeviceInfo("GIGE123", "192.168.1.50", net_export="192.168.1.10", tlayer_type=4)]
        last_transport_flag = None

        @staticmethod
        def MV_CC_EnumDevices(transport_flag, device_list) -> int:
            _FakeMvCameraGigE.last_transport_flag = transport_flag
            device_list.nDeviceNum = len(_FakeMvCameraGigE._devices)
            device_list.pDeviceInfo = list(_FakeMvCameraGigE._devices)
            return 0

    class _FakeMVSGigE(_FakeMVS):
        MvCamera = _FakeMvCameraGigE

    monkeypatch.setattr("lerobot.cameras.hikrobot.camera_hikrobot._load_mvs_sdk", lambda: _FakeMVSGigE)
    cameras = HikrobotCamera.find_cameras()
    assert _FakeMvCameraGigE.last_transport_flag == (_FakeMVSGigE.MV_USB_DEVICE | _FakeMVSGigE.MV_GIGE_DEVICE | _FakeMVSGigE.MV_GENTL_GIGE_DEVICE)

    assert cameras == [
        {
            "name": "Hikrobot Camera @ GIGE123",
            "type": "Hikrobot",
            "id": "GIGE123",
            "device_index": 0,
            "serial": "GIGE123",
            "manufacturer": "HIKROBOT",
            "model": "MV-CE060-10GC",
            "transport_layer": "gige",
            "current_ip": "192.168.1.50",
            "net_export": "192.168.1.10",
        }
    ]


def test_default_config_matches_hikrobot_recording_profile():
    config = HikrobotCameraConfig(serial="LEFT123", width=2, height=2, fps=30)

    assert config.color_mode == HIKROBOT_DEFAULT_COLOR_MODE
    assert config.warmup_s == HIKROBOT_DEFAULT_WARMUP_S
    assert config.exposure_us == HIKROBOT_DEFAULT_EXPOSURE_US
    assert config.gain_db == HIKROBOT_DEFAULT_GAIN_DB
    assert config.gamma == HIKROBOT_DEFAULT_GAMMA
    assert config.white_balance_auto == "continuous"
    assert config.lock_white_balance_after_warmup is HIKROBOT_DEFAULT_LOCK_WHITE_BALANCE_AFTER_WARMUP


def test_connect_read_disconnect():
    config = HikrobotCameraConfig(serial="LEFT123", width=2, height=2, fps=30, warmup_s=0, timeout_ms=50)
    camera = HikrobotCamera(config, mvs_module=_FakeMVS)

    camera.connect(warmup=False)
    assert camera.is_connected

    frame = camera.read()
    assert frame.shape == (2, 2, 3)
    assert frame.dtype == np.uint8

    latest = camera.read_latest(max_age_ms=500)
    assert latest.shape == (2, 2, 3)

    camera.disconnect()
    assert not camera.is_connected


def test_connect_gige_camera_by_serial():
    class _FakeMvCameraGigE(_FakeMvCamera):
        _devices = [_FakeGigEDeviceInfo("GIGE123", "192.168.1.50"), _FakeGigEDeviceInfo("GIGE456", "192.168.1.51")]

        @staticmethod
        def MV_CC_EnumDevices(_transport_flag, device_list) -> int:
            device_list.nDeviceNum = len(_FakeMvCameraGigE._devices)
            device_list.pDeviceInfo = list(_FakeMvCameraGigE._devices)
            return 0

    class _FakeMVSGigE(_FakeMVS):
        MvCamera = _FakeMvCameraGigE

    camera = HikrobotCamera(
        HikrobotCameraConfig(serial="GIGE456", width=2, height=2, fps=30, warmup_s=0, transport_layer="gige"),
        mvs_module=_FakeMVSGigE,
    )

    camera.connect(warmup=False)
    assert camera.is_connected

    camera.disconnect()
    assert not camera.is_connected


def test_connect_requires_device_rgb888_support():
    class _FakeMvCameraNoRgb(_FakeMvCamera):
        def MV_CC_GetEnumValue(self, key, enum_value) -> int:
            if key != "PixelFormat":
                return -1
            enum_value.nCurValue = self.pixel_format
            enum_value.nSupportedNum = 1
            enum_value.nSupportValue[0] = self._pixel_format_bgr8
            return 0

    class _FakeMVSNoRgb(_FakeMVS):
        MvCamera = _FakeMvCameraNoRgb

    camera = HikrobotCamera(
        HikrobotCameraConfig(serial="LEFT123", width=2, height=2, warmup_s=0, color_mode="rgb"),
        mvs_module=_FakeMVSNoRgb,
    )

    try:
        camera.connect(warmup=False)
    except RuntimeError as exc:
        assert "does not support device-side RGB888 output" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when RGB888 is not supported")


def test_connect_requires_device_bgr888_support_for_bgr_mode():
    class _FakeMvCameraNoBgr(_FakeMvCamera):
        def MV_CC_GetEnumValue(self, key, enum_value) -> int:
            if key != "PixelFormat":
                return -1
            enum_value.nCurValue = self.pixel_format
            enum_value.nSupportedNum = 1
            enum_value.nSupportValue[0] = self._pixel_format_rgb8
            return 0

    class _FakeMVSNoBgr(_FakeMVS):
        MvCamera = _FakeMvCameraNoBgr

    camera = HikrobotCamera(
        HikrobotCameraConfig(serial="LEFT123", width=2, height=2, warmup_s=0, color_mode="bgr"),
        mvs_module=_FakeMVSNoBgr,
    )

    try:
        camera.connect(warmup=False)
    except RuntimeError as exc:
        assert "does not support device-side BGR888 output" in str(exc)
    else:
        raise AssertionError("Expected RuntimeError when BGR888 is not supported")


def test_gamma_is_configured_before_stream_starts():
    config = HikrobotCameraConfig(serial="LEFT123", width=2, height=2, warmup_s=0, gamma=1.3)
    camera = HikrobotCamera(config, mvs_module=_FakeMVS)

    camera.connect(warmup=False)
    calls = camera._cam.calls
    gamma_call_index = calls.index(("set_float", "Gamma", 1.3))
    start_call_index = calls.index(("start_grabbing", None, None))
    assert gamma_call_index < start_call_index

    camera.disconnect()


def test_manual_white_balance_is_configured_before_stream_starts():
    config = HikrobotCameraConfig(
        serial="LEFT123",
        width=2,
        height=2,
        warmup_s=0,
        white_balance_auto="off",
        white_balance_red=1100,
        white_balance_blue=1400,
    )
    camera = HikrobotCamera(config, mvs_module=_FakeMVS)

    camera.connect(warmup=False)
    calls = camera._cam.calls
    red_call_index = calls.index(("set_int_ex", "BalanceRatio", 1100))
    blue_call_index = calls.index(("set_int_ex", "BalanceRatio", 1400))
    start_call_index = calls.index(("start_grabbing", None, None))
    assert red_call_index < start_call_index
    assert blue_call_index < start_call_index
    assert camera._cam.balance_ratios["red"] == 1100
    assert camera._cam.balance_ratios["blue"] == 1400

    camera.disconnect()


def test_bgr_mode_keeps_device_bgr_channel_order():
    config = HikrobotCameraConfig(serial="LEFT123", width=2, height=2, warmup_s=0, color_mode="bgr")
    camera = HikrobotCamera(config, mvs_module=_FakeMVS)

    camera.connect(warmup=False)
    frame = camera.read()

    assert camera._cam.pixel_format == camera._cam._pixel_format_bgr8
    np.testing.assert_array_equal(frame, np.arange(12, dtype=np.uint8).reshape(2, 2, 3))

    camera.disconnect()


def test_get_white_balance_ratios_restores_continuous_mode_when_it_is_still_active():
    config = HikrobotCameraConfig(serial="LEFT123", width=2, height=2, warmup_s=0, white_balance_auto="continuous")
    camera = HikrobotCamera(config, mvs_module=_FakeMVS)

    camera.connect(warmup=False)
    ratios = camera.get_white_balance_ratios()

    assert ratios == {"red": 900, "green": 900, "blue": 900}
    assert camera._cam.balance_white_auto == 2

    camera.disconnect()


def test_get_white_balance_ratios_does_not_restart_awb_after_locking():
    config = HikrobotCameraConfig(
        serial="LEFT123",
        width=2,
        height=2,
        warmup_s=0,
        white_balance_auto="continuous",
        lock_white_balance_after_warmup=True,
    )
    camera = HikrobotCamera(config, mvs_module=_FakeMVS)

    camera.connect(warmup=False)
    camera._lock_white_balance()
    set_enum_count_before = len([call for call in camera._cam.calls if call[:2] == ("set_enum", "BalanceWhiteAuto")])

    ratios = camera.get_white_balance_ratios()

    assert ratios == {"red": 900, "green": 900, "blue": 900}
    assert camera._cam.balance_white_auto == 0
    assert len([call for call in camera._cam.calls if call[:2] == ("set_enum", "BalanceWhiteAuto")]) == set_enum_count_before

    camera.disconnect()
