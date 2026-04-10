#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import yaml

from tools.hikrobot import export_hikrobot_static_config as export_tool


def _encode_ipv4(ip: str) -> int:
    octets = [int(part) for part in ip.split(".")]
    return (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]


class _FakeGigEInfo:
    def __init__(self, serial: str, ip: str, net_export: str = "192.168.1.10") -> None:
        self.chSerialNumber = serial.encode("utf-8") + b"\x00"
        self.chModelName = b"MV-CE060-10GC\x00"
        self.chManufacturerName = b"HIKROBOT\x00"
        self.nCurrentIp = _encode_ipv4(ip)
        self.nNetExport = _encode_ipv4(net_export)


class _FakeGigESpecialInfo:
    def __init__(self, serial: str, ip: str, net_export: str = "192.168.1.10") -> None:
        self.stGigEInfo = _FakeGigEInfo(serial, ip, net_export)


class _FakeGigEDeviceInfo:
    def __init__(self, serial: str, ip: str, net_export: str = "192.168.1.10") -> None:
        self.nTLayerType = 4
        self.SpecialInfo = _FakeGigESpecialInfo(serial, ip, net_export)


class _FakeDeviceInfoList:
    def __init__(self) -> None:
        self.nDeviceNum = 0
        self.pDeviceInfo: list[_FakeGigEDeviceInfo] = []


class _FakeIntValue:
    def __init__(self) -> None:
        self.nCurValue = 0
        self.nMax = 0
        self.nMin = 0
        self.nInc = 0


class _FakeFloatValue:
    def __init__(self) -> None:
        self.fCurValue = 0.0
        self.fMax = 0.0
        self.fMin = 0.0


class _FakeEnumValue:
    def __init__(self) -> None:
        self.nCurValue = 0
        self.nSupportedNum = 0
        self.nSupportValue = [0] * 64


class _FakeStringValue:
    def __init__(self) -> None:
        self.chCurValue = b""
        self.nMaxLength = 256


class _FakeMvCamera:
    _devices = [_FakeGigEDeviceInfo("GIGE123", "192.168.1.50")]

    @staticmethod
    def MV_CC_Initialize() -> int:
        return 0

    @staticmethod
    def MV_CC_Finalize() -> int:
        return 0

    @staticmethod
    def MV_CC_EnumDevices(_transport_flag, device_list) -> int:
        device_list.nDeviceNum = len(_FakeMvCamera._devices)
        device_list.pDeviceInfo = list(_FakeMvCamera._devices)
        return 0

    def __init__(self) -> None:
        self.balance_ratio_selector = 0
        self.balance_ratios = {
            0: 1100,
            1: 1000,
            2: 1200,
        }

    def MV_CC_CreateHandle(self, _device_info) -> int:
        return 0

    def MV_CC_OpenDevice(self, *_args) -> int:
        return 0

    def MV_CC_CloseDevice(self) -> int:
        return 0

    def MV_CC_DestroyHandle(self) -> int:
        return 0

    def MV_CC_GetIntValueEx(self, key, int_value) -> int:
        values = {
            "Width": (1920, 64, 2448, 8),
            "Height": (1080, 64, 2048, 2),
            "OffsetX": (0, 0, 100, 8),
            "OffsetY": (0, 0, 100, 2),
            "PayloadSize": (6220800, 0, 6220800, 1),
            "GevSCPSPacketSize": (1500, 576, 9000, 4),
        }
        if key == "BalanceRatio":
            value = self.balance_ratios[self.balance_ratio_selector]
            int_value.nCurValue = value
            int_value.nMin = 1
            int_value.nMax = 16376
            int_value.nInc = 1
            return 0
        if key not in values:
            return 1
        value, min_value, max_value, inc = values[key]
        int_value.nCurValue = value
        int_value.nMin = min_value
        int_value.nMax = max_value
        int_value.nInc = inc
        return 0

    def MV_CC_GetFloatValue(self, key, float_value) -> int:
        values = {
            "AcquisitionFrameRate": (30.0, 1.0, 120.0),
            "ResultingFrameRate": (29.97, 1.0, 120.0),
            "ExposureTime": (8000.0, 20.0, 2000000.0),
            "Gain": (9.5, 0.0, 24.0),
            "Gamma": (1.2, 0.1, 4.0),
        }
        if key not in values:
            return 1
        value, min_value, max_value = values[key]
        float_value.fCurValue = value
        float_value.fMin = min_value
        float_value.fMax = max_value
        return 0

    def MV_CC_GetBoolValue(self, key, bool_value) -> int:
        values = {
            "AcquisitionFrameRateEnable": True,
            "GammaEnable": True,
            "ReverseX": False,
            "ReverseY": False,
        }
        if key not in values:
            return 1
        bool_value.value = values[key]
        return 0

    def MV_CC_GetEnumValue(self, key, enum_value) -> int:
        values = {
            "PixelFormat": (export_tool.PIXEL_FORMAT_BGR8, [export_tool.PIXEL_FORMAT_RGB8, export_tool.PIXEL_FORMAT_BGR8]),
            "ExposureAuto": (0, [0, 1, 2]),
            "GainAuto": (2, [0, 1, 2]),
            "BalanceWhiteAuto": (2, [0, 1, 2]),
            "TriggerMode": (0, [0, 1]),
            "TriggerSource": (7, [7]),
            "BalanceRatioSelector": (self.balance_ratio_selector, [0, 1, 2]),
        }
        if key not in values:
            return 1
        current_value, supported_values = values[key]
        enum_value.nCurValue = current_value
        enum_value.nSupportedNum = len(supported_values)
        for index, value in enumerate(supported_values):
            enum_value.nSupportValue[index] = value
        return 0

    def MV_CC_GetStringValue(self, key, string_value) -> int:
        values = {
            "DeviceUserID": "front_left",
            "DeviceVersion": "1.2.3",
        }
        if key not in values:
            return 1
        string_value.chCurValue = values[key].encode("utf-8") + b"\x00"
        string_value.nMaxLength = 256
        return 0

    def MV_CC_SetEnumValueByString(self, key, value) -> int:
        if key != "BalanceRatioSelector":
            return 1
        selector_map = {
            "Red": 0,
            "Green": 1,
            "Blue": 2,
        }
        self.balance_ratio_selector = selector_map[value]
        return 0

    def MV_CC_SetEnumValue(self, key, value) -> int:
        if key != "BalanceRatioSelector":
            return 1
        self.balance_ratio_selector = value
        return 0


class _FakeMVS:
    MV_USB_DEVICE = 1
    MV_GIGE_DEVICE = 2
    MV_GENTL_GIGE_DEVICE = 4
    MV_ACCESS_Exclusive = 1
    MV_CC_DEVICE_INFO_LIST = _FakeDeviceInfoList
    MV_CC_DEVICE_INFO = _FakeGigEDeviceInfo
    MVCC_INTVALUE_EX = _FakeIntValue
    MVCC_FLOATVALUE = _FakeFloatValue
    MVCC_ENUMVALUE = _FakeEnumValue
    MVCC_STRINGVALUE = _FakeStringValue
    MvCamera = _FakeMvCamera


def test_export_camera_state_builds_expected_payload(monkeypatch):
    monkeypatch.setattr(export_tool, "_load_mvs_sdk", lambda: _FakeMVS)

    payload = export_tool.export_camera_state(serial="GIGE123", device_index=None, transport_layer="gige")

    assert payload["device"]["serial"] == "GIGE123"
    assert payload["device"]["transport_layer"] == "gige"
    assert payload["device"]["current_ip"] == "192.168.1.50"
    assert payload["lerobot_camera_config"] == {
        "type": "hikrobot",
        "serial": "GIGE123",
        "device_index": 0,
        "transport_layer": "gige",
        "width": 1920,
        "height": 1080,
        "fps": 30.0,
        "exposure_us": 8000.0,
        "gain_db": 9.5,
        "gamma": 1.2,
        "white_balance_auto": "continuous",
        "color_mode": "bgr",
    }
    assert payload["white_balance_ratios"] == {
        "red": 1100,
        "green": 1000,
        "blue": 1200,
    }
    assert payload["nodes"]["PixelFormat"]["label"] == "BGR8Packed"
    assert payload["nodes"]["DeviceUserID"]["value"] == "front_left"


def test_main_writes_yaml_to_requested_output(monkeypatch, tmp_path: Path):
    monkeypatch.setattr(export_tool, "_load_mvs_sdk", lambda: _FakeMVS)
    output_path = tmp_path / "hikrobot.yaml"

    exit_code = export_tool.main(
        [
            "--serial",
            "GIGE123",
            "--transport-layer",
            "gige",
            "--output",
            str(output_path),
        ]
    )

    assert exit_code == 0
    payload = yaml.safe_load(output_path.read_text(encoding="utf-8"))
    assert payload["selector"]["requested_serial"] == "GIGE123"
    assert payload["device"]["serial"] == "GIGE123"
    assert payload["lerobot_camera_config"]["color_mode"] == "bgr"
