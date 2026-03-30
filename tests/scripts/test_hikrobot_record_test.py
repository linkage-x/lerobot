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

import json
from pathlib import Path

from tools.hikrobot import hikrobot_record_test


class _FakeGigEInfo:
    def __init__(self, serial: str, ip: str) -> None:
        self.chSerialNumber = serial.encode("utf-8") + b"\x00"
        self.chModelName = b"MV-CE060-10GC\x00"
        self.chManufacturerName = b"HIKROBOT\x00"
        octets = [int(part) for part in ip.split(".")]
        self.nCurrentIp = (octets[0] << 24) | (octets[1] << 16) | (octets[2] << 8) | octets[3]


class _FakeGigESpecialInfo:
    def __init__(self, serial: str, ip: str) -> None:
        self.stGigEInfo = _FakeGigEInfo(serial, ip)


class _FakeGigEDeviceInfo:
    def __init__(self, serial: str, ip: str) -> None:
        self.nTLayerType = 2
        self.SpecialInfo = _FakeGigESpecialInfo(serial, ip)


class _FakeFloatValue:
    def __init__(self) -> None:
        self.fMin = 0.0
        self.fMax = 24.0


class _FakeDeviceInfoList:
    def __init__(self) -> None:
        self.nDeviceNum = 0
        self.pDeviceInfo: list[_FakeGigEDeviceInfo] = []


class _FakeMvCamera:
    _devices = [_FakeGigEDeviceInfo("GIGE123", "192.168.1.50")]

    @staticmethod
    def MV_CC_EnumDevices(_transport_flag, device_list) -> int:
        device_list.nDeviceNum = len(_FakeMvCamera._devices)
        device_list.pDeviceInfo = list(_FakeMvCamera._devices)
        return 0

    def MV_CC_CreateHandle(self, _device_info) -> int:
        return 0

    def MV_CC_OpenDevice(self, *_args) -> int:
        return 0

    def MV_CC_GetFloatValue(self, _key, float_value) -> int:
        float_value.fMin = 0.0
        float_value.fMax = 24.0
        return 0

    def MV_CC_CloseDevice(self) -> int:
        return 0

    def MV_CC_DestroyHandle(self) -> int:
        return 0


class _FakeMVS:
    MV_USB_DEVICE = 1
    MV_GIGE_DEVICE = 2
    MV_ACCESS_Exclusive = 1
    MV_CC_DEVICE_INFO_LIST = _FakeDeviceInfoList
    MVCC_FLOATVALUE = _FakeFloatValue
    MvCamera = _FakeMvCamera


class _FakeConnectedCamera:
    def __init__(self, _config) -> None:
        self._mvs = _FakeMVS
        self._cam = _FakeMvCamera()

    def connect(self, warmup: bool = True) -> None:
        del warmup

    def get_white_balance_ratios(self) -> dict[str, int]:
        return {"red": 900, "green": 900, "blue": 900}

    def disconnect(self) -> None:
        return None


class _FakeVideoWriter:
    def __init__(self, *_args, **_kwargs) -> None:
        self.frames = []

    def isOpened(self) -> bool:
        return True

    def write(self, frame) -> None:
        self.frames.append(frame)

    def release(self) -> None:
        return None


def test_parse_args_accepts_transport_layer():
    args = hikrobot_record_test.parse_args(["--transport-layer", "gige", "--serial", "GIGE123"])

    assert args.transport_layer == "gige"
    assert args.serial == "GIGE123"


def test_resolve_gain_db_uses_gige_metadata(monkeypatch):
    monkeypatch.setattr(hikrobot_record_test, "_load_mvs_sdk", lambda: _FakeMVS)

    resolved_gain_db, max_gain_db, metadata = hikrobot_record_test._resolve_gain_db(
        "GIGE123",
        "gige",
        "manual",
        12.0,
    )

    assert resolved_gain_db == 12.0
    assert max_gain_db == 24.0
    assert metadata["transport_layer"] == "gige"
    assert metadata["current_ip"] == "192.168.1.50"


def test_main_writes_transport_layer_and_current_ip_to_metadata(tmp_path: Path, monkeypatch):
    video_path = tmp_path / "record.mp4"
    metadata_path = tmp_path / "record.json"

    monkeypatch.setattr(hikrobot_record_test, "build_output_paths", lambda _args: (video_path, metadata_path))
    monkeypatch.setattr(
        hikrobot_record_test,
        "_resolve_gain_db",
        lambda *_args: (12.0, 24.0, {"transport_layer": "gige", "current_ip": "192.168.1.50"}),
    )
    monkeypatch.setattr(hikrobot_record_test, "_get_camera_float", lambda *_args: (12.0, 0.0, 24.0))
    monkeypatch.setattr(hikrobot_record_test, "HikrobotCamera", _FakeConnectedCamera)
    monkeypatch.setattr(hikrobot_record_test.cv2, "VideoWriter", _FakeVideoWriter)

    exit_code = hikrobot_record_test.main(
        [
            "--serial",
            "GIGE123",
            "--transport-layer",
            "gige",
            "--duration-s",
            "0",
            "--output-dir",
            str(tmp_path),
        ]
    )

    assert exit_code == 0

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert metadata["serial"] == "GIGE123"
    assert metadata["transport_layer"] == "gige"
    assert metadata["current_ip"] == "192.168.1.50"

