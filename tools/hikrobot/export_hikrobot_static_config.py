#!/usr/bin/env python3

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

# Example:
#   PYTHONPATH=src python tools/hikrobot/export_hikrobot_static_config.py \
#     --serial DA9342471 \
#     --transport-layer gige \
#     --output /tmp/hikrobot_DA9342471_static.yaml
#
#   PYTHONPATH=src python tools/hikrobot/export_hikrobot_static_config.py \
#     --device-index 0 \
#     --transport-layer all

import argparse
from ctypes import POINTER, c_bool, cast
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from lerobot.cameras.hikrobot.camera_hikrobot import (
    _extract_camera_metadata,
    _extract_device_info,
    _load_mvs_sdk,
)

PIXEL_FORMAT_RGB8 = 0x02180014
PIXEL_FORMAT_BGR8 = 0x02180015

AUTO_MODE_LABELS = {
    0: "off",
    1: "once",
    2: "continuous",
}
BALANCE_RATIO_SELECTOR_LABELS = {
    0: "red",
    1: "green",
    2: "blue",
}
PIXEL_FORMAT_LABELS = {
    PIXEL_FORMAT_RGB8: "RGB8Packed",
    PIXEL_FORMAT_BGR8: "BGR8Packed",
}
NODE_SPECS: tuple[tuple[str, str], ...] = (
    ("int", "Width"),
    ("int", "Height"),
    ("int", "OffsetX"),
    ("int", "OffsetY"),
    ("int", "PayloadSize"),
    ("int", "GevSCPSPacketSize"),
    ("float", "AcquisitionFrameRate"),
    ("float", "ResultingFrameRate"),
    ("float", "ExposureTime"),
    ("float", "Gain"),
    ("float", "Gamma"),
    ("bool", "AcquisitionFrameRateEnable"),
    ("bool", "GammaEnable"),
    ("bool", "ReverseX"),
    ("bool", "ReverseY"),
    ("enum", "PixelFormat"),
    ("enum", "ExposureAuto"),
    ("enum", "GainAuto"),
    ("enum", "BalanceWhiteAuto"),
    ("enum", "TriggerMode"),
    ("enum", "TriggerSource"),
    ("enum", "BalanceRatioSelector"),
    ("string", "DeviceUserID"),
    ("string", "DeviceVersion"),
)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the current static configuration of a Hikrobot camera to YAML.")
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--serial", help="Target Hikrobot camera serial number.")
    group.add_argument("--device-index", type=int, help="Target enumerated Hikrobot device index.")
    parser.add_argument(
        "--transport-layer",
        default="all",
        choices=["usb", "gige", "all"],
        help="Transport layer used when enumerating the target camera.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output YAML path. Defaults to outputs/hikrobot/hikrobot_<serial>_static_config.yaml.",
    )
    return parser.parse_args(argv)


def _transport_layer_flag(mvs: Any, transport_layer: str) -> int:
    if transport_layer == "usb":
        return int(getattr(mvs, "MV_USB_DEVICE", 0))
    if transport_layer == "gige":
        return int(getattr(mvs, "MV_GIGE_DEVICE", 0)) | int(getattr(mvs, "MV_GENTL_GIGE_DEVICE", 0))
    return (
        int(getattr(mvs, "MV_USB_DEVICE", 0))
        | int(getattr(mvs, "MV_GIGE_DEVICE", 0))
        | int(getattr(mvs, "MV_GENTL_GIGE_DEVICE", 0))
    )


def _enum_devices(mvs: Any, transport_layer: str) -> list[tuple[int, Any, dict[str, Any]]]:
    device_list = mvs.MV_CC_DEVICE_INFO_LIST()
    ret = mvs.MvCamera.MV_CC_EnumDevices(_transport_layer_flag(mvs, transport_layer), device_list)
    if ret != 0:
        raise RuntimeError(f"MVS EnumDevices failed: 0x{ret:08x}")

    devices = []
    for idx in range(device_list.nDeviceNum):
        entry = device_list.pDeviceInfo[idx]
        if hasattr(entry, "SpecialInfo"):
            device_info = entry
        else:
            device_info = cast(entry, POINTER(mvs.MV_CC_DEVICE_INFO)).contents
        metadata = _extract_camera_metadata(device_info, mvs)
        devices.append((idx, device_info, metadata))
    return devices


def _select_device(
    devices: list[tuple[int, Any, dict[str, Any]]],
    *,
    serial: str | None,
    device_index: int | None,
) -> tuple[int, Any, dict[str, Any]]:
    if not devices:
        raise RuntimeError("No Hikrobot devices found.")

    if serial is not None:
        for idx, device_info, metadata in devices:
            if metadata.get("serial") == serial:
                return idx, device_info, metadata
        raise RuntimeError(f"Hikrobot device with serial {serial!r} not found.")

    if device_index is not None:
        for idx, device_info, metadata in devices:
            if idx == device_index:
                return idx, device_info, metadata
        raise RuntimeError(f"Hikrobot device_index {device_index} not found in the current enumeration result.")

    return devices[0]


def _query_int(cam: Any, mvs: Any, key: str) -> dict[str, Any]:
    int_value = mvs.MVCC_INTVALUE_EX()
    ret = cam.MV_CC_GetIntValueEx(key, int_value)
    if ret != 0:
        raise RuntimeError(f"0x{ret:08x}")
    return {
        "kind": "int",
        "value": int(int_value.nCurValue),
        "min": int(int_value.nMin),
        "max": int(int_value.nMax),
        "inc": int(int_value.nInc),
    }


def _query_float(cam: Any, mvs: Any, key: str) -> dict[str, Any]:
    float_value = mvs.MVCC_FLOATVALUE()
    ret = cam.MV_CC_GetFloatValue(key, float_value)
    if ret != 0:
        raise RuntimeError(f"0x{ret:08x}")
    return {
        "kind": "float",
        "value": float(float_value.fCurValue),
        "min": float(float_value.fMin),
        "max": float(float_value.fMax),
    }


def _query_bool(cam: Any, _mvs: Any, key: str) -> dict[str, Any]:
    bool_value = c_bool(False)
    ret = cam.MV_CC_GetBoolValue(key, bool_value)
    if ret != 0:
        raise RuntimeError(f"0x{ret:08x}")
    return {
        "kind": "bool",
        "value": bool(bool_value.value),
    }


def _query_enum(cam: Any, mvs: Any, key: str) -> dict[str, Any]:
    enum_value = mvs.MVCC_ENUMVALUE()
    ret = cam.MV_CC_GetEnumValue(key, enum_value)
    if ret != 0:
        raise RuntimeError(f"0x{ret:08x}")
    supported_num = int(getattr(enum_value, "nSupportedNum", 0))
    supported_values = [int(enum_value.nSupportValue[idx]) for idx in range(supported_num)]
    labels = _enum_labels_for_key(key)
    result: dict[str, Any] = {
        "kind": "enum",
        "value": int(enum_value.nCurValue),
        "supported_values": supported_values,
    }
    if labels:
        current_label = labels.get(result["value"])
        if current_label is not None:
            result["label"] = current_label
        result["supported_labels"] = {str(value): labels.get(value, f"unknown_{value}") for value in supported_values}
    return result


def _query_string(cam: Any, mvs: Any, key: str) -> dict[str, Any]:
    if not hasattr(mvs, "MVCC_STRINGVALUE"):
        raise RuntimeError("string query type is unavailable in this MVS binding")
    string_value = mvs.MVCC_STRINGVALUE()
    ret = cam.MV_CC_GetStringValue(key, string_value)
    if ret != 0:
        raise RuntimeError(f"0x{ret:08x}")
    raw = bytes(string_value.chCurValue)
    return {
        "kind": "string",
        "value": raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore"),
        "max_length": int(string_value.nMaxLength),
    }


def _query_node(cam: Any, mvs: Any, kind: str, key: str) -> dict[str, Any]:
    try:
        if kind == "int":
            return _query_int(cam, mvs, key)
        if kind == "float":
            return _query_float(cam, mvs, key)
        if kind == "bool":
            return _query_bool(cam, mvs, key)
        if kind == "enum":
            return _query_enum(cam, mvs, key)
        if kind == "string":
            return _query_string(cam, mvs, key)
    except Exception as exc:  # noqa: BLE001
        return {
            "kind": kind,
            "error": str(exc),
        }
    raise ValueError(f"Unsupported node kind: {kind}")


def _enum_labels_for_key(key: str) -> dict[int, str]:
    if key in {"ExposureAuto", "GainAuto", "BalanceWhiteAuto"}:
        return AUTO_MODE_LABELS
    if key == "BalanceRatioSelector":
        return BALANCE_RATIO_SELECTOR_LABELS
    if key == "PixelFormat":
        return PIXEL_FORMAT_LABELS
    if key == "TriggerMode":
        return {0: "off", 1: "on"}
    return {}


def _query_balance_ratios(cam: Any, mvs: Any, nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    selector_state = nodes.get("BalanceRatioSelector")
    previous_selector = selector_state.get("value") if selector_state and "value" in selector_state else None
    string_setter = getattr(cam, "MV_CC_SetEnumValueByString", None)
    enum_setter = getattr(cam, "MV_CC_SetEnumValue", None)

    ratios: dict[str, Any] = {}
    for selector_name in ("red", "green", "blue"):
        try:
            if string_setter is not None:
                ret = string_setter("BalanceRatioSelector", selector_name.capitalize())
                if ret not in (0, None):
                    raise RuntimeError(f"0x{ret:08x}")
            elif enum_setter is not None:
                ret = enum_setter("BalanceRatioSelector", next(key for key, value in BALANCE_RATIO_SELECTOR_LABELS.items() if value == selector_name))
                if ret not in (0, None):
                    raise RuntimeError(f"0x{ret:08x}")
            else:
                raise RuntimeError("camera handle does not expose a balance ratio selector setter")
            ratio_state = _query_int(cam, mvs, "BalanceRatio")
            ratios[selector_name] = ratio_state["value"]
        except Exception as exc:  # noqa: BLE001
            ratios[selector_name] = {"error": str(exc)}

    if previous_selector is not None and enum_setter is not None:
        try:
            enum_setter("BalanceRatioSelector", previous_selector)
        except Exception:
            pass

    return ratios


def build_lerobot_config(device_index: int, metadata: dict[str, Any], nodes: dict[str, dict[str, Any]]) -> dict[str, Any]:
    pixel_format = nodes.get("PixelFormat", {}).get("value")
    color_mode = None
    if pixel_format == PIXEL_FORMAT_BGR8:
        color_mode = "bgr"
    elif pixel_format == PIXEL_FORMAT_RGB8:
        color_mode = "rgb"

    config: dict[str, Any] = {
        "type": "hikrobot",
        "serial": metadata.get("serial") or None,
        "device_index": device_index,
        "transport_layer": metadata.get("transport_layer"),
    }
    if "value" in nodes.get("Width", {}):
        config["width"] = nodes["Width"]["value"]
    if "value" in nodes.get("Height", {}):
        config["height"] = nodes["Height"]["value"]
    if "value" in nodes.get("AcquisitionFrameRate", {}):
        config["fps"] = nodes["AcquisitionFrameRate"]["value"]
    if "value" in nodes.get("ExposureTime", {}):
        config["exposure_us"] = nodes["ExposureTime"]["value"]
    if "value" in nodes.get("Gain", {}):
        config["gain_db"] = nodes["Gain"]["value"]
    if "value" in nodes.get("Gamma", {}):
        config["gamma"] = nodes["Gamma"]["value"]
    if "label" in nodes.get("BalanceWhiteAuto", {}):
        config["white_balance_auto"] = nodes["BalanceWhiteAuto"]["label"]
    if color_mode is not None:
        config["color_mode"] = color_mode
    return config


def export_camera_state(
    *,
    serial: str | None,
    device_index: int | None,
    transport_layer: str,
) -> dict[str, Any]:
    mvs = _load_mvs_sdk()
    initialize = getattr(mvs.MvCamera, "MV_CC_Initialize", None)
    if initialize is not None:
        ret = initialize()
        if ret not in (0, None):
            raise RuntimeError(f"MVS SDK initialization failed: 0x{ret:08x}")

    cam = mvs.MvCamera()
    try:
        devices = _enum_devices(mvs, transport_layer)
        resolved_device_index, device_info, metadata = _select_device(
            devices,
            serial=serial,
            device_index=device_index,
        )
        # Normalize through the same helper used by the runtime camera path.
        metadata = _extract_camera_metadata(_extract_device_info(device_info, mvs), mvs)

        ret = cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            raise RuntimeError(f"MVS CreateHandle failed: 0x{ret:08x}")
        ret = cam.MV_CC_OpenDevice(getattr(mvs, "MV_ACCESS_Exclusive", 1), 0)
        if ret != 0:
            raise RuntimeError(f"MVS OpenDevice failed: 0x{ret:08x}")

        nodes = {key: _query_node(cam, mvs, kind, key) for kind, key in NODE_SPECS}
        white_balance_ratios = _query_balance_ratios(cam, mvs, nodes)

        return {
            "schema_version": 1,
            "queried_at": datetime.now(timezone.utc).isoformat(),
            "selector": {
                "requested_serial": serial,
                "requested_device_index": device_index,
                "transport_layer": transport_layer,
            },
            "device": {
                "device_index": resolved_device_index,
                **metadata,
            },
            "lerobot_camera_config": build_lerobot_config(resolved_device_index, metadata, nodes),
            "white_balance_ratios": white_balance_ratios,
            "nodes": nodes,
        }
    finally:
        try:
            cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            cam.MV_CC_DestroyHandle()
        except Exception:
            pass
        finalize = getattr(mvs.MvCamera, "MV_CC_Finalize", None)
        if finalize is not None:
            try:
                finalize()
            except Exception:
                pass


def build_default_output_path(serial: str | None, device_index: int | None) -> Path:
    identifier = serial or f"index_{device_index if device_index is not None else 0}"
    return Path("outputs/hikrobot") / f"hikrobot_{identifier}_static_config.yaml"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    payload = export_camera_state(
        serial=args.serial,
        device_index=args.device_index,
        transport_layer=args.transport_layer,
    )
    output_path = args.output or build_default_output_path(
        payload["device"].get("serial"),
        payload["device"].get("device_index"),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=True), encoding="utf-8")
    print(output_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
