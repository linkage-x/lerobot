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

import argparse
import fcntl
import json
import socket
import struct
import subprocess
import sys
from ctypes import POINTER, cast
from typing import Any

_MVS_PYTHON_PATHS = [
    "/opt/MVS/Samples/64/Python",
    "/opt/MVS/Samples/32/Python",
]
_SIOCGIFADDR = 0x8915


def _load_mvs_sdk() -> Any:
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
            "Hikrobot MVS SDK Python bindings not found under /opt/MVS/Samples/64/Python or /opt/MVS/Samples/32/Python."
        ) from exc


def _decode_char_buffer(field: Any) -> str:
    raw = memoryview(field).tobytes()
    return raw.split(b"\x00", 1)[0].decode("utf-8", errors="ignore")


def _decode_ipv4_address(raw_ip: Any) -> str:
    value = int(raw_ip)
    return ".".join(
        str((value & mask) >> shift)
        for mask, shift in (
            (0xFF000000, 24),
            (0x00FF0000, 16),
            (0x0000FF00, 8),
            (0x000000FF, 0),
        )
    )


def _list_local_ipv4_interfaces() -> dict[str, str]:
    interfaces: dict[str, str] = {}
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    except OSError:
        return _list_local_ipv4_interfaces_via_ip()
    try:
        for _, ifname in socket.if_nameindex():
            ifreq = struct.pack("256s", ifname.encode("utf-8"))
            try:
                response = fcntl.ioctl(sock.fileno(), _SIOCGIFADDR, ifreq)
            except OSError:
                continue
            ipv4 = socket.inet_ntoa(response[20:24])
            interfaces[ipv4] = ifname
    finally:
        sock.close()
    if interfaces:
        return interfaces
    return _list_local_ipv4_interfaces_via_ip()


def _list_local_ipv4_interfaces_via_ip() -> dict[str, str]:
    interfaces: dict[str, str] = {}
    try:
        output = subprocess.check_output(
            ["ip", "-o", "-4", "addr", "show"],
            text=True,
            stderr=subprocess.DEVNULL,
        )
    except (OSError, subprocess.CalledProcessError):
        return interfaces

    for line in output.splitlines():
        parts = line.split()
        if len(parts) < 4:
            continue
        ifname = parts[1]
        cidr = parts[3]
        ipv4 = cidr.split("/", 1)[0]
        interfaces[ipv4] = ifname
    return interfaces


def _device_to_dict(device_info: Any, mvs: Any) -> dict[str, Any] | None:
    gige_types = {int(getattr(mvs, "MV_GIGE_DEVICE", 0))}
    gentl_gige_type = getattr(mvs, "MV_GENTL_GIGE_DEVICE", None)
    if gentl_gige_type is not None:
        gige_types.add(int(gentl_gige_type))

    if int(getattr(device_info, "nTLayerType", 0)) not in gige_types:
        return None

    gige_info = device_info.SpecialInfo.stGigEInfo
    return {
        "transport_layer": "gige",
        "manufacturer": _decode_char_buffer(gige_info.chManufacturerName),
        "model": _decode_char_buffer(gige_info.chModelName),
        "serial": _decode_char_buffer(gige_info.chSerialNumber),
        "user_defined_name": _decode_char_buffer(gige_info.chUserDefinedName),
        "current_ip": _decode_ipv4_address(gige_info.nCurrentIp),
        "net_export": _decode_ipv4_address(gige_info.nNetExport),
    }


def _resolve_net_export_filter(args: argparse.Namespace) -> str | None:
    if args.net_export and args.interface:
        raise ValueError("Use either --net-export or --interface, not both.")

    if args.net_export:
        return args.net_export

    if args.interface is None:
        return None

    local_interfaces = _list_local_ipv4_interfaces()
    for ipv4, ifname in local_interfaces.items():
        if ifname == args.interface:
            return ipv4

    raise ValueError(f"Interface {args.interface!r} does not have an IPv4 address in the current environment.")


def list_gige_cameras(net_export_filter: str | None = None) -> tuple[list[dict[str, Any]], dict[str, str]]:
    mvs = _load_mvs_sdk()
    local_interfaces = _list_local_ipv4_interfaces()

    init_ret = mvs.MvCamera.MV_CC_Initialize()
    if init_ret not in (0, None):
        raise RuntimeError(f"MVS SDK initialization failed: 0x{init_ret:08x}")

    try:
        transport_flag = int(getattr(mvs, "MV_GIGE_DEVICE", 0)) | int(getattr(mvs, "MV_GENTL_GIGE_DEVICE", 0))
        device_list = mvs.MV_CC_DEVICE_INFO_LIST()
        ret = mvs.MvCamera.MV_CC_EnumDevices(transport_flag, device_list)
        if ret != 0:
            raise RuntimeError(f"MVS EnumDevices failed: 0x{ret:08x}")

        results: list[dict[str, Any]] = []
        for device_index in range(device_list.nDeviceNum):
            device_info = cast(
                device_list.pDeviceInfo[device_index],
                POINTER(mvs.MV_CC_DEVICE_INFO),
            ).contents
            device_entry = _device_to_dict(device_info, mvs)
            if device_entry is None:
                continue
            if net_export_filter is not None and device_entry["net_export"] != net_export_filter:
                continue
            device_entry["net_export_interface"] = local_interfaces.get(device_entry["net_export"], "")
            results.append({"index": device_index, **device_entry})

        return results, local_interfaces
    finally:
        finalize = getattr(mvs.MvCamera, "MV_CC_Finalize", None)
        if finalize is not None:
            finalize()


def _print_text(results: list[dict[str, Any]], net_export_filter: str | None, local_interfaces: dict[str, str]) -> None:
    if net_export_filter is not None:
        nic_name = local_interfaces.get(net_export_filter, "")
        if nic_name:
            print(f"net_export_filter: {net_export_filter} ({nic_name})")
        else:
            print(f"net_export_filter: {net_export_filter}")

    if not results:
        print("total_gige_devices: 0")
        print("note: PoE cameras are enumerated as GigE devices; use --interface or --net-export to filter by NIC.")
        return

    unique_nics = {device["net_export"] for device in results}
    if net_export_filter is None and len(unique_nics) == 1:
        auto_ip = next(iter(unique_nics))
        auto_ifname = local_interfaces.get(auto_ip, "")
        if auto_ifname:
            print(f"auto_detected_net_export: {auto_ip} ({auto_ifname})")
        else:
            print(f"auto_detected_net_export: {auto_ip}")
        print()

    for device_entry in results:
        label = device_entry["user_defined_name"] or device_entry["serial"] or device_entry["model"] or "unknown"
        print(f"device[{device_entry['index']}]: {label}")
        print(f"  model: {device_entry['model']}")
        print(f"  serial: {device_entry['serial']}")
        print(f"  ip: {device_entry['current_ip']}")
        if device_entry["net_export_interface"]:
            print(f"  net_export: {device_entry['net_export']} ({device_entry['net_export_interface']})")
        else:
            print(f"  net_export: {device_entry['net_export']}")
        if device_entry["manufacturer"]:
            print(f"  manufacturer: {device_entry['manufacturer']}")
        print()

    print(f"total_gige_devices: {len(results)}")
    print("note: PoE cameras are enumerated as GigE devices; net_export is the NIC IP used to reach the camera.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="List Hikrobot GigE/PoE cameras without opening devices.")
    parser.add_argument("--net-export", help="Filter by NIC IP address reported in stGigEInfo.nNetExport.")
    parser.add_argument("--interface", help="Filter by local interface name, for example enp7s0.")
    parser.add_argument("--json", action="store_true", help="Print JSON instead of text.")
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    net_export_filter = _resolve_net_export_filter(args)
    results, local_interfaces = list_gige_cameras(net_export_filter=net_export_filter)
    if args.json:
        print(
            json.dumps(
                {
                    "filter_net_export": net_export_filter,
                    "local_interfaces": local_interfaces,
                    "devices": results,
                },
                ensure_ascii=False,
                indent=2,
            )
        )
    else:
        _print_text(results, net_export_filter, local_interfaces)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
