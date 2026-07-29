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

"""
Run a host-side preflight before FR3 teleoperation recording.

This script is intentionally conservative: it validates host imports, FR3 arm
reachability, Franka Hand connectivity, and Hikrobot GigE camera open/read
probes without starting the full teleoperation/recording loop.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import os
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"
_HOST_ENV_MARKER = "LEROBOT_FR3_PREFLIGHT_ENV_READY"
for extra_path in (REPO_ROOT, SRC_ROOT):
    extra_path_str = str(extra_path)
    if extra_path_str not in sys.path:
        sys.path.insert(0, extra_path_str)

from tools.fr3 import fr3_record
from lerobot.cameras.hikrobot.camera_hikrobot import HikrobotCamera
from lerobot.cameras.hikrobot.configuration_hikrobot import Cv2Rotation, HikrobotCameraConfig
from tools.hikrobot.list_hikrobot_gige_cameras import _resolve_net_export_filter, list_gige_cameras

DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("fr3_record_hikrobot_example.yaml")


@dataclass
class CheckResult:
    name: str
    ok: bool
    details: str


@dataclass
class HikrobotSummary:
    ok: bool
    details: str
    expected_cameras: list[dict[str, Any]]
    detected_cameras: list[dict[str, Any]]
    matched_camera_names: list[str]
    missing_camera_names: list[str]
    missing_serials: list[str]
    unspecified_camera_names: list[str]
    suggestion: str | None


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a host-side FR3 + Franka Hand + Hikrobot preflight.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=REPO_ROOT,
        help="Repository root. Used to resolve host-side config paths.",
    )
    parser.add_argument(
        "--config-path",
        type=Path,
        default=DEFAULT_CONFIG_PATH,
        help="FR3 record YAML config to validate against.",
    )
    parser.add_argument("--robot-ip", default=None, help="Optional FR3 controller IP override.")
    parser.add_argument("--net-export", default=None, help="Filter Hikrobot GigE devices by NIC IPv4 address.")
    parser.add_argument("--interface", default=None, help="Filter Hikrobot GigE devices by NIC name.")
    parser.add_argument("--ping-count", type=int, default=1, help="Ping attempts for the FR3 reachability check.")
    parser.add_argument(
        "--hikrobot-frame-timeout-ms",
        type=int,
        default=1500,
        help="Timeout for the first Hikrobot frame probe.",
    )
    parser.add_argument(
        "--skip-host-imports",
        action="store_true",
        help="Skip the generic recording/teleoperation import bundle; device checks still import what they use.",
    )
    parser.add_argument(
        "--skip-ping",
        action="store_true",
        help="Skip the FR3 ping reachability check.",
    )
    parser.add_argument(
        "--skip-arm",
        action="store_true",
        help="Skip the panda_py FR3 arm connectivity check.",
    )
    parser.add_argument(
        "--skip-gripper",
        action="store_true",
        help="Skip the Franka Hand connectivity check.",
    )
    parser.add_argument(
        "--skip-hikrobot",
        action="store_true",
        help="Skip Hikrobot GigE camera enumeration and open/read probes.",
    )
    parser.add_argument(
        "--skip-camera-open",
        action="store_true",
        help="Skip per-camera open/read probes after Hikrobot enumeration succeeds.",
    )
    return parser.parse_args(argv)


def _load_runtime_config(config_path: Path, workspace: Path) -> dict[str, Any]:
    config_payload = fr3_record._load_yaml_config(config_path, workspace)
    return fr3_record._translate_workspace_bound_value_for_host(config_payload, workspace)


def _robot_config(config_payload: dict[str, Any]) -> dict[str, Any]:
    robot_cfg = config_payload.get("robot")
    if not isinstance(robot_cfg, dict):
        raise ValueError("Config must contain a mapping-style `robot` section.")
    return robot_cfg


def extract_expected_hikrobot_gige_cameras(config_payload: dict[str, Any]) -> list[dict[str, Any]]:
    robot_cfg = config_payload.get("robot")
    if not isinstance(robot_cfg, dict):
        return []
    cameras_cfg = robot_cfg.get("cameras")
    if not isinstance(cameras_cfg, dict):
        return []

    cameras: list[dict[str, Any]] = []
    for camera_name, raw_cfg in cameras_cfg.items():
        if not isinstance(raw_cfg, dict):
            continue
        if str(raw_cfg.get("type", "")).lower() != "hikrobot":
            continue
        transport_layer = str(raw_cfg.get("transport_layer", "usb")).lower()
        if transport_layer != "gige":
            continue
        camera_cfg = dict(raw_cfg)
        camera_cfg["name"] = str(camera_name)
        camera_cfg["transport_layer"] = transport_layer
        camera_cfg["serial"] = str(camera_cfg["serial"]) if camera_cfg.get("serial") is not None else None
        cameras.append(camera_cfg)
    return cameras


def build_camera_serial_suggestion(
    detected_cameras: list[dict[str, Any]],
    *,
    expected_camera_names: list[str] | None = None,
) -> str | None:
    if not detected_cameras:
        return None

    ordered_cameras = sorted(
        detected_cameras,
        key=lambda item: (
            str(item.get("net_export", "")),
            str(item.get("current_ip", "")),
            str(item.get("serial", "")),
        ),
    )
    names = list(expected_camera_names or [])
    if len(names) < len(ordered_cameras):
        names.extend(f"cam_{idx}" for idx in range(len(names), len(ordered_cameras)))

    lines = ["robot:", "  cameras:"]
    for idx, camera in enumerate(ordered_cameras):
        serial = str(camera.get("serial", "")).strip()
        if not serial:
            continue
        camera_name = names[idx]
        lines.append(f"    {camera_name}:")
        lines.append('      type: hikrobot')
        lines.append(f'      serial: "{serial}"')
        lines.append('      transport_layer: gige')
    return "\n".join(lines) if len(lines) > 2 else None


def summarize_hikrobot_cameras(
    expected_cameras: list[dict[str, Any]],
    detected_cameras: list[dict[str, Any]],
) -> HikrobotSummary:
    detected_by_serial = {
        str(camera.get("serial", "")).strip(): camera
        for camera in detected_cameras
        if str(camera.get("serial", "")).strip()
    }

    matched_camera_names: list[str] = []
    missing_camera_names: list[str] = []
    missing_serials: list[str] = []
    unspecified_camera_names: list[str] = []
    expected_names = [str(camera["name"]) for camera in expected_cameras]

    for camera in expected_cameras:
        camera_name = str(camera["name"])
        serial = str(camera.get("serial") or "").strip()
        if not serial:
            unspecified_camera_names.append(camera_name)
            continue
        if serial in detected_by_serial:
            matched_camera_names.append(camera_name)
            continue
        missing_camera_names.append(camera_name)
        missing_serials.append(serial)

    detail_parts = [f"detected_gige={len(detected_cameras)}"]
    if matched_camera_names:
        detail_parts.append(f"matched={matched_camera_names}")
    if missing_camera_names:
        detail_parts.append(f"missing={missing_camera_names}")
    if missing_serials:
        detail_parts.append(f"missing_serials={missing_serials}")
    if unspecified_camera_names:
        detail_parts.append(f"serial_missing_in_config={unspecified_camera_names}")

    ok = not missing_camera_names and not unspecified_camera_names
    if expected_cameras:
        detail_parts.append(f"expected={expected_names}")
    else:
        detail_parts.append("expected=[]")

    suggestion = None
    if expected_cameras and (missing_camera_names or unspecified_camera_names):
        suggestion = build_camera_serial_suggestion(detected_cameras, expected_camera_names=expected_names)
    elif not expected_cameras and detected_cameras:
        suggestion = build_camera_serial_suggestion(detected_cameras)

    return HikrobotSummary(
        ok=ok,
        details=", ".join(detail_parts),
        expected_cameras=expected_cameras,
        detected_cameras=detected_cameras,
        matched_camera_names=matched_camera_names,
        missing_camera_names=missing_camera_names,
        missing_serials=missing_serials,
        unspecified_camera_names=unspecified_camera_names,
        suggestion=suggestion,
    )


def _record(results: list[CheckResult], name: str, ok: bool, details: str) -> None:
    results.append(CheckResult(name=name, ok=ok, details=details))


def _exception_details(exc: BaseException) -> str:
    return "".join(traceback.format_exception_only(type(exc), exc)).strip()


def check_host_runtime_imports() -> tuple[bool, str]:
    modules = ("placo", "panda_py", "ruckig", "pyspacemouse", "easyhid")
    try:
        imported = []
        for module_name in modules:
            __import__(module_name)
            imported.append(module_name)
        from MvImport import MvCameraControl_class as mvs

        return True, f"modules={imported}, mvs={mvs.__file__}"
    except Exception as exc:
        return False, _exception_details(exc)


def check_ping(robot_ip: str, count: int) -> tuple[bool, str]:
    try:
        completed = subprocess.run(
            ["ping", "-c", str(count), robot_ip],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return True, "ping binary unavailable; skipped"

    details = completed.stdout.strip() or completed.stderr.strip() or f"returncode={completed.returncode}"
    return completed.returncode == 0, details


def check_fr3_arm(robot_ip: str) -> tuple[bool, str]:
    robot = None
    try:
        from panda_py import Panda

        robot = Panda(robot_ip)
        state = robot.get_state()
        q = getattr(state, "q", None)
        if q is None:
            return False, "connected but state.q missing"
        pose = robot.get_pose()
        pose_shape = getattr(pose, "shape", None)
        return True, f"connected, dof={len(q)}, q0={float(q[0]):.4f}, pose_shape={pose_shape}"
    except Exception as exc:
        return False, _exception_details(exc)
    finally:
        if robot is not None and hasattr(robot, "stop_controller"):
            try:
                robot.stop_controller()
            except Exception:
                pass


def check_franka_hand(robot_ip: str) -> tuple[bool, str]:
    gripper = None
    try:
        from panda_py import libfranka

        gripper = libfranka.Gripper(robot_ip)
        state = gripper.read_once()
        width = float(getattr(state, "width", 0.0))
        max_width = float(getattr(state, "max_width", 0.0))
        is_grasped = getattr(state, "is_grasped", None)
        return True, f"connected, width_m={width:.4f}, max_width_m={max_width:.4f}, is_grasped={is_grasped}"
    except Exception as exc:
        return False, _exception_details(exc)
    finally:
        if gripper is not None and hasattr(gripper, "stop"):
            try:
                gripper.stop()
            except Exception:
                pass


def _normalize_rotation(rotation: Any) -> Cv2Rotation:
    if rotation is None:
        return Cv2Rotation.NO_ROTATION
    if isinstance(rotation, Cv2Rotation):
        return rotation
    if isinstance(rotation, int):
        return Cv2Rotation(rotation)
    if isinstance(rotation, str):
        normalized = rotation.strip().lower()
        aliases = {
            "no_rotation": Cv2Rotation.NO_ROTATION,
            "rotate_90": Cv2Rotation.ROTATE_90,
            "rotate_180": Cv2Rotation.ROTATE_180,
            "rotate_270": Cv2Rotation.ROTATE_270,
            "0": Cv2Rotation.NO_ROTATION,
            "90": Cv2Rotation.ROTATE_90,
            "180": Cv2Rotation.ROTATE_180,
            "-90": Cv2Rotation.ROTATE_270,
        }
        if normalized in aliases:
            return aliases[normalized]
    return Cv2Rotation(rotation)


def _build_hikrobot_camera_config(camera_cfg: dict[str, Any]) -> HikrobotCameraConfig:
    image_shape = camera_cfg.get("image_shape")
    width = camera_cfg.get("width")
    height = camera_cfg.get("height")
    if image_shape is not None:
        if not isinstance(image_shape, (list, tuple)) or len(image_shape) != 2:
            raise ValueError(f"hikrobot camera {camera_cfg.get('name', '<unknown>')!r} must use image_shape=[height, width]")
        height, width = int(image_shape[0]), int(image_shape[1])
    if width is None or height is None:
        raise ValueError(f"hikrobot camera {camera_cfg.get('name', '<unknown>')!r} requires width/height or image_shape")

    fps = camera_cfg.get("fps")
    if fps is None:
        raise ValueError(f"hikrobot camera {camera_cfg.get('name', '<unknown>')!r} requires fps")

    return HikrobotCameraConfig(
        serial=str(camera_cfg["serial"]) if camera_cfg.get("serial") is not None else None,
        device_index=int(camera_cfg["device_index"]) if camera_cfg.get("device_index") is not None else None,
        width=int(width),
        height=int(height),
        fps=int(fps),
        warmup_s=0,
        transport_layer=str(camera_cfg.get("transport_layer", "gige")),
        color_mode=camera_cfg.get("color_mode", "bgr"),
        rotation=_normalize_rotation(camera_cfg.get("rotation")),
        exposure_us=float(camera_cfg["exposure_us"]) if camera_cfg.get("exposure_us") is not None else None,
        gain_db=float(camera_cfg["gain_db"]) if camera_cfg.get("gain_db") is not None else None,
        gamma=float(camera_cfg["gamma"]) if camera_cfg.get("gamma") is not None else None,
        white_balance_auto=str(camera_cfg.get("white_balance_auto", "continuous")),
        white_balance_red=int(camera_cfg["white_balance_red"]) if camera_cfg.get("white_balance_red") is not None else None,
        white_balance_green=int(camera_cfg["white_balance_green"]) if camera_cfg.get("white_balance_green") is not None else None,
        white_balance_blue=int(camera_cfg["white_balance_blue"]) if camera_cfg.get("white_balance_blue") is not None else None,
        lock_white_balance_after_warmup=bool(camera_cfg.get("lock_white_balance_after_warmup", True)),
        timeout_ms=int(camera_cfg.get("timeout_ms", 1000)),
    )


def probe_hikrobot_camera(camera_cfg: dict[str, Any], *, frame_timeout_ms: int) -> tuple[bool, str]:
    camera = None
    camera_name = str(camera_cfg.get("name", camera_cfg.get("serial", "<unknown>")))
    try:
        config = _build_hikrobot_camera_config(camera_cfg)
        camera = HikrobotCamera(config)
        camera.connect(warmup=False)
        frame = camera.async_read(timeout_ms=max(frame_timeout_ms, config.timeout_ms))
        return True, f"name={camera_name}, serial={config.serial}, frame_shape={tuple(frame.shape)}"
    except Exception as exc:
        return False, f"name={camera_name}, {_exception_details(exc)}"
    finally:
        if camera is not None and camera.is_connected:
            try:
                camera.disconnect()
            except Exception:
                pass


def check_hikrobot_cameras(
    config_payload: dict[str, Any],
    *,
    net_export_filter: str | None,
    probe_open: bool,
    frame_timeout_ms: int,
) -> tuple[list[CheckResult], HikrobotSummary | None]:
    results: list[CheckResult] = []
    expected_cameras = extract_expected_hikrobot_gige_cameras(config_payload)
    if not expected_cameras:
        _record(results, "hikrobot_config", True, "no Hikrobot GigE cameras declared in config")
        return results, None

    try:
        detected_cameras, local_interfaces = list_gige_cameras(net_export_filter=net_export_filter)
        nic_details = ""
        if net_export_filter is not None:
            nic_name = local_interfaces.get(net_export_filter, "")
            nic_details = f", net_export_filter={net_export_filter}" + (f" ({nic_name})" if nic_name else "")
        summary = summarize_hikrobot_cameras(expected_cameras, detected_cameras)
        _record(results, "hikrobot_enum", summary.ok, f"{summary.details}{nic_details}")
        if not summary.ok or not probe_open:
            return results, summary

        detected_by_serial = {
            str(camera.get("serial", "")).strip(): camera
            for camera in detected_cameras
            if str(camera.get("serial", "")).strip()
        }
        for camera_cfg in expected_cameras:
            serial = str(camera_cfg.get("serial") or "").strip()
            if not serial or serial not in detected_by_serial:
                continue
            probe_ok, probe_details = probe_hikrobot_camera(camera_cfg, frame_timeout_ms=frame_timeout_ms)
            _record(results, f"hikrobot_open:{camera_cfg['name']}", probe_ok, probe_details)
        return results, summary
    except Exception as exc:
        _record(results, "hikrobot_enum", False, _exception_details(exc))
        return results, None


def build_record_command(args: argparse.Namespace) -> str:
    config_path = fr3_record._to_host_path(args.config_path, args.workspace.resolve())
    return (
        "uv run --python .venv-fr3/bin/python python tools/fr3/fr3_record.py --runtime host "
        f"--config-path {config_path}"
    )


def ensure_host_env(args: argparse.Namespace) -> None:
    if os.environ.get(_HOST_ENV_MARKER) == "1":
        return

    env = fr3_record.build_host_env(args.workspace.resolve())
    env[_HOST_ENV_MARKER] = "1"
    os.execvpe(sys.executable, [sys.executable, *sys.argv], env)


def run_preflight(args: argparse.Namespace) -> tuple[list[CheckResult], HikrobotSummary | None]:
    config_payload = _load_runtime_config(args.config_path, args.workspace.resolve())
    robot_cfg = _robot_config(config_payload)
    robot_ip = args.robot_ip or str(robot_cfg.get("robot_ip", "")).strip()
    if not robot_ip:
        raise ValueError("Could not resolve robot IP from --robot-ip or config.robot.robot_ip.")

    results: list[CheckResult] = []
    if args.skip_host_imports:
        _record(results, "host_runtime_imports", True, "skipped for replay-only preflight")
    else:
        imports_ok, imports_details = check_host_runtime_imports()
        _record(results, "host_runtime_imports", imports_ok, imports_details)

    if not args.skip_ping:
        ping_ok, ping_details = check_ping(robot_ip, args.ping_count)
        _record(results, "fr3_ping", ping_ok, ping_details)

    if not args.skip_arm:
        arm_ok, arm_details = check_fr3_arm(robot_ip)
        _record(results, "fr3_arm", arm_ok, arm_details)

    gripper_backend = str(robot_cfg.get("gripper_backend", "pika")).lower()
    if not args.skip_gripper:
        if gripper_backend == "franka_hand":
            gripper_ok, gripper_details = check_franka_hand(robot_ip)
            _record(results, "franka_hand", gripper_ok, gripper_details)
        else:
            _record(results, "franka_hand", True, f"skipped because gripper_backend={gripper_backend}")

    hikrobot_summary = None
    if not args.skip_hikrobot:
        hikrobot_results, hikrobot_summary = check_hikrobot_cameras(
            config_payload,
            net_export_filter=_resolve_net_export_filter(args),
            probe_open=not args.skip_camera_open,
            frame_timeout_ms=args.hikrobot_frame_timeout_ms,
        )
        results.extend(hikrobot_results)

    return results, hikrobot_summary


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    ensure_host_env(args)
    try:
        results, hikrobot_summary = run_preflight(args)
    except Exception as exc:
        print(f"[FAIL] preflight_setup: {_exception_details(exc)}")
        return 1

    failures = [result.name for result in results if not result.ok]
    for result in results:
        status = "PASS" if result.ok else "FAIL"
        print(f"[{status}] {result.name}: {result.details}")

    if hikrobot_summary is not None and hikrobot_summary.suggestion is not None:
        print()
        print("hikrobot_camera_serial_suggestion:")
        print(hikrobot_summary.suggestion)

    if failures:
        print()
        print(f"fr3_record_preflight=FAIL failed_checks={failures}")
        return 1

    print()
    print("fr3_record_preflight=PASS")
    print("next_command:")
    print(f"  {build_record_command(args)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
