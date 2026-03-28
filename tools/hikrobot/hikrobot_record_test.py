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
import json
import time
from pathlib import Path

import cv2  # type: ignore
import numpy as np

from lerobot.cameras.hikrobot import HikrobotCamera, HikrobotCameraConfig
from lerobot.cameras.hikrobot.camera_hikrobot import _decode_char_buffer, _extract_device_info, _load_mvs_sdk
from lerobot.cameras.hikrobot.configuration_hikrobot import ColorMode


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a short Hikrobot camera validation video.")
    parser.add_argument("--serial", default="DA5404760", help="Target Hikrobot camera serial number.")
    parser.add_argument("--width", type=int, default=1280, help="Capture width.")
    parser.add_argument("--height", type=int, default=720, help="Capture height.")
    parser.add_argument("--fps", type=int, default=30, help="Target capture and video FPS.")
    parser.add_argument("--exposure-us", type=float, default=10000.0, help="Manual exposure time in microseconds.")
    parser.add_argument(
        "--gain-mode",
        choices=("manual", "max"),
        default="manual",
        help="Gain strategy. Defaults to an indoor-safe manual gain instead of the camera maximum.",
    )
    parser.add_argument(
        "--gain-db",
        type=float,
        default=12.0,
        help="Manual analog gain in dB when --gain-mode=manual. Defaults to 12 dB for indoor scenes.",
    )
    parser.add_argument(
        "--gamma",
        type=float,
        default=1.3,
        help="Gamma correction value. Set to <= 0 to disable. Defaults to 1.3.",
    )
    parser.add_argument(
        "--white-balance-mode",
        choices=("auto_continuous", "auto_once", "manual"),
        default="auto_continuous",
        help="White-balance strategy. Use manual with --wb-red/--wb-blue or --load-white-balance-preset.",
    )
    parser.add_argument("--wb-red", type=int, default=None, help="Manual white-balance red ratio.")
    parser.add_argument("--wb-green", type=int, default=None, help="Manual white-balance green ratio.")
    parser.add_argument("--wb-blue", type=int, default=None, help="Manual white-balance blue ratio.")
    parser.add_argument(
        "--load-white-balance-preset",
        type=Path,
        default=None,
        help="Load a white-balance preset JSON file and apply it as manual white balance.",
    )
    parser.add_argument(
        "--save-white-balance-preset",
        type=Path,
        default=None,
        help="After warmup, read the current white-balance ratios and save them to this JSON file.",
    )
    parser.add_argument(
        "--duration-s",
        type=float,
        default=10.0,
        help="Recording duration in seconds.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("outputs/hikrobot"),
        help="Directory for the recorded video and metadata.",
    )
    parser.add_argument(
        "--codec",
        default="mp4v",
        help="OpenCV fourcc codec. Defaults to mp4v for broad compatibility.",
    )
    parser.add_argument(
        "--warmup-s",
        type=int,
        default=2,
        help="Warmup duration passed to the camera backend.",
    )
    parser.add_argument(
        "--keep-awb-running-during-recording",
        action="store_true",
        help=(
            "Keep continuous auto white balance active while writing video. "
            "By default auto_continuous is only used during warmup, then locked before recording."
        ),
    )
    return parser.parse_args()


def build_output_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    stem = f"hikrobot_{args.serial}_{args.width}x{args.height}_{args.fps}fps_{timestamp}"
    return args.output_dir / f"{stem}.mp4", args.output_dir / f"{stem}.json"


def _resolve_white_balance(args: argparse.Namespace) -> tuple[str, dict[str, int | None]]:
    white_balance_mode = args.white_balance_mode
    manual_ratios = {"red": args.wb_red, "green": args.wb_green, "blue": args.wb_blue}

    if args.load_white_balance_preset is not None:
        preset = json.loads(args.load_white_balance_preset.read_text(encoding="utf-8"))
        manual_ratios = {
            "red": preset.get("red"),
            "green": preset.get("green"),
            "blue": preset.get("blue"),
        }
        white_balance_mode = "manual"

    if white_balance_mode == "manual" and manual_ratios["red"] is None and manual_ratios["blue"] is None:
        raise ValueError("Manual white balance requires --wb-red/--wb-blue or --load-white-balance-preset.")

    return white_balance_mode, manual_ratios


def _get_camera_float(camera: HikrobotCamera, key: str) -> tuple[float, float, float]:
    float_value = camera._mvs.MVCC_FLOATVALUE()
    ret = camera._cam.MV_CC_GetFloatValue(key, float_value)
    if ret != 0:
        raise RuntimeError(f"Failed to query {key}: 0x{ret:08x}")
    return float(float_value.fCurValue), float(float_value.fMin), float(float_value.fMax)


def _resolve_gain_db(serial: str, gain_mode: str, requested_gain_db: float) -> tuple[float, float]:
    mvs = _load_mvs_sdk()
    transport = getattr(mvs, "MV_USB_DEVICE", 0) | getattr(mvs, "MV_GIGE_DEVICE", 0)
    device_list = mvs.MV_CC_DEVICE_INFO_LIST()
    ret = mvs.MvCamera.MV_CC_EnumDevices(transport, device_list)
    if ret != 0:
        raise RuntimeError(f"MVS EnumDevices failed while resolving gain: 0x{ret:08x}")

    device_info = None
    for idx in range(device_list.nDeviceNum):
        candidate = _extract_device_info(device_list.pDeviceInfo[idx], mvs)
        usb_info = getattr(candidate.SpecialInfo, "stUsb3VInfo", None)
        candidate_serial = _decode_char_buffer(usb_info.chSerialNumber) if usb_info is not None else ""
        if candidate_serial == serial:
            device_info = candidate
            break
    if device_info is None:
        raise RuntimeError(f"Hikrobot device with serial {serial!r} not found while resolving gain.")

    cam = mvs.MvCamera()
    try:
        ret = cam.MV_CC_CreateHandle(device_info)
        if ret != 0:
            raise RuntimeError(f"MVS CreateHandle failed while resolving gain: 0x{ret:08x}")
        ret = cam.MV_CC_OpenDevice(getattr(mvs, "MV_ACCESS_Exclusive", 1), 0)
        if ret != 0:
            raise RuntimeError(f"MVS OpenDevice failed while resolving gain: 0x{ret:08x}")
        float_value = mvs.MVCC_FLOATVALUE()
        ret = cam.MV_CC_GetFloatValue("Gain", float_value)
        if ret != 0:
            raise RuntimeError(f"Failed to query gain range: 0x{ret:08x}")
        min_gain_db = float(float_value.fMin)
        max_gain_db = float(float_value.fMax)
    finally:
        try:
            cam.MV_CC_CloseDevice()
        except Exception:
            pass
        try:
            cam.MV_CC_DestroyHandle()
        except Exception:
            pass

    resolved_gain_db = max_gain_db if gain_mode == "max" else float(np.clip(requested_gain_db, min_gain_db, max_gain_db))
    return resolved_gain_db, max_gain_db


def main() -> int:
    args = parse_args()
    video_path, metadata_path = build_output_paths(args)
    resolved_gain_db, max_gain_db = _resolve_gain_db(args.serial, args.gain_mode, args.gain_db)
    white_balance_mode, manual_white_balance = _resolve_white_balance(args)
    lock_white_balance_after_warmup = (
        white_balance_mode == "auto_continuous" and not args.keep_awb_running_during_recording
    )

    config = HikrobotCameraConfig(
        serial=args.serial,
        width=args.width,
        height=args.height,
        fps=args.fps,
        color_mode=ColorMode.BGR,
        warmup_s=args.warmup_s,
        exposure_us=args.exposure_us,
        gain_db=resolved_gain_db,
        gamma=args.gamma if args.gamma > 0 else None,
        white_balance_auto="continuous" if white_balance_mode == "auto_continuous" else "once",
        white_balance_red=manual_white_balance["red"],
        white_balance_green=manual_white_balance["green"],
        white_balance_blue=manual_white_balance["blue"],
        lock_white_balance_after_warmup=lock_white_balance_after_warmup,
        timeout_ms=max(1000, int(2000 / max(args.fps, 1))),
    )
    if white_balance_mode == "manual":
        config.white_balance_auto = "off"

    camera = HikrobotCamera(config)
    writer = cv2.VideoWriter(
        str(video_path),
        cv2.VideoWriter_fourcc(*args.codec),
        float(args.fps),
        (args.width, args.height),
    )
    if not writer.isOpened():
        raise RuntimeError(f"Failed to open video writer for {video_path}")

    started_at = time.perf_counter()
    frame_count = 0
    timestamps: list[float] = []
    applied_gain_db: float | None = None
    applied_white_balance: dict[str, int] | None = None
    gamma_mode = "disabled" if args.gamma <= 0 else "pre_stream"

    print(
        f"[INFO] Recording Hikrobot camera {args.serial} at "
        f"{args.width}x{args.height}, {args.fps} fps, {args.exposure_us:.0f} us exposure, "
        f"gain {'max' if args.gain_mode == 'max' else f'{args.gain_db:.1f} dB'}, "
        f"gamma {'off' if args.gamma <= 0 else f'{args.gamma:.2f}'}, "
        f"white balance mode {white_balance_mode}"
        f"{' (locked after warmup)' if lock_white_balance_after_warmup else ''}."
    )
    print(f"[INFO] Output video: {video_path}")

    try:
        camera.connect(warmup=True)
        applied_gain_db, _, _ = _get_camera_float(camera, "Gain")
        should_read_white_balance_before_recording = (
            white_balance_mode != "auto_continuous"
            or lock_white_balance_after_warmup
            or args.save_white_balance_preset is not None
        )
        if should_read_white_balance_before_recording:
            applied_white_balance = camera.get_white_balance_ratios()

        print(f"[INFO] Applied gain: {applied_gain_db:.3f} dB (camera max {max_gain_db:.3f} dB), gamma mode: {gamma_mode}.")
        if applied_white_balance is not None:
            print(f"[INFO] White balance ratios: {applied_white_balance}.")
        if args.save_white_balance_preset is not None:
            args.save_white_balance_preset.parent.mkdir(parents=True, exist_ok=True)
            args.save_white_balance_preset.write_text(json.dumps(applied_white_balance, indent=2), encoding="utf-8")
            print(f"[INFO] Saved white-balance preset: {args.save_white_balance_preset}")
        capture_deadline = time.perf_counter() + args.duration_s

        while time.perf_counter() < capture_deadline:
            frame = camera.async_read(timeout_ms=max(1000, int(2000 / max(args.fps, 1))))
            if frame.shape[:2] != (args.height, args.width):
                raise RuntimeError(f"Unexpected frame shape {frame.shape}; expected {(args.height, args.width, 3)}")
            writer.write(frame)
            frame_count += 1
            timestamps.append(time.perf_counter())
        if applied_white_balance is None:
            applied_white_balance = camera.get_white_balance_ratios()
    finally:
        writer.release()
        try:
            camera.disconnect()
        except Exception:
            pass

    elapsed_s = max(time.perf_counter() - started_at, 1e-9)
    intervals = [b - a for a, b in zip(timestamps, timestamps[1:])]
    measured_fps = (1.0 / (sum(intervals) / len(intervals))) if intervals else 0.0

    metadata = {
        "serial": args.serial,
        "width": args.width,
        "height": args.height,
        "fps_requested": args.fps,
        "fps_measured": round(measured_fps, 3),
        "duration_requested_s": args.duration_s,
        "duration_elapsed_s": round(elapsed_s, 3),
        "exposure_us": args.exposure_us,
        "gain_mode": args.gain_mode,
        "gain_db_requested": args.gain_db,
        "gain_db_applied": None if applied_gain_db is None else round(applied_gain_db, 3),
        "gain_db_max": None if max_gain_db is None else round(max_gain_db, 3),
        "gamma": args.gamma,
        "gamma_mode": gamma_mode,
        "white_balance_mode": white_balance_mode,
        "white_balance_locked_after_warmup": lock_white_balance_after_warmup,
        "white_balance_ratios": applied_white_balance,
        "frames_written": frame_count,
        "video_path": str(video_path.resolve()),
        "codec": args.codec,
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(f"[INFO] Frames written: {frame_count}")
    print(f"[INFO] Measured FPS: {measured_fps:.3f}")
    print(f"[INFO] Metadata: {metadata_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
