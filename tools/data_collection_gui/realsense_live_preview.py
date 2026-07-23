#!/usr/bin/env python3
"""Write the latest RealSense color frame to a JPEG for the replay UI."""

from __future__ import annotations

import argparse
import json
import signal
import time
from pathlib import Path


def _atomic_write(path: Path, payload: bytes) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_bytes(payload)
    temporary.replace(path)


def _write_status(path: Path, **payload: object) -> None:
    payload["updated_at"] = time.time()
    _atomic_write(path, (json.dumps(payload) + "\n").encode("utf-8"))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--status", type=Path, required=True)
    parser.add_argument("--serial", default="", help="Empty selects the first connected RealSense.")
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=15)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.status.parent.mkdir(parents=True, exist_ok=True)
    try:
        import cv2
        import numpy as np
        import pyrealsense2 as rs
    except Exception as exc:  # pragma: no cover - depends on Thor runtime
        _write_status(args.status, available=False, running=False, error=f"RealSense runtime unavailable: {exc}")
        return 2

    devices = list(rs.context().query_devices())
    if not devices:
        _write_status(args.status, available=False, running=False, error="No RealSense camera detected")
        return 3
    serials = [device.get_info(rs.camera_info.serial_number) for device in devices]
    selected_serial = str(args.serial).strip() or serials[0]
    if selected_serial not in serials:
        _write_status(
            args.status,
            available=False,
            running=False,
            error=f"RealSense serial {selected_serial} not found; connected: {serials}",
        )
        return 4

    running = True

    def request_stop(_signum: int, _frame: object) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(selected_serial)
    config.enable_stream(rs.stream.color, int(args.width), int(args.height), rs.format.bgr8, int(args.fps))
    try:
        pipeline.start(config)
        _write_status(
            args.status,
            available=True,
            running=True,
            serial=selected_serial,
            width=int(args.width),
            height=int(args.height),
            fps=int(args.fps),
            error="",
        )
        while running:
            frames = pipeline.wait_for_frames(timeout_ms=1500)
            color = frames.get_color_frame()
            if not color:
                continue
            image = np.asanyarray(color.get_data())
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, 82])
            if ok:
                _atomic_write(args.output, encoded.tobytes())
    except Exception as exc:
        _write_status(
            args.status,
            available=True,
            running=False,
            serial=selected_serial,
            error=f"RealSense preview failed: {exc}",
        )
        return 5
    finally:
        try:
            pipeline.stop()
        except Exception:
            pass
    _write_status(args.status, available=True, running=False, serial=selected_serial, error="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
