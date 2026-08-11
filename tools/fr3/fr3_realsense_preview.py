#!/usr/bin/env python3

from __future__ import annotations

import argparse
import signal
import sys
import time

import cv2
import numpy as np
import pyrealsense2 as rs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Stream concatenated RealSense JPEG frames to stdout.")
    parser.add_argument("--serial", required=True)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--output-width", type=int, default=480)
    parser.add_argument("--output-fps", type=float, default=10.0)
    parser.add_argument("--quality", type=int, default=70)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    running = True

    def stop(_signum, _frame) -> None:
        nonlocal running
        running = False

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)

    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(args.serial)
    config.enable_stream(rs.stream.color, args.width, args.height, rs.format.bgr8, args.fps)
    pipeline.start(config)
    output_height = max(1, round(args.output_width * args.height / max(args.width, 1)))
    frame_period_s = 1.0 / max(args.output_fps, 0.1)
    next_frame_s = time.monotonic()
    output = sys.stdout.buffer
    try:
        while running:
            frames = pipeline.wait_for_frames(timeout_ms=1000)
            color_frame = frames.get_color_frame()
            if not color_frame:
                continue
            now = time.monotonic()
            if now < next_frame_s:
                continue
            next_frame_s = now + frame_period_s
            image = cv2.resize(
                np.asanyarray(color_frame.get_data()),
                (args.output_width, output_height),
                interpolation=cv2.INTER_AREA,
            )
            ok, encoded = cv2.imencode(".jpg", image, [cv2.IMWRITE_JPEG_QUALITY, args.quality])
            if not ok:
                continue
            output.write(encoded.tobytes())
            output.flush()
    finally:
        pipeline.stop()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
