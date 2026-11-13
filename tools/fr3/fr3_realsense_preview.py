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
    parser.add_argument("--settle-min-s", type=float, default=0.5)
    parser.add_argument("--settle-max-s", type=float, default=3.0)
    return parser.parse_args()


# The gateway reaps an idle preview after 5 s, so a still asked for between rollouts is nearly
# always this pipeline's first frame -- taken before auto exposure and white balance have run
# (11-13: side camera B34 G54 R4 against ~110 settled). The side camera tape check then failed on
# it as "tape covered". Hold frames back until the image stops changing.
SETTLE_WINDOW = 8  # consecutive frames (~0.27 s at 30 fps)
SETTLE_TOLERANCE = 0.03  # relative change of any channel mean across the window


class ExposureSettle:
    """Says when a freshly started colour stream has settled: per-channel mean brightness stable
    over SETTLE_WINDOW frames, no sooner than min_s, and in any case by max_s."""

    def __init__(self, min_s: float, max_s: float) -> None:
        self.min_s = min_s
        self.max_s = max_s
        self.history: list[np.ndarray] = []
        self.settled = False

    def update(self, image: np.ndarray, elapsed_s: float) -> bool:
        if self.settled:
            return True
        self.history = (self.history + [image[::8, ::8].reshape(-1, 3).mean(axis=0)])[-SETTLE_WINDOW:]
        if elapsed_s >= self.max_s:
            self.settled = True
        elif elapsed_s >= self.min_s and len(self.history) == SETTLE_WINDOW:
            stack = np.stack(self.history)
            spread = (stack.max(axis=0) - stack.min(axis=0)) / np.maximum(stack.mean(axis=0), 1.0)
            self.settled = bool(spread.max() <= SETTLE_TOLERANCE)
        return self.settled


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
    started_s = time.monotonic()
    settle = ExposureSettle(args.settle_min_s, args.settle_max_s)
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
            if not settle.update(np.asanyarray(color_frame.get_data()), now - started_s):
                continue
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
