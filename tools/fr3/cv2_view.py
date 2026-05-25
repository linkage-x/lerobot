#!/usr/bin/env python3
"""Cycle through /dev/video* streams with an OpenCV preview window."""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview /dev/video* camera streams one at a time.")
    parser.add_argument("--device-glob", default="/dev/video*", help="Glob pattern used to find video devices.")
    parser.add_argument("--backend", choices=("any", "v4l2"), default="v4l2", help="OpenCV capture backend.")
    parser.add_argument("--width", type=int, default=None, help="Optional requested capture width.")
    parser.add_argument("--height", type=int, default=None, help="Optional requested capture height.")
    parser.add_argument("--fps", type=float, default=None, help="Optional requested capture FPS.")
    parser.add_argument("--window-prefix", default="cv2_view", help="Prefix for the OpenCV window title.")
    return parser.parse_args(argv)


def import_cv2() -> Any:
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "OpenCV python package not found. Install `opencv-python` or run inside an environment with cv2."
        ) from exc
    return cv2


def list_video_devices(device_glob: str) -> list[Path]:
    return sorted(Path("/").glob(device_glob.lstrip("/")), key=lambda path: path.name)


def cv_backend(cv2: Any, backend_name: str) -> int:
    if backend_name == "v4l2":
        return int(cv2.CAP_V4L2)
    return int(cv2.CAP_ANY)


def open_capture(cv2: Any, device: Path, args: argparse.Namespace) -> Any | None:
    cap = cv2.VideoCapture(str(device), cv_backend(cv2, args.backend))
    if not cap.isOpened():
        cap.release()
        return None

    if args.width is not None:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, args.width)
    if args.height is not None:
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, args.height)
    if args.fps is not None:
        cap.set(cv2.CAP_PROP_FPS, args.fps)

    return cap


def frame_label(cv2: Any, frame: Any, label: str) -> Any:
    output = frame.copy()
    cv2.rectangle(output, (0, 0), (output.shape[1], 36), (0, 0, 0), thickness=-1)
    cv2.putText(
        output,
        label,
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 255, 0),
        2,
        cv2.LINE_AA,
    )
    return output


def device_info(cv2: Any, cap: Any, device: Path, index: int, total: int) -> str:
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    if fps and fps > 0:
        return f"{index + 1}/{total} {device} | {width}x{height} @ {fps:.1f} FPS"
    return f"{index + 1}/{total} {device} | {width}x{height}"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cv2 = import_cv2()
    devices = list_video_devices(args.device_glob)
    if not devices:
        print(f"No devices matched {args.device_glob}")
        return 1

    print("Discovered video devices:")
    for idx, device in enumerate(devices):
        print(f"  [{idx}] {device}")
    print("Controls: n = next device, q/Esc = quit")

    current = 0
    window_name: str | None = None

    while True:
        device = devices[current]
        cap = open_capture(cv2, device, args)
        if cap is None:
            print(f"Could not open {device}; skipping.")
            current = (current + 1) % len(devices)
            if current == 0:
                print("No readable video devices found.")
                return 1
            continue

        title = f"{args.window_prefix}: {device}"
        if window_name is not None and window_name != title:
            cv2.destroyWindow(window_name)
        window_name = title
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        print(f"Showing {device}; press n for next, q/Esc to quit.")

        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                print(f"Failed to read from {device}; moving to next device.")
                key = ord("n")
                break

            label = device_info(cv2, cap, device, current, len(devices))
            cv2.imshow(window_name, frame_label(cv2, frame, label))
            key = cv2.waitKey(1) & 0xFF

            if key in (ord("q"), 27, ord("n")):
                break

        cap.release()

        if key in (ord("q"), 27):
            break
        current = (current + 1) % len(devices)

    cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
