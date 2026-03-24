#!/usr/bin/env python3
"""Scan /dev/video* devices, capture one frame per readable stream, and save probe images."""

from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path
from typing import Any

_REPO_ROOT = Path(__file__).resolve().parents[2]
_DEFAULT_OUTPUT_ROOT = _REPO_ROOT / "outputs" / "video_device_scan"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scan /dev/video* devices and save one frame from each readable stream.")
    parser.add_argument("--device-glob", default="/dev/video*", help="Glob pattern for camera devices.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory to save probe images and summary.json. Defaults to outputs/video_device_scan/<timestamp>.",
    )
    parser.add_argument("--warmup-reads", type=int, default=5, help="How many frames to read before saving the last one.")
    parser.add_argument("--width", type=int, default=None, help="Optional requested capture width.")
    parser.add_argument("--height", type=int, default=None, help="Optional requested capture height.")
    parser.add_argument("--fps", type=float, default=None, help="Optional requested capture FPS.")
    parser.add_argument(
        "--backend",
        choices=["any", "v4l2"],
        default="v4l2",
        help="OpenCV backend to use on open. 'v4l2' is usually the safest choice on Linux.",
    )
    return parser.parse_args(argv)


def import_cv2():
    try:
        import cv2  # type: ignore
    except ModuleNotFoundError as exc:
        raise SystemExit(
            "OpenCV python package not found. Install `cv2` in this environment or run the script inside the runtime/container that has OpenCV."
        ) from exc
    return cv2


def resolve_output_dir(output_dir: Path | None) -> Path:
    if output_dir is not None:
        return output_dir.resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return (_DEFAULT_OUTPUT_ROOT / timestamp).resolve()


def list_video_devices(device_glob: str) -> list[Path]:
    return sorted(Path("/").glob(device_glob.lstrip("/")), key=lambda path: path.name)


def _cv_backend(cv2: Any, backend_name: str) -> int:
    if backend_name == "v4l2":
        return int(cv2.CAP_V4L2)
    return int(cv2.CAP_ANY)


def sanitize_device_name(device_path: Path) -> str:
    return device_path.name.replace("/", "_")


def probe_device(
    device_path: Path,
    *,
    output_dir: Path,
    warmup_reads: int,
    width: int | None,
    height: int | None,
    fps: float | None,
    backend_name: str,
) -> dict[str, Any]:
    cv2 = import_cv2()
    result: dict[str, Any] = {
        "device": str(device_path),
        "opened": False,
        "readable": False,
        "saved_image": None,
        "backend": backend_name,
    }

    capture = cv2.VideoCapture(str(device_path), _cv_backend(cv2, backend_name))
    try:
        result["opened"] = bool(capture.isOpened())
        if not result["opened"]:
            result["error"] = "open_failed"
            return result

        if width is not None:
            capture.set(cv2.CAP_PROP_FRAME_WIDTH, float(width))
        if height is not None:
            capture.set(cv2.CAP_PROP_FRAME_HEIGHT, float(height))
        if fps is not None:
            capture.set(cv2.CAP_PROP_FPS, float(fps))

        frame = None
        attempts = max(int(warmup_reads), 1)
        for _ in range(attempts):
            ok, current = capture.read()
            if ok:
                frame = current

        result["actual_width"] = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        result["actual_height"] = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        result["actual_fps"] = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
        result["fourcc"] = decode_fourcc(capture.get(cv2.CAP_PROP_FOURCC))
        try:
            result["backend_name"] = capture.getBackendName()
        except Exception:
            result["backend_name"] = backend_name

        if frame is None:
            result["error"] = "read_failed"
            return result

        image_path = output_dir / f"{sanitize_device_name(device_path)}.png"
        if not cv2.imwrite(str(image_path), frame):
            result["error"] = "save_failed"
            return result

        result["readable"] = True
        result["saved_image"] = str(image_path)
        return result
    finally:
        capture.release()


def decode_fourcc(raw_value: float) -> str:
    code = int(raw_value)
    chars = [chr((code >> (8 * idx)) & 0xFF) for idx in range(4)]
    decoded = "".join(chars).strip("\x00")
    return decoded or "unknown"


def save_summary(results: list[dict[str, Any]], output_dir: Path) -> Path:
    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    return summary_path


def print_result(result: dict[str, Any]) -> None:
    status = "OK" if result["readable"] else "FAIL"
    suffix = ""
    if result["readable"]:
        suffix = f" -> {result['saved_image']}"
    elif result.get("error"):
        suffix = f" ({result['error']})"
    print(f"[{status}] {result['device']}{suffix}")


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = resolve_output_dir(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    devices = list_video_devices(args.device_glob)
    if not devices:
        print(f"No devices matched {args.device_glob}")
        return 1

    results = [
        probe_device(
            device,
            output_dir=output_dir,
            warmup_reads=args.warmup_reads,
            width=args.width,
            height=args.height,
            fps=args.fps,
            backend_name=args.backend,
        )
        for device in devices
    ]

    for result in results:
        print_result(result)

    summary_path = save_summary(results, output_dir)
    readable_count = sum(1 for result in results if result["readable"])
    print(f"Saved {readable_count} readable stream snapshots under {output_dir}")
    print(f"Summary written to {summary_path}")
    return 0 if readable_count > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
