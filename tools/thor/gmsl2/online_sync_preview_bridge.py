"""Bridge ``argus_online_sync`` frame bus clusters into UI preview JPEGs."""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import signal
import sys
import time

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.thor.gmsl2.online_sync_frame_client import ThorOnlineSyncFrameClient


_STOP = False


def _handle_signal(signum, frame) -> None:  # noqa: ANN001 - signal handler API
    del signum, frame
    global _STOP
    _STOP = True


def parse_camera_specs(specs: list[str]) -> dict[str, Path]:
    cameras: dict[str, Path] = {}
    for spec in specs:
        if "=" not in spec:
            raise ValueError(f"camera spec must be NAME=JPEG_PATH, got {spec!r}")
        name, path = spec.split("=", 1)
        name = name.strip()
        path = path.strip()
        if not name or not path:
            raise ValueError(f"camera spec must be NAME=JPEG_PATH, got {spec!r}")
        cameras[name] = Path(path)
    return cameras


def _write_preview_jpeg(frame, out_path: Path, *, preview_width: int, quality: int) -> None:
    try:
        import cv2  # type: ignore
    except ImportError as exc:  # pragma: no cover - optional runtime dependency
        raise RuntimeError("online-sync preview bridge requires cv2") from exc

    rgb = frame.as_rgb()
    height, width = rgb.shape[:2]
    if preview_width > 0 and width > preview_width:
        preview_height = max(1, int(round(height * (preview_width / width))))
        rgb = cv2.resize(rgb, (preview_width, preview_height), interpolation=cv2.INTER_AREA)
    bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_name(f".{out_path.name}.tmp")
    ok, encoded = cv2.imencode(
        ".jpg",
        bgr,
        [int(cv2.IMWRITE_JPEG_QUALITY), int(quality)],
    )
    if not ok:
        raise RuntimeError(f"failed to encode preview JPEG for {frame.camera}")
    tmp_path.write_bytes(encoded.tobytes())
    os.replace(tmp_path, out_path)


def run_bridge(
    *,
    frame_bus_dir: Path,
    cameras: dict[str, Path],
    fps: float,
    preview_width: int,
    quality: int,
) -> int:
    if not cameras:
        raise ValueError("at least one --camera NAME=JPEG_PATH is required")
    client = ThorOnlineSyncFrameClient(
        frame_bus_dir,
        cameras=sorted(cameras),
        validate_files=True,
        poll_interval_s=0.01,
    )
    min_period_s = 1.0 / max(0.1, float(fps))
    next_publish_seq: int | None = None
    last_emit_s = 0.0

    while not _STOP:
        cluster = client.get_latest(timeout_s=0.2, min_publish_seq=next_publish_seq)
        if cluster is None:
            continue
        next_publish_seq = cluster.publish_seq + 1

        now = time.monotonic()
        delay_s = min_period_s - (now - last_emit_s)
        if delay_s > 0:
            time.sleep(delay_s)
        for name, out_path in cameras.items():
            frame = cluster.frames.get(name)
            if frame is None:
                continue
            _write_preview_jpeg(
                frame,
                out_path,
                preview_width=preview_width,
                quality=quality,
            )
        last_emit_s = time.monotonic()
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Convert argus_online_sync frame-bus NV12 clusters to preview JPEGs"
    )
    parser.add_argument("--frame-bus-dir", required=True, type=Path)
    parser.add_argument("--camera", action="append", default=[], metavar="NAME=JPEG_PATH")
    parser.add_argument("--fps", type=float, default=5.0)
    parser.add_argument("--preview-width", type=int, default=480)
    parser.add_argument("--quality", type=int, default=80)
    args = parser.parse_args()

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    cameras = parse_camera_specs(args.camera)
    return run_bridge(
        frame_bus_dir=args.frame_bus_dir,
        cameras=cameras,
        fps=args.fps,
        preview_width=args.preview_width,
        quality=max(1, min(100, int(args.quality))),
    )


if __name__ == "__main__":
    raise SystemExit(main())
