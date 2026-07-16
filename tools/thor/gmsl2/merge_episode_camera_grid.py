#!/usr/bin/env python3
"""Create per-episode camera grid videos for Thor GMSL2 datasets."""

from __future__ import annotations

import argparse
import math
import shutil
import subprocess
from pathlib import Path


VIDEO_SUFFIXES = (".mkv", ".mp4", ".mov")


def _camera_sort_key(path: Path) -> tuple[int, str]:
    stem = path.stem
    if stem.startswith("cam_"):
        try:
            return (int(stem.split("_", 1)[1]), stem)
        except ValueError:
            pass
    return (10_000, stem)


def _find_episode_dirs(root: Path) -> list[Path]:
    def has_camera_videos(path: Path) -> bool:
        return any(p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES for p in path.glob("cam_*"))

    if root.is_dir() and root.name.startswith("episode_") and has_camera_videos(root):
        return [root]

    candidates: list[Path] = []
    episodes_root = root / "episodes"
    if episodes_root.is_dir():
        candidates.extend(sorted(p for p in episodes_root.glob("episode_*") if p.is_dir()))
    candidates.extend(sorted(p for p in root.glob("episode_*") if p.is_dir()))

    seen: set[Path] = set()
    result: list[Path] = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in seen:
            continue
        seen.add(resolved)
        if has_camera_videos(candidate):
            result.append(candidate)
    return result


def _find_camera_videos(ep_dir: Path, camera_order: list[str] | None) -> list[Path]:
    by_stem = {
        p.stem: p
        for p in ep_dir.glob("cam_*")
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES
    }
    if camera_order:
        missing = [name for name in camera_order if name not in by_stem]
        if missing:
            raise FileNotFoundError(f"{ep_dir.name}: missing camera videos: {', '.join(missing)}")
        return [by_stem[name] for name in camera_order]
    return sorted(by_stem.values(), key=_camera_sort_key)


def _escape_drawtext(text: str) -> str:
    return (
        text.replace("\\", "\\\\")
        .replace(":", "\\:")
        .replace("'", "\\'")
        .replace("[", "\\[")
        .replace("]", "\\]")
    )


def _build_filter(
    cameras: list[Path],
    *,
    cols: int,
    tile_width: int,
    tile_height: int,
    labels: bool,
) -> str:
    parts: list[str] = []
    for idx, camera in enumerate(cameras):
        chain = (
            f"[{idx}:v]setpts=PTS-STARTPTS,"
            f"scale={tile_width}:{tile_height}:force_original_aspect_ratio=decrease,"
            f"pad={tile_width}:{tile_height}:(ow-iw)/2:(oh-ih)/2:color=black"
        )
        if labels:
            label = _escape_drawtext(camera.stem)
            chain += (
                ",drawtext="
                f"text='{label}':x=10:y=10:fontsize=24:fontcolor=white:"
                "box=1:boxcolor=black@0.55:boxborderw=6"
            )
        chain += f"[v{idx}]"
        parts.append(chain)

    layout = []
    for idx in range(len(cameras)):
        x = (idx % cols) * tile_width
        y = (idx // cols) * tile_height
        layout.append(f"{x}_{y}")
    stacked_inputs = "".join(f"[v{idx}]" for idx in range(len(cameras)))
    parts.append(f"{stacked_inputs}xstack=inputs={len(cameras)}:layout={'|'.join(layout)}:fill=black[v]")
    return ";".join(parts)


def _ffprobe_summary(path: Path) -> str:
    if shutil.which("ffprobe") is None:
        return "ffprobe unavailable"
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-select_streams",
        "v:0",
        "-show_entries",
        "stream=nb_frames,duration",
        "-of",
        "default=noprint_wrappers=1:nokey=0",
        str(path),
    ]
    proc = subprocess.run(cmd, text=True, capture_output=True, check=False)
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout).strip()
        return f"ffprobe failed: {detail}"
    return " ".join(line.strip() for line in proc.stdout.splitlines() if line.strip())


def merge_episode(
    ep_dir: Path,
    *,
    out_dir: Path,
    camera_order: list[str] | None,
    cols: int,
    tile_width: int,
    tile_height: int,
    fps: int,
    overwrite: bool,
    labels: bool,
    encoder: str,
    crf: int,
    cq: int | None,
    preset: str,
    probe: bool,
) -> Path:
    cameras = _find_camera_videos(ep_dir, camera_order)
    if not cameras:
        raise FileNotFoundError(f"{ep_dir}: no cam_* video files found")
    cols = max(1, min(cols, len(cameras)))
    rows = math.ceil(len(cameras) / cols)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{ep_dir.name}_{len(cameras)}cam_grid.mp4"
    if out_path.exists() and not overwrite:
        print(f"skip existing {out_path}")
        return out_path

    filter_complex = _build_filter(
        cameras,
        cols=cols,
        tile_width=tile_width,
        tile_height=tile_height,
        labels=labels,
    )
    cmd = ["ffmpeg", "-hide_banner", "-loglevel", "error"]
    if overwrite:
        cmd.append("-y")
    else:
        cmd.append("-n")
    for camera in cameras:
        cmd.extend(["-i", str(camera)])
    cmd.extend(["-filter_complex", filter_complex, "-map", "[v]", "-an", "-r", str(fps)])
    cmd.extend(_encoder_args(encoder=encoder, preset=preset, crf=crf, cq=cq))
    cmd.extend(["-pix_fmt", "yuv420p", "-movflags", "+faststart", "-shortest", str(out_path)])

    camera_names = ", ".join(p.stem for p in cameras)
    print(f"merge {ep_dir.name}: {len(cameras)} camera(s), grid={cols}x{rows}, cameras={camera_names}")
    subprocess.run(cmd, check=True)
    if probe:
        print(f"  wrote {out_path} ({_ffprobe_summary(out_path)})")
    else:
        print(f"  wrote {out_path}")
    return out_path


def _parse_camera_order(value: str | None) -> list[str] | None:
    if value is None or not value.strip():
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _encoder_args(*, encoder: str, preset: str, crf: int, cq: int | None) -> list[str]:
    if "nvenc" in encoder:
        return [
            "-c:v",
            encoder,
            "-preset",
            preset,
            "-rc",
            "vbr",
            "-cq",
            str(cq if cq is not None else crf),
            "-b:v",
            "0",
        ]
    if encoder in {"libx264", "libx265"}:
        return ["-c:v", encoder, "-preset", preset, "-crf", str(crf)]
    return ["-c:v", encoder]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("dataset_root", type=Path, help="Dataset root or one episode directory")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory. Default: <dataset_root>/grid_videos",
    )
    parser.add_argument(
        "--camera-order",
        default=None,
        help="Comma-separated camera order, e.g. cam_03,cam_06,cam_07",
    )
    parser.add_argument("--cols", type=int, default=4, help="Grid columns")
    parser.add_argument("--tile-width", type=int, default=480)
    parser.add_argument("--tile-height", type=int, default=270)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument(
        "--encoder",
        default="libx264",
        help="Output encoder. Default: libx264. Use h264_nvenc for NVIDIA GPU encode.",
    )
    parser.add_argument(
        "--gpu-encode",
        action="store_true",
        help="Shortcut for --encoder h264_nvenc with an NVENC-compatible preset.",
    )
    parser.add_argument("--crf", type=int, default=24)
    parser.add_argument("--cq", type=int, default=None, help="NVENC constant quality value")
    parser.add_argument("--preset", default="ultrafast")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-labels", action="store_true")
    parser.add_argument("--probe", action="store_true", help="Run ffprobe on each merged output")
    args = parser.parse_args()

    if shutil.which("ffmpeg") is None:
        raise SystemExit("ffmpeg is required but was not found in PATH")

    dataset_root = args.dataset_root.resolve()
    if not dataset_root.exists():
        raise SystemExit(f"dataset root does not exist: {dataset_root}")
    out_dir = args.out_dir.resolve() if args.out_dir else dataset_root / "grid_videos"
    episodes = _find_episode_dirs(dataset_root)
    if not episodes:
        raise SystemExit(f"no episode directories with cam_* videos found under {dataset_root}")

    camera_order = _parse_camera_order(args.camera_order)
    encoder = "h264_nvenc" if args.gpu_encode else args.encoder
    preset = args.preset
    if "nvenc" in encoder and preset == "ultrafast":
        preset = "p4"
    print(f"dataset={dataset_root}")
    print(f"episodes={len(episodes)}")
    print(f"out_dir={out_dir}")
    print(f"encoder={encoder} preset={preset}")
    for ep_dir in episodes:
        merge_episode(
            ep_dir,
            out_dir=out_dir,
            camera_order=camera_order,
            cols=args.cols,
            tile_width=args.tile_width,
            tile_height=args.tile_height,
            fps=args.fps,
            overwrite=args.overwrite,
            labels=not args.no_labels,
            encoder=encoder,
            crf=args.crf,
            cq=args.cq,
            preset=preset,
            probe=args.probe,
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
