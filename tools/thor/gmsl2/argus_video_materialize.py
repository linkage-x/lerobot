"""Materialize Argus-aligned camera videos.

The metadata sync report tells us which encoded frame range is common across
all cameras.  This module rewrites each raw camera MKV to that exact range, so
the episode files themselves obey the synchronization contract instead of
requiring every downstream consumer to interpret the report.
"""

from __future__ import annotations

import shutil
import subprocess
from dataclasses import asdict, dataclass
from functools import lru_cache
from pathlib import Path

from tools.thor.gmsl2.argus_frame_sync import (
    ArgusFrameMetadata,
    CameraFrameWindow,
    EpisodeAlignment,
    camera_frame_windows,
    frame_metadata_sidecar_path,
    write_frame_metadata_csv,
)


@dataclass(frozen=True)
class MaterializedVideo:
    camera: str
    path: str
    start_frame_index: int
    stop_frame_index: int
    frame_count: int
    raw_frame_count: int | None
    rewritten: bool


def _ffprobe_frame_count(path: Path, *, timeout_s: float = 30.0) -> int | None:
    if shutil.which("ffprobe") is None:
        return None
    cmd = [
        "ffprobe", "-v", "error",
        "-count_frames",
        "-select_streams", "v:0",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=nokey=1:noprint_wrappers=1",
        str(path),
    ]
    try:
        result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    text = (result.stdout or "").strip()
    if result.returncode != 0 or not text or text == "N/A":
        return None
    try:
        return int(text.splitlines()[-1])
    except ValueError:
        return None


def _encoder_for_codec(codec: str) -> str:
    codec = codec.lower()
    if codec == "h265":
        return "libx265"
    if codec == "h264":
        return "libx264"
    raise ValueError(f"unsupported codec for materialization: {codec!r}")


@lru_cache(maxsize=1)
def _available_ffmpeg_encoders() -> frozenset[str]:
    if shutil.which("ffmpeg") is None:
        return frozenset()
    try:
        result = subprocess.run(
            ["ffmpeg", "-hide_banner", "-encoders"],
            text=True,
            capture_output=True,
            timeout=10.0,
            check=False,
        )
    except (OSError, subprocess.TimeoutExpired):
        return frozenset()
    if result.returncode != 0:
        return frozenset()
    encoders: set[str] = set()
    for line in (result.stdout or "").splitlines():
        parts = line.split()
        if len(parts) >= 2 and parts[0].startswith("V"):
            encoders.add(parts[1])
    return frozenset(encoders)


def _select_materialization_encoder(
    codec: str,
    *,
    available_encoders: set[str] | frozenset[str] | None = None,
) -> str:
    encoder = _encoder_for_codec(codec)
    available = (
        _available_ffmpeg_encoders()
        if available_encoders is None else frozenset(available_encoders)
    )
    if encoder not in available:
        raise RuntimeError(
            f"ffmpeg encoder {encoder!r} is required to materialize "
            f"{codec.lower()} Argus-aligned videos"
        )
    return encoder


def build_ffmpeg_select_command(
    src: Path,
    dst: Path,
    window: CameraFrameWindow,
    *,
    fps: int,
    codec: str,
    encoder: str | None = None,
) -> list[str]:
    """Build a frame-exact ffmpeg command for one contiguous frame window."""

    if window.frame_count <= 0:
        raise ValueError(f"{window.camera} frame_count must be positive")
    end = window.stop_frame_index - 1
    vf = (
        f"select=between(n\\,{window.start_frame_index}\\,{end}),"
        f"setpts=N/({int(fps)}*TB)"
    )
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src),
        "-vf", vf,
        "-frames:v", str(window.frame_count),
        "-r", str(int(fps)),
        "-fps_mode", "cfr",
        "-an",
        "-c:v", encoder or _encoder_for_codec(codec),
        "-preset", "ultrafast",
        "-crf", "18",
    ]
    if codec.lower() == "h265":
        cmd.extend(["-x265-params", "log-level=error"])
    cmd.append(str(dst))
    return cmd


def _rewrite_sidecar(
    episode_dir: Path,
    camera: str,
    frames: list[ArgusFrameMetadata],
    window: CameraFrameWindow,
) -> None:
    by_encoded = {row.encoded_frame_index: row for row in frames}
    selected: list[ArgusFrameMetadata] = []
    for final_index, source_index in enumerate(
        range(window.start_frame_index, window.stop_frame_index)
    ):
        row = by_encoded[source_index]
        selected.append(
            ArgusFrameMetadata(
                camera=row.camera,
                encoded_frame_index=final_index,
                local_frame_number=row.local_frame_number,
                sensor_timestamp_ns=row.sensor_timestamp_ns,
                sof_tsc_ns=row.sof_tsc_ns,
                eof_tsc_ns=row.eof_tsc_ns,
                internal_frame_count=row.internal_frame_count,
            )
        )
    sidecar = frame_metadata_sidecar_path(episode_dir, camera)
    raw_sidecar = sidecar.with_suffix(sidecar.suffix + ".raw")
    if sidecar.exists():
        sidecar.replace(raw_sidecar)
    try:
        write_frame_metadata_csv(sidecar, selected)
    except Exception:
        if raw_sidecar.exists():
            raw_sidecar.replace(sidecar)
        raise
    try:
        raw_sidecar.unlink(missing_ok=True)
    except OSError:
        pass


def _ensure_selected_frames_exist(
    frames: list[ArgusFrameMetadata],
    window: CameraFrameWindow,
) -> None:
    available = {row.encoded_frame_index for row in frames}
    missing = [
        idx
        for idx in range(window.start_frame_index, window.stop_frame_index)
        if idx not in available
    ]
    if missing:
        raise ValueError(f"{window.camera} sidecar missing selected frames: {missing}")


def materialize_aligned_videos(
    episode_dir: Path,
    fragments: dict[str, Path],
    frames_by_camera: dict[str, list[ArgusFrameMetadata]],
    alignment: EpisodeAlignment,
    *,
    fps: int,
    codec: str,
    timeout_s: float = 600.0,
) -> dict[str, dict]:
    """Rewrite raw videos to the synchronized frame window.

    Returns a JSON-serializable payload keyed by camera.
    """

    if shutil.which("ffmpeg") is None:
        raise RuntimeError("ffmpeg is required to materialize Argus-aligned videos")

    windows = camera_frame_windows(alignment)
    encoder = _select_materialization_encoder(codec)
    results: dict[str, dict] = {}
    for camera, window in windows.items():
        src = fragments[camera]
        if not src.exists():
            raise FileNotFoundError(f"{camera} video not found: {src}")
        _ensure_selected_frames_exist(frames_by_camera[camera], window)

        raw_count = _ffprobe_frame_count(src)
        rewritten = False
        if raw_count != window.frame_count or window.start_frame_index != 0:
            tmp = src.with_name(f".{src.stem}.aligned.tmp{src.suffix}")
            backup = src.with_name(f"{src.name}.raw")
            if tmp.exists():
                tmp.unlink()
            cmd = build_ffmpeg_select_command(
                src,
                tmp,
                window,
                fps=fps,
                codec=codec,
                encoder=encoder,
            )
            result = subprocess.run(cmd, text=True, capture_output=True, timeout=timeout_s, check=False)
            if result.returncode != 0 or not tmp.exists() or tmp.stat().st_size <= 0:
                tmp.unlink(missing_ok=True)
                detail = (result.stderr or result.stdout or "").strip()
                raise RuntimeError(f"{camera} ffmpeg materialization failed: {detail}")
            out_count = _ffprobe_frame_count(tmp)
            if out_count is not None and out_count != window.frame_count:
                tmp.unlink(missing_ok=True)
                raise RuntimeError(
                    f"{camera} materialized frame count {out_count} != {window.frame_count}"
                )
            src.replace(backup)
            try:
                tmp.replace(src)
            except Exception:
                backup.replace(src)
                raise
            try:
                backup.unlink(missing_ok=True)
            except OSError:
                pass
            rewritten = True

        _rewrite_sidecar(episode_dir, camera, frames_by_camera[camera], window)
        results[camera] = asdict(MaterializedVideo(
            camera=camera,
            path=str(src),
            start_frame_index=window.start_frame_index,
            stop_frame_index=window.stop_frame_index,
            frame_count=window.frame_count,
            raw_frame_count=raw_count,
            rewritten=rewritten,
        ))

    return results
