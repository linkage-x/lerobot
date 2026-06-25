"""Consolidate a task's separately-recorded GMSL2 sessions into one LeRobot v3 dataset.

Each Connect→record session writes its own ``<name>_<timestamp>/`` directory
(raw per-camera MKV under ``episodes/episode_NNNNNN/`` plus, when BOX is
enabled, a per-session ``data/chunk-000/file-000.parquet`` of box state/action
on the camera-frame grid). Progress tracking already groups these sessions by
the ``repo_id`` trailing name; this script is the *consolidation* step the GUI
calls from Dataset Export: gather every session whose directory name shares the
task's base name, concatenate their episodes with a contiguous global
``episode_index``, and write a single loadable LeRobot v3 dataset under a
separate exports root.

**Runs in the Thor minimal env.** The Thor capture host only has
``pyarrow`` + ``gstreamer`` (no torch / datasets / av / ffmpeg — see
DEPLOYMENT.md §3), so this writer hand-rolls the v3 layout with pyarrow and
transcodes video with gstreamer rather than going through the heavy
``LeRobotDataset`` API. The output still loads through ``LeRobotDataset``
(validated on a full-deps host).

Video: source MKV is HEVC (``nvv4l2h265enc``). Each episode's per-camera MKV is
transcoded to a **H.264** ``.mp4`` (max decoder compatibility — LeRobot's
torchvision fallback is H.264-centric) and written as its own v3 video file
(``videos/<key>/chunk-000/file-<ep>.mp4``); the episode points at it with
``from_timestamp=0``. Transcode prefers the Jetson ``nvv4l2`` gst pipeline and
falls back to ``ffmpeg`` libx264 on dev hosts.

Multi-sensor time sync (see tools/thor/ts_sync.md): the camera grid is
authoritative — the minimum PWM-locked frame count across an episode's cameras,
transcoded to a constant ``i/fps`` grid so one query timestamp serves every
camera. Box state is matched onto that grid per episode, in priority order:

  1. the recorder's pre-synced session box parquet — paired to camera frames by
     ``frame_index`` (not list position), so a box grid that is longer than the
     camera clip (the ``round(duration*fps)`` rounding tail) is dropped and a
     shorter one carry-forwards its last reading;
  2. fallback when no session parquet exists — re-derive box state from the raw
     per-episode ``box_sensors.jsonl`` (MCU-clock calibration + per-sensor
     nearest-neighbour on the ``pts_offset + N/fps`` grid), i.e. the recorder's
     L3b alignment redone at export time.

The 6 box sensors are already nearest-neighbour-merged onto the camera grid (by
the recorder, or by the fallback); per-episode output ``timestamp`` is re-based
to ``i/fps`` to match the re-anchored per-episode video.
"""

from __future__ import annotations

import argparse
import csv
import json
import logging
import math
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

_DEFAULT_JOBS = 8

logger = logging.getLogger("export_v3")

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

# pyarrow-only helpers shared with the recorder's box v3 writer.
from tools.thor.gmsl2 import thor_lerobot_v3 as lr3  # noqa: E402

_CHUNKS_SIZE = 1000
_DATA_PATH = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
_VIDEO_PATH = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
_TRAJ_SIDECAR_NAME = "april_cube_tracking_in_robot_base"
_TRACKING_RUN_SUFFIX = "_thor_april_tracking_in_robot_base"
_POSE7_NAMES = ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")
_POSE7_FEATURE_NAMES = ("pose.x_m", "pose.y_m", "pose.z_m", "pose.qx", "pose.qy", "pose.qz", "pose.qw")
_POSE7_NAN = [math.nan] * 7


def _emit(text: str) -> None:
    """Single point for progress lines so the gateway can stream them."""
    print(text, flush=True)


# --------------------------------------------------------------- discovery ---


def _name_prefixes(name: str) -> set[str]:
    """Mirror of gateway._dataset_name_prefixes: strip a capture timestamp."""
    prefixes = {name}
    match = re.match(r"^(?P<base>.+)_\d{8}_\d{6}(?:_\d{2})?$", name)
    if match:
        prefixes.add(match.group("base"))
    return prefixes


def _has_gmsl2_episodes(path: Path) -> bool:
    eps = path / "episodes"
    return eps.is_dir() and any(eps.glob("episode_*/meta.json"))


def find_task_sessions(datasets_root: Path, base_name: str) -> list[Path]:
    """Session directories under ``datasets_root`` for ``base_name``, sorted."""
    base_name = base_name.split("/")[-1].strip()
    if not base_name or not datasets_root.is_dir():
        return []
    sessions = [
        entry
        for entry in datasets_root.iterdir()
        if entry.is_dir()
        and base_name in _name_prefixes(entry.name)
        and _has_gmsl2_episodes(entry)
    ]
    return sorted(sessions, key=lambda p: p.name)


def _episode_dirs(session_dir: Path) -> list[Path]:
    eps = session_dir / "episodes"
    if not eps.is_dir():
        return []
    return sorted(
        (d for d in eps.iterdir() if d.is_dir() and d.name.startswith("episode_")),
        key=lambda d: d.name,
    )


@dataclass
class EpisodeSource:
    session_dir: Path
    ep_dir: Path
    local_index: int  # episode_index within its session (for the box parquet join)


def gather_episodes(sessions: list[Path]) -> list[EpisodeSource]:
    out: list[EpisodeSource] = []
    for session in sessions:
        for ep_dir in _episode_dirs(session):
            match = re.search(r"episode_(\d+)$", ep_dir.name)
            local_index = int(match.group(1)) if match else len(out)
            out.append(EpisodeSource(session, ep_dir, local_index))
    return out


# ------------------------------------------------------------------ inputs ---


def _load_meta(ep_dir: Path) -> dict[str, Any]:
    meta_path = ep_dir / "meta.json"
    if not meta_path.is_file():
        return {}
    try:
        return json.loads(meta_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}


def _camera_entries(meta: dict[str, Any], ep_dir: Path) -> list[tuple[str, Path]]:
    """(camera_key, mkv_path) pairs, from meta when present else by globbing."""
    cams: list[tuple[str, Path]] = []
    for cam in meta.get("cameras", []) or []:
        name = str(cam.get("name") or "").strip()
        file = str(cam.get("file") or "").strip()
        if name and file and (ep_dir / file).is_file():
            cams.append((name, ep_dir / file))
    if cams:
        return sorted(cams, key=lambda c: c[0])
    return [(p.stem, p) for p in sorted(ep_dir.glob("cam_*.mkv"))]


def _load_box_rows(session_dir: Path) -> dict[int, list[dict[str, Any]]]:
    """Per-session box parquet rows grouped by local episode_index ({} if none)."""
    parquet = session_dir / "data" / "chunk-000" / "file-000.parquet"
    if not parquet.is_file():
        return {}
    import pyarrow.parquet as pq

    cols = pq.read_table(str(parquet)).to_pydict()
    by_ep: dict[int, list[dict[str, Any]]] = {}
    for i in range(len(cols["episode_index"])):
        ep = int(cols["episode_index"][i])
        row: dict[str, Any] = {
            "frame_index": int(cols["frame_index"][i]),
            "observation.state": [float(v) for v in cols["observation.state"][i]],
            "action": [float(v) for v in cols["action"][i]],
        }
        if "box.timestamps" in cols:
            row["box.timestamps"] = [float(v) for v in cols["box.timestamps"][i]]
        by_ep.setdefault(ep, []).append(row)
    for rows in by_ep.values():
        rows.sort(key=lambda r: r["frame_index"])
    return by_ep


# ----------------------------------------------------- box ↔ camera sync ---


def _pts_offset_s(meta: dict[str, Any]) -> float:
    """Camera pipeline-start delay correction for the frame-time grid.

    ``pts_offset = mean(camera_first_wall_s[cam] - t0_wall_s)`` from
    ``meta.json.sync_reference`` — the same value the recorder bakes into the box
    parquet ``timestamp`` column (ts_sync.md §5.1). Used as the t0-relative
    origin when re-aligning raw box samples onto the camera frame grid. Returns
    0.0 when the anchors are missing.
    """
    sync_reference = meta.get("sync_reference") if isinstance(meta, dict) else None
    if not isinstance(sync_reference, dict):
        return 0.0
    t0_wall_s = sync_reference.get("t0_wall_s")
    camera_first_wall_s = sync_reference.get("camera_first_wall_s")
    if not isinstance(t0_wall_s, (int, float)) or not isinstance(camera_first_wall_s, dict):
        return 0.0
    deltas = [
        float(wall_s) - float(t0_wall_s)
        for wall_s in camera_first_wall_s.values()
        if isinstance(wall_s, (int, float))
    ]
    if not deltas:
        return 0.0
    return sum(deltas) / len(deltas)


def _align_box_rows_by_frame_index(
    box_rows: list[dict[str, Any]],
    n_frames: int,
    state_width: int,
    ts_width: int,
) -> tuple[list[list[float]], list[list[float]], list[list[float]] | None, int]:
    """Align pre-synced box parquet rows to the camera frame grid by
    ``frame_index`` (not list position).

    The recorder already nearest-neighbour-merges the 6 box sensors onto the
    ``pts_offset + N/fps`` grid (ts_sync.md §6), so row ``frame_index == N``
    belongs to camera frame ``N``. Indexing by ``frame_index`` (rather than the
    old positional ``box_rows[:n]`` slice) keeps that pairing correct even if the
    box grid is longer than the camera clip (phantom duration-rounding tail),
    shorter (carry-forward the last sample), or not 0-based contiguous.

    Returns ``(state_rows, action_rows, ts_rows_or_None, missing_count)``.
    """
    by_frame = {int(r["frame_index"]): r for r in box_rows}
    state_rows: list[list[float]] = []
    action_rows: list[list[float]] = []
    ts_rows: list[list[float]] | None = [] if ts_width else None
    last: dict[str, Any] | None = None
    missing = 0
    for i in range(n_frames):
        row = by_frame.get(i)
        if row is None:
            missing += 1
            row = last  # camera frame with no box sample -> hold last reading
        else:
            last = row
        if row is None:
            state_rows.append([0.0] * state_width)
            action_rows.append([0.0] * state_width)
            if ts_rows is not None:
                ts_rows.append([0.0] * ts_width)
        else:
            state_rows.append(list(row["observation.state"]))
            action_rows.append(list(row["action"]))
            if ts_rows is not None:
                ts_rows.append(list(row.get("box.timestamps", [0.0] * ts_width)))
    return state_rows, action_rows, ts_rows, missing


def _box_snapshots_from_meta(meta: dict[str, Any]) -> list[dict[str, Any]]:
    box_meta = meta.get("box_collection") if isinstance(meta, dict) else None
    snapshots = box_meta.get("snapshots") if isinstance(box_meta, dict) else None
    if not isinstance(snapshots, list):
        return []
    return [snap for snap in snapshots if isinstance(snap, dict)]


def _rows_from_lr3_rows(
    rows: list[dict[str, Any]],
) -> tuple[list[list[float]], list[list[float]], list[list[float]]]:
    return (
        [list(row.get("observation.state") or []) for row in rows],
        [list(row.get("action") or []) for row in rows],
        [list(row.get("box.timestamps") or []) for row in rows],
    )


def _box_rows_from_snapshots(
    meta: dict[str, Any],
    fps: int,
    n_frames: int,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[str], list[str]] | None:
    snapshots = _box_snapshots_from_meta(meta)
    if not snapshots:
        return None
    box_ids = lr3.box_ids_from_snapshots(snapshots)
    rows = lr3._build_episode_rows(
        fps=fps,
        episode_index=0,
        snapshots=snapshots,
        duration_s=n_frames / max(int(fps), 1),
        box_ids=box_ids,
    )
    if not rows:
        return None
    state_rows, action_rows, ts_rows = _rows_from_lr3_rows(rows)
    return state_rows, action_rows, ts_rows, list(lr3.box_state_names(box_ids)), list(lr3.box_timestamp_names(box_ids))


def _box_rows_from_raw(
    ep_dir: Path,
    meta: dict[str, Any],
    fps: int,
    n_frames: int,
) -> tuple[list[list[float]], list[list[float]], list[list[float]], list[str], list[str]] | None:
    """Fallback: re-derive box state on the camera frame grid from the raw
    per-sensor ``box_sensors.jsonl`` when no session box parquet exists.

    Mirrors the recorder's L3b alignment (ts_sync.md §5/§6): per-sensor MCU-clock
    calibration, then nearest-neighbour selection at each grid time
    ``pts_offset + N/fps`` for N in ``range(n_frames)``. Returns canonical-width
    ``(state_rows, action_rows, ts_rows)`` or ``None`` when the file is absent or
    has no usable samples.
    """
    jsonl = ep_dir / "box_sensors.jsonl"
    if not jsonl.is_file():
        return None
    sensor_samples: dict[str, list[dict[str, Any]]] = {}
    try:
        with jsonl.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                rec = json.loads(line)
                sid = rec.get("sid")
                if not sid:
                    continue
                sensor_samples.setdefault(sid, []).append(
                    {
                        "data": rec.get("data", {}),
                        "t_rel_s": float(rec.get("t_rel_s", 0.0)),
                        "wall_s": float(rec.get("wall_s", 0.0)),
                    }
                )
    except (OSError, json.JSONDecodeError, ValueError, TypeError):
        return None
    if not any(sensor_samples.values()):
        return None

    sync_reference = meta.get("sync_reference") if isinstance(meta, dict) else {}
    t0_wall_s = float((sync_reference or {}).get("t0_wall_s") or 0.0)
    pts_offset = _pts_offset_s(meta)

    box_ids = lr3.box_ids_from_sensor_samples(sensor_samples)
    rows = lr3._build_episode_rows(
        fps=fps,
        episode_index=0,
        snapshots=[],
        duration_s=n_frames / max(int(fps), 1),
        sensor_samples=sensor_samples,
        t0_wall_s=t0_wall_s,
        pts_offset_s=pts_offset,
        box_ids=box_ids,
    )
    if not rows:
        return None
    state_rows, action_rows, ts_rows = _rows_from_lr3_rows(rows)
    return state_rows, action_rows, ts_rows, list(lr3.box_state_names(box_ids)), list(lr3.box_timestamp_names(box_ids))


# -------------------------------------------------------------- transcoding ---


def _gst_transcode_to_h264(src_mkv: Path, dst_mp4: Path, src_codec: str, fps: int) -> bool:
    """HEVC/H264 MKV -> CFR H.264 MP4 via the Jetson nvv4l2 pipeline (Thor).

    ``videorate`` forces a constant frame rate so the output PTS land exactly on
    the ``i/fps`` grid. This is essential: the v3 loader queries every camera of
    an episode with a single timestamp, so all cameras must share that grid (the
    GMSL2 frames are PWM-locked 60fps; only the recorded container PTS jitter).
    """
    if shutil.which("gst-launch-1.0") is None:
        return False
    parse = "h265parse" if src_codec == "h265" else "h264parse"
    cmd = [
        "gst-launch-1.0", "-q",
        "filesrc", f"location={src_mkv}",
        "!", "matroskademux",
        "!", parse,
        "!", "nvv4l2decoder",
        "!", "nvvidconv",
        "!", "videorate",
        "!", f"video/x-raw,framerate={fps}/1",
        "!", "nvvidconv",
        "!", "video/x-raw(memory:NVMM)",
        "!", "nvv4l2h264enc", "bitrate=10000000", "insert-sps-pps=1",
        "!", "h264parse", "config-interval=-1",
        "!", "mp4mux", "faststart=true",
        "!", "filesink", f"location={dst_mp4}",
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0 and dst_mp4.is_file() and dst_mp4.stat().st_size > 0


def _ffmpeg_transcode_to_h264(src_mkv: Path, dst_mp4: Path, fps: int) -> bool:
    """CFR H.264 fallback for dev hosts without nvv4l2 (uses libx264).

    ``-vf fps`` resamples to a constant frame rate so PTS == i/fps (same grid
    contract as the gst path above)."""
    if shutil.which("ffmpeg") is None:
        return False
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", str(src_mkv),
        "-vf", f"fps={fps}",
        "-fps_mode", "cfr",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-an",
        "-movflags", "+faststart",
        str(dst_mp4),
    ]
    try:
        result = subprocess.run(cmd, capture_output=True, timeout=180, check=False)
    except (subprocess.TimeoutExpired, OSError):
        return False
    return result.returncode == 0 and dst_mp4.is_file() and dst_mp4.stat().st_size > 0


def transcode_to_h264_mp4(src_mkv: Path, dst_mp4: Path, src_codec: str, fps: int) -> None:
    dst_mp4.parent.mkdir(parents=True, exist_ok=True)
    if dst_mp4.exists():
        dst_mp4.unlink()
    if _gst_transcode_to_h264(src_mkv, dst_mp4, src_codec, fps):
        return
    if _ffmpeg_transcode_to_h264(src_mkv, dst_mp4, fps):
        return
    raise RuntimeError(
        f"could not transcode {src_mkv.name} to H.264 (need gst-launch-1.0 with "
        "nvv4l2, or ffmpeg with libx264)"
    )


def _mkv_frame_count(mkv_path: Path) -> int:
    """Camera frame count, read from the source MKV's demuxed packet PTS.

    ``extract_pts`` reads container PTS without decoding (fast) and works on the
    Thor minimal env via gstreamer. The CFR transcode is 1:1 for the PWM-locked
    60fps source, so this also bounds the output mp4's frame count.
    """
    try:
        return len(lr3.extract_pts(mkv_path))
    except Exception:  # noqa: BLE001
        return 0


# ----------------------------------------------------------- tracking poses ---


def _parse_float(value: Any, default: float = math.nan) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return default
    return out if math.isfinite(out) else default


def _parse_int(value: Any, default: int = -1) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _pose_feature() -> dict[str, Any]:
    return {"dtype": "float32", "shape": [7], "names": list(_POSE7_FEATURE_NAMES)}


def _pose7_from_row(row: dict[str, Any], prefix: str) -> list[float] | None:
    values = [_parse_float(row.get(f"{prefix}_{suffix}")) for suffix in _POSE7_NAMES]
    if not all(math.isfinite(value) for value in values):
        return None
    return values


def _read_pose_csv(
    csv_path: Path,
    *,
    episode_index: int,
    prefix: str,
    cube_name: str | None = None,
) -> dict[int, list[float]]:
    poses: dict[int, list[float]] = {}
    try:
        with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            for row in reader:
                if cube_name is not None and row.get("cube_name") not in (None, "", cube_name):
                    continue
                if _parse_int(row.get("episode_index"), 0) != int(episode_index):
                    continue
                frame_index = _parse_int(row.get("frame_index"), -1)
                if frame_index < 0:
                    continue
                pose = _pose7_from_row(row, prefix)
                if pose is not None:
                    poses[frame_index] = pose
    except OSError:
        return {}
    return poses


def _pose_rows_for_frames(poses_by_frame: dict[int, list[float]], n_frames: int) -> list[list[float]]:
    return [list(poses_by_frame.get(frame_index, _POSE7_NAN)) for frame_index in range(int(n_frames))]


def _sidecar_dir(session_dir: Path) -> Path:
    return session_dir / "derived" / _TRAJ_SIDECAR_NAME


def _tracking_run_dir(session_dir: Path) -> Path:
    return _REPO_ROOT / "outputs" / "tracking_analysis" / f"{session_dir.name}{_TRACKING_RUN_SUFFIX}"


def _state_action_csv(session_dir: Path, cube: str) -> Path:
    return _sidecar_dir(session_dir) / f"state_action.{cube}.csv"


def _cube_camera_csv(session_dir: Path, cube: str, camera: str) -> Path:
    return _sidecar_dir(session_dir) / f"cube_pose.{cube}.{camera}.csv"


def _cube_pose_csvs(session_dir: Path, cube: str) -> list[Path]:
    return sorted(_sidecar_dir(session_dir).glob(f"cube_pose.{cube}.*.csv"))


def _fused_cube_csv_candidates(session_dir: Path, cube: str) -> list[Path]:
    run_dir = _tracking_run_dir(session_dir)
    return [
        run_dir / f"fused_ee_pose_in_robot_base_records_{cube}.csv",
        run_dir / "fused_ee_pose_in_robot_base_records.csv",
    ]


def _fused_cube_csv(session_dir: Path, cube: str) -> Path | None:
    for candidate in _fused_cube_csv_candidates(session_dir, cube):
        if candidate.is_file():
            return candidate
    return None


def _read_cube_base_from_camera_csvs(session_dir: Path, *, cube: str, episode_index: int) -> dict[int, list[float]]:
    poses: dict[int, list[float]] = {}
    used_for_fusion: dict[int, bool] = {}
    for csv_path in _cube_pose_csvs(session_dir, cube):
        try:
            with csv_path.open("r", encoding="utf-8", newline="") as csv_file:
                reader = csv.DictReader(csv_file)
                for row in reader:
                    if row.get("cube_name") not in (None, "", cube):
                        continue
                    if _parse_int(row.get("episode_index"), 0) != int(episode_index):
                        continue
                    frame_index = _parse_int(row.get("frame_index"), -1)
                    if frame_index < 0:
                        continue
                    pose = _pose7_from_row(row, "cube_base")
                    if pose is None:
                        continue
                    row_used = str(row.get("used_for_fusion", "")).strip().lower() in {"1", "true", "yes", "y"}
                    if frame_index not in poses or (row_used and not used_for_fusion.get(frame_index, False)):
                        poses[frame_index] = pose
                        used_for_fusion[frame_index] = row_used
        except OSError:
            continue
    return poses


def _discover_tracking_pose_columns(sessions: list[Path]) -> list[_PoseColumn]:
    columns: dict[str, _PoseColumn] = {}
    for session_dir in sessions:
        sidecar = _sidecar_dir(session_dir)
        if not sidecar.is_dir():
            continue
        for state_action_csv in sorted(sidecar.glob("state_action.*.csv")):
            cube = state_action_csv.name.removeprefix("state_action.").removesuffix(".csv")
            if not cube:
                continue
            columns.setdefault(
                f"observation.ee_pose.{cube}.base",
                _PoseColumn(f"observation.ee_pose.{cube}.base", "ee_state", cube),
            )
            columns.setdefault(
                f"action.ee_pose.{cube}.base",
                _PoseColumn(f"action.ee_pose.{cube}.base", "ee_action", cube),
            )
            if _fused_cube_csv(session_dir, cube) is not None or _cube_pose_csvs(session_dir, cube):
                columns.setdefault(
                    f"observation.cube_pose.{cube}.base",
                    _PoseColumn(f"observation.cube_pose.{cube}.base", "cube_base", cube),
                )
        for cube_pose_csv in sorted(sidecar.glob("cube_pose.*.*.csv")):
            stem = cube_pose_csv.name.removeprefix("cube_pose.").removesuffix(".csv")
            cube, sep, camera = stem.partition(".")
            if not sep or not cube or not camera:
                continue
            columns.setdefault(
                f"observation.cube_pose.{cube}.camera.{camera}",
                _PoseColumn(f"observation.cube_pose.{cube}.camera.{camera}", "cube_camera", cube, camera),
            )
    return [columns[key] for key in sorted(columns)]


def _load_tracking_pose_rows(
    src: EpisodeSource,
    *,
    pose_columns: list[_PoseColumn],
    n_frames: int,
) -> dict[str, list[list[float]]]:
    out: dict[str, list[list[float]]] = {}
    cache: dict[tuple[Path, str, str | None], dict[int, list[float]]] = {}
    for col in pose_columns:
        csv_path: Path | None = None
        prefix = ""
        cube_filter: str | None = None
        if col.kind == "ee_state":
            csv_path = _state_action_csv(src.session_dir, col.cube)
            prefix = "state"
        elif col.kind == "ee_action":
            csv_path = _state_action_csv(src.session_dir, col.cube)
            prefix = "action"
        elif col.kind == "cube_camera" and col.camera is not None:
            csv_path = _cube_camera_csv(src.session_dir, col.cube, col.camera)
            prefix = "cube_cam"
        elif col.kind == "cube_base":
            csv_path = _fused_cube_csv(src.session_dir, col.cube)
            prefix = "cube_base"
            cube_filter = col.cube
            if csv_path is None or not csv_path.is_file():
                poses = _read_cube_base_from_camera_csvs(
                    src.session_dir,
                    cube=col.cube,
                    episode_index=src.local_index,
                )
                out[col.key] = _pose_rows_for_frames(poses, n_frames)
                continue
        if csv_path is None or not csv_path.is_file():
            out[col.key] = _pose_rows_for_frames({}, n_frames)
            continue
        cache_key = (csv_path, prefix, cube_filter)
        if cache_key not in cache:
            cache[cache_key] = _read_pose_csv(
                csv_path,
                episode_index=src.local_index,
                prefix=prefix,
                cube_name=cube_filter,
            )
        out[col.key] = _pose_rows_for_frames(cache[cache_key], n_frames)
    return out


def _pose_table_column_stats(table, col_name: str) -> dict[str, list[Any]]:
    import numpy as np

    arr = table[col_name]
    n = arr.length()
    if n == 0:
        empty = [None] * 7
        return {
            "min": empty, "max": empty, "mean": empty, "std": empty,
            "count": [0] * 7,
            "q01": empty, "q10": empty, "q50": empty, "q90": empty, "q99": empty,
        }
    flat = arr.combine_chunks().flatten().to_numpy(zero_copy_only=False).astype(np.float64, copy=False)
    np_arr = flat.reshape(n, 7)
    stat_names = ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99")
    out: dict[str, list[Any]] = {name: [] for name in stat_names}
    for col in range(7):
        values = np_arr[:, col]
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            for name in ("min", "max", "mean", "std", "q01", "q10", "q50", "q90", "q99"):
                out[name].append(None)
            out["count"].append(0)
            continue
        quantiles = np.percentile(finite, [1.0, 10.0, 50.0, 90.0, 99.0])
        out["min"].append(float(np.min(finite)))
        out["max"].append(float(np.max(finite)))
        out["mean"].append(float(np.mean(finite)))
        out["std"].append(float(np.std(finite)))
        out["count"].append(int(finite.size))
        out["q01"].append(float(quantiles[0]))
        out["q10"].append(float(quantiles[1]))
        out["q50"].append(float(quantiles[2]))
        out["q90"].append(float(quantiles[3]))
        out["q99"].append(float(quantiles[4]))
    return out


# ---------------------------------------------------------------- v3 writer ---


@dataclass
class _VideoKey:
    feature: str  # observation.images.cam_00
    camera: str  # cam_00


@dataclass(frozen=True)
class _PoseColumn:
    key: str
    kind: str
    cube: str
    camera: str | None = None


class _V3Writer:
    """pyarrow-only LeRobot v3 writer: numeric features in parquet, per-episode
    H.264 mp4 video files, hand-written episodes/info/stats/tasks meta."""

    def __init__(
        self,
        dataset_root: Path,
        *,
        repo_id: str,
        task: str,
        fps: int,
        height: int,
        width: int,
        video_keys: list[_VideoKey],
        state_width: int,
        state_names: list[str] | None,
        ts_width: int = 0,
        ts_names: list[str] | None = None,
        pose_columns: list[_PoseColumn] | None = None,
    ) -> None:
        import pyarrow as pa
        import pyarrow.parquet as pq

        self.pa = pa
        self.pq = pq
        self.root = dataset_root
        self.repo_id = repo_id
        self.task = task
        self.fps = int(fps)
        self.height = height
        self.width = width
        self.video_keys = video_keys
        self.state_width = state_width
        self.state_names = state_names
        self.ts_width = ts_width
        self.ts_names = ts_names
        self.pose_columns = list(pose_columns or [])

        self.meta_dir = dataset_root / "meta"
        self.episodes_dir = self.meta_dir / "episodes" / "chunk-000"
        self.data_path = dataset_root / "data" / "chunk-000" / "file-000.parquet"
        self.data_path.parent.mkdir(parents=True, exist_ok=True)
        self.episodes_dir.mkdir(parents=True, exist_ok=True)

        self.total_frames = 0
        self._episode_rows: list[dict[str, Any]] = []
        self._schema = self._build_schema()
        self._writer = pq.ParquetWriter(self.data_path, schema=self._schema, compression="snappy")

    def _build_schema(self):
        pa = self.pa
        fields = []
        if self.state_width > 0:
            fields.append(("observation.state", pa.list_(pa.float32(), self.state_width)))
            fields.append(("action", pa.list_(pa.float32(), self.state_width)))
        if self.ts_width > 0:
            fields.append(("box.timestamps", pa.list_(pa.float64(), self.ts_width)))
        for pose_col in self.pose_columns:
            fields.append((pose_col.key, pa.list_(pa.float32(), 7)))
        fields += [
            ("timestamp", pa.float32()),
            ("frame_index", pa.int64()),
            ("episode_index", pa.int64()),
            ("index", pa.int64()),
            ("task_index", pa.int64()),
        ]
        return pa.schema(fields)

    def append_episode(
        self,
        *,
        episode_index: int,
        n_frames: int,
        state_rows: list[list[float]] | None,
        action_rows: list[list[float]] | None,
        video_files: dict[str, Path],  # camera -> mp4 path already written
        ts_rows: list[list[float]] | None = None,
        pose_rows: dict[str, list[list[float]]] | None = None,
    ) -> None:
        start = self.total_frames
        cols: dict[str, list[Any]] = {
            "timestamp": [float(i) / self.fps for i in range(n_frames)],
            "frame_index": list(range(n_frames)),
            "episode_index": [episode_index] * n_frames,
            "index": list(range(start, start + n_frames)),
            "task_index": [0] * n_frames,
        }
        if self.state_width > 0:
            cols["observation.state"] = state_rows or [[0.0] * self.state_width] * n_frames
            cols["action"] = action_rows or [[0.0] * self.state_width] * n_frames
        if self.ts_width > 0:
            cols["box.timestamps"] = ts_rows or [[0.0] * self.ts_width] * n_frames
        pose_rows = pose_rows or {}
        for pose_col in self.pose_columns:
            rows = pose_rows.get(pose_col.key)
            if rows is None:
                rows = _pose_rows_for_frames({}, n_frames)
            if len(rows) != n_frames:
                raise ValueError(f"{pose_col.key} row count mismatch: {len(rows)} != {n_frames}")
            cols[pose_col.key] = rows
        self._writer.write_table(self.pa.table(cols, schema=self._schema))

        stop = start + n_frames
        self.total_frames = stop
        duration = n_frames / self.fps
        row: dict[str, Any] = {
            "episode_index": int(episode_index),
            "tasks": [self.task],
            "length": int(n_frames),
            "data/chunk_index": 0,
            "data/file_index": 0,
            "dataset_from_index": int(start),
            "dataset_to_index": int(stop),
            "meta/episodes/chunk_index": 0,
            "meta/episodes/file_index": 0,
        }
        for vk in self.video_keys:
            row[f"videos/{vk.feature}/chunk_index"] = 0
            row[f"videos/{vk.feature}/file_index"] = int(episode_index)
            row[f"videos/{vk.feature}/from_timestamp"] = 0.0
            row[f"videos/{vk.feature}/to_timestamp"] = float(duration)
        self._episode_rows.append(row)

    def finalize(self) -> None:
        self._writer.close()
        self._write_episodes()
        self._write_tasks()
        self._write_stats()
        self._write_info()

    # -- meta writers --

    def _features(self) -> dict[str, Any]:
        features: dict[str, Any] = {}
        if self.state_width > 0:
            names = self.state_names if self.state_names and len(self.state_names) == self.state_width else None
            features["observation.state"] = lr3._feature("float32", [self.state_width], names)
            features["action"] = lr3._feature("float32", [self.state_width], names)
        if self.ts_width > 0:
            ts_names = self.ts_names if self.ts_names and len(self.ts_names) == self.ts_width else None
            features["box.timestamps"] = lr3._feature("float64", [self.ts_width], ts_names)
        for pose_col in self.pose_columns:
            features[pose_col.key] = _pose_feature()
        for vk in self.video_keys:
            features[vk.feature] = {
                "dtype": "video",
                "shape": [self.height, self.width, 3],
                "names": ["height", "width", "channels"],
                "info": {
                    "video.height": self.height,
                    "video.width": self.width,
                    "video.codec": "h264",
                    "video.pix_fmt": "yuv420p",
                    "video.is_depth_map": False,
                    "video.fps": self.fps,
                    "video.channels": 3,
                    "has_audio": False,
                },
            }
        features["timestamp"] = lr3._feature("float32", [1])
        features["frame_index"] = lr3._feature("int64", [1])
        features["episode_index"] = lr3._feature("int64", [1])
        features["index"] = lr3._feature("int64", [1])
        features["task_index"] = lr3._feature("int64", [1])
        return features

    def _write_episodes(self) -> None:
        rows = sorted(self._episode_rows, key=lambda r: int(r["episode_index"]))
        self.pq.write_table(self.pa.Table.from_pylist(rows), self.episodes_dir / "file-000.parquet")

    def _write_tasks(self) -> None:
        self.pq.write_table(
            self.pa.Table.from_pylist([{"task_index": 0, "task": self.task}]),
            self.meta_dir / "tasks.parquet",
        )

    def _write_stats(self) -> None:
        table = self.pq.read_table(self.data_path)
        stats: dict[str, Any] = {
            "timestamp": lr3._table_column_stats(table, "timestamp", width=1),
            "frame_index": lr3._table_column_stats(table, "frame_index", width=1),
            "episode_index": lr3._table_column_stats(table, "episode_index", width=1),
            "index": lr3._table_column_stats(table, "index", width=1),
            "task_index": lr3._table_column_stats(table, "task_index", width=1),
        }
        if self.state_width > 0:
            stats["observation.state"] = lr3._table_column_stats(table, "observation.state", width=self.state_width)
            stats["action"] = lr3._table_column_stats(table, "action", width=self.state_width)
        if self.ts_width > 0:
            stats["box.timestamps"] = lr3._table_column_stats(table, "box.timestamps", width=self.ts_width)
        for pose_col in self.pose_columns:
            stats[pose_col.key] = _pose_table_column_stats(table, pose_col.key)
        (self.meta_dir / "stats.json").write_text(json.dumps(stats, indent=4), encoding="utf-8")

    def _write_info(self) -> None:
        n_eps = len(self._episode_rows)
        info = {
            "codebase_version": "v3.0",
            "robot_type": "thor_gmsl2_box",
            "repo_id": self.repo_id,
            "total_episodes": int(n_eps),
            "total_frames": int(self.total_frames),
            "total_tasks": 1,
            "chunks_size": _CHUNKS_SIZE,
            "data_files_size_in_mb": 100,
            "video_files_size_in_mb": 200,
            "fps": int(self.fps),
            "splits": {"train": f"0:{n_eps}"},
            "data_path": _DATA_PATH,
            "video_path": _VIDEO_PATH if self.video_keys else None,
            "features": self._features(),
        }
        (self.meta_dir / "info.json").write_text(json.dumps(info, indent=4), encoding="utf-8")


# ------------------------------------------------------------------ export ---


def export_task_to_v3(
    *,
    datasets_root: Path,
    exports_root: Path,
    base_name: str,
    repo_id: str,
    task: str,
    overwrite: bool = False,
    jobs: int = _DEFAULT_JOBS,
) -> Path:
    name = base_name.split("/")[-1].strip()
    if not name:
        raise RuntimeError(f"Invalid task base name: {base_name!r}")
    sessions = find_task_sessions(datasets_root, name)
    if not sessions:
        raise RuntimeError(f"No recorded sessions found for '{name}' under {datasets_root}")
    episodes = gather_episodes(sessions)
    if not episodes:
        raise RuntimeError(f"Sessions for '{name}' contain no episodes")
    _emit(f"Export plan: {len(episodes)} episodes from {len(sessions)} session(s) -> {repo_id}")

    out_root = exports_root / name
    if out_root.exists():
        if not overwrite:
            raise RuntimeError(f"Output already exists: {out_root} (pass --overwrite to replace)")
        shutil.rmtree(out_root)

    first = episodes[0]
    first_meta = _load_meta(first.ep_dir)
    video = first_meta.get("video") or {}
    fps = int(video.get("fps") or 60)
    height = int(video.get("height") or 0)
    width = int(video.get("width") or 0)
    camera_entries = _camera_entries(first_meta, first.ep_dir)
    if not camera_entries:
        raise RuntimeError(f"No camera MKVs in {first.ep_dir}")
    if not (height and width):
        raise RuntimeError(f"meta.json for {first.ep_dir} lacks video.height/width")
    camera_keys = [name for name, _ in camera_entries]
    video_keys = [_VideoKey(feature=f"observation.images.{c}", camera=c) for c in camera_keys]
    pose_columns = _discover_tracking_pose_columns(sessions)
    if pose_columns:
        _emit(f"Tracking pose schema: {len(pose_columns)} column(s) from {_TRAJ_SIDECAR_NAME}")
    else:
        _emit(f"Tracking pose schema: no {_TRAJ_SIDECAR_NAME} sidecar found; exporting video/box only")

    box_state_names = list(lr3.BOX_STATE_NAMES)
    box_ts_names = list(lr3.BOX_TIMESTAMP_NAMES)
    first_box = _load_box_rows(first.session_dir)
    state_width = 0
    ts_width = 0

    first_snapshots = _box_snapshots_from_meta(first_meta)
    first_snapshot_box_ids = lr3.box_ids_from_snapshots(first_snapshots) if first_snapshots else ("",)
    if first_snapshots and first_snapshot_box_ids != ("",):
        box_state_names = list(lr3.box_state_names(first_snapshot_box_ids))
        box_ts_names = list(lr3.box_timestamp_names(first_snapshot_box_ids))
        state_width = len(box_state_names)
        ts_width = len(box_ts_names)

    # Prefer the recorder's already-synced session parquet when it already has
    # the selected schema. Multi-BOX snapshots win over old all-zero single-BOX
    # parquet so re-exporting affected sessions can recover visible BOX data.
    if state_width == 0 and first_box:
        sample_row = None
        if first.local_index in first_box and first_box[first.local_index]:
            sample_row = first_box[first.local_index][0]
        else:
            for rows in first_box.values():
                if rows:
                    sample_row = rows[0]
                    break
        if sample_row is not None:
            state_width = len(sample_row["observation.state"])
            ts_width = len(sample_row.get("box.timestamps", []))

    # No session parquet, but raw per-sensor samples exist -> export re-aligns
    # from box_sensors.jsonl using the canonical/dynamic box schema.
    if state_width == 0 and (first.ep_dir / "box_sensors.jsonl").is_file():
        state_width = len(box_state_names)
        ts_width = len(box_ts_names)
    has_box = state_width > 0

    _emit(f"Schema: {len(camera_keys)} camera(s) {width}x{height} @ {fps}fps, state_width={state_width}")

    writer = _V3Writer(
        out_root,
        repo_id=repo_id,
        task=task,
        fps=fps,
        height=height,
        width=width,
        video_keys=video_keys,
        state_width=state_width,
        state_names=box_state_names if state_width else None,
        ts_width=ts_width,
        ts_names=box_ts_names if ts_width else None,
        pose_columns=pose_columns,
    )

    box_cache: dict[Path, dict[int, list[dict[str, Any]]]] = {}
    sources: list[dict[str, Any]] = []
    global_index = 0
    for src in episodes:
        meta = _load_meta(src.ep_dir)
        cams = _camera_entries(meta, src.ep_dir)
        if [c for c, _ in cams] != camera_keys:
            _emit(f"WARNING: skipping {src.session_dir.name}/{src.ep_dir.name}: camera set mismatch")
            continue
        src_codec = str((meta.get("video") or {}).get("codec") or "h265").lower()

        if src.session_dir not in box_cache:
            box_cache[src.session_dir] = _load_box_rows(src.session_dir)

        # Per-camera frame counts can differ (uneven frame drops during capture).
        # One query timestamp serves all cameras, so cap at the *minimum* count.
        # The camera grid is authoritative: it defines how many frames the v3
        # loader can actually serve, and box samples are matched onto it.
        cam_counts = {cam: _mkv_frame_count(mkv) for cam, mkv in cams}
        cam_frame_count = min(cam_counts.values()) if cam_counts else 0
        if cam_frame_count <= 0:
            _emit(f"WARNING: skipping {src.session_dir.name}/{src.ep_dir.name}: could not read camera frame count")
            continue
        if max(cam_counts.values()) != cam_frame_count:
            _emit(f"  note: {src.ep_dir.name} camera frame counts vary {dict(sorted(cam_counts.items()))}; truncating to {cam_frame_count}")

        n_frames = cam_frame_count
        state_rows = action_rows = ts_rows = None
        if has_box:
            box_rows = box_cache[src.session_dir].get(src.local_index, [])
            parquet_width = len(box_rows[0].get("observation.state", [])) if box_rows else 0
            if box_rows and parquet_width == state_width:
                # Default: align the recorder's pre-synced box parquet to the
                # camera grid by frame_index (ts_sync.md §6), not list position.
                state_rows, action_rows, ts_rows, missing = _align_box_rows_by_frame_index(
                    box_rows, n_frames, state_width, ts_width
                )
                box_grid_len = max(int(r["frame_index"]) for r in box_rows) + 1
                if missing:
                    _emit(
                        f"  note: {src.ep_dir.name} {missing}/{n_frames} camera frames had no box "
                        f"row (box grid {box_grid_len} vs camera {n_frames}); held last reading"
                    )
                elif box_grid_len != n_frames:
                    _emit(
                        f"  note: {src.ep_dir.name} box grid {box_grid_len} vs camera {n_frames}; "
                        f"aligned by frame_index"
                    )
            else:
                # Fallback: no usable session parquet rows for this episode ->
                # re-align raw samples, then fall back to low-rate meta snapshots.
                raw = _box_rows_from_raw(src.ep_dir, meta, fps, n_frames)
                if (
                    raw is not None
                    and len(raw[0][0]) == state_width
                    and (ts_width == 0 or len(raw[2][0]) == ts_width)
                ):
                    state_rows, action_rows, raw_ts, _raw_state_names, _raw_ts_names = raw
                    ts_rows = raw_ts if ts_width else None
                    _emit(
                        f"  note: {src.ep_dir.name} re-aligned {n_frames} frames "
                        f"from box_sensors.jsonl"
                    )
                else:
                    snap = _box_rows_from_snapshots(meta, fps, n_frames)
                    if (
                        snap is not None
                        and len(snap[0][0]) == state_width
                        and (ts_width == 0 or len(snap[2][0]) == ts_width)
                    ):
                        state_rows, action_rows, snap_ts, _snap_state_names, _snap_ts_names = snap
                        ts_rows = snap_ts if ts_width else None
                        _emit(
                            f"  note: {src.ep_dir.name} used BOX meta snapshots for {n_frames} frames"
                        )
                    else:
                        _emit(
                            f"WARNING: skipping {src.session_dir.name}/{src.ep_dir.name}: "
                            f"no usable box parquet, box_sensors.jsonl, or BOX snapshots"
                        )
                        continue

        # Transcode each camera's clip to a per-episode CFR H.264 mp4 (PTS=i/fps).
        # Cameras are independent nvv4l2 jobs, so run them concurrently (the
        # recorder already drives 10+ parallel nvv4l2 streams) — this is the main
        # speedup over the old one-clip-at-a-time loop.
        def _transcode_cam(cam_mkv: tuple[str, Path]) -> tuple[str, Path]:
            cam, mkv = cam_mkv
            feature = f"observation.images.{cam}"
            dst = out_root / _VIDEO_PATH.format(video_key=feature, chunk_index=0, file_index=global_index)
            transcode_to_h264_mp4(mkv, dst, src_codec, fps)
            return cam, dst

        video_files: dict[str, Path] = {}
        with ThreadPoolExecutor(max_workers=max(1, jobs)) as pool:
            for cam, dst in pool.map(_transcode_cam, cams):
                video_files[cam] = dst

        pose_rows = (
            _load_tracking_pose_rows(src, pose_columns=pose_columns, n_frames=n_frames)
            if pose_columns
            else None
        )
        writer.append_episode(
            episode_index=global_index,
            n_frames=n_frames,
            state_rows=state_rows,
            action_rows=action_rows,
            video_files=video_files,
            ts_rows=ts_rows,
            pose_rows=pose_rows,
        )
        sources.append(
            {
                "global_episode_index": global_index,
                "session": src.session_dir.name,
                "source_episode": src.ep_dir.name,
                "frames": n_frames,
            }
        )
        _emit(f"Episode {global_index} written ({n_frames} frames) from {src.session_dir.name}/{src.ep_dir.name}")
        global_index += 1

    if global_index == 0:
        raise RuntimeError("No episodes were written (all skipped); nothing to export")

    writer.finalize()
    (out_root / "meta" / "export_sources.json").write_text(
        json.dumps({"repo_id": repo_id, "task": task, "base_name": name, "episodes": sources}, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
    _emit(f"Export complete: {global_index} episodes at {out_root}")
    return out_root


# -------------------------------------------------------------------- main ---


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Consolidate a task's GMSL2 sessions into one LeRobot v3 dataset")
    ap.add_argument("--datasets-root", required=True, type=Path)
    ap.add_argument("--exports-root", required=True, type=Path)
    ap.add_argument("--base-name", required=True, help="task dataset base name, e.g. pick_and_place")
    ap.add_argument("--repo-id", required=True, help="e.g. local/pick_and_place")
    ap.add_argument("--task", required=True, help="single_task prompt string")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--jobs", type=int, default=_DEFAULT_JOBS, help="parallel camera transcodes")
    args = ap.parse_args(argv)

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s %(message)s", stream=sys.stderr)
    try:
        export_task_to_v3(
            datasets_root=args.datasets_root.resolve(),
            exports_root=args.exports_root.resolve(),
            base_name=args.base_name,
            repo_id=args.repo_id,
            task=args.task,
            overwrite=args.overwrite,
            jobs=args.jobs,
        )
    except Exception as exc:  # noqa: BLE001
        _emit(f"ERROR: {exc}")
        logger.exception("export failed")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
