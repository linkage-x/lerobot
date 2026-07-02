#!/usr/bin/env python
"""Prepare a selectable LeRobot dataset view and train ACT or Diffusion Policy.

This script is intentionally conservative: it never mutates the source dataset.
It writes a derived dataset view under outputs/datasets and launches the standard
LeRobot training entrypoint on that view.
"""

from __future__ import annotations

import argparse
import copy
import json
import re
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml


_REPO_ROOT = Path(__file__).resolve().parents[2]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

DEFAULT_DATASET_ROOT = Path("dataset_test/single_cube2_20260429_165325")
DEFAULT_CAMERAS = "observation.images.cam_1,observation.images.cam_3"
DEFAULT_STATE_KEYS = "observation.state"
DEFAULT_DERIVED_ACTION = Path("derived/hikon_cube_tracking_in_robot_base/action.npy")
DEFAULT_ACTION_APPEND_SELECTORS = "observation.state_raw:handheld_gripper.pika_left.width_mm"
DEFAULT_ACTION_APPEND_NAMES = "gripper"
DEFAULT_GMSL2_EXPORT_ROOT = Path("outputs/exported_datasets")


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    if value.strip().lower() in {"", "none", "null"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_camera_key(key: str) -> str:
    return key if key.startswith("observation.images.") else f"observation.images.{key}"


def load_json(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return json.load(f)


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        json.dump(value, f, indent=2)


def write_yaml(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as f:
        yaml.safe_dump(value, f, sort_keys=False)


def cli_arg_was_supplied(name: str) -> bool:
    return any(arg == name or arg.startswith(f"{name}=") for arg in sys.argv[1:])


def append_resume_override(cmd: list[str], helper_flag: str, config_key: str, value: Any) -> None:
    if cli_arg_was_supplied(helper_flag):
        cmd.append(f"--{config_key}={value}")


def resolve_resume_config_path(checkpoint: Path) -> Path:
    checkpoint = checkpoint.resolve()
    candidates = [
        checkpoint / "pretrained_model" / "train_config.json",
        checkpoint / "train_config.json",
    ]
    if checkpoint.name == "pretrained_model":
        candidates.insert(0, checkpoint / "train_config.json")
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        f"Could not find train_config.json for resume under {checkpoint}. "
        "Pass a checkpoint directory such as outputs/train/<job_name>/checkpoints/last."
    )


def feature_dim(feature: dict[str, Any]) -> int:
    shape = feature.get("shape")
    if not isinstance(shape, list) or len(shape) != 1:
        raise ValueError(f"Expected a 1D feature, got shape={shape}")
    return int(shape[0])


def as_matrix(series: pd.Series, key: str, dim: int) -> np.ndarray:
    values = np.asarray(series.to_list(), dtype=np.float32)
    if values.ndim == 1 and dim == 1:
        values = values.reshape(-1, 1)
    if values.ndim != 2 or values.shape[1] != dim:
        raise ValueError(f"{key} has array shape {values.shape}, expected (*, {dim})")
    return values


def shifted_series_within_episode(df: pd.DataFrame, key: str, shift: int) -> pd.Series:
    if shift == 0:
        return df[key]
    shifted = df.groupby("episode_index", sort=False)[key].shift(-shift)
    return shifted.where(shifted.notna(), df[key])


def parse_feature_selector(selector: str, features: dict[str, Any]) -> tuple[str, int | None, str | None]:
    if ":" in selector:
        key, dim_selector = selector.split(":", 1)
    else:
        key, dim_selector = selector, None

    if key not in features:
        raise KeyError(f"Feature not found: {key}")

    if dim_selector is None or dim_selector == "":
        return key, None, None

    names = features[key].get("names") or []
    if dim_selector.isdigit():
        index = int(dim_selector)
        if index >= feature_dim(features[key]):
            raise IndexError(f"{selector} index {index} out of range")
        name = names[index] if index < len(names) else str(index)
        return key, index, name

    if dim_selector not in names:
        raise KeyError(f"{dim_selector} not found in feature names for {key}: {names}")
    return key, names.index(dim_selector), dim_selector


def action_append_dim_names(features: dict[str, Any], selectors: list[str]) -> list[str]:
    if not selectors:
        return []

    names = []
    for selector in selectors:
        key, index, dim_name = parse_feature_selector(selector, features)
        dim = feature_dim(features[key])
        ft_names = features[key].get("names") or [str(i) for i in range(dim)]
        if index is None:
            names.extend([f"{key}.{name}" for name in ft_names])
            continue
        names.append(f"{key}.{dim_name}")
    return names


def select_action_append_matrix(
    df: pd.DataFrame,
    features: dict[str, Any],
    selectors: list[str],
    shift: int,
) -> np.ndarray | None:
    if not selectors:
        return None

    parts = []
    for selector in selectors:
        key, index, _ = parse_feature_selector(selector, features)
        dim = feature_dim(features[key])
        values = as_matrix(shifted_series_within_episode(df, key, shift), key, dim)
        if index is None:
            parts.append(values)
            continue

        parts.append(values[:, index : index + 1])

    return np.concatenate(parts, axis=1).astype(np.float32)


def vector_stats(values: np.ndarray) -> dict[str, Any]:
    return {
        "min": values.min(axis=0).tolist(),
        "max": values.max(axis=0).tolist(),
        "mean": values.mean(axis=0).tolist(),
        "std": values.std(axis=0).tolist(),
        "count": [int(values.shape[0])],
        "q01": np.quantile(values, 0.01, axis=0).tolist(),
        "q10": np.quantile(values, 0.10, axis=0).tolist(),
        "q50": np.quantile(values, 0.50, axis=0).tolist(),
        "q90": np.quantile(values, 0.90, axis=0).tolist(),
        "q99": np.quantile(values, 0.99, axis=0).tolist(),
    }


def fill_nonfinite_matrix_within_episode(
    values: np.ndarray,
    episode_indices: pd.Series,
    *,
    label: str,
) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if np.isfinite(values).all():
        return values

    repaired = pd.DataFrame(values).replace([np.inf, -np.inf], np.nan)
    repaired["__episode_index"] = episode_indices.to_numpy()
    value_columns = [col for col in repaired.columns if col != "__episode_index"]
    grouped_values = repaired.groupby("__episode_index", sort=False)[value_columns]
    repaired_values_df = grouped_values.ffill()
    repaired_values_df = repaired_values_df.groupby(repaired["__episode_index"], sort=False).bfill()
    repaired_values = repaired_values_df.to_numpy(dtype=np.float32)
    if not np.isfinite(repaired_values).all():
        bad_rows, bad_cols = np.where(~np.isfinite(repaired_values))
        examples = [
            {"row": int(row), "dim": int(col), "episode_index": int(episode_indices.iloc[row])}
            for row, col in zip(bad_rows[:10], bad_cols[:10], strict=False)
        ]
        raise ValueError(f"{label} contains non-finite values that cannot be repaired: {examples}")

    repaired_count = int((~np.isfinite(values)).sum())
    print(f"[prepare] repaired {repaired_count} non-finite values in {label} via per-episode ffill/bfill")
    return repaired_values


def unrepairable_nonfinite_episodes(values: np.ndarray, episode_indices: pd.Series) -> set[int]:
    values = np.asarray(values, dtype=np.float32)
    if values.size == 0 or np.isfinite(values).all():
        return set()

    bad: set[int] = set()
    episodes = episode_indices.to_numpy()
    for episode in pd.unique(episode_indices):
        mask = episodes == episode
        repaired = pd.DataFrame(values[mask]).replace([np.inf, -np.inf], np.nan).ffill().bfill()
        if not np.isfinite(repaired.to_numpy(dtype=np.float32)).all():
            bad.add(int(episode))
    return bad


def copy_or_symlink_file(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def discover_dataset_roots(dataset_root: Path) -> list[Path]:
    dataset_root = dataset_root.resolve()
    if (dataset_root / "meta/info.json").is_file():
        return [dataset_root]

    roots = [
        child.resolve()
        for child in sorted(dataset_root.iterdir())
        if child.is_dir() and (child / "meta/info.json").is_file()
    ]
    if roots:
        return roots
    raise FileNotFoundError(
        f"{dataset_root} is neither a LeRobot dataset root nor a directory containing dataset roots."
    )


def dataset_has_training_contract(
    dataset_root: Path,
    *,
    camera_keys: list[str],
    state_keys: list[str],
    action_key: str,
    action_npy: Path | None,
    action_append_selectors: list[str],
) -> tuple[bool, str]:
    try:
        roots = discover_dataset_roots(dataset_root)
    except FileNotFoundError as exc:
        return False, str(exc)

    for root in roots:
        info_path = root / "meta/info.json"
        if not info_path.is_file():
            return False, f"missing {info_path}"
        try:
            features = load_json(info_path)["features"]
        except (KeyError, json.JSONDecodeError, OSError) as exc:
            return False, f"could not read features from {info_path}: {exc}"

        for key in camera_keys:
            feature = features.get(key)
            if not isinstance(feature, dict) or feature.get("dtype") not in ("video", "image"):
                return False, f"camera feature not found or not decodable image/video in {root}: {key}"
        for selector in state_keys:
            try:
                key, _, _ = parse_feature_selector(selector, features)
            except (KeyError, IndexError, ValueError) as exc:
                return False, f"state selector unavailable in {root}: {selector} ({exc})"
            if not key.startswith("observation.") or features[key].get("dtype") not in ("float32", "float64"):
                return False, f"state selector is not a numeric observation in {root}: {selector}"
        if action_npy is None:
            if action_key not in features:
                return False, f"action feature not found in {root}: {action_key}"
            try:
                feature_dim(features[action_key])
            except ValueError as exc:
                return False, f"action feature has unsupported shape in {root}: {action_key} ({exc})"
        for selector in action_append_selectors:
            try:
                parse_feature_selector(selector, features)
            except (KeyError, IndexError, ValueError) as exc:
                return False, f"action append selector unavailable in {root}: {selector} ({exc})"
    return True, "ok"


def looks_like_gmsl2_raw_dataset(dataset_root: Path) -> bool:
    episodes = dataset_root / "episodes"
    return episodes.is_dir() and any(episodes.glob("episode_*/meta.json")) and any(episodes.glob("episode_*/*.mkv"))


def _parse_rate(value: str) -> float | None:
    if "/" in value:
        num, den = value.split("/", 1)
        try:
            den_f = float(den)
            return float(num) / den_f if den_f else None
        except ValueError:
            return None
    try:
        return float(value)
    except ValueError:
        return None


def video_file_has_cfr_grid(video_path: Path, fps: int) -> tuple[bool, str]:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=avg_frame_rate,r_frame_rate,nb_frames,duration",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads(result.stdout)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError) as exc:
        return False, f"ffprobe failed for {video_path}: {exc}"

    streams = info.get("streams") or []
    if not streams:
        return False, f"no video stream in {video_path}"
    stream = streams[0]
    avg = _parse_rate(str(stream.get("avg_frame_rate", "")))
    nominal = _parse_rate(str(stream.get("r_frame_rate", "")))
    try:
        duration = float(stream.get("duration"))
        nb_frames = int(stream.get("nb_frames"))
    except (TypeError, ValueError):
        return False, f"missing duration/nb_frames in {video_path}"

    if avg is None or abs(avg - fps) > 1e-6:
        return False, f"{video_path} avg_frame_rate={stream.get('avg_frame_rate')} expected {fps}/1"
    if nominal is None or abs(nominal - fps) > 1e-6:
        return False, f"{video_path} r_frame_rate={stream.get('r_frame_rate')} expected {fps}/1"
    if abs(duration - nb_frames / fps) > max(1e-4, 0.25 / fps):
        return False, f"{video_path} duration={duration} nb_frames={nb_frames} expected duration={nb_frames / fps}"
    return True, "ok"


def video_file_frame_count(video_path: Path) -> int | None:
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames,nb_frames",
                "-of",
                "json",
                str(video_path),
            ],
            check=True,
            capture_output=True,
            text=True,
        )
        info = json.loads(result.stdout)
        stream = (info.get("streams") or [{}])[0]
        for key in ("nb_read_frames", "nb_frames"):
            value = stream.get(key)
            if value not in (None, "N/A"):
                return int(value)
    except (OSError, subprocess.CalledProcessError, json.JSONDecodeError, ValueError):
        return None
    return None


def dataset_videos_have_cfr_grid(dataset_root: Path, camera_keys: list[str]) -> tuple[bool, str]:
    info = load_json(dataset_root / "meta/info.json")
    fps = int(info.get("fps") or 0)
    video_path_template = info.get("video_path")
    if fps <= 0 or not video_path_template:
        return False, f"missing fps/video_path in {dataset_root / 'meta/info.json'}"

    episodes_files = sorted((dataset_root / "meta/episodes").glob("*/*.parquet"))
    if not episodes_files:
        return False, f"missing episode metadata under {dataset_root / 'meta/episodes'}"
    episodes = pd.concat([pq.read_table(path).to_pandas() for path in episodes_files], ignore_index=True)
    for cam in camera_keys:
        chunk_col = f"videos/{cam}/chunk_index"
        file_col = f"videos/{cam}/file_index"
        if chunk_col not in episodes or file_col not in episodes:
            return False, f"missing video metadata columns for {cam}"
        pairs = sorted({(int(chunk), int(file)) for chunk, file in zip(episodes[chunk_col], episodes[file_col], strict=True)})
        for chunk_index, file_index in pairs:
            video_path = dataset_root / chunk_file_path(
                video_path_template,
                video_key=cam,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if not video_path.is_file():
                return False, f"missing video file: {video_path}"
            ok, reason = video_file_has_cfr_grid(video_path, fps)
            if not ok:
                return False, reason
            frame_count = video_file_frame_count(video_path)
            if frame_count is None:
                return False, f"could not read frame count for {video_path}"
            rows = episodes[
                (episodes[chunk_col].astype(int) == chunk_index)
                & (episodes[file_col].astype(int) == file_index)
            ]
            max_length = int(rows["length"].max())
            if max_length > frame_count:
                return (
                    False,
                    f"{video_path} has {frame_count} frames but episode metadata requires {max_length}",
                )
    return True, "ok"


def dataset_camera_shapes_match(
    dataset_root: Path,
    *,
    camera_keys: list[str],
    image_resize_shape: list[int] | None,
) -> tuple[bool, str]:
    if image_resize_shape is None:
        return True, "ok"
    info = load_json(dataset_root / "meta/info.json")
    features = info.get("features") or {}
    expected = [int(image_resize_shape[0]), int(image_resize_shape[1]), 3]
    for cam in camera_keys:
        shape = (features.get(cam) or {}).get("shape")
        if shape != expected:
            return False, f"{cam} shape={shape} expected {expected}"
    return True, "ok"


def optimized_lerobot_video_export_root(dataset_root: Path, export_root: Path, image_resize_shape: list[int]) -> Path:
    height, width = int(image_resize_shape[0]), int(image_resize_shape[1])
    return (export_root / f"{dataset_root.name}_{height}x{width}").resolve()


def lerobot_video_dataset_selected_cameras(dataset_root: Path, camera_keys: list[str]) -> bool:
    info_path = dataset_root / "meta/info.json"
    if not info_path.is_file():
        return False
    try:
        features = load_json(info_path)["features"]
    except (KeyError, json.JSONDecodeError, OSError):
        return False
    return all((features.get(key) or {}).get("dtype") == "video" for key in camera_keys)


def copy_lerobot_dataset_without_videos(src_root: Path, dst_root: Path) -> None:
    for child in src_root.iterdir():
        if child.name == "videos":
            continue
        dst_child = dst_root / child.name
        if child.is_dir():
            shutil.copytree(child, dst_child)
        else:
            dst_child.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(child, dst_child)


def optimize_lerobot_video_dataset(
    *,
    src_root: Path,
    dst_root: Path,
    camera_keys: list[str],
    image_resize_shape: list[int],
    jobs: int,
) -> Path:
    from tools.thor.gmsl2.export_v3 import transcode_to_h264_mp4

    if dst_root.exists():
        shutil.rmtree(dst_root)
    dst_root.mkdir(parents=True)
    copy_lerobot_dataset_without_videos(src_root, dst_root)

    info = load_json(src_root / "meta/info.json")
    fps = int(info.get("fps") or 0)
    video_path_template = info.get("video_path")
    if fps <= 0 or not video_path_template:
        raise ValueError(f"missing fps/video_path in {src_root / 'meta/info.json'}")

    episodes_files = sorted((src_root / "meta/episodes").glob("*/*.parquet"))
    if not episodes_files:
        raise FileNotFoundError(f"missing episode metadata under {src_root / 'meta/episodes'}")
    episodes = pd.concat([pq.read_table(path).to_pandas() for path in episodes_files], ignore_index=True)

    work_items: list[tuple[str, tuple[int, int], Path, Path, int]] = []
    for cam in camera_keys:
        chunk_col = f"videos/{cam}/chunk_index"
        file_col = f"videos/{cam}/file_index"
        if chunk_col not in episodes or file_col not in episodes:
            raise KeyError(f"missing video metadata columns for {cam} in {src_root}")
        pairs = sorted({(int(chunk), int(file)) for chunk, file in zip(episodes[chunk_col], episodes[file_col], strict=True)})
        for chunk_index, file_index in pairs:
            rows = episodes[
                (episodes[chunk_col].astype(int) == chunk_index)
                & (episodes[file_col].astype(int) == file_index)
            ]
            required_frames = int(rows["length"].max())
            src_video = src_root / chunk_file_path(
                video_path_template,
                video_key=cam,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            dst_video = dst_root / chunk_file_path(
                video_path_template,
                video_key=cam,
                chunk_index=chunk_index,
                file_index=file_index,
            )
            if not src_video.is_file():
                raise FileNotFoundError(f"missing video file: {src_video}")
            work_items.append((cam, (chunk_index, file_index), src_video, dst_video, required_frames))

    resize_shape = (int(image_resize_shape[0]), int(image_resize_shape[1]))
    print(
        f"[prepare] optimizing {len(work_items)} selected camera video(s) to "
        f"CFR H.264 {resize_shape[1]}x{resize_shape[0]} under {dst_root}"
    )

    def _transcode(item: tuple[str, tuple[int, int], Path, Path, int]) -> tuple[str, tuple[int, int], Path, int, int]:
        cam, pair, src_video, dst_video, required_frames = item
        transcode_to_h264_mp4(src_video, dst_video, "h264", fps, resize_shape=resize_shape)
        frame_count = video_file_frame_count(dst_video)
        if frame_count is None:
            raise RuntimeError(f"could not read optimized video frame count: {dst_video}")
        return cam, pair, dst_video, frame_count, required_frames

    max_workers = max(1, int(jobs))
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        for cam, pair, dst_video, frame_count, required_frames in pool.map(_transcode, work_items):
            if frame_count < required_frames:
                raise RuntimeError(
                    f"optimized video for {cam} {pair} has {frame_count} frames but episode metadata "
                    f"requires {required_frames}: {dst_video}"
                )

    optimized_info = copy.deepcopy(info)
    optimized_features = copy.deepcopy(info["features"])
    for key, feature in list(optimized_features.items()):
        if isinstance(feature, dict) and feature.get("dtype") == "video" and key not in camera_keys:
            del optimized_features[key]
    for cam in camera_keys:
        optimized_features[cam] = resize_camera_feature(optimized_features[cam], image_resize_shape)
    optimized_info["features"] = optimized_features
    optimized_info["repo_id"] = f"{info.get('repo_id', src_root.name)}_optimized_{resize_shape[0]}x{resize_shape[1]}"
    write_json(dst_root / "meta/info.json", optimized_info)

    stats_path = dst_root / "meta/stats.json"
    if stats_path.is_file():
        stats = load_json(stats_path)
        for key in list(stats):
            feature = info.get("features", {}).get(key)
            if isinstance(feature, dict) and feature.get("dtype") == "video" and key not in camera_keys:
                del stats[key]
        write_json(stats_path, stats)

    manifest = {
        "source_dataset_root": str(src_root),
        "camera_keys": camera_keys,
        "image_resize_shape": image_resize_shape,
        "fps": fps,
        "video_count": len(work_items),
    }
    write_json(dst_root / "meta/video_optimization_manifest.json", manifest)
    return dst_root


def maybe_optimize_lerobot_video_dataset(
    args: argparse.Namespace,
    *,
    dataset_root: Path,
    camera_keys: list[str],
    image_resize_shape: list[int] | None,
) -> Path:
    if not args.auto_export_gmsl2 or image_resize_shape is None:
        return dataset_root
    if not lerobot_video_dataset_selected_cameras(dataset_root, camera_keys):
        return dataset_root

    shape_ok, shape_reason = dataset_camera_shapes_match(
        dataset_root,
        camera_keys=camera_keys,
        image_resize_shape=image_resize_shape,
    )
    timing_ok = False
    timing_reason = "skipped timing check because camera shape is incompatible"
    if shape_ok:
        timing_ok, timing_reason = dataset_videos_have_cfr_grid(dataset_root, camera_keys=camera_keys)

    if shape_ok and timing_ok and not args.force_export_gmsl2:
        return dataset_root

    export_root = optimized_lerobot_video_export_root(dataset_root, args.gmsl2_export_root.resolve(), image_resize_shape)
    if not args.force_export_gmsl2 and export_root.exists():
        try:
            export_shape_ok, export_shape_reason = dataset_camera_shapes_match(
                export_root,
                camera_keys=camera_keys,
                image_resize_shape=image_resize_shape,
            )
            export_timing_ok, export_timing_reason = (
                dataset_videos_have_cfr_grid(export_root, camera_keys=camera_keys)
                if export_shape_ok
                else (False, "skipped timing check because camera shape is incompatible")
            )
        except (OSError, KeyError, json.JSONDecodeError) as exc:
            export_shape_ok = False
            export_timing_ok = False
            export_shape_reason = f"could not read optimized export metadata: {exc}"
            export_timing_reason = export_shape_reason
        if export_shape_ok and export_timing_ok:
            print(f"[prepare] using existing optimized LeRobot video export: {export_root}")
            return export_root
        print(
            "[prepare] existing optimized LeRobot video export is incompatible: "
            f"{export_shape_reason if not export_shape_ok else export_timing_reason}"
        )

    if not shape_ok:
        print(f"[prepare] LeRobot video dataset needs optimization: {shape_reason}")
    elif not timing_ok:
        print(f"[prepare] LeRobot video dataset needs CFR optimization: {timing_reason}")
    else:
        print("[prepare] rebuilding optimized LeRobot video export because --force-export-gmsl2 was supplied")
    optimized = optimize_lerobot_video_dataset(
        src_root=dataset_root,
        dst_root=export_root,
        camera_keys=camera_keys,
        image_resize_shape=image_resize_shape,
        jobs=args.gmsl2_export_jobs,
    )
    optimized_shape_ok, optimized_shape_reason = dataset_camera_shapes_match(
        optimized,
        camera_keys=camera_keys,
        image_resize_shape=image_resize_shape,
    )
    if not optimized_shape_ok:
        raise RuntimeError(f"optimized LeRobot video export has incompatible camera shape: {optimized_shape_reason}")
    optimized_timing_ok, optimized_timing_reason = dataset_videos_have_cfr_grid(optimized, camera_keys=camera_keys)
    if not optimized_timing_ok:
        raise RuntimeError(f"optimized LeRobot video export has incompatible video timing: {optimized_timing_reason}")
    return optimized


def infer_gmsl2_task_name(dataset_root: Path) -> str:
    name = dataset_root.name
    match = re.match(r"^(?P<base>.+)_\d{8}_\d{6}(?:_\d{2})?$", name)
    if match:
        name = match.group("base")
    return name.replace("_", " ").replace("-", " ").strip() or dataset_root.name


def maybe_export_gmsl2_dataset(
    args: argparse.Namespace,
    *,
    camera_keys: list[str],
    state_keys: list[str],
    action_append_selectors: list[str],
    image_resize_shape: list[int] | None,
) -> Path:
    dataset_root = args.dataset_root.resolve()
    ok, reason = dataset_has_training_contract(
        dataset_root,
        camera_keys=camera_keys,
        state_keys=state_keys,
        action_key=args.action_key,
        action_npy=args.action_npy,
        action_append_selectors=action_append_selectors,
    )
    if ok:
        return maybe_optimize_lerobot_video_dataset(
            args,
            dataset_root=dataset_root,
            camera_keys=camera_keys,
            image_resize_shape=image_resize_shape,
        )

    if not args.auto_export_gmsl2:
        print(f"[prepare] dataset is not directly trainable: {reason}")
        return dataset_root

    if not looks_like_gmsl2_raw_dataset(dataset_root):
        print(f"[prepare] dataset does not look like a raw GMSL2 recording: {dataset_root}")
        print(f"[prepare] keeping original dataset root; prepare step will report the exact error. First mismatch: {reason}")
        return dataset_root

    export_root = (args.gmsl2_export_root / dataset_root.name).resolve()
    if not args.force_export_gmsl2 and export_root.exists():
        export_ok, export_reason = dataset_has_training_contract(
            export_root,
            camera_keys=camera_keys,
            state_keys=state_keys,
            action_key=args.action_key,
            action_npy=args.action_npy,
            action_append_selectors=action_append_selectors,
        )
        if export_ok:
            shape_ok, shape_reason = dataset_camera_shapes_match(
                export_root,
                camera_keys=camera_keys,
                image_resize_shape=image_resize_shape,
            )
            if not shape_ok:
                print(f"[prepare] existing GMSL2 export has incompatible camera shape: {shape_reason}")
                timing_ok = False
                timing_reason = "skipped timing check because camera shape is incompatible"
            else:
                timing_ok, timing_reason = dataset_videos_have_cfr_grid(export_root, camera_keys=camera_keys)
            if not timing_ok:
                print(f"[prepare] existing GMSL2 export has incompatible video timing: {timing_reason}")
            else:
                print(f"[prepare] using existing GMSL2 export: {export_root}")
                return export_root
        else:
            print(f"[prepare] existing GMSL2 export is incomplete or incompatible: {export_reason}")

    from tools.thor.gmsl2.export_v3 import export_task_to_v3

    task = args.gmsl2_export_task or infer_gmsl2_task_name(dataset_root)
    repo_id = args.gmsl2_export_repo_id or f"local/{dataset_root.name}"
    print(f"[prepare] exporting raw GMSL2 dataset before training: {dataset_root} -> {export_root}")
    exported = export_task_to_v3(
        datasets_root=dataset_root.parent,
        exports_root=args.gmsl2_export_root.resolve(),
        base_name=dataset_root.name,
        repo_id=repo_id,
        task=task,
        overwrite=True,
        jobs=args.gmsl2_export_jobs,
        resize_shape=tuple(image_resize_shape) if image_resize_shape is not None else None,
    ).resolve()
    export_ok, export_reason = dataset_has_training_contract(
        exported,
        camera_keys=camera_keys,
        state_keys=state_keys,
        action_key=args.action_key,
        action_npy=args.action_npy,
        action_append_selectors=action_append_selectors,
    )
    if not export_ok:
        raise RuntimeError(f"GMSL2 export completed but still does not match training contract: {export_reason}")
    shape_ok, shape_reason = dataset_camera_shapes_match(
        exported,
        camera_keys=camera_keys,
        image_resize_shape=image_resize_shape,
    )
    if not shape_ok:
        raise RuntimeError(f"GMSL2 export completed but camera shape is still incompatible: {shape_reason}")
    timing_ok, timing_reason = dataset_videos_have_cfr_grid(exported, camera_keys=camera_keys)
    if not timing_ok:
        raise RuntimeError(f"GMSL2 export completed but video timing is still incompatible: {timing_reason}")
    return exported


def chunk_file_from_path(path: Path) -> tuple[int, int]:
    if not path.parent.name.startswith("chunk-") or not path.stem.startswith("file-"):
        raise ValueError(f"Expected chunk/file path, got {path}")
    return int(path.parent.name.removeprefix("chunk-")), int(path.stem.removeprefix("file-"))


def chunk_file_for_index(index: int, chunks_size: int) -> tuple[int, int]:
    return index // chunks_size, index % chunks_size


def chunk_file_path(template: str, *, chunk_index: int, file_index: int, **kwargs: Any) -> Path:
    return Path(template.format(chunk_index=chunk_index, file_index=file_index, **kwargs))


def selected_state_names(features: dict[str, Any], state_keys: list[str]) -> list[str]:
    names: list[str] = []
    for selector in state_keys:
        key, index, dim_name = parse_feature_selector(selector, features)
        ft = features[key]
        ft_names = ft.get("names") or [str(i) for i in range(feature_dim(ft))]
        if index is not None:
            name = str(dim_name or ft_names[index])
            names.append(name if key == "observation.state" else f"{key}.{name}")
        elif key == "observation.state":
            names.extend([str(name) for name in ft_names])
        else:
            names.extend([f"{key}.{name}" for name in ft_names])
    return names


def select_state_matrix(df: pd.DataFrame, features: dict[str, Any], selectors: list[str]) -> np.ndarray:
    parts: list[np.ndarray] = []
    for selector in selectors:
        key, index, _ = parse_feature_selector(selector, features)
        values = as_matrix(df[key], key, feature_dim(features[key]))
        if index is None:
            parts.append(values)
        else:
            parts.append(values[:, index : index + 1])
    return np.concatenate(parts, axis=1).astype(np.float32)


def resize_camera_feature(feature: dict[str, Any], image_resize_shape: list[int] | None) -> dict[str, Any]:
    resized = copy.deepcopy(feature)
    if image_resize_shape is None:
        return resized

    height, width = int(image_resize_shape[0]), int(image_resize_shape[1])
    shape = resized.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError(f"Expected camera feature shape [H, W, C], got {shape}")
    resized["shape"] = [height, width, int(shape[2])]
    info = resized.setdefault("info", {})
    if isinstance(info, dict):
        info["video.height"] = height
        info["video.width"] = width
    return resized


def prepare_dataset_view(
    *,
    src_root: Path,
    dst_root: Path,
    repo_id: str,
    camera_keys: list[str],
    state_keys: list[str],
    action_key: str,
    action_npy: Path | None,
    action_append_selectors: list[str],
    action_append_names: list[str],
    action_append_shift: int,
    image_resize_shape: list[int] | None,
    copy_videos: bool,
    drop_nonfinite_episodes: bool,
    overwrite: bool,
) -> None:
    src_roots = discover_dataset_roots(src_root)
    if overwrite and dst_root.exists():
        shutil.rmtree(dst_root)
    if dst_root.exists():
        raise FileExistsError(f"{dst_root} already exists. Pass --overwrite-view to replace it.")

    source_infos = [load_json(root / "meta/info.json") for root in src_roots]
    first_info = source_infos[0]
    first_features = first_info["features"]
    first_stats = (
        load_json(src_roots[0] / "meta/stats.json") if (src_roots[0] / "meta/stats.json").exists() else {}
    )
    chunks_size = int(first_info.get("chunks_size", 1000))

    for root, info in zip(src_roots, source_infos, strict=True):
        features = info["features"]
        for key in camera_keys:
            if key not in features or features[key]["dtype"] not in ("video", "image"):
                raise KeyError(f"Camera feature not found in {root}: {key}")
            if features[key].get("shape") != first_features[key].get("shape"):
                raise ValueError(f"Camera feature shape mismatch for {key} in {root}")
        for selector in state_keys:
            key, _, _ = parse_feature_selector(selector, features)
            if not key.startswith("observation.") or features[key]["dtype"] not in ("float32", "float64"):
                raise ValueError(f"State key must be a numeric observation feature in {root}: {key}")
            if feature_dim(features[key]) != feature_dim(first_features[key]):
                raise ValueError(f"State feature dimension mismatch for {key} in {root}")
        source_action_npy = action_npy
        if source_action_npy is not None and not source_action_npy.is_absolute():
            source_action_npy = root / source_action_npy
        if source_action_npy is None and action_key not in features:
            raise KeyError(f"Action feature not found in {root}: {action_key}")
        if source_action_npy is None and feature_dim(features[action_key]) != feature_dim(first_features[action_key]):
            raise ValueError(f"Action feature dimension mismatch for {action_key} in {root}")
        for selector in action_append_selectors:
            parse_feature_selector(selector, features)

    if action_npy is not None and action_npy.is_absolute() and len(src_roots) > 1:
        raise ValueError(
            "Absolute --action-npy is ambiguous with multiple source datasets. "
            "Pass a relative path present under each dataset root, or use --use-derived-action."
        )

    default_append_names = action_append_dim_names(first_features, action_append_selectors)
    if action_append_names:
        if len(action_append_names) != len(default_append_names):
            raise ValueError(
                "--action-append-names length must match appended action dimension count: "
                f"{len(action_append_names)} != {len(default_append_names)}"
            )
        append_feature_names = action_append_names
    else:
        append_feature_names = default_append_names

    dst_root.mkdir(parents=True)
    (dst_root / "meta").mkdir(parents=True, exist_ok=True)

    global_task_to_index: dict[str, int] = {}
    task_index_maps: list[dict[int, int]] = []
    for root in src_roots:
        tasks = pd.read_parquet(root / "meta/tasks.parquet")
        source_map: dict[int, int] = {}
        for task, row in tasks.iterrows():
            task_name = str(task)
            if task_name not in global_task_to_index:
                global_task_to_index[task_name] = len(global_task_to_index)
            source_map[int(row["task_index"])] = global_task_to_index[task_name]
        task_index_maps.append(source_map)
    tasks_df = pd.DataFrame(
        {"task_index": [idx for _, idx in sorted(global_task_to_index.items(), key=lambda item: item[1])]},
        index=pd.Index(
            [task for task, _ in sorted(global_task_to_index.items(), key=lambda item: item[1])],
            name="task",
        ),
    )
    tasks_df.to_parquet(dst_root / "meta/tasks.parquet")

    state_parts: list[np.ndarray] = []
    action_parts: list[np.ndarray] = []
    scalar_parts: dict[str, list[np.ndarray]] = {
        key: [] for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]
    }
    source_data_files: list[list[Path]] = []
    source_episodes: list[pd.DataFrame] = []
    source_file_maps: list[dict[tuple[int, int], tuple[int, int]]] = []

    for root in src_roots:
        data_files = sorted((root / "data").glob("*/*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
        episodes_files = sorted((root / "meta/episodes").glob("*/*.parquet"))
        if not episodes_files:
            raise FileNotFoundError(f"No episode metadata files found under {root / 'meta/episodes'}")
        episodes = pd.concat([pq.read_table(path).to_pandas() for path in episodes_files], ignore_index=True)
        source_data_files.append(data_files)
        source_episodes.append(episodes)
        file_map: dict[tuple[int, int], tuple[int, int]] = {}
        for src_file in data_files:
            old_pair = chunk_file_from_path(src_file.relative_to(root))
            file_map[old_pair] = chunk_file_for_index(len(file_map) + sum(len(m) for m in source_file_maps), chunks_size)
        source_file_maps.append(file_map)

    dropped_episodes_by_source: list[set[int]] = []
    for source_idx, (root, data_files) in enumerate(zip(src_roots, source_data_files, strict=True)):
        features = source_infos[source_idx]["features"]
        bad_episodes: set[int] = set()
        source_action_npy = action_npy
        if source_action_npy is not None and not source_action_npy.is_absolute():
            source_action_npy = root / source_action_npy
        loaded_action_npy = np.load(source_action_npy).astype(np.float32) if source_action_npy is not None else None
        source_processed_rows = 0
        for src_file in data_files:
            df = pq.read_table(src_file).to_pandas()
            if state_keys:
                state = select_state_matrix(df, features, state_keys)
                bad_episodes.update(unrepairable_nonfinite_episodes(state, df["episode_index"]))
            if loaded_action_npy is not None:
                action = loaded_action_npy[source_processed_rows : source_processed_rows + len(df)]
            else:
                action = as_matrix(df[action_key], action_key, feature_dim(features[action_key]))
            action_append = select_action_append_matrix(
                df,
                features,
                action_append_selectors,
                action_append_shift,
            )
            if action_append is not None:
                action = np.concatenate([action, action_append], axis=1)
            bad_episodes.update(unrepairable_nonfinite_episodes(action, df["episode_index"]))
            source_processed_rows += len(df)
        if bad_episodes:
            if not drop_nonfinite_episodes:
                examples = sorted(bad_episodes)[:10]
                raise ValueError(
                    f"Selected state/action contains non-finite values that cannot be repaired in "
                    f"{root}; bad episode_index examples: {examples}. "
                    "Pass --drop-nonfinite-episodes to skip these episodes."
                )
            print(f"[prepare] dropping {len(bad_episodes)} non-finite episode(s) from {root}: {sorted(bad_episodes)}")
        dropped_episodes_by_source.append(bad_episodes)

    source_episode_maps: list[dict[int, int]] = []
    source_episode_frame_ranges: list[dict[int, tuple[int, int]]] = []
    total_rows = 0
    total_episodes = 0
    for episodes, dropped in zip(source_episodes, dropped_episodes_by_source, strict=True):
        episode_map: dict[int, int] = {}
        frame_ranges: dict[int, tuple[int, int]] = {}
        for _, row in episodes.sort_values("episode_index").iterrows():
            old_episode = int(row["episode_index"])
            if old_episode in dropped:
                continue
            length = int(row["length"])
            episode_map[old_episode] = total_episodes
            frame_ranges[old_episode] = (total_rows, total_rows + length)
            total_rows += length
            total_episodes += 1
        source_episode_maps.append(episode_map)
        source_episode_frame_ranges.append(frame_ranges)

    source_video_maps: list[dict[str, dict[tuple[int, int], tuple[int, int]]]] = []
    next_video_file_index = {cam: 0 for cam in camera_keys}
    for source_idx, episodes in enumerate(source_episodes):
        dropped = dropped_episodes_by_source[source_idx]
        kept_episodes = episodes[~episodes["episode_index"].astype(int).isin(dropped)]
        camera_maps: dict[str, dict[tuple[int, int], tuple[int, int]]] = {}
        for cam in camera_keys:
            chunk_col = f"videos/{cam}/chunk_index"
            file_col = f"videos/{cam}/file_index"
            if chunk_col not in kept_episodes or file_col not in kept_episodes:
                raise KeyError(f"Episode metadata missing video columns for {cam} in {src_roots[source_idx]}")
            pairs = sorted(
                {
                    (int(chunk), int(file))
                    for chunk, file in zip(kept_episodes[chunk_col], kept_episodes[file_col], strict=True)
                }
            )
            camera_map: dict[tuple[int, int], tuple[int, int]] = {}
            for pair in pairs:
                camera_map[pair] = chunk_file_for_index(next_video_file_index[cam], chunks_size)
                next_video_file_index[cam] += 1
            camera_maps[cam] = camera_map
        source_video_maps.append(camera_maps)

    for source_idx, root in enumerate(src_roots):
        for cam in camera_keys:
            for old_pair, new_pair in source_video_maps[source_idx][cam].items():
                old_chunk, old_file = old_pair
                new_chunk, new_file = new_pair
                src_video = root / chunk_file_path(
                    source_infos[source_idx]["video_path"],
                    video_key=cam,
                    chunk_index=old_chunk,
                    file_index=old_file,
                )
                if not src_video.exists():
                    raise FileNotFoundError(f"Video file not found: {src_video}")
                dst_video = dst_root / chunk_file_path(
                    first_info["video_path"],
                    video_key=cam,
                    chunk_index=new_chunk,
                    file_index=new_file,
                )
                copy_or_symlink_file(src_video, dst_video, copy=copy_videos)

    processed_rows = 0
    for source_idx, (root, data_files) in enumerate(zip(src_roots, source_data_files, strict=True)):
        features = source_infos[source_idx]["features"]
        file_map = source_file_maps[source_idx]
        episode_map = source_episode_maps[source_idx]
        task_index_map = task_index_maps[source_idx]
        source_processed_rows = 0

        source_action_npy = action_npy
        if source_action_npy is not None and not source_action_npy.is_absolute():
            source_action_npy = root / source_action_npy
        loaded_action_npy = np.load(source_action_npy).astype(np.float32) if source_action_npy is not None else None

        for src_file in data_files:
            old_pair = chunk_file_from_path(src_file.relative_to(root))
            new_chunk, new_file = file_map[old_pair]
            dst_file = dst_root / chunk_file_path(
                first_info["data_path"],
                chunk_index=new_chunk,
                file_index=new_file,
            )
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            df_raw = pq.read_table(src_file).to_pandas()
            source_row_count = len(df_raw)
            keep_mask = df_raw["episode_index"].astype(int).isin(episode_map)
            df = df_raw.loc[keep_mask].reset_index(drop=True)
            if df.empty:
                source_processed_rows += source_row_count
                continue

            out = pd.DataFrame()
            out["timestamp"] = df["timestamp"]
            out["frame_index"] = df["frame_index"]
            out["episode_index"] = df["episode_index"].astype(int).map(episode_map)
            out["index"] = np.arange(processed_rows, processed_rows + len(df), dtype=np.int64)
            out["task_index"] = df["task_index"].map(task_index_map)
            if out["task_index"].isna().any():
                missing = sorted(set(df.loc[out["task_index"].isna(), "task_index"].tolist()))
                raise ValueError(f"Missing task index mapping for {root}: {missing}")
            out["task_index"] = out["task_index"].astype(df["task_index"].dtype)

            if state_keys:
                state = select_state_matrix(df, features, state_keys)
                state = fill_nonfinite_matrix_within_episode(
                    state,
                    out["episode_index"],
                    label=f"state in {src_file.relative_to(root)}",
                )
                out["observation.state"] = list(state)
                state_parts.append(state)

            if loaded_action_npy is not None:
                action_raw = loaded_action_npy[source_processed_rows : source_processed_rows + source_row_count]
                action = action_raw[keep_mask.to_numpy()]
            else:
                action = as_matrix(df[action_key], action_key, feature_dim(features[action_key]))
            action_append = select_action_append_matrix(
                df,
                features,
                action_append_selectors,
                action_append_shift,
            )
            if action_append is not None:
                action = np.concatenate([action, action_append], axis=1)
            action = fill_nonfinite_matrix_within_episode(
                action.astype(np.float32),
                out["episode_index"],
                label=f"action in {src_file.relative_to(root)}",
            )
            out["action"] = list(action)
            action_parts.append(action)
            for key in scalar_parts:
                scalar_parts[key].append(np.asarray(out[key]).reshape(-1, 1))

            pq.write_table(pa.Table.from_pandas(out, preserve_index=False), dst_file)
            source_processed_rows += source_row_count
            processed_rows += len(df)

        if loaded_action_npy is not None and len(loaded_action_npy) != source_processed_rows:
            raise ValueError(f"{source_action_npy} has {len(loaded_action_npy)} rows, dataset has {source_processed_rows}")

    total_rows = processed_rows

    all_state = np.concatenate(state_parts, axis=0) if state_parts else np.empty((total_rows, 0), dtype=np.float32)
    all_action = np.concatenate(action_parts, axis=0)

    episode_views: list[pd.DataFrame] = []
    episode_keep_cols = [
        "episode_index",
        "tasks",
        "length",
        "data/chunk_index",
        "data/file_index",
        "dataset_from_index",
        "dataset_to_index",
    ]
    for cam in camera_keys:
        episode_keep_cols.extend(
            [
                f"videos/{cam}/chunk_index",
                f"videos/{cam}/file_index",
                f"videos/{cam}/from_timestamp",
                f"videos/{cam}/to_timestamp",
            ]
        )
    for source_idx, episodes in enumerate(source_episodes):
        file_map = source_file_maps[source_idx]
        episode_map = source_episode_maps[source_idx]
        frame_ranges = source_episode_frame_ranges[source_idx]
        out_episodes = episodes[episodes["episode_index"].astype(int).isin(episode_map)].copy()
        out_episodes["episode_index"] = out_episodes["episode_index"].astype(int).map(episode_map)
        out_episodes["dataset_from_index"] = [
            frame_ranges[int(old_episode)][0] for old_episode in episodes.loc[out_episodes.index, "episode_index"]
        ]
        out_episodes["dataset_to_index"] = [
            frame_ranges[int(old_episode)][1] for old_episode in episodes.loc[out_episodes.index, "episode_index"]
        ]

        data_pairs = [
            file_map[(int(chunk), int(file))]
            for chunk, file in zip(out_episodes["data/chunk_index"], out_episodes["data/file_index"], strict=True)
        ]
        out_episodes["data/chunk_index"] = [chunk for chunk, _ in data_pairs]
        out_episodes["data/file_index"] = [file for _, file in data_pairs]
        for cam in camera_keys:
            chunk_col = f"videos/{cam}/chunk_index"
            file_col = f"videos/{cam}/file_index"
            if chunk_col not in out_episodes or file_col not in out_episodes:
                continue
            video_map = source_video_maps[source_idx][cam]
            new_pairs = [
                video_map[(int(chunk), int(file))]
                for chunk, file in zip(out_episodes[chunk_col], out_episodes[file_col], strict=True)
            ]
            out_episodes[chunk_col] = [chunk for chunk, _ in new_pairs]
            out_episodes[file_col] = [file for _, file in new_pairs]
        missing_cols = [col for col in episode_keep_cols if col not in out_episodes]
        if missing_cols:
            raise KeyError(f"Episode metadata missing columns in {src_roots[source_idx]}: {missing_cols}")
        episode_views.append(out_episodes[episode_keep_cols])

    all_episodes = pd.concat(episode_views, ignore_index=True)
    episodes_path = dst_root / "meta/episodes/chunk-000/file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(all_episodes, preserve_index=False), episodes_path)

    new_features = {
        key: first_features[key]
        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index"]
        if key in first_features
    }
    if state_keys:
        new_features["observation.state"] = {
            "dtype": "float32",
            "shape": [int(all_state.shape[1])],
            "names": selected_state_names(first_features, state_keys),
        }
    for cam in camera_keys:
        new_features[cam] = resize_camera_feature(first_features[cam], image_resize_shape)
    action_names = first_features.get(action_key, {}).get("names")
    base_action_dim = all_action.shape[1] - len(append_feature_names)
    if action_names is None or len(action_names) != base_action_dim:
        action_names = [f"action.{i}" for i in range(base_action_dim)]
    action_names = [*action_names, *append_feature_names]
    new_features["action"] = {
        "dtype": "float32",
        "shape": [int(all_action.shape[1])],
        "names": action_names,
    }

    new_info = copy.deepcopy(first_info)
    new_info["robot_type"] = f"{first_info.get('robot_type', 'robot')}_il_view"
    new_info["features"] = new_features
    new_info["total_frames"] = int(total_rows)
    new_info["total_episodes"] = int(total_episodes)
    new_info["total_tasks"] = int(len(global_task_to_index))
    new_info["splits"] = {"train": f"0:{total_episodes}"}
    new_info["data_path"] = "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet"
    new_info["video_path"] = "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4"
    write_json(dst_root / "meta/info.json", new_info)

    new_stats = {
        key: first_stats[key]
        for key in ["timestamp", "frame_index", "episode_index", "index", "task_index", *camera_keys]
        if key in first_stats
    }
    for key, parts in scalar_parts.items():
        if parts:
            new_stats[key] = vector_stats(np.concatenate(parts, axis=0).astype(np.float32))
    if state_keys:
        new_stats["observation.state"] = vector_stats(all_state)
    new_stats["action"] = vector_stats(all_action)
    write_json(dst_root / "meta/stats.json", new_stats)

    manifest = {
        "source_dataset_root": str(src_roots[0]) if len(src_roots) == 1 else None,
        "source_dataset_roots": [str(root) for root in src_roots],
        "repo_id": repo_id,
        "cameras": camera_keys,
        "state_keys": state_keys,
        "action_key": None if action_npy else action_key,
        "action_npy": str(action_npy) if action_npy else None,
        "action_append_selectors": action_append_selectors,
        "action_append_names": append_feature_names,
        "action_append_shift": action_append_shift,
        "image_resize_shape": image_resize_shape,
        "state_dim": int(all_state.shape[1]) if state_keys else 0,
        "action_dim": int(all_action.shape[1]),
        "total_episodes": int(total_episodes),
        "total_rows": int(total_rows),
    }
    write_json(dst_root / "meta/il_view_manifest.json", manifest)


def make_train_config(args: argparse.Namespace, view_root: Path, repo_id: str, config_path: Path) -> None:
    image_resize_shape = parse_hw(args.image_resize_shape)
    peak_lr = args.act_lr if args.policy == "act" else args.dp_lr
    policy: dict[str, Any] = {
        "type": args.policy,
        "device": None if args.device == "auto" else args.device,
        "use_amp": args.use_amp,
        "push_to_hub": False,
    }
    if args.policy == "act":
        policy.update(
            {
                "chunk_size": args.act_chunk_size,
                "n_action_steps": args.act_n_action_steps,
                "vision_backbone": args.vision_backbone,
                "pretrained_backbone_weights": args.act_pretrained_backbone_weights,
                "optimizer_lr": args.act_lr,
                "optimizer_lr_backbone": args.act_lr_backbone,
            }
        )
    else:
        policy.update(
            {
                "n_obs_steps": args.dp_n_obs_steps,
                "horizon": args.dp_horizon,
                "n_action_steps": args.dp_n_action_steps,
                "drop_n_last_frames": args.dp_horizon - args.dp_n_action_steps - args.dp_n_obs_steps + 1,
                "vision_backbone": args.vision_backbone,
                "resize_shape": None if image_resize_shape is not None else parse_hw(args.dp_resize_shape),
                "optimizer_lr": args.dp_lr,
            }
        )

    dataset_cfg: dict[str, Any] = {
        "repo_id": repo_id,
        "root": str(view_root),
        "streaming": False,
        "use_imagenet_stats": args.use_imagenet_stats,
        "video_backend": args.video_backend,
    }
    if image_resize_shape is not None:
        dataset_cfg["image_transforms"] = {
            "enable": True,
            "max_num_transforms": 1,
            "random_order": False,
            "tfs": {
                "resize": {
                    "weight": 1.0,
                    "type": "Resize",
                    "kwargs": {
                        "size": image_resize_shape,
                        "antialias": True,
                    },
                }
            },
        }

    config = {
        "dataset": dataset_cfg,
        "policy": policy,
        "output_dir": str(args.output_dir),
        "job_name": args.job_name,
        "seed": args.seed,
        "num_workers": args.num_workers,
        "batch_size": args.batch_size,
        "steps": args.steps,
        "eval_freq": 0,
        "log_freq": args.log_freq,
        "save_checkpoint": True,
        "save_freq": args.save_freq,
        "tolerance_s": args.tolerance_s,
        "wandb": {"enable": args.wandb},
        "wandb_log_images_n_steps": args.wandb_log_images_n_steps,
        "wandb_log_images_n_samples": args.wandb_log_images_n_samples,
    }
    if args.lr_scheduler == "cosine_decay_with_warmup":
        config["scheduler"] = {
            "type": "cosine_decay_with_warmup",
            "num_warmup_steps": args.lr_warmup_steps,
            "num_decay_steps": args.lr_decay_steps or args.steps,
            "peak_lr": peak_lr,
            "decay_lr": args.lr_decay_final_lr if args.lr_decay_final_lr is not None else peak_lr * 0.1,
        }
    if args.wandb_project:
        config["wandb"]["project"] = args.wandb_project
    if args.wandb_entity:
        config["wandb"]["entity"] = args.wandb_entity
    if args.wandb_mode:
        config["wandb"]["mode"] = args.wandb_mode
    write_json(config_path, config)


def make_inference_config(
    args: argparse.Namespace,
    *,
    view_root: Path,
    repo_id: str,
    camera_keys: list[str],
    state_keys: list[str],
    action_append_selectors: list[str],
    action_append_names: list[str],
    image_resize_shape: list[int] | None,
    train_config_path: Path,
    inference_config_path: Path,
) -> None:
    camera_suffixes = [key.removeprefix("observation.images.") for key in camera_keys]
    checkpoint_path = args.output_dir / "checkpoints" / "last"
    config = {
        "version": 1,
        "training": {
            "policy": args.policy,
            "repo_id": repo_id,
            "dataset_root": str(view_root),
            "train_config": str(train_config_path),
            "checkpoint": str(checkpoint_path),
            "camera_features": camera_keys,
            "camera_keys": camera_suffixes,
            "image_resize_shape": image_resize_shape,
            "state_keys": state_keys,
            "state_observation": "observation.state" if state_keys else None,
            "action_key": "action",
            "action_source_key": None if args.action_npy else args.action_key,
            "action_npy": str(args.action_npy) if args.action_npy else None,
            "action_append_selectors": action_append_selectors,
            "action_append_names": action_append_names,
            "action_append_shift": args.action_append_shift,
        },
        "runtime": {
            "checkpoint": str(checkpoint_path),
            "dataset_root": str(view_root),
            "camera_config": "tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml",
            "policy_fps": None,
            "max_steps": None,
            "preview": True,
            "hardware": {
                "robot_ip": "192.168.1.208",
                "gripper_backend": "pika",
                "gripper_port": "/dev/ttyUSB0",
            },
            "startup": {
                "move_to_das_start": True,
                "align_gripper_to_dataset_start": True,
                "dataset_start_gripper_tolerance": 0.05,
            },
            "safety": {
                "first_frame_max_pos_delta_mm": 20.0,
                "first_frame_max_rot_delta_deg": 8.0,
                "max_step_pos_delta_mm": 3.0,
                "max_step_rot_delta_deg": 2.0,
            },
            "debug_step0_dump_dir": str(Path("outputs/debug") / f"{args.job_name}_step0"),
        },
    }
    write_yaml(inference_config_path, config)


def parse_hw(value: str | None) -> list[int] | None:
    if value in (None, "", "none", "None"):
        return None
    parts = parse_csv(value)
    if len(parts) != 2:
        raise ValueError("Expected H,W, for example 360,640")
    return [int(parts[0]), int(parts[1])]


def run_smoke(args: argparse.Namespace, view_root: Path, repo_id: str) -> None:
    import torch
    from torch.utils.data._utils.collate import default_collate

    from lerobot.datasets.factory import resolve_delta_timestamps
    from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
    from lerobot.datasets.transforms import ImageTransformConfig, ImageTransforms, ImageTransformsConfig
    from lerobot.policies.factory import make_policy
    from lerobot.policies.act.configuration_act import ACTConfig
    from lerobot.policies.diffusion.configuration_diffusion import DiffusionConfig

    meta = LeRobotDatasetMetadata(repo_id, root=view_root)
    if args.policy == "act":
        cfg = ACTConfig(
            push_to_hub=False,
            chunk_size=args.act_chunk_size,
            n_action_steps=args.act_n_action_steps,
        )
    else:
        cfg = DiffusionConfig(
            push_to_hub=False,
            n_obs_steps=args.dp_n_obs_steps,
            horizon=args.dp_horizon,
            n_action_steps=args.dp_n_action_steps,
        )
    delta_timestamps = resolve_delta_timestamps(cfg, meta)
    image_transforms = None
    image_resize_shape = parse_hw(args.image_resize_shape)
    if image_resize_shape is not None:
        image_transforms = ImageTransforms(
            ImageTransformsConfig(
                enable=True,
                max_num_transforms=1,
                random_order=False,
                tfs={
                    "resize": ImageTransformConfig(
                        weight=1.0,
                        type="Resize",
                        kwargs={"size": image_resize_shape, "antialias": True},
                    )
                },
            )
        )
    ds = LeRobotDataset(
        repo_id,
        root=view_root,
        delta_timestamps=delta_timestamps,
        image_transforms=image_transforms,
        video_backend=args.video_backend,
        tolerance_s=args.tolerance_s,
    )
    item = ds[0]
    print(f"[smoke] dataset len={len(ds)}")
    if "observation.state" in item:
        print(f"[smoke] observation.state shape={tuple(item['observation.state'].shape)}")
    else:
        print("[smoke] observation.state=<absent>")
    print(f"[smoke] action shape={tuple(item['action'].shape)}")
    for cam in meta.camera_keys:
        print(f"[smoke] {cam} shape={tuple(item[cam].shape)}")

    if ds.delta_indices is not None and "action" in ds.delta_indices:
        ds._ensure_hf_dataset_loaded()
        checked_indices: set[int] = set()
        for episode in meta.episodes:
            start = int(episode["dataset_from_index"])
            end = int(episode["dataset_to_index"])
            episode_index = int(episode["episode_index"])
            tail_start = max(start, end - max(1, len(ds.delta_indices["action"])))
            for abs_idx in range(tail_start, end):
                query_indices, _ = ds._get_query_indices(abs_idx, episode_index)
                ds._query_hf_dataset({"action": query_indices["action"]})
                checked_indices.add(abs_idx)
        print(f"[smoke] temporal action tail queries checked={len(checked_indices)}")

    if args.policy == "act":
        smoke_device = args.device
        if smoke_device == "auto":
            smoke_device = "cuda" if torch.cuda.is_available() else "cpu"
        cfg.device = smoke_device
        policy = make_policy(cfg=cfg, ds_meta=meta)
        policy.train()
        batch = default_collate([item])
        if "action_is_pad" not in batch:
            batch["action_is_pad"] = torch.zeros(batch["action"].shape[:2], dtype=torch.bool)
        batch = {
            key: value.to(smoke_device) if hasattr(value, "to") else value
            for key, value in batch.items()
        }
        with torch.no_grad():
            loss, loss_dict = policy.forward(batch)
        print(f"[smoke] act forward loss={float(loss.detach().cpu()):.6f} details={loss_dict}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--repo-id", default="single_cube2_il_view")
    parser.add_argument("--auto-export-gmsl2", action="store_true")
    parser.add_argument("--force-export-gmsl2", action="store_true")
    parser.add_argument("--gmsl2-export-root", type=Path, default=DEFAULT_GMSL2_EXPORT_ROOT)
    parser.add_argument("--gmsl2-export-repo-id", default=None)
    parser.add_argument("--gmsl2-export-task", default=None)
    parser.add_argument("--gmsl2-export-jobs", type=int, default=8)
    parser.add_argument("--policy", choices=["act", "diffusion"], default="act")
    parser.add_argument("--cameras", default=DEFAULT_CAMERAS)
    parser.add_argument("--state-keys", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--image-resize-shape", default=None)
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--action-npy", type=Path, default=None)
    parser.add_argument("--use-derived-action", action="store_true")
    parser.add_argument("--action-append-selectors", default=DEFAULT_ACTION_APPEND_SELECTORS)
    parser.add_argument("--action-append-names", default=DEFAULT_ACTION_APPEND_NAMES)
    parser.add_argument("--action-append-shift", type=int, default=1)
    parser.add_argument("--view-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--overwrite-view", action="store_true")
    parser.add_argument("--copy-videos", action="store_true")
    parser.add_argument("--drop-nonfinite-episodes", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument("--smoke", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--resume-checkpoint", type=Path, default=None)

    parser.add_argument("--device", default="auto")
    parser.add_argument("--use-amp", action="store_true")
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--log-freq", type=int, default=20)
    parser.add_argument("--save-freq", type=int, default=500)
    parser.add_argument("--seed", type=int, default=1000)
    parser.add_argument("--tolerance-s", type=float, default=1e-3)
    parser.add_argument("--video-backend", default="pyav")
    parser.add_argument("--wandb", action="store_true")
    parser.add_argument("--wandb-project", default="lerobot")
    parser.add_argument("--wandb-entity", default=None)
    parser.add_argument("--wandb-mode", choices=["online", "offline", "disabled"], default=None)
    parser.add_argument("--wandb-log-images-n-steps", type=int, default=0)
    parser.add_argument("--wandb-log-images-n-samples", type=int, default=2)
    parser.add_argument("--use-imagenet-stats", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--lr-scheduler", choices=["none", "cosine_decay_with_warmup"], default="none")
    parser.add_argument("--lr-warmup-steps", type=int, default=0)
    parser.add_argument("--lr-decay-steps", type=int, default=None)
    parser.add_argument("--lr-decay-final-lr", type=float, default=None)

    parser.add_argument("--vision-backbone", default="resnet18")
    parser.add_argument("--act-chunk-size", type=int, default=30)
    parser.add_argument("--act-n-action-steps", type=int, default=30)
    parser.add_argument("--act-lr", type=float, default=1e-5)
    parser.add_argument("--act-lr-backbone", type=float, default=1e-5)
    parser.add_argument("--act-pretrained-backbone-weights", default="ResNet18_Weights.IMAGENET1K_V1")

    parser.add_argument("--dp-n-obs-steps", type=int, default=2)
    parser.add_argument("--dp-horizon", type=int, default=16)
    parser.add_argument("--dp-n-action-steps", type=int, default=8)
    parser.add_argument("--dp-resize-shape", default="224,224")
    parser.add_argument("--dp-lr", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    steps_supplied = cli_arg_was_supplied("--steps")
    cameras = [normalize_camera_key(key) for key in parse_csv(args.cameras)]
    state_keys = parse_csv(args.state_keys)
    action_append_selectors = parse_csv(args.action_append_selectors)
    action_append_names = parse_csv(args.action_append_names)
    image_resize_shape = parse_hw(args.image_resize_shape)
    if not cameras:
        raise ValueError("At least one camera must be selected.")
    if args.policy != "act" and not state_keys:
        raise ValueError("Image-only low-dimensional observation is currently supported for ACT only.")
    if args.resume and args.lr_scheduler != "none":
        print("[resume] ignoring --lr-scheduler for resume; scheduler comes from the checkpoint config.")

    if args.use_derived_action:
        args.action_npy = DEFAULT_DERIVED_ACTION

    tag = f"{args.policy}_{'_'.join(key.split('.')[-1] for key in cameras)}"
    args.job_name = args.job_name or f"single_cube2_{tag}"
    view_root = args.view_root or Path("outputs/datasets") / args.job_name
    args.output_dir = args.output_dir or Path("outputs/train") / args.job_name
    config_path = view_root / "train_config.generated.json"
    inference_config_path = view_root / "inference_config.generated.yaml"

    if args.resume and view_root.exists() and not args.overwrite_view:
        print(f"[prepare] resume: keeping existing dataset view: {view_root}")
    else:
        args.dataset_root = maybe_export_gmsl2_dataset(
            args,
            camera_keys=cameras,
            state_keys=state_keys,
            action_append_selectors=action_append_selectors,
            image_resize_shape=image_resize_shape,
        )
        prepare_dataset_view(
            src_root=args.dataset_root,
            dst_root=view_root,
            repo_id=args.repo_id,
            camera_keys=cameras,
            state_keys=state_keys,
            action_key=args.action_key,
            action_npy=args.action_npy,
            action_append_selectors=action_append_selectors,
            action_append_names=action_append_names,
            action_append_shift=args.action_append_shift,
            image_resize_shape=image_resize_shape,
            copy_videos=args.copy_videos,
            drop_nonfinite_episodes=args.drop_nonfinite_episodes,
            overwrite=args.overwrite_view,
        )
        manifest = load_json(view_root / "meta/il_view_manifest.json")
        make_train_config(args, view_root, args.repo_id, config_path)
        make_inference_config(
            args,
            view_root=view_root,
            repo_id=args.repo_id,
            camera_keys=manifest["cameras"],
            state_keys=manifest["state_keys"],
            action_append_selectors=manifest["action_append_selectors"],
            action_append_names=manifest["action_append_names"],
            image_resize_shape=manifest["image_resize_shape"],
            train_config_path=config_path,
            inference_config_path=inference_config_path,
        )

        print(f"[prepare] dataset view: {view_root}")
        print(f"[prepare] train config: {config_path}")
        print(f"[prepare] inference config: {inference_config_path}")
    print(f"[prepare] output dir: {args.output_dir}")
    if args.smoke:
        run_smoke(args, view_root, args.repo_id)
    if args.prepare_only:
        return

    if args.resume:
        checkpoint = args.resume_checkpoint or (args.output_dir / "checkpoints" / "last")
        resume_config_path = resolve_resume_config_path(checkpoint)
        cmd = [
            sys.executable,
            "-m",
            "lerobot.scripts.lerobot_train",
            f"--config_path={resume_config_path}",
            "--resume=true",
        ]
        if steps_supplied:
            cmd.append(f"--steps={args.steps}")
        append_resume_override(cmd, "--batch-size", "batch_size", args.batch_size)
        append_resume_override(cmd, "--num-workers", "num_workers", args.num_workers)
        append_resume_override(cmd, "--log-freq", "log_freq", args.log_freq)
        append_resume_override(cmd, "--save-freq", "save_freq", args.save_freq)
        append_resume_override(cmd, "--wandb-log-images-n-steps", "wandb_log_images_n_steps", args.wandb_log_images_n_steps)
        append_resume_override(
            cmd,
            "--wandb-log-images-n-samples",
            "wandb_log_images_n_samples",
            args.wandb_log_images_n_samples,
        )
        print(f"[resume] checkpoint: {resume_config_path.parent.parent}")
        print(f"[resume] config: {resume_config_path}")
        if not steps_supplied:
            print("[resume] --steps was not supplied; using total steps saved in the checkpoint config.")
    else:
        cmd = [sys.executable, "-m", "lerobot.scripts.lerobot_train", f"--config_path={config_path}"]
    print("[train] " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
