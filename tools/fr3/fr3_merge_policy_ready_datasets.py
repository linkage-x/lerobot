#!/usr/bin/env python3
"""Merge a policy-ready training view with finalized DAgger correction datasets.

This is deliberately not the raw-recording training-view exporter. DAgger corrections are
already written in the policy action space of the checkpoint that produced them, so running the
normal delta-action derivation on them would transform the action column a second time.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

IS_INTERVENTION_KEY = "is_intervention"
BOOKKEEPING_FEATURES = {"timestamp", "frame_index", "episode_index", "index", "task_index"}


class MergeError(RuntimeError):
    """A merge refusal the operator can fix."""


@dataclass
class SourceSummary:
    role: str
    root: str
    episodes: int
    frames: int
    fps: int
    tasks: list[str]
    qc_status: str = ""


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = json.load(handle)
    if not isinstance(loaded, dict):
        raise MergeError(f"{path} is not a JSON object")
    return loaded


def write_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2)


def normalize_prompt(value: str) -> str:
    return " ".join(str(value or "").split())


def dataset_data_files(root: Path) -> list[Path]:
    return sorted((root / "data").glob("chunk-*/*.parquet"))


def dataset_episode_files(root: Path) -> list[Path]:
    return sorted((root / "meta" / "episodes").glob("chunk-*/*.parquet"))


def chunk_file_from_path(path: Path) -> tuple[int, int]:
    parts = path.parts
    try:
        chunk_name = next(part for part in parts if part.startswith("chunk-"))
        file_name = next(part for part in parts if part.startswith("file-") and part.endswith(".parquet"))
        return int(chunk_name.removeprefix("chunk-")), int(file_name.removeprefix("file-").removesuffix(".parquet"))
    except StopIteration as exc:
        raise MergeError(f"Cannot parse chunk/file from {path}") from exc


def chunk_file_for_index(index: int, chunks_size: int) -> tuple[int, int]:
    return index // chunks_size, index % chunks_size


def chunk_file_path(template: str, *, chunk_index: int, file_index: int, **kwargs: Any) -> Path:
    return Path(template.format(chunk_index=chunk_index, file_index=file_index, **kwargs))


def copy_or_symlink_file(src: Path, dst: Path, *, copy: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists() or dst.is_symlink():
        dst.unlink()
    if copy:
        shutil.copy2(src, dst)
    else:
        os.symlink(src.resolve(), dst)


# The quantiles a LeRobot dataset writer records alongside min/max/mean/std. They are not
# decoration: a policy whose `normalization_mapping` asks for QUANTILES -- pi0.5 does, for STATE
# -- reads q01 and q99 and raises at the first batch when they are missing. A merge that emitted
# only the four moments produced a dataset that loaded, then died in the normalizer.
STATS_QUANTILES = {"q01": 1.0, "q10": 10.0, "q50": 50.0, "q90": 90.0, "q99": 99.0}
STATS_KEYS = ("count", "min", "max", "mean", "std", *STATS_QUANTILES)


def vector_stats(values: np.ndarray) -> dict[str, Any]:
    """Per-dimension statistics over the merged rows, in the vocabulary LeRobot writes.

    Recomputed from the concatenated data rather than combined from the sources' own stats:
    quantiles do not compose, and the merge keeps only a subset of the base episodes, so the
    base view's q01 describes frames that may not be in this dataset.
    """
    arr = np.asarray(values, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.size == 0:
        return {key: [] for key in STATS_KEYS}
    stats: dict[str, Any] = {
        "count": [int(arr.shape[0])],
        "min": np.nanmin(arr, axis=0).astype(float).tolist(),
        "max": np.nanmax(arr, axis=0).astype(float).tolist(),
        "mean": np.nanmean(arr, axis=0).astype(float).tolist(),
        "std": np.nanstd(arr, axis=0).astype(float).tolist(),
    }
    for key, percentile in STATS_QUANTILES.items():
        stats[key] = np.nanpercentile(arr, percentile, axis=0).astype(float).tolist()
    return stats


def assert_stats_keep_what_the_sources_had(output_root: Path, base_view: Path) -> None:
    """Refuse a merge whose stats.json says less about a feature than the base view's did.

    Every statistic here is read by something: the normalizer picks its inputs from
    `normalization_mapping`, so which keys matter depends on the policy that will be trained, not
    on anything visible at merge time. Dropping one produces a dataset that merges, checks,
    loads -- and then raises inside the training step. Comparing key sets costs nothing and keeps
    that failure here.
    """
    stats_path = output_root / "meta" / "stats.json"
    base_path = base_view / "meta" / "stats.json"
    if not stats_path.is_file() or not base_path.is_file():
        return
    merged = load_json(stats_path)
    base = load_json(base_path)
    missing = sorted(
        f"{feature}: {', '.join(sorted(set(keys) - set(merged.get(feature) or {})))}"
        for feature, keys in base.items()
        if isinstance(keys, dict) and set(keys) - set(merged.get(feature) or {})
    )
    if missing:
        raise MergeError(
            "The merged stats.json drops statistics the base view recorded, which a policy that "
            "normalizes on them cannot train without: " + "; ".join(missing)
        )


def feature_without_intervention(features: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in features.items() if key != IS_INTERVENTION_KEY}


def canonical_feature(value: Any) -> str:
    return json.dumps(value, sort_keys=True, default=str)


def task_table(root: Path) -> pd.DataFrame:
    path = root / "meta" / "tasks.parquet"
    if not path.is_file():
        raise MergeError(f"{root} has no meta/tasks.parquet")
    table = pd.read_parquet(path)
    if "task_index" not in table.columns:
        raise MergeError(f"{path} has no task_index column")
    return table


def task_names(root: Path) -> list[str]:
    table = task_table(root)
    return [normalize_prompt(task) for task in table.index.tolist()]


def load_episodes(root: Path) -> pd.DataFrame:
    files = dataset_episode_files(root)
    if not files:
        raise MergeError(f"{root} has no meta/episodes parquet files; dataset is not finalized")
    episodes = pd.concat([pq.read_table(path).to_pandas() for path in files], ignore_index=True)
    required = {"episode_index", "length", "dataset_from_index", "dataset_to_index", "data/chunk_index", "data/file_index"}
    missing = sorted(required - set(episodes.columns))
    if missing:
        raise MergeError(f"{root} episode metadata is missing columns {missing}")
    return episodes.sort_values("dataset_from_index").reset_index(drop=True)


def qc_status(root: Path) -> str:
    path = root / "meta" / "processing.json"
    if not path.is_file():
        return "missing"
    try:
        meta = load_json(path)
    except (OSError, json.JSONDecodeError, MergeError):
        return "unreadable"
    active = meta.get("active_version")
    versions = meta.get("versions") if isinstance(meta.get("versions"), dict) else {}
    version = versions.get(active) if isinstance(active, str) else None
    qc = version.get("qc") if isinstance(version, dict) else None
    if not isinstance(qc, dict):
        return "missing"
    return str(qc.get("status") or "").lower() or "missing"


def intervention_feature() -> dict[str, Any]:
    return {"dtype": "float32", "shape": [1], "names": [IS_INTERVENTION_KEY]}


def validate_policy_ready_merge(
    base_view: Path, dagger_roots: Sequence[Path], *, base_episodes: Sequence[int] | None = None
) -> dict[str, Any]:
    if not dagger_roots:
        raise MergeError("Pass at least one DAgger dataset root.")
    base_info = load_json(base_view / "meta" / "info.json")
    base_manifest = load_json(base_view / "meta" / "il_view_manifest.json")
    if not dataset_data_files(base_view):
        raise MergeError(f"{base_view} has no data parquet files")
    if not dataset_episode_files(base_view):
        raise MergeError(f"{base_view} has no finalized episode metadata")

    base_features = base_info.get("features") if isinstance(base_info.get("features"), dict) else {}
    if not base_features:
        raise MergeError(f"{base_view} has no features in meta/info.json")
    base_feature_sig = canonical_feature(feature_without_intervention(base_features))
    base_fps = int(base_info.get("fps") or 0)
    base_tasks = task_names(base_view)
    base_task_set = sorted(set(base_tasks))
    action_mode = str(base_manifest.get("action_mode") or "")
    if not action_mode:
        raise MergeError(f"{base_view} manifest has no action_mode; it is not a training view")

    base_episode_rows = load_episodes(base_view)
    base_episode_filter = sorted({int(episode) for episode in (base_episodes or [])})
    if base_episode_filter:
        known_base_episodes = {int(index) for index in base_episode_rows["episode_index"]}
        unknown = sorted(set(base_episode_filter) - known_base_episodes)
        if unknown:
            raise MergeError(f"{base_view.name} has no base episode(s) {unknown}")
        base_episode_rows = base_episode_rows[base_episode_rows["episode_index"].isin(base_episode_filter)]
        if base_episode_rows.empty:
            raise MergeError("baseEpisodes excluded every base episode; nothing to merge from the base view")

    sources = [
        SourceSummary(
            role="base",
            root=str(base_view),
            episodes=int(len(base_episode_rows)),
            frames=int(base_episode_rows["length"].sum()),
            fps=base_fps,
            tasks=base_tasks,
        )
    ]
    base_message = f"{base_view.name} action_mode={action_mode}"
    if base_episode_filter:
        base_message += f", baseEpisodes={base_episode_filter}"
    checks: list[dict[str, Any]] = [
        {"name": "base_training_view", "status": "pass", "message": base_message}
    ]

    for dagger_root in dagger_roots:
        info = load_json(dagger_root / "meta" / "info.json")
        features = info.get("features") if isinstance(info.get("features"), dict) else {}
        if canonical_feature(feature_without_intervention(features)) != base_feature_sig:
            base_keys = sorted(feature_without_intervention(base_features))
            dagger_keys = sorted(feature_without_intervention(features))
            raise MergeError(
                f"{dagger_root.name} schema differs from base view. "
                f"base keys={base_keys}; dagger keys={dagger_keys}"
            )
        fps = int(info.get("fps") or 0)
        if fps != base_fps:
            raise MergeError(f"{dagger_root.name} fps={fps}, base view fps={base_fps}")
        tasks = task_names(dagger_root)
        if sorted(set(tasks)) != base_task_set:
            raise MergeError(
                f"{dagger_root.name} prompt set {sorted(set(tasks))} does not match base view {base_task_set}"
            )
        status = qc_status(dagger_root)
        if status != "pass":
            raise MergeError(f"{dagger_root.name} must be QC PASS before merge; current QC status is {status}")
        if not dataset_data_files(dagger_root) or not dataset_episode_files(dagger_root):
            raise MergeError(f"{dagger_root.name} is not a finalized LeRobot v3 dataset")
        sources.append(
            SourceSummary(
                role="dagger",
                root=str(dagger_root),
                episodes=int(info.get("total_episodes") or 0),
                frames=int(info.get("total_frames") or 0),
                fps=fps,
                tasks=tasks,
                qc_status=status,
            )
        )
        checks.append(
            {
                "name": "dagger_dataset",
                "status": "pass",
                "message": f"{dagger_root.name}: {int(info.get('total_episodes') or 0)} episode(s), QC PASS",
            }
        )

    total_episodes = sum(source.episodes for source in sources)
    total_frames = sum(source.frames for source in sources)
    return {
        "ok": True,
        "actionMode": action_mode,
        "fps": base_fps,
        "baseView": str(base_view),
        "daggerRoots": [str(root) for root in dagger_roots],
        "totalEpisodes": total_episodes,
        "totalFrames": total_frames,
        "sources": [asdict(source) for source in sources],
        "checks": checks,
        "summary": f"Ready to merge {len(dagger_roots)} DAgger dataset(s): {total_episodes} episode(s), {total_frames} frame(s).",
    }


def source_task_index_map(root: Path, global_task_to_index: dict[str, int]) -> dict[int, int]:
    table = task_table(root)
    mapping: dict[int, int] = {}
    for task, row in table.iterrows():
        task_name = normalize_prompt(str(task))
        if task_name not in global_task_to_index:
            raise MergeError(f"{root.name} task {task_name!r} was not present in the base view")
        mapping[int(row["task_index"])] = int(global_task_to_index[task_name])
    return mapping


def ensure_intervention_column(df: pd.DataFrame, *, is_dagger: bool) -> None:
    """Give every row an ``is_intervention`` flag, as the scalar LeRobot will ask parquet for.

    A shape-``[1]`` feature is read back as a ``datasets.Value``, not a length-1 ``Sequence``
    (``get_hf_features_from_features`` in ``lerobot/datasets/utils.py``), so the column has to be
    a bare float. Writing ``np.array([value])`` per row produces ``list<float>`` instead, which
    survives the merge and every check that reads ``meta/info.json`` -- the feature dict is
    identical either way -- and then fails with ``Couldn't cast array of type list<element:
    float> to float`` the moment training opens the dataset. Worse, it fails *unevenly*: the
    recorder's own writer already applies this rule, so the DAgger files were right and only the
    base-derived ones were wrong.

    An existing column is flattened rather than trusted, because a source that spelled the flag
    as a length-1 list is exactly the case that has to stop being propagated here.
    """
    if IS_INTERVENTION_KEY in df.columns:
        df[IS_INTERVENTION_KEY] = np.asarray(
            [float(np.reshape(value, -1)[0]) for value in df[IS_INTERVENTION_KEY]],
            dtype=np.float32,
        )
        return
    df[IS_INTERVENTION_KEY] = np.full(len(df), 1.0 if is_dagger else 0.0, dtype=np.float32)


def assert_uniform_data_schema(output_root: Path) -> None:
    """Refuse to hand back a merge whose data files disagree on their arrow types.

    The output concatenates parquet from two different producers and nothing upstream compares
    what they physically wrote: ``meta/info.json`` is what the schema check reads, and it stays
    identical while the files underneath differ. Such a dataset merges cleanly, passes
    ``--check-only``, appears on the training page, and dies seconds into a run. Catching it here
    costs one schema read per file and turns a failed training launch into a re-merge.
    """
    files = dataset_data_files(output_root)
    if not files:
        return
    reference = pq.read_schema(files[0])
    reference_types = {name: str(reference.field(name).type) for name in reference.names}
    for path in files[1:]:
        schema = pq.read_schema(path)
        types = {name: str(schema.field(name).type) for name in schema.names}
        if types == reference_types:
            continue
        differences = sorted(
            f"{key} is {reference_types.get(key, 'absent')} in {files[0].name} "
            f"but {types.get(key, 'absent')} in {path.name}"
            for key in set(reference_types) | set(types)
            if reference_types.get(key) != types.get(key)
        )
        raise MergeError(
            "The merged data files disagree on their parquet schema, which training cannot "
            "read: " + "; ".join(differences)
        )


def matrix_column(df: pd.DataFrame, key: str) -> np.ndarray | None:
    if key not in df.columns or df.empty:
        return None
    values = np.asarray(df[key].to_list(), dtype=np.float32)
    if values.ndim == 1:
        values = values.reshape(-1, 1)
    return values


def merge_policy_ready_datasets(
    *,
    base_view: Path,
    dagger_roots: Sequence[Path],
    output_root: Path,
    repo_id: str,
    base_episodes: Sequence[int] | None = None,
    overwrite: bool = False,
    copy_videos: bool = False,
) -> dict[str, Any]:
    validation = validate_policy_ready_merge(base_view, dagger_roots, base_episodes=base_episodes)
    roots = [base_view, *dagger_roots]
    roles = ["base", *["dagger" for _ in dagger_roots]]
    if overwrite and output_root.exists():
        shutil.rmtree(output_root)
    if output_root.exists():
        raise MergeError(f"{output_root} already exists; pass --overwrite to replace it.")

    infos = [load_json(root / "meta" / "info.json") for root in roots]
    base_info = copy.deepcopy(infos[0])
    base_features = copy.deepcopy(base_info["features"])
    base_features[IS_INTERVENTION_KEY] = intervention_feature()
    chunks_size = int(base_info.get("chunks_size") or 1000)
    data_template = str(base_info.get("data_path") or "data/chunk-{chunk_index:03d}/file-{file_index:03d}.parquet")
    video_template = str(base_info.get("video_path") or "videos/{video_key}/chunk-{chunk_index:03d}/file-{file_index:03d}.mp4")
    camera_keys = [
        key for key, feature in base_features.items()
        if isinstance(feature, dict) and key.startswith("observation.images.") and feature.get("dtype") in ("video", "image")
    ]

    output_root.mkdir(parents=True)
    (output_root / "meta").mkdir(parents=True, exist_ok=True)

    base_tasks = task_names(base_view)
    global_task_to_index = {task: index for index, task in enumerate(base_tasks)}
    tasks_df = pd.DataFrame(
        {"task_index": [idx for _, idx in sorted(global_task_to_index.items(), key=lambda item: item[1])]},
        index=pd.Index([task for task, _ in sorted(global_task_to_index.items(), key=lambda item: item[1])], name="task"),
    )
    tasks_df.to_parquet(output_root / "meta" / "tasks.parquet")

    source_file_maps: list[dict[tuple[int, int], tuple[int, int]]] = []
    source_episode_maps: list[dict[int, int]] = []
    source_frame_offsets: list[int] = []
    total_rows = 0
    total_episodes = 0
    all_episode_frames: list[pd.DataFrame] = []
    stats_parts: dict[str, list[np.ndarray]] = {key: [] for key in ["timestamp", "frame_index", "episode_index", "index", "task_index", "observation.state", "action", IS_INTERVENTION_KEY]}

    base_episode_filter = sorted({int(episode) for episode in (base_episodes or [])})

    next_file_index = 0
    for source_idx, root in enumerate(roots):
        data_files = dataset_data_files(root)
        episodes = load_episodes(root)
        if source_idx == 0 and base_episode_filter:
            episodes = episodes[episodes["episode_index"].isin(base_episode_filter)].reset_index(drop=True)
        source_frame_offsets.append(total_rows)
        episode_map = {
            int(source_index): total_episodes + position
            for position, source_index in enumerate(episodes["episode_index"])
        }
        source_episode_maps.append(episode_map)
        file_map: dict[tuple[int, int], tuple[int, int]] = {}
        for src_file in data_files:
            old_pair = chunk_file_from_path(src_file.relative_to(root))
            file_map[old_pair] = chunk_file_for_index(next_file_index, chunks_size)
            next_file_index += 1
        source_file_maps.append(file_map)
        total_rows += int(episodes["length"].sum())
        total_episodes += len(episodes)

    for source_idx, (root, role, info) in enumerate(zip(roots, roles, infos, strict=True)):
        file_map = source_file_maps[source_idx]
        for cam in camera_keys:
            for old_pair, new_pair in file_map.items():
                old_chunk, old_file = old_pair
                new_chunk, new_file = new_pair
                src_video = root / chunk_file_path(
                    str(info.get("video_path") or video_template),
                    video_key=cam,
                    chunk_index=old_chunk,
                    file_index=old_file,
                )
                if not src_video.exists():
                    continue
                dst_video = output_root / chunk_file_path(
                    video_template,
                    video_key=cam,
                    chunk_index=new_chunk,
                    file_index=new_file,
                )
                copy_or_symlink_file(src_video, dst_video, copy=copy_videos)

        episode_map = source_episode_maps[source_idx]
        task_map = source_task_index_map(root, global_task_to_index)
        frame_offset = source_frame_offsets[source_idx]
        source_written_rows = 0
        for src_file in dataset_data_files(root):
            old_pair = chunk_file_from_path(src_file.relative_to(root))
            new_chunk, new_file = file_map[old_pair]
            dst_file = output_root / chunk_file_path(data_template, chunk_index=new_chunk, file_index=new_file)
            dst_file.parent.mkdir(parents=True, exist_ok=True)

            df = pq.read_table(src_file).to_pandas()
            if df.empty:
                pq.write_table(pa.Table.from_pandas(df, preserve_index=False), dst_file)
                continue
            keep = df["episode_index"].isin(episode_map).to_numpy()
            out = df[keep].reset_index(drop=True).copy()
            ensure_intervention_column(out, is_dagger=role == "dagger")
            out["episode_index"] = out["episode_index"].map(episode_map).astype(df["episode_index"].dtype)
            out["index"] = np.arange(len(out), dtype=np.int64) + frame_offset + source_written_rows
            out["task_index"] = out["task_index"].map(task_map).astype(df["task_index"].dtype)
            source_written_rows += len(out)
            for key in stats_parts:
                values = matrix_column(out, key)
                if values is not None:
                    stats_parts[key].append(values)
            pq.write_table(pa.Table.from_pandas(out, preserve_index=False), dst_file)

        episodes = load_episodes(root).copy()
        if source_idx == 0 and base_episode_filter:
            episodes = episodes[episodes["episode_index"].isin(base_episode_filter)].reset_index(drop=True)
        episodes["episode_index"] = [episode_map[int(index)] for index in episodes["episode_index"]]
        lengths = episodes["length"].to_numpy()
        starts = frame_offset + np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(lengths.dtype)
        episodes["dataset_from_index"] = starts
        episodes["dataset_to_index"] = starts + lengths
        for col_prefix in ["data", *[f"videos/{cam}" for cam in camera_keys]]:
            chunk_col = f"{col_prefix}/chunk_index"
            file_col = f"{col_prefix}/file_index"
            if chunk_col not in episodes or file_col not in episodes:
                continue
            pairs = [
                file_map[(int(chunk), int(file))]
                for chunk, file in zip(episodes[chunk_col], episodes[file_col], strict=True)
            ]
            episodes[chunk_col] = [chunk for chunk, _ in pairs]
            episodes[file_col] = [file for _, file in pairs]
        all_episode_frames.append(episodes)

    assert_uniform_data_schema(output_root)

    all_episodes = pd.concat(all_episode_frames, ignore_index=True)
    episodes_path = output_root / "meta" / "episodes" / "chunk-000" / "file-000.parquet"
    episodes_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(all_episodes, preserve_index=False), episodes_path)

    merged_info = copy.deepcopy(base_info)
    merged_info["repo_id"] = repo_id
    merged_info["features"] = base_features
    merged_info["total_frames"] = int(total_rows)
    merged_info["total_episodes"] = int(total_episodes)
    merged_info["total_tasks"] = int(len(global_task_to_index))
    merged_info["splits"] = {"train": f"0:{total_episodes}"}
    merged_info["data_path"] = data_template
    merged_info["video_path"] = video_template
    write_json(output_root / "meta" / "info.json", merged_info)

    base_stats = load_json(base_view / "meta" / "stats.json") if (base_view / "meta" / "stats.json").is_file() else {}
    stats: dict[str, Any] = {
        key: base_stats[key]
        for key in camera_keys
        if key in base_stats
    }
    for key, parts in stats_parts.items():
        if parts:
            stats[key] = vector_stats(np.concatenate(parts, axis=0))
    write_json(output_root / "meta" / "stats.json", stats)
    assert_stats_keep_what_the_sources_had(output_root, base_view)

    source_digest = hashlib.sha256(
        json.dumps(
            {
                "merge_type": "policy_ready_dagger",
                "base_view": str(base_view),
                "dagger_roots": [str(root) for root in dagger_roots],
                "base_episodes": base_episode_filter,
                "tasks": base_tasks,
                "features": feature_without_intervention(base_features),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]
    base_manifest = load_json(base_view / "meta" / "il_view_manifest.json")
    manifest = {
        **base_manifest,
        "source_dataset_root": None,
        "source_dataset_roots": [str(root) for root in roots],
        "base_training_view_root": str(base_view),
        "dagger_dataset_roots": [str(root) for root in dagger_roots],
        "base_episode_filter": base_episode_filter,
        "merge_type": "policy_ready_dagger",
        "build_id": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_digest": source_digest,
        "repo_id": repo_id,
        "task_prompts": base_tasks,
        "source_task_prompts": base_tasks,
        "fps": int(merged_info.get("fps") or 0),
        "total_episodes": int(total_episodes),
        "total_rows": int(total_rows),
        "is_intervention_key": IS_INTERVENTION_KEY,
        "episode_source_index": [
            {
                "episode_index": int(view_episode),
                "source_dataset_root": str(root),
                "source_episode_index": int(source_episode),
            }
            for root, episode_map in zip(roots, source_episode_maps, strict=True)
            for source_episode, view_episode in sorted(episode_map.items(), key=lambda item: item[1])
        ],
    }
    write_json(output_root / "meta" / "il_view_manifest.json", manifest)

    return {
        **validation,
        "ok": True,
        "outputRoot": str(output_root),
        "repoId": repo_id,
        "totalEpisodes": int(total_episodes),
        "totalFrames": int(total_rows),
        "summary": f"Merged policy-ready view: {total_episodes} episode(s), {total_rows} frame(s) -> {output_root}",
    }


def parse_episode_list(value: str) -> list[int]:
    if not str(value or "").strip():
        return []
    episodes: list[int] = []
    for item in str(value).split(","):
        item = item.strip()
        if not item:
            continue
        episodes.append(int(item))
    return episodes


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-view", type=Path, required=True)
    parser.add_argument("--dagger-roots", type=Path, nargs="+", required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--repo-id", required=True)
    parser.add_argument("--base-episodes", default="", help="Comma-separated base view episode indices to keep.")
    parser.add_argument("--check-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--copy-videos", action="store_true")
    parser.add_argument("--json", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        base_episodes = parse_episode_list(args.base_episodes)
        if args.check_only:
            result = validate_policy_ready_merge(args.base_view, args.dagger_roots, base_episodes=base_episodes)
        else:
            result = merge_policy_ready_datasets(
                base_view=args.base_view,
                dagger_roots=args.dagger_roots,
                output_root=args.output_root,
                repo_id=args.repo_id,
                base_episodes=base_episodes,
                overwrite=args.overwrite,
                copy_videos=args.copy_videos,
            )
    except Exception as exc:  # noqa: BLE001 - CLI should report JSON instead of traceback when asked.
        if args.json:
            print(json.dumps({"ok": False, "error": str(exc)}, ensure_ascii=False))
            return 1
        raise
    if args.json:
        print(json.dumps(result, ensure_ascii=False))
    else:
        print(result["summary"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
