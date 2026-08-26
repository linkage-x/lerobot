#!/usr/bin/env python
"""Prepare a selectable LeRobot dataset view and train any LeRobot policy on it.

This script is intentionally conservative: it never mutates the source dataset.
It writes a derived dataset view under outputs/datasets and launches the standard
LeRobot training entrypoint on that view.

``--policy`` accepts any type the LeRobot registry knows (``act``, ``diffusion``,
``smolvla``, ``pi0``, ``pi05``, ``groot``, ``xvla``, ``wall_x``, ``vqbet``, ...).
Only ``act`` and ``diffusion`` get dedicated hyperparameter flags, and those flags
default to the upstream policy dataclass values (ACT ``chunk_size`` 100, diffusion
``horizon`` 16, ...), so a run nobody overrode trains what LeRobot itself would train.
Every other type is emitted with the common training keys only, so its own dataclass
defaults apply. ``--policy-config`` overrides either, and is the single place a
rig-specific number belongs: a value that suited one recording, left standing as the
default for every later one, is a guess wearing the costume of a default.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import shutil
import subprocess
import sys
from collections.abc import Callable, Iterable, Sequence
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import yaml

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _import_root in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from lerobot.robots.franka_research3.action_modes import (  # noqa: E402
    ACTION_MODE_ABSOLUTE_EE,
    ACTION_MODES,
    is_delta_action_mode,
)

from tools.fr3.fr3_delta_action_transform import (  # noqa: E402
    derive_delta_action,
    summarize_delta_scale,
)


DEFAULT_DATASET_ROOT = Path("dataset_test/single_cube2_20260429_165325")
DEFAULT_CAMERAS = "observation.images.cam_1,observation.images.cam_3"
DEFAULT_STATE_KEYS = "observation.state"
DEFAULT_DERIVED_ACTION = Path("derived/hikon_cube_tracking_in_robot_base/action.npy")
DEFAULT_ACTION_APPEND_SELECTORS = "observation.state_raw:handheld_gripper.pika_left.width_mm"
DEFAULT_ACTION_APPEND_NAMES = "gripper"

# Types with dedicated hyperparameter flags below. Those flags default to the upstream
# dataclass values, so being on this list changes how a type can be overridden, not what
# it trains untouched. Everything else is still accepted -- it is validated against the
# LeRobot registry at run time, not against this list.
FLAGGED_POLICY_TYPES = ("act", "diffusion")
# Advertised in --help and by the GUI. Kept as a literal because resolving it needs
# lerobot.policies.factory, and importing that to print a usage string would make
# `--help` depend on every policy's optional dependencies.
KNOWN_POLICY_TYPES = (
    "act",
    "diffusion",
    "vqbet",
    "tdmpc",
    "pi0",
    "pi0_fast",
    "pi05",
    "smolvla",
    "groot",
    "xvla",
    "wall_x",
    "sac",
    "sarm",
    "reward_classifier",
)


DEFAULT_RECORD_CONFIG = "tools/fr3/fr3_record_config.yaml"
# Every inference camera config in the tree. The generated config names one of these; which one
# is decided by matching cameras against the recording, never by a default.
INFER_CAMERA_CONFIG_CANDIDATES = (
    "tools/fr3/fr3_il_infer_realsense_camera_config.yaml",
    "tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml",
    "tools/fr3/fr3_il_infer_gmsl2_corenetic_camera_config.yaml",
    "tools/fr3/fr3_act_infer_camera_config.yaml",
)


def load_yaml(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        loaded = yaml.safe_load(handle)
    return loaded if isinstance(loaded, dict) else {}


def camera_identities(config: dict[str, Any]) -> dict[str, tuple[str, str]]:
    """(type, device id) per camera key.

    The device id is what makes this an identity rather than a label: two configs can both
    call a camera `ee` and open different hardware, which is precisely the mistake a policy
    cannot detect -- it gets images of the right shape from the wrong viewpoint.
    """
    cameras = ((config.get("robot") or {}).get("cameras")) or {}
    identities: dict[str, tuple[str, str]] = {}
    for key, camera in cameras.items():
        if not isinstance(camera, dict):
            continue
        device = camera.get("serial_number_or_name", camera.get("index_or_path", ""))
        identities[str(key)] = (str(camera.get("type", "")).lower(), str(device))
    return identities


def resolve_inference_camera_config(
    record_config: dict[str, Any], camera_keys: list[str], repo_root: Path
) -> str | None:
    """Pick the inference camera config that opens the same cameras the view was recorded on.

    Matched by camera identity against the record config rather than assumed, because the two
    disagreeing is not a loud failure: the wrong file either names cameras the checkpoint never
    asks for (raises at startup, fine) or names the right keys pointed at different hardware
    (runs, and is wrong for the whole rollout).

    Returns None when nothing matches, and the caller writes null. A null that fails loudly at
    deployment is worth more than a plausible default that is silently wrong.
    """
    wanted_keys = [key.removeprefix("observation.images.") for key in camera_keys]
    recorded = camera_identities(record_config)
    wanted = {key: recorded[key] for key in wanted_keys if key in recorded}
    if not wanted:
        return None
    for candidate in INFER_CAMERA_CONFIG_CANDIDATES:
        path = repo_root / candidate
        if not path.exists():
            continue
        available = camera_identities(load_yaml(path))
        if all(available.get(key) == identity for key, identity in wanted.items()):
            return candidate
    return None


def resolve_frame_strides(
    src_roots: list[Path], source_infos: list[dict], view_fps: int
) -> tuple[int, list[int]]:
    """Decide how many source frames each view frame stands for, per source dataset.

    ``view_fps <= 0`` means "keep the sources' own rate", which then requires every source
    to already agree -- merging 30 fps and 60 fps recordings into one view without saying
    so produces an action column whose per-frame delta is twice as large in half the rows,
    and nothing downstream can tell the two halves apart.

    Only integer decimation is allowed. 60 -> 25 would need interpolation, and picking the
    nearest frame instead would jitter the sample interval between 1 and 2 source frames --
    a 2x swing in every delta, distributed unevenly through the episode.
    """
    source_fps: list[int] = []
    for root, info in zip(src_roots, source_infos, strict=True):
        fps = int(info.get("fps") or 0)
        if fps <= 0:
            raise ValueError(f"{root} has no usable fps in meta/info.json (got {info.get('fps')!r}).")
        source_fps.append(fps)

    if view_fps <= 0:
        distinct = sorted(set(source_fps))
        if len(distinct) > 1:
            raise ValueError(
                f"Source datasets disagree on fps ({distinct}); pass --view-fps to resample them "
                "to a common rate. Merging them as-is would put two different per-frame action "
                "scales in one column."
            )
        return distinct[0], [1] * len(src_roots)

    strides: list[int] = []
    for root, fps in zip(src_roots, source_fps, strict=True):
        if fps < view_fps:
            raise ValueError(
                f"{root} is {fps} fps, below the requested --view-fps {view_fps}. Upsampling would "
                "invent frames; lower --view-fps instead."
            )
        if fps % view_fps != 0:
            raise ValueError(
                f"{root} is {fps} fps, which is not an integer multiple of --view-fps {view_fps}. "
                "Only integer decimation is supported; pick a divisor of the source rate "
                f"(for example {', '.join(str(fps // n) for n in range(1, 5) if fps % n == 0)})."
            )
        strides.append(fps // view_fps)
    return view_fps, strides


def rewrite_episode_tasks(value: Iterable[Any], rewrite: Callable[[str], str]) -> list[str]:
    """Apply a prompt rewrite to one episode's `tasks` list, keeping order and dropping dupes.

    Order is preserved rather than sorted because this column is read by people, and dedup
    happens because a rewrite can map two of an episode's prompts onto one.
    """
    rewritten: list[str] = []
    for task in list(value):
        replacement = rewrite(str(task))
        if replacement not in rewritten:
            rewritten.append(replacement)
    return rewritten


def parse_task_prompt_map(value: str | None) -> dict[str, str]:
    """Parse --task-prompt-map, a JSON object of {recorded prompt: prompt to train on}.

    JSON rather than `old=new` pairs because a task string is a sentence: it contains spaces,
    commas and (given what people actually type into a recorder) `=`. Any separator this could
    have used is a character a prompt is allowed to hold.
    """
    if not value or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--task-prompt-map must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"--task-prompt-map must be a JSON object, got {type(parsed).__name__}")
    mapping: dict[str, str] = {}
    for key, replacement in parsed.items():
        if not isinstance(replacement, str) or not replacement.strip():
            raise ValueError(f"--task-prompt-map value for {key!r} must be a non-empty string.")
        mapping[str(key)] = replacement.strip()
    return mapping


def normalize_task_prompt(value: str | None) -> str:
    """The prompt as the policy will see it, or "" for "leave the recording's prompt alone".

    Collapsed to single spaces because pi0/pi0.5 tokenize the task into a fixed-width prompt
    (`Task: {task}, State: ...`), so a trailing newline or a double space is not cosmetic -- it
    is a different token sequence conditioning every frame in the dataset.
    """
    return " ".join((value or "").split())


def parse_policy_config(value: str | None) -> dict[str, Any]:
    """Parse --policy-config, which is JSON so it can carry non-string types.

    Policy hyperparameters are ints, floats, bools, lists and nested dicts. A
    ``key=value`` mini-language would have to guess which of those a token is, and
    guessing wrong on ``optimizer_lr=1e-5`` (str vs float) fails deep inside the
    optimizer instead of here.
    """
    if not value or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError as exc:
        raise ValueError(f"--policy-config must be valid JSON: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ValueError(f"--policy-config must be a JSON object, got {type(parsed).__name__}")
    return parsed


def validate_policy_type(policy_type: str) -> None:
    """Fail here, with the list, rather than inside the training subprocess.

    Names in KNOWN_POLICY_TYPES are accepted without importing anything: they are the
    factory's own branches, and resolving them for real would pull in torch and every
    optional backbone dependency just to spell-check a string -- which `--prepare-only`
    has no other reason to do. A missing dependency still surfaces, at the point the
    policy is actually built, as the ModuleNotFoundError naming the package.

    Anything else is looked up in the draccus registry, which is where a policy that
    lives outside this repo registers itself.
    """
    if policy_type in KNOWN_POLICY_TYPES:
        return
    try:
        from lerobot.configs.policies import PreTrainedConfig

        PreTrainedConfig.get_choice_class(policy_type)
    except Exception as exc:
        known = ", ".join(KNOWN_POLICY_TYPES)
        raise ValueError(
            f"Unknown policy type {policy_type!r}. Known types: {known}. "
            "Third-party types must be registered with draccus before this script sees them."
        ) from exc


def parse_csv(value: str | None) -> list[str]:
    if value is None:
        return []
    if value.strip().lower() in {"", "none", "null"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


def normalize_camera_key(key: str) -> str:
    return key if key.startswith("observation.images.") else f"observation.images.{key}"


def parse_camera_crop_specs(value: str | None) -> dict[str, list[int]]:
    if value is None or value.strip().lower() in {"", "none", "null", "{}"}:
        return {}
    raw = value.strip()
    parsed: dict[str, Any]
    if raw.startswith("{"):
        loaded = json.loads(raw)
        if not isinstance(loaded, dict):
            raise ValueError("--camera-crops JSON must be an object keyed by camera name.")
        parsed = loaded
    else:
        parsed = {}
        for item in raw.split(";"):
            if not item.strip():
                continue
            key, sep, spec = item.partition(":")
            if not sep:
                raise ValueError(f"Invalid camera crop spec {item!r}; expected camera:x,y,w,h.")
            parsed[key.strip()] = [part.strip() for part in spec.split(",")]

    crops: dict[str, list[int]] = {}
    for key, spec in parsed.items():
        if isinstance(spec, dict):
            values = [spec.get(name) for name in ("x", "y", "w", "h")]
        elif isinstance(spec, (list, tuple)):
            values = list(spec)
        else:
            raise ValueError(f"Invalid crop for {key!r}: expected [x,y,w,h] or object.")
        if len(values) != 4:
            raise ValueError(f"Invalid crop for {key!r}: expected four values x,y,w,h.")
        try:
            crop = [int(value) for value in values]
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Invalid crop for {key!r}: values must be integers.") from exc
        crops[normalize_camera_key(str(key))] = crop
    return crops


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


def copy_or_symlink_file(src: Path, dst: Path, *, copy: bool) -> None:
    if dst.exists() or dst.is_symlink():
        return
    dst.parent.mkdir(parents=True, exist_ok=True)
    if copy:
        shutil.copy2(src, dst)
    else:
        dst.symlink_to(src.resolve())


def discover_dataset_roots(dataset_root: Path | Sequence[Path]) -> list[Path]:
    """Resolve the build's source datasets.

    A sequence is an *explicit* selection and is taken literally: every entry must be a dataset
    root itself. Expanding a directory that appears inside an explicit list would pull in
    datasets the operator did not pick, and the training set would then differ from what the
    GUI said it was building -- the one failure mode a merge must not have.
    """
    if isinstance(dataset_root, (list, tuple)):
        roots: list[Path] = []
        for entry in dataset_root:
            root = Path(entry).resolve()
            if not (root / "meta/info.json").is_file():
                raise FileNotFoundError(f"{root} is not a LeRobot dataset root (no meta/info.json).")
            if root not in roots:
                roots.append(root)
        if not roots:
            raise ValueError("No source datasets selected.")
        return roots

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


def action_drop_column_indices(base_action_names: list[str], drop_dims: list[str]) -> list[int]:
    """Columns of the *base* action (before appended selectors) that ``drop_dims`` names.

    Refuses a name that is not an action dim rather than silently dropping nothing: a typo here
    would otherwise produce a view that still carries the axis it was built to remove, and the
    only symptom would be a policy that keeps predicting it.
    """
    missing = [name for name in drop_dims if name not in base_action_names]
    if missing:
        raise ValueError(
            f"--action-drop-dims names are not action dims: {missing}; have {base_action_names}"
        )
    if len(drop_dims) >= len(base_action_names):
        raise ValueError("--action-drop-dims would drop every action dim.")
    return sorted(base_action_names.index(name) for name in drop_dims)


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


def resize_camera_feature(
    feature: dict[str, Any],
    image_resize_shape: list[int] | None,
    crop: list[int] | None = None,
) -> dict[str, Any]:
    resized = copy.deepcopy(feature)
    shape = resized.get("shape")
    if not isinstance(shape, list) or len(shape) != 3:
        raise ValueError(f"Expected camera feature shape [H, W, C], got {shape}")
    if image_resize_shape is not None:
        height, width = int(image_resize_shape[0]), int(image_resize_shape[1])
    elif crop is not None:
        height, width = int(crop[3]), int(crop[2])
    else:
        return resized
    resized["shape"] = [height, width, int(shape[2])]
    info = resized.setdefault("info", {})
    if isinstance(info, dict):
        info["video.height"] = height
        info["video.width"] = width
    return resized


def validate_camera_crop_specs(
    camera_crop_specs: dict[str, list[int]], features: dict[str, Any], camera_keys: list[str]
) -> dict[str, list[int]]:
    unknown = sorted(set(camera_crop_specs) - set(camera_keys))
    if unknown:
        raise ValueError(f"Crop specified for camera(s) not selected in --cameras: {unknown}")
    validated: dict[str, list[int]] = {}
    for camera_key, crop in camera_crop_specs.items():
        shape = features[camera_key].get("shape")
        if not isinstance(shape, list) or len(shape) != 3:
            raise ValueError(f"Expected camera feature shape [H, W, C] for {camera_key}, got {shape}")
        image_h, image_w = int(shape[0]), int(shape[1])
        x, y, w, h = [int(value) for value in crop]
        if x < 0 or y < 0 or w <= 0 or h <= 0:
            raise ValueError(f"Crop for {camera_key} must be non-negative x/y and positive w/h, got {crop}")
        if x + w > image_w or y + h > image_h:
            raise ValueError(f"Crop for {camera_key} exceeds {image_w}x{image_h}: {crop}")
        if any(value % 2 for value in (x, y, w, h)):
            raise ValueError(f"Crop for {camera_key} must use even x,y,w,h for yuv420p video, got {crop}")
        if x == 0 and y == 0 and w == image_w and h == image_h:
            continue
        validated[camera_key] = [x, y, w, h]
    return validated


def crop_video_file(src: Path, dst: Path, crop: list[int]) -> None:
    x, y, w, h = [int(value) for value in crop]
    dst.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        f"crop={w}:{h}:{x}:{y}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        "-preset",
        "veryfast",
        "-crf",
        "18",
        str(dst),
    ]
    subprocess.run(command, check=True)


ANNOTATION_STORE_RELATIVE_PATH = Path("meta") / "gui_annotations.json"


def annotated_excluded_episodes(dataset_root: Path) -> set[int]:
    """Episodes the operator marked *not* for training, from the GUI's annotation store.

    Read here rather than passed in as a list, so the Episode Replay checkbox and a command-line
    build reach the same answer. It used to be neither: the flag was written to
    ``meta/gui_annotations.json`` and nothing downstream ever read it, which made "exclude from
    training" a note to oneself -- the only way to actually drop an episode was to delete it.
    """
    store_path = dataset_root / ANNOTATION_STORE_RELATIVE_PATH
    if not store_path.is_file():
        return set()
    try:
        store = json.loads(store_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"{store_path} is unreadable: {exc}") from exc
    annotations = store.get("annotations") if isinstance(store, dict) else None
    if not isinstance(annotations, dict):
        return set()
    excluded: set[int] = set()
    for key, annotation in annotations.items():
        if not isinstance(annotation, dict):
            continue
        if annotation.get("includeInTraining", True):
            continue
        try:
            excluded.add(int(annotation.get("episode", key)))
        except (TypeError, ValueError):
            continue
    return excluded


def resolve_excluded_episodes(
    src_roots: list[Path],
    *,
    explicit: set[int] | None,
    respect_annotations: bool,
) -> dict[Path, set[int]]:
    """Per-source-root exclusion sets, from the annotation store and the explicit flag."""
    if explicit and len(src_roots) > 1:
        raise ValueError(
            "--exclude-episodes names episode indices of one dataset, but this build has "
            f"{len(src_roots)} source roots. Mark the episodes in the GUI instead, which records "
            "the choice per dataset."
        )
    excluded: dict[Path, set[int]] = {}
    for root in src_roots:
        root_excluded = set(explicit or ())
        if respect_annotations:
            root_excluded |= annotated_excluded_episodes(root)
        excluded[root] = root_excluded
    return excluded


def prepare_dataset_view(
    *,
    src_root: Path | Sequence[Path],
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
    camera_crop_specs: dict[str, list[int]] | None,
    copy_videos: bool,
    overwrite: bool,
    action_drop_dims: list[str] | None = None,
    action_mode: str = ACTION_MODE_ABSOLUTE_EE,
    exclude_episodes: set[int] | None = None,
    respect_annotations: bool = True,
    view_fps: int = 0,
    task_prompt: str | None = None,
    task_prompt_map: dict[str, str] | None = None,
) -> None:
    src_roots = discover_dataset_roots(src_root)
    excluded_by_root = resolve_excluded_episodes(
        src_roots, explicit=exclude_episodes, respect_annotations=respect_annotations
    )
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
    camera_crop_specs = validate_camera_crop_specs(camera_crop_specs or {}, first_features, camera_keys)
    chunks_size = int(first_info.get("chunks_size", 1000))
    resolved_fps, source_strides = resolve_frame_strides(src_roots, source_infos, view_fps)
    if any(stride > 1 for stride in source_strides):
        print(
            "[prepare] resampling to "
            f"{resolved_fps} fps: "
            + ", ".join(
                f"{root.name} {int(info['fps'])}->{resolved_fps} (keep 1 of {stride})"
                for root, info, stride in zip(src_roots, source_infos, source_strides, strict=True)
            )
        )

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

    # The prompt is training data, not a run setting: for pi0/pi0.5 the task string is tokenized
    # into every sample's context, so which words are in it changes what the model is conditioned
    # on exactly as much as which pixels are. That is why it is rewritten *here*, into the view,
    # rather than at train time -- the view is the thing a checkpoint's manifest identifies, and a
    # prompt that lived on the training command line would leave two runs over the same
    # `source_digest` having learned different language with nothing on disk saying so.
    #
    # The recording is never touched. A prompt chosen badly during capture is not a reason to
    # rewrite the only primary record of what was captured.
    task_prompt = normalize_task_prompt(task_prompt)
    task_prompt_map = {
        normalize_task_prompt(key): normalize_task_prompt(value)
        for key, value in (task_prompt_map or {}).items()
    }
    if task_prompt and task_prompt_map:
        raise ValueError(
            "--task-prompt and --task-prompt-map both rewrite the same column. Pass one: "
            "--task-prompt to give every episode the same instruction, --task-prompt-map to "
            "rewrite each recorded prompt separately."
        )

    def rewrite_task(name: str) -> str:
        if task_prompt:
            return task_prompt
        return task_prompt_map.get(normalize_task_prompt(name), str(name))

    global_task_to_index: dict[str, int] = {}
    task_index_maps: list[dict[int, int]] = []
    source_task_names: list[str] = []
    for root in src_roots:
        tasks = pd.read_parquet(root / "meta/tasks.parquet")
        source_map: dict[int, int] = {}
        for task, row in tasks.iterrows():
            source_name = str(task)
            if source_name not in source_task_names:
                source_task_names.append(source_name)
            # Two source prompts can rewrite to one string, and then they *are* one task: the
            # view's task_index space is rebuilt here, so the frames of both end up pointing at
            # the same row and the policy sees one instruction. That is the point of the map.
            task_name = rewrite_task(source_name)
            if task_name not in global_task_to_index:
                global_task_to_index[task_name] = len(global_task_to_index)
            source_map[int(row["task_index"])] = global_task_to_index[task_name]
        task_index_maps.append(source_map)
    # Named a prompt no recording has: almost always a typo or a stale copy of the recorder
    # config, and silently ignoring it would train the old wording while the command line says
    # otherwise. Compared after normalization so a trailing space is not what makes it "absent".
    unmatched = sorted(
        key
        for key in task_prompt_map
        if key not in {normalize_task_prompt(name) for name in source_task_names}
    )
    if unmatched:
        raise ValueError(
            f"--task-prompt-map names prompt(s) no source dataset records: {unmatched}. "
            f"Recorded prompts are: {sorted(source_task_names)}"
        )

    dst_root.mkdir(parents=True)
    (dst_root / "meta").mkdir(parents=True, exist_ok=True)

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
    source_frame_offsets: list[int] = []
    # Source episode index -> index in this view, per source root. Excluded episodes are absent,
    # and the survivors are renumbered contiguously: LeRobot addresses episodes by position
    # (`splits: 0:N`, one meta/episodes row each), so a gap would be a dataset that claims
    # episodes it does not have.
    source_episode_maps: list[dict[int, int]] = []
    total_rows = 0
    total_episodes = 0

    for source_root_index, root in enumerate(src_roots):
        data_files = sorted((root / "data").glob("*/*.parquet"))
        if not data_files:
            raise FileNotFoundError(f"No parquet files found under {root / 'data'}")
        episodes_files = sorted((root / "meta/episodes").glob("*/*.parquet"))
        if not episodes_files:
            raise FileNotFoundError(f"No episode metadata files found under {root / 'meta/episodes'}")
        episodes = pd.concat([pq.read_table(path).to_pandas() for path in episodes_files], ignore_index=True)
        # Physical row order, which is what the re-counted dataset_from/to_index below assume.
        # It is normally episode order too, but that is a convention, not a guarantee.
        episodes = episodes.sort_values("dataset_from_index").reset_index(drop=True)

        excluded = excluded_by_root.get(root, set())
        unknown = sorted(excluded - {int(index) for index in episodes["episode_index"]})
        if unknown:
            raise ValueError(f"{root} has no episode(s) {unknown} to exclude.")
        if excluded:
            # Keep source order: the rows are laid out that way in the parquet files, and the new
            # dataset_from/to_index below count through them in exactly that order.
            episodes = episodes[~episodes["episode_index"].isin(excluded)].reset_index(drop=True)
            print(f"[prepare] excluding {len(excluded)} episode(s) from {root.name}: {sorted(excluded)}")
        if episodes.empty:
            raise ValueError(f"Every episode of {root} is excluded; there is nothing to build.")

        stride = source_strides[source_root_index]
        if stride > 1:
            # Ceiling, because frame 0 of every episode is always kept: an episode of 5 frames
            # at stride 2 keeps frames 0, 2 and 4. This has to happen before the row offsets
            # below are accumulated -- they count through these lengths.
            episodes = episodes.copy()
            lengths = episodes["length"].to_numpy()
            episodes["length"] = ((lengths + stride - 1) // stride).astype(episodes["length"].dtype)

        source_data_files.append(data_files)
        source_episodes.append(episodes)
        source_frame_offsets.append(total_rows)
        source_episode_maps.append(
            {
                int(source_index): total_episodes + position
                for position, source_index in enumerate(episodes["episode_index"])
            }
        )
        file_map: dict[tuple[int, int], tuple[int, int]] = {}
        for src_file in data_files:
            old_pair = chunk_file_from_path(src_file.relative_to(root))
            file_map[old_pair] = chunk_file_for_index(len(file_map) + sum(len(m) for m in source_file_maps), chunks_size)
        source_file_maps.append(file_map)
        total_rows += int(episodes["length"].sum())
        total_episodes += len(episodes)

    for source_idx, root in enumerate(src_roots):
        file_map = source_file_maps[source_idx]
        for cam in camera_keys:
            for old_pair, new_pair in file_map.items():
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
                crop = camera_crop_specs.get(cam)
                if crop is not None:
                    crop_video_file(src_video, dst_video, crop)
                else:
                    copy_or_symlink_file(src_video, dst_video, copy=copy_videos)

    processed_rows = 0
    delta_action_names: list[str] = []
    delta_reports: list[dict] = []
    for source_idx, (root, data_files) in enumerate(zip(src_roots, source_data_files, strict=True)):
        features = source_infos[source_idx]["features"]
        stride = source_strides[source_idx]
        file_map = source_file_maps[source_idx]
        frame_offset = source_frame_offsets[source_idx]
        episode_map = source_episode_maps[source_idx]
        task_index_map = task_index_maps[source_idx]
        # Rows read from this source (which indexes --action-npy, written for the *source*) and
        # rows actually written (which numbers the view). Exclusions make the two diverge.
        source_read_rows = 0
        source_written_rows = 0

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

            source_df = pq.read_table(src_file).to_pandas()
            # Excluded rows are dropped before anything is derived from them: a delta computed
            # across the seam of a removed episode would be an operator command that never
            # happened, and the statistics would describe frames the view does not contain.
            keep_mask = source_df["episode_index"].isin(episode_map).to_numpy()
            if stride > 1:
                # Decimation joins the same mask, and for the same reason: the delta must be
                # derived *after* the rows are thinned, so that it spans one view frame rather
                # than one source frame. Differencing first and dropping rows second would
                # discard the motion that happened in the dropped frames outright.
                #
                # frame_index is 0-based within its episode, so this keeps frame 0 of every
                # episode -- the pose the whole rollout aligns on.
                keep_mask = keep_mask & ((source_df["frame_index"].to_numpy() % stride) == 0)
            df = source_df[keep_mask].reset_index(drop=True)
            file_source_rows = len(source_df)
            if df.empty:
                # Every episode in this file was excluded. The video file it maps to stays
                # symlinked and unreferenced, which costs nothing and keeps file numbering stable.
                source_read_rows += file_source_rows
                continue

            out = pd.DataFrame()
            # timestamp is carried over untouched. The kept rows already sit at 1/view_fps
            # spacing (source frame n*stride is at n*stride/source_fps == n/view_fps), and it is
            # what addresses the mp4, which is not re-encoded. Recomputing it from the new index
            # would give the same numbers with float drift against the actual video frame times.
            out["timestamp"] = df["timestamp"]
            # Exact because only multiples of stride survive the mask above.
            out["frame_index"] = (df["frame_index"] // stride) if stride > 1 else df["frame_index"]
            out["episode_index"] = df["episode_index"].map(episode_map).astype(source_df["episode_index"].dtype)
            # Renumbered rather than offset: `index` is the row's position in the whole dataset,
            # and dropping rows moves every later one.
            out["index"] = np.arange(len(df), dtype=np.int64) + frame_offset + source_written_rows
            out["task_index"] = df["task_index"].map(task_index_map)
            if out["task_index"].isna().any():
                missing = sorted(set(df.loc[out["task_index"].isna(), "task_index"].tolist()))
                raise ValueError(f"Missing task index mapping for {root}: {missing}")
            out["task_index"] = out["task_index"].astype(df["task_index"].dtype)

            if state_keys:
                state = select_state_matrix(df, features, state_keys)
                out["observation.state"] = list(state)
                state_parts.append(state)

            if loaded_action_npy is not None:
                # Sliced by *source* position and then filtered with the same mask: the npy was
                # written against the recording, so it still has a row per excluded frame.
                action = loaded_action_npy[source_read_rows : source_read_rows + file_source_rows]
                action = action[keep_mask]
            else:
                action = as_matrix(df[action_key], action_key, feature_dim(features[action_key]))
            if is_delta_action_mode(action_mode):
                # Differenced here, on the base action and before any appended columns, so the
                # delta spans exactly one dataset frame. Episode boundaries are respected inside
                # derive_delta_action; the call self-checks by rebuilding the absolute stream.
                action, delta_action_names, delta_report = derive_delta_action(
                    absolute_action=action,
                    action_names=list(features[action_key]["names"]),
                    observation_state=as_matrix(
                        df["observation.state"],
                        "observation.state",
                        feature_dim(features["observation.state"]),
                    ),
                    observation_names=list(features["observation.state"]["names"]),
                    episode_index=np.asarray(df["episode_index"]),
                    action_mode=action_mode,
                )
                action = action.astype(np.float32)
                delta_reports.append(delta_report)
            if action_drop_dims:
                # Dropped after the delta is derived, never before: the reference pose is rebuilt
                # from all six components, so differencing a already-truncated action would put
                # the rotation error into the axes that are kept. Deployment restores the dropped
                # axes as exact zeros -- see fr3_act_infer_real_runtime.decode_action_to_robot_command.
                base_action_names = (
                    delta_action_names
                    if is_delta_action_mode(action_mode)
                    else list(features[action_key].get("names") or [])
                )
                if len(base_action_names) != action.shape[1]:
                    raise ValueError(
                        f"--action-drop-dims needs named action dims; {action_key} has "
                        f"{action.shape[1]} columns and {len(base_action_names)} names."
                    )
                keep_columns = [
                    index
                    for index in range(action.shape[1])
                    if index not in set(action_drop_column_indices(base_action_names, action_drop_dims))
                ]
                action = action[:, keep_columns]
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
            source_read_rows += file_source_rows
            source_written_rows += len(df)
            processed_rows += len(df)

        if loaded_action_npy is not None and len(loaded_action_npy) != source_read_rows:
            raise ValueError(f"{source_action_npy} has {len(loaded_action_npy)} rows, dataset has {source_read_rows}")

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
        frame_offset = source_frame_offsets[source_idx]
        episode_map = source_episode_maps[source_idx]
        out_episodes = episodes.copy()
        out_episodes["episode_index"] = [
            episode_map[int(index)] for index in out_episodes["episode_index"]
        ]
        # Kept in step with tasks.parquet by hand. Nothing in the training path reads this column
        # -- LeRobotDataset resolves a frame's prompt through `task_index` into tasks.parquet --
        # but it is what a person opens to ask what an episode was, and a view whose two records
        # of the same fact disagree is worse than one that never had the second.
        out_episodes["tasks"] = [
            rewrite_episode_tasks(value, rewrite_task) for value in out_episodes["tasks"]
        ]
        # Recounted from the kept lengths rather than shifted by a constant: with an episode
        # removed, every later episode starts earlier than it did in the source. The video
        # timestamps are deliberately untouched -- the mp4 files are symlinked whole, so each
        # surviving episode's from/to range still addresses its own frames.
        lengths = out_episodes["length"].to_numpy()
        starts = frame_offset + np.concatenate([[0], np.cumsum(lengths)[:-1]]).astype(lengths.dtype)
        out_episodes["dataset_from_index"] = starts
        out_episodes["dataset_to_index"] = starts + lengths
        for col_prefix in ["data", *[f"videos/{cam}" for cam in camera_keys]]:
            chunk_col = f"{col_prefix}/chunk_index"
            file_col = f"{col_prefix}/file_index"
            if chunk_col not in out_episodes or file_col not in out_episodes:
                continue
            new_pairs = [
                file_map[(int(chunk), int(file))]
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
        new_features[cam] = resize_camera_feature(first_features[cam], image_resize_shape, camera_crop_specs.get(cam))
    action_names = first_features.get(action_key, {}).get("names")
    base_action_dim = all_action.shape[1] - len(append_feature_names)
    if is_delta_action_mode(action_mode):
        # The delta names carry the reference, which is what makes the view self-describing:
        # an offline tool can tell from the column names alone how to integrate it back.
        action_names = delta_action_names
    if action_drop_dims and action_names is not None:
        dropped = set(action_drop_dims)
        action_names = [name for name in action_names if name not in dropped]
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
    new_info["fps"] = int(resolved_fps)
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

    # What this build was made of, as one comparable value. Views are rebuilt under the same
    # name when a task gains a session, so `view.root` alone no longer identifies the frames a
    # checkpoint trained on -- a checkpoint that records this digest can still tell whether the
    # view on disk is the one it saw, and the build id says when it was replaced.
    source_digest = hashlib.sha256(
        json.dumps(
            {
                "sources": [
                    {
                        "root": str(root),
                        "excluded": sorted(excluded_by_root.get(root, set())),
                        "stride": int(stride),
                    }
                    for root, stride in zip(src_roots, source_strides, strict=True)
                ],
                "cameras": camera_keys,
                "state_keys": state_keys,
                # In the digest because the prompt is part of what the frames teach: two views
                # over the same episodes with different instructions must not compare equal, or a
                # checkpoint could claim to have trained on a view it never saw.
                "task_prompts": [
                    task for task, _ in sorted(global_task_to_index.items(), key=lambda item: item[1])
                ],
                "action_mode": action_mode,
                "action_key": None if action_npy else action_key,
                "action_drop_dims": action_drop_dims or [],
                "action_append_selectors": action_append_selectors,
                "action_append_shift": action_append_shift,
                "image_resize_shape": image_resize_shape,
                "camera_crop_specs": camera_crop_specs,
                "fps": int(resolved_fps),
            },
            sort_keys=True,
            default=str,
        ).encode("utf-8")
    ).hexdigest()[:16]

    manifest = {
        "source_dataset_root": str(src_roots[0]) if len(src_roots) == 1 else None,
        "source_dataset_roots": [str(root) for root in src_roots],
        "build_id": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "source_digest": source_digest,
        "repo_id": repo_id,
        "cameras": camera_keys,
        "state_keys": state_keys,
        "action_key": None if action_npy else action_key,
        "action_npy": str(action_npy) if action_npy else None,
        "action_drop_dims": action_drop_dims or [],
        "action_append_selectors": action_append_selectors,
        "action_append_names": append_feature_names,
        "action_append_shift": action_append_shift,
        "image_resize_shape": image_resize_shape,
        "camera_crop_specs": camera_crop_specs,
        # The language the policy is conditioned on, in task_index order, and where it came
        # from. A rollout has to send the same string it trained on -- pi0.5 takes the prompt
        # from the caller at inference, not from the checkpoint -- so this is the value the
        # generated inference config carries forward.
        "task_prompts": [
            task for task, _ in sorted(global_task_to_index.items(), key=lambda item: item[1])
        ],
        "source_task_prompts": source_task_names,
        "task_prompt_override": task_prompt or None,
        "task_prompt_map": task_prompt_map,
        "state_dim": int(all_state.shape[1]) if state_keys else 0,
        "action_dim": int(all_action.shape[1]),
        # The view's rate and what it was resampled from. Without this the fps in info.json is
        # indistinguishable from a recording that was captured at that rate, and a later merge
        # cannot tell whether these frames have already been thinned.
        "fps": int(resolved_fps),
        "source_fps": {
            str(root): int(info.get("fps") or 0)
            for root, info in zip(src_roots, source_infos, strict=True)
        },
        "frame_stride": {
            str(root): int(stride) for root, stride in zip(src_roots, source_strides, strict=True)
        },
        "total_episodes": int(total_episodes),
        "total_rows": int(total_rows),
        # What was left out and where each surviving episode came from. Episodes are renumbered
        # here, so without this a view's episode 4 could not be traced back to the recording --
        # and a training set that silently differs from its source is not auditable.
        "excluded_episodes": {
            str(root): sorted(excluded_by_root.get(root, set())) for root in src_roots
        },
        "episode_source_index": [
            {"episode_index": view_index, "source_dataset_root": str(root), "source_episode_index": source_index}
            for root, episode_map in zip(src_roots, source_episode_maps, strict=True)
            for source_index, view_index in sorted(episode_map.items(), key=lambda item: item[1])
        ],
        # Recorded so the action contract of this view, and the evidence that the conversion was
        # invertible, are auditable from the dataset rather than only from the command line.
        "action_mode": action_mode,
        "delta_transform": _summarize_delta_reports(
            delta_reports,
            delta_action=all_action,
            delta_names=action_names,
            append_names=append_feature_names,
            # The view's rate, not the recording's: this is what turns a per-frame delta into
            # mm/s, and using the source rate on a decimated view reports a 2x speed.
            fps=int(resolved_fps),
        )
        if is_delta_action_mode(action_mode)
        else None,
    }
    write_json(dst_root / "meta/il_view_manifest.json", manifest)
    if is_delta_action_mode(action_mode):
        scale = manifest["delta_transform"]["per_frame_scale"]
        print(
            f"[prepare] action_mode={action_mode} "
            f"per-frame translation p95={scale['p95_translation_per_frame_mm']:.3f} mm "
            f"(implied {scale['implied_p95_speed_mm_s']:.1f} mm/s at {resolved_fps} fps), "
            f"reconstruction max error {manifest['delta_transform']['reconstruction_max_position_error_mm']:.5f} mm"
        )


def _summarize_delta_reports(
    reports: list[dict],
    *,
    delta_action: np.ndarray,
    delta_names: list[str],
    append_names: list[str],
    fps: int,
) -> dict:
    base_columns = delta_action.shape[1] - len(append_names)
    return {
        "frames": int(sum(int(report["frames"]) for report in reports)),
        "episodes": int(sum(int(report["episodes"]) for report in reports)),
        "reconstruction_max_position_error_mm": max(
            (float(report["reconstruction_max_position_error_mm"]) for report in reports), default=0.0
        ),
        "reconstruction_max_rotation_error_deg": max(
            (float(report["reconstruction_max_rotation_error_deg"]) for report in reports), default=0.0
        ),
        "per_frame_scale": summarize_delta_scale(
            delta_action=delta_action[:, :base_columns].astype(np.float64),
            delta_names=delta_names[:base_columns],
            fps=fps,
        ),
    }


def build_policy_section(args: argparse.Namespace, image_resize_shape: list[int] | None) -> dict[str, Any]:
    """The `policy` block of the generated train config.

    `act` and `diffusion` are spelled out explicitly, from flags that default to the
    upstream dataclass values -- writing them into the generated config records what the
    run used without changing it. Any other type gets the common keys only: its own
    dataclass defaults are the closest thing to a measured value that exists for it here,
    and overriding them with numbers borrowed from ACT would be worse than not overriding
    them at all. `--policy-config` is applied last so an operator who *has* measured
    something can say so.
    """
    policy: dict[str, Any] = {
        "type": args.policy,
        "device": None if args.device == "auto" else args.device,
        "use_amp": args.use_amp,
        "push_to_hub": False,
    }
    if args.pretrained_path:
        # Weights only. `make_policy` passes this config into `from_pretrained`, so the
        # checkpoint supplies parameters and everything here supplies shape and hyperparameters
        # -- which is what lets a pi0.5 base trained on someone else's robot finetune onto this
        # rig's action dimension without the two configs having to agree.
        #
        # Deliberately not `use_peft`: that flag means "the path points at an adapter", and
        # setting it for a LoRA *run* would send make_policy looking for an adapter config in a
        # base model. lerobot_train wraps the policy itself once it sees the `peft` block.
        policy["pretrained_path"] = str(args.pretrained_path)
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
    elif args.policy == "diffusion":
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
    policy.update(parse_policy_config(args.policy_config))
    return policy


def build_peft_section(args: argparse.Namespace) -> dict[str, Any] | None:
    """The top-level `peft` block of the generated train config, or None for a dense run.

    Mirrors `lerobot.configs.default.PeftConfig` exactly, which is a shorter list than the LoRA
    knobs people expect: rank, targets, method, init and which modules stay fully trainable.
    There is no `lora_alpha` or `lora_dropout` here because that dataclass has no field for
    them -- draccus parses this block into it, and an extra key is a hard parse error rather
    than an ignored one. Adapting those two means widening PeftConfig upstream.

    Keys the operator did not set are omitted rather than sent as null so the policy's own
    defaults apply: `PreTrainedPolicy._build_peft_config` skips None values, but only after
    `PeftConfig`'s defaults have already overwritten what the policy asked for -- and for pi0.5
    the policy's default target set is the tuned one.
    """
    if not args.lora:
        return None
    peft: dict[str, Any] = {"method_type": args.peft_method, "r": int(args.lora_r)}
    if args.lora_target_modules is not None:
        spec = args.lora_target_modules.strip()
        # A single token is passed through as-is: 'all-linear' is a PEFT keyword and a regex is
        # one string. Splitting either on commas would turn it into a list of one and change
        # what PEFT matches.
        peft["target_modules"] = parse_csv(spec) if "," in spec else spec
    if args.lora_full_training_modules is not None:
        peft["full_training_modules"] = parse_csv(args.lora_full_training_modules)
    if args.lora_init_type is not None:
        peft["init_type"] = args.lora_init_type
    return peft


def validate_prompt_args(args: argparse.Namespace) -> None:
    """A prompt rewrite belongs to a view being built; refuse it when no view is being built.

    `--skip-prepare` trains frames someone else already wrote, and their task column is already
    on disk. Accepting the flag there would silently train the old wording while the command
    line -- and the operator's memory of what they asked for -- said otherwise.
    """
    if not (args.task_prompt or args.task_prompt_map):
        return
    if args.skip_prepare:
        raise ValueError(
            "--task-prompt rewrites the view's task column while the view is built, and "
            "--skip-prepare trains a view that already exists. Rebuild the view with the new "
            "prompt (Dataset Export page, or --prepare-only), then train it."
        )
    if args.resume:
        raise ValueError(
            "--task-prompt cannot be applied to a resumed run: the checkpoint was trained on "
            "the prompt already in its view, and changing the language mid-run trains one model "
            "on two different instructions. Start a fresh run over a rebuilt view."
        )


def validate_finetune_args(args: argparse.Namespace) -> None:
    """Refuse LoRA without a base model, here rather than 300 seconds into a training run.

    `PreTrainedPolicy._validate_peft_config` raises the same refusal, but only after the
    dataset has been scanned, the dataloader built and the policy constructed -- and on a
    remote training host that failure arrives as a line in a log file nobody is watching yet.
    """
    if args.lora and not args.pretrained_path:
        raise ValueError(
            "--lora adapts a pretrained model and there is nothing here to adapt. Pass "
            "--pretrained-path (for example --pretrained-path lerobot/pi05_base), or drop "
            "--lora to train from scratch."
        )


def resolve_peak_lr(args: argparse.Namespace, policy: dict[str, Any]) -> float:
    """Peak LR for the warmup/decay scheduler.

    Read back off the resolved policy block so `--policy-config` wins, and so an
    untuned policy that never set `optimizer_lr` fails here with an instruction
    rather than scheduling a decay from a learning rate borrowed from ACT.
    """
    lr = policy.get("optimizer_lr")
    if lr is not None:
        return float(lr)
    raise ValueError(
        f"--lr-scheduler=cosine_decay_with_warmup needs a peak learning rate, and policy "
        f"{args.policy!r} has no tuned default here. Pass it explicitly, for example "
        f"--policy-config '{{\"optimizer_lr\": 1e-5}}'."
    )


def adopt_existing_view(args: argparse.Namespace, view_root: Path) -> dict[str, Any]:
    """Load the manifest of a view built earlier and make `args` describe *that* view.

    Training and export are separate steps here: the Dataset Export page builds a view with
    ``--prepare-only`` and QC gates it, and the Training page trains that view, possibly on
    another machine that never had the source recording. Without this the train step would
    have to name a ``--dataset-root``, and the only path it can be sure exists is the view
    itself -- which would re-derive a delta action column from frames whose action column is
    already a delta, silently squaring the contract.

    The shape keys are taken from the manifest rather than from the CLI because they are
    facts about frames that already exist. A ``--cameras`` that disagreed with the view
    would not fail here; it would fail thousands of steps into training, or not at all.
    """
    manifest_path = view_root / "meta" / "il_view_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(
            f"--skip-prepare needs a training view at {view_root}, but {manifest_path} is not there. "
            "Build the view first (Dataset Export page, or --prepare-only)."
        )
    manifest = load_json(manifest_path)
    args.cameras = ",".join(manifest["cameras"])
    args.state_keys = ",".join(manifest["state_keys"])
    args.action_drop_dims = ",".join(manifest.get("action_drop_dims") or [])
    args.action_append_selectors = ",".join(manifest["action_append_selectors"])
    args.action_append_names = ",".join(manifest["action_append_names"])
    args.action_mode = manifest["action_mode"]
    # Only when the caller did not ask for one: a resize is a training-time transform, so
    # overriding it is legitimate, but inheriting it is what keeps the generated config
    # consistent with a view whose videos were already re-encoded at that size.
    if args.image_resize_shape is None and manifest.get("image_resize_shape"):
        args.image_resize_shape = ",".join(str(value) for value in manifest["image_resize_shape"])
    if manifest.get("repo_id"):
        args.repo_id = manifest["repo_id"]
    print(
        f"[prepare] view manifest: {manifest['total_episodes']} episodes / "
        f"{manifest['total_rows']} rows @ {manifest.get('fps', '?')} fps, "
        f"action_mode={manifest['action_mode']}"
    )
    return manifest


def make_train_config(args: argparse.Namespace, view_root: Path, repo_id: str, config_path: Path) -> None:
    image_resize_shape = parse_hw(args.image_resize_shape)
    policy = build_policy_section(args, image_resize_shape)

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
    peft = build_peft_section(args)
    if peft is not None:
        config["peft"] = peft
    if args.lr_scheduler == "cosine_decay_with_warmup":
        peak_lr = resolve_peak_lr(args, policy)
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


def view_task_prompts(view_root: Path) -> list[str]:
    """The prompts a built view actually holds, in task_index order.

    Read from the view's own tasks.parquet rather than from its manifest: this is the table
    `LeRobotDataset.__getitem__` indexes to attach `task` to a sample, so it is the only record
    that cannot disagree with what a policy will be trained on. It also works for views built
    before the manifest carried prompts at all.
    """
    tasks_path = view_root / "meta" / "tasks.parquet"
    if not tasks_path.is_file():
        return []
    tasks = pd.read_parquet(tasks_path)
    return [str(task) for task, _ in sorted(tasks.iterrows(), key=lambda item: int(item[1]["task_index"]))]


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
    task_prompts = view_task_prompts(view_root)
    checkpoint_path = args.output_dir / "checkpoints" / "last"

    # Hardware comes from the config the data was recorded with, not from literals here. Both of
    # the literals this replaced were wrong for the workstation (a hikrobot camera file on a
    # RealSense rig, and 192.168.1.208 for a robot at .206), and neither was reachable from the
    # record config, so nothing could have caught the drift. The launcher script derives the same
    # values independently; the point of deriving both from one file is that they cannot disagree.
    record_config_path = Path(args.record_config)
    if not record_config_path.is_absolute():
        record_config_path = _REPO_ROOT / record_config_path
    record_config: dict[str, Any] = {}
    if record_config_path.exists():
        record_config = load_yaml(record_config_path)
    else:
        print(f"[prepare] WARNING: record config not found: {record_config_path}")
    record_robot = record_config.get("robot") or {}
    camera_config = resolve_inference_camera_config(record_config, camera_keys, _REPO_ROOT)
    if camera_config is None:
        print(
            "[prepare] WARNING: no inference camera config matches the recording's cameras "
            f"({', '.join(camera_suffixes)}); writing camera_config: null. Set it by hand, or "
            "pass --record-config for the rig this view came from."
        )
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
            "action_drop_dims": parse_csv(args.action_drop_dims),
            "action_append_selectors": action_append_selectors,
            "action_append_names": action_append_names,
            "action_append_shift": args.action_append_shift,
            "pretrained_path": str(args.pretrained_path) if args.pretrained_path else None,
            "peft": build_peft_section(args),
            "task_prompts": task_prompts,
        },
        "runtime": {
            "checkpoint": str(checkpoint_path),
            "dataset_root": str(view_root),
            "camera_config": camera_config,
            # The rate the view was built at, so a rollout can be paced to the data it learned
            # from rather than to whatever the recorder happened to run at.
            "policy_fps": None,
            "max_steps": None,
            "preview": True,
            # The instruction a language-conditioned policy must be given at rollout. pi0/pi0.5
            # read the task from the caller, not from the checkpoint, so a rollout that sends a
            # different string than the view was built with is running a prompt the model was
            # never trained on -- and nothing raises, it just behaves worse.
            #
            # A view with several prompts leaves this null on purpose: which one a given rollout
            # wants is the operator's choice, and picking the first would look like an answer.
            # The FR3 rollout runtime feeds this to language-conditioned policies. A single-task
            # view can be auto-resolved; multi-task views must provide an explicit rollout prompt.
            "task_prompt": task_prompts[0] if len(task_prompts) == 1 else None,
            "task_prompts": task_prompts,
            "rtc": {
                # auto keeps ACT on its checkpoint queue and enables RTC for pi0/pi0.5/SmolVLA.
                "mode": "auto",
                "execution_horizon": 10,
                "max_guidance_weight": 10.0,
                "prefix_attention_schedule": "EXP",
                "replan_queue_size": 30,
                "inference_delay_steps": None,
            },
            "hardware": {
                "robot_ip": record_robot.get("robot_ip"),
                "gripper_backend": "pika",
                "gripper_port": record_robot.get("gripper_port"),
                # Not read by the runtime yet -- recorded so a rollout can check the frame it is
                # about to solve IK in against the frame this view's poses are expressed in. The
                # two Pika frames are 410.85 mm apart and naming the wrong one does not fail.
                "target_frame_name": record_robot.get("target_frame_name"),
            },
            "recorded_with": str(record_config_path.relative_to(_REPO_ROOT))
            if record_config_path.is_relative_to(_REPO_ROOT)
            else str(record_config_path),
            "startup": {
                # The DAS rig's joint configuration, not this one's. T_B_Ws is solved from the
                # first observation against the dataset start pose, so homing somewhere the
                # episodes were not recorded from offsets every target. Home with
                # robot_init_state, or the launcher's own homing step.
                "move_to_das_start": False,
                "align_gripper_to_dataset_start": True,
                "dataset_start_gripper_tolerance": 0.05,
            },
            "safety": {
                "first_frame_max_pos_delta_mm": 20.0,
                "first_frame_max_rot_delta_deg": 8.0,
                # Bounds the policy's own per-step delta, measured against prev_cmd. Sized from the
                # recorded action distribution: 5 mm admits 99.90% of demonstrated frames.
                "max_step_pos_delta_mm": 5.0,
                "max_step_rot_delta_deg": 2.0,
                # Bounds how far the command may lead the measured pose. That gap is servo tracking
                # lag, which reaches p95 10.65 mm in the demonstrations, so it needs its own much
                # looser limit -- holding it to the per-step budget clamps every step of a healthy
                # rollout.
                "max_leash_pos_delta_mm": 20.0,
                "max_leash_rot_delta_deg": 8.0,
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
    from lerobot.policies.factory import make_policy, make_policy_config

    meta = LeRobotDatasetMetadata(repo_id, root=view_root)
    # Built from the same block the train config gets, minus the keys that describe the
    # training run rather than the policy. A smoke test that constructed the policy a
    # different way from training would be testing the smoke test.
    policy_section = build_policy_section(args, parse_hw(args.image_resize_shape))
    cfg_kwargs = {
        key: value
        for key, value in policy_section.items()
        if key not in ("type", "device", "use_amp") and value is not None
    }
    cfg = make_policy_config(args.policy, **cfg_kwargs)
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

    # Run the forward pass for whichever policy was asked for. Previously this was gated
    # on `act`, so `--smoke` on any other type silently checked the dataset and nothing
    # else -- an all-clear that had never touched the model.
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
    print(f"[smoke] {args.policy} forward loss={float(loss.detach().cpu()):.6f} details={loss_dict}")


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument(
        "--dataset-roots",
        type=Path,
        nargs="*",
        default=[],
        metavar="ROOT",
        help=(
            "Explicit list of source datasets to merge into one view, overriding --dataset-root. "
            "Each entry must be a dataset root. Merging happens here rather than after the fact "
            "because a view renumbers its episodes and computes meta/stats.json over the whole "
            "set: two views built separately cannot be combined later, and adding a session to "
            "an existing view means rebuilding it from every source at once."
        ),
    )
    parser.add_argument("--repo-id", default="single_cube2_il_view")
    parser.add_argument(
        "--policy",
        default="act",
        metavar="TYPE",
        help=(
            "Policy type to train. Any type in the LeRobot registry is accepted; known ones are "
            + ", ".join(KNOWN_POLICY_TYPES)
            + ". Only "
            + " and ".join(FLAGGED_POLICY_TYPES)
            + " have dedicated hyperparameter flags, and those default to the upstream policy "
            "dataclass values; for the rest, the policy's own dataclass defaults apply. "
            "--policy-config overrides either."
        ),
    )
    parser.add_argument(
        "--policy-config",
        default="",
        metavar="JSON",
        help=(
            "JSON object of policy hyperparameters merged into the generated train config, "
            'for example \'{"chunk_size": 50, "optimizer_lr": 2.5e-5}\'. Applied after the '
            "built-in defaults, so it wins."
        ),
    )
    parser.add_argument(
        "--pretrained-path",
        default=None,
        metavar="REPO_OR_DIR",
        help=(
            "Base weights to finetune from: a Hugging Face repo id (lerobot/pi05_base) or a "
            "local directory saved by `save_pretrained`. Written to policy.pretrained_path, so "
            "the weights come from there while every hyperparameter still comes from this "
            "script's generated config. Required by --lora, which has nothing to adapt without "
            "a pretrained model."
        ),
    )
    parser.add_argument(
        "--lora",
        action="store_true",
        help=(
            "Train a PEFT adapter instead of the whole network: the base weights are frozen and "
            "only the adapter (plus --lora-full-training-modules) gets gradients. Emits the "
            "top-level `peft` block lerobot_train reads. Needs --pretrained-path and the `peft` "
            "package (pip install -e '.[fr3-train]')."
        ),
    )
    parser.add_argument(
        "--peft-method",
        default="LORA",
        metavar="METHOD",
        help="PEFT method name, resolved against peft.PeftType (LORA, MISS, ...). Default LORA.",
    )
    parser.add_argument(
        "--lora-r",
        type=int,
        default=16,
        help="Adapter rank. Upstream PeftConfig default is 16; higher is closer to full finetuning.",
    )
    parser.add_argument(
        "--lora-target-modules",
        default=None,
        metavar="SPEC",
        help=(
            "Which modules to adapt: a comma-separated list of name suffixes, the literal "
            "'all-linear', or a regex. Left unset, the policy's own default is used -- pi0.5 "
            "targets the action expert's q/v projections plus the state/action projections "
            "(see PI05Policy._get_default_peft_targets), which is the tuned answer for it."
        ),
    )
    parser.add_argument(
        "--lora-full-training-modules",
        default=None,
        metavar="LIST",
        help=(
            "Comma-separated modules to fully finetune and save alongside the adapter (PEFT's "
            "modules_to_save). Pass an empty string to force none. Left unset, the policy's "
            "default applies."
        ),
    )
    parser.add_argument(
        "--lora-init-type",
        default=None,
        metavar="INIT",
        help="Adapter initialization, passed through to the PEFT method's own init option.",
    )
    parser.add_argument("--cameras", default=DEFAULT_CAMERAS)
    parser.add_argument("--state-keys", default=DEFAULT_STATE_KEYS)
    parser.add_argument("--image-resize-shape", default=None)
    parser.add_argument(
        "--camera-crops",
        default="",
        help="JSON or semicolon specs keyed by camera, e.g. side:224,0,416,346",
    )
    parser.add_argument(
        "--task-prompt",
        default=None,
        metavar="TEXT",
        help=(
            "Rewrite every episode's language instruction to this string in the view being "
            "built. The recording is not touched. This is the prompt pi0/pi0.5/smolvla are "
            "conditioned on -- it is tokenized into every training sample -- so it is worth more "
            "than the label the recorder happened to be configured with. Rebuild the view to "
            "change it; a checkpoint's manifest records the prompt it trained on."
        ),
    )
    parser.add_argument(
        "--task-prompt-map",
        default=None,
        metavar="JSON",
        help=(
            'JSON object rewriting recorded prompts one by one, e.g. \'{"Pick and place": '
            '"pick up the red cube and place it in the box"}\'. Use instead of --task-prompt '
            "when merging recordings that carry different prompts. Naming a prompt no source "
            "records is an error, not a no-op."
        ),
    )
    parser.add_argument("--action-key", default="action")
    parser.add_argument("--action-npy", type=Path, default=None)
    parser.add_argument("--use-derived-action", action="store_true")
    parser.add_argument(
        "--action-drop-dims",
        default="",
        help=(
            "Comma-separated action dim names to leave out of the view, e.g. "
            "delta_ee_from_prev_cmd.drx,delta_ee_from_prev_cmd.dry. For axes the teleop rig locks: "
            "their recorded signal is numerical noise, and normalising a noise-only range hands it "
            "a real share of the loss. Deployment restores a dropped delta axis as an exact zero."
        ),
    )
    parser.add_argument("--action-append-selectors", default=DEFAULT_ACTION_APPEND_SELECTORS)
    parser.add_argument("--action-append-names", default=DEFAULT_ACTION_APPEND_NAMES)
    parser.add_argument("--action-append-shift", type=int, default=1)
    parser.add_argument("--view-root", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--job-name", default=None)
    parser.add_argument("--overwrite-view", action="store_true")
    parser.add_argument(
        "--action-mode",
        choices=ACTION_MODES,
        default=ACTION_MODE_ABSOLUTE_EE,
        help=(
            "Action contract for the generated training view. Recording always stores absolute EE; "
            "the delta modes are derived here as consecutive-dataset-frame differences."
        ),
    )
    parser.add_argument(
        "--exclude-episodes",
        default="",
        help=(
            "Comma-separated source episode indices to leave out of the view, on top of whatever "
            "the annotation store excludes. Single-source builds only."
        ),
    )
    parser.add_argument(
        "--respect-annotations",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Drop episodes marked includeInTraining=false in <root>/meta/gui_annotations.json. "
            "On by default: the operator's review is the point of recording it."
        ),
    )
    parser.add_argument(
        "--view-fps",
        type=int,
        default=30,
        metavar="FPS",
        help=(
            "Frame rate of the generated view. Sources faster than this are decimated to it "
            "(60 fps -> keep 1 frame of 2), which is what lets recordings captured at different "
            "rates share one view: the action column is a per-frame delta, so mixing rates puts "
            "two scales in one column. Only integer ratios are accepted. Pass 0 to keep the "
            "sources' own rate, which then requires them to already agree. Videos are not "
            "re-encoded -- rows address the mp4 by timestamp."
        ),
    )
    parser.add_argument(
        "--record-config",
        default=DEFAULT_RECORD_CONFIG,
        metavar="PATH",
        help=(
            "Recorder config the source dataset was captured with. The generated inference "
            "config takes its robot IP, gripper port, tool frame and camera file from here, so "
            "that deployment meets the hardware the data came off instead of a literal."
        ),
    )
    parser.add_argument("--copy-videos", action="store_true")
    parser.add_argument("--prepare-only", action="store_true")
    parser.add_argument(
        "--skip-prepare",
        action="store_true",
        help=(
            "Train a training view that already exists at --view-root instead of building one. "
            "The view's own manifest supplies the cameras, state keys and action contract, so "
            "--dataset-root is not consulted and need not be present on this machine. This is "
            "how the Training page runs: the view was built and QC gated as a separate step."
        ),
    )
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

    # Every default below is the upstream one, copied from ACTConfig and DiffusionConfig in
    # src/lerobot/policies/{act,diffusion}/configuration_*.py. The flags exist so a measured
    # value can be named on the command line, not so this script can hold an opinion the
    # policy's authors do not: a helper default that quietly differs from the library's turns
    # every "I ran ACT" into "I ran something near ACT", and the difference only surfaces
    # after the checkpoint disappoints. Change one here only to follow upstream changing it.
    parser.add_argument("--vision-backbone", default="resnet18")
    parser.add_argument("--act-chunk-size", type=int, default=100)
    parser.add_argument("--act-n-action-steps", type=int, default=100)
    parser.add_argument("--act-lr", type=float, default=1e-5)
    parser.add_argument("--act-lr-backbone", type=float, default=1e-5)
    parser.add_argument("--act-pretrained-backbone-weights", default="ResNet18_Weights.IMAGENET1K_V1")

    parser.add_argument("--dp-n-obs-steps", type=int, default=2)
    parser.add_argument("--dp-horizon", type=int, default=16)
    parser.add_argument("--dp-n-action-steps", type=int, default=8)
    parser.add_argument(
        "--dp-resize-shape",
        default="",
        metavar="H,W",
        help=(
            "DP-internal image resize as H,W. Empty is upstream's default: no DP-internal "
            "resize. Prefer --image-resize-shape for real-robot work, because that one is "
            "shared by training metadata, dataloader and inference; setting it disables this "
            "resize so images are not resized twice."
        ),
    )
    parser.add_argument("--dp-lr", type=float, default=1e-4)
    return parser


def main() -> None:
    args = build_arg_parser().parse_args()
    steps_supplied = cli_arg_was_supplied("--steps")
    cameras = [normalize_camera_key(key) for key in parse_csv(args.cameras)]
    state_keys = parse_csv(args.state_keys)
    action_drop_dims = parse_csv(args.action_drop_dims)
    action_append_selectors = parse_csv(args.action_append_selectors)
    action_append_names = parse_csv(args.action_append_names)
    image_resize_shape = parse_hw(args.image_resize_shape)
    camera_crop_specs = parse_camera_crop_specs(args.camera_crops)
    if not cameras:
        raise ValueError("At least one camera must be selected.")
    validate_policy_type(args.policy)
    validate_finetune_args(args)
    validate_prompt_args(args)
    if args.skip_prepare and args.prepare_only:
        raise ValueError("--skip-prepare and --prepare-only ask for opposite halves of this script.")
    if args.policy != "act" and not state_keys:
        raise ValueError(
            f"Image-only observation (no --state-keys) is only wired up for ACT here; "
            f"{args.policy!r} needs a state feature. Pass --state-keys observation.state."
        )
    if args.resume and args.lr_scheduler != "none":
        print("[resume] ignoring --lr-scheduler for resume; scheduler comes from the checkpoint config.")

    if args.use_derived_action:
        args.action_npy = DEFAULT_DERIVED_ACTION

    tag = f"{args.policy}_{'_'.join(key.split('.')[-1] for key in cameras)}"
    # The job name names the training output dir and the checkpoint path the generated inference
    # config points at, so it has to identify *this* run. A fixed default (it used to be the
    # single_cube2 task this script was written for) makes every dataset train into one directory
    # and quietly overwrite the previous checkpoints.
    # An explicit list wins over the single root: --dataset-root keeps its default, so it is
    # always set and could never signal "not asked for".
    source_roots: Path | list[Path] = list(args.dataset_roots) if args.dataset_roots else args.dataset_root
    default_job_source = (
        Path(args.dataset_roots[0]).name
        if len(args.dataset_roots) == 1
        else f"{Path(args.dataset_roots[0]).name}_plus{len(args.dataset_roots) - 1}"
        if args.dataset_roots
        else Path(args.dataset_root).name
    )
    args.job_name = args.job_name or f"{default_job_source}_{tag}"
    view_root = args.view_root or Path("outputs/datasets") / args.job_name
    args.output_dir = args.output_dir or Path("outputs/train") / args.job_name
    # A run that builds its own view puts the generated configs at the view root, where the
    # export step has always left them. A run that trains a view someone else built puts them
    # under `runs/<job>/` instead, because the view is shared: several jobs train the same
    # frames with different policies and step counts, and writing to the root would leave the
    # last job's settings sitting in the view with an inference config naming a checkpoint
    # nobody asked about.
    #
    # Not `args.output_dir`, tempting as that is: lerobot_train refuses to start when its
    # output directory already exists and it is not resuming, so creating it here to hold the
    # config would make every fresh run fail on its own scaffolding.
    config_dir = (view_root / "runs" / args.job_name) if args.skip_prepare else view_root
    config_path = config_dir / "train_config.generated.json"
    inference_config_path = config_dir / "inference_config.generated.yaml"

    if args.skip_prepare:
        manifest = adopt_existing_view(args, view_root)
        # Regenerated rather than reused: the run's policy, step count, batch size and W&B
        # settings live in this file, and the one left behind by --prepare-only describes
        # whatever was asked for at export time. The *data* keys all come from the manifest,
        # so regenerating cannot make the config disagree with the frames it points at.
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
        print(f"[prepare] training existing view: {view_root}")
        print(f"[prepare] train config: {config_path}")
        print(f"[prepare] inference config: {inference_config_path}")
    elif args.resume and view_root.exists() and not args.overwrite_view:
        print(f"[prepare] resume: keeping existing dataset view: {view_root}")
    else:
        prepare_dataset_view(
            src_root=source_roots,
            dst_root=view_root,
            repo_id=args.repo_id,
            camera_keys=cameras,
            state_keys=state_keys,
            action_key=args.action_key,
            action_npy=args.action_npy,
            action_drop_dims=action_drop_dims,
            action_append_selectors=action_append_selectors,
            action_append_names=action_append_names,
            action_append_shift=args.action_append_shift,
            image_resize_shape=image_resize_shape,
            camera_crop_specs=camera_crop_specs,
            copy_videos=args.copy_videos,
            overwrite=args.overwrite_view,
            action_mode=args.action_mode,
            exclude_episodes={int(value) for value in parse_csv(args.exclude_episodes)},
            respect_annotations=args.respect_annotations,
            view_fps=args.view_fps,
            task_prompt=args.task_prompt,
            task_prompt_map=parse_task_prompt_map(args.task_prompt_map),
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
    if args.prepare_only:
        # Nothing is written here in prepare-only mode; say so, or the last line of the run reads
        # as if a training directory had just been produced.
        print(f"[prepare] training output dir (created when this view is trained): {args.output_dir}")
    else:
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
