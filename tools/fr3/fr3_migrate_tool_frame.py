#!/usr/bin/env python

"""Re-express a recorded FR3 dataset from ``pika_task_tcp`` to ``pika_gripper_ee``.

The workstation recorded against ``pika_task_tcp`` until it switched to the tool point (see
tools/fr3/WORKSTATION_RECORDING.md). The two frames are fixed on the same URDF and share an
orientation exactly, so their separation expressed in the *tool* frame is a rigid constant and every
recorded position converts exactly::

    p_ee = p_tcp + R(quat) @ d,    d = (-0.366842, 0, 0.185) m

Nothing about that is approximate. What makes a migration wrong in practice is everything around it,
so this script does the three jobs together:

1. **Three position triplets, each with its own rotation.** ``observation.state`` carries both the
   measured pose and the previous *command* pose, and ``action`` carries a third. They are converted
   with ``ee.q``, ``prev_cmd.ee.q`` and the action's own ``ee.q`` respectively -- pairs discovered by
   feature *name* rather than by hardcoded offsets, so a schema change fails loudly instead of
   silently converting the wrong columns.
2. **Statistics recomputed, not transformed.** ``meta/stats.json`` and the per-episode
   ``stats/<feature>/*`` columns in ``meta/episodes/*.parquet`` carry min/max/mean/std/count and
   q01..q99. The conversion is nonlinear in ``quat``, so no closed form maps the old quantiles onto
   the new ones; both are recomputed from the migrated values with the same estimator
   ``fr3_train_il_policy.vector_stats`` uses.
3. **A new series root.** Nothing in a LeRobot dataset records which tool frame it is in -- ``ee.x``
   is ``ee.x`` in both -- so a migrated and an unmigrated copy are indistinguishable by schema. The
   destination is therefore always an explicit new root, and the script writes
   ``meta/fr3_tool_frame.json`` next to it as the marker the schema lacks.

The exit check is not a round trip through this file's own arithmetic. Every episode begins from the
``home`` keyframe, so frame 0 of each one is that pose measured in whichever frame the dataset uses.
The script computes the model's forward kinematics for *both* candidate frames and asks which one the
data actually sits on -- the same decision ``fr3_sim_record_replay_runtime.py`` makes at replay time,
anchored on the keyframe instead of on IK. The input must come out as ``pika_task_tcp`` (a dataset
that already reads as ``pika_gripper_ee`` is refused rather than migrated twice, which would be a
silent 822 mm error) and the output must come out as ``pika_gripper_ee``, by a margin no arithmetic
slip could fake.

Usage, one dataset at a time -- the rename is the safety mechanism, so it is never derived::

    .venv-fr3/bin/python tools/fr3/fr3_migrate_tool_frame.py \\
        --src outputs/datasets/fr3_spacemouse_20260813_160401 \\
        --dst outputs/datasets/fr3_spacemouse_eeframe_20260813_160401

Needs ``mujoco`` (for the keyframe check), ``pyarrow`` and ``numpy``: run it with ``.venv-fr3``.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

sys.path.insert(0, str(Path(__file__).resolve().parent))

# The frame identification lives in fr3_tool_frame_check so the replay preflight and this tool
# cannot end up with two answers to the same geometric question.
from fr3_tool_frame_check import (  # noqa: E402  (path set up above)
    NEW_FRAME,
    OLD_FRAME,
    TOOL_FRAME_OFFSET_M,
    ToolFrameError,
    home_frame_positions,
    require_frame,
)

# Feature name prefixes that carry a pose. Each yields one (position, quaternion) pair per feature
# that declares all seven names.
POSE_PREFIXES = ("", "prev_cmd.")
STAT_FEATURES = ("observation.state", "action")


class MigrationError(RuntimeError):
    """Anything that would produce a dataset nobody can tell is wrong by looking at it."""


def check_frame(label: str, first_positions: np.ndarray, home: dict[str, np.ndarray], expected: str) -> None:
    """require_frame, with the two failures this tool can produce spelled out for the operator."""
    try:
        errors = require_frame(label, first_positions, expected, home)
    except ToolFrameError as exc:
        hint = (
            " It looks like it has already been migrated; running again would move it another 410.85 mm."
            if expected == OLD_FRAME
            else " The conversion did not land where it should have."
        )
        raise MigrationError(f"{exc}{hint}") from exc
    report = ", ".join(f"{name} {errors[name] * 1e3:.1f} mm" for name in sorted(errors))
    print(f"  [frame] {label}: {expected}  ({report})")


# --------------------------------------------------------------------------- pose conversion ---


def quaternion_matrices(quats: np.ndarray) -> np.ndarray:
    """Rotation matrices from (x, y, z, w) rows, the order the recorded columns are named in."""
    norms = np.linalg.norm(quats, axis=1, keepdims=True)
    if np.any(norms == 0):
        raise MigrationError("a recorded quaternion is all zeros; it does not describe a rotation")
    x, y, z, w = (quats / norms).T
    matrices = np.empty((len(x), 3, 3), dtype=np.float64)
    matrices[:, 0, 0] = 1 - 2 * (y * y + z * z)
    matrices[:, 0, 1] = 2 * (x * y - z * w)
    matrices[:, 0, 2] = 2 * (x * z + y * w)
    matrices[:, 1, 0] = 2 * (x * y + z * w)
    matrices[:, 1, 1] = 1 - 2 * (x * x + z * z)
    matrices[:, 1, 2] = 2 * (y * z - x * w)
    matrices[:, 2, 0] = 2 * (x * z - y * w)
    matrices[:, 2, 1] = 2 * (y * z + x * w)
    matrices[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return matrices


def pose_column_pairs(names: list[str], feature: str) -> list[tuple[list[int], list[int]]]:
    """(position, quaternion) index pairs, matched by name so a schema change cannot pass silently."""
    pairs: list[tuple[list[int], list[int]]] = []
    index = {name: position for position, name in enumerate(names)}
    for prefix in POSE_PREFIXES:
        position_names = [f"{prefix}ee.{axis}" for axis in ("x", "y", "z")]
        quaternion_names = [f"{prefix}ee.{axis}" for axis in ("qx", "qy", "qz", "qw")]
        present = [name for name in position_names + quaternion_names if name in index]
        if not present:
            continue
        if len(present) != 7:
            raise MigrationError(
                f"{feature} declares {sorted(present)} for prefix '{prefix}' but a pose needs all of "
                f"{position_names + quaternion_names}. Refusing to guess which columns are the pose"
            )
        pairs.append(([index[name] for name in position_names], [index[name] for name in quaternion_names]))
    if not pairs:
        raise MigrationError(f"{feature} declares no ee pose columns; nothing here is migratable")
    return pairs


def convert_block(block: np.ndarray, pairs: list[tuple[list[int], list[int]]]) -> np.ndarray:
    """Apply p_ee = p_tcp + R(quat) @ d to each (position, quaternion) pair, in float64."""
    converted = block.astype(np.float64, copy=True)
    for position_indices, quaternion_indices in pairs:
        rotations = quaternion_matrices(converted[:, quaternion_indices])
        converted[:, position_indices] += np.einsum("nij,j->ni", rotations, TOOL_FRAME_OFFSET_M)
    return converted


def assert_rigid_shift(before: np.ndarray, after: np.ndarray, pairs, feature: str) -> None:
    """|p_new - p_old| is |d| for every row, and nothing outside the position columns moved.

    A rotation matrix preserves length, so any row that shifted by something other than 410.85 mm
    means the quaternion it was built from was not a rotation.
    """
    expected = float(np.linalg.norm(TOOL_FRAME_OFFSET_M))
    moved: set[int] = set()
    for position_indices, _ in pairs:
        shift = np.linalg.norm(after[:, position_indices] - before[:, position_indices], axis=1)
        worst = float(np.max(np.abs(shift - expected)))
        if worst > 1e-6:
            raise MigrationError(
                f"{feature}: a row moved by {shift.max():.6f} m instead of the rigid {expected:.6f} m "
                f"(worst deviation {worst * 1e3:.4f} mm)"
            )
        moved.update(position_indices)

    untouched = [column for column in range(before.shape[1]) if column not in moved]
    if untouched and not np.array_equal(before[:, untouched], after[:, untouched]):
        raise MigrationError(f"{feature}: a non-position column changed; only the ee positions may move")


# --------------------------------------------------------------------------------- statistics ---


def vector_stats(values: np.ndarray) -> dict[str, Any]:
    """The estimator fr3_train_il_policy.vector_stats uses, so a view built later agrees with this."""
    values = np.asarray(values, dtype=np.float32)
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


def rewrite_global_stats(path: Path, migrated: dict[str, np.ndarray]) -> None:
    if not path.exists():
        print(f"  [stats] {path.name} absent; nothing to rewrite")
        return
    stats = json.loads(path.read_text(encoding="utf-8"))
    for feature, values in migrated.items():
        if feature not in stats:
            raise MigrationError(f"{path.name} has no '{feature}' entry but the data does")
        stats[feature] = vector_stats(values)
    path.write_text(json.dumps(stats, indent=4) + "\n", encoding="utf-8")
    print(f"  [stats] {path.name}: recomputed {', '.join(sorted(migrated))}")


def rewrite_episode_stats(root: Path, migrated: dict[str, np.ndarray], episodes: np.ndarray) -> None:
    """Per-episode min/max/mean/std/count/q01..q99, in the row order the meta parquet already uses."""
    files = sorted((root / "meta" / "episodes").rglob("*.parquet"))
    if not files:
        print("  [stats] no meta/episodes parquet; nothing to rewrite")
        return

    for path in files:
        table = pq.read_table(path)
        episode_column = table["episode_index"].to_numpy()
        for feature, values in migrated.items():
            for statistic in ("min", "max", "mean", "std", "count", "q01", "q10", "q50", "q90", "q99"):
                column = f"stats/{feature}/{statistic}"
                if column not in table.schema.names:
                    continue
                rows = []
                for episode in episode_column:
                    selected = values[episodes == episode]
                    if not len(selected):
                        raise MigrationError(f"episode {episode} is in {path.name} but has no frames")
                    rows.append(vector_stats(selected)[statistic])
                position = table.schema.get_field_index(column)
                field = table.schema.field(position)
                table = table.set_column(position, field, pa.array(rows, type=field.type))
        pq.write_table(table, path)
        print(f"  [stats] {path.relative_to(root)}: recomputed {len(episode_column)} episodes")


# ------------------------------------------------------------------------------------ driving ---


def read_feature_names(info: dict[str, Any], feature: str) -> list[str]:
    declared = (info.get("features") or {}).get(feature)
    if not declared:
        raise MigrationError(f"meta/info.json declares no '{feature}' feature")
    names = declared.get("names")
    if not names:
        raise MigrationError(f"meta/info.json gives '{feature}' no column names; they are what pairs the pose")
    return list(names)


def copy_tree(src: Path, dst: Path, video_mode: str) -> None:
    if dst.exists():
        raise MigrationError(f"{dst} already exists; migrating into it would mix two frames")

    def copy_function(source: str, destination: str) -> Any:
        if video_mode != "copy" and Path(source).suffix.lower() == ".mp4":
            try:
                if video_mode == "hardlink":
                    return Path(destination).hardlink_to(source)
                return Path(destination).symlink_to(source)
            except OSError:
                pass  # different filesystem, or no permission -- a copy is always correct
        return shutil.copy2(source, destination)

    shutil.copytree(src, dst, copy_function=copy_function)
    print(f"  [copy] {src} -> {dst}  (videos: {video_mode})")


def migrate(src: Path, dst: Path, video_mode: str, dry_run: bool) -> None:
    info = json.loads((src / "meta" / "info.json").read_text(encoding="utf-8"))
    home = home_frame_positions()

    src_files = sorted((src / "data").rglob("*.parquet"))
    if not src_files:
        raise MigrationError(f"{src} has no data parquet files")

    # Identify the source frame before writing anything, so a refusal costs nothing.
    state_positions = pose_column_pairs(read_feature_names(info, "observation.state"), "observation.state")[0][0]
    firsts = []
    for path in src_files:
        table = pq.read_table(path, columns=["observation.state", "episode_index"])
        block = np.stack(table["observation.state"].to_numpy(zero_copy_only=False)).astype(np.float64)
        episodes = table["episode_index"].to_numpy()
        for episode in np.unique(episodes):
            firsts.append(block[episodes == episode][0, state_positions])
    check_frame(f"{src.name} (source)", np.asarray(firsts), home, OLD_FRAME)

    if dry_run:
        print(f"  [dry-run] would write {len(src_files)} data file(s) to {dst}")
        return

    dst.parent.mkdir(parents=True, exist_ok=True)
    copy_tree(src, dst, video_mode)

    migrated: dict[str, list[np.ndarray]] = {}
    episode_order: list[np.ndarray] = []
    for path in sorted((dst / "data").rglob("*.parquet")):
        table = pq.read_table(path)
        episode_order.append(table["episode_index"].to_numpy())
        for feature in STAT_FEATURES:
            if feature not in table.schema.names:
                continue
            pairs = pose_column_pairs(read_feature_names(info, feature), feature)
            before = np.stack(table[feature].to_numpy(zero_copy_only=False)).astype(np.float64)
            after = convert_block(before, pairs)
            assert_rigid_shift(before, after, pairs, feature)

            position = table.schema.get_field_index(feature)
            field = table.schema.field(position)
            values = pa.array(after.astype(np.float32).ravel(), type=field.type.value_type)
            table = table.set_column(
                position, field, pa.FixedSizeListArray.from_arrays(values, field.type.list_size)
            )
            migrated.setdefault(feature, []).append(after)
        pq.write_table(table, path)
        print(f"  [data] {path.relative_to(dst)}: {table.num_rows} frames")

    stacked = {feature: np.concatenate(parts) for feature, parts in migrated.items()}
    episodes = np.concatenate(episode_order)
    rewrite_global_stats(dst / "meta" / "stats.json", stacked)
    rewrite_episode_stats(dst, stacked, episodes)

    marker = {
        "target_frame_name": NEW_FRAME,
        "migrated_from": {"target_frame_name": OLD_FRAME, "dataset_root": str(src)},
        "tool_frame_offset_m": TOOL_FRAME_OFFSET_M.tolist(),
        "note": (
            "A LeRobot dataset does not record its tool frame -- ee.x is ee.x in either. This file is "
            "that marker. Do not train on a mixture of frames."
        ),
    }
    (dst / "meta" / "fr3_tool_frame.json").write_text(json.dumps(marker, indent=4) + "\n", encoding="utf-8")

    # The exit check: ask the model which frame the written data is in, not this file's arithmetic.
    state = stacked["observation.state"]
    firsts = [state[episodes == episode][0, state_positions] for episode in np.unique(episodes)]
    check_frame(f"{dst.name} (result)", np.asarray(firsts), home, NEW_FRAME)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--src", type=Path, required=True, help="dataset root recorded against pika_task_tcp")
    parser.add_argument("--dst", type=Path, required=True, help="new dataset root; must not exist")
    parser.add_argument(
        "--videos",
        choices=("hardlink", "copy", "symlink"),
        default="hardlink",
        help="how to carry the mp4s over; they are unchanged by the migration (default: hardlink)",
    )
    parser.add_argument("--dry-run", action="store_true", help="identify the source frame and stop")
    args = parser.parse_args()

    print(f"Migrating {OLD_FRAME} -> {NEW_FRAME}")
    try:
        migrate(args.src.resolve(), args.dst.resolve(), args.videos, args.dry_run)
    except MigrationError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
