#!/usr/bin/env python

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


MODULE_PATH = (
    Path(__file__).resolve().parents[2]
    / "third_party/opencv_kalibr/hikon_cube_tracking_offline/interpolate_missing_ee_pose.py"
)
SPEC = importlib.util.spec_from_file_location("interpolate_missing_ee_pose", MODULE_PATH)
interpolate_missing_ee_pose = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
sys.modules["interpolate_missing_ee_pose"] = interpolate_missing_ee_pose
SPEC.loader.exec_module(interpolate_missing_ee_pose)


def test_interpolates_bounded_missing_ee_pose_and_rebuilds_action(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _write_dataset(
        dataset_root,
        states=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [3.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    exit_code = interpolate_missing_ee_pose.main([str(dataset_root), "--no-backup"])

    assert exit_code == 0
    table = pq.read_table(dataset_root / "data/chunk-000/file-000.parquet")
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
    actions = np.asarray(table.column("action").to_pylist(), dtype=np.float64)
    assert np.allclose(states[:, 0], [0.0, 1.0, 2.0, 3.0])
    assert np.allclose(states[:, 3:7], [[0.0, 0.0, 0.0, 1.0]] * 4)
    assert np.allclose(actions[:, 0], [1.0, 2.0, 3.0, 3.0])

    report_path = dataset_root / "derived/hikon_cube_tracking_in_robot_base/ee_pose_interpolation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["initial_missing_frames"] == 2
    assert report["final_missing_frames"] == 0
    assert report["filled_frames"] == 2


def test_dry_run_does_not_modify_parquet(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _write_dataset(
        dataset_root,
        states=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    args = interpolate_missing_ee_pose.parse_args([str(dataset_root), "--dry-run"])
    result = interpolate_missing_ee_pose.run(args)

    table = pq.read_table(dataset_root / "data/chunk-000/file-000.parquet")
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)
    assert np.isnan(states[1]).all()
    assert result["filled_frames"] == 1
    report_path = dataset_root / "derived/hikon_cube_tracking_in_robot_base/ee_pose_interpolation_report.json"
    assert not report_path.exists()


def test_config_can_supply_dataset_root(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _write_dataset(
        dataset_root,
        states=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )
    config_path = tmp_path / "interpolate.yaml"
    config_path.write_text(
        "\n".join(
            [
                "input:",
                f"  dataset_root: {dataset_root}",
                "runtime:",
                "  dry_run: true",
            ]
        ),
        encoding="utf-8",
    )

    args = interpolate_missing_ee_pose.parse_args(["--config", str(config_path)])
    result = interpolate_missing_ee_pose.run(args)

    assert args.dataset_root == dataset_root
    assert result["filled_frames"] == 1
    assert result["dry_run"] is True


def test_unbounded_missing_segment_is_reported_not_filled(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _write_dataset(
        dataset_root,
        states=[
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    result = interpolate_missing_ee_pose.run(
        interpolate_missing_ee_pose.parse_args([str(dataset_root), "--dry-run"])
    )

    assert result["filled_frames"] == 0
    assert result["skipped_frames"] == 1
    assert result["skipped_gaps"][0]["reason"] == "unbounded"


def test_velocity_hermite_uses_history_velocity_not_just_linear(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _write_dataset(
        dataset_root,
        states=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    interpolate_missing_ee_pose.main(
        [
            str(dataset_root),
            "--no-backup",
            "--method",
            "velocity_hermite",
            "--history-velocity-frames",
            "2",
        ]
    )
    table = pq.read_table(dataset_root / "data/chunk-000/file-000.parquet")
    states = np.asarray(table.column("observation.state").to_pylist(), dtype=np.float64)

    linear_x = np.array([7.333333333333333, 8.666666666666666], dtype=np.float64)
    hermite_x = states[3:5, 0]
    assert np.all(hermite_x > linear_x)
    assert np.allclose(hermite_x, [9.11111111111111, 9.555555555555555], atol=1e-6)

    report_path = dataset_root / "derived/hikon_cube_tracking_in_robot_base/ee_pose_interpolation_report.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["filled_gaps"][0]["method"] == "velocity_hermite"


def test_velocity_hermite_uses_history_acceleration_when_available(tmp_path: Path):
    dataset_root_no_accel = tmp_path / "dataset_no_accel"
    dataset_root_with_accel = tmp_path / "dataset_with_accel"
    states = [
        [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [6.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
        [10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
    ]
    _write_dataset(dataset_root_no_accel, states=states)
    _write_dataset(dataset_root_with_accel, states=states)

    interpolate_missing_ee_pose.main(
        [
            str(dataset_root_no_accel),
            "--no-backup",
            "--method",
            "velocity_hermite",
            "--history-velocity-frames",
            "1",
        ]
    )
    interpolate_missing_ee_pose.main(
        [
            str(dataset_root_with_accel),
            "--no-backup",
            "--method",
            "velocity_hermite",
            "--history-velocity-frames",
            "2",
        ]
    )

    states_no_accel = np.asarray(
        pq.read_table(dataset_root_no_accel / "data/chunk-000/file-000.parquet")
        .column("observation.state")
        .to_pylist(),
        dtype=np.float64,
    )
    states_with_accel = np.asarray(
        pq.read_table(dataset_root_with_accel / "data/chunk-000/file-000.parquet")
        .column("observation.state")
        .to_pylist(),
        dtype=np.float64,
    )

    # With acceleration estimate available, interpolation should carry stronger forward motion.
    assert np.all(states_with_accel[3:5, 0] > states_no_accel[3:5, 0])


def test_velocity_hermite_falls_back_to_linear_when_no_history(tmp_path: Path):
    dataset_root = tmp_path / "dataset"
    _write_dataset(
        dataset_root,
        states=[
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
            [np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan],
            [2.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
        ],
    )

    result = interpolate_missing_ee_pose.run(
        interpolate_missing_ee_pose.parse_args(
            [str(dataset_root), "--dry-run", "--method", "velocity_hermite"]
        )
    )

    assert result["filled_frames"] == 1
    assert result["filled_gaps"][0]["method"] == "linear"


def _write_dataset(dataset_root: Path, *, states: list[list[float]]) -> None:
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data/chunk-000").mkdir(parents=True)
    names = ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw"]
    (dataset_root / "meta/info.json").write_text(
        json.dumps(
            {
                "fps": 30,
                "features": {
                    "observation.state": {"dtype": "float32", "shape": [7], "names": names},
                    "action": {"dtype": "float32", "shape": [7], "names": names},
                },
            }
        ),
        encoding="utf-8",
    )
    table = pa.table(
        {
            "observation.state": pa.array(states, type=pa.list_(pa.float32())),
            "action": pa.array(states, type=pa.list_(pa.float32())),
            "episode_index": pa.array([0] * len(states), type=pa.int64()),
            "frame_index": pa.array(list(range(len(states))), type=pa.int64()),
        }
    )
    pq.write_table(table, dataset_root / "data/chunk-000/file-000.parquet")
