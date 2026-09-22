from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

from tools.thor import p0_native_arm_dataset_replay as replay


FIELDS = [
    "episode_index",
    "frame_index",
    "state_x_m",
    "state_y_m",
    "state_z_m",
    "state_qx",
    "state_qy",
    "state_qz",
    "state_qw",
    "pose_source",
    "smoothing",
]


def _write_sidecar(path: Path) -> None:
    rows = [
        [0, 0, 0.4, -0.1, 0.3, 0, 0, 0, 1, "corner_ba", "offline_batch"],
        [1, 4, 0.5, -0.2, 0.4, 0, 0, 0, 2, "corner_ba", "offline_batch"],
        [1, 3, 0.4, -0.2, 0.4, 0, 0, 0, 1, "corner_ba", "offline_batch"],
        [1, 5, "nan", -0.2, 0.4, 0, 0, 0, 1, "corner_ba", "offline_batch"],
    ]
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(FIELDS)
        writer.writerows(rows)


def test_load_episode_targets_selects_sorts_and_normalizes(tmp_path: Path) -> None:
    path = tmp_path / "state_action.left.csv"
    _write_sidecar(path)

    targets, summary = replay._load_episode_targets(path, 1)

    assert [target.frame_index for target in targets] == [3, 4]
    np.testing.assert_allclose(targets[1].quaternion_xyzw, [0, 0, 0, 1])
    assert summary == {
        "frames": 2,
        "first_frame": 3,
        "last_frame": 4,
        "skipped_invalid_rows": 1,
        "pose_sources": ["corner_ba"],
        "smoothing_modes": ["offline_batch"],
    }


def test_drop_consecutive_duplicate_knots_preserves_path_order() -> None:
    q = np.asarray(
        [
            [0, 1, 2, 3, 4, 5, 6],
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
            [1, 2, 3, 4, 5, 6, 7],
            [2, 3, 4, 5, 6, 7, 8],
        ],
        dtype=np.float64,
    )

    filtered, removed = replay._drop_consecutive_duplicate_knots(q)

    assert removed == 2
    np.testing.assert_array_equal(filtered, q[[0, 2, 4]])


def test_parse_args_accepts_explicit_dataset_and_episode_option() -> None:
    args = replay._parse_args(
        [
            "--episode-index",
            "7",
            "--dataset-root",
            "/tmp/example",
            "--side",
            "right",
            "--gripper-width-mm",
            "88",
        ]
    )

    assert args.episode is None
    assert args.episode_index == 7
    assert args.dataset_root == Path("/tmp/example")
    assert args.side == "right"
    assert args.gripper_width_mm == 88.0
    assert args.chunk_size == 10
    assert args.execute is False
    assert args.unchecked_execution is False
    assert args.gripper_mode == "off"
    assert args.replay_speed_factor == replay.NATIVE_SPEED_FACTOR
    assert args.start_speed_factor == replay.START_SPEED_FACTOR


def test_parse_args_defaults_to_requested_dataset_and_episode_zero() -> None:
    args = replay._parse_args(["--side", "left", "--gripper-width-mm", "88"])

    assert args.episode_index == 0
    assert args.dataset_root == replay.DEFAULT_DATASET_ROOT


def test_parse_args_keeps_legacy_positional_episode() -> None:
    args = replay._parse_args(["4", "--side", "left", "--gripper-width-mm", "88"])

    assert args.episode == 4
    assert args.episode_index == 4


def test_parse_args_accepts_explicit_unchecked_execution() -> None:
    args = replay._parse_args(
        [
            "--episode-index",
            "2",
            "--side",
            "left",
            "--gripper-width-mm",
            "88",
            "--execute",
            "--unchecked-execution",
            "--gripper-mode",
            "dataset",
            "--replay-speed-factor",
            "0.1",
            "--start-speed-factor",
            "0.05",
        ]
    )

    assert args.execute is True
    assert args.unchecked_execution is True
    assert args.gripper_mode == "dataset"
    assert args.replay_speed_factor == 0.1
    assert args.start_speed_factor == 0.05


def test_build_unchecked_trajectories_uses_fixed_overlapping_chunks() -> None:
    calls = []

    class FakeCore:
        @staticmethod
        def JointTrajectory(waypoints, speed, deviation, timeout):
            calls.append((waypoints, speed, deviation, timeout))
            return object()

    q = np.arange(5 * 7, dtype=float).reshape(5, 7)

    trajectories = replay._build_unchecked_trajectories(FakeCore(), q, 3, 0.1)

    assert [(start, end) for start, end, _trajectory in trajectories] == [(0, 3), (2, 5)]
    assert calls[0] == (
        q[0:3].tolist(),
        0.1,
        replay.MAX_DEVIATION_RAD,
        replay.PLANNING_TIMEOUT_S,
    )
    assert calls[1][0] == q[2:5].tolist()


def test_gripper_first_command_happens_after_arm_replay_controller_starts() -> None:
    events = []

    class FakeTrajectory:
        @staticmethod
        def get_duration():
            return 0.0

    class FakeController:
        def __init__(self, *_args):
            pass

    class FakeCore:
        NativeJointTrajectoryController = FakeController

    class FakePanda:
        @staticmethod
        def start_controller_guarded(_controller):
            events.append("arm_start")

        @staticmethod
        def control_thread_active():
            return False

        @staticmethod
        def stop_controller():
            events.append("arm_stop")

        @staticmethod
        def raise_error():
            pass

    class FakeGripper:
        @staticmethod
        def set_position(_position):
            events.append("gripper_command")

    replay._run_unchecked_trajectory(
        FakePanda(),
        FakeCore(),
        FakeTrajectory(),
        gripper=FakeGripper(),
        gripper_widths_m=np.asarray([0.05, 0.06]),
        joint_knots=np.zeros((2, 7)),
    )

    assert events.index("arm_start") < events.index("gripper_command")


def test_requested_ik_initial_joint_vector_matches_operator_capture() -> None:
    np.testing.assert_allclose(
        replay.IK_INITIAL_JOINTS_RAD,
        [
            -0.2982022354854064,
            -0.20546837567339093,
            0.2008775163648066,
            -2.707162497847623,
            -0.09350475554363503,
            2.9366955831629005,
            0.8043834376561214,
        ],
        rtol=0.0,
        atol=0.0,
    )


def test_native_audit_splits_only_planner_failures(tmp_path: Path) -> None:
    class FakeNative:
        def prepare(self, _core, q, _speed, _width, _folder):
            if len(q) > 2:
                raise RuntimeError("Trajectory generation faild.")

    q = np.arange(5 * 7, dtype=float).reshape(5, 7)

    partitions = replay._audit_native_partitions(
        FakeNative(),
        object(),
        q,
        chunk_size=5,
        speed_factor=0.01,
        width_m=0.088,
        out_dir=tmp_path,
    )

    assert partitions == [(0, 2), (1, 3), (2, 4), (3, 5)]
    assert list(tmp_path.glob("*/planner_split.json"))


def test_load_episode_gripper_widths_uses_named_box_distance(tmp_path: Path) -> None:
    import pyarrow as pa
    import pyarrow.parquet as pq

    dataset_root = tmp_path / "dataset"
    (dataset_root / "meta").mkdir(parents=True)
    (dataset_root / "data/chunk-000").mkdir(parents=True)
    (dataset_root / "meta/info.json").write_text(
        json.dumps(
            {
                "features": {
                    "observation.state": {
                        "names": ["unrelated", "box_gripper.distance_m"],
                    }
                }
            }
        ),
        encoding="utf-8",
    )
    table = pa.table(
        {
            "observation.state": [[7.0, 0.052], [8.0, 0.061], [9.0, 0.073]],
            "episode_index": [2, 2, 3],
            "frame_index": [4, 5, 0],
        }
    )
    pq.write_table(table, dataset_root / "data/chunk-000/file-000.parquet")
    targets = [
        replay.PoseTarget(4, np.zeros(3), np.asarray([0, 0, 0, 1], dtype=float)),
        replay.PoseTarget(5, np.zeros(3), np.asarray([0, 0, 0, 1], dtype=float)),
    ]

    widths, summary = replay._load_episode_gripper_widths(dataset_root, targets, 2)

    np.testing.assert_allclose(widths, [0.052, 0.061])
    assert summary["source"] == "observation.state"
    assert summary["feature_name"] == "box_gripper.distance_m"
    assert summary["frames"] == 2


def test_network_probe_result_accepts_original_thresholds() -> None:
    result = replay._network_probe_result(
        "10000 packets transmitted, 10000 received, 0% packet loss\n"
        "rtt min/avg/max/mdev = 0.153/0.217/0.837/0.074 ms\n",
        "",
        0,
    )

    assert result["passed"] is True
    assert result["metrics"]["rtt_max_ms"] == 0.837


def test_network_probe_result_rejects_latency_spike() -> None:
    result = replay._network_probe_result(
        "10000 packets transmitted, 10000 received, 0% packet loss\n"
        "rtt min/avg/max/mdev = 0.134/0.232/10.133/0.188 ms\n",
        "",
        0,
    )

    assert result["passed"] is False


def test_safe_multistart_seeds_are_deterministic_and_bounded() -> None:
    lower = np.arange(7, dtype=float)
    upper = lower + 2.0
    preferred = np.asarray([-10, 1.5, 2.5, 3.5, 4.5, 5.5, 20], dtype=float)

    first = replay._safe_multistart_seeds(lower, upper, preferred, count=4)
    second = replay._safe_multistart_seeds(lower, upper, preferred, count=4)

    assert [name for name, _seed in first] == [
        "preferred",
        "halton_001",
        "halton_002",
        "halton_003",
        "halton_004",
    ]
    np.testing.assert_allclose(first[0][1], np.clip(preferred, lower, upper))
    for (_first_name, first_seed), (_second_name, second_seed) in zip(first, second, strict=True):
        np.testing.assert_allclose(first_seed, second_seed)
        assert np.all(first_seed >= lower)
        assert np.all(first_seed <= upper)


def test_verify_fixed_base_provenance_rejects_auxiliary_alignment(tmp_path: Path) -> None:
    dataset_root = tmp_path / "outputs" / "datasets" / "example"
    dataset_root.mkdir(parents=True)
    tracking = (
        tmp_path
        / "outputs"
        / "tracking_analysis"
        / f"example_{replay.TRACKING_RUN_SUFFIX}"
    )
    tracking.mkdir(parents=True)
    fixed_summary = tmp_path / "fixed_summary.json"
    fixed_summary.write_text("{}", encoding="utf-8")
    summary = {
        "robot_base_mode": "fixed",
        "alignment": {"enabled": True, "applied": True, "method": "aruco"},
        "calibration_inputs": {
            "intrinsics_summary": "/tmp/intrinsics.json",
            "fixed_camera_summary": str(fixed_summary),
            "auxiliary_marker_required": True,
            "auxiliary_marker_summary": "/tmp/auxiliary.json",
            "auxiliary_marker_run_name": "legacy_auxiliary",
        },
    }
    (tracking / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    with np.testing.assert_raises_regex(RuntimeError, "not approved for direct robot-base replay"):
        replay._verify_fixed_base_provenance(dataset_root)


def test_verify_fixed_base_provenance_accepts_no_auxiliary_marker(tmp_path: Path) -> None:
    dataset_root = tmp_path / "outputs" / "datasets" / "example"
    dataset_root.mkdir(parents=True)
    tracking = (
        tmp_path
        / "outputs"
        / "tracking_analysis"
        / f"example_{replay.TRACKING_RUN_SUFFIX}"
    )
    tracking.mkdir(parents=True)
    fixed_summary = tmp_path / "fixed_summary.json"
    fixed_summary.write_text("{}", encoding="utf-8")
    summary = {
        "robot_base_mode": "fixed",
        "alignment": {"enabled": False, "applied": False, "method": "none"},
        "calibration_inputs": {
            "intrinsics_summary": "/tmp/intrinsics.json",
            "fixed_camera_summary": str(fixed_summary),
            "auxiliary_marker_required": False,
            "auxiliary_marker_summary": "",
            "auxiliary_marker_run_name": "",
        },
    }
    (tracking / "summary.json").write_text(json.dumps(summary), encoding="utf-8")

    provenance = replay._verify_fixed_base_provenance(dataset_root)

    assert provenance["robot_base_mode"] == "fixed"
    assert provenance["alignment"] == {"enabled": False, "applied": False, "method": "none"}
    assert provenance["auxiliary_marker_required"] is False
