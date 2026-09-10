from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import yaml
from scipy.spatial.transform import Rotation

from tools.thor.p1_eye_hand_core import (
    EyeHandObservation,
    assess_solution,
    invert_transform,
    select_diverse_pose_records,
    solve_eye_hand,
    transform_error,
)
from tools.thor.p1_simple_eye_hand_calibration import (
    CAMERA_PROXIMITY_POOL_MULTIPLIER,
    MIN_TCP_CLEARANCE_M,
    MIN_TCP_Z_M,
    TABLE_CONTACT_TCP_Z_M,
    _load_pose_records,
    prepare_capture,
)
from tools.thor.p1_native_preposition import load_first_target


def _pose(rotvec, xyz):
    T = np.eye(4)
    T[:3, :3] = Rotation.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = xyz
    return T


def test_joint_solver_recovers_moved_base_and_both_tcp_tags() -> None:
    rng = np.random.default_rng(7)
    expected_base = _pose([0.03, -0.05, 0.08], [0.12, -0.04, 0.025])
    expected_tags = {
        56: _pose([0.2, 0.1, -0.15], [0.03, 0.01, 0.08]),
        57: _pose([-0.1, 0.3, 0.12], [-0.025, 0.02, 0.075]),
    }
    observations = []
    for index in range(20):
        tcp = _pose(rng.normal(0, 0.5, 3), [0.45, 0.0, 0.4] + rng.normal(0, 0.09, 3))
        for tag_id in (56, 57):
            world_tag = expected_base @ tcp @ expected_tags[tag_id]
            observations.append(
                EyeHandObservation(index, f"cam_{6 + index % 3:02d}", tag_id, tcp, world_tag)
            )

    solution = solve_eye_hand(observations)

    trans, rot = transform_error(solution["T_world_base"], expected_base)
    assert np.linalg.norm(trans) < 1e-7
    assert np.linalg.norm(rot) < 1e-7
    for tag_id in (56, 57):
        trans, rot = transform_error(solution["T_tcp_tags"][tag_id], expected_tags[tag_id])
        assert np.linalg.norm(trans) < 1e-7
        assert np.linalg.norm(rot) < 1e-7
    assert assess_solution(solution)[0]


def test_diverse_pose_selection_keeps_exact_teach_records() -> None:
    rows = [
        {
            "index": index,
            "pose": {"position_xyz_m": [0.4 + index * 0.01, 0.0, 0.4], "rotvec_xyz_rad": [0, index * 0.1, 0]},
        }
        for index in range(30)
    ]

    selected = select_diverse_pose_records(rows, 20)

    assert len(selected) == 20
    assert len({row["index"] for row in selected}) == 20
    assert all(row in rows for row in selected)
    assert [row["index"] for row in selected] == sorted(row["index"] for row in selected)


def test_prepare_capture_writes_pose_document_expected_by_existing_runner(tmp_path) -> None:
    run_dir = tmp_path / "run_1"
    run_dir.mkdir()
    source = Path("outputs/calibration/thor_gmsl2_extrinisics_robot_base_0720")

    selected_path, config_path = prepare_capture(run_dir, source, 50)

    selected = json.loads(selected_path.read_text(encoding="utf-8"))
    assert isinstance(selected, dict)
    assert selected["generated_by"] == "P1_simple_eye_hand_calibration"
    assert len(selected["records"]) == 50
    assert selected["pose_contract"].startswith("exact database")
    assert selected["selection_policy"]["table_contact_tcp_z_m"] == TABLE_CONTACT_TCP_Z_M
    assert selected["selection_policy"]["minimum_tcp_clearance_m"] == MIN_TCP_CLEARANCE_M
    assert selected["selection_policy"]["minimum_tcp_z_m"] == MIN_TCP_Z_M
    assert selected["selection_policy"]["camera_proximity_pool_multiplier"] == (
        CAMERA_PROXIMITY_POOL_MULTIPLIER
    )
    assert all("p1_pitch_mirror" not in record for record in selected["records"])
    assert all(record["pose"]["position_xyz_m"][2] >= MIN_TCP_Z_M for record in selected["records"])
    assert all(
        record["extrinsics_importance"]["nearest_visible_camera_distance_m"] is not None
        for record in selected["records"]
    )
    coverage = {
        camera: sum(camera in record["extrinsics_importance"]["visible_cameras"] for record in selected["records"])
        for camera in ("cam_06", "cam_07", "cam_08", "cam_09", "cam_12", "cam_13", "cam_14")
    }
    assert min(coverage.values()) >= 5
    database_records = _load_pose_records(source)
    database_by_frame = {record["database_frame_index"]: record for record in database_records}
    eligible = [
        record
        for record in database_records
        if record["pose"]["position_xyz_m"][2] >= MIN_TCP_Z_M
        and record["extrinsics_importance"]["visible_camera_count"] >= 4
    ]
    pool_count = int(np.ceil(50 * CAMERA_PROXIMITY_POOL_MULTIPLIER))
    proximity_cutoff = sorted(
        record["extrinsics_importance"]["nearest_visible_camera_distance_m"]
        for record in eligible
    )[pool_count - 1]
    for record in selected["records"]:
        source_record = database_by_frame[record["database_frame_index"]]
        assert record["extrinsics_importance"]["nearest_visible_camera_distance_m"] <= (
            proximity_cutoff
        )
        np.testing.assert_allclose(record["joint_values_rad"], source_record["joint_values_rad"])
        np.testing.assert_allclose(record["pose"]["quaternion_xyzw"], source_record["pose"]["quaternion_xyzw"])
    assert config_path.is_file()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert config["robot"]["target_frame_name"] == "fr3_ee"
    assert config["robot"]["urdf_path"].endswith("/fr3_corenetic_gripper.urdf")
    assert config["robot"]["gripper_backend"] == "mock"
    assert config["execution"]["home_before_start"] is False
    assert config["execution"]["max_command_steps"] == 240
    assert config["execution"]["joint_tolerance_rad"] == 0.02
    assert config["execution"]["fail_on_unreached_pose"] is True
    assert config["thor"]["skip_hardware_sync"] is True
    assert config["thor"]["skip_argus_probe"] is True

    from third_party.opencv_kalibr.fr3_calibration.execute_pose_and_capture import (
        _load_pose_records as load_capture_pose_records,
    )

    assert len(load_capture_pose_records(selected_path)) == 50
    np.testing.assert_allclose(load_first_target(selected_path), selected["records"][0]["joint_values_rad"])
