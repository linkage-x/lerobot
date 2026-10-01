import json
from pathlib import Path

import numpy as np

from tools.thor.single_tag_camera_calibration import (
    _filter_joint_observation_outliers,
    _fit_for_views,
    load_existing_fisheye_intrinsics,
    tag_object_points,
)


def test_joint_rigid_consistency_filter_rejects_pose_outlier() -> None:
    observations = {}
    camera_poses = {}
    for camera in ("cam_06", "cam_07"):
        samples = []
        for index in range(21):
            samples.append(
                {
                    "frame_index": index,
                    "T_b_tool": np.eye(4),
                    "T_c_board": np.eye(4),
                    "reprojection_rmse_px": 0.1,
                }
            )
        outlier = np.eye(4)
        angle = np.deg2rad(10.0)
        outlier[:3, :3] = [
            [np.cos(angle), -np.sin(angle), 0.0],
            [np.sin(angle), np.cos(angle), 0.0],
            [0.0, 0.0, 1.0],
        ]
        samples.append(
            {
                "frame_index": 99,
                "T_b_tool": np.eye(4),
                "T_c_board": outlier,
                "reprojection_rmse_px": 0.2,
            }
        )
        observations[camera] = samples
        camera_poses[camera] = np.eye(4)

    filtered, reports = _filter_joint_observation_outliers(
        observations,
        np.eye(4),
        camera_poses,
        max_rotation_deg=3.0,
        max_translation_m=0.02,
        min_samples=20,
    )

    assert all(len(samples) == 21 for samples in filtered.values())
    assert reports["cam_06"]["num_rejected_outliers"] == 1
    assert reports["cam_06"]["rejected_outliers"][0]["capture_index"] == 99
    assert reports["cam_06"]["rejected_outliers"][0]["reasons"] == ["rotation"]


def _write_intrinsics_tree(tmp_path: Path, model: str = "opencv_fisheye") -> Path:
    camera_path = tmp_path / "per_camera/cam_13_cam_13/intrinsics.json"
    camera_path.parent.mkdir(parents=True)
    camera_path.write_text(
        json.dumps(
            {
                "model": model,
                "image_width": 1920,
                "image_height": 1080,
                "camera_matrix": [[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]],
                "dist_coeffs": [[-0.1, 0.01, 0.0, 0.0]],
            }
        ),
        encoding="utf-8",
    )
    summary = tmp_path / "summary.json"
    summary.write_text(
        json.dumps(
            {
                "cameras": [
                    {
                        "camera_name": "cam_13",
                        "status": "ok",
                        "intrinsics_json": str(camera_path),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return summary


def test_loads_only_declared_fisheye_intrinsics(tmp_path: Path) -> None:
    summary = _write_intrinsics_tree(tmp_path)
    loaded = load_existing_fisheye_intrinsics(summary)
    assert set(loaded) == {"cam_13"}
    assert loaded["cam_13"]["D"].shape == (4, 1)

    invalid_summary = _write_intrinsics_tree(tmp_path / "invalid", model="opencv_rational")
    try:
        load_existing_fisheye_intrinsics(invalid_summary)
    except ValueError as exc:
        assert "expected existing fisheye" in str(exc)
    else:
        raise AssertionError("non-fisheye intrinsics must be rejected")


def test_scales_existing_intrinsics_to_capture_resolution() -> None:
    fit = _fit_for_views(
        {
            "K": np.asarray([[1000.0, 0.0, 960.0], [0.0, 1000.0, 540.0], [0.0, 0.0, 1.0]]),
            "D": np.zeros((4, 1)),
            "width": 1920,
            "height": 1080,
        },
        [{"width": 960, "height": 540}],
    )
    np.testing.assert_allclose(fit["K"], [[500.0, 0.0, 480.0], [0.0, 500.0, 270.0], [0.0, 0.0, 1.0]])
    assert tag_object_points(0.16).shape == (4, 3)
