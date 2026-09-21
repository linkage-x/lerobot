import json
from pathlib import Path

import numpy as np

from tools.thor.visualize_p0_camera_extrinsics import visualize


def test_visualizes_camera_extrinsics_and_writes_machine_summary(tmp_path: Path) -> None:
    cameras = {}
    for index, camera in enumerate(("cam_06", "cam_07")):
        matrix = np.eye(4)
        matrix[:3, 3] = [1.0, float(index) * 0.5 - 0.25, 0.6]
        cameras[camera] = {
            "base_to_camera": {
                "matrix_4x4": matrix.tolist(),
                "rpy_deg": [0.0, 0.0, 0.0],
                "quaternion_xyzw": [0.0, 0.0, 0.0, 1.0],
            },
            "num_input_observations": 30,
            "num_observations": 28,
            "num_rejected_outliers": 2,
            "sample_residuals": {"rotation_deg_mean": 0.5},
            "intrinsics_source_camera": camera,
            "temporary_intrinsics_reuse": False,
        }
    source = tmp_path / "summary.json"
    source.write_text(
        json.dumps(
            {
                "status": "passed",
                "world": {"world_frame_id": "fr3_base"},
                "joint_solution": {
                    "sample_residuals": {
                        "rotation_deg_mean": 0.5,
                        "translation_m_mean": 0.004,
                    },
                    "robust_filter": {"num_rejected_outliers": 4},
                    "cameras": cameras,
                },
            }
        ),
        encoding="utf-8",
    )

    image_path, visualization_summary_path, calibration_copy_path = visualize(
        source, tmp_path / "visualization"
    )

    assert image_path.is_file() and image_path.stat().st_size > 1000
    assert calibration_copy_path.read_bytes() == source.read_bytes()
    payload = json.loads(visualization_summary_path.read_text(encoding="utf-8"))
    assert payload["world_frame_id"] == "fr3_base"
    assert set(payload["cameras"]) == {"cam_06", "cam_07"}
    assert payload["cameras"]["cam_06"]["optical_axis_base_xyz"] == [0.0, 0.0, 1.0]
