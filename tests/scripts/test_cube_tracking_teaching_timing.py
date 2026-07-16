import csv
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np


def _load_evaluator_module():
    path = Path("third_party/opencv_kalibr/evaluate/cube_tracking_teaching_error_eval.py")
    spec = importlib.util.spec_from_file_location("cube_tracking_teaching_error_eval", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_load_thor_video_timing_maps_logical_sof_to_monotonic(tmp_path: Path) -> None:
    evaluator = _load_evaluator_module()
    episode_dir = tmp_path / "episode_000001"
    episode_dir.mkdir()
    video_path = episode_dir / "cam_06.mkv"
    meta_path = episode_dir / "meta.json"
    meta_path.write_text(json.dumps({
        "video": {"fps": 60},
        "sync_reference": {
            "t0_wall_s": 1000.0,
            "t0_mono_s": 100.0,
            "camera_clock_bridge": {
                "scale": 1.0,
                "offset_ns": -900_000_000,
                "acquire_delay_residual_ns": {"p95": 100_000},
            },
        },
        "cameras": [{"name": "cam_06", "file": "cam_06.mkv", "first_wall_s": 1000.0}],
    }))
    with (episode_dir / "cam_06.argus_frame_metadata.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=[
            "camera",
            "logical_frame_index",
            "local_frame_number",
            "sensor_timestamp_ns",
            "sof_tsc_ns",
            "eof_tsc_ns",
            "internal_frame_count",
            "host_acquired_monotonic_ns",
        ])
        writer.writeheader()
        writer.writerow({
            "camera": "cam_06",
            "logical_frame_index": 0,
            "local_frame_number": 1,
            "sensor_timestamp_ns": 0,
            "sof_tsc_ns": 1_000_000_000,
            "eof_tsc_ns": 1_014_000_000,
            "internal_frame_count": 1,
            "host_acquired_monotonic_ns": 114_100_000,
        })

    timing = evaluator._load_thor_video_timing(
        {"cam_06": str(video_path)},
        {"motion_meta_json": str(meta_path)},
    )

    assert timing is not None
    assert timing["camera_frame_monotonic_s"]["cam_06"][0] == 0.1
    assert timing["camera_clock_bridge"]["offset_ns"] == -900_000_000


def test_fk_interpolator_uses_monotonic_time() -> None:
    evaluator = _load_evaluator_module()
    rows = []
    for monotonic_s, x in ((10.0, 0.0), (12.0, 2.0)):
        rows.append({
            "monotonic_s": monotonic_s,
            "actual_ee_x_m": x,
            "actual_ee_y_m": 0.0,
            "actual_ee_z_m": 0.0,
            "actual_ee_rx_rad": 0.0,
            "actual_ee_ry_rad": 0.0,
            "actual_ee_rz_rad": 0.0,
        })

    interpolate = evaluator._build_fk_monotonic_time_interpolator(rows)

    assert interpolate is not None
    pose = interpolate(11.0)
    assert pose is not None
    assert np.allclose(pose[:3, 3], [1.0, 0.0, 0.0])


def test_legacy_fk_wall_time_is_anchored_to_episode_monotonic() -> None:
    evaluator = _load_evaluator_module()

    rows = evaluator._robot_rows_with_monotonic_time(
        [{"wall_s": 1002.5}],
        {"sync_reference": {"t0_wall_s": 1000.0, "t0_mono_s": 100.0}},
    )

    assert rows[0]["monotonic_s"] == 102.5
    assert rows[0]["robot_state_time_source"] == "episode_wall_to_monotonic_anchor"
