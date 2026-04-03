#!/usr/bin/env python

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch

from tools.fr3 import fr3_das_replay_real_runtime
from lerobot.utils.constants import ACTION, OBS_STATE


def test_load_episode_uses_public_dataset_api(tmp_path: Path, empty_lerobot_dataset_factory):
    features = {
        OBS_STATE: {"dtype": "float32", "shape": (2,), "names": ["x", "y"]},
        ACTION: {"dtype": "float32", "shape": (2,), "names": ["ax", "ay"]},
    }
    dataset = empty_lerobot_dataset_factory(root=tmp_path / "fr3_runtime_dataset", features=features, use_videos=False)

    for episode_idx in range(2):
        for frame_idx in range(2):
            dataset.add_frame(
                {
                    OBS_STATE: torch.tensor([episode_idx, frame_idx], dtype=torch.float32),
                    ACTION: torch.tensor([episode_idx + 0.5, frame_idx + 1.5], dtype=torch.float32),
                    "task": f"task_{episode_idx}",
                }
            )
        dataset.save_episode()

    dataset.finalize()
    loaded = fr3_das_replay_real_runtime.load_episode(str(dataset.root), 1)

    assert loaded.keys() == {"state", "action", "timestamp"}
    assert loaded["state"].dtype == np.float64
    assert loaded["action"].dtype == np.float64
    assert loaded["timestamp"].dtype == np.float64
    assert np.allclose(loaded["state"], np.array([[1.0, 0.0], [1.0, 1.0]], dtype=np.float64))
    assert np.allclose(loaded["action"], np.array([[1.5, 1.5], [1.5, 2.5]], dtype=np.float64))
    assert np.allclose(loaded["timestamp"], np.array([0.0, 1.0 / 30.0], dtype=np.float64))


def test_describe_first_frame_tilt_detects_legacy_pitch():
    pytest.importorskip("scipy")
    legacy_pose = np.array([0.0, 0.0, 0.0, 0.0, np.sin(np.deg2rad(7.5)), 0.0, np.cos(np.deg2rad(7.5))])
    T_Ws_I0 = fr3_das_replay_real_runtime.pose_from_xyzquat(legacy_pose)

    tilt = fr3_das_replay_real_runtime.describe_first_frame_tilt(T_Ws_I0)

    assert tilt["legacy_tilt"] is True
    assert tilt["angle_deg"] > 14.0
    assert np.allclose(np.asarray(tilt["axis"]), np.array([0.0, 1.0, 0.0]), atol=1e-3)


def test_interpolate_pose_blends_translation_and_rotation():
    pytest.importorskip("scipy")
    T_start = np.eye(4, dtype=np.float64)
    T_end = np.eye(4, dtype=np.float64)
    T_end[:3, 3] = np.array([0.2, -0.1, 0.4], dtype=np.float64)
    T_end[:3, :3] = fr3_das_replay_real_runtime._rotation_class().from_euler("y", 90.0, degrees=True).as_matrix()

    T_half = fr3_das_replay_real_runtime.interpolate_pose(T_start, T_end, 0.5)

    assert np.allclose(T_half[:3, 3], np.array([0.1, -0.05, 0.2]), atol=1e-9)
    rot_half = fr3_das_replay_real_runtime._rotation_class().from_matrix(T_half[:3, :3])
    assert np.allclose(rot_half.as_euler("xyz", degrees=True), np.array([0.0, 45.0, 0.0]), atol=1e-6)


def test_bbox_corners_returns_eight_unique_points():
    corners = fr3_das_replay_real_runtime.bbox_corners(
        np.array([-1.0, -2.0, -3.0], dtype=np.float64),
        np.array([4.0, 5.0, 6.0], dtype=np.float64),
    )

    assert corners.shape == (8, 3)
    assert len({tuple(row.tolist()) for row in corners}) == 8


def test_estimate_finger_lowest_z_uses_conservative_envelope():
    T_identity = np.eye(4, dtype=np.float64)

    lowest_z = fr3_das_replay_real_runtime.estimate_finger_lowest_z(T_identity)

    assert np.isclose(lowest_z, fr3_das_replay_real_runtime._FINGER_SWEEP_BBOX_E_MIN[2])
