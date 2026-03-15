#!/usr/bin/env python

import numpy as np
import torch

from lerobot.scripts.lerobot_dataset_viz import (
    EE_RULER_AXIS_COLORS,
    build_ee_axis_ruler_strips,
    extract_ee_pose,
    get_ee_pose_state_indices,
    has_ee_pose,
)


def test_has_ee_pose_requires_all_pose_fields():
    assert has_ee_pose(
        {
            "ee.x": torch.tensor([0.1]),
            "ee.y": torch.tensor([0.2]),
            "ee.z": torch.tensor([0.3]),
            "ee.wx": torch.tensor([0.0]),
            "ee.wy": torch.tensor([0.0]),
            "ee.wz": torch.tensor([0.0]),
        }
    )
    assert not has_ee_pose(
        {
            "ee.x": torch.tensor([0.1]),
            "ee.y": torch.tensor([0.2]),
            "ee.z": torch.tensor([0.3]),
            "ee.wx": torch.tensor([0.0]),
            "ee.wy": torch.tensor([0.0]),
        }
    )


def test_extract_ee_pose_returns_position_and_rotation_matrix():
    batch = {
        "ee.x": torch.tensor([0.4], dtype=torch.float32),
        "ee.y": torch.tensor([-0.1], dtype=torch.float32),
        "ee.z": torch.tensor([0.25], dtype=torch.float32),
        "ee.wx": torch.tensor([0.0], dtype=torch.float32),
        "ee.wy": torch.tensor([0.0], dtype=torch.float32),
        "ee.wz": torch.tensor([np.pi / 2], dtype=torch.float32),
    }

    position, rotation = extract_ee_pose(batch, 0)

    assert np.allclose(position, np.array([0.4, -0.1, 0.25], dtype=np.float32))
    expected_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(rotation, expected_rotation, atol=1e-6)


def test_get_ee_pose_state_indices_from_packed_observation_state():
    indices = get_ee_pose_state_indices(
        ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "gripper.pos"]
    )

    assert indices == {
        "ee.x": 0,
        "ee.y": 1,
        "ee.z": 2,
        "ee.wx": 3,
        "ee.wy": 4,
        "ee.wz": 5,
    }


def test_extract_ee_pose_supports_packed_observation_state():
    ee_pose_state_indices = get_ee_pose_state_indices(
        ["ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "gripper.pos"]
    )
    batch = {
        "observation.state": torch.tensor([[0.4, -0.1, 0.25, 0.0, 0.0, np.pi / 2, 0.5]], dtype=torch.float32)
    }

    assert has_ee_pose(batch, ee_pose_state_indices=ee_pose_state_indices)
    position, rotation = extract_ee_pose(batch, 0, ee_pose_state_indices=ee_pose_state_indices)

    assert np.allclose(position, np.array([0.4, -0.1, 0.25], dtype=np.float32))
    expected_rotation = np.array(
        [
            [0.0, -1.0, 0.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )
    assert np.allclose(rotation, expected_rotation, atol=1e-6)


def test_build_ee_axis_ruler_strips_for_x_axis_creates_main_line_and_ticks():
    strips = build_ee_axis_ruler_strips("x", 0.1)

    assert len(strips) == 12
    assert np.allclose(
        strips[0],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.1, 0.0, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert np.allclose(strips[1], np.array([[0.0, 0.0, 0.0], [0.0, 0.01, 0.0]], dtype=np.float32))
    assert np.allclose(strips[6], np.array([[0.05, 0.0, 0.0], [0.05, 0.01, 0.0]], dtype=np.float32))
    assert np.allclose(strips[2], np.array([[0.01, 0.0, 0.0], [0.01, 0.005, 0.0]], dtype=np.float32))


def test_build_ee_axis_ruler_strips_for_y_and_z_axes_use_origin_and_perpendicular_ticks():
    y_strips = build_ee_axis_ruler_strips("y", 0.1)
    z_strips = build_ee_axis_ruler_strips("z", 0.1)

    assert np.allclose(
        y_strips[0],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.1, 0.0],
            ],
            dtype=np.float32,
        ),
    )
    assert np.allclose(y_strips[2], np.array([[0.0, 0.01, 0.0], [0.0, 0.01, 0.005]], dtype=np.float32))

    assert np.allclose(
        z_strips[0],
        np.array(
            [
                [0.0, 0.0, 0.0],
                [0.0, 0.0, 0.1],
            ],
            dtype=np.float32,
        ),
    )
    assert np.allclose(z_strips[2], np.array([[0.0, 0.0, 0.01], [0.005, 0.0, 0.01]], dtype=np.float32))


def test_ee_ruler_axis_colors_follow_rgb_convention():
    assert EE_RULER_AXIS_COLORS == {
        "x": [255, 0, 0, 255],
        "y": [0, 255, 0, 255],
        "z": [0, 0, 255, 255],
    }
