#!/usr/bin/env python

from __future__ import annotations

import numpy as np

from tools.fr3 import fr3_compare_current_pose_to_dataset_starts as compare_tool


def test_quaternion_angle_deg_matches_expected_rotation():
    identity = np.array([0.0, 0.0, 0.0, 1.0], dtype=np.float64)
    yaw_90 = np.array([0.0, 0.0, np.sin(np.deg2rad(45.0)), np.cos(np.deg2rad(45.0))], dtype=np.float64)

    angle_deg = compare_tool.quaternion_angle_deg(identity, yaw_90)

    assert np.isclose(angle_deg, 90.0)


def test_compute_episode_deltas_ranks_closest_start_first():
    current_state = np.array([0.1, -0.5, 0.28, 0.0, 0.0, 0.0, 1.0, 0.9], dtype=np.float64)
    episode_start_states = [
        compare_tool.EpisodeStartState(
            episode_index=3,
            state=np.array([0.11, -0.49, 0.27, 0.0, 0.0, 0.0, 1.0, 0.85], dtype=np.float64),
        ),
        compare_tool.EpisodeStartState(
            episode_index=7,
            state=np.array([0.25, 0.02, 0.05, 0.0, 0.0, 0.0, 1.0, 0.2], dtype=np.float64),
        ),
    ]

    deltas = compare_tool.compute_episode_deltas(
        current_state,
        episode_start_states,
        position_weight_mm=1.0,
        rotation_weight_deg=1.0,
        gripper_weight=50.0,
    )

    assert [item.episode_index for item in deltas] == [3, 7]
    assert np.allclose(deltas[0].position_delta_m, [0.01, 0.01, -0.01])
    assert deltas[0].rotation_delta_deg == 0.0
    assert deltas[0].gripper_delta == 0.05
    assert deltas[1].weighted_score > deltas[0].weighted_score
