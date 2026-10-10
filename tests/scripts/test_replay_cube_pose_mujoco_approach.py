"""The MuJoCo preview's initial approach must follow the real replay's joint-jump guard."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest

pytest.importorskip("pinocchio")

_SCRIPT = (
    Path(__file__).resolve().parents[2]
    / "third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base_mujoco.py"
)
_spec = importlib.util.spec_from_file_location("replay_cube_pose_mujoco", _SCRIPT)
replay = importlib.util.module_from_spec(_spec)
# dataclasses resolve their module through sys.modules.
sys.modules[_spec.name] = replay
_spec.loader.exec_module(replay)


class _TranslationArm:
    """Toy kinematics: joints 0-2 are the TCP position, orientation fixed; IK is exact."""

    def forward_kinematics(self, q):
        pose = np.eye(4)
        pose[:3, 3] = np.asarray(q)[:3]
        return pose

    def inverse_kinematics(self, q, target):
        out = np.asarray(q, dtype=float).copy()
        out[:3] = target[:3, 3]
        return out


def _args(**overrides):
    values = dict(
        approach_max_steps=180,
        approach_position_tolerance_m=0.012,
        approach_orientation_tolerance_deg=6.0,
        guard_max_abs_delta_rad=0.35,
        guard_max_l2_delta_rad=0.70,
        guard_max_shrink=4,
    )
    values.update(overrides)
    return argparse.Namespace(**values)


def test_fr3_start_is_the_move_to_start_pose():
    expected = [0.0, -np.pi / 4, 0.0, -3 * np.pi / 4, 0.0, np.pi / 2, np.pi / 4]
    assert np.allclose(replay._INITIAL_JOINTS["fr3_start"], expected)


def test_guarded_approach_walks_in_shrunk_steps_like_the_real_guard():
    target = np.eye(4)
    target[:3, 3] = [1.0, 0.0, 0.0]

    result = replay._approach_like_real_replay(_TranslationArm(), np.zeros(7), target, _args())

    assert result["reached"]
    # Each accepted step is the full remaining distance halved until it is <= 0.35:
    # 1.0 -> 0.25 per step, then the remainder, so several steps rather than one jump.
    assert result["steps"] > 1
    assert abs(result["joints_rad"][0] - 1.0) <= 0.012


def test_guarded_approach_reports_a_hold_the_real_replay_would_stop_on():
    target = np.eye(4)
    target[:3, 3] = [1.0, 0.0, 0.0]

    result = replay._approach_like_real_replay(
        _TranslationArm(), np.zeros(7), target, _args(guard_max_abs_delta_rad=0.01)
    )

    assert not result["reached"]
    assert result["reason"] == "joint_jump_guard_hold"
