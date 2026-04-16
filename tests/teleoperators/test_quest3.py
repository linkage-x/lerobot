#!/usr/bin/env python

import numpy as np
import pytest

from lerobot.teleoperators.quest3.configuration_quest3 import Quest3TeleopConfig
from lerobot.teleoperators.quest3.teleop_quest3 import Quest3Teleop


def test_controller_pose_robot_frame_flips_planar_translation_for_pika_scene():
    teleop = Quest3Teleop(Quest3TeleopConfig(use_hand_tracking=False))
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    teleop._controller_mats["right"] = pose
    teleop._controller_last_update_s["right"] = 0.0

    pose_robot, _states, _last_update_s = teleop._controller_pose_robot_frame("right")

    np.testing.assert_allclose(pose_robot[:3, 3], np.array([3.0, 1.0, 2.0], dtype=np.float64))


def test_controller_gripper_uses_right_trigger_to_close_and_release_to_open():
    teleop = Quest3Teleop(Quest3TeleopConfig(use_hand_tracking=False))

    assert teleop._controller_gripper({"trigger": 0.0}, {"trigger": 0.0}) == pytest.approx(1.0)
    assert teleop._controller_gripper({"trigger": 0.7}, {"trigger": 0.0}) == pytest.approx(0.0)


def test_controller_mode_emits_per_frame_deltas_without_repeating_motion(monkeypatch):
    monkeypatch.setattr("lerobot.teleoperators.quest3.teleop_quest3.time.perf_counter", lambda: 100.0)

    teleop = Quest3Teleop(Quest3TeleopConfig(use_hand_tracking=False))
    teleop._controller_states["right"] = {"trigger": 0.0, "grip": 1.0, "button_a": False, "button_b": False}
    teleop._controller_last_update_s["right"] = 100.0

    pose0 = np.eye(4, dtype=np.float64)
    teleop._controller_mats["right"] = pose0
    action0 = teleop.get_action()
    assert action0["enabled"] is True
    np.testing.assert_allclose(
        [action0["target_x"], action0["target_y"], action0["target_z"]],
        np.zeros(3, dtype=np.float64),
    )

    pose1 = np.eye(4, dtype=np.float64)
    pose1[0, 3] = 0.05
    teleop._controller_mats["right"] = pose1
    action1 = teleop.get_action()
    assert np.linalg.norm([action1["target_x"], action1["target_y"], action1["target_z"]]) > 0.0

    teleop._controller_mats["right"] = pose1
    action2 = teleop.get_action()
    np.testing.assert_allclose(
        [action2["target_x"], action2["target_y"], action2["target_z"]],
        np.zeros(3, dtype=np.float64),
    )
