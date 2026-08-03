#!/usr/bin/env python

import numpy as np
import pytest

from lerobot.teleoperators.quest3.configuration_quest3 import Quest3TeleopConfig
from lerobot.teleoperators.quest3.teleop_quest3 import Quest3Teleop


def test_controller_pose_robot_frame_maps_openxr_axes_onto_the_robot_base():
    # OpenXR is +X right, +Y up, -Z forward; the FR3 base is +X forward, +Y left, +Z up.
    # So robot_x = -vr_z, robot_y = -vr_x, robot_z = +vr_y. Flipping the planar signs
    # here would mirror every controller motion, so pin the mapping axis by axis.
    teleop = Quest3Teleop(Quest3TeleopConfig(use_hand_tracking=False))

    def robot_translation(vr_xyz):
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.array(vr_xyz, dtype=np.float64)
        teleop._controller_mats["right"] = pose
        teleop._controller_last_update_s["right"] = 0.0
        pose_robot, _states, _last_update_s = teleop._controller_pose_robot_frame("right")
        return pose_robot[:3, 3]

    # Pushing the controller forward (-Z in OpenXR) moves the arm forward (+X).
    np.testing.assert_allclose(robot_translation([0.0, 0.0, -1.0]), [1.0, 0.0, 0.0])
    # Moving it right (+X in OpenXR) moves the arm right (-Y).
    np.testing.assert_allclose(robot_translation([1.0, 0.0, 0.0]), [0.0, -1.0, 0.0])
    # Raising it (+Y in OpenXR) raises the arm (+Z).
    np.testing.assert_allclose(robot_translation([0.0, 1.0, 0.0]), [0.0, 0.0, 1.0])
    np.testing.assert_allclose(robot_translation([1.0, 2.0, 3.0]), [-3.0, -1.0, 2.0])


def test_controller_gripper_uses_right_trigger_to_close_and_release_to_open():
    # 1.0 is open, 0.0 is closed, as everywhere else in this class and in the robot
    # backends. Squeezing the trigger must close, not open.
    teleop = Quest3Teleop(Quest3TeleopConfig(use_hand_tracking=False))

    assert teleop._controller_gripper({"trigger": 0.0}, {"trigger": 0.0}) == pytest.approx(1.0)
    assert teleop._controller_gripper({"trigger": 0.7}, {"trigger": 0.0}) == pytest.approx(0.0)


def test_controller_mode_emits_per_frame_deltas_without_repeating_motion(monkeypatch):
    monkeypatch.setattr("lerobot.teleoperators.quest3.teleop_quest3.time.perf_counter", lambda: 100.0)

    teleop = Quest3Teleop(Quest3TeleopConfig(use_hand_tracking=False))
    # get_action() is guarded by @check_if_not_connected; this test drives the pose
    # bookkeeping directly, so mark the teleop connected without touching the headset.
    teleop._is_connected = True
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
