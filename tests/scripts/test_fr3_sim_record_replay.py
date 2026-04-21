#!/usr/bin/env python

from __future__ import annotations

import numpy as np

from lerobot.utils.rotation import Rotation
from tools.fr3 import fr3_sim_record_replay_runtime


def test_pose_from_xyzquat():
    xyzquat = np.array([0.1, 0.2, 0.3, 0.0, 0.0, 0.0, 1.0])
    T = fr3_sim_record_replay_runtime.pose_from_xyzquat(xyzquat)

    assert T.shape == (4, 4)
    assert np.allclose(T[:3, 3], [0.1, 0.2, 0.3])
    assert np.allclose(T[3, :], [0, 0, 0, 1])
    assert np.isclose(np.linalg.det(T[:3, :3]), 1.0)


def test_pose_from_xyzquat_and_back():
    original = np.array([0.5, -0.3, 0.8, 0.707, 0.0, 0.707, 0.0])
    T = fr3_sim_record_replay_runtime.pose_from_xyzquat(original)

    assert np.allclose(T[:3, 3], original[:3])
    rot = Rotation.from_quat(original[3:7])
    assert np.allclose(T[:3, :3], rot.as_matrix())


def test_get_joint_ids_raises_on_missing_joint():
    import mujoco

    xml = """
    <mujoco model="test">
        <worldbody>
            <body name="base">
                <joint name="test_joint" type="hinge" axis="1 0 0"/>
            </body>
        </worldbody>
    </mujoco>
    """
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)

    try:
        fr3_sim_record_replay_runtime.get_joint_ids(mujoco, model, ["nonexistent_joint"])
        assert False, "Should have raised ValueError"
    except ValueError as e:
        assert "not in MuJoCo model" in str(e)


def test_load_episode_structure():
    ep = {
        "state": np.zeros((10, 22)),
        "action": np.zeros((10, 8)),
        "timestamp": np.zeros(10),
    }
    assert ep["state"].shape == (10, 22)
    assert ep["action"].shape == (10, 8)
    assert ep["timestamp"].shape == (10,)
