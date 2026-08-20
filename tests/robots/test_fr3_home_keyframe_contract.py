#!/usr/bin/env python

"""The workstation's start pose lives in three files; only one of them is the source of truth.

``fr3_pika_gripper.xml``'s ``home`` keyframe is what the recorder homes to between episodes, and
``T_B_Ws`` is solved from the first observation against the dataset's start pose -- so that pose is
what places an entire recorded trajectory in the workspace. It is deliberately *not* the DAS rig's
joint configuration and not ``Panda.move_to_start()``.

The seven angles are then hand-copied into ``fr3_record_config.yaml`` (the recorder) and
``fr3_move_to_start_runtime.py`` (the rollout launcher's homing step). Both copies carry a comment
naming the XML as authoritative, which is a claim nothing checked. Editing the keyframe without
touching the copies would move the recording start pose and leave rollouts homing to the old one --
a distribution shift with no error message anywhere.
"""

from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pytest
import yaml

from tools.fr3 import fr3_move_to_start_runtime

ASSETS = Path("src/lerobot/robots/franka_research3/assets/franka_fr3")
PIKA_XML = ASSETS / "fr3_pika_gripper.xml"
SCENE_XML = ASSETS / "fr3_pika_gripper_scene.xml"
RECORD_CONFIG = Path("tools/fr3/fr3_record_config.yaml")

ARM_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
# Not the runtime's arrival tolerance: that one absorbs controller residual on real hardware, and
# is two orders of magnitude looser than anything a *file* should be allowed to drift by. Two
# hand-copied numbers either agree or they do not.
MAX_JOINT_DRIFT_RAD = 1e-9


def _home_keyframe_qpos() -> np.ndarray:
    """Read the keyframe out of the XML text.

    Deliberately a text parse, not ``MjModel.from_xml_path``: the base model cannot be compiled on
    its own (see the standalone-compile test below), and this is the file every comment in the tree
    points at, so it is the file the test has to read.
    """
    root = ElementTree.parse(PIKA_XML).getroot()
    keys = root.findall("./keyframe/key[@name='home']")
    assert len(keys) == 1, f"expected exactly one 'home' keyframe in {PIKA_XML.name}, found {len(keys)}"
    return np.asarray([float(value) for value in keys[0].get("qpos").split()], dtype=np.float64)


def _home_arm_joints() -> np.ndarray:
    return _home_keyframe_qpos()[: len(ARM_JOINT_NAMES)]


def test_the_first_seven_keyframe_values_really_are_the_arm_joints():
    """The copies slice ``qpos[:7]``. That is only the arm if MuJoCo lays the model out that way."""
    mujoco = pytest.importorskip("mujoco")

    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)

    expected = _home_arm_joints()
    for index, joint_name in enumerate(ARM_JOINT_NAMES):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert joint_id >= 0, f"joint '{joint_name}' missing from {SCENE_XML.name}"
        address = model.jnt_qposadr[joint_id]
        assert address == index, (
            f"{joint_name} sits at qpos[{address}], not qpos[{index}]; the tools that slice the "
            "keyframe's first seven values are no longer reading the arm"
        )
        assert data.qpos[address] == pytest.approx(expected[index], abs=MAX_JOINT_DRIFT_RAD)


def test_the_base_model_cannot_be_compiled_without_its_scene():
    """A guard on the parse above, and a signpost for anyone who tries to open the file directly.

    ``home``'s qpos is 16 long: 7 arm joints, 2 gripper slides, and 7 more belonging to the
    ``workspace_object_body`` freejoint that only exists in the scene. So the file every comment
    calls the source of truth raises ``invalid qpos size, expected length 9`` on its own. Harmless
    today -- every consumer loads the scene -- but it is why this test reads the XML as text.
    """
    mujoco = pytest.importorskip("mujoco")

    assert len(_home_keyframe_qpos()) == 16
    with pytest.raises(ValueError, match="invalid qpos size"):
        mujoco.MjModel.from_xml_path(str(PIKA_XML))


def test_the_homing_runtime_matches_the_keyframe():
    expected = _home_arm_joints()
    actual = np.asarray(fr3_move_to_start_runtime.FR3_PIKA_HOME_JOINTS_RAD, dtype=np.float64)

    assert actual.shape == expected.shape
    np.testing.assert_allclose(
        actual,
        expected,
        atol=MAX_JOINT_DRIFT_RAD,
        err_msg=(
            "fr3_move_to_start_runtime.FR3_PIKA_HOME_JOINTS_RAD has drifted from "
            f"{PIKA_XML.name}:keyframe/home. The rollout would home the arm somewhere the episodes "
            "were not recorded from."
        ),
    )


def test_the_record_config_start_pose_matches_the_keyframe():
    config = yaml.safe_load(RECORD_CONFIG.read_text(encoding="utf-8"))
    configured = config["robot"].get("start_joint_positions")
    assert configured is not None, (
        f"{RECORD_CONFIG.name} no longer pins start_joint_positions; the recorder would fall back "
        "to the arm backend's own start pose, which on this rig is not the recording contract"
    )

    np.testing.assert_allclose(
        np.asarray(configured, dtype=np.float64),
        _home_arm_joints(),
        atol=MAX_JOINT_DRIFT_RAD,
        err_msg=(
            f"{RECORD_CONFIG.name}:robot.start_joint_positions has drifted from "
            f"{PIKA_XML.name}:keyframe/home. New episodes would start from a different pose than "
            "the ones already recorded."
        ),
    )
