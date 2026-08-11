#!/usr/bin/env python

"""The URDF and the MJCF of one FR3 rig must agree about where the tool is.

The hardware arm solves IK against the URDF; the simulated arm solves it against the MJCF. When
the two files disagree about the flange-to-tool transform, nothing fails loudly -- a sim
recording is self-consistent, a hardware recording is self-consistent, and only replaying one
through the other reveals that they were never the same robot. Copying the URDF's ``rpy``
triplets into MuJoCo ``euler`` attributes did exactly that: same numbers, different rotation
convention (fixed-axis ``Rz*Ry*Rx`` versus ``eulerseq="xyz"``'s ``Rx*Ry*Rz``), 165 deg apart at
the gripper mount and 684 mm at the TCP.
"""

from pathlib import Path

import numpy as np
import pytest

from lerobot.robots.franka_research3.validate_frame_contract import parse_urdf_joint_transform

ASSETS = Path("src/lerobot/robots/franka_research3/assets/franka_fr3")
ARM_JOINT_NAMES = tuple(f"fr3_joint{index}" for index in range(1, 8))
# Nothing special about these angles beyond being away from zero on every joint, so a frame error
# cannot hide behind a coincidentally aligned axis.
PROBE_JOINT_POSITIONS = (-0.0953, 0.3979, -0.2361, -2.3901, 1.7232, 0.6167, 2.1407)

MAX_POSITION_ERROR_M = 1e-3
MAX_ROTATION_ERROR_DEG = 0.1

# (mjcf body, urdf joint chain from fr3_link8)
TOOL_FRAMES = (
    ("gripper_base", ("fr3_gripper_joint",)),
    ("pika_gripper_ee", ("fr3_gripper_joint", "fr3_hand_tcp_joint")),
    ("pika_task_tcp", ("fr3_gripper_joint", "pika_task_tcp_joint")),
)


def _se3_inverse(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -transform[:3, :3].T @ transform[:3, 3]
    return inverse


def _rotation_error_deg(expected: np.ndarray, actual: np.ndarray) -> float:
    cosine = (np.trace(expected[:3, :3].T @ actual[:3, :3]) - 1.0) / 2.0
    return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))


def _mjcf_flange_relative_poses(scene_path: Path) -> dict[str, np.ndarray]:
    mujoco = pytest.importorskip("mujoco")

    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    for joint_name, position in zip(ARM_JOINT_NAMES, PROBE_JOINT_POSITIONS, strict=True):
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        assert joint_id >= 0, f"joint '{joint_name}' missing from {scene_path.name}"
        data.qpos[model.jnt_qposadr[joint_id]] = position
    mujoco.mj_forward(model, data)

    def body_pose(name: str) -> np.ndarray:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        assert body_id >= 0, f"body '{name}' missing from {scene_path.name}"
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.asarray(data.xpos[body_id], dtype=np.float64)
        pose[:3, :3] = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
        return pose

    flange_inverse = _se3_inverse(body_pose("fr3_link8"))
    return {name: flange_inverse @ body_pose(name) for name, _ in TOOL_FRAMES}


def _urdf_flange_relative_pose(urdf_path: Path, joint_chain: tuple[str, ...]) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    for joint_name in joint_chain:
        transform = parse_urdf_joint_transform(str(urdf_path), joint_name)
        assert transform is not None, f"joint '{joint_name}' missing from {urdf_path.name}"
        pose = pose @ transform
    return pose


@pytest.mark.parametrize(("body_name", "joint_chain"), TOOL_FRAMES)
def test_pika_gripper_mjcf_tool_frames_match_the_urdf(body_name: str, joint_chain: tuple[str, ...]):
    mjcf_poses = _mjcf_flange_relative_poses(ASSETS / "fr3_pika_gripper_scene.xml")
    urdf_pose = _urdf_flange_relative_pose(ASSETS / "fr3_pika_gripper.urdf", joint_chain)

    position_error_m = float(np.linalg.norm(mjcf_poses[body_name][:3, 3] - urdf_pose[:3, 3]))
    rotation_error_deg = _rotation_error_deg(urdf_pose, mjcf_poses[body_name])

    assert position_error_m <= MAX_POSITION_ERROR_M, (
        f"{body_name}: MJCF and URDF place it {position_error_m * 1e3:.2f} mm apart "
        f"relative to fr3_link8"
    )
    assert rotation_error_deg <= MAX_ROTATION_ERROR_DEG, (
        f"{body_name}: MJCF and URDF orient it {rotation_error_deg:.2f} deg apart "
        f"relative to fr3_link8"
    )
