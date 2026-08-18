#!/usr/bin/env python

"""Where the FR3 Pika tool frames actually sit, measured against the finger meshes.

``test_fr3_model_frames.py`` proves the URDF and the MJCF agree with *each other*. Agreement is not
correctness: both files can carry the same wrong literal, and here they do. ``pika_task_tcp`` is
documented as "midpoint between the two finger working points" but sits 411.8 mm from it, because
its ``0.366842`` was measured in ``quest3_pika_gripper_scene.xml`` -- a free-flying wrist frame
whose tool axis is +z, mounted 0.1765 m above the pika mesh. In the arm-mounted model
``gripper_base`` *is* the pika mesh frame and the tool axis is +x, so the same literal points across
the gripper rather than along it.

That is not a bug this test is allowed to fix. The workstation records against ``pika_task_tcp``
(``tools/fr3/fr3_record_config.yaml``) and every dataset collected so far is anchored to it; it is a
rigid frame, so record/replay/rollout stay self-consistent as long as all three name it. What the
test does is stop the two facts from drifting apart silently: that ``pika_gripper_ee`` is the real
tool point, and that ``pika_task_tcp``'s offset from it is a dataset-compatibility constant rather
than a measurement of anything.
"""

from pathlib import Path

import numpy as np
import pytest

ASSETS = Path("src/lerobot/robots/franka_research3/assets/franka_fr3")

# The base model's keyframe carries the scene's workspace-object freejoint, so it only compiles
# through the scene. See test_fr3_home_keyframe_contract.py.
SCENE = ASSETS / "fr3_pika_gripper_scene.xml"

FINGER_GEOMS = ("gripper_left_collision", "gripper_right_collision")
# A finger's "working point" is the part that touches the object: the leading slab of its mesh,
# measured along the approach axis. 5 mm is thick enough to average over mesh tessellation and thin
# enough that the jaw's shank does not pull the centroid backwards.
WORKING_POINT_SLAB_M = 0.005

# pika_gripper_ee lands on the finger-working-point midpoint to within a hair of the jaw
# tessellation; anything beyond a few mm means the meshes or the frame moved.
MAX_TOOL_POINT_ERROR_M = 0.005

# Not a measurement of anything physical: the offset every existing workstation dataset is anchored
# to. The two frames are (0.185, 0, 0) and (0, 0, 0.366842) in gripper_base, so this is exactly
# hypot(0.185, 0.366842). Changing it silently reinterprets recorded poses, so it is pinned tight
# and deliberately.
PIKA_TASK_TCP_OFFSET_M = 0.41085040217091184
MAX_OFFSET_DRIFT_M = 1e-5

# Expressed in the *tool* frame -- the frame the recorded rotvec columns describe -- the
# pika_task_tcp -> pika_gripper_ee offset is this constant, for every arm configuration. That is
# what makes an existing dataset convertible rather than dead: p_ee = p_tcp + R(rotvec) @ d.
TASK_TCP_TO_GRIPPER_EE_IN_TOOL_FRAME_M = np.array([-0.366842, 0.0, 0.185])


@pytest.fixture(scope="module")
def model_and_data():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def _gripper_base_frame(mujoco, model, data):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "gripper_base")
    assert body_id >= 0, "gripper_base missing from the scene"
    rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    origin = np.asarray(data.xpos[body_id], dtype=np.float64)
    return rotation, origin


def _body_pose_in_gripper_base(mujoco, model, data, name: str):
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body_id >= 0, f"body '{name}' missing from {SCENE.name}"
    rotation, origin = _gripper_base_frame(mujoco, model, data)
    position = rotation.T @ (np.asarray(data.xpos[body_id], dtype=np.float64) - origin)
    orientation = rotation.T @ np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    return position, orientation


def _finger_vertices_in_gripper_base(mujoco, model, data, geom_name: str) -> np.ndarray:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, geom_name)
    assert geom_id >= 0, f"geom '{geom_name}' missing from {SCENE.name}"
    mesh_id = model.geom_dataid[geom_id]
    assert mesh_id >= 0, f"geom '{geom_name}' is not a mesh; the working point cannot be measured"

    start = model.mesh_vertadr[mesh_id]
    count = model.mesh_vertnum[mesh_id]
    local = np.asarray(model.mesh_vert[start : start + count], dtype=np.float64)

    world = (np.asarray(data.geom_xmat[geom_id], dtype=np.float64).reshape(3, 3) @ local.T).T
    world += np.asarray(data.geom_xpos[geom_id], dtype=np.float64)

    rotation, origin = _gripper_base_frame(mujoco, model, data)
    return (rotation.T @ (world - origin).T).T


def _finger_working_point_midpoint(mujoco, model, data, approach_axis: np.ndarray) -> np.ndarray:
    working_points = []
    for geom_name in FINGER_GEOMS:
        vertices = _finger_vertices_in_gripper_base(mujoco, model, data, geom_name)
        reach = vertices @ approach_axis
        leading = vertices[reach > reach.max() - WORKING_POINT_SLAB_M]
        working_points.append(leading.mean(axis=0))
    return np.mean(working_points, axis=0)


def test_pika_gripper_ee_is_the_finger_working_point_midpoint(model_and_data):
    mujoco, model, data = model_and_data
    position, orientation = _body_pose_in_gripper_base(mujoco, model, data, "pika_gripper_ee")
    # The frame's own z is the approach axis; deriving it rather than hardcoding +x means this
    # still measures the right thing if the mount rotation is ever revised.
    midpoint = _finger_working_point_midpoint(mujoco, model, data, orientation[:, 2])

    error_m = float(np.linalg.norm(position - midpoint))
    assert error_m <= MAX_TOOL_POINT_ERROR_M, (
        f"pika_gripper_ee is {error_m * 1e3:.1f} mm from the finger working-point midpoint "
        f"{np.round(midpoint, 4).tolist()} in gripper_base; it is supposed to be the tool point"
    )


def test_pika_task_tcp_is_not_the_finger_working_point_midpoint(model_and_data):
    """The name and the old comment claim it is. Measurement says otherwise -- keep saying so."""
    mujoco, model, data = model_and_data
    position, orientation = _body_pose_in_gripper_base(mujoco, model, data, "pika_task_tcp")
    midpoint = _finger_working_point_midpoint(mujoco, model, data, orientation[:, 2])

    error_m = float(np.linalg.norm(position - midpoint))
    assert error_m > 0.1, (
        "pika_task_tcp now sits on the finger working-point midpoint. That is the physically right "
        "place for it, but it is not where the recorded datasets think it is: every workstation "
        "recording made against target_frame_name=pika_task_tcp is now misinterpreted by "
        f"{PIKA_TASK_TCP_OFFSET_M * 1e3:.1f} mm. Re-record, or migrate the datasets, before "
        "relaxing this."
    )


def test_the_two_tool_frames_differ_by_a_pure_translation(model_and_data):
    """Same orientation means a dataset recorded against one can be re-expressed in the other."""
    mujoco, model, data = model_and_data
    ee_position, ee_orientation = _body_pose_in_gripper_base(mujoco, model, data, "pika_gripper_ee")
    tcp_position, tcp_orientation = _body_pose_in_gripper_base(mujoco, model, data, "pika_task_tcp")

    cosine = (np.trace(ee_orientation.T @ tcp_orientation) - 1.0) / 2.0
    rotation_error_deg = float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))
    assert rotation_error_deg <= 0.1, (
        f"pika_gripper_ee and pika_task_tcp are {rotation_error_deg:.2f} deg apart; the recorded "
        "quaternions can no longer be shared between the two frames by translation alone"
    )

    offset_m = float(np.linalg.norm(tcp_position - ee_position))
    assert abs(offset_m - PIKA_TASK_TCP_OFFSET_M) <= MAX_OFFSET_DRIFT_M, (
        f"the pika_task_tcp -> pika_gripper_ee offset moved from {PIKA_TASK_TCP_OFFSET_M * 1e3:.3f} mm "
        f"to {offset_m * 1e3:.3f} mm. Existing workstation datasets are anchored to the old value"
    )


def test_the_tool_frames_put_z_along_the_approach_axis(model_and_data):
    """The IK and the recorded quaternions assume z points out of the jaws, not across them."""
    mujoco, model, data = model_and_data
    for name in ("pika_gripper_ee", "pika_task_tcp"):
        _, orientation = _body_pose_in_gripper_base(mujoco, model, data, name)
        # gripper_base +x is the direction the fingers extend; +y is the jaw-closing direction.
        assert orientation[0, 2] > 0.999, (
            f"{name}: z_tool is {np.round(orientation[:, 2], 4).tolist()} in gripper_base, not the "
            "approach axis (1, 0, 0)"
        )
        assert abs(orientation[1, 1]) > 0.999, (
            f"{name}: y_tool is {np.round(orientation[:, 1], 4).tolist()} in gripper_base, not the "
            "jaw-closing axis"
        )


def test_the_offset_is_the_same_constant_in_the_tool_frame_for_every_configuration(model_and_data):
    """Why a frame switch is a dataset migration and not a re-record.

    The two frames share an orientation, so their separation expressed in the tool frame is a rigid
    constant -- it cannot depend on the arm's pose. Together with the recorded rotvec that makes
    every position column exactly convertible:

        p_ee = p_tcp + R(rotvec) @ d

    applied to ``ee.*``, ``prev_cmd.ee.*`` and the action's ``target_*``. Joint and gripper columns
    are frame-independent and the rotations do not change at all. If this test ever fails, the
    conversion has silently stopped being exact and the migration has to be re-derived.
    """
    mujoco, model, data = model_and_data
    rng = np.random.default_rng(0)
    home = np.array(data.qpos, dtype=np.float64)

    offsets = []
    for trial in range(6):
        data.qpos[:] = home
        if trial:
            data.qpos[:7] += rng.uniform(-0.8, 0.8, 7)
        mujoco.mj_forward(model, data)

        ee_position, ee_orientation = _body_pose_in_gripper_base(mujoco, model, data, "pika_gripper_ee")
        tcp_position, _ = _body_pose_in_gripper_base(mujoco, model, data, "pika_task_tcp")
        # gripper_base is itself rigid on the wrist, so the tool-frame offset is what has to be
        # constant; express it through the tool's own orientation, which is what the datasets store.
        offsets.append(ee_orientation.T @ (ee_position - tcp_position))

    data.qpos[:] = home
    mujoco.mj_forward(model, data)

    offsets = np.asarray(offsets)
    spread_m = float(np.abs(offsets.max(axis=0) - offsets.min(axis=0)).max())
    assert spread_m <= 1e-9, (
        f"the tool-frame offset varies by {spread_m * 1e3:.6f} mm across arm configurations; "
        "p_ee = p_tcp + R @ d is no longer an exact dataset conversion"
    )

    np.testing.assert_allclose(
        offsets.mean(axis=0),
        TASK_TCP_TO_GRIPPER_EE_IN_TOOL_FRAME_M,
        atol=1e-9,
        err_msg=(
            "the pika_task_tcp -> pika_gripper_ee tool-frame offset moved. Any dataset already "
            "migrated with the old constant is now inconsistent with one migrated with the new."
        ),
    )
