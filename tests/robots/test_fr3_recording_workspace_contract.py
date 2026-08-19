#!/usr/bin/env python

"""The recording rig's tool frame and its workspace fence only mean anything together.

``send_action`` clips the *target frame origin* to ``workspace_min/max`` (franka_research3.py), so
the box describes wherever the configured frame happens to be -- not a region of the room. The two
FR3 tool frames are 410.85 mm apart, so naming the other one silently moves the fence by that much
while every number in it stays the same. The rig switched to ``pika_gripper_ee`` (the finger
working-point midpoint) so that a rotation command pivots at the fingertips instead of swinging
them through a 0.41 m arc; the box was re-derived from the workstation table in the same change.

What this file pins is the *joint* contract: the frame the config names, the fence it names, the
table those numbers were read off, and the home pose the recorder returns to between episodes.
Changing any one of them without the others is the failure this catches -- none of which raises,
and one of which (a fence 411 mm from where the comment says) is invisible until an operator
notices the arm refusing to move in a direction it should.
"""

from pathlib import Path

import numpy as np
import pytest
import yaml

ASSETS = Path("src/lerobot/robots/franka_research3/assets/franka_fr3")
SCENE_XML = ASSETS / "fr3_pika_gripper_scene.xml"
RECORD_CONFIG = Path("tools/fr3/fr3_record_config.yaml")

# The frame the rig records against. Named here as well as in the YAML so a change has to be
# deliberate in two places, one of which explains why.
EXPECTED_TARGET_FRAME = "pika_gripper_ee"

# Homing to a pose that already sits against the fence means the operator's first nudge in that
# direction is clipped, with no message. 50 mm is small next to the box (520 x 900 x 700 mm) and
# large enough to fail if the frame is switched without re-deriving the box: at pika_task_tcp the
# home pose clears the +x wall by 25 mm.
MIN_HOME_MARGIN_M = 0.05

# The FR3's reach. A box is not a sphere, so its far-high corners are outside it by construction;
# what matters is that the box is not mostly unreachable.
FR3_REACH_M = 0.855
MIN_REACHABLE_CORNERS = 6


@pytest.fixture(scope="module")
def record_config() -> dict:
    return yaml.safe_load(RECORD_CONFIG.read_text())


@pytest.fixture(scope="module")
def robot_config(record_config) -> dict:
    robot = record_config.get("robot")
    assert isinstance(robot, dict), f"{RECORD_CONFIG} has no robot: block"
    return robot


@pytest.fixture(scope="module")
def scene():
    mujoco = pytest.importorskip("mujoco")
    model = mujoco.MjModel.from_xml_path(str(SCENE_XML))
    data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, data, 0)
    mujoco.mj_forward(model, data)
    return mujoco, model, data


def _base_origin(mujoco, model, data) -> np.ndarray:
    """The scene mounts the arm on a 0.4 m pedestal; every config number is in the base frame."""
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link0")
    assert body_id >= 0, "fr3_link0 missing from the scene"
    rotation = np.asarray(data.xmat[body_id], dtype=np.float64).reshape(3, 3)
    assert np.allclose(rotation, np.eye(3), atol=1e-9), (
        "fr3_link0 is rotated in the scene; the config's axis-aligned box is no longer axis-aligned "
        "in the base frame and every bound below has to be re-derived"
    )
    return np.asarray(data.xpos[body_id], dtype=np.float64)


def _frame_origin_in_base(mujoco, model, data, name: str) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    assert body_id >= 0, f"body '{name}' missing from {SCENE_XML.name}"
    return np.asarray(data.xpos[body_id], dtype=np.float64) - _base_origin(mujoco, model, data)


def _table_bounds_in_base(mujoco, model, data) -> tuple[np.ndarray, np.ndarray]:
    geom_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "table")
    assert geom_id >= 0, f"the 'table' geom is missing from {SCENE_XML.name}"
    centre = np.asarray(data.geom_xpos[geom_id], dtype=np.float64) - _base_origin(mujoco, model, data)
    half = np.asarray(model.geom_size[geom_id], dtype=np.float64)
    return centre - half, centre + half


def _box(robot_config) -> tuple[np.ndarray, np.ndarray]:
    lo, hi = robot_config.get("workspace_min"), robot_config.get("workspace_max")
    assert lo is not None and hi is not None, (
        "fr3_record_config.yaml does not set workspace_min/workspace_max, so the fence falls back "
        "to FrankaResearch3Config's (0.2,-0.6,0.05)-(0.9,0.6,0.8). That default was never derived "
        "for this rig and describes a region 411 mm from where this config's tool frame is"
    )
    return np.asarray(lo, dtype=np.float64), np.asarray(hi, dtype=np.float64)


def test_the_rig_records_against_the_tool_point(robot_config):
    assert robot_config.get("target_frame_name") == EXPECTED_TARGET_FRAME, (
        f"the recording config names '{robot_config.get('target_frame_name')}'. Switching the tool "
        "frame is a dataset boundary: existing episodes are anchored to the frame they were "
        "recorded in and are not replayable against the other one without converting them "
        "(p_ee = p_tcp + R(quat) @ [-0.366842, 0, 0.185]). If the switch is intended, re-derive "
        "workspace_min/max at the new frame and update this test with the reason"
    )


def test_the_workspace_floor_is_the_tabletop(robot_config, scene):
    """z_min is not a round number that happens to look safe -- it is where the table is."""
    mujoco, model, data = scene
    lo, _hi = _box(robot_config)
    _table_lo, table_hi = _table_bounds_in_base(mujoco, model, data)

    assert lo[2] == pytest.approx(table_hi[2], abs=1e-9), (
        f"the workspace floor is z={lo[2]:.4f} but the tabletop is at z={table_hi[2]:.4f} in the "
        "base frame. Below the table the fence stops being the thing that keeps the fingertips out "
        "of the surface; above it, part of the table is unreachable with no note saying so"
    )


def test_the_workspace_stays_over_the_table(robot_config, scene):
    """x/y were read off the table footprint. If the table moves, the derivation is stale."""
    mujoco, model, data = scene
    lo, hi = _box(robot_config)
    table_lo, table_hi = _table_bounds_in_base(mujoco, model, data)

    for axis, name in ((0, "x"), (1, "y")):
        assert lo[axis] >= table_lo[axis] - 1e-9 and hi[axis] <= table_hi[axis] + 1e-9, (
            f"the workspace spans {name} [{lo[axis]:.3f}, {hi[axis]:.3f}] but the table only covers "
            f"[{table_lo[axis]:.3f}, {table_hi[axis]:.3f}]. The fence now permits commanding the "
            "fingertips past the edge of the surface the task sits on"
        )


def test_the_home_pose_sits_well_inside_the_fence(robot_config, scene):
    """The recorder homes here between every episode, and teleop starts from it."""
    mujoco, model, data = scene
    lo, hi = _box(robot_config)
    home = _frame_origin_in_base(mujoco, model, data, robot_config["target_frame_name"])

    margins = np.minimum(home - lo, hi - home)
    assert margins.min() >= MIN_HOME_MARGIN_M, (
        f"{robot_config['target_frame_name']} homes to {np.round(home, 4).tolist()}, which clears "
        f"the workspace fence by only {np.round(margins, 4).tolist()} m. The recorder returns here "
        "after every episode, so the operator's first command in the tight direction is silently "
        "clipped. Either the frame changed without re-deriving the box, or the box is too small"
    )


def test_most_of_the_fence_is_actually_reachable(robot_config):
    lo, hi = _box(robot_config)
    corners = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    reachable = int((np.linalg.norm(corners, axis=1) <= FR3_REACH_M).sum())

    assert reachable >= MIN_REACHABLE_CORNERS, (
        f"only {reachable}/8 corners of the workspace are within the FR3's {FR3_REACH_M * 1e3:.0f} mm "
        "reach. A fence that mostly encloses unreachable space is not doing the job the clip is "
        "there for; IK failures start standing in for it"
    )


def test_no_rotation_axis_is_pinned_off(record_config):
    """The reason the frame was switched. Re-pinning them undoes it without touching the frame."""
    teleop = record_config.get("teleop") or {}
    for axis in ("scale_wx", "scale_wy", "scale_wz"):
        assert teleop.get(axis) != 0, (
            f"teleop.{axis} is pinned to 0. Roll and pitch were disabled because rotation about "
            "pika_task_tcp swung the fingertips through a 0.41 m arc; recording against "
            f"{EXPECTED_TARGET_FRAME} is what makes them usable. Turning them back off means the "
            "frame switch bought nothing -- say why here if that is intended"
        )


def test_the_per_step_clamps_are_still_declared(robot_config):
    """They are the real speed guard. At the tool point 1 mm/step is the fingertip step itself."""
    for field in ("max_target_delta_pos", "max_target_delta_rot"):
        values = robot_config.get(field)
        assert values is not None and len(values) == 3, (
            f"{field} is no longer a 3-vector in the recording config; without it "
            "FrankaResearch3Config leaves the per-step delta unclamped"
        )
        assert all(float(value) > 0 for value in values), f"{field} must be positive"
