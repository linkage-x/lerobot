"""Contract tests for the FR3 delta-EE action modes.

The properties pinned here are the ones a silent regression would hide: the delta must be an
exact inverse of the reconstruction, the rotation convention must stay body-frame
right-multiplied, and the two references must remain distinguishable from a dataset alone.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.processor.core import TransitionKey
from lerobot.robots.franka_research3.action_modes import (
    ACTION_MODE_ABSOLUTE_EE,
    ACTION_MODE_DELTA_EE_FROM_CURRENT,
    ACTION_MODE_DELTA_EE_FROM_PREV_CMD,
    delta_reference_for_action_mode,
    is_delta_action_mode,
    validate_action_mode,
)
from lerobot.robots.franka_research3.processor_franka_research3 import (
    AbsoluteEEToDeltaEEAction,
    DeltaEEToAbsoluteEEAction,
    delta_ee_action_keys,
    delta_ee_position_keys,
    delta_ee_rotvec_keys,
    delta_reference_from_action_names,
)
from lerobot.utils.rotation import Rotation

REFERENCES = ("prev_cmd", "current")


def _observation(reference_position, reference_rotvec, *, measured_offset=0.0):
    """Observation carrying both delta references, offset so they are distinguishable."""
    measured = np.asarray(reference_position, dtype=np.float64) + measured_offset
    return {
        "ee.x": float(measured[0]),
        "ee.y": float(measured[1]),
        "ee.z": float(measured[2]),
        "ee.wx": float(reference_rotvec[0]),
        "ee.wy": float(reference_rotvec[1]),
        "ee.wz": float(reference_rotvec[2]),
        "prev_cmd.ee.x": float(reference_position[0]),
        "prev_cmd.ee.y": float(reference_position[1]),
        "prev_cmd.ee.z": float(reference_position[2]),
        "prev_cmd.ee.wx": float(reference_rotvec[0]),
        "prev_cmd.ee.wy": float(reference_rotvec[1]),
        "prev_cmd.ee.wz": float(reference_rotvec[2]),
    }


def _absolute_action(position, rotation_matrix, gripper=0.4):
    quaternion = Rotation.from_matrix(rotation_matrix).as_quat()
    return {
        "enabled": True,
        "ee.x": float(position[0]),
        "ee.y": float(position[1]),
        "ee.z": float(position[2]),
        "ee.qx": float(quaternion[0]),
        "ee.qy": float(quaternion[1]),
        "ee.qz": float(quaternion[2]),
        "ee.qw": float(quaternion[3]),
        "gripper.pos": gripper,
    }


def _apply(step, action, observation):
    result = step({TransitionKey.ACTION: dict(action), TransitionKey.OBSERVATION: dict(observation)})
    return result[TransitionKey.ACTION]


# --------------------------------------------------------------------------- action modes ---


def test_action_mode_helpers_agree_on_which_modes_are_delta():
    assert not is_delta_action_mode(ACTION_MODE_ABSOLUTE_EE)
    assert is_delta_action_mode(ACTION_MODE_DELTA_EE_FROM_PREV_CMD)
    assert is_delta_action_mode(ACTION_MODE_DELTA_EE_FROM_CURRENT)
    assert delta_reference_for_action_mode(ACTION_MODE_DELTA_EE_FROM_PREV_CMD) == "prev_cmd"
    assert delta_reference_for_action_mode(ACTION_MODE_DELTA_EE_FROM_CURRENT) == "current"
    with pytest.raises(ValueError, match="not a delta action mode"):
        delta_reference_for_action_mode(ACTION_MODE_ABSOLUTE_EE)
    with pytest.raises(ValueError, match="action_mode must be one of"):
        validate_action_mode("delta")


def test_the_two_delta_references_are_distinguishable_from_feature_names_alone():
    """A dataset must say which reference it used, or an offline tool can integrate it wrongly."""
    prev_cmd_keys = delta_ee_action_keys("prev_cmd")
    current_keys = delta_ee_action_keys("current")
    assert set(prev_cmd_keys) != set(current_keys)
    assert delta_reference_from_action_names(prev_cmd_keys) == "prev_cmd"
    assert delta_reference_from_action_names(current_keys) == "current"
    assert delta_reference_from_action_names(["ee.x", "ee.qx", "gripper.pos"]) is None


def test_rotation_delta_is_a_three_component_rotvec_not_a_quaternion():
    """rotvec, deliberately: a clamped delta's quaternion qw carries ~8 bits in float32, and
    recovering the angle through acos near 1 amplifies regression error by ~80x."""
    keys = delta_ee_action_keys("prev_cmd")
    assert len(delta_ee_rotvec_keys("prev_cmd")) == 3
    assert not any(component in key for key in keys for component in (".qx", ".qy", ".qz", ".qw"))
    # 3 translation + 3 rotation + absolute gripper, with the gripper last.
    assert len(keys) == 7
    assert keys[-1] == "gripper.pos"


# -------------------------------------------------------------------------- round trip ---


@pytest.mark.parametrize("reference", REFERENCES)
def test_delta_is_the_exact_inverse_of_the_reconstruction(reference):
    forward = AbsoluteEEToDeltaEEAction(reference=reference)
    inverse = DeltaEEToAbsoluteEEAction(reference=reference)
    rng = np.random.default_rng(1234)

    worst_position_error_m = 0.0
    worst_rotation_error_deg = 0.0
    for _ in range(200):
        reference_position = rng.uniform(0.3, 0.7, 3)
        axis = rng.normal(0.0, 1.0, 3)
        reference_rotvec = axis / np.linalg.norm(axis) * rng.uniform(0.0, np.pi)
        # Measured pose deliberately offset from the commanded one, so a mode that used the wrong
        # reference would produce a different delta and fail the recovery assertion below.
        observation = _observation(reference_position, reference_rotvec, measured_offset=-0.01)

        delta_position = rng.uniform(-0.001, 0.001, 3)
        delta_rotvec = rng.uniform(-0.01, 0.01, 3)
        if reference == "current":
            base_position = reference_position - 0.01
        else:
            base_position = reference_position
        base_rotation = Rotation.from_rotvec(reference_rotvec).as_matrix()
        desired_position = base_position + delta_position
        desired_rotation = base_rotation @ Rotation.from_rotvec(delta_rotvec).as_matrix()

        delta = _apply(forward, _absolute_action(desired_position, desired_rotation), observation)
        recovered_position = np.array([delta[key] for key in delta_ee_position_keys(reference)])
        recovered_rotvec = np.array([delta[key] for key in delta_ee_rotvec_keys(reference)])
        assert np.allclose(recovered_position, delta_position, atol=1e-12)
        assert np.allclose(recovered_rotvec, delta_rotvec, atol=1e-10)

        rebuilt = _apply(inverse, delta, observation)
        rebuilt_position = np.array([rebuilt["ee.x"], rebuilt["ee.y"], rebuilt["ee.z"]])
        rebuilt_rotation = Rotation.from_quat(
            [rebuilt["ee.qx"], rebuilt["ee.qy"], rebuilt["ee.qz"], rebuilt["ee.qw"]]
        ).as_matrix()
        worst_position_error_m = max(
            worst_position_error_m, float(np.abs(rebuilt_position - desired_position).max())
        )
        worst_rotation_error_deg = max(
            worst_rotation_error_deg,
            float(np.degrees(np.linalg.norm(Rotation.from_matrix(rebuilt_rotation.T @ desired_rotation).as_rotvec()))),
        )

    assert worst_position_error_m < 1e-12
    assert worst_rotation_error_deg < 1e-9


def test_rotation_delta_is_body_frame_right_multiplied():
    """desired_R = reference_R @ delta_R. A left-multiplied (world-frame) delta is a different
    rotation whenever the reference is not identity, and would corrupt orientation silently."""
    forward = AbsoluteEEToDeltaEEAction(reference="prev_cmd")
    reference_rotvec = np.array([0.0, 0.0, np.pi / 2.0])
    reference_rotation = Rotation.from_rotvec(reference_rotvec).as_matrix()
    delta_rotvec = np.array([0.01, 0.0, 0.0])
    observation = _observation([0.5, 0.0, 0.3], reference_rotvec)

    right = reference_rotation @ Rotation.from_rotvec(delta_rotvec).as_matrix()
    delta = _apply(forward, _absolute_action([0.5, 0.0, 0.3], right), observation)
    assert np.allclose(
        [delta[key] for key in delta_ee_rotvec_keys("prev_cmd")], delta_rotvec, atol=1e-12
    )

    # The same delta applied on the left yields a different absolute rotation, so the recorded
    # delta must differ from delta_rotvec -- proving the convention is not accidentally symmetric.
    left = Rotation.from_rotvec(delta_rotvec).as_matrix() @ reference_rotation
    delta_left = _apply(forward, _absolute_action([0.5, 0.0, 0.3], left), observation)
    assert not np.allclose(
        [delta_left[key] for key in delta_ee_rotvec_keys("prev_cmd")], delta_rotvec, atol=1e-6
    )


def test_translation_delta_is_world_frame():
    """Translation is a plain world-frame difference, matching DeltaActionToAbsoluteEEAction,
    even when the reference orientation is far from identity."""
    forward = AbsoluteEEToDeltaEEAction(reference="prev_cmd")
    reference_rotvec = np.array([0.0, 0.0, np.pi / 2.0])
    observation = _observation([0.5, 0.0, 0.3], reference_rotvec)
    desired_position = np.array([0.5 + 0.002, 0.0, 0.3])

    delta = _apply(
        forward,
        _absolute_action(desired_position, Rotation.from_rotvec(reference_rotvec).as_matrix()),
        observation,
    )
    assert np.isclose(delta["delta_ee_from_prev_cmd.dx"], 0.002, atol=1e-12)
    assert np.isclose(delta["delta_ee_from_prev_cmd.dy"], 0.0, atol=1e-12)


# ------------------------------------------------------------------------------ guards ---


def test_absurd_rotation_delta_is_rejected_rather_than_aliased():
    forward = AbsoluteEEToDeltaEEAction(reference="prev_cmd")
    observation = _observation([0.5, 0.0, 0.3], [0.0, 0.0, 0.0])
    huge = Rotation.from_rotvec([0.0, 0.0, 2.5]).as_matrix()
    with pytest.raises(ValueError, match="exceeds"):
        _apply(forward, _absolute_action([0.5, 0.0, 0.3], huge), observation)


def test_missing_reference_keys_fail_instead_of_falling_back_to_the_other_reference():
    """Substituting the measured pose for a missing prev_cmd would silently change the action
    contract, which is exactly the kind of quiet basis switch that must never happen."""
    forward = AbsoluteEEToDeltaEEAction(reference="prev_cmd")
    observation = {
        "ee.x": 0.5,
        "ee.y": 0.0,
        "ee.z": 0.3,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
    }
    with pytest.raises(ValueError, match="missing"):
        _apply(forward, _absolute_action([0.5, 0.0, 0.3], np.eye(3)), observation)


def test_reconstruction_passes_absolute_actions_through_untouched():
    """One pipeline serves both contracts: a hold frame from the absolute path must not be
    mangled by the delta reconstruction step sitting in front of the robot adapter."""
    inverse = DeltaEEToAbsoluteEEAction(reference="prev_cmd")
    observation = _observation([0.5, 0.0, 0.3], [0.0, 0.0, 0.0])
    idle = {
        "enabled": False,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }
    assert _apply(inverse, idle, observation) == idle


def test_reconstruction_clamps_to_the_workspace_when_bounds_are_given():
    """At deployment the policy's delta is not clamped by the recorder, so the reconstruction is
    the guard that keeps it inside the safe envelope."""
    inverse = DeltaEEToAbsoluteEEAction(
        reference="prev_cmd", workspace_min=(0.2, -0.6, 0.05), workspace_max=(0.55, 0.6, 0.8)
    )
    observation = _observation([0.5, 0.0, 0.3], [0.0, 0.0, 0.0])
    runaway = dict.fromkeys(delta_ee_rotvec_keys("prev_cmd"), 0.0)
    runaway.update(dict(zip(delta_ee_position_keys("prev_cmd"), (0.5, 0.0, 0.0), strict=True)))
    runaway["gripper.pos"] = 0.5

    rebuilt = _apply(inverse, runaway, observation)
    assert np.isclose(rebuilt["ee.x"], 0.55)


@pytest.mark.parametrize("reference", REFERENCES)
def test_zero_delta_reconstructs_to_the_reference_pose(reference):
    """The property that makes the delta contract deployable without an `enabled` flag."""
    inverse = DeltaEEToAbsoluteEEAction(reference=reference)
    reference_position = np.array([0.5, 0.1, 0.3])
    observation = _observation(reference_position, [0.0, 0.0, 0.0], measured_offset=-0.01)
    zero = dict.fromkeys(delta_ee_position_keys(reference) + delta_ee_rotvec_keys(reference), 0.0)
    zero["gripper.pos"] = 0.5

    rebuilt = _apply(inverse, zero, observation)
    expected_x = reference_position[0] - (0.01 if reference == "current" else 0.0)
    assert np.isclose(rebuilt["ee.x"], expected_x, atol=1e-12)
