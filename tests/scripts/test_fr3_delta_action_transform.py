"""Tests for deriving a delta-EE action column from a recorded absolute-EE one.

The headline property is the one a capture-time delta got wrong: the delta must span exactly one
*dataset* frame. `test_delta_spans_one_dataset_frame_not_one_control_tick` is the regression that
would have caught it.
"""

from __future__ import annotations

import numpy as np
import pytest

from lerobot.robots.franka_research3.processor_franka_research3 import (
    delta_ee_action_keys,
    delta_ee_position_keys,
    delta_ee_rotvec_keys,
)
from lerobot.utils.rotation import Rotation

from tools.fr3.fr3_delta_action_transform import (
    DeltaTransformError,
    derive_delta_action,
    summarize_delta_scale,
)

ACTION_NAMES = ["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"]
OBSERVATION_NAMES = [
    "ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw",
    "prev_cmd.ee.x", "prev_cmd.ee.y", "prev_cmd.ee.z",
    "prev_cmd.ee.qx", "prev_cmd.ee.qy", "prev_cmd.ee.qz", "prev_cmd.ee.qw",
    "gripper.pos",
]


def _episode(
    *,
    frames: int,
    step_m: float,
    step_rad: float = 0.0,
    start=(0.50, 0.10, 0.35),
    tracking_lag_m: float = 0.0,
):
    """One episode of absolute commands advancing by step_m per dataset frame."""
    commanded_positions = np.array(
        [[start[0] + step_m * (i + 1), start[1], start[2]] for i in range(frames)], dtype=np.float64
    )
    commanded_rotations = [Rotation.from_rotvec([0.0, 0.0, step_rad * (i + 1)]) for i in range(frames)]
    commanded_quaternions = np.array([r.as_quat() for r in commanded_rotations], dtype=np.float64)

    # prev_cmd(k) is the command from frame k-1; frame 0 has none, so the robot reports the
    # measured pose (its _last_command_pose is cleared by move_to_start).
    prev_cmd_positions = np.vstack([np.asarray(start, dtype=np.float64), commanded_positions[:-1]])
    prev_cmd_quaternions = np.vstack(
        [Rotation.from_rotvec([0.0, 0.0, 0.0]).as_quat(), commanded_quaternions[:-1]]
    )
    measured_positions = commanded_positions - np.array([tracking_lag_m, 0.0, 0.0])
    if tracking_lag_m == 0.0:
        measured_positions = prev_cmd_positions.copy()
    measured_quaternions = prev_cmd_quaternions.copy()

    action = np.hstack([commanded_positions, commanded_quaternions, np.full((frames, 1), 0.5)])
    observation = np.hstack(
        [
            measured_positions,
            measured_quaternions,
            prev_cmd_positions,
            prev_cmd_quaternions,
            np.full((frames, 1), 0.5),
        ]
    )
    return action, observation, np.zeros(frames, dtype=np.int64), commanded_positions


def test_delta_spans_one_dataset_frame_not_one_control_tick():
    """Regression for the capture-time bug: the delta must equal the per-frame advance.

    Computing the delta inside the teleop pipeline stored a single control tick's increment
    (measured 0.5 mm) where the command had advanced a full frame (1.0-1.5 mm), so a policy
    trained on it drove the arm ~6.7x too slow. Differencing consecutive dataset frames is exact.
    """
    step_m = 0.0015
    action, observation, episode_index, _ = _episode(frames=10, step_m=step_m)
    delta, names, report = derive_delta_action(
        absolute_action=action,
        action_names=ACTION_NAMES,
        observation_state=observation,
        observation_names=OBSERVATION_NAMES,
        episode_index=episode_index,
        action_mode="delta_ee_from_prev_cmd",
    )
    dx = delta[:, names.index(delta_ee_position_keys("prev_cmd")[0])]
    # Every frame, including the seed frame, advances by exactly one frame's worth.
    assert np.allclose(dx, step_m, atol=1e-12)
    assert report["reconstruction_max_position_error_mm"] < 1e-6


def test_delta_scale_summary_exposes_the_implied_speed():
    """The manifest must make an implausible cadence visible without re-deriving anything."""
    step_m = 0.0015
    action, observation, episode_index, _ = _episode(frames=10, step_m=step_m)
    delta, names, _ = derive_delta_action(
        absolute_action=action,
        action_names=ACTION_NAMES,
        observation_state=observation,
        observation_names=OBSERVATION_NAMES,
        episode_index=episode_index,
        action_mode="delta_ee_from_prev_cmd",
    )
    scale = summarize_delta_scale(delta_action=delta, delta_names=names, fps=30)
    assert scale["median_translation_per_frame_mm"] == pytest.approx(1.5, abs=1e-6)
    assert scale["implied_p95_speed_mm_s"] == pytest.approx(45.0, abs=1e-3)


def test_deltas_never_cross_an_episode_boundary():
    """A delta spanning the seam would encode the move_to_start homing move as an action."""
    step_m = 0.001
    first_action, first_obs, _, _ = _episode(frames=4, step_m=step_m, start=(0.50, 0.10, 0.35))
    # Second episode starts somewhere else entirely, as it would after a re-home.
    second_action, second_obs, _, _ = _episode(frames=4, step_m=step_m, start=(0.30, -0.20, 0.50))
    action = np.vstack([first_action, second_action])
    observation = np.vstack([first_obs, second_obs])
    episode_index = np.array([0] * 4 + [1] * 4, dtype=np.int64)

    delta, names, report = derive_delta_action(
        absolute_action=action,
        action_names=ACTION_NAMES,
        observation_state=observation,
        observation_names=OBSERVATION_NAMES,
        episode_index=episode_index,
        action_mode="delta_ee_from_prev_cmd",
    )
    assert report["episodes"] == 2
    dx = delta[:, names.index(delta_ee_position_keys("prev_cmd")[0])]
    # Frame 4 is episode 1's seed; it must be one step, not the 200 mm jump between episodes.
    assert np.allclose(dx, step_m, atol=1e-12)


def test_rotation_delta_is_per_frame_and_stored_as_rotvec():
    step_rad = 0.008
    action, observation, episode_index, _ = _episode(frames=8, step_m=0.0, step_rad=step_rad)
    delta, names, _ = derive_delta_action(
        absolute_action=action,
        action_names=ACTION_NAMES,
        observation_state=observation,
        observation_names=OBSERVATION_NAMES,
        episode_index=episode_index,
        action_mode="delta_ee_from_prev_cmd",
    )
    drz = delta[:, names.index(delta_ee_rotvec_keys("prev_cmd")[2])]
    assert np.allclose(drz, step_rad, atol=1e-10)
    # rotvec, not quaternion: three rotation components and no q* column anywhere.
    assert len(delta_ee_rotvec_keys("prev_cmd")) == 3
    assert not any(name.endswith((".qx", ".qy", ".qz", ".qw")) for name in names)


def test_gripper_stays_absolute():
    action, observation, episode_index, _ = _episode(frames=5, step_m=0.001)
    delta, names, _ = derive_delta_action(
        absolute_action=action,
        action_names=ACTION_NAMES,
        observation_state=observation,
        observation_names=OBSERVATION_NAMES,
        episode_index=episode_index,
        action_mode="delta_ee_from_prev_cmd",
    )
    assert names[-1] == "gripper.pos"
    assert np.allclose(delta[:, -1], 0.5)


def test_from_current_reference_encodes_the_rigs_tracking_lag():
    """Both modes are exact, but `current` folds the rig's lag into every action.

    This is why prev_cmd is the default, and why the scale summary is worth reading before
    training on a `current` view.
    """
    lag_m = 0.004
    action, observation, episode_index, _ = _episode(frames=8, step_m=0.001, tracking_lag_m=lag_m)
    delta, names, report = derive_delta_action(
        absolute_action=action,
        action_names=ACTION_NAMES,
        observation_state=observation,
        observation_names=OBSERVATION_NAMES,
        episode_index=episode_index,
        action_mode="delta_ee_from_current",
    )
    dx = delta[:, names.index(delta_ee_position_keys("current")[0])]
    assert np.allclose(dx, lag_m, atol=1e-12)
    # Still exactly invertible -- it is a different contract, not a broken one.
    assert report["reconstruction_max_position_error_mm"] < 1e-6


def test_action_names_carry_the_reference_so_the_view_is_self_describing():
    action, observation, episode_index, _ = _episode(frames=3, step_m=0.001)
    for mode, reference in (("delta_ee_from_prev_cmd", "prev_cmd"), ("delta_ee_from_current", "current")):
        _delta, names, _report = derive_delta_action(
            absolute_action=action,
            action_names=ACTION_NAMES,
            observation_state=observation,
            observation_names=OBSERVATION_NAMES,
            episode_index=episode_index,
            action_mode=mode,
        )
        assert names == list(delta_ee_action_keys(reference))


def test_absolute_mode_is_rejected_by_the_transform():
    action, observation, episode_index, _ = _episode(frames=3, step_m=0.001)
    with pytest.raises(DeltaTransformError, match="not a delta action mode"):
        derive_delta_action(
            absolute_action=action,
            action_names=ACTION_NAMES,
            observation_state=observation,
            observation_names=OBSERVATION_NAMES,
            episode_index=episode_index,
            action_mode="absolute_ee",
        )


def test_a_delta_source_dataset_is_rejected():
    """Differencing an already-delta action would produce a second-difference, silently."""
    action, observation, episode_index, _ = _episode(frames=3, step_m=0.001)
    with pytest.raises(DeltaTransformError, match="not an absolute EE contract"):
        derive_delta_action(
            absolute_action=action[:, :7],
            action_names=list(delta_ee_action_keys("prev_cmd")),
            observation_state=observation,
            observation_names=OBSERVATION_NAMES,
            episode_index=episode_index,
            action_mode="delta_ee_from_prev_cmd",
        )


def test_missing_current_pose_blocks_the_from_current_mode():
    action, observation, episode_index, _ = _episode(frames=3, step_m=0.001)
    prev_cmd_only = [name for name in OBSERVATION_NAMES if name.startswith("prev_cmd")]
    columns = [OBSERVATION_NAMES.index(name) for name in prev_cmd_only]
    with pytest.raises(DeltaTransformError, match="delta_ee_from_current needs"):
        derive_delta_action(
            absolute_action=action,
            action_names=ACTION_NAMES,
            observation_state=observation[:, columns],
            observation_names=prev_cmd_only,
            episode_index=episode_index,
            action_mode="delta_ee_from_current",
        )
