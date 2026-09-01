"""The one place a commanded pose is smoothed and bounded before it reaches the arm.

Split out of ``fr3_act_infer_real_runtime`` so that everything which can drive this arm shares
a single copy of these rules. Inference is no longer the only such thing: the DAgger rehearsal
in ``dagger_sim_dryrun`` hands the arm back and forth between a scripted stream and a human,
and the whole point of rehearsing it is that the guard it exercises is the guard the real
rollout will use. A second implementation would rehearse the wrong thing.

The module deliberately imports nothing beyond numpy and ``lerobot.utils.rotation``. The
runtime's own import chain reaches ``lerobot.policies``, which needs a GPU-scale install and is
currently unimportable on the workstation; keeping the guard clear of it is what lets the
handoff be tested on a laptop, which is where the reasoning errors are cheapest to find.

Two references, two different meanings -- see ``limit_command_for_safety``.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from lerobot.utils.rotation import Rotation

# Copied rather than imported from ``processor_franka_research3`` for the reason in the module
# docstring. ``fr3_act_infer_real_runtime`` asserts these against the processor's own tuples at
# import time, so a rename there fails loudly on the rig instead of silently disabling the step
# guard here (a missing key makes ``_extract_previous_command_pose`` return None).
PREV_CMD_POSITION_KEYS = ('prev_cmd.ee.x', 'prev_cmd.ee.y', 'prev_cmd.ee.z')
PREV_CMD_ROTVEC_KEYS = ('prev_cmd.ee.wx', 'prev_cmd.ee.wy', 'prev_cmd.ee.wz')

RobotObservation = dict[str, Any]


def _extract_observation_pose(robot_observation: RobotObservation) -> tuple[np.ndarray, Rotation]:
    position = np.asarray(
        [robot_observation['ee.x'], robot_observation['ee.y'], robot_observation['ee.z']],
        dtype=np.float64,
    )
    rotation = Rotation.from_rotvec(
        [robot_observation['ee.wx'], robot_observation['ee.wy'], robot_observation['ee.wz']]
    )
    return position, rotation


def _extract_command_pose(robot_command: dict[str, float]) -> tuple[np.ndarray, Rotation]:
    position = np.asarray([robot_command['ee.x'], robot_command['ee.y'], robot_command['ee.z']], dtype=np.float64)
    rotation = Rotation.from_rotvec([robot_command['ee.wx'], robot_command['ee.wy'], robot_command['ee.wz']])
    return position, rotation


def compute_pose_delta_from_current(
    robot_command: dict[str, float],
    robot_observation: RobotObservation,
) -> tuple[np.ndarray, np.ndarray]:
    current_position, current_rotation = _extract_observation_pose(robot_observation)
    target_position, target_rotation = _extract_command_pose(robot_command)
    position_delta = target_position - current_position
    rotation_delta = (current_rotation.inv() * target_rotation).as_rotvec()
    return position_delta, rotation_delta


def smooth_robot_command_ema(
    robot_command: dict[str, float],
    previous_command: dict[str, float] | None,
    *,
    alpha: float | None,
) -> dict[str, float]:
    if alpha is None:
        return dict(robot_command)
    alpha = float(alpha)
    if not 0.0 < alpha <= 1.0:
        raise ValueError('--command-ema-alpha must be in (0, 1] when provided.')
    if previous_command is None or alpha >= 1.0:
        return dict(robot_command)

    previous_position, previous_rotation = _extract_command_pose(previous_command)
    target_position, target_rotation = _extract_command_pose(robot_command)
    smoothed_position = previous_position + alpha * (target_position - previous_position)
    relative_rotation = previous_rotation.inv() * target_rotation
    smoothed_rotation = previous_rotation * Rotation.from_rotvec(alpha * relative_rotation.as_rotvec())
    smoothed_rotvec = smoothed_rotation.as_rotvec()

    smoothed_command = dict(robot_command)
    smoothed_command.update(
        {
            'ee.x': float(smoothed_position[0]),
            'ee.y': float(smoothed_position[1]),
            'ee.z': float(smoothed_position[2]),
            'ee.wx': float(smoothed_rotvec[0]),
            'ee.wy': float(smoothed_rotvec[1]),
            'ee.wz': float(smoothed_rotvec[2]),
        }
    )
    return smoothed_command


def _extract_previous_command_pose(
    robot_observation: RobotObservation,
) -> tuple[np.ndarray, Rotation] | None:
    """The pose the policy's delta is defined against, or None if the robot does not report it.

    This is the driver's own last sent command -- the same field the recorder wrote into
    ``observation.state.prev_cmd.*`` and the same one ``DeltaEEToAbsoluteEEAction`` rebuilds the
    absolute target from. Reading it here is what lets the step guard judge the policy by the
    quantity the policy actually produced, rather than by that quantity plus the servo lag.
    """
    keys = PREV_CMD_POSITION_KEYS + PREV_CMD_ROTVEC_KEYS
    if not all(key in robot_observation for key in keys):
        return None
    position = np.asarray([robot_observation[key] for key in PREV_CMD_POSITION_KEYS], dtype=np.float64)
    rotation = Rotation.from_rotvec([robot_observation[key] for key in PREV_CMD_ROTVEC_KEYS])
    return position, rotation


def observation_with_prev_cmd(
    robot_observation: RobotObservation,
    previous_sent_command: dict[str, float] | None,
) -> RobotObservation:
    """The observation with this loop's own last command written into the prev_cmd fields.

    The hardware robot reports ``prev_cmd.ee.*`` and the step guard measures against it. A
    backend that does not -- the simulated arm, for one -- makes the guard fall back to the
    *measured* pose, which bounds a different quantity: measured-pose deltas carry servo lag, so
    the same policy would be clamped differently in simulation than on hardware and a rehearsal
    would rehearse a guard nobody runs.

    Nothing is invented here. The value written is the caller's last sent command, which is
    exactly what the field means on the real robot.
    """
    if previous_sent_command is None:
        return robot_observation
    enriched = dict(robot_observation)
    for key, source_key in zip(PREV_CMD_POSITION_KEYS, ('ee.x', 'ee.y', 'ee.z'), strict=True):
        enriched[key] = float(previous_sent_command[source_key])
    for key, source_key in zip(PREV_CMD_ROTVEC_KEYS, ('ee.wx', 'ee.wy', 'ee.wz'), strict=True):
        enriched[key] = float(previous_sent_command[source_key])
    return enriched


def _shorten_to_limit(delta: np.ndarray, limit: float) -> tuple[np.ndarray, bool]:
    """Shorten an over-long delta without turning it.

    Per-axis ``np.clip`` silently changes the commanded *direction*: a mostly-downward reach of
    (0.2, 2.5, -4.9) mm clips to (0.2, 2.5, -3.0) mm, which points somewhere the policy never
    asked to go. Worse, it bends hardest along whichever axis is carrying the motion, so a descent
    gets throttled while the lateral drift rides through untouched -- on the 299/299 run, 77% of
    all discarded displacement was the z component. Scaling the whole vector keeps the heading and
    only reduces the distance.
    """
    magnitude = float(np.linalg.norm(delta))
    if magnitude <= limit or magnitude == 0.0:
        return delta, False
    return delta * (limit / magnitude), True


def limit_command_for_safety(
    robot_command: dict[str, float],
    robot_observation: RobotObservation,
    *,
    max_step_pos_delta_m: float,
    max_step_rot_delta_rad: float,
    max_leash_pos_delta_m: float,
    max_leash_rot_delta_rad: float,
) -> tuple[dict[str, float], dict[str, Any]]:
    """Bound the command twice, against the two references that mean different things.

    **Step** -- how much motion the policy asked for, measured from ``prev_cmd``, the pose its
    delta is defined against. This is the guard on policy aggression, and the training data sizes
    it directly.

    **Leash** -- how far the resulting command sits from where the arm actually is. That gap is
    tracking lag, not intent; it is large and healthy whenever the arm is moving, and it only
    means trouble when the arm has stopped tracking altogether.

    Collapsing the two -- judging intent by the gap -- is what made a well-behaved rollout report
    every one of its 299 steps as clamped while the arm was in fact following the policy closely.
    """
    current_position, current_rotation = _extract_observation_pose(robot_observation)
    target_position, target_rotation = _extract_command_pose(robot_command)

    # Reported unchanged for the log, so ``pos_delta_mm`` keeps meaning what it always meant.
    raw_gap_position_delta = target_position - current_position
    raw_gap_rotation_delta = (current_rotation.inv() * target_rotation).as_rotvec()

    reference = _extract_previous_command_pose(robot_observation)
    if reference is None:
        # No prev_cmd in the observation: fall back to the measured pose. The step guard then
        # degrades to the old behaviour rather than silently vanishing.
        reference_position, reference_rotation = current_position, current_rotation
    else:
        reference_position, reference_rotation = reference

    step_position_delta = target_position - reference_position
    step_rotation_delta = (reference_rotation.inv() * target_rotation).as_rotvec()
    limited_step_position, step_position_limited = _shorten_to_limit(
        step_position_delta, float(max_step_pos_delta_m)
    )
    limited_step_rotation, step_rotation_limited = _shorten_to_limit(
        step_rotation_delta, float(max_step_rot_delta_rad)
    )
    target_position = reference_position + limited_step_position
    target_rotation = reference_rotation * Rotation.from_rotvec(limited_step_rotation)

    gap_position_delta = target_position - current_position
    gap_rotation_delta = (current_rotation.inv() * target_rotation).as_rotvec()
    limited_gap_position, leash_position_limited = _shorten_to_limit(
        gap_position_delta, float(max_leash_pos_delta_m)
    )
    limited_gap_rotation, leash_rotation_limited = _shorten_to_limit(
        gap_rotation_delta, float(max_leash_rot_delta_rad)
    )

    safe_position = current_position + limited_gap_position
    safe_rotation = current_rotation * Rotation.from_rotvec(limited_gap_rotation)
    safe_rotvec = safe_rotation.as_rotvec()
    safe_command = dict(robot_command)
    safe_command.update(
        {
            'ee.x': float(safe_position[0]),
            'ee.y': float(safe_position[1]),
            'ee.z': float(safe_position[2]),
            'ee.wx': float(safe_rotvec[0]),
            'ee.wy': float(safe_rotvec[1]),
            'ee.wz': float(safe_rotvec[2]),
        }
    )

    step_limited = bool(step_position_limited or step_rotation_limited)
    leash_limited = bool(leash_position_limited or leash_rotation_limited)
    guard: dict[str, Any] = {
        'position_delta': raw_gap_position_delta,
        'rotation_delta': raw_gap_rotation_delta,
        'step_position_delta': step_position_delta,
        'step_rotation_delta': step_rotation_delta,
        'step_limited': step_limited,
        'leash_limited': leash_limited,
        'has_prev_cmd_reference': reference is not None,
        # The leash firing is the one worth reacting to: it means the command is running away
        # from an arm that is not following it.
        'status': 'leash_limited' if leash_limited else ('step_limited' if step_limited else 'pass'),
    }
    return safe_command, guard
