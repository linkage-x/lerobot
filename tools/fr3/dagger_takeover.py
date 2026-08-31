"""Handing the arm to the operator mid-rollout, and marking what they did with it.

DAgger's whole value is that the correction happens on *the state the policy walked itself
into*. Re-recording demonstrations cannot reach those states: there is no frame in the dataset
of "the peg has already been knocked over", because a demonstrator never gets there. So the
correction has to be applied online, inside the same loop that is running the policy, and the
segment it produced has to come back labelled.

Two design decisions carry the rest of this file.

**One reference for both action sources.** The policy's action is a delta against ``prev_cmd``
-- the pose the last command asked for, not the pose the arm reached. The expert's delta is
defined against exactly the same thing, so the step guard in ``limit_command_for_safety``
bounds the two sources identically and the handoff in either direction is continuous: an
operator who has not yet moved the SpaceMouse re-issues the command that was already in
flight. Anchoring the expert to the *measured* pose instead would make taking over a step
backwards of the size of the tracking lag, which during fast motion is tens of millimetres --
a jolt at the exact moment the operator reached for the control because something was already
going wrong.

Integrating against the last *sent* command also removes windup for free. A reference that
integrated the expert's raw target would keep running while the clamp shortened what was
actually sent, and the operator -- who is closing the loop with their eyes -- would push
harder against a target that had already left the arm behind.

**Taking over does not by itself move the gripper.** The SpaceMouse reports an absolute
gripper position, so passing it through on the first engaged step would snap the gripper to
wherever the operator's buttons last left it. The failure this exists to correct happens
within a few centimetres of the object, often with something already between the fingers.
So the commanded gripper is held at whatever the policy last asked for until the operator
actually presses a button, and only then does the device take over that axis too.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from lerobot.utils.rotation import Rotation

EE_POSITION_KEYS = ('ee.x', 'ee.y', 'ee.z')
EE_ROTVEC_KEYS = ('ee.wx', 'ee.wy', 'ee.wz')
GRIPPER_KEY = 'gripper.pos'

# How far the device's gripper reading must move from what it read when the operator engaged
# before the device owns that axis. Below this it is the same button state, not a press: the
# incremental tool mode drifts by `incremental_step` per poll while a button is held, and the
# EMA in the teleoperator means even a released button settles rather than snaps.
_GRIPPER_TOUCH_EPS = 0.02


def _pose_from_command(command: Mapping[str, float]) -> tuple[np.ndarray, Rotation]:
    position = np.array([float(command[key]) for key in EE_POSITION_KEYS], dtype=np.float64)
    rotation = Rotation.from_rotvec(np.array([float(command[key]) for key in EE_ROTVEC_KEYS], dtype=np.float64))
    return position, rotation


def _pose_from_observation(observation: Mapping[str, float]) -> tuple[np.ndarray, Rotation]:
    position = np.array([float(observation[key]) for key in EE_POSITION_KEYS], dtype=np.float64)
    rotation = Rotation.from_rotvec(np.array([float(observation[key]) for key in EE_ROTVEC_KEYS], dtype=np.float64))
    return position, rotation


def expert_spans(sources: list[str], *, expert_label: str = 'expert') -> list[tuple[int, int]]:
    """The inclusive `(first, last)` step of each stretch the operator was driving.

    Reduced from the per-step source column rather than recorded separately while it happens.
    The trace file is the thing that survives the rollout, so a span the marker reports and the
    CSV does not is a span that cannot be checked afterwards -- and two records of one event is
    two things that can disagree.
    """
    spans: list[tuple[int, int]] = []
    start: int | None = None
    for index, source in enumerate(sources):
        if source == expert_label:
            if start is None:
                start = index
        elif start is not None:
            spans.append((start, index - 1))
            start = None
    if start is not None:
        spans.append((start, len(sources) - 1))
    return spans


class ExpertTakeover:
    """Turns SpaceMouse motion into the same command shape the policy produces.

    Holds one piece of state only -- whether the operator has touched the gripper since they
    engaged. Everything else is derived from the command that was last sent, which the caller
    already tracks for the step guard.
    """

    def __init__(self, teleop: Any):
        self._teleop = teleop
        self._engaged = False
        self._gripper_hold: float = 0.0
        self._gripper_at_engage: float | None = None
        self._gripper_owned = False
        self._poll_failed = False

    @property
    def engaged(self) -> bool:
        return self._engaged

    def _engage(self, *, previous_sent_command: Mapping[str, float] | None, robot_observation: Mapping[str, float]) -> None:
        self._engaged = True
        self._gripper_owned = False
        self._gripper_at_engage = None
        self._poll_failed = False
        if previous_sent_command is not None and GRIPPER_KEY in previous_sent_command:
            self._gripper_hold = float(previous_sent_command[GRIPPER_KEY])
        else:
            self._gripper_hold = float(robot_observation.get(GRIPPER_KEY, 0.0))
        print(f'[INFO] dagger_takeover=engaged gripper_hold={self._gripper_hold:.3f}')

    def _release(self) -> None:
        self._engaged = False
        self._gripper_owned = False
        self._gripper_at_engage = None
        print('[INFO] dagger_takeover=released')

    def command(
        self,
        *,
        engaged: bool,
        policy_command: dict[str, float],
        previous_sent_command: Mapping[str, float] | None,
        robot_observation: Mapping[str, float],
    ) -> tuple[dict[str, float], dict[str, Any]]:
        """The command for this step, plus what it was and why.

        Returns the policy's own command untouched whenever the operator is not driving, so a
        run with takeover available but never used is byte-identical to one without it.
        """
        if engaged and not self._engaged:
            self._engage(previous_sent_command=previous_sent_command, robot_observation=robot_observation)
        elif not engaged and self._engaged:
            self._release()

        if not self._engaged:
            return policy_command, {'source': 'policy', 'engaged': False}

        # The arm holds where the last command put it. Used both as the delta reference and as
        # the fallback when the device cannot be read: an operator who asked for control and
        # got a policy command back instead is the one surprise worth ruling out.
        if previous_sent_command is not None:
            reference_position, reference_rotation = _pose_from_command(previous_sent_command)
        else:
            reference_position, reference_rotation = _pose_from_observation(robot_observation)

        try:
            action = self._teleop.get_action()
        except Exception as exc:  # noqa: BLE001 - a device read must never end a rollout mid-motion
            if not self._poll_failed:
                self._poll_failed = True
                print(f'[WARN] dagger_takeover_poll_failed holding_position: {exc}')
            held = dict(policy_command)
            held.update(
                {
                    'ee.x': float(reference_position[0]),
                    'ee.y': float(reference_position[1]),
                    'ee.z': float(reference_position[2]),
                }
            )
            rotvec = reference_rotation.as_rotvec()
            held.update({'ee.wx': float(rotvec[0]), 'ee.wy': float(rotvec[1]), 'ee.wz': float(rotvec[2])})
            held[GRIPPER_KEY] = float(self._gripper_hold)
            return held, {'source': 'expert', 'engaged': True, 'status': 'poll_failed'}
        self._poll_failed = False

        delta_position = np.array(
            [float(action.get('target_x', 0.0)), float(action.get('target_y', 0.0)), float(action.get('target_z', 0.0))],
            dtype=np.float64,
        )
        delta_rotvec = np.array(
            [float(action.get('target_wx', 0.0)), float(action.get('target_wy', 0.0)), float(action.get('target_wz', 0.0))],
            dtype=np.float64,
        )
        if not np.all(np.isfinite(delta_position)) or not np.all(np.isfinite(delta_rotvec)):
            delta_position = np.zeros(3, dtype=np.float64)
            delta_rotvec = np.zeros(3, dtype=np.float64)

        target_position = reference_position + delta_position
        # Right-multiplied, matching the recorder's own delta convention
        # (processor_franka_research3: `desired_R = reference_R @ delta_R`). A left-multiplied
        # rotation here would mean the SpaceMouse turned the tool about the base axes during
        # takeover and about the tool axes during recording -- the same device, two behaviours.
        target_rotation = reference_rotation * Rotation.from_rotvec(delta_rotvec)
        target_rotvec = target_rotation.as_rotvec()

        gripper_reading = action.get('gripper')
        if gripper_reading is None or not np.isfinite(float(gripper_reading)):
            gripper_command = float(self._gripper_hold)
        else:
            gripper_reading = float(np.clip(float(gripper_reading), 0.0, 1.0))
            if self._gripper_at_engage is None:
                self._gripper_at_engage = gripper_reading
            if not self._gripper_owned and abs(gripper_reading - self._gripper_at_engage) >= _GRIPPER_TOUCH_EPS:
                self._gripper_owned = True
                print(f'[INFO] dagger_takeover=gripper_taken value={gripper_reading:.3f}')
            gripper_command = gripper_reading if self._gripper_owned else float(self._gripper_hold)

        expert_command = dict(policy_command)
        expert_command.update(
            {
                'ee.x': float(target_position[0]),
                'ee.y': float(target_position[1]),
                'ee.z': float(target_position[2]),
                'ee.wx': float(target_rotvec[0]),
                'ee.wy': float(target_rotvec[1]),
                'ee.wz': float(target_rotvec[2]),
                GRIPPER_KEY: float(gripper_command),
            }
        )
        moved = bool(np.any(delta_position != 0.0) or np.any(delta_rotvec != 0.0))
        return expert_command, {
            'source': 'expert',
            'engaged': True,
            'status': 'moving' if moved else 'holding',
            'moved': moved,
            'gripper_owned': self._gripper_owned,
            'step_mm': float(np.linalg.norm(delta_position) * 1000.0),
        }

    def close(self) -> None:
        """Release the device. Never raises: this runs in the same `finally` that disconnects
        the robot, and a teleoperator that will not let go must not stop the arm from being
        put down safely."""
        self._engaged = False
        try:
            self._teleop.disconnect()
        except Exception as exc:  # noqa: BLE001
            print(f'[WARN] dagger_takeover_disconnect_failed: {exc}')
