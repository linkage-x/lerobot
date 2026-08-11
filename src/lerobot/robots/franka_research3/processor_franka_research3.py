#!/usr/bin/env python

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field

import numpy as np

from lerobot.configs.types import FeatureType, PipelineFeatureType, PolicyFeature
from lerobot.processor import ObservationProcessorStep, RobotAction, RobotActionProcessorStep, RobotObservation
from lerobot.processor.core import TransitionKey
from lerobot.utils.rotation import Rotation

EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
EE_ROTVEC_KEYS = ("ee.wx", "ee.wy", "ee.wz")
EE_QUAT_KEYS = ("ee.qx", "ee.qy", "ee.qz", "ee.qw")
PREV_CMD_POSITION_KEYS = ("prev_cmd.ee.x", "prev_cmd.ee.y", "prev_cmd.ee.z")
PREV_CMD_ROTVEC_KEYS = ("prev_cmd.ee.wx", "prev_cmd.ee.wy", "prev_cmd.ee.wz")
PREV_CMD_QUAT_KEYS = ("prev_cmd.ee.qx", "prev_cmd.ee.qy", "prev_cmd.ee.qz", "prev_cmd.ee.qw")
PREV_CMD_GRIPPER_KEY = "prev_cmd.gripper.pos"


def _canonicalize_quaternion(quaternion_xyzw: np.ndarray) -> np.ndarray:
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float64)
    dominant_component_index = int(np.argmax(np.abs(quaternion_xyzw)))
    if float(quaternion_xyzw[dominant_component_index]) < 0.0:
        quaternion_xyzw = -quaternion_xyzw
    return quaternion_xyzw


def _continuous_quaternion(quaternion_xyzw: np.ndarray, previous_quaternion_xyzw: np.ndarray | None) -> np.ndarray:
    quaternion_xyzw = np.asarray(quaternion_xyzw, dtype=np.float64)
    if previous_quaternion_xyzw is None:
        return _canonicalize_quaternion(quaternion_xyzw)
    if float(np.dot(quaternion_xyzw, previous_quaternion_xyzw)) < 0.0:
        quaternion_xyzw = -quaternion_xyzw
    return quaternion_xyzw


def _pose_from_position_and_rotvec(position_xyz: np.ndarray, rotvec_xyz: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    pose[:3, :3] = Rotation.from_rotvec(rotvec_xyz).as_matrix()
    return pose


def _pose_from_position_and_quaternion(position_xyz: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = np.asarray(position_xyz, dtype=np.float64)
    pose[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    return pose


@dataclass
class KeepAbsoluteEEObservation(ObservationProcessorStep):
    """Keep only absolute EE pose, gripper position, and camera observations using quaternions."""

    _prev_quaternion_xyzw: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_prev_cmd_quaternion_xyzw: np.ndarray | None = field(default=None, init=False, repr=False)

    def _extract_quaternion(
        self,
        observation: RobotObservation,
        *,
        rotvec_keys: tuple[str, str, str],
        quat_keys: tuple[str, str, str, str],
    ) -> np.ndarray | None:
        if all(key in observation for key in rotvec_keys):
            return Rotation.from_rotvec([observation[key] for key in rotvec_keys]).as_quat()
        if all(key in observation for key in quat_keys):
            return np.array([observation[key] for key in quat_keys], dtype=np.float64)
        return None

    def observation(self, observation: RobotObservation) -> RobotObservation:
        quaternion_xyzw = self._extract_quaternion(
            observation,
            rotvec_keys=EE_ROTVEC_KEYS,
            quat_keys=EE_QUAT_KEYS,
        )
        prev_cmd_quaternion_xyzw = self._extract_quaternion(
            observation,
            rotvec_keys=PREV_CMD_ROTVEC_KEYS,
            quat_keys=PREV_CMD_QUAT_KEYS,
        )

        ignored_keys = set(EE_POSITION_KEYS + EE_ROTVEC_KEYS + EE_QUAT_KEYS + ("gripper.pos",))
        ignored_keys.update(PREV_CMD_POSITION_KEYS + PREV_CMD_ROTVEC_KEYS + PREV_CMD_QUAT_KEYS + (PREV_CMD_GRIPPER_KEY,))

        passthrough_observation = {
            key: value
            for key, value in observation.items()
            if key not in ignored_keys and not key.endswith(".pos")
        }

        processed_observation: RobotObservation = {
            **{key: observation[key] for key in EE_POSITION_KEYS if key in observation},
            "gripper.pos": observation["gripper.pos"],
            **passthrough_observation,
        }

        if quaternion_xyzw is not None:
            quaternion_xyzw = _continuous_quaternion(quaternion_xyzw, self._prev_quaternion_xyzw)
            self._prev_quaternion_xyzw = quaternion_xyzw.copy()
            processed_observation.update(
                {key: float(value) for key, value in zip(EE_QUAT_KEYS, quaternion_xyzw, strict=True)}
            )

        if all(key in observation for key in PREV_CMD_POSITION_KEYS):
            processed_observation.update({key: observation[key] for key in PREV_CMD_POSITION_KEYS})
        if PREV_CMD_GRIPPER_KEY in observation:
            processed_observation[PREV_CMD_GRIPPER_KEY] = observation[PREV_CMD_GRIPPER_KEY]
        if prev_cmd_quaternion_xyzw is not None:
            prev_cmd_quaternion_xyzw = _continuous_quaternion(
                prev_cmd_quaternion_xyzw,
                self._prev_prev_cmd_quaternion_xyzw,
            )
            self._prev_prev_cmd_quaternion_xyzw = prev_cmd_quaternion_xyzw.copy()
            processed_observation.update(
                {key: float(value) for key, value in zip(PREV_CMD_QUAT_KEYS, prev_cmd_quaternion_xyzw, strict=True)}
            )

        return processed_observation

    def reset(self) -> None:
        self._prev_quaternion_xyzw = None
        self._prev_prev_cmd_quaternion_xyzw = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        observation_features = features[PipelineFeatureType.OBSERVATION]
        for key in list(observation_features):
            if (
                key.endswith(".pos")
                and key not in {"gripper.pos", PREV_CMD_GRIPPER_KEY}
                and key not in EE_POSITION_KEYS
                and key not in PREV_CMD_POSITION_KEYS
            ):
                observation_features.pop(key, None)
        for key in EE_ROTVEC_KEYS + PREV_CMD_ROTVEC_KEYS:
            observation_features.pop(key, None)
        gripper_feature = observation_features.pop("gripper.pos", None)
        prev_cmd_gripper_feature = observation_features.pop(PREV_CMD_GRIPPER_KEY, None)
        for key in EE_QUAT_KEYS:
            observation_features[key] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
        for key in PREV_CMD_QUAT_KEYS:
            observation_features[key] = PolicyFeature(type=FeatureType.STATE, shape=(1,))
        if gripper_feature is not None:
            observation_features["gripper.pos"] = gripper_feature
        if prev_cmd_gripper_feature is not None:
            observation_features[PREV_CMD_GRIPPER_KEY] = prev_cmd_gripper_feature
        return features


@dataclass
class DeltaActionToAbsoluteEEAction(RobotActionProcessorStep):
    """Convert FR3 delta teleop actions into absolute EE action targets using quaternions."""

    workspace_min: tuple[float, float, float]
    workspace_max: tuple[float, float, float]
    max_target_delta_pos: tuple[float, float, float] | None = None
    max_target_delta_rot: tuple[float, float, float] | None = None

    _reference_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _last_command_pose: np.ndarray | None = field(default=None, init=False, repr=False)
    _prev_enabled: bool = field(default=False, init=False, repr=False)
    _prev_output_quaternion_xyzw: np.ndarray | None = field(default=None, init=False, repr=False)

    def _pose_from_observation(self, observation: RobotObservation) -> np.ndarray:
        return _pose_from_position_and_rotvec(
            position_xyz=np.array([observation[key] for key in EE_POSITION_KEYS], dtype=np.float64),
            rotvec_xyz=np.array([observation[key] for key in EE_ROTVEC_KEYS], dtype=np.float64),
        )

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required to map FR3 delta teleop commands to absolute EE actions.")

        current_pose = self._pose_from_observation(observation)
        enabled = bool(action.pop("enabled"))
        gripper = float(np.clip(action.pop("gripper"), 0.0, 1.0))
        raw_target_x = float(action.pop("target_x"))
        raw_target_y = float(action.pop("target_y"))
        raw_target_z = float(action.pop("target_z"))
        raw_target_wx = float(action.pop("target_wx"))
        raw_target_wy = float(action.pop("target_wy"))
        raw_target_wz = float(action.pop("target_wz"))

        if enabled:
            if not self._prev_enabled or self._reference_pose is None:
                self._reference_pose = current_pose.copy()

            delta_pos = np.array(
                [raw_target_x, raw_target_y, raw_target_z],
                dtype=np.float64,
            )
            if self.max_target_delta_pos is not None:
                delta_pos = np.clip(
                    delta_pos,
                    -np.asarray(self.max_target_delta_pos, dtype=np.float64),
                    np.asarray(self.max_target_delta_pos, dtype=np.float64),
                )

            delta_rotvec = np.array(
                [raw_target_wx, raw_target_wy, raw_target_wz],
                dtype=np.float64,
            )
            if self.max_target_delta_rot is not None:
                delta_rotvec = np.clip(
                    delta_rotvec,
                    -np.asarray(self.max_target_delta_rot, dtype=np.float64),
                    np.asarray(self.max_target_delta_rot, dtype=np.float64),
                )
            delta_rot = Rotation.from_rotvec(delta_rotvec)

            desired_pose = np.eye(4, dtype=np.float64)
            desired_pose[:3, :3] = self._reference_pose[:3, :3] @ delta_rot.as_matrix()
            desired_pose[:3, 3] = self._reference_pose[:3, 3] + delta_pos
            desired_pose[:3, 3] = np.clip(
                desired_pose[:3, 3],
                np.asarray(self.workspace_min, dtype=np.float64),
                np.asarray(self.workspace_max, dtype=np.float64),
            )
            self._last_command_pose = desired_pose.copy()
            self._reference_pose = desired_pose.copy()
        else:
            desired_pose = self._last_command_pose.copy() if self._last_command_pose is not None else current_pose.copy()
            self._reference_pose = None

        desired_quaternion_xyzw = Rotation.from_matrix(desired_pose[:3, :3]).as_quat()
        desired_quaternion_xyzw = _continuous_quaternion(desired_quaternion_xyzw, self._prev_output_quaternion_xyzw)
        self._prev_output_quaternion_xyzw = desired_quaternion_xyzw.copy()
        self._prev_enabled = enabled
        return {
            "enabled": enabled,
            "target_x": raw_target_x,
            "target_y": raw_target_y,
            "target_z": raw_target_z,
            "target_wx": raw_target_wx,
            "target_wy": raw_target_wy,
            "target_wz": raw_target_wz,
            "gripper": gripper,
            "ee.x": float(desired_pose[0, 3]),
            "ee.y": float(desired_pose[1, 3]),
            "ee.z": float(desired_pose[2, 3]),
            "ee.qx": float(desired_quaternion_xyzw[0]),
            "ee.qy": float(desired_quaternion_xyzw[1]),
            "ee.qz": float(desired_quaternion_xyzw[2]),
            "ee.qw": float(desired_quaternion_xyzw[3]),
            "gripper.pos": gripper,
        }

    def reset(self) -> None:
        self._reference_pose = None
        self._last_command_pose = None
        self._prev_enabled = False
        self._prev_output_quaternion_xyzw = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        for key in ("enabled", "target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz", "gripper"):
            action_features.pop(key, None)

        for key in ("ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"):
            action_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features


# Which pose the delta is measured against. Both are exactly invertible, but they teach a
# policy different things and require different deployment code, so the choice is explicit.
DELTA_REFERENCE_PREV_CMD = "prev_cmd"
DELTA_REFERENCE_CURRENT = "current"
DELTA_REFERENCES = (DELTA_REFERENCE_PREV_CMD, DELTA_REFERENCE_CURRENT)
# The reference is part of the *feature name*, not just a config value. Two datasets whose deltas
# are measured against different poses need different absolute trajectories to be rebuilt from
# them, so sharing one name would make a dataset un-self-describing and let an offline tool
# integrate the wrong way while producing a plausible-looking result.
_DELTA_EE_KEY_ROOTS = {
    DELTA_REFERENCE_PREV_CMD: "delta_ee_from_prev_cmd",
    DELTA_REFERENCE_CURRENT: "delta_ee_from_current",
}
# Gripper stays absolute in every mode: 0..1 is an opening, and a delta on it would accumulate
# drift with no reference in the observation to correct against.
DELTA_EE_GRIPPER_KEY = "gripper.pos"


def delta_ee_key_root(reference: str) -> str:
    if reference not in _DELTA_EE_KEY_ROOTS:
        raise ValueError(f"delta reference must be one of {DELTA_REFERENCES}, got {reference!r}")
    return _DELTA_EE_KEY_ROOTS[reference]


def delta_ee_position_keys(reference: str) -> tuple[str, str, str]:
    root = delta_ee_key_root(reference)
    return (f"{root}.dx", f"{root}.dy", f"{root}.dz")


def delta_ee_rotvec_keys(reference: str) -> tuple[str, str, str]:
    root = delta_ee_key_root(reference)
    return (f"{root}.drx", f"{root}.dry", f"{root}.drz")


def delta_ee_action_keys(reference: str) -> tuple[str, ...]:
    return (
        *delta_ee_position_keys(reference),
        *delta_ee_rotvec_keys(reference),
        DELTA_EE_GRIPPER_KEY,
    )


def delta_reference_from_action_names(action_names: Iterable[str]) -> str | None:
    """Recover which delta reference a recorded dataset used, from its action feature names."""
    names = set(action_names)
    for reference in DELTA_REFERENCES:
        if set(delta_ee_position_keys(reference)) <= names:
            return reference
    return None
# A delta this large is not a teleop increment; it means the reference pose was wrong (e.g. a
# left/right multiplication mix-up, or a reference taken from the wrong frame). rotvec is
# well-conditioned far below this, and near pi it would alias -- so refuse instead of aliasing.
_MAX_SANE_DELTA_ROTATION_RAD = np.pi / 2.0


def _reference_pose_from_observation(observation: RobotObservation, reference: str) -> np.ndarray:
    """Pose the recorded delta is relative to, read from the raw robot observation.

    Both references are present in every FR3 observation, so neither mode needs the recorder to
    carry extra state -- and at deployment the policy's delta can be turned back into an
    absolute target from the same observation it was conditioned on.
    """
    if reference == DELTA_REFERENCE_PREV_CMD:
        position_keys, rotvec_keys, quat_keys = (
            PREV_CMD_POSITION_KEYS,
            PREV_CMD_ROTVEC_KEYS,
            PREV_CMD_QUAT_KEYS,
        )
    elif reference == DELTA_REFERENCE_CURRENT:
        position_keys, rotvec_keys, quat_keys = EE_POSITION_KEYS, EE_ROTVEC_KEYS, EE_QUAT_KEYS
    else:
        raise ValueError(f"delta reference must be one of {DELTA_REFERENCES}, got {reference!r}")

    if not all(key in observation for key in position_keys):
        raise ValueError(
            f"Observation is missing {position_keys} needed for delta reference {reference!r}."
        )
    position_xyz = np.array([observation[key] for key in position_keys], dtype=np.float64)
    if all(key in observation for key in rotvec_keys):
        return _pose_from_position_and_rotvec(
            position_xyz=position_xyz,
            rotvec_xyz=np.array([observation[key] for key in rotvec_keys], dtype=np.float64),
        )
    if all(key in observation for key in quat_keys):
        return _pose_from_position_and_quaternion(
            position_xyz=position_xyz,
            quaternion_xyzw=np.array([observation[key] for key in quat_keys], dtype=np.float64),
        )
    raise ValueError(
        f"Observation carries neither {rotvec_keys} nor {quat_keys} for delta reference {reference!r}."
    )


@dataclass
class AbsoluteEEToDeltaEEAction(RobotActionProcessorStep):
    """Re-express an absolute EE target as a delta against the reference pose, for recording.

    Runs after :class:`DeltaActionToAbsoluteEEAction`, so the delta describes the command that
    was actually issued -- including the workspace clip -- rather than the raw teleop increment.

    Conventions, which the reconstruction must mirror exactly:

    * **translation is world-frame** (``delta = desired_pos - reference_pos``)
    * **rotation is body/tool-frame and right-multiplied**
      (``delta_R = reference_R^T @ desired_R``, so ``desired_R = reference_R @ delta_R``)

    That mixed convention is not a preference; it is what
    :class:`DeltaActionToAbsoluteEEAction` already does when it builds the target, so anything
    else would make the recorded delta disagree with the clamp that produced it.

    Rotation is stored as a **rotvec**, not a quaternion. Deltas are clamped to ~0.01 rad/frame,
    where a quaternion's ``qw = cos(theta/2)`` spans only ~1.25e-5 -- about 8 bits of float32 --
    and recovering the angle through ``acos`` near 1 amplifies regression error by ~80x. rotvec
    is 1:1 in that range, carries no unit-norm constraint for a regressor to violate, and can be
    linearly averaged for action chunking. The pi singularity that makes quaternions necessary
    for *absolute* pose is two orders of magnitude away from a clamped delta.
    """

    reference: str = DELTA_REFERENCE_PREV_CMD

    def __post_init__(self) -> None:
        if self.reference not in DELTA_REFERENCES:
            raise ValueError(f"reference must be one of {DELTA_REFERENCES}, got {self.reference!r}")

    def action(self, action: RobotAction) -> RobotAction:
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required to express an FR3 EE target as a delta.")
        if not all(key in action for key in EE_POSITION_KEYS + EE_QUAT_KEYS):
            raise ValueError(
                "AbsoluteEEToDeltaEEAction expects an absolute EE action "
                f"({EE_POSITION_KEYS + EE_QUAT_KEYS}); got keys {sorted(action)}."
            )

        desired_pose = _pose_from_position_and_quaternion(
            position_xyz=np.array([action[key] for key in EE_POSITION_KEYS], dtype=np.float64),
            quaternion_xyzw=np.array([action[key] for key in EE_QUAT_KEYS], dtype=np.float64),
        )
        reference_pose = _reference_pose_from_observation(observation, self.reference)

        delta_position = desired_pose[:3, 3] - reference_pose[:3, 3]
        delta_rotation = Rotation.from_matrix(reference_pose[:3, :3].T @ desired_pose[:3, :3])
        delta_rotvec = delta_rotation.as_rotvec()
        delta_angle_rad = float(np.linalg.norm(delta_rotvec))
        if delta_angle_rad > _MAX_SANE_DELTA_ROTATION_RAD:
            raise ValueError(
                f"FR3 delta rotation {np.degrees(delta_angle_rad):.1f} deg exceeds "
                f"{np.degrees(_MAX_SANE_DELTA_ROTATION_RAD):.1f} deg against the "
                f"{self.reference!r} reference; the reference pose or the multiplication order "
                "is wrong, and a rotvec this large would alias near pi."
            )

        gripper = float(np.clip(action.get("gripper.pos", action.get("gripper", 0.0)), 0.0, 1.0))
        # Everything the downstream reconstruction and the robot's hold path need is passed
        # through; only delta_ee_action_keys(reference) are declared as dataset features.
        return {
            **{key: bool(action["enabled"]) for key in ("enabled",) if "enabled" in action},
            **{
                key: float(action[key])
                for key in ("target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz")
                if key in action
            },
            **dict(
                zip(delta_ee_position_keys(self.reference), (float(v) for v in delta_position), strict=True)
            ),
            **dict(
                zip(delta_ee_rotvec_keys(self.reference), (float(v) for v in delta_rotvec), strict=True)
            ),
            "gripper": gripper,
            "gripper.pos": gripper,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        action_features = features[PipelineFeatureType.ACTION]
        # Drop the absolute-EE keys the previous step declared: a delta dataset must not also
        # advertise an absolute action, or a policy could be trained against the wrong one.
        for key in (
            "enabled",
            "target_x",
            "target_y",
            "target_z",
            "target_wx",
            "target_wy",
            "target_wz",
            "gripper",
            *EE_POSITION_KEYS,
            *EE_QUAT_KEYS,
            # Popped and re-added last so the ordering matches the absolute contract
            # (pose components first, gripper last) instead of depending on insertion history.
            DELTA_EE_GRIPPER_KEY,
        ):
            action_features.pop(key, None)
        for key in delta_ee_action_keys(self.reference):
            action_features[key] = PolicyFeature(type=FeatureType.ACTION, shape=(1,))
        return features


@dataclass
class DeltaEEToAbsoluteEEAction(RobotActionProcessorStep):
    """Rebuild the absolute EE target from a delta action.

    This is the single implementation of the reconstruction, used in both directions: during
    recording it turns the delta back into the command sent to the robot, and at deployment it
    turns the policy's delta into that same command. Sharing it is what makes "training equals
    deployment" structural rather than a convention two files have to agree on.

    ``enabled`` is optional and defaults to True: a policy emits no such flag, and it does not
    need one -- a held frame is recorded as a zero delta, which reconstructs to "stay put".
    """

    reference: str = DELTA_REFERENCE_PREV_CMD
    workspace_min: tuple[float, float, float] | None = None
    workspace_max: tuple[float, float, float] | None = None

    _prev_output_quaternion_xyzw: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        if self.reference not in DELTA_REFERENCES:
            raise ValueError(f"reference must be one of {DELTA_REFERENCES}, got {self.reference!r}")

    def action(self, action: RobotAction) -> RobotAction:
        position_keys = delta_ee_position_keys(self.reference)
        rotvec_keys = delta_ee_rotvec_keys(self.reference)
        if not all(key in action for key in position_keys + rotvec_keys):
            # Already absolute (or a hold frame from an absolute pipeline): pass it through so
            # one pipeline can serve both action modes without a caller-side branch.
            return action
        observation = self.transition.get(TransitionKey.OBSERVATION)
        if observation is None:
            raise ValueError("Observation is required to rebuild an absolute EE target from a delta.")

        reference_pose = _reference_pose_from_observation(observation, self.reference)
        delta_position = np.array([float(action[key]) for key in position_keys], dtype=np.float64)
        delta_rotvec = np.array([float(action[key]) for key in rotvec_keys], dtype=np.float64)

        desired_position = reference_pose[:3, 3] + delta_position
        if self.workspace_min is not None and self.workspace_max is not None:
            # Clamped on the way out as well as on the way in. During recording this is a no-op
            # (the delta already encodes a clipped target); at deployment it is the guard that
            # keeps a policy from commanding its way out of the safe envelope.
            desired_position = np.clip(
                desired_position,
                np.asarray(self.workspace_min, dtype=np.float64),
                np.asarray(self.workspace_max, dtype=np.float64),
            )
        # Right-multiplied, body frame -- the inverse of AbsoluteEEToDeltaEEAction.
        desired_rotation = reference_pose[:3, :3] @ Rotation.from_rotvec(delta_rotvec).as_matrix()

        desired_quaternion_xyzw = Rotation.from_matrix(desired_rotation).as_quat()
        desired_quaternion_xyzw = _continuous_quaternion(
            desired_quaternion_xyzw, self._prev_output_quaternion_xyzw
        )
        self._prev_output_quaternion_xyzw = desired_quaternion_xyzw.copy()

        gripper = float(np.clip(action.get("gripper.pos", action.get("gripper", 0.0)), 0.0, 1.0))
        rebuilt: RobotAction = {
            **{key: action[key] for key in ("enabled",) if key in action},
            **{
                key: action[key]
                for key in ("target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz")
                if key in action
            },
            "ee.x": float(desired_position[0]),
            "ee.y": float(desired_position[1]),
            "ee.z": float(desired_position[2]),
            **dict(zip(EE_QUAT_KEYS, (float(v) for v in desired_quaternion_xyzw), strict=True)),
            "gripper": gripper,
            "gripper.pos": gripper,
        }
        return rebuilt

    def reset(self) -> None:
        self._prev_output_quaternion_xyzw = None

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        # Reconstruction is a robot-command step, not a dataset-schema step: the recorded action
        # stays the delta. Rewriting the features here would re-advertise the absolute action.
        return features


@dataclass
class AbsoluteEEActionToRobotAction(RobotActionProcessorStep):
    """Adapt ee2ee record actions so idle frames still use the robot's hold-current-joints path."""

    def action(self, action: RobotAction) -> RobotAction:
        if not all(key in action for key in ("ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw")):
            return action

        gripper = float(np.clip(action.get("gripper.pos", action.get("gripper", 0.0)), 0.0, 1.0))
        enabled = bool(action.get("enabled", True))
        if not enabled:
            return {
                "enabled": False,
                "target_x": float(action.get("target_x", 0.0)),
                "target_y": float(action.get("target_y", 0.0)),
                "target_z": float(action.get("target_z", 0.0)),
                "target_wx": float(action.get("target_wx", 0.0)),
                "target_wy": float(action.get("target_wy", 0.0)),
                "target_wz": float(action.get("target_wz", 0.0)),
                "gripper": gripper,
            }

        desired_pose = _pose_from_position_and_quaternion(
            position_xyz=np.array([action[key] for key in EE_POSITION_KEYS], dtype=np.float64),
            quaternion_xyzw=np.array([action[key] for key in EE_QUAT_KEYS], dtype=np.float64),
        )
        desired_rotvec = Rotation.from_matrix(desired_pose[:3, :3]).as_rotvec()

        return {
            "ee.x": float(action["ee.x"]),
            "ee.y": float(action["ee.y"]),
            "ee.z": float(action["ee.z"]),
            "ee.wx": float(desired_rotvec[0]),
            "ee.wy": float(desired_rotvec[1]),
            "ee.wz": float(desired_rotvec[2]),
            "gripper.pos": gripper,
        }

    def transform_features(
        self, features: dict[PipelineFeatureType, dict[str, PolicyFeature]]
    ) -> dict[PipelineFeatureType, dict[str, PolicyFeature]]:
        return features
