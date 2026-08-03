#!/usr/bin/env python3

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

"""Derive a delta-EE action column from a recorded absolute-EE one.

Why this is an offline transform and not something the recorder does
-------------------------------------------------------------------
A delta's magnitude is tied to the interval it spans, so it must span the interval at which the
policy will be queried -- one dataset frame. The recorder's teleop pipeline runs at
``control_fps`` (200 Hz on the workstation) while frames are captured at ``dataset.fps`` (30 Hz),
and a processor step in that pipeline cannot tell which of its ~6.7 invocations per frame is the
one that gets recorded. Computing the delta at capture time therefore stored a *one-control-tick*
increment against a per-frame grid: measured 0.5 mm recorded where the command had actually
advanced 1.0-1.5 mm, i.e. a policy trained on it would drive the arm ~6.7x too slow.

Differencing consecutive *dataset* frames offline is exact by construction, and it has three
further consequences worth stating: the capture path stays the proven absolute contract, the
delta definition can be changed or fixed without re-recording on hardware, and both references
can be derived from one recording for comparison.

Nothing is lost by not recording the delta: the absolute action plus
``observation.state.prev_cmd.ee.*`` determine it exactly, and the absolute command stream remains
the source of truth.
"""

from __future__ import annotations

from pathlib import Path
import sys
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _path in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from lerobot.processor.core import TransitionKey  # noqa: E402
from lerobot.robots.franka_research3.action_modes import (  # noqa: E402
    delta_reference_for_action_mode,
    is_delta_action_mode,
    validate_action_mode,
)
from lerobot.robots.franka_research3.processor_franka_research3 import (  # noqa: E402
    AbsoluteEEToDeltaEEAction,
    DeltaEEToAbsoluteEEAction,
    delta_ee_action_keys,
)
from lerobot.utils.rotation import Rotation  # noqa: E402

ABSOLUTE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
ABSOLUTE_QUAT_KEYS = ("ee.qx", "ee.qy", "ee.qz", "ee.qw")
PREV_CMD_POSITION_KEYS = ("prev_cmd.ee.x", "prev_cmd.ee.y", "prev_cmd.ee.z")
PREV_CMD_QUAT_KEYS = ("prev_cmd.ee.qx", "prev_cmd.ee.qy", "prev_cmd.ee.qz", "prev_cmd.ee.qw")
GRIPPER_KEY = "gripper.pos"

# The transform must be exactly invertible; these bound float32 storage round-off only, not any
# modelling slack. Anything larger means the reference or the convention is wrong.
SELF_CHECK_POSITION_TOL_M = 1e-5
SELF_CHECK_ROTATION_TOL_DEG = 1e-2


class DeltaTransformError(RuntimeError):
    """Raised when a dataset cannot be converted to a delta contract."""


def _column_index(names: list[str], key: str) -> int:
    try:
        return names.index(key)
    except ValueError as exc:
        raise DeltaTransformError(f"Column {key!r} is missing; have {names}.") from exc


def _pose_matrix(position_xyz: np.ndarray, quaternion_xyzw: np.ndarray) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, 3] = position_xyz
    pose[:3, :3] = Rotation.from_quat(quaternion_xyzw).as_matrix()
    return pose


def extract_pose_stream(
    values: np.ndarray,
    names: list[str],
    *,
    position_keys: tuple[str, ...],
    quat_keys: tuple[str, ...],
) -> np.ndarray | None:
    """(N, 4, 4) poses for the given key group, or None when the group is absent."""
    if not all(key in names for key in position_keys + quat_keys):
        return None
    positions = np.stack([values[:, _column_index(names, key)] for key in position_keys], axis=1)
    quaternions = np.stack([values[:, _column_index(names, key)] for key in quat_keys], axis=1)
    return np.stack(
        [_pose_matrix(positions[i], quaternions[i]) for i in range(values.shape[0])], axis=0
    )


def _episode_slices(episode_index: np.ndarray) -> list[slice]:
    """Contiguous row ranges per episode.

    Differencing must never cross an episode boundary: the first frame of an episode follows a
    ``move_to_start()``, so a delta spanning that seam would encode a homing move as if it were
    an operator command.
    """
    episode_index = np.asarray(episode_index).reshape(-1)
    if episode_index.size == 0:
        return []
    boundaries = np.flatnonzero(np.diff(episode_index) != 0) + 1
    starts = [0, *boundaries.tolist()]
    ends = [*boundaries.tolist(), int(episode_index.size)]
    return [slice(start, end) for start, end in zip(starts, ends, strict=True)]


def build_reference_poses(
    *,
    absolute_poses: np.ndarray,
    episode_index: np.ndarray,
    recorded_prev_cmd_poses: np.ndarray | None,
    measured_poses: np.ndarray | None,
    reference: str,
) -> np.ndarray:
    """The pose each frame's delta is measured against, (N, 4, 4).

    ``prev_cmd``: the command issued on the previous *dataset* frame -- which is precisely what
    the deployed robot reports as ``prev_cmd`` when inference runs once per frame. Each episode's
    first frame has no predecessor, so it uses the recorded ``prev_cmd`` (which the robot reports
    as the measured pose right after ``move_to_start()`` clears its last command), exactly as the
    recorder's own semantics did.

    ``current``: the measured pose on the same frame, taken from the recorded observation.
    """
    if reference == "current":
        if measured_poses is None:
            raise DeltaTransformError(
                "delta_ee_from_current needs the absolute EE pose (ee.x/y/z + ee.qx/qy/qz/qw) "
                "in observation.state."
            )
        return measured_poses

    reference_poses = np.empty_like(absolute_poses)
    for episode_slice in _episode_slices(episode_index):
        start = episode_slice.start
        if recorded_prev_cmd_poses is not None:
            reference_poses[start] = recorded_prev_cmd_poses[start]
        elif measured_poses is not None:
            reference_poses[start] = measured_poses[start]
        else:
            raise DeltaTransformError(
                "Seeding a prev_cmd delta needs prev_cmd.ee.* (preferred) or ee.* in "
                "observation.state."
            )
        if episode_slice.stop - start > 1:
            reference_poses[start + 1 : episode_slice.stop] = absolute_poses[start : episode_slice.stop - 1]
    return reference_poses


def derive_delta_action(
    *,
    absolute_action: np.ndarray,
    action_names: list[str],
    observation_state: np.ndarray,
    observation_names: list[str],
    episode_index: np.ndarray,
    action_mode: str,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Convert an absolute-EE action block to a delta one.

    Returns ``(delta_action, delta_action_names, report)``. The per-frame maths comes from
    :class:`AbsoluteEEToDeltaEEAction`, the same step the deployment inverse pairs with, so the
    transform cannot drift from the reconstruction that will run on the robot.
    """
    validate_action_mode(action_mode)
    if not is_delta_action_mode(action_mode):
        raise DeltaTransformError(f"{action_mode!r} is not a delta action mode.")
    reference = delta_reference_for_action_mode(action_mode)

    absolute_action = np.asarray(absolute_action, dtype=np.float64)
    absolute_poses = extract_pose_stream(
        absolute_action,
        action_names,
        position_keys=ABSOLUTE_POSITION_KEYS,
        quat_keys=ABSOLUTE_QUAT_KEYS,
    )
    if absolute_poses is None:
        raise DeltaTransformError(
            "Source action is not an absolute EE contract "
            f"(need {ABSOLUTE_POSITION_KEYS + ABSOLUTE_QUAT_KEYS}); got {action_names}."
        )

    observation_state = np.asarray(observation_state, dtype=np.float64)
    recorded_prev_cmd_poses = extract_pose_stream(
        observation_state,
        observation_names,
        position_keys=PREV_CMD_POSITION_KEYS,
        quat_keys=PREV_CMD_QUAT_KEYS,
    )
    measured_poses = extract_pose_stream(
        observation_state,
        observation_names,
        position_keys=ABSOLUTE_POSITION_KEYS,
        quat_keys=ABSOLUTE_QUAT_KEYS,
    )
    reference_poses = build_reference_poses(
        absolute_poses=absolute_poses,
        episode_index=episode_index,
        recorded_prev_cmd_poses=recorded_prev_cmd_poses,
        measured_poses=measured_poses,
        reference=reference,
    )

    gripper_index = _column_index(action_names, GRIPPER_KEY)
    forward = AbsoluteEEToDeltaEEAction(reference=reference)
    delta_names = list(delta_ee_action_keys(reference))
    total_frames = int(absolute_action.shape[0])
    delta_action = np.zeros((total_frames, len(delta_names)), dtype=np.float64)

    for frame_index in range(total_frames):
        # Synthesised observation carrying only the reference this mode reads, so a mode that
        # looked at the wrong group would raise instead of silently using the other one.
        observation = _reference_observation(reference_poses[frame_index], reference)
        action = {
            "ee.x": float(absolute_poses[frame_index][0, 3]),
            "ee.y": float(absolute_poses[frame_index][1, 3]),
            "ee.z": float(absolute_poses[frame_index][2, 3]),
            **dict(
                zip(
                    ABSOLUTE_QUAT_KEYS,
                    (
                        float(v)
                        for v in Rotation.from_matrix(absolute_poses[frame_index][:3, :3]).as_quat()
                    ),
                    strict=True,
                )
            ),
            GRIPPER_KEY: float(absolute_action[frame_index, gripper_index]),
        }
        delta = forward({TransitionKey.ACTION: action, TransitionKey.OBSERVATION: observation})[
            TransitionKey.ACTION
        ]
        delta_action[frame_index] = [float(delta[name]) for name in delta_names]

    report = verify_delta_reconstruction(
        delta_action=delta_action,
        delta_names=delta_names,
        reference_poses=reference_poses,
        absolute_poses=absolute_poses,
        reference=reference,
    )
    report["action_mode"] = action_mode
    report["frames"] = total_frames
    report["episodes"] = len(_episode_slices(episode_index))
    return delta_action, delta_names, report


def _reference_observation(reference_pose: np.ndarray, reference: str) -> dict[str, float]:
    quaternion = Rotation.from_matrix(reference_pose[:3, :3]).as_quat()
    position_keys = PREV_CMD_POSITION_KEYS if reference == "prev_cmd" else ABSOLUTE_POSITION_KEYS
    quat_keys = PREV_CMD_QUAT_KEYS if reference == "prev_cmd" else ABSOLUTE_QUAT_KEYS
    return {
        **dict(zip(position_keys, (float(v) for v in reference_pose[:3, 3]), strict=True)),
        **dict(zip(quat_keys, (float(v) for v in quaternion), strict=True)),
    }


def verify_delta_reconstruction(
    *,
    delta_action: np.ndarray,
    delta_names: list[str],
    reference_poses: np.ndarray,
    absolute_poses: np.ndarray,
    reference: str,
) -> dict[str, Any]:
    """Rebuild the absolute stream from the deltas and compare against the source.

    This is the check that would have caught the capture-time rate bug: a delta spanning the
    wrong interval reconstructs a trajectory that no longer matches the recorded commands, even
    though it still looks like a plausible smooth path.
    """
    inverse = DeltaEEToAbsoluteEEAction(reference=reference)
    worst_position_error_m = 0.0
    worst_rotation_error_deg = 0.0
    for frame_index in range(delta_action.shape[0]):
        action = {name: float(delta_action[frame_index, i]) for i, name in enumerate(delta_names)}
        rebuilt = inverse(
            {
                TransitionKey.ACTION: action,
                TransitionKey.OBSERVATION: _reference_observation(
                    reference_poses[frame_index], reference
                ),
            }
        )[TransitionKey.ACTION]
        rebuilt_position = np.array([rebuilt["ee.x"], rebuilt["ee.y"], rebuilt["ee.z"]])
        rebuilt_rotation = Rotation.from_quat(
            [rebuilt[key] for key in ABSOLUTE_QUAT_KEYS]
        ).as_matrix()
        worst_position_error_m = max(
            worst_position_error_m,
            float(np.abs(rebuilt_position - absolute_poses[frame_index][:3, 3]).max()),
        )
        worst_rotation_error_deg = max(
            worst_rotation_error_deg,
            float(
                np.degrees(
                    np.linalg.norm(
                        Rotation.from_matrix(
                            rebuilt_rotation.T @ absolute_poses[frame_index][:3, :3]
                        ).as_rotvec()
                    )
                )
            ),
        )

    if (
        worst_position_error_m > SELF_CHECK_POSITION_TOL_M
        or worst_rotation_error_deg > SELF_CHECK_ROTATION_TOL_DEG
    ):
        raise DeltaTransformError(
            "Delta transform is not invertible: rebuilding the absolute stream differs from the "
            f"source by {worst_position_error_m * 1e3:.4f} mm / {worst_rotation_error_deg:.4f} deg "
            f"(limits {SELF_CHECK_POSITION_TOL_M * 1e3:.4f} mm / {SELF_CHECK_ROTATION_TOL_DEG:.4f} deg). "
            "The reference poses or the rotation convention are wrong."
        )
    return {
        "reconstruction_max_position_error_mm": worst_position_error_m * 1e3,
        "reconstruction_max_rotation_error_deg": worst_rotation_error_deg,
    }


def summarize_delta_scale(
    *,
    delta_action: np.ndarray,
    delta_names: list[str],
    fps: int,
) -> dict[str, Any]:
    """Per-frame delta magnitudes, so an implausible cadence is visible in the manifest.

    A delta that spans the wrong interval shows up here as a translation speed that does not
    match how fast the arm actually moved.
    """
    position_columns = [i for i, name in enumerate(delta_names) if name.endswith((".dx", ".dy", ".dz"))]
    rotation_columns = [i for i, name in enumerate(delta_names) if name.endswith((".drx", ".dry", ".drz"))]
    translation_norm_m = np.linalg.norm(delta_action[:, position_columns], axis=1)
    rotation_norm_rad = np.linalg.norm(delta_action[:, rotation_columns], axis=1)
    return {
        "median_translation_per_frame_mm": float(np.median(translation_norm_m) * 1e3),
        "p95_translation_per_frame_mm": float(np.percentile(translation_norm_m, 95) * 1e3),
        "max_translation_per_frame_mm": float(np.max(translation_norm_m) * 1e3),
        "implied_p95_speed_mm_s": float(np.percentile(translation_norm_m, 95) * 1e3 * max(fps, 1)),
        "median_rotation_per_frame_deg": float(np.degrees(np.median(rotation_norm_rad))),
        "max_rotation_per_frame_deg": float(np.degrees(np.max(rotation_norm_rad))),
    }
