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

"""Choose the joint configuration a replayed TCP trajectory should start from.

The FR3 is redundant: one TCP path has a family of joint paths, and frame-by-frame IK
follows whichever branch it starts on. A branch can be fine at frame 0 and walk into a
joint limit hundreds of frames later (2026-10-09: the branch reached from the arm's start
pose needed j6 = 4.15 rad at frame 479, past panda_py's 3.7525 wall). So the start is
chosen for the whole trajectory: every frame-0 solution found from many seeds is followed
through all frames the way the replay follows it. Of those that track every frame with room
to spare at every joint limit, the one nearest a reference posture (the arm's start pose)
wins: the margins of the branches that pass barely differ, and the nearest one is the
shortest joint move to the start and the least unusual posture.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Any, Protocol

import numpy as np


class _Kinematics(Protocol):
    def forward_kinematics(self, joint_positions_rad: np.ndarray) -> np.ndarray: ...

    def inverse_kinematics(self, current_joint_positions_rad: np.ndarray, desired_pose: np.ndarray) -> np.ndarray: ...


@dataclass
class TrajectoryStartPlan:
    feasible: bool
    joints_rad: list[float]
    # Smallest distance (rad) any joint came to its limit over the trajectory, and where.
    min_limit_margin_rad: float
    min_margin_joint: int
    min_margin_frame: int
    max_position_error_m: float
    max_orientation_error_deg: float
    max_joint_step_rad: float
    # First frame (index into the poses) the branch could not track; -1 when it tracked all.
    failed_at_frame: int
    candidates_tried: int
    candidates_feasible: int
    reason: str = ""
    rejected: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "feasible": bool(self.feasible),
            "joints_rad": [float(v) for v in self.joints_rad],
            "min_limit_margin_rad": float(self.min_limit_margin_rad),
            "min_margin_joint": int(self.min_margin_joint),
            "min_margin_frame": int(self.min_margin_frame),
            "max_position_error_m": float(self.max_position_error_m),
            "max_orientation_error_deg": float(self.max_orientation_error_deg),
            "max_joint_step_rad": float(self.max_joint_step_rad),
            "failed_at_frame": int(self.failed_at_frame),
            "candidates_tried": int(self.candidates_tried),
            "candidates_feasible": int(self.candidates_feasible),
            "reason": self.reason,
        }


def _pose_error(actual: np.ndarray, target: np.ndarray) -> tuple[float, float]:
    position_error_m = float(np.linalg.norm(actual[:3, 3] - target[:3, 3]))
    relative = actual[:3, :3].T @ target[:3, :3]
    cos_angle = float(np.clip((np.trace(relative) - 1.0) / 2.0, -1.0, 1.0))
    return position_error_m, math.degrees(math.acos(cos_angle))


@dataclass
class _Branch:
    joints_rad: np.ndarray
    min_margin_rad: float = math.inf
    min_margin_joint: int = -1
    min_margin_frame: int = -1
    max_position_error_m: float = 0.0
    max_orientation_error_deg: float = 0.0
    max_joint_step_rad: float = 0.0
    failed_at_frame: int = -1
    reason: str = ""


def _follow_branch(
    kinematics: _Kinematics,
    poses: Sequence[np.ndarray],
    q0: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    margin_rad: float,
    position_tolerance_m: float,
    orientation_tolerance_deg: float,
    max_joint_step_rad: float,
) -> _Branch:
    branch = _Branch(joints_rad=q0.copy())
    q = q0.copy()
    for index, pose in enumerate(poses):
        if index > 0:
            q_next = np.asarray(kinematics.inverse_kinematics(q, pose), dtype=np.float64).reshape(q.shape)
            step = float(np.max(np.abs(q_next - q)))
            branch.max_joint_step_rad = max(branch.max_joint_step_rad, step)
            q = q_next
            if step > max_joint_step_rad:
                branch.failed_at_frame, branch.reason = index, f"joint step {step:.3f} rad"
                return branch
        position_error_m, orientation_error_deg = _pose_error(kinematics.forward_kinematics(q), pose)
        branch.max_position_error_m = max(branch.max_position_error_m, position_error_m)
        branch.max_orientation_error_deg = max(branch.max_orientation_error_deg, orientation_error_deg)
        if position_error_m > position_tolerance_m or orientation_error_deg > orientation_tolerance_deg:
            branch.failed_at_frame = index
            branch.reason = f"IK misses by {position_error_m * 1000.0:.1f} mm / {orientation_error_deg:.2f} deg"
            return branch
        margins = np.minimum(q - lower, upper - q)
        joint = int(np.argmin(margins))
        if margins[joint] < branch.min_margin_rad:
            branch.min_margin_rad = float(margins[joint])
            branch.min_margin_joint, branch.min_margin_frame = joint, index
        if margins[joint] < margin_rad:
            branch.failed_at_frame = index
            branch.reason = f"j{joint + 1} within {margins[joint]:.3f} rad of its limit"
            return branch
    return branch


def plan_trajectory_start(
    kinematics: _Kinematics,
    poses: Sequence[np.ndarray],
    lower: Sequence[float],
    upper: Sequence[float],
    *,
    seeds: Sequence[Sequence[float]] = (),
    reference_joints: Sequence[float] | None = None,
    num_random_seeds: int = 64,
    margin_rad: float = 0.1,
    position_tolerance_m: float = 0.002,
    orientation_tolerance_deg: float = 1.0,
    max_joint_step_rad: float = 0.35,
    rng_seed: int = 0,
) -> TrajectoryStartPlan:
    """Pick the frame-0 joints from which IK tracks all ``poses`` inside ``lower``/``upper``.

    ``kinematics`` must already clip to the limits the arm really has (for panda_py, the
    intersection with PANDA_PY_JOINT_LIMITS_*). Each candidate is a frame-0 IK solution from
    one seed (the given ``seeds`` first, then ``num_random_seeds`` uniform draws, seeded by
    ``rng_seed`` so the gateway's preview and a rerun agree). A candidate is feasible when
    every frame is tracked within tolerance, no joint comes nearer than ``margin_rad`` to a
    limit, and no frame-to-frame step exceeds ``max_joint_step_rad`` (the replay's joint-jump
    guard). Among feasible ones the nearest to ``reference_joints`` (default: the first seed)
    wins. When none is feasible, the plan is the candidate that got furthest, with
    ``feasible=False``.
    """
    if not poses:
        raise ValueError("plan_trajectory_start needs at least one pose.")
    poses = [np.asarray(pose, dtype=np.float64).reshape(4, 4) for pose in poses]
    lower_arr = np.asarray(lower, dtype=np.float64)
    upper_arr = np.asarray(upper, dtype=np.float64)
    rng = np.random.default_rng(int(rng_seed))
    seed_list = [np.clip(np.asarray(seed, dtype=np.float64), lower_arr, upper_arr) for seed in seeds]
    seed_list += [rng.uniform(lower_arr, upper_arr) for _ in range(int(num_random_seeds))]
    if reference_joints is not None:
        reference = np.asarray(reference_joints, dtype=np.float64)
    elif seeds:
        reference = seed_list[0]
    else:
        reference = (lower_arr + upper_arr) / 2.0

    starts: list[np.ndarray] = []
    for seed in seed_list:
        q0 = np.asarray(kinematics.inverse_kinematics(seed, poses[0]), dtype=np.float64)
        position_error_m, orientation_error_deg = _pose_error(kinematics.forward_kinematics(q0), poses[0])
        if position_error_m > position_tolerance_m or orientation_error_deg > orientation_tolerance_deg:
            continue
        if any(float(np.max(np.abs(q0 - other))) < 0.05 for other in starts):
            continue
        starts.append(q0)

    best: _Branch | None = None
    feasible_count = 0
    rejected: list[dict[str, Any]] = []
    for q0 in starts:
        branch = _follow_branch(
            kinematics,
            poses,
            q0,
            lower_arr,
            upper_arr,
            margin_rad=margin_rad,
            position_tolerance_m=position_tolerance_m,
            orientation_tolerance_deg=orientation_tolerance_deg,
            max_joint_step_rad=max_joint_step_rad,
        )
        if branch.failed_at_frame < 0:
            feasible_count += 1
        else:
            rejected.append({"joints_rad": q0.tolist(), "failed_at_frame": branch.failed_at_frame, "reason": branch.reason})
        if best is None or _better(branch, best, reference):
            best = branch

    if best is None:
        return TrajectoryStartPlan(
            feasible=False,
            joints_rad=[],
            min_limit_margin_rad=math.nan,
            min_margin_joint=-1,
            min_margin_frame=-1,
            max_position_error_m=math.nan,
            max_orientation_error_deg=math.nan,
            max_joint_step_rad=math.nan,
            failed_at_frame=0,
            candidates_tried=len(seed_list),
            candidates_feasible=0,
            reason="no seed solved frame 0 inside the joint limits",
        )
    return TrajectoryStartPlan(
        feasible=best.failed_at_frame < 0,
        joints_rad=best.joints_rad.tolist(),
        min_limit_margin_rad=best.min_margin_rad,
        min_margin_joint=best.min_margin_joint,
        min_margin_frame=best.min_margin_frame,
        max_position_error_m=best.max_position_error_m,
        max_orientation_error_deg=best.max_orientation_error_deg,
        max_joint_step_rad=best.max_joint_step_rad,
        failed_at_frame=best.failed_at_frame,
        candidates_tried=len(seed_list),
        candidates_feasible=feasible_count,
        reason=best.reason,
        rejected=rejected,
    )


def _better(candidate: _Branch, incumbent: _Branch, reference: np.ndarray) -> bool:
    candidate_done = candidate.failed_at_frame < 0
    incumbent_done = incumbent.failed_at_frame < 0
    if candidate_done != incumbent_done:
        return candidate_done
    if not candidate_done:
        return candidate.failed_at_frame > incumbent.failed_at_frame
    return float(np.linalg.norm(candidate.joints_rad - reference)) < float(
        np.linalg.norm(incumbent.joints_rad - reference)
    )
