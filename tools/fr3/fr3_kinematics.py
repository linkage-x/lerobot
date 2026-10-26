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

"""
FR3 kinematics in pure numpy, for tools that must not import the lerobot package (torch, ...).

Craig DH with the joint zero offsets as a fourth column, batched FK + Jacobian, a damped-least-squares IK
inside the joint walls, and the fixed URDF chain from the flange to a tool frame. Shared by the G3
registration study (``dual_fr3_registration_sim``) and the dual-arm replay runtime's fake arms and
separation gate (``dual_fr3_replay_runtime``).
"""

# ruff: noqa: N802, N803, N806  (matrix names: R, T, J follow the maths)
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
FR3_URDF = (
    REPO
    / "src"
    / "lerobot"
    / "robots"
    / "franka_research3"
    / "assets"
    / "franka_fr3"
    / "fr3_corenetic_gripper.urdf"
)
#: The URDF link that FK's flange is (fr3_link7 + 0.107 along z).
FLANGE_LINK = "fr3_link8"

# FR3, Craig (modified) DH: (a_{i-1}, d_i, alpha_{i-1}) as payload_identification._DH, plus the joint's zero
# offset -- the fourth column is what a joint calibration error moves.
NOMINAL_DH = np.array(
    [
        (0.0, 0.333, 0.0, 0.0),
        (0.0, 0.0, -math.pi / 2, 0.0),
        (0.0, 0.316, math.pi / 2, 0.0),
        (0.0825, 0.0, math.pi / 2, 0.0),
        (-0.0825, 0.384, -math.pi / 2, 0.0),
        (0.0, 0.0, math.pi / 2, 0.0),
        (0.088, 0.0, math.pi / 2, 0.0),
    ]
)
FLANGE_D = 0.107
# tools/fr3/fr3_tcp_pivot.py FR3_JOINT_LIMITS and tools/fr3/fr3_pivot_auto.py's panda_py walls, whichever binds
_FR3_LOWER = np.array([-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159])
_FR3_UPPER = np.array([2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159])
_WALL_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
_WALL_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])
LIMIT_MARGIN_DEG = 3.0  # fr3_pivot_auto.LIMIT_MARGIN_DEG
Q_LOWER = np.maximum(_FR3_LOWER, _WALL_LOWER) + math.radians(LIMIT_MARGIN_DEG)
Q_UPPER = np.minimum(_FR3_UPPER, _WALL_UPPER) - math.radians(LIMIT_MARGIN_DEG)
Q_READY = np.array([0.0, -math.pi / 4, 0.0, -3 * math.pi / 4, 0.0, math.pi / 2, math.pi / 4])

IK_TOL_MM = 0.05
IK_TOL_DEG = 0.05


def _dh_batch(a: float, d: float, alpha: float, theta: np.ndarray) -> np.ndarray:
    ca, sa = math.cos(alpha), math.sin(alpha)
    ct, st = np.cos(theta), np.sin(theta)
    T = np.zeros((len(theta), 4, 4))
    T[:, 0, 0], T[:, 0, 1], T[:, 0, 3] = ct, -st, a
    T[:, 1, 0], T[:, 1, 1], T[:, 1, 2], T[:, 1, 3] = st * ca, ct * ca, -sa, -d * sa
    T[:, 2, 0], T[:, 2, 1], T[:, 2, 2], T[:, 2, 3] = st * sa, ct * sa, ca, d * ca
    T[:, 3, 3] = 1.0
    return T


@dataclass(frozen=True)
class Arm:
    """One FR3's kinematics in its own base frame; ``dh`` as NOMINAL_DH."""

    dh: np.ndarray
    flange_d: float = FLANGE_D

    def kinematics(self, q: np.ndarray, point: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Base->flange ``T`` (n,4,4) and the 6x7 Jacobian [Jv; Jw] at ``point`` (flange frame; origin if None)."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        axes, origins = [], []
        for i, (a, d, alpha, offset) in enumerate(self.dh):
            T = T @ _dh_batch(a, d, alpha, q[:, i] + offset)
            axes.append(T[:, :3, 2].copy())
            origins.append(T[:, :3, 3].copy())
        T[:, :3, 3] += T[:, :3, 2] * self.flange_d
        p = T[:, :3, 3] if point is None else T[:, :3, 3] + T[:, :3, :3] @ np.asarray(point, dtype=np.float64)
        J = np.zeros((len(q), 6, 7))
        for i, (z, o) in enumerate(zip(axes, origins, strict=True)):
            J[:, :3, i] = np.cross(z, p - o)
            J[:, 3:, i] = z
        return T, J

    def fk(self, q: np.ndarray) -> np.ndarray:
        return self.kinematics(q)[0]

    def point(self, q: np.ndarray, point: np.ndarray) -> np.ndarray:
        T = self.fk(q)
        return T[:, :3, 3] + T[:, :3, :3] @ np.asarray(point, dtype=np.float64)


NOMINAL = Arm(NOMINAL_DH.copy())


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vectors of (n,3,3) rotations, angles below pi."""
    cos = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1)
    s = np.sin(angle)
    factor = np.where(s > 1e-9, angle / (2.0 * np.where(s > 1e-9, s, 1.0)), 0.5)
    return v * factor[:, None]


def ik(
    arm: Arm, T_target: np.ndarray, seed: np.ndarray, *, iters: int = 400
) -> tuple[np.ndarray, np.ndarray]:
    """Flange poses ``T_target`` (n,4,4) by damped least squares, all at once; ``(q, converged)``.

    The nullspace pulls toward the middle of the joint range, and the joints are kept inside it with
    LIMIT_MARGIN_DEG to spare; a pose that cannot be reached there does not converge.
    """
    q = np.array(np.broadcast_to(seed, (len(T_target), 7)), dtype=np.float64)
    mid = (Q_LOWER + Q_UPPER) / 2.0
    eye6 = np.eye(6)
    for _ in range(iters):
        T, J = arm.kinematics(q)
        e = np.concatenate(
            [
                T_target[:, :3, 3] - T[:, :3, 3],
                _rotvec(T_target[:, :3, :3] @ T[:, :3, :3].transpose(0, 2, 1)),
            ],
            axis=1,
        )
        Jt = J.transpose(0, 2, 1)
        A = J @ Jt + 1e-6 * eye6
        dq = (Jt @ np.linalg.solve(A, e[:, :, None]))[:, :, 0]
        Jpinv = Jt @ np.linalg.inv(A)
        null = np.eye(7) - Jpinv @ J
        dq += (null @ (0.05 * (mid - q))[:, :, None])[:, :, 0]
        step = np.abs(dq).max(axis=1, keepdims=True)
        dq *= np.minimum(1.0, 0.2 / np.maximum(step, 1e-12))
        q = np.clip(q + dq, Q_LOWER, Q_UPPER)
    T = arm.fk(q)
    pos = np.linalg.norm(T_target[:, :3, 3] - T[:, :3, 3], axis=1)
    rot = np.linalg.norm(_rotvec(T_target[:, :3, :3] @ T[:, :3, :3].transpose(0, 2, 1)), axis=1)
    return q, (pos * 1e3 < IK_TOL_MM) & (np.degrees(rot) < IK_TOL_DEG)


#: Retried from these when the ready-posture seed does not converge: the wrist turned, the elbow bent, the
#: shoulder forward. Without them a reachable pose read as unreachable about one time in five.
_SEED_NUDGES = (
    np.zeros(7),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, math.pi / 2]),
    np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, -math.pi / 2]),
    np.array([0.0, 0.3, 0.0, 0.4, 0.0, 0.3, 0.0]),
    np.array([0.0, -0.3, 0.0, -0.5, 0.0, -0.3, 0.0]),
)


def ik_retry(arm: Arm, T_target: np.ndarray, seed: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``ik`` from the seed, then from each of _SEED_NUDGES for the poses still not converged."""
    seeds = np.array(np.broadcast_to(seed, (len(T_target), 7)), dtype=np.float64)
    q = np.zeros_like(seeds)
    ok = np.zeros(len(T_target), dtype=bool)
    for nudge in _SEED_NUDGES:
        todo = ~ok
        if not todo.any():
            break
        q_try, ok_try = ik(arm, T_target[todo], np.clip(seeds[todo] + nudge, Q_LOWER, Q_UPPER))
        q[todo], ok[todo] = q_try, ok_try
    return q, ok


def _rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (
        math.cos(roll),
        math.sin(roll),
        math.cos(pitch),
        math.sin(pitch),
        math.cos(yaw),
        math.sin(yaw),
    )
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def urdf_fixed_transform(
    child_link: str, parent_link: str = FLANGE_LINK, urdf: Path | str = FR3_URDF
) -> np.ndarray:
    """``T_parent_child`` through fixed joints only; a movable joint on the way is an error, not zero."""
    root = ET.parse(str(urdf)).getroot()
    joints = {j.find("child").get("link"): j for j in root.findall("joint")}
    T = np.eye(4)
    link = child_link
    while link != parent_link:
        joint = joints.get(link)
        if joint is None:
            raise ValueError(f"{urdf}: {child_link} does not descend from {parent_link}")
        if joint.get("type") != "fixed":
            raise ValueError(
                f"{urdf}: joint {joint.get('name')} between {parent_link} and {child_link} is not fixed"
            )
        origin = joint.find("origin")
        xyz = [float(v) for v in (origin.get("xyz") if origin is not None else "0 0 0").split()]
        rpy = [float(v) for v in (origin.get("rpy") if origin is not None else "0 0 0").split()]
        step = np.eye(4)
        step[:3, :3], step[:3, 3] = _rpy_matrix(*rpy), xyz
        T = step @ T
        link = joint.find("parent").get("link")
    return T


def link_points(arm: Arm, q: np.ndarray, tool: np.ndarray | None = None) -> np.ndarray:
    """(n, m, 3) points along the arm in its base frame -- base, shoulder, elbow, wrist, flange (and the tool
    point when given) -- the polyline a coarse capsule model of the links hangs on."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    origins = [T[:, :3, 3].copy()]
    for i, (a, d, alpha, offset) in enumerate(arm.dh):
        T = T @ _dh_batch(a, d, alpha, q[:, i] + offset)
        origins.append(T[:, :3, 3].copy())
    flange = T[:, :3, 3] + T[:, :3, 2] * arm.flange_d
    points = [origins[0], origins[1], origins[3], origins[4], origins[5], origins[7], flange]
    if tool is not None:
        points.append(flange + (T[:, :3, :3] @ np.asarray(tool, dtype=np.float64)))
    return np.stack(points, axis=1)
