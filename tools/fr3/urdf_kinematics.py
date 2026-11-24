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
Kinematics of any serial chain read from a URDF, in pure numpy (roadmap docs/monte02_left_arm_roadmap.html, M1.3).

``fr3_kinematics`` hard-codes the FR3's DH table; Monte02's arm (tilted shoulder axis, joints on a trunk that
itself moves) has none, and its URDF will be replaced by the one the delivered robot publishes. So the chain is
read from the URDF as it is: every joint between ``base_link`` and ``tip_link``, fixed ones folded into
constants, movable ones (revolute / continuous / prismatic) along their ``<axis>``. Joints the caller names in
``locked`` are folded in at that value too: Monte02's trunk (knee, hip, waist) is locked for a session
(roadmap D4), so the arm is a 7-joint chain from the base, for that trunk posture.

Same calling conventions as ``fr3_kinematics``: batched ``kinematics(q, point) -> (T, J)`` with J = [Jv; Jw]
in the base frame at ``point`` (tip frame), a damped-least-squares ``ik`` inside the joint limits (less a
margin) with a nullspace pull, and ``ik_retry`` from a few nudged seeds.

Nothing here talks to a robot.
"""

# ruff: noqa: N802, N803, N806  (matrix names: R, T, J follow the maths)
from __future__ import annotations

import hashlib
import math
import xml.etree.ElementTree as ET
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

MOVABLE = ("revolute", "continuous", "prismatic")
LIMIT_MARGIN_DEG = 3.0  # as fr3_kinematics
LIMIT_MARGIN_M = 0.002  # prismatic joints
IK_TOL_MM = 0.05
IK_TOL_DEG = 0.05
EARLY_EXIT_ITERS = 60  # fr3_kinematics.EARLY_EXIT_ITERS


def rpy_matrix(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF ``rpy``: R = Rz(yaw) Ry(pitch) Rx(roll)."""
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


def _origin(joint: ET.Element) -> np.ndarray:
    origin = joint.find("origin")
    xyz = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
    rpy = [float(v) for v in (origin.get("rpy", "0 0 0") if origin is not None else "0 0 0").split()]
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = rpy_matrix(*rpy), xyz
    return T


def _axis(joint: ET.Element) -> np.ndarray:
    axis = joint.find("axis")
    a = np.array([float(v) for v in (axis.get("xyz") if axis is not None else "1 0 0").split()])
    n = np.linalg.norm(a)
    if n < 1e-12:
        raise ValueError(f"joint {joint.get('name')}: zero axis")
    return a / n


def _motion(kind: str, axis: np.ndarray, q: np.ndarray) -> np.ndarray:
    """(n,4,4) transform of a joint at positions ``q`` (n,) about / along the unit ``axis``."""
    T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    if kind == "prismatic":
        T[:, :3, 3] = q[:, None] * axis
        return T
    x, y, z = axis
    K = np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]])
    s, c = np.sin(q)[:, None, None], np.cos(q)[:, None, None]
    T[:, :3, :3] = np.eye(3) + s * K + (1.0 - c) * (K @ K)
    return T


@dataclass(frozen=True)
class Chain:
    """A serial chain ``base_link`` -> ``tip_link`` of a URDF; build it with :func:`load_chain`."""

    urdf: str
    base_link: str
    tip_link: str
    joint_names: tuple[str, ...]
    kinds: tuple[str, ...]
    #: per movable joint: the constant transform from the previous joint's moving frame (or the base) to this
    #: joint's frame, then the last constant to the tip
    pre: np.ndarray = field(repr=False)
    post: np.ndarray = field(repr=False)
    axes: np.ndarray = field(repr=False)
    #: URDF limits; continuous joints get +-inf
    lower: np.ndarray = field(repr=False)
    upper: np.ndarray = field(repr=False)
    velocity: np.ndarray = field(repr=False)
    locked: tuple[tuple[str, float], ...] = ()
    urdf_sha256: str = ""

    @property
    def dof(self) -> int:
        return len(self.joint_names)

    @property
    def fingerprint(self) -> str:
        """What a gate approved: the URDF bytes, the two ends and the locked joints' values."""
        h = hashlib.sha256()
        h.update(self.urdf_sha256.encode())
        h.update(f"|{self.base_link}|{self.tip_link}|".encode())
        for name, value in self.locked:
            h.update(f"{name}={value!r};".encode())
        return h.hexdigest()

    def q_limits(self, margin_deg: float = LIMIT_MARGIN_DEG, margin_m: float = LIMIT_MARGIN_M) -> tuple[np.ndarray, np.ndarray]:
        """Limits less a margin (radians for revolute joints, metres for prismatic ones)."""
        margin = np.array([margin_m if k == "prismatic" else math.radians(margin_deg) for k in self.kinds])
        lo, hi = self.lower + margin, self.upper - margin
        if np.any(lo > hi):
            raise ValueError(f"margin larger than half the range of {np.array(self.joint_names)[lo > hi].tolist()}")
        return lo, hi

    def _forward(self, q: np.ndarray) -> tuple[np.ndarray, list[np.ndarray]]:
        """Base->tip (n,4,4) and, per movable joint, its frame (n,4,4) before it moves."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        if q.shape[1] != self.dof:
            raise ValueError(f"q has {q.shape[1]} columns, the chain has {self.dof} joints")
        T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        frames = []
        for i in range(self.dof):
            T = T @ self.pre[i]
            frames.append(T)
            T = T @ _motion(self.kinds[i], self.axes[i], q[:, i])
        return T @ self.post, frames

    def kinematics(self, q: np.ndarray, point: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Base->tip ``T`` (n,4,4) and the 6 x dof Jacobian [Jv; Jw] at ``point`` (tip frame; origin if None)."""
        T, frames = self._forward(q)
        p = T[:, :3, 3] if point is None else T[:, :3, 3] + T[:, :3, :3] @ np.asarray(point, dtype=np.float64)
        J = np.zeros((len(T), 6, self.dof))
        for i, F in enumerate(frames):
            z = F[:, :3, :3] @ self.axes[i]
            if self.kinds[i] == "prismatic":
                J[:, :3, i] = z
            else:
                J[:, :3, i] = np.cross(z, p - F[:, :3, 3])
                J[:, 3:, i] = z
        return T, J

    def fk(self, q: np.ndarray) -> np.ndarray:
        return self._forward(q)[0]

    def point(self, q: np.ndarray, point: np.ndarray) -> np.ndarray:
        T = self.fk(q)
        return T[:, :3, 3] + T[:, :3, :3] @ np.asarray(point, dtype=np.float64)

    def joint_frames(self, q: np.ndarray) -> np.ndarray:
        """(n, dof+1, 4, 4): each movable joint's child-link frame after it moves, then the tip."""
        q = np.atleast_2d(np.asarray(q, dtype=np.float64))
        T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
        out = []
        for i in range(self.dof):
            T = T @ self.pre[i] @ _motion(self.kinds[i], self.axes[i], q[:, i])
            out.append(T)
        out.append(T @ self.post)
        return np.stack(out, axis=1)


def _path(root: ET.Element, base_link: str, tip_link: str) -> list[ET.Element]:
    """Joints from ``base_link`` down to ``tip_link``, in that order."""
    by_child = {j.find("child").get("link"): j for j in root.findall("joint")}
    links = {link.get("name") for link in root.findall("link")}
    for name in (base_link, tip_link):
        if name not in links:
            raise ValueError(f"no link {name!r} in the URDF")
    path, link = [], tip_link
    while link != base_link:
        joint = by_child.get(link)
        if joint is None:
            raise ValueError(f"{tip_link} does not descend from {base_link}")
        path.append(joint)
        link = joint.find("parent").get("link")
    return path[::-1]


def load_chain(
    urdf: Path | str, base_link: str, tip_link: str, locked: Mapping[str, float] | None = None
) -> Chain:
    """The chain ``base_link`` -> ``tip_link`` of ``urdf``; ``locked`` joints (on the path) fixed at those values.

    A mimic joint on the path is an error (none of our arms has one), and so is a locked joint that is not on the
    path or a locked value outside its limits: either is a typo that would silently give another robot.
    """
    urdf = Path(urdf)
    data = urdf.read_bytes()
    root = ET.fromstring(data)
    locked = dict(locked or {})
    path = _path(root, base_link, tip_link)
    on_path = {j.get("name") for j in path}
    stray = sorted(set(locked) - on_path)
    if stray:
        raise ValueError(f"locked joints not between {base_link} and {tip_link}: {stray}")

    names, kinds, pres, axes, lower, upper, velocity = [], [], [], [], [], [], []
    acc = np.eye(4)
    for joint in path:
        name, kind = joint.get("name"), joint.get("type")
        acc = acc @ _origin(joint)
        if kind == "fixed":
            continue
        if kind not in MOVABLE:
            raise ValueError(f"joint {name}: type {kind!r} is not supported")
        if joint.find("mimic") is not None:
            raise ValueError(f"joint {name} is a mimic joint; not supported on a chain")
        limit = joint.find("limit")
        lo = float(limit.get("lower", "nan")) if limit is not None and kind != "continuous" else -math.inf
        hi = float(limit.get("upper", "nan")) if limit is not None and kind != "continuous" else math.inf
        axis = _axis(joint)
        if name in locked:
            value = float(locked[name])
            if not lo <= value <= hi:
                raise ValueError(f"locked {name}={value} outside its limits [{lo}, {hi}]")
            acc = acc @ _motion(kind, axis, np.array([value]))[0]
            continue
        if not (math.isfinite(lo) or kind == "continuous") or not (math.isfinite(hi) or kind == "continuous"):
            raise ValueError(f"joint {name}: no limits")
        names.append(name)
        kinds.append(kind)
        pres.append(acc)
        axes.append(axis)
        lower.append(lo)
        upper.append(hi)
        velocity.append(float(limit.get("velocity", "nan")) if limit is not None else math.nan)
        acc = np.eye(4)
    return Chain(
        urdf=str(urdf),
        base_link=base_link,
        tip_link=tip_link,
        joint_names=tuple(names),
        kinds=tuple(kinds),
        pre=np.array(pres).reshape(len(pres), 4, 4),
        post=acc,
        axes=np.array(axes).reshape(len(axes), 3),
        lower=np.array(lower),
        upper=np.array(upper),
        velocity=np.array(velocity),
        locked=tuple(sorted((k, float(v)) for k, v in locked.items())),
        urdf_sha256=hashlib.sha256(data).hexdigest(),
    )


def fixed_transform(urdf: Path | str, parent_link: str, child_link: str) -> np.ndarray:
    """``T_parent_child`` through fixed joints only; a movable joint on the way is an error, not zero."""
    root = ET.parse(str(urdf)).getroot()
    T = np.eye(4)
    for joint in _path(root, parent_link, child_link):
        if joint.get("type") != "fixed":
            raise ValueError(f"joint {joint.get('name')} between {parent_link} and {child_link} is not fixed")
        T = T @ _origin(joint)
    return T


def _rotvec(R: np.ndarray) -> np.ndarray:
    """Rotation vectors of (n,3,3) rotations, angles below pi (fr3_kinematics._rotvec)."""
    cos = np.clip((np.trace(R, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
    angle = np.arccos(cos)
    v = np.stack([R[:, 2, 1] - R[:, 1, 2], R[:, 0, 2] - R[:, 2, 0], R[:, 1, 0] - R[:, 0, 1]], axis=1)
    s = np.sin(angle)
    factor = np.where(s > 1e-9, angle / (2.0 * np.where(s > 1e-9, s, 1.0)), 0.5)
    return v * factor[:, None]


def pose_error(T_target: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Position error (mm) and rotation error (deg) per pose."""
    pos = np.linalg.norm(T_target[:, :3, 3] - T[:, :3, 3], axis=1) * 1e3
    rot = np.degrees(np.linalg.norm(_rotvec(T_target[:, :3, :3] @ T[:, :3, :3].transpose(0, 2, 1)), axis=1))
    return pos, rot


def ik(
    chain: Chain,
    T_target: np.ndarray,
    seed: np.ndarray,
    *,
    iters: int = 400,
    early_exit: bool = False,
    rest: np.ndarray | None = None,
    limits: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Tip poses ``T_target`` (n,4,4) by damped least squares, all at once; ``(q, converged)``.

    As ``fr3_kinematics.ik``: joints kept inside ``limits`` (default the URDF limits less LIMIT_MARGIN_DEG),
    the nullspace pulls toward ``rest`` (default the middle of the range; a control loop passes where the arm
    stands), ``early_exit`` stops once every pose has converged. A pose not reachable inside the limits does not
    converge. Continuous joints are not pulled (their middle is undefined) unless ``rest`` is given.
    """
    lo, hi = chain.q_limits() if limits is None else limits
    n, dof = len(T_target), chain.dof
    q = np.array(np.broadcast_to(seed, (n, dof)), dtype=np.float64)
    finite = np.isfinite(lo) & np.isfinite(hi)
    if rest is None:
        mid = np.where(finite, (np.where(finite, lo, 0.0) + np.where(finite, hi, 0.0)) / 2.0, 0.0)
        mid = np.broadcast_to(mid, (n, dof))
        pull = np.broadcast_to(finite.astype(float), (n, dof))
    else:
        mid = np.array(np.broadcast_to(rest, (n, dof)), dtype=np.float64)
        pull = np.ones((n, dof))
    eye6 = np.eye(6)
    tol = np.array([IK_TOL_MM * 1e-3, math.radians(IK_TOL_DEG)])
    for _ in range(iters):
        T, J = chain.kinematics(q)
        e = np.concatenate(
            [T_target[:, :3, 3] - T[:, :3, 3], _rotvec(T_target[:, :3, :3] @ T[:, :3, :3].transpose(0, 2, 1))],
            axis=1,
        )
        if early_exit and np.all(np.linalg.norm(e[:, :3], axis=1) < tol[0] * 0.5) and np.all(
            np.linalg.norm(e[:, 3:], axis=1) < tol[1] * 0.5
        ):
            break
        Jt = J.transpose(0, 2, 1)
        A = J @ Jt + 1e-6 * eye6
        Ainv = np.linalg.inv(A)
        dq = (Jt @ (Ainv @ e[:, :, None]))[:, :, 0]
        null = np.eye(dof) - Jt @ Ainv @ J
        dq += (null @ (0.05 * pull * (mid - q))[:, :, None])[:, :, 0]
        step = np.abs(dq).max(axis=1, keepdims=True)
        dq *= np.minimum(1.0, 0.2 / np.maximum(step, 1e-12))
        q = np.clip(q + dq, lo, hi)
    pos, rot = pose_error(T_target, chain.fk(q))
    return q, (pos < IK_TOL_MM) & (rot < IK_TOL_DEG)


def default_nudges(chain: Chain) -> tuple[np.ndarray, ...]:
    """Seeds tried after the given one: none, the whole arm bent +-0.3 rad, then the last three joints (the
    wrist) turned a quarter of their range each way. Deterministic, so a QC verdict repeats."""
    dof = chain.dof
    lo, hi = chain.q_limits()
    span = np.where(np.isfinite(hi - lo), hi - lo, 2.0 * math.pi)
    sign = np.array([1.0 if i % 2 == 0 else -1.0 for i in range(dof)])
    wrist = np.zeros(dof)
    wrist[max(0, dof - 3) :] = 0.25 * span[max(0, dof - 3) :]
    return (np.zeros(dof), 0.3 * sign, -0.3 * sign, wrist, -wrist)


def ik_retry(
    chain: Chain,
    T_target: np.ndarray,
    seed: np.ndarray,
    *,
    early_exit: bool = False,
    rest: np.ndarray | None = None,
    nudges: Sequence[np.ndarray] | None = None,
    limits: tuple[np.ndarray, np.ndarray] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """``ik`` from the seed, then from each nudge for the poses still not converged (``fr3_kinematics.ik_retry``)."""
    lo, hi = chain.q_limits() if limits is None else limits
    n, dof = len(T_target), chain.dof
    seeds = np.array(np.broadcast_to(seed, (n, dof)), dtype=np.float64)
    rests = None if rest is None else np.array(np.broadcast_to(rest, seeds.shape), dtype=np.float64)
    q = np.zeros_like(seeds)
    ok = np.zeros(n, dtype=bool)
    for nudge in default_nudges(chain) if nudges is None else nudges:
        todo = ~ok
        if not todo.any():
            break
        q_try, ok_try = ik(
            chain,
            T_target[todo],
            np.clip(seeds[todo] + nudge, lo, hi),
            early_exit=early_exit,
            rest=None if rests is None else rests[todo],
            iters=EARLY_EXIT_ITERS if early_exit else 400,
            limits=(lo, hi),
        )
        q[todo], ok[todo] = q_try, ok_try
    return q, ok
