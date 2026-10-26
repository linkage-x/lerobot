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
Both arms' joint paths for a whole episode at once: the joint retarget (roadmap C3).

Frame-by-frame IK per arm (what the single-arm replay does) answers "can this arm reach this pose from where
it was", one frame and one arm at a time. It cannot trade a millimetre at one frame for a wall it would hit
two hundred frames later, nor keep two arms apart. This optimises both joint paths over the episode together:

    minimise  sum_t  |TCP_R(q_t) - target_R,t|^2 / sigma_p^2 + |rot error|^2 / sigma_r^2   (both arms)
            + sum_t  |q_t - q_{t-1}|^2 / sigma_step^2                                      (continuity)
            + hinges: joint-wall margin, singularity margin, arm-arm gap > d_safe, arm-environment gap > d_env

with Gauss-Newton / Levenberg-Marquardt. Every per-frame term depends on that frame's 14 joints only, so the
normal equations are block-tridiagonal (14x14 blocks) and are solved exactly in O(n). The hinges are stiff
(sigmas of a fraction of a millimetre or milliradian): a constraint is given up only when nothing else can be.
The start is chosen as the single-arm replay chooses it (``trajectory_start``): IK branches from many seeds,
each followed through the episode; the one that tracks most frames inside the walls, nearest the ready pose.

``d_safe`` is not a constant: it is the placement's claimed sigma times ``K_PLACEMENT`` plus both arms'
execution-error tails (``derive_d_safe``); the capsules enclose the meshes, so the model needs no margin of
its own.

Geometry and time are separate. The retarget produces a path; ``retime`` gives it a reference time axis that
keeps every joint under its velocity limit (scaled), slowing segments down, never speeding them up, and
``resample`` puts it on the runtime's fixed command period. The raw (recorded), reference (retimed) and,
after a run, executed time axes are all kept.

What comes out is the reference trajectory and how far it is from the demonstration: per arm the TCP change
(p50/p95/max), the change of the hands' relative pose T_LR, the smallest arm-arm and arm-environment gaps,
the smallest joint-wall and singularity margins, and a verdict with reasons (``verdict``) -- what C4 writes
into the dataset. The demonstration's labels are never changed.
"""

# ruff: noqa: N802, N803, N806  (matrix names: R, T, J follow the maths)
from __future__ import annotations

import math
import sys
import time
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import dual_fr3_collision as col, fr3_kinematics as kin  # noqa: E402

SCHEMA = "dual_fr3_retarget/v1"
#: FR3 joint velocity limits, rad/s (Franka FR3 datasheet).
FR3_VELOCITY_LIMITS = np.array([2.62, 2.62, 2.62, 2.62, 5.26, 4.18, 5.26])
#: Placement sigmas the collision margin covers.
K_PLACEMENT = 3.0
#: The arm's TCP tracking error against its command, tail: 10-10 laser replay, executed (FK, 0.27 mm from the
#: laser) against desired, 7.8 mm -- 5.5 mm sag plus 150 ms lag. Per arm; until D5 measures it on two arms.
EXECUTION_TAIL_MM = 7.8
#: The gap to the cameras / table on top of one arm's execution tail: their registration (~3 mm) and that the
#: camera balls' size is assumed.
ENV_MARGIN_MM = 10.0
#: Never plan two arms closer than this, whatever the sigmas claim.
D_SAFE_FLOOR_MM = 20.0
#: Lever for the singularity measure's rotation rows: sigma_min of [Jv; L*Jw] is in metres.
SINGULAR_LEVER_M = 0.2


def derive_d_safe(
    placement_sigma_mm: float | None,
    *,
    execution_tail_mm: tuple[float, float] = (EXECUTION_TAIL_MM, EXECUTION_TAIL_MM),
    unknown_sigma_mm: float = 20.0,
) -> dict[str, Any]:
    """``d_safe`` (arm-arm) and ``d_env`` (arm-environment), metres, with the parts they are made of."""
    sigma = unknown_sigma_mm if placement_sigma_mm is None else float(placement_sigma_mm)
    arm_arm = K_PLACEMENT * sigma + sum(execution_tail_mm)
    d_safe = max(arm_arm, D_SAFE_FLOOR_MM)
    d_env = max(execution_tail_mm) + ENV_MARGIN_MM
    return {
        "d_safe_m": d_safe / 1e3,
        "d_env_m": d_env / 1e3,
        "placement_sigma_mm": sigma,
        "placement_sigma_known": placement_sigma_mm is not None,
        "k_placement": K_PLACEMENT,
        "execution_tail_mm": list(execution_tail_mm),
        "env_margin_mm": ENV_MARGIN_MM,
        "floor_mm": D_SAFE_FLOOR_MM,
    }


@dataclass(frozen=True)
class Settings:
    sigma_position_m: float = 0.001
    sigma_rotation_deg: float = 1.0
    sigma_step_rad: float = 0.05
    #: kept inside kin.Q_LOWER/Q_UPPER (already LIMIT_MARGIN_DEG inside panda_py's walls); the walls are hard
    limit_margin_rad: float = 0.05
    sigma_limit_rad: float = 0.001
    #: sigma_min of the TCP Jacobian (rotation rows x SINGULAR_LEVER_M): random joint configurations have
    #: a median of 0.05 and a 5th percentile of 0.005; recorded episodes keep above ~0.025
    singular_min_m: float = 0.01
    sigma_singular_m: float = 0.0005
    d_safe_m: float = 0.05
    d_env_m: float = 0.018
    sigma_clearance_m: float = 0.0005
    iterations: int = 40
    #: what a retarget may change and still replay the demonstration
    max_change_position_mm: float = 20.0
    p95_change_position_mm: float = 5.0
    max_change_rotation_deg: float = 8.0
    p95_change_rotation_deg: float = 2.0
    max_t_lr_change_mm: float = 20.0
    #: a joint moving more than this between two frames is a jump to another IK branch, not a motion
    #: (trajectory_start's joint-jump guard)
    max_joint_step_rad: float = 0.35
    #: fraction of FR3_VELOCITY_LIMITS a reference trajectory may use
    velocity_scale: float = 0.5
    #: frames whose label was not solved still get a target (the held pose) but with this weight
    unseen_weight: float = 0.1


# ------------------------------------------------------------------ start ---


def track(
    arm: kin.Arm,
    flange: np.ndarray,
    starts: np.ndarray,
    *,
    iters: int = 30,
    rotation_weight_m: float = 0.001 / math.radians(1.0),
    max_frame_step_rad: float = 0.1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Follow ``flange`` (n,4,4) from each start (B,7), as frame-by-frame IK does: ``(q (B,n,7), position
    error (B,n) m, rotation error (B,n) deg)``. Joints clip at Q_LOWER/Q_UPPER, so a branch that runs into a
    wall stays at it and misses. Rotation is weighed as the optimisation weighs it (1 deg ~ 1 mm): with metres
    and radians at par, an unreachable frame is met by giving up most of the orientation. No joint moves more
    than ``max_frame_step_rad`` between frames: past the reach, IK otherwise swings the arm round (3 rad in
    one frame) after a target it cannot have, and the path that starts the optimisation is not a path."""
    B, n = len(starts), len(flange)
    q = np.array(starts, dtype=np.float64)
    Q = np.zeros((B, n, 7))
    ep, er = np.zeros((B, n)), np.zeros((B, n))
    eye6 = np.eye(6)
    W = np.array([1.0, 1.0, 1.0, rotation_weight_m, rotation_weight_m, rotation_weight_m])
    for t in range(n):
        target = np.broadcast_to(flange[t], (B, 4, 4))
        q_prev = q.copy()
        for _ in range(iters):
            T, J = arm.kinematics(q)
            e = np.concatenate(
                [
                    target[:, :3, 3] - T[:, :3, 3],
                    kin._rotvec(target[:, :3, :3] @ T[:, :3, :3].transpose(0, 2, 1)),
                ],
                axis=1,
            )
            if np.abs(e).max() < 1e-6:
                break
            e, J = e * W, J * W[:, None]
            Jt = J.transpose(0, 2, 1)
            dq = (Jt @ np.linalg.solve(J @ Jt + 1e-4 * eye6, e[:, :, None]))[:, :, 0]
            step = np.abs(dq).max(axis=1, keepdims=True)
            dq *= np.minimum(1.0, 0.2 / np.maximum(step, 1e-12))
            q = np.clip(q + dq, kin.Q_LOWER, kin.Q_UPPER)
        if t > 0:
            delta = q - q_prev
            big = np.abs(delta).max(axis=1, keepdims=True)
            q = q_prev + delta * np.minimum(1.0, max_frame_step_rad / np.maximum(big, 1e-12))
        T = arm.fk(q)
        Q[:, t] = q
        ep[:, t] = np.linalg.norm(target[:, :3, 3] - T[:, :3, 3], axis=1)
        er[:, t] = np.degrees(
            np.linalg.norm(kin._rotvec(target[:, :3, :3] @ T[:, :3, :3].transpose(0, 2, 1)), axis=1)
        )
    return Q, ep, er


def choose_start(
    arm: kin.Arm,
    flange: np.ndarray,
    *,
    random_seeds: int = 24,
    rng_seed: int = 0,
    tol_mm: float = 2.0,
    tol_deg: float = 1.0,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The joint path of the IK branch that tracks most frames inside the walls (then the largest wall margin,
    then nearest Q_READY), from frame-0 solutions found from Q_READY, its nudges and random seeds."""
    rng = np.random.default_rng(rng_seed)
    seeds = [np.clip(kin.Q_READY + nudge, kin.Q_LOWER, kin.Q_UPPER) for nudge in kin._SEED_NUDGES]
    seeds += list(rng.uniform(kin.Q_LOWER, kin.Q_UPPER, size=(random_seeds, 7)))
    q0, ok = kin.ik(arm, np.broadcast_to(flange[0], (len(seeds), 4, 4)).copy(), np.array(seeds))
    starts: list[np.ndarray] = []
    for q in q0[ok]:
        if all(np.abs(q - s).max() > 0.05 for s in starts):
            starts.append(q)
    if not starts:  # frame 0 itself out of reach: follow from the closest attempt
        T = arm.fk(q0)
        miss = np.linalg.norm(T[:, :3, 3] - flange[0, :3, 3], axis=1)
        starts = [q0[int(np.argmin(miss))]]
    Q, ep, er = track(arm, flange, np.array(starts))
    tracked = ((ep * 1e3 <= tol_mm) & (er <= tol_deg)).sum(axis=1)
    margin = np.minimum(Q - kin.Q_LOWER, kin.Q_UPPER - Q).min(axis=(1, 2))
    near = np.linalg.norm(Q[:, 0] - kin.Q_READY, axis=1)
    best = int(np.lexsort((near, -margin, -tracked))[0])
    return Q[best], {
        "branches": len(starts),
        "frames_tracked": int(tracked[best]),
        "frames": len(flange),
        "start_joints": Q[best, 0].tolist(),
    }


# ------------------------------------------------------------------ the problem ---


@dataclass
class Problem:
    targets: dict[str, np.ndarray]  # arm -> (n,4,4) TCP targets in its own base
    weights: dict[str, np.ndarray]  # arm -> (n,) tracking weight
    tool: np.ndarray  # (4,4) flange -> TCP (both arms)
    T_right_left: np.ndarray  # left base in the right base
    model: col.CapsuleModel
    obstacles: Sequence[Mapping[str, Any]]  # in the right base
    settings: Settings
    arm: kin.Arm = kin.NOMINAL

    def __post_init__(self) -> None:
        self.n = len(self.targets["right"])
        self._mask = self.model.table_mask()
        self._K = len(self.model.names)
        # obstacles in each arm's base
        T_lr_inv = np.linalg.inv(self.T_right_left)
        self._env = {
            "right": list(self.obstacles),
            "left": [_obstacle_in(ob, T_lr_inv) for ob in self.obstacles],
        }

    def arm_terms(self, side: str, q: np.ndarray) -> np.ndarray:
        """(n, 12 + 14 + 1) of one arm: tracking, joint-wall margin, singularity margin."""
        g = self.settings
        T, J = self.arm.kinematics(q, self.tool[:3, 3])
        T = T @ self.tool
        target = self.targets[side]
        w = np.sqrt(self.weights[side])[:, None]
        e_p = (T[:, :3, 3] - target[:, :3, 3]) / g.sigma_position_m
        e_r = kin._rotvec(T[:, :3, :3] @ target[:, :3, :3].transpose(0, 2, 1)) / math.radians(
            g.sigma_rotation_deg
        )
        low = np.maximum(0.0, g.limit_margin_rad - (q - kin.Q_LOWER)) / g.sigma_limit_rad
        high = np.maximum(0.0, g.limit_margin_rad - (kin.Q_UPPER - q)) / g.sigma_limit_rad
        singular = np.maximum(0.0, g.singular_min_m - singular_measure(J))[:, None] / g.sigma_singular_m
        return np.concatenate([w * e_p, w * e_r, low, high, singular], axis=1)

    def gaps(self, Q: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Surface gaps: right arm to environment (n, K*M), left arm to environment, arm to arm (n, K*K)."""
        A_r, B_r = self.model.place(self.arm, Q[:, :7])
        A_l, B_l = self.model.place(self.arm, Q[:, 7:])
        env_r = col.obstacle_gaps(self._env["right"], A_r, B_r, self.model.r, self._mask).reshape(len(Q), -1)
        env_l = col.obstacle_gaps(self._env["left"], A_l, B_l, self.model.r, self._mask).reshape(len(Q), -1)
        R, p = self.T_right_left[:3, :3], self.T_right_left[:3, 3]
        arm_arm = col.arm_arm_gaps(self.model, A_r, B_r, A_l @ R.T + p, B_l @ R.T + p).reshape(len(Q), -1)
        return env_r, env_l, arm_arm

    def clearance_terms(self, Q: np.ndarray) -> np.ndarray:
        g = self.settings
        env_r, env_l, arm_arm = self.gaps(Q)
        return np.concatenate(
            [
                np.maximum(0.0, g.d_env_m - env_r) / g.sigma_clearance_m,
                np.maximum(0.0, g.d_env_m - env_l) / g.sigma_clearance_m,
                np.maximum(0.0, g.d_safe_m - arm_arm) / g.sigma_clearance_m,
            ],
            axis=1,
        )

    def per_frame(self, Q: np.ndarray) -> np.ndarray:
        """(n, m) residuals that depend on each frame's own joints only."""
        return np.concatenate(
            [self.arm_terms("right", Q[:, :7]), self.arm_terms("left", Q[:, 7:]), self.clearance_terms(Q)],
            axis=1,
        )

    def active(self, Q: np.ndarray, buffer_m: float = 0.02) -> np.ndarray:
        """Frames where some gap is within ``buffer_m`` of its margin; elsewhere every clearance hinge and its
        derivative are zero."""
        g = self.settings
        env_r, env_l, arm_arm = self.gaps(Q)
        near_env = (
            np.minimum(env_r.min(axis=1, initial=np.inf), env_l.min(axis=1, initial=np.inf))
            < g.d_env_m + buffer_m
        )
        return near_env | (arm_arm.min(axis=1) < g.d_safe_m + buffer_m)

    def jacobian(self, Q: np.ndarray, r0: np.ndarray, h: float = 1e-6) -> np.ndarray:
        """(n, m, 14) by forward differences, all frames at once (each frame's residuals see only its joints).
        An arm's own terms are differenced over its own joints only, and the clearance terms only on the frames
        where they can be non-zero."""
        J = np.zeros((*r0.shape, 14))
        width = r0.shape[1]
        own = (width - self.clearance_width) // 2
        active = self.active(Q)
        Qa = Q[active]
        r0_clear = r0[active, 2 * own :]
        for j in range(14):
            side, cols, offset = ("right", slice(0, 7), 0) if j < 7 else ("left", slice(7, 14), own)
            q = Q[:, cols].copy()
            q[:, j % 7] += h
            J[:, offset : offset + own, j] = (self.arm_terms(side, q) - r0[:, offset : offset + own]) / h
            if active.any():
                Qh = Qa.copy()
                Qh[:, j] += h
                J[active, 2 * own :, j] = (self.clearance_terms(Qh) - r0_clear) / h
        return J

    @property
    def clearance_width(self) -> int:
        K, M = self._K, len(self.obstacles)
        return 2 * K * M + K * K


def _obstacle_in(ob: Mapping[str, Any], T: np.ndarray) -> dict[str, Any]:
    """``ob`` re-expressed by ``T`` (new <- old)."""
    out = dict(ob)
    if ob["kind"] == "halfspace":
        normal = T[:3, :3] @ np.asarray(ob["normal"], dtype=np.float64)
        normal_old = np.asarray(ob["normal"], dtype=np.float64)
        point_old = normal_old / np.linalg.norm(normal_old) * float(ob["offset"])
        out["normal"] = normal.tolist()
        out["offset"] = float((T[:3, :3] @ point_old + T[:3, 3]) @ (normal / np.linalg.norm(normal)))
    elif ob["kind"] == "sphere":
        out["centre"] = (T[:3, :3] @ np.asarray(ob["centre"]) + T[:3, 3]).tolist()
    else:
        out["a"] = (T[:3, :3] @ np.asarray(ob["a"]) + T[:3, 3]).tolist()
        out["b"] = (T[:3, :3] @ np.asarray(ob["b"]) + T[:3, 3]).tolist()
    return out


def singular_measure(J: np.ndarray) -> np.ndarray:
    """(n,) smallest singular value of the TCP Jacobian with its rotation rows times SINGULAR_LEVER_M, metres
    per radian: how fast the worst direction of TCP motion is still available."""
    Js = J.copy()
    Js[:, 3:] *= SINGULAR_LEVER_M
    return np.linalg.svd(Js, compute_uv=False)[:, -1]


def _block_tridiagonal_solve(D: np.ndarray, c: float, rhs: np.ndarray) -> np.ndarray:
    """Solve the system with diagonal blocks ``D`` (n,k,k), off-diagonal blocks ``c * I``."""
    n, k = rhs.shape
    Dp = np.empty_like(D)
    dp = np.empty_like(rhs)
    Dp[0], dp[0] = D[0], rhs[0]
    inv_prev = np.linalg.inv(Dp[0])
    for t in range(1, n):
        Dp[t] = D[t] - c * c * inv_prev
        dp[t] = rhs[t] - c * (inv_prev @ dp[t - 1])
        inv_prev = np.linalg.inv(Dp[t])
    x = np.empty_like(rhs)
    x[-1] = np.linalg.solve(Dp[-1], dp[-1])
    for t in range(n - 2, -1, -1):
        x[t] = np.linalg.solve(Dp[t], dp[t] - c * x[t + 1])
    return x


def optimise(
    problem: Problem, Q0: np.ndarray, *, log: list | None = None
) -> tuple[np.ndarray, dict[str, Any]]:
    """Levenberg-Marquardt from ``Q0`` (n,14); joints stay inside Q_LOWER/Q_UPPER (projected)."""
    g = problem.settings
    w_s = 1.0 / g.sigma_step_rad**2
    lower, upper = np.tile(kin.Q_LOWER, 2), np.tile(kin.Q_UPPER, 2)
    Q = np.clip(Q0, lower, upper)

    def cost(Q: np.ndarray, r: np.ndarray) -> float:
        return 0.5 * float((r * r).sum()) + 0.5 * w_s * float((np.diff(Q, axis=0) ** 2).sum())

    r = problem.per_frame(Q)
    f = cost(Q, r)
    lam, it = 1e-3, 0
    started = time.monotonic()
    for it in range(1, g.iterations + 1):
        J = problem.jacobian(Q, r)
        JtJ = np.einsum("nmi,nmj->nij", J, J)
        grad = np.einsum("nmi,nm->ni", J, r)
        dQ = np.diff(Q, axis=0)
        grad[1:] += w_s * dQ
        grad[:-1] -= w_s * dQ
        diag_s = np.full(len(Q), 2.0 * w_s)
        diag_s[0] = diag_s[-1] = w_s
        improved = False
        for _ in range(8):
            D = JtJ + (diag_s[:, None, None] + lam * (1.0 + np.einsum("nii->ni", JtJ))[:, :, None]) * np.eye(
                14
            )
            step = _block_tridiagonal_solve(D, -w_s, -grad)
            Q_new = np.clip(Q + step, lower, upper)
            r_new = problem.per_frame(Q_new)
            f_new = cost(Q_new, r_new)
            if f_new < f:
                improved = True
                rel = (f - f_new) / max(f, 1e-12)
                Q, r, f = Q_new, r_new, f_new
                lam = max(lam / 3.0, 1e-7)
                break
            lam *= 4.0
        if log is not None:
            log.append({"iteration": it, "cost": f, "lambda": lam})
        if not improved or rel < 1e-5 or float(np.abs(step).max()) < 1e-6:
            break
    return Q, {"iterations": it, "cost": f, "seconds": round(time.monotonic() - started, 2)}


# ------------------------------------------------------------------ time ---


def retime(Q: np.ndarray, t_raw: np.ndarray, *, speed: float, velocity_scale: float) -> np.ndarray:
    """The reference time axis: the recorded intervals stretched by 1/``speed``, and each further stretched
    until no joint of either arm exceeds ``velocity_scale`` of its limit. Never faster than ``speed``."""
    vmax = np.tile(FR3_VELOCITY_LIMITS, Q.shape[1] // 7) * velocity_scale
    nominal = np.diff(t_raw) / speed
    needed = (np.abs(np.diff(Q, axis=0)) / vmax).max(axis=1)
    return np.concatenate([[0.0], np.cumsum(np.maximum(nominal, needed))])


def resample(Q: np.ndarray, t_ref: np.ndarray, period_s: float) -> tuple[np.ndarray, np.ndarray]:
    """``Q`` on the runtime's command clock ``0, period, 2 period, ...`` (linear in time); and, for each
    command, the fractional index into the path it came from."""
    t_cmd = np.arange(0.0, t_ref[-1] + 1e-9, period_s)
    index = np.interp(t_cmd, t_ref, np.arange(len(t_ref), dtype=np.float64))
    lo = np.floor(index).astype(int)
    hi = np.minimum(lo + 1, len(Q) - 1)
    frac = (index - lo)[:, None]
    return Q[lo] * (1.0 - frac) + Q[hi] * frac, index


# ------------------------------------------------------------------ metrics ---


def _stats(values: np.ndarray) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {
        "p50": round(float(np.percentile(values, 50)), 3),
        "p95": round(float(np.percentile(values, 95)), 3),
        "max": round(float(values.max()), 3),
    }


def metrics(problem: Problem, Q: np.ndarray) -> dict[str, Any]:
    g = problem.settings
    out: dict[str, Any] = {"arms": {}}
    achieved = {}
    for side, cols in (("right", slice(0, 7)), ("left", slice(7, 14))):
        q = Q[:, cols]
        T, J = problem.arm.kinematics(q, problem.tool[:3, 3])
        T = T @ problem.tool
        achieved[side] = T
        target = problem.targets[side]
        dp = np.linalg.norm(T[:, :3, 3] - target[:, :3, 3], axis=1) * 1e3
        dr = np.degrees(
            np.linalg.norm(kin._rotvec(T[:, :3, :3] @ target[:, :3, :3].transpose(0, 2, 1)), axis=1)
        )
        margins = np.minimum(q - kin.Q_LOWER, kin.Q_UPPER - q)
        frame, joint = np.unravel_index(int(np.argmin(margins)), margins.shape)
        s = singular_measure(J)
        out["arms"][side] = {
            "change_position_mm": _stats(dp),
            "change_rotation_deg": _stats(dr),
            "worst_change_frame": int(np.argmax(dp)),
            "min_limit_margin_rad": round(float(margins[frame, joint]), 4),
            "min_limit_margin_joint": int(joint) + 1,
            "min_limit_margin_frame": int(frame),
            "min_singular_m": round(float(s.min()), 4),
            "min_singular_frame": int(np.argmin(s)),
            "max_joint_step_rad": round(float(np.abs(np.diff(q, axis=0)).max()) if len(q) > 1 else 0.0, 4),
        }
    # the hands' relation: their position difference in the right base, and their relative rotation. Not the
    # translation of inv(T_L) T_R -- that is read in the left TCP's frame, where a fraction of a degree at the
    # left wrist times the ~0.8 m between the hands reads as millimetres of change (A1's lesson).
    left_in_right = problem.T_right_left @ achieved["left"]
    target_left_in_right = problem.T_right_left @ problem.targets["left"]
    d_achieved = achieved["right"][:, :3, 3] - left_in_right[:, :3, 3]
    d_target = problem.targets["right"][:, :3, 3] - target_left_in_right[:, :3, 3]
    out["t_lr_change_position_mm"] = _stats(np.linalg.norm(d_achieved - d_target, axis=1) * 1e3)
    R_lr = left_in_right[:, :3, :3].transpose(0, 2, 1) @ achieved["right"][:, :3, :3]
    R_lr_target = target_left_in_right[:, :3, :3].transpose(0, 2, 1) @ problem.targets["right"][:, :3, :3]
    out["t_lr_change_rotation_deg"] = _stats(
        np.degrees(np.linalg.norm(kin._rotvec(R_lr @ R_lr_target.transpose(0, 2, 1)), axis=1))
    )
    c = col.clearance(problem.model, Q[:, :7], Q[:, 7:], problem.T_right_left, problem.obstacles, problem.arm)
    k = int(np.argmin(c["arm_arm_m"]))
    out["min_arm_arm_gap_mm"] = round(float(c["arm_arm_m"][k]) * 1e3, 1)
    out["min_arm_arm_gap_frame"] = k
    out["min_arm_arm_gap_capsules"] = list(c["arm_arm_pair"][k])
    for side in ("right", "left"):
        env = c[f"env_{side}_m"]
        j = int(np.argmin(env))
        out["arms"][side]["min_env_gap_mm"] = (
            None if not np.isfinite(env[j]) else round(float(env[j]) * 1e3, 1)
        )
        out["arms"][side]["min_env_gap_with"] = (
            None if not np.isfinite(env[j]) else list(c[f"env_{side}_with"][j])
        )
        out["arms"][side]["min_env_gap_frame"] = j
    out["d_safe_mm"] = round(g.d_safe_m * 1e3, 1)
    out["d_env_mm"] = round(g.d_env_m * 1e3, 1)
    return out


def verdict(m: Mapping[str, Any], g: Settings) -> tuple[bool, list[str]]:
    """Executable, and every reason it is not."""
    reasons = []
    tol = 0.5  # mm: the optimum sits on a hinge's edge, a hair inside it
    if m["min_arm_arm_gap_mm"] < g.d_safe_m * 1e3 - tol:
        reasons.append(
            f"arms {m['min_arm_arm_gap_mm']:.0f} mm apart at frame {m['min_arm_arm_gap_frame']} "
            f"({'/'.join(m['min_arm_arm_gap_capsules'])}), d_safe {g.d_safe_m * 1e3:.0f} mm"
        )
    for side, row in m["arms"].items():
        if row["min_env_gap_mm"] is not None and row["min_env_gap_mm"] < g.d_env_m * 1e3 - tol:
            reasons.append(
                f"{side}: {row['min_env_gap_mm']:.0f} mm from {'/'.join(row['min_env_gap_with'])} at frame "
                f"{row['min_env_gap_frame']}, d_env {g.d_env_m * 1e3:.0f} mm"
            )
        cp, cr = row["change_position_mm"], row["change_rotation_deg"]
        if cp["max"] > g.max_change_position_mm or cp["p95"] > g.p95_change_position_mm:
            reasons.append(
                f"{side}: retarget moves the TCP p95 {cp['p95']:.1f} / max {cp['max']:.1f} mm (frame "
                f"{row['worst_change_frame']}; allowed {g.p95_change_position_mm:.0f} / {g.max_change_position_mm:.0f})"
            )
        if cr["max"] > g.max_change_rotation_deg or cr["p95"] > g.p95_change_rotation_deg:
            reasons.append(
                f"{side}: retarget turns the TCP p95 {cr['p95']:.1f} / max {cr['max']:.1f} deg (allowed "
                f"{g.p95_change_rotation_deg:.0f} / {g.max_change_rotation_deg:.0f})"
            )
        if row["min_limit_margin_rad"] < 1e-6:
            reasons.append(
                f"{side}: j{row['min_limit_margin_joint']} at its wall at frame {row['min_limit_margin_frame']}"
            )
        if row["max_joint_step_rad"] > g.max_joint_step_rad:
            reasons.append(
                f"{side}: the joint path jumps {row['max_joint_step_rad']:.2f} rad between two frames (another IK branch)"
            )
        if row["min_singular_m"] < 0.5 * g.singular_min_m:
            reasons.append(f"{side}: near a singularity at frame {row['min_singular_frame']}")
    if m["t_lr_change_position_mm"]["max"] > g.max_t_lr_change_mm:
        reasons.append(
            f"the hands' relative position changes by up to {m['t_lr_change_position_mm']['max']:.1f} mm "
            f"(allowed {g.max_t_lr_change_mm:.0f})"
        )
    return not reasons, reasons


# ------------------------------------------------------------------ all of it ---


def retarget(
    targets: Mapping[str, np.ndarray],
    seen: Mapping[str, np.ndarray],
    *,
    tool: np.ndarray,
    T_right_left: np.ndarray,
    obstacles: Sequence[Mapping[str, Any]],
    t_raw: np.ndarray,
    speed: float,
    settings: Settings,
    model: col.CapsuleModel | None = None,
) -> dict[str, Any]:
    """C3 for one episode: start, joint optimisation, metrics, verdict, time axes. Returns the report (JSON-
    able) and, under ``"arrays"``, the reference path and time axes."""
    model = model or col.load()
    tool = np.asarray(tool, dtype=np.float64)
    tool_inv = np.linalg.inv(tool)
    starts, Q0 = {}, []
    for side in ("right", "left"):
        q, info = choose_start(kin.NOMINAL, np.asarray(targets[side]) @ tool_inv)
        starts[side] = info
        Q0.append(q)
    weights = {
        side: np.where(np.asarray(seen[side]), 1.0, settings.unseen_weight) for side in ("right", "left")
    }
    problem = Problem(
        targets={side: np.asarray(targets[side], dtype=np.float64) for side in ("right", "left")},
        weights=weights,
        tool=tool,
        T_right_left=np.asarray(T_right_left, dtype=np.float64),
        model=model,
        obstacles=obstacles,
        settings=settings,
    )
    Q0 = np.concatenate(Q0, axis=1)
    before = metrics(problem, Q0)
    Q, solve = optimise(problem, Q0)
    after = metrics(problem, Q)
    ok, reasons = verdict(after, settings)
    t_ref = retime(Q, t_raw, speed=speed, velocity_scale=settings.velocity_scale)
    nominal = float(t_raw[-1] - t_raw[0]) / speed
    return {
        "schema": SCHEMA,
        "executable": ok,
        "reasons": reasons,
        "settings": asdict(settings),
        "capsule_model_sha256": model.fingerprint,
        "start": starts,
        "solve": solve,
        "frame_by_frame_ik": before,
        "retarget": after,
        "timing": {
            "speed": speed,
            "raw_duration_s": round(float(t_raw[-1] - t_raw[0]), 3),
            "nominal_duration_s": round(nominal, 3),
            "reference_duration_s": round(float(t_ref[-1]), 3),
            "slowdown_vs_nominal": round(float(t_ref[-1]) / nominal, 3) if nominal > 0 else None,
            "velocity_scale": settings.velocity_scale,
        },
        "arrays": {
            "q": Q,
            "q_frame_by_frame": Q0,
            "t_raw": np.asarray(t_raw, dtype=np.float64),
            "t_ref": t_ref,
        },
    }
