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
Two hands on one rigid object: what the arms' set points squeeze into it, and how soft the arms must be
(roadmap D6).

While both grippers hold one object, the two arms and the object are one closed chain. Each arm is a spring
round its set point (panda_py's ``JointPosition``: tau = K (q_d - q) + D (dq_d - dq), on libfranka's gravity
compensation), so whatever the set points ask that the object does not allow -- the left hand anywhere but
where the clamped object puts it -- is taken up by the two springs in series, as a force inside the object.
The object is not told; the grippers slip, or the arms reflex.

What the set points ask that the object does not allow (``relative_mismatch``), per held frame, against the
frame the grippers closed on (the object is clamped where the hands are then, so a constant offset costs
nothing):

* **placement** -- the plan is rigid in the *assumed* T_right_left. With the left base off by eps, the left
  hand's motion is the right's conjugated by eps: a translation error turns into ``(I - R_obj) p_eps`` as the
  object turns, a rotation error into ``eps_R x`` the object's travel. Sampled per scenario (``PLACEMENTS``).
* **skew** -- one arm's set points ``skew_s`` behind the other's (D2's execution skew, and the two
  controllers' lag differing): ``skew_s * v`` of the hand.
* **the plan itself** -- a retarget that bent T_LR (C3's ``t_lr_change``); in with eps = identity.
* **measured** (``from_run``) -- the two arms' read-back joints of a run, which is the same sum as it really
  happened, the placement excepted (the run reads both arms in the assumed geometry). A gripperless free-space
  run of a held plan measures it without any object.

How hard it squeezes (``internal_wrench``): the mismatch twist at the hands' midpoint over the two arms'
compliances in series, ``w = (C_R + C_L + C_grasp)^-1 delta``, each arm's ``C = J K^-1 J^T`` moved to the
midpoint. Scaling the joint stiffness by s scales w by s (without grasp compliance) -- and the object's sag
under its own weight (the two arms in parallel) by 1/s, and the friction deadband (a joint does not move
until its spring beats ~1 Nm of friction: ``JOINT_FRICTION_NM / (s K)``) by 1/s. ``budget`` scans s for the
window where all three hold, and, where there is none, how soft the grasp itself must be.

All of it is a model: rigid links, nominal kinematics, a constant friction, a quasi-static object. It says
which way and roughly how far; the guard in the runtime (``dual_fr3_replay_runtime``: the internal force
from both arms' external torques, stops both) is what protects the object.
"""

# ruff: noqa: N802, N803, N806  (matrix names: R, T, J, C follow the maths)
from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import fr3_kinematics as kin  # noqa: E402

SCHEMA = "dual_fr3_closed_chain/v1"

#: panda_py's JointPosition defaults (src/controllers/joint_position.cpp): what D1 drives the arms with.
JOINT_STIFFNESS = np.array([600.0, 600.0, 600.0, 600.0, 250.0, 150.0, 50.0])
JOINT_DAMPING = np.array([50.0, 50.0, 50.0, 20.0, 20.0, 20.0, 10.0])

#: What a held object may take before the plan is not fit to run: inside the object, in N and Nm. A grip
#: slips at mu * grip force; the gripper's is not measured yet, so these are the budget, not its limit.
INTERNAL_FORCE_BUDGET_N = 10.0
INTERNAL_TORQUE_BUDGET_NM = 1.0
#: the object under its own weight, held by both arms at their scaled stiffness
SAG_BUDGET_MM = 10.0
#: a set point a joint's friction keeps it from: 10-19's pivot runs, 1 Nm leaves 0.4 deg on j6, 1.1 on j7
JOINT_FRICTION_NM = 1.0
FRICTION_DEADBAND_BUDGET_MM = 5.0
OBJECT_MASS_KG = 1.0
GRAVITY = 9.81
#: (translation sigma per axis mm, rotation sigma per axis deg) of the left base against the right:
#: by tape measure (C3's unknown 20 mm, a degree), and after B3's arm-to-arm sphere (G3's ~3 mm)
PLACEMENTS: dict[str, tuple[float, float]] = {"assumed": (20.0, 1.0), "measured_b3": (3.0, 0.2)}
#: one arm's set points behind the other's: D2's start skew is under a millisecond on the fakes, the two
#: controllers' tracking lags differ by more -- a guess until B1's runs measure it
SKEW_S = 0.010
SAMPLES = 200
#: the joint stiffness scales ``budget`` tries, stiffest first
STIFFNESS_SCALES = (1.0, 0.7, 0.5, 0.35, 0.25, 0.15, 0.1)
#: grasp stiffnesses tried when no joint scale fits (N/m, per hand, translation); its rotation is the
#: translation's over a pad pair GRASP_LEVER_M apart (k * a^2)
GRASP_STIFFNESSES_N_M = (20000.0, 10000.0, 5000.0, 2000.0, 1000.0, 500.0)
GRASP_LEVER_M = 0.03


# ------------------------------------------------------------------ the plan ---

#: what a held plan npz carries (``dual_fr3_coop_carry`` writes one): joints per arm on the command clock,
#: which frames both hands hold the object, the geometry it was planned in, its rate
PLAN_KEYS = ("q_right", "q_left", "held", "T_right_left", "tool", "fps")


def read_plan(path: Path | str) -> dict[str, Any]:
    with np.load(path) as data:
        missing = [k for k in PLAN_KEYS if k not in data.files]
        if missing:
            raise ValueError(f"{path}: not a held plan, no {', '.join(missing)}")
        plan = {k: np.asarray(data[k]) for k in data.files}
    plan["held"] = plan["held"].astype(bool)
    plan["fps"] = float(plan["fps"])
    return plan


def held_segments(held: np.ndarray) -> list[tuple[int, int]]:
    """``[start, stop)`` of each run of held frames."""
    h = np.r_[False, np.asarray(held, dtype=bool), False]
    edges = np.flatnonzero(np.diff(h.astype(np.int8)))
    return [(int(a), int(b)) for a, b in zip(edges[::2], edges[1::2], strict=True)]


def compliance_profile(held: np.ndarray, scale: float, ramp_frames: int) -> np.ndarray:
    """Per frame, the factor on the arms' joint stiffness: ``scale`` through every held segment, 1 away from
    them, and a linear ramp of ``ramp_frames`` frames on each side -- down *before* the first held frame, so
    the arms are soft by the time the fingers seat (they centre the hand on the object), up after the last."""
    held = np.asarray(held, dtype=bool)
    out = np.ones(len(held))
    if scale >= 1.0:
        return out
    for a, b in held_segments(held):
        out[a:b] = scale
        for i in range(1, ramp_frames + 1):
            value = scale + (1.0 - scale) * i / (ramp_frames + 1)
            if a - i >= 0:
                out[a - i] = min(out[a - i], value)
            if b - 1 + i < len(out):
                out[b - 1 + i] = min(out[b - 1 + i], value)
    return out


# ------------------------------------------------------------------ geometry ---


def _skew(v: np.ndarray) -> np.ndarray:
    v = np.asarray(v, dtype=np.float64)
    S = np.zeros((*v.shape[:-1], 3, 3))
    S[..., 0, 1], S[..., 0, 2] = -v[..., 2], v[..., 1]
    S[..., 1, 0], S[..., 1, 2] = v[..., 2], -v[..., 0]
    S[..., 2, 0], S[..., 2, 1] = -v[..., 1], v[..., 0]
    return S


def _rotation(w: np.ndarray) -> np.ndarray:
    """Rotation matrices of (n, 3) rotation vectors."""
    w = np.atleast_2d(np.asarray(w, dtype=np.float64))
    th = np.linalg.norm(w, axis=1)
    K = _skew(w / np.where(th > 1e-12, th, 1.0)[:, None])
    s, c = np.sin(th)[:, None, None], (1.0 - np.cos(th))[:, None, None]
    return np.eye(3) + s * K + c * (K @ K)


def _blockdiag(R: np.ndarray) -> np.ndarray:
    B = np.zeros((*R.shape[:-2], 6, 6))
    B[..., :3, :3] = R
    B[..., 3:, 3:] = R
    return B


def _at_point(C: np.ndarray, r: np.ndarray) -> np.ndarray:
    """A compliance (twist per wrench) at a point p, as seen at p + r: the wrench there carries ``r x f``
    more moment at p, and a turn at p moves the point by ``-r x dtheta`` more."""
    A = np.broadcast_to(np.eye(6), C.shape).copy()
    A[..., :3, 3:] = -_skew(r)
    return A @ C @ np.swapaxes(A, -1, -2)


def tcp_compliance(
    q: np.ndarray, tool: np.ndarray, joint_stiffness: np.ndarray = JOINT_STIFFNESS
) -> np.ndarray:
    """(n, 6, 6) ``J K^-1 J^T`` at the TCP, in the arm's base axes: twist (m, rad) per wrench (N, Nm)."""
    _, J = kin.NOMINAL.kinematics(q, point=np.asarray(tool)[:3, 3])
    return J @ (J.transpose(0, 2, 1) / np.asarray(joint_stiffness, dtype=np.float64)[None, :, None])


def relative_mismatch(
    q_right: np.ndarray,
    q_left: np.ndarray,
    held: np.ndarray,
    T_right_left: np.ndarray,
    tool: np.ndarray,
    *,
    T_true: np.ndarray | None = None,
    left_delay_frames: float = 0.0,
) -> dict[str, np.ndarray]:
    """Per held frame, the twist (at the hands' midpoint, in the right base) by which the left set point is off
    where the object, clamped on the segment's first frame, holds it -- the right hand taken as where it is
    asked to be (only the relative part squeezes). ``T_true`` is the left base as it really is (default: as
    planned); ``left_delay_frames`` streams the left that many frames late (a skew)."""
    q_right, q_left = np.asarray(q_right, np.float64), np.asarray(q_left, np.float64)
    tool = np.asarray(tool, np.float64)
    n = len(q_right)
    if left_delay_frames:
        k = np.clip(np.arange(n) - left_delay_frames, 0, n - 1)
        q_left = np.stack([np.interp(k, np.arange(n), q_left[:, j]) for j in range(7)], axis=1)
    T_R = kin.NOMINAL.fk(q_right) @ tool
    T_L = np.asarray(T_right_left if T_true is None else T_true) @ kin.NOMINAL.fk(q_left) @ tool
    frames, segment, twist, mid = [], [], [], []
    for s, (a, b) in enumerate(held_segments(held)):
        M_g = np.linalg.inv(T_R[a]) @ T_L[a]
        required = T_R[a:b] @ M_g
        D = T_L[a:b] @ np.linalg.inv(required)
        c = 0.5 * (T_R[a:b, :3, 3] + required[:, :3, 3])
        moved = np.einsum("nij,nj->ni", D[:, :3, :3], c) + D[:, :3, 3] - c
        twist.append(np.concatenate([moved, kin._rotvec(D[:, :3, :3])], axis=1))
        mid.append(c)
        frames.append(np.arange(a, b))
        segment.append(np.full(b - a, s))
    if not frames:
        empty = np.zeros((0, 6))
        return {
            "frames": np.zeros(0, int),
            "segment": np.zeros(0, int),
            "twist": empty,
            "midpoint": empty[:, :3],
        }
    return {
        "frames": np.concatenate(frames),
        "segment": np.concatenate(segment),
        "twist": np.concatenate(twist),
        "midpoint": np.concatenate(mid),
    }


#: a grasp: a stiffness (N/m, the same every way; its rotation k a^2 over ``GRASP_LEVER_M``), a (6, 6) compliance
#: at the TCP in the TCP's own axes (a designed part: ``dual_fr3_compliant_handle``), or None for rigid
Grasp = float | np.ndarray | None


def grasp_compliance(T_tcp: np.ndarray, grasp: Grasp, lever_m: float = GRASP_LEVER_M) -> np.ndarray:
    """(n, 6, 6) one hand's grasp as a spring at its TCP, in the base axes; zero for a rigid grasp."""
    n = len(T_tcp)
    if grasp is None:
        return np.zeros((n, 6, 6))
    if np.ndim(grasp) == 0:
        k = float(grasp)
        if not k or not math.isfinite(k):
            return np.zeros((n, 6, 6))
        local = np.diag([1.0 / k] * 3 + [1.0 / (k * lever_m**2)] * 3)
    else:
        local = np.asarray(grasp, dtype=np.float64)
        if local.shape != (6, 6):
            raise ValueError(f"a grasp compliance is (6, 6), not {local.shape}")
    B = _blockdiag(np.asarray(T_tcp)[:, :3, :3])
    return B @ local @ np.swapaxes(B, -1, -2)


class ClosedChain:
    """The two arms' compliances on a held plan's frames, at the hands' midpoint, in the right base -- once,
    so that scales and grasps are a solve each."""

    def __init__(
        self,
        q_right: np.ndarray,
        q_left: np.ndarray,
        frames: np.ndarray,
        midpoint: np.ndarray,
        T_right_left: np.ndarray,
        tool: np.ndarray,
        joint_stiffness: np.ndarray = JOINT_STIFFNESS,
    ):
        tool = np.asarray(tool, np.float64)
        qr, ql = np.asarray(q_right)[frames], np.asarray(q_left)[frames]
        T_RL = np.asarray(T_right_left, np.float64)
        T_R = kin.NOMINAL.fk(qr) @ tool
        T_L = T_RL @ kin.NOMINAL.fk(ql) @ tool
        C_R = tcp_compliance(qr, tool, joint_stiffness)
        C_L = (
            _blockdiag(T_RL[:3, :3]) @ tcp_compliance(ql, tool, joint_stiffness) @ _blockdiag(T_RL[:3, :3]).T
        )
        self.T_R, self.T_L = T_R, T_L
        self.C_R = _at_point(C_R, np.asarray(midpoint) - T_R[:, :3, 3])
        self.C_L = _at_point(C_L, np.asarray(midpoint) - T_L[:, :3, 3])
        self.midpoint = np.asarray(midpoint)
        # the friction deadband at scale 1: each joint JOINT_FRICTION_NM / K short, at the TCP (RSS)
        deadband = []
        for q, T_tool in ((qr, tool), (ql, tool)):
            _, J = kin.NOMINAL.kinematics(q, point=T_tool[:3, 3])
            per_joint = np.linalg.norm(J[:, :3, :], axis=1) * (
                JOINT_FRICTION_NM / np.asarray(joint_stiffness)
            )
            deadband.append(np.sqrt((per_joint**2).sum(axis=1)))
        self.deadband_m = np.maximum(*deadband)

    def _grasp(self, grasp: Grasp) -> tuple[np.ndarray, np.ndarray]:
        g_R = _at_point(grasp_compliance(self.T_R, grasp), self.midpoint - self.T_R[:, :3, 3])
        g_L = _at_point(grasp_compliance(self.T_L, grasp), self.midpoint - self.T_L[:, :3, 3])
        return g_R, g_L

    def internal_wrench(self, twist: np.ndarray, scale: float = 1.0, grasp: Grasp = None) -> np.ndarray:
        """(..., n, 6) force (N) and moment (Nm) inside the object for mismatch twists (..., n, 6)."""
        g_R, g_L = self._grasp(grasp)
        C = (self.C_R + self.C_L) / scale + g_R + g_L
        return np.linalg.solve(C, np.asarray(twist)[..., None])[..., 0]

    def grasp_moments(self, wrench: np.ndarray) -> np.ndarray:
        """(..., n) the larger of the two hands' moments (Nm) for internal wrenches at the midpoint: what each
        grip has to hold, the force's lever to the midpoint included."""
        f, m = np.asarray(wrench)[..., :3], np.asarray(wrench)[..., 3:]
        out = [
            np.linalg.norm(m + np.cross(self.midpoint - T[:, :3, 3], f), axis=-1)
            for T in (self.T_R, self.T_L)
        ]
        return np.maximum(*out)

    def grasp_give(self, wrench: np.ndarray, grasp: Grasp) -> np.ndarray:
        """(..., n, 2) the larger of the two grasps' give under internal wrenches at the midpoint: how far
        (m) and how much (rad) each hand's grasp spring is deflected -- what a handle's stops must allow."""
        f, m = np.asarray(wrench)[..., :3], np.asarray(wrench)[..., 3:]
        out = []
        for T in (self.T_R, self.T_L):
            at_grip = np.concatenate([f, m + np.cross(self.midpoint - T[:, :3, 3], f)], axis=-1)
            d = np.einsum("nij,...nj->...ni", grasp_compliance(T, grasp), at_grip)
            out.append(
                np.stack([np.linalg.norm(d[..., :3], axis=-1), np.linalg.norm(d[..., 3:], axis=-1)], -1)
            )
        return np.maximum(*out)

    def sag_mm(self, mass_kg: float, scale: float = 1.0, grasp: Grasp = None) -> np.ndarray:
        """(n,) how far the object's weight pushes the midpoint down with both arms holding it (in parallel)."""
        g_R, g_L = self._grasp(grasp)
        K = np.linalg.inv(self.C_R / scale + g_R) + np.linalg.inv(self.C_L / scale + g_L)
        weight = np.array([0.0, 0.0, -mass_kg * GRAVITY, 0.0, 0.0, 0.0])
        return (
            np.linalg.norm(
                np.linalg.solve(K, np.broadcast_to(weight, (len(K), 6))[..., None])[:, :3, 0], axis=1
            )
            * 1e3
        )


# ------------------------------------------------------------------ the budget ---


def _placement_errors(sigma_mm: float, sigma_deg: float, samples: int, seed: int) -> np.ndarray:
    """(samples, 4, 4) placement errors of the left base, in its own frame."""
    rng = np.random.default_rng(seed)
    E = np.broadcast_to(np.eye(4), (samples, 4, 4)).copy()
    E[:, :3, :3] = _rotation(rng.normal(0.0, math.radians(sigma_deg), (samples, 3)))
    E[:, :3, 3] = rng.normal(0.0, sigma_mm / 1e3, (samples, 3))
    return E


def _peaks(twists: np.ndarray, chain: ClosedChain, scale: float, grasp: Grasp) -> dict[str, np.ndarray]:
    """Per sample, the worst held frame: mismatch (mm, deg), internal force/moment (N, Nm), and with a grasp
    spring its give (mm, deg)."""
    w = chain.internal_wrench(twists, scale, grasp)
    out = {
        "mismatch_mm": np.linalg.norm(twists[..., :3], axis=-1).max(axis=-1) * 1e3,
        "mismatch_deg": np.degrees(np.linalg.norm(twists[..., 3:], axis=-1).max(axis=-1)),
        "force_n": np.linalg.norm(w[..., :3], axis=-1).max(axis=-1),
        "moment_nm": chain.grasp_moments(w).max(axis=-1),
    }
    if grasp is not None:
        give = chain.grasp_give(w, grasp).max(axis=-2)
        out["give_mm"], out["give_deg"] = give[..., 0] * 1e3, np.degrees(give[..., 1])
    return out


def _p95(x: np.ndarray) -> float:
    return round(float(np.percentile(x, 95)), 2)


def budget(
    plan: Mapping[str, Any],
    *,
    placements: Mapping[str, tuple[float, float]] = PLACEMENTS,
    skew_s: float = SKEW_S,
    mass_kg: float = OBJECT_MASS_KG,
    samples: int = SAMPLES,
    seed: int = 0,
    scales: tuple[float, ...] = STIFFNESS_SCALES,
    force_budget_n: float = INTERNAL_FORCE_BUDGET_N,
    torque_budget_nm: float = INTERNAL_TORQUE_BUDGET_NM,
    sag_budget_mm: float = SAG_BUDGET_MM,
    deadband_budget_mm: float = FRICTION_DEADBAND_BUDGET_MM,
    designs: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, Any]:
    """Per placement scenario: the internal force over the held frames (the peak per sampled placement, p95
    over the samples), at each joint stiffness scale, with the sag and friction deadband that scale costs; the
    stiffest scale inside every budget, or -- when there is none -- the softest grasp that would be needed.

    ``designs`` are grasp parts to check as well, at the arms' full stiffness: name -> ``{"compliance": (6, 6)
    at the TCP in its axes, "travel_mm": the give its stops allow, "travel_deg": the turn (optional)}``
    (``dual_fr3_compliant_handle``). One fits when force, moment and sag are inside the budgets and its give
    inside its travel -- past that it is rigid again."""
    T_RL, tool = np.asarray(plan["T_right_left"]), np.asarray(plan["tool"])
    q_r, q_l, held = plan["q_right"], plan["q_left"], plan["held"]
    nominal = relative_mismatch(q_r, q_l, held, T_RL, tool)
    if not len(nominal["frames"]):
        return {"schema": SCHEMA, "held_frames": 0, "verdict": "no held frames: nothing closes the chain"}
    chain = ClosedChain(q_r, q_l, nominal["frames"], nominal["midpoint"], T_RL, tool)
    delay = skew_s * float(plan["fps"])
    sag1 = float(chain.sag_mm(mass_kg).max())
    deadband1 = float(chain.deadband_m.max() * 1e3)
    out: dict[str, Any] = {
        "schema": SCHEMA,
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "held_frames": len(nominal["frames"]),
        "held_segments": [list(s) for s in held_segments(held)],
        "model": {
            "joint_stiffness": JOINT_STIFFNESS.tolist(),
            "joint_friction_nm": JOINT_FRICTION_NM,
            "skew_s": skew_s,
            "object_mass_kg": mass_kg,
            "samples": samples,
            "budgets": {
                "force_n": force_budget_n,
                "torque_nm": torque_budget_nm,
                "sag_mm": sag_budget_mm,
                "deadband_mm": deadband_budget_mm,
            },
        },
        "plan_only": {
            "mismatch_mm_max": round(float(np.linalg.norm(nominal["twist"][:, :3], axis=1).max() * 1e3), 3),
            "mismatch_deg_max": round(
                float(np.degrees(np.linalg.norm(nominal["twist"][:, 3:], axis=1).max())), 4
            ),
        },
        "scenarios": {},
    }
    for name, (sigma_mm, sigma_deg) in placements.items():
        E = _placement_errors(sigma_mm, sigma_deg, samples, seed)
        signs = np.where(np.random.default_rng(seed + 1).random(samples) < 0.5, -1.0, 1.0)
        twists = np.stack(
            [
                relative_mismatch(q_r, q_l, held, T_RL, tool, T_true=T_RL @ e, left_delay_frames=s * delay)[
                    "twist"
                ]
                for e, s in zip(E, signs, strict=True)
            ]
        )
        rows = []
        for scale in scales:
            peaks = _peaks(twists, chain, scale, None)
            row = {
                "scale": scale,
                "force_n_p95": _p95(peaks["force_n"]),
                "moment_nm_p95": _p95(peaks["moment_nm"]),
                "sag_mm": round(sag1 / scale, 2),
                "deadband_mm": round(deadband1 / scale, 2),
            }
            row["ok"] = bool(
                row["force_n_p95"] <= force_budget_n
                and row["moment_nm_p95"] <= torque_budget_nm
                and row["sag_mm"] <= sag_budget_mm
                and row["deadband_mm"] <= deadband_budget_mm
            )
            rows.append(row)
        peaks1 = _peaks(twists, chain, 1.0, None)
        scenario: dict[str, Any] = {
            "placement_sigma": {"mm": sigma_mm, "deg": sigma_deg},
            "mismatch_mm_p95": _p95(peaks1["mismatch_mm"]),
            "mismatch_deg_p95": _p95(peaks1["mismatch_deg"]),
            "scales": rows,
        }
        if designs:
            scenario["designs"] = []
            for label, part in designs.items():
                C = np.asarray(part["compliance"], dtype=np.float64)
                peaks = _peaks(twists, chain, 1.0, C)
                row = {
                    "design": label,
                    "force_n_p95": _p95(peaks["force_n"]),
                    "moment_nm_p95": _p95(peaks["moment_nm"]),
                    "sag_mm": round(float(chain.sag_mm(mass_kg, 1.0, C).max()), 2),
                    "give_mm_p95": _p95(peaks["give_mm"]),
                    "give_deg_p95": _p95(peaks["give_deg"]),
                    "travel_mm": float(part["travel_mm"]),
                    "travel_deg": float(part.get("travel_deg", math.inf)),
                }
                row["ok"] = bool(
                    row["force_n_p95"] <= force_budget_n
                    and row["moment_nm_p95"] <= torque_budget_nm
                    and row["sag_mm"] <= sag_budget_mm
                    and row["give_mm_p95"] <= row["travel_mm"]
                    and row["give_deg_p95"] <= row["travel_deg"]
                )
                scenario["designs"].append(row)
        fit = next((r for r in rows if r["ok"]), None)
        if fit is not None:
            scenario["verdict"] = f"joint stiffness x{fit['scale']:g} through the held frames"
            scenario["scale"] = fit["scale"]
        else:
            grasps = []
            for k in GRASP_STIFFNESSES_N_M:
                peaks = _peaks(twists, chain, 1.0, k)
                grasps.append(
                    {
                        "grasp_n_m": k,
                        "grasp_nm_rad": round(k * GRASP_LEVER_M**2, 2),
                        "force_n_p95": _p95(peaks["force_n"]),
                        "moment_nm_p95": _p95(peaks["moment_nm"]),
                        "sag_mm": round(float(chain.sag_mm(mass_kg, 1.0, k).max()), 2),
                    }
                )
            scenario["grasps"] = grasps
            soft = next(
                (
                    g
                    for g in grasps
                    if g["force_n_p95"] <= force_budget_n
                    and g["moment_nm_p95"] <= torque_budget_nm
                    and g["sag_mm"] <= sag_budget_mm
                ),
                None,
            )
            scenario["verdict"] = (
                f"no joint stiffness fits; a compliant grasp of {soft['grasp_n_m']:g} N/m and "
                f"{soft['grasp_nm_rad']:g} Nm/rad per hand would"
                if soft
                else "no joint stiffness fits, nor a grasp down to "
                f"{GRASP_STIFFNESSES_N_M[-1]:g} N/m: the placement has to be measured first (B3)"
            )
        out["scenarios"][name] = scenario
    return out


def from_run(
    q_right: np.ndarray,
    q_left: np.ndarray,
    held: np.ndarray,
    T_right_left: np.ndarray,
    tool: np.ndarray,
    *,
    scale: float = 1.0,
) -> dict[str, Any]:
    """A run's read-back joints of a held plan (``dual_fr3_replay_runtime``'s ``state_q``, one row per frame):
    the mismatch the two arms really made against the frame each held segment began on, and the internal
    force it would put in a rigid object at the stiffness the run used. Without an object (a gripperless
    free-space run) that is the execution's share of the squeeze, measured; the placement's is not in it."""
    n = min(len(q_right), len(q_left), len(held))
    if n == 0:
        return {"held_frames": 0}
    held = np.asarray(held[:n], dtype=bool)
    m = relative_mismatch(q_right[:n], q_left[:n], held, T_right_left, tool)
    if not len(m["frames"]):
        return {"held_frames": 0}
    chain = ClosedChain(q_right[:n], q_left[:n], m["frames"], m["midpoint"], T_right_left, tool)
    w = chain.internal_wrench(m["twist"], scale)
    per_segment = []
    for s, (a, b) in enumerate(held_segments(held)):
        sel = m["segment"] == s
        per_segment.append(
            {
                "frames": [a, b],
                "mismatch_mm_max": round(float(np.linalg.norm(m["twist"][sel, :3], axis=1).max() * 1e3), 2),
                "mismatch_deg_max": round(
                    float(np.degrees(np.linalg.norm(m["twist"][sel, 3:], axis=1).max())), 3
                ),
                "force_n_max": round(float(np.linalg.norm(w[sel, :3], axis=1).max()), 2),
                "moment_nm_max": round(float(chain.grasp_moments(w[sel]).max()), 3),
            }
        )
    return {"held_frames": int(held.sum()), "scale": scale, "segments": per_segment}


def _summary_lines(report: Mapping[str, Any]) -> list[str]:
    lines = [
        f"held frames {report['held_frames']}  plan's own T_LR bend {report['plan_only']['mismatch_mm_max']} mm"
    ]
    for name, sc in report["scenarios"].items():
        sigma = sc["placement_sigma"]
        lines.append(
            f"[{name}] placement {sigma['mm']} mm / {sigma['deg']} deg: mismatch p95 {sc['mismatch_mm_p95']} mm, "
            f"{sc['mismatch_deg_p95']} deg"
        )
        for r in sc["scales"]:
            lines.append(
                f"   x{r['scale']:<5g} force {r['force_n_p95']:7.1f} N  moment {r['moment_nm_p95']:5.2f} Nm  "
                f"sag {r['sag_mm']:5.1f} mm  deadband {r['deadband_mm']:5.1f} mm  {'ok' if r['ok'] else '-'}"
            )
        for g in sc.get("grasps", []):
            lines.append(
                f"   grasp {g['grasp_n_m']:>7g} N/m {g['grasp_nm_rad']:>5g} Nm/rad: force {g['force_n_p95']:6.1f} N  moment "
                f"{g['moment_nm_p95']:5.2f} Nm  sag {g['sag_mm']:5.1f} mm"
            )
        for d in sc.get("designs", []):
            lines.append(
                f"   {d['design']}: force {d['force_n_p95']:6.1f} N  moment {d['moment_nm_p95']:5.2f} Nm  "
                f"sag {d['sag_mm']:5.1f} mm  give {d['give_mm_p95']:4.1f} mm / {d['give_deg_p95']:4.2f} deg "
                f"(travel {d['travel_mm']:g} mm / {d['travel_deg']:g} deg)  {'ok' if d['ok'] else '-'}"
            )
        lines.append(f"   -> {sc['verdict']}")
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Internal force budget of a two-hand held plan (D6)")
    parser.add_argument("plan", type=Path, help="a held plan npz (dual_fr3_coop_carry writes one)")
    parser.add_argument("--mass", type=float, default=OBJECT_MASS_KG, help="the held object, kg")
    parser.add_argument(
        "--skew", type=float, default=SKEW_S, help="one arm's set points behind the other's, s"
    )
    parser.add_argument("--samples", type=int, default=SAMPLES)
    parser.add_argument("--out", type=Path, default=None, help="report json (default: next to the plan)")
    args = parser.parse_args(argv)
    report = budget(read_plan(args.plan), mass_kg=args.mass, skew_s=args.skew, samples=args.samples)
    report["plan"] = str(args.plan)
    out = args.out or args.plan.with_name(args.plan.stem + "_closed_chain.json")
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    if "scenarios" in report:
        print("\n".join(_summary_lines(report)))
    else:
        print(report["verdict"])
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
