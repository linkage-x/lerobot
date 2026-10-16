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
The FR3 pivot about the table ball, planned and driven instead of moved by hand.

  capture  real robot, read only: the socket is seated on the ball (Desk guiding); FK of
           corenetic_socket_ee there is the pivot centre to plan about
  plan     the pivot motion about that centre -- tilt into a cone, sweep it, spin about the tool
           axis -- solved by IK and checked in MuJoCo (limits incl. panda_py's walls, speed, wrist
           travel, self-contact, table clearance) before anything may run it
  execute  real robot: a soft Cartesian impedance follows the plan, pressing the socket into the ball
  guide    real robot, the recommended one: a virtual ball joint. The impedance only holds the socket
           pressed down on the ball (translation spring, zero rotational stiffness, elbow free); the operator
           turns the gripper by hand and so steers clear of the table and everything else, which no
           model has to know about

Not circular, although the plan is built on the very frame being measured: the arm is soft
(600 N/m) and pressed into the ball, so the ball -- not the command -- decides where the socket
is, and ``fr3_tcp_pivot.py fit`` uses the *measured* joint angles. A few mm of error in
corenetic_socket_ee is a newton or two in the spring. Proof by data: run the same centre with the
pivot point deliberately biased +3 mm and -3 mm (``--bias-mm``); if the socket was seated, both fits
agree, if it followed the command they differ by 6 mm. Each run also records the contact force.

The spring holds one point of the socket frame. By default that is corenetic_socket_ee's origin; when
the true pivot is s mm off it, every turn by θ drags that origin ~|s|·θ sideways against the ball
(10-13: s ≈ 6-7 mm, 1-3 N at 20°). ``guide --anchor-mm`` (the s an earlier fit found, same grip) moves
the held point onto the ball centre, so the spring stops fighting the turn. It is fitted on one grip;
a re-clamped insert seats again within 0.2 mm (10-14), so the GUI only warns after a re-clamp.

On Thor, from ~/lerobot (panda_py needs the cmeel libraries):
  V=~/Code/infer/.venv-fr3; LD_LIBRARY_PATH=$V/lib/python3.12/site-packages/cmeel.prefix/lib \
  PYTHONPATH=src $V/bin/python tools/fr3/fr3_pivot_auto.py capture
"""

# ruff: noqa: N803, N806  (matrix names: R, T, J follow the maths)
from __future__ import annotations

import argparse
import datetime as dt
import json
import signal
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from tools.fr3.fr3_tcp_pivot import (  # noqa: E402
    DEFAULT_ROBOT_IP,
    DEFAULT_URDF,
    FR3_JOINT_LIMITS,
    JOINTS,
    SOCKET_FRAME,
    contact_summary,  # noqa: F401  (re-exported for the fit report)
    teaching_wall_problems,
)

REPO = Path(__file__).resolve().parents[2]
DEFAULT_XML = REPO / "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper.xml"
OUT_ROOT = REPO / "outputs" / "fr3_tcp_pivot"
CENTRE_SCHEMA = "fr3_pivot_centre/v1"
PLAN_SCHEMA = "fr3_pivot_plan/v1"

# panda_py.constants (panda_python 0.8.1): every panda_py torque controller runs virtual walls on
# these Panda limits, and a joint past one aborts the process (see fr3_tcp_pivot.FR3_JOINT_LIMITS).
PANDA_WALL_LOWER = np.array([-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973])
PANDA_WALL_UPPER = np.array([2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973])

# Executed by the planner's checks and the execute monitors.
LIMIT_MARGIN_DEG = 3.0
MAX_JOINT_SPEED_DEG_S = 25.0
# The arm BOX lost power on a 123° j6 swing at 89°/s (10-13). A +-25° tilt about j6's axis alone
# is 50° of j6; what is capped is the size and above all the speed of the swing.
MAX_J6_TRAVEL_DEG = 90.0
MAX_J6_SPEED_DEG_S = 10.0
MAX_IK_POS_MM = 0.5
MAX_IK_ROT_DEG = 0.2
MIN_TABLE_CLEARANCE_MM = 20.0
# The table top, in fr3_base: the FR3 stands about 12 cm below it (the operator, 10-13).
DEFAULT_TABLE_Z_M = 0.12
# The ball sits on the table; a centre below this is a mis-capture (or the wrong table height).
MIN_CENTRE_ABOVE_TABLE_MM = 10.0
START_TOLERANCE_DEG = 1.5
START_TOLERANCE_MM = 3.0
STOP_FORCE_N = 15.0
STOP_TRACKING_MM = 15.0
STOP_WALL_MARGIN_DEG = 1.0
# The impedance holds the socket; only the nullspace term holds the elbow where MuJoCo checked it.
# 0.5 (panda_py's default) let it drift; 15 is panda_py's move_to_pose default. And a joint off the
# plan by this much means the posture is no longer the one that was checked against the table.
NULLSPACE_STIFFNESS = 15.0
STOP_JOINT_DEVIATION_DEG = 5.0
# guide: a hand on the gripper pushes too, so the force stop sits higher than the planned run's.
GUIDE_STOP_FORCE_N = 20.0
GUIDE_DEFAULT_DURATION_S = 60.0
GUIDE_DURATION_RANGE_S = (20.0, 180.0)
GUIDE_RAMP_S = 2.0
GUIDE_RATE_HZ = 100.0
# Damping for the free directions: nullspace target = the current posture, so this term only damps.
GUIDE_NULLSPACE_STIFFNESS = 2.0
GUIDE_WALL_WARN_DEG = 6.0
GUIDE_PHASES = ["press", "guide", "release"]
# A held point further than this from the socket frame is not a fitted ball centre but a bad fit.
GUIDE_MAX_ANCHOR_MM = 20.0
# The insert's cup sits on top of the ball (the ball on a stand on the table, the gripper reaching in
# roughly level -- the 10-13 photo): the cup opens downwards, across the tool axis, not along it. So the
# press is straight down in fr3_base, onto the stand, and the pivot turns about axes through the ball:
# two level ones (pitch the gripper, roll it about its own axis) and the vertical (swing it round).
PRESS_DIR = np.array([0.0, 0.0, -1.0])
# Bodies bolted to the table: never "too low".
TABLE_EXEMPT_BODIES = {"world", "base", "fr3_link0", "fr3_link1"}


@dataclass
class PlanParams:
    # 10-13, with the cup down and the gripper about level: +-25°/+-60° rarely fits inside panda_py's
    # walls (j6 at 215°); +-20°/+-30° does over most of 0.45-0.65 m, gripper heading 60-90° off radial.
    tilt_deg: float = 20.0
    spin_deg: float = 30.0
    speed_deg_s: float = 5.0  # peak angular speed of the tool
    press_mm: float = 3.0  # pivot point commanded this far into the ball, along the tool axis
    bias_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])  # fr3_base, deliberate
    ramp_s: float = 2.0  # press/bias in and out
    rate_hz: float = 100.0
    translational_stiffness: float = 600.0
    rotational_stiffness: float = 40.0


# --- rotations -------------------------------------------------------------------------------------


def rot(axis, angle: float) -> np.ndarray:
    a = np.asarray(axis, dtype=float)
    a = a / np.linalg.norm(a)
    K = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    return np.eye(3) + np.sin(angle) * K + (1 - np.cos(angle)) * K @ K


def rot_angle(R: np.ndarray) -> float:
    return float(np.arccos(np.clip((np.trace(R) - 1) / 2, -1.0, 1.0)))


def quat_xyzw(R: np.ndarray) -> np.ndarray:
    """Unit quaternion (x, y, z, w) of a rotation matrix (Shepperd)."""
    t = np.trace(R)
    if t > 0:
        s = 2 * np.sqrt(1 + t)
        q = [(R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s, s / 4]
    else:
        i = int(np.argmax(np.diag(R)))
        j, k = (i + 1) % 3, (i + 2) % 3
        s = 2 * np.sqrt(1 + R[i, i] - R[j, j] - R[k, k])
        q = [0.0, 0.0, 0.0, (R[k, j] - R[j, k]) / s]
        q[i] = s / 4
        q[j] = (R[j, i] + R[i, j]) / s
        q[k] = (R[k, i] + R[i, k]) / s
    q = np.asarray(q)
    return q / np.linalg.norm(q)


def smoothstep(u):
    u = np.clip(u, 0.0, 1.0)
    return u * u * (3 - 2 * u)


# --- the path --------------------------------------------------------------------------------------


def pivot_axes(R0: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Two level axes through the ball, in fr3_base: along the tool's level heading (rolls the gripper
    about its own axis) and across it (pitches the gripper). Spin is about the vertical."""
    h = np.array([R0[0, 2], R0[1, 2], 0.0])
    if np.linalg.norm(h) < 0.3:  # tool near vertical: any level pair will do
        return np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0])
    a1 = h / np.linalg.norm(h)
    return a1, np.cross([0.0, 0.0, 1.0], a1)


def _phases(p: PlanParams, a1=(1.0, 0.0, 0.0), a2=(0.0, 1.0, 0.0)) -> list[tuple[str, Any, str]]:
    """(name, u -> rotation about the ball in fr3_base, how press/bias ramp) in order.

    A star, not a cone: tilt +-``tilt_deg`` about the level axis ``a1``, then about ``a2``, then spin
    +-``spin_deg`` about the vertical. Two tilt axes and the spin pin the socket down; a full cone also
    tilts sideways to the forearm, which near a vertical forearm is a wrist flip.
    """
    tilt, spin = np.radians(p.tilt_deg), np.radians(p.spin_deg)
    z = np.array([0.0, 0.0, 1.0])
    I3 = np.eye(3)

    def swing(axis, name):  # 0 -> +tilt -> -tilt -> 0
        return [
            (f"{name}+", lambda u: rot(axis, u * tilt), "on"),
            (f"{name}-", lambda u: rot(axis, (1 - 2 * u) * tilt), "on"),
            (f"{name}0", lambda u: rot(axis, (u - 1) * tilt), "on"),
        ]

    return [
        ("press", lambda u: I3, "in"),
        *swing(np.asarray(a1, float), "tiltA"),
        *swing(np.asarray(a2, float), "tiltB"),
        ("spin+", lambda u: rot(z, u * spin), "on"),
        ("spin-", lambda u: rot(z, (1 - 2 * u) * spin), "on"),
        ("spin0", lambda u: rot(z, (u - 1) * spin), "on"),
        ("release", lambda u: I3, "out"),
    ]


def build_path(
    R0: np.ndarray, c0: np.ndarray, p: PlanParams, a1=(1.0, 0.0, 0.0), a2=(0.0, 1.0, 0.0)
) -> dict[str, np.ndarray]:
    """Socket poses sampled at ``rate_hz``: each phase timed by its arc length, smoothstep in and out."""
    dt_s = 1.0 / p.rate_hz
    speed = np.radians(p.speed_deg_s)
    bias = np.asarray(p.bias_mm, dtype=float) / 1e3
    press = p.press_mm / 1e3
    ts, Rs, ps, phase_idx, factor = [], [], [], [], []
    t0 = 0.0
    phases = _phases(p, a1, a2)
    for k, (_, fn, ramp) in enumerate(phases):
        us = np.linspace(0.0, 1.0, 401)
        dense = [fn(u) for u in us]
        steps = [rot_angle(a.T @ b) for a, b in zip(dense[:-1], dense[1:], strict=True)]
        arc = np.concatenate([[0.0], np.cumsum(steps)])
        length = arc[-1]
        # smoothstep peaks at 1.5x its mean speed
        duration = p.ramp_s if length < 1e-9 else 1.5 * length / speed
        n = max(2, int(round(duration / dt_s)))
        for i in range(n):
            s = float(smoothstep(i / n))
            u = s if length < 1e-9 else float(np.interp(s * length, arc, us))
            R_rel = fn(u)
            f = {"in": s, "out": 1.0 - s, "on": 1.0}[ramp]
            R = R_rel @ R0  # turned about the ball, in fr3_base
            ts.append(t0 + i * dt_s)
            Rs.append(R)
            ps.append(c0 + f * (bias + press * PRESS_DIR))
            phase_idx.append(k)
            factor.append(f)
        t0 += n * dt_s
    # end exactly where it started, nothing pressed
    ts.append(t0)
    Rs.append(R0.copy())
    ps.append(c0.copy())
    phase_idx.append(len(phases) - 1)
    factor.append(0.0)
    return {
        "t": np.asarray(ts),
        "R": np.asarray(Rs),
        "p": np.asarray(ps),
        "phase": np.asarray(phase_idx),
        "factor": np.asarray(factor),
    }


# --- kinematics ------------------------------------------------------------------------------------


IK_LOWER = np.maximum(np.asarray(FR3_JOINT_LIMITS)[:, 0], PANDA_WALL_LOWER) + np.radians(0.5)
IK_UPPER = np.minimum(np.asarray(FR3_JOINT_LIMITS)[:, 1], PANDA_WALL_UPPER) - np.radians(0.5)


class Kinematics:
    """pinocchio FK/IK of one URDF frame over the seven arm joints."""

    def __init__(self, urdf: Path = DEFAULT_URDF, frame: str = SOCKET_FRAME):
        import pinocchio as pin

        self.pin = pin
        full = pin.buildModelFromUrdf(str(urdf))
        lock = [i for i, n in enumerate(full.names) if i > 0 and n not in JOINTS]
        self.model = pin.buildReducedModel(full, lock, pin.neutral(full))
        self.data = self.model.createData()
        self.fid = self.model.getFrameId(frame)

    def fk(self, q) -> np.ndarray:
        pin = self.pin
        pin.forwardKinematics(self.model, self.data, np.asarray(q, dtype=float))
        pin.updateFramePlacements(self.model, self.data)
        return self.data.oMf[self.fid].homogeneous.copy()

    def ik(
        self,
        T_des: np.ndarray,
        seed,
        q_null,
        *,
        iters: int = 60,
        damping: float = 1e-4,
        null_gain: float = 0.05,
    ):
        """Damped least squares from ``seed``; the nullspace leans towards ``q_null`` (keeps the posture)."""
        pin = self.pin
        q = np.asarray(seed, dtype=float).copy()
        q_null = np.asarray(q_null, dtype=float)
        e_p = e_r = np.full(3, np.inf)
        for _ in range(iters):
            T = self.fk(q)
            e_p = T_des[:3, 3] - T[:3, 3]
            e_r = pin.log3(T_des[:3, :3] @ T[:3, :3].T)
            if np.linalg.norm(e_p) < 1e-7 and np.linalg.norm(e_r) < 1e-6:
                break
            J = pin.computeFrameJacobian(self.model, self.data, q, self.fid, pin.LOCAL_WORLD_ALIGNED)
            JJt = J @ J.T + damping * np.eye(6)
            dq = J.T @ np.linalg.solve(JJt, np.r_[e_p, e_r])
            N = np.eye(7) - np.linalg.pinv(J) @ J
            # Kept inside the FR3's limits and panda_py's walls: a solution past either is no solution.
            q = np.clip(q + dq + N @ (null_gain * (q_null - q)), IK_LOWER, IK_UPPER)
        return q, float(np.linalg.norm(e_p)), float(np.linalg.norm(e_r))


# --- MuJoCo checks ---------------------------------------------------------------------------------


class SceneCheck:
    """Contact pairs the start pose does not have, and how close any moving mesh comes to the table."""

    def __init__(
        self, xml: Path = DEFAULT_XML, table_z_m: float = DEFAULT_TABLE_Z_M, max_vertices: int = 300
    ):
        import mujoco

        self.mj = mujoco
        self.model = mujoco.MjModel.from_xml_path(str(xml))
        self.data = mujoco.MjData(self.model)
        self.table_z = table_z_m
        self.qadr = [self.model.jnt_qposadr[self.model.joint(n).id] for n in JOINTS]
        m = self.model
        self.geoms = []  # (geom id, local points)
        for g in range(m.ngeom):
            body = m.body(m.geom_bodyid[g]).name
            if body in TABLE_EXEMPT_BODIES:
                continue
            if m.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[g]
                v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                if len(v) > max_vertices:
                    v = v[np.linspace(0, len(v) - 1, max_vertices).astype(int)]
                self.geoms.append((g, np.asarray(v, dtype=float), 0.0))
            elif m.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                self.geoms.append((g, np.zeros((1, 3)), float(m.geom_size[g][0])))
        self.baseline: set[tuple[str, str]] | None = None

    def _set(self, q) -> None:
        for adr, x in zip(self.qadr, q, strict=True):
            self.data.qpos[adr] = x
        self.mj.mj_forward(self.model, self.data)

    def contacts(self, q) -> set[tuple[str, str]]:
        self._set(q)
        m, d = self.model, self.data
        pairs = set()
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist < 0:
                a, b = m.body(m.geom_bodyid[c.geom1]).name, m.body(m.geom_bodyid[c.geom2]).name
                pairs.add(tuple(sorted((a, b))))
        return pairs

    def clearance(self, q) -> tuple[float, str]:
        """Lowest point of any moving geom above the table (m), and whose."""
        self._set(q)
        d, m = self.data, self.model
        best, who = np.inf, ""
        for g, pts, radius in self.geoms:
            z = (pts @ d.geom_xmat[g].reshape(3, 3).T)[:, 2] + d.geom_xpos[g][2] - radius
            if z.min() < best:
                best, who = float(z.min()), m.body(m.geom_bodyid[g]).name
        return best - self.table_z, who

    def new_contacts(self, q) -> set[tuple[str, str]]:
        pairs = self.contacts(q)
        assert self.baseline is not None, "set the baseline at the start pose first"
        return pairs - self.baseline


# --- plan ------------------------------------------------------------------------------------------


def limit_margins_deg(qs: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per joint, the smallest distance (deg) to the FR3 limits and to panda_py's walls over ``qs``."""
    fr3 = np.asarray(FR3_JOINT_LIMITS)
    m_fr3 = np.minimum(qs - fr3[:, 0], fr3[:, 1] - qs).min(axis=0)
    m_wall = np.minimum(qs - PANDA_WALL_LOWER, PANDA_WALL_UPPER - qs).min(axis=0)
    return np.degrees(m_fr3), np.degrees(m_wall)


def plan(
    centre: dict[str, Any],
    params: PlanParams,
    kin: Kinematics,
    scene: SceneCheck | None,
    *,
    check_every: int = 10,
    table_z_m: float = DEFAULT_TABLE_Z_M,
):
    q0 = np.asarray(centre["q"], dtype=float)
    T0 = kin.fk(q0)
    a1, a2 = pivot_axes(T0[:3, :3])
    path = build_path(T0[:3, :3], T0[:3, 3], params, a1, a2)
    n = len(path["t"])
    qs = np.zeros((n, 7))
    q = q0
    for i in range(n):
        T = np.eye(4)
        T[:3, :3], T[:3, 3] = path["R"][i], path["p"][i]
        q, _, _ = kin.ik(T, q, q0)
        qs[i] = q
    summary = check_path(
        path, qs, q0, kin, scene, params.rate_hz, check_every=check_every, table_z_m=table_z_m
    )
    summary["tilt_axes_base"] = [np.round(a1, 3).tolist(), np.round(a2, 3).tolist()]
    return path, qs, summary


def check_path(
    path: dict[str, np.ndarray],
    qs: np.ndarray,
    q0: np.ndarray,
    kin: Kinematics,
    scene: SceneCheck | None,
    rate_hz: float,
    *,
    check_every: int = 10,
    table_z_m: float = DEFAULT_TABLE_Z_M,
) -> dict[str, Any]:
    """Everything a path must pass before ``execute`` may run it: limits and walls, speeds, wrist travel,
    IK error against the commanded poses, and in MuJoCo self-contact and the table."""
    T0 = kin.fk(q0)
    n = len(qs)
    ik_pos, ik_rot = np.zeros(n), np.zeros(n)
    for i in range(n):
        T = kin.fk(qs[i])
        ik_pos[i] = np.linalg.norm(T[:3, 3] - path["p"][i])
        ik_rot[i] = rot_angle(T[:3, :3].T @ path["R"][i])

    problems: list[str] = []
    above_mm = (T0[2, 3] - table_z_m) * 1e3
    if above_mm < MIN_CENTRE_ABOVE_TABLE_MM:
        problems.append(
            f"球心只比桌面高 {above_mm:.0f} mm（桌面按 fr3_base z = {table_z_m * 1e3:.0f} mm）：球心取错了，或桌面高度不对"
        )
    m_fr3, m_wall = limit_margins_deg(qs)
    for j in range(7):
        if m_fr3[j] < LIMIT_MARGIN_DEG:
            problems.append(f"j{j + 1} 离 FR3 限位只有 {m_fr3[j]:.1f}°（要 ≥ {LIMIT_MARGIN_DEG:.0f}°）")
        if m_wall[j] < LIMIT_MARGIN_DEG:
            problems.append(
                f"j{j + 1} 离 panda_py 的墙只有 {m_wall[j]:.1f}°（要 ≥ {LIMIT_MARGIN_DEG:.0f}°；越墙进程会直接退出）"
            )
    dq = np.degrees(np.abs(np.diff(qs, axis=0))) * rate_hz
    speed = dq.max(axis=0)
    for j in range(7):
        if speed[j] > MAX_JOINT_SPEED_DEG_S:
            problems.append(f"j{j + 1} 最快 {speed[j]:.0f}°/s（上限 {MAX_JOINT_SPEED_DEG_S:.0f}°/s）")
    travel = np.degrees(qs.max(axis=0) - qs.min(axis=0))
    if travel[5] > MAX_J6_TRAVEL_DEG:
        problems.append(
            f"j6 总转角 {travel[5]:.0f}°（上限 {MAX_J6_TRAVEL_DEG:.0f}°：手腕大幅摆动曾让夹爪 BOX 掉电）"
        )
    if speed[5] > MAX_J6_SPEED_DEG_S:
        problems.append(
            f"j6 最快 {speed[5]:.0f}°/s（上限 {MAX_J6_SPEED_DEG_S:.0f}°/s：10-13 掉电那次是 89°/s 的快摆）"
        )
    if ik_pos.max() * 1e3 > MAX_IK_POS_MM or np.degrees(ik_rot.max()) > MAX_IK_ROT_DEG:
        problems.append(
            f"IK 没解到（最差 {ik_pos.max() * 1e3:.2f} mm / {np.degrees(ik_rot.max()):.2f}°）：这个球心附近转不开"
        )

    clearance_mm, clearance_body, contacts = None, "", []
    if scene is not None:
        scene.baseline = scene.contacts(q0)
        new: set[tuple[str, str]] = set()
        low, low_body = np.inf, ""
        for i in list(range(0, n, check_every)) + [n - 1]:
            new |= scene.new_contacts(qs[i])
            c, who = scene.clearance(qs[i])
            if c < low:
                low, low_body = c, who
        clearance_mm, clearance_body, contacts = low * 1e3, low_body, sorted(new)
        for a, b in contacts:
            problems.append(f"MuJoCo 里 {a} 和 {b} 碰上了")
        if clearance_mm < MIN_TABLE_CLEARANCE_MM:
            problems.append(
                f"{clearance_body} 最低离桌面 {clearance_mm:.0f} mm（要 ≥ {MIN_TABLE_CLEARANCE_MM:.0f} mm，"
                f"桌面按 fr3_base z = {scene.table_z * 1e3:.0f} mm）"
            )

    summary = {
        "passed": not problems,
        "problems": problems,
        "duration_s": round(float(path["t"][-1]), 1),
        "samples": n,
        "joint_travel_deg": np.round(travel, 1).tolist(),
        "max_joint_speed_deg_s": np.round(speed, 1).tolist(),
        "min_margin_fr3_deg": np.round(m_fr3, 1).tolist(),
        "min_margin_wall_deg": np.round(m_wall, 1).tolist(),
        "ik_worst_mm": round(float(ik_pos.max() * 1e3), 3),
        "ik_worst_deg": round(float(np.degrees(ik_rot.max())), 3),
        "table_clearance_mm": None if clearance_mm is None else round(clearance_mm, 1),
        "table_clearance_body": clearance_body,
        "mujoco_checked": scene is not None,
        "table_z_mm": round((scene.table_z if scene is not None else table_z_m) * 1e3, 1),
        "centre_above_table_mm": round(float(above_mm), 1),
    }
    return summary


# --- commands --------------------------------------------------------------------------------------


def _stamp() -> str:
    return f"{dt.datetime.now():%Y%m%d_%H%M%S}"


def _write(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1), encoding="utf-8")


def _ee_matrix(state) -> np.ndarray:
    return np.asarray(state.O_T_EE, dtype=float).reshape(4, 4).T  # libfranka is column-major


def fresh_state(panda):
    """A new 1 kHz frame. Without a controller running, panda_py's get_state() is a frozen cache (10-12)."""
    return panda.get_robot().read_once()


def capture(args) -> int:
    from panda_py import Panda

    panda = Panda(args.robot_ip)
    state = fresh_state(panda)
    q = np.asarray(state.q, dtype=float)
    beyond, close = teaching_wall_problems(q, PANDA_WALL_LOWER, PANDA_WALL_UPPER)
    kin = Kinematics(args.urdf)
    T = kin.fk(q)
    out = args.out or OUT_ROOT / "centres" / f"{_stamp()}_centre.json"
    payload = {
        "schema": CENTRE_SCHEMA,
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "robot_ip": args.robot_ip,
        "note": args.note,
        "frame": SOCKET_FRAME,
        "q": q.tolist(),
        "O_T_EE": _ee_matrix(state).tolist(),
        "centre_m": T[:3, 3].tolist(),
        "R": T[:3, :3].tolist(),
        "tool_from_level_deg": round(float(np.degrees(np.arcsin(np.clip(T[2, 2], -1, 1)))), 1),
        "wall_problems": beyond + close,
    }
    _write(out, payload)
    print(
        f"centre (socket FK) {np.round(T[:3, 3] * 1e3, 1).tolist()} mm in fr3_base, tool axis {payload['tool_from_level_deg']}° from level (+ = up)"
    )
    for line in beyond:
        print(f"ERROR: {line}")
    for line in close:
        print(f"WARNING: {line}")
    print(f"wrote {out}")
    return 0


def plan_cmd(args) -> int:
    centre = json.loads(args.centre.read_text(encoding="utf-8"))
    if centre.get("schema") != CENTRE_SCHEMA:
        print(f"ERROR: {args.centre} is not a {CENTRE_SCHEMA}")
        return 2
    if not args.star:
        return plan_dwells_cmd(args, centre)
    params = PlanParams(
        tilt_deg=args.tilt_deg,
        spin_deg=args.spin_deg,
        speed_deg_s=args.speed_deg_s,
        press_mm=args.press_mm,
        bias_mm=[float(x) for x in args.bias_mm],
    )
    kin = Kinematics(args.urdf)
    scene = None if args.no_mujoco else SceneCheck(args.xml, table_z_m=args.table_z_m)
    path, qs, summary = plan(centre, params, kin, scene, table_z_m=args.table_z_m)
    out = args.out or OUT_ROOT / "plans" / f"{_stamp()}_plan.json"
    header = {
        "schema": PLAN_SCHEMA,
        "kind": "star",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "centre": args.centre.name,
        "centre_m": centre["centre_m"],
        "params": asdict(params),
        **summary,
    }
    _write(out.with_name(out.name.removesuffix(".json") + ".summary.json"), header)
    _write(
        out,
        {
            **header,
            "q_start": centre["q"],
            "t": np.round(path["t"], 4).tolist(),
            "q": np.round(qs, 6).tolist(),
            "p": np.round(path["p"], 6).tolist(),
            "quat_xyzw": np.round([quat_xyzw(R) for R in path["R"]], 7).tolist(),
            "phase": path["phase"].tolist(),
            "phase_names": [name for name, _, _ in _phases(params)],
        },
    )
    verdict = "PASS" if summary["passed"] else "FAIL"
    print(
        f"plan {verdict}: {summary['duration_s']} s, joint travel {summary['joint_travel_deg']}°, "
        f"wall margin {summary['min_margin_wall_deg']}°, table clearance {summary['table_clearance_mm']} mm"
    )
    for line in summary["problems"]:
        print(f"  {line}")
    print(f"wrote {out}")
    return 0 if summary["passed"] else 3


def plan_dwells_cmd(args, centre: dict[str, Any]) -> int:
    """Survey the orientations about the ball, pick the dwells, route them, check the timed path."""
    from tools.fr3 import fr3_pivot_plan as pp

    if np.linalg.norm(args.anchor_mm) > GUIDE_MAX_ANCHOR_MM:
        print(
            f"ERROR: --anchor-mm {args.anchor_mm} is further than {GUIDE_MAX_ANCHOR_MM:g} mm off the socket frame"
        )
        return 2
    params = pp.DwellParams(
        dwells=args.dwells,
        dwell_s=args.dwell_s,
        speed_deg_s=args.speed_deg_s,
        press_mm=args.press_mm,
        bias_mm=[float(x) for x in args.bias_mm],
        anchor_mm=[float(x) for x in args.anchor_mm],
    )
    kin = Kinematics(args.urdf)
    scene = None if args.no_mujoco else SceneCheck(args.xml, table_z_m=args.table_z_m)
    stand = pp.Stand.load()
    t0 = time.monotonic()
    path, qs, names, info, survey = pp.plan_dwells(
        centre, kin, scene, stand, params, pp.SurveyParams(table_z_m=args.table_z_m)
    )
    problems = list(info["problems"])
    summary: dict[str, Any] = {"passed": False, "problems": problems}
    if len(qs):
        summary = check_path(
            path, qs, np.asarray(centre["q"]), kin, scene, params.rate_hz, table_z_m=args.table_z_m
        )
        more, stand_info = pp.stand_problems(survey, path, qs)
        summary.update(stand_info)
        summary["problems"] = problems + summary["problems"] + more
        summary["passed"] = not summary["problems"]
    out = args.out or OUT_ROOT / "plans" / f"{_stamp()}_plan.json"
    header = {
        "schema": PLAN_SCHEMA,
        "kind": "dwells",
        "created": dt.datetime.now().isoformat(timespec="seconds"),
        "centre": args.centre.name,
        "centre_m": centre["centre_m"],
        "ball_m": np.round(survey.centre_m, 6).tolist(),
        "anchor_from": args.anchor_from,
        "params": asdict(params),
        **summary,
        "survey": info["survey"],
        "dwells_deg": info.get("dwells_deg", []),
        "sigma_per_unit": info.get("sigma_per_unit"),
        "notes": info.get("notes", []),
        "planning_s": round(time.monotonic() - t0, 1),
    }
    _write(out.with_name(out.name.removesuffix(".json") + ".summary.json"), header)
    body = {**header, "q_start": centre["q"], "dwell_q": info.get("dwell_q", []), "phase_names": names}
    if len(qs):
        body.update(
            {
                "t": np.round(path["t"], 4).tolist(),
                "q": np.round(qs, 6).tolist(),
                "p": np.round(path["p"], 6).tolist(),
                "quat_xyzw": np.round([quat_xyzw(R) for R in path["R"]], 7).tolist(),
                "phase": path["phase"].tolist(),
            }
        )
    _write(out, body)
    sv = info["survey"]
    print(
        f"survey: {sv['transit']} of {sv['cells']} orientations pass (heading {sv['span_deg']['heading']}, "
        f"pitch {sv['span_deg']['pitch']}, roll {sv['span_deg']['roll']} deg); failing: {sv['why']}"
    )
    for line in info.get("notes", []):
        print(f"  {line}")
    if info.get("sigma_per_unit"):
        sig = info["sigma_per_unit"]
        print(
            f"{len(info['dwells_deg'])} dwells: predicted 1-sigma per mm of dwell error s {sig['s']}, c {sig['c']} "
            f"(at 0.5 mm per dwell: s <= {0.5 * max(sig['s']):.2f} mm)"
        )
    verdict = "PASS" if summary["passed"] else "FAIL"
    if len(qs):
        print(
            f"plan {verdict}: {summary['duration_s']} s, joint travel {summary['joint_travel_deg']}°, "
            f"wall margin {summary['min_margin_wall_deg']}°, table clearance {summary['table_clearance_mm']} mm, "
            f"stand: socket {summary['socket_stand_mm']} mm, gripper {summary['gripper_stand_mm']} mm "
            f"({summary['gripper_stand_body']})"
        )
    else:
        print(f"plan {verdict}")
    for line in summary["problems"]:
        print(f"  {line}")
    print(f"wrote {out}")
    return 0 if summary["passed"] else 3


def start_problems(q_now, socket_now, plan_data: dict[str, Any]) -> list[str]:
    """Why the arm is not where the plan starts (it must still be seated where the centre was taken)."""
    out = []
    dq = np.degrees(np.abs(np.asarray(q_now) - np.asarray(plan_data["q"][0])))
    if dq.max() > START_TOLERANCE_DEG:
        j = int(dq.argmax())
        out.append(f"j{j + 1} 离规划起点差 {dq[j]:.1f}°：取球心之后机械臂被动过，重新取球心并演算")
    dp = np.linalg.norm(np.asarray(socket_now) - np.asarray(plan_data["p"][0])) * 1e3
    if dp > START_TOLERANCE_MM:
        out.append(f"球窝离规划起点 {dp:.1f} mm")
    return out


@dataclass
class Monitor:
    """Why to stop now, or ""."""

    force_n: float = STOP_FORCE_N
    tracking_mm: float = STOP_TRACKING_MM
    wall_deg: float = STOP_WALL_MARGIN_DEG
    joint_deg: float = STOP_JOINT_DEVIATION_DEG

    def check(self, q, f_ext, socket_actual, socket_cmd, q_plan=None) -> str:
        f = float(np.linalg.norm(np.asarray(f_ext)[:3]))
        if f > self.force_n:
            return f"外力 {f:.1f} N 超过 {self.force_n:.0f} N"
        e = float(np.linalg.norm(np.asarray(socket_actual) - np.asarray(socket_cmd)) * 1e3)
        if e > self.tracking_mm:
            return f"球窝偏离指令 {e:.1f} mm 超过 {self.tracking_mm:.0f} mm"
        q = np.asarray(q)
        margin = np.degrees(np.minimum(q - PANDA_WALL_LOWER, PANDA_WALL_UPPER - q))
        if margin.min() < self.wall_deg:
            return f"j{int(margin.argmin()) + 1} 离 panda_py 的墙不到 {self.wall_deg:.0f}°"
        if q_plan is not None:
            dev = np.degrees(np.abs(q - np.asarray(q_plan)))
            if dev.max() > self.joint_deg:
                return f"j{int(dev.argmax()) + 1} 偏离规划 {dev.max():.1f}°（姿态已不是演算过的那个）"
        return ""


def execute(args) -> int:
    plan_data = json.loads(args.plan.read_text(encoding="utf-8"))
    if plan_data.get("schema") != PLAN_SCHEMA:
        print(f"ERROR: {args.plan} is not a {PLAN_SCHEMA}")
        return 2
    if not plan_data.get("passed"):
        print("ERROR: the plan did not pass its checks; nothing was started")
        return 2
    from panda_py import Panda, controllers

    kin = Kinematics(args.urdf)
    panda = Panda(args.robot_ip)
    state = fresh_state(panda)
    q_now = np.asarray(state.q, dtype=float)
    beyond, _ = teaching_wall_problems(q_now, PANDA_WALL_LOWER, PANDA_WALL_UPPER)
    T_sock = kin.fk(q_now)
    problems = beyond + start_problems(q_now, T_sock[:3, 3], plan_data)
    if problems:
        print("ERROR: not starting:")
        for line in problems:
            print(f"  {line}")
        return 2

    # The controller drives Desk's EE (O_T_EE); the plan is in corenetic_socket_ee. Constant between them.
    X = np.linalg.inv(_ee_matrix(state)) @ T_sock
    X_inv = np.linalg.inv(X)
    params = plan_data["params"]
    K = np.diag([params["translational_stiffness"]] * 3 + [params["rotational_stiffness"]] * 3)
    ctrl = controllers.CartesianImpedance(
        impedance=K, damping_ratio=1.0, nullspace_stiffness=NULLSPACE_STIFFNESS, filter_coeff=1.0
    )

    stop = {"why": ""}

    def _sigint(signum, frame):  # noqa: ARG001
        stop["why"] = "operator"

    signal.signal(signal.SIGINT, _sigint)
    signal.signal(signal.SIGTERM, _sigint)
    t_plan = np.asarray(plan_data["t"])
    P, Q, QN = np.asarray(plan_data["p"]), np.asarray(plan_data["quat_xyzw"]), np.asarray(plan_data["q"])
    rec: dict[str, list] = {
        "t": [],
        "q": [],
        "dq": [],
        "O_T_EE": [],
        "F_ext": [],
        "cmd_socket": [],
        "phase": [],
    }
    monitor = Monitor()
    print(
        f"execute: {t_plan[-1]:.0f} s pivot, soft impedance {params['translational_stiffness']:.0f} N/m, "
        f"press {params['press_mm']} mm, bias {params['bias_mm']} mm"
    )
    panda.start_controller(ctrl)
    try:
        ee0 = _ee_matrix(state)
        ctrl.set_control(ee0[:3, 3], quat_xyzw(ee0[:3, :3]), q_now)
        t0 = time.monotonic()
        next_print = 0.0
        while not stop["why"]:
            elapsed = time.monotonic() - t0
            if elapsed > t_plan[-1]:
                break
            k = int(min(np.searchsorted(t_plan, elapsed), len(t_plan) - 1))
            T_cmd = np.eye(4)
            x, y, z, w = Q[k]
            T_cmd[:3, :3] = _quat_matrix(x, y, z, w)
            T_cmd[:3, 3] = P[k]
            T_ee = T_cmd @ X_inv
            ctrl.set_control(T_ee[:3, 3], quat_xyzw(T_ee[:3, :3]), QN[k])
            s = panda.get_state()
            ee = _ee_matrix(s)
            socket = (ee @ X)[:3, 3]
            f_ext = list(s.O_F_ext_hat_K)
            rec["t"].append(elapsed)
            rec["q"].append(list(s.q))
            rec["dq"].append(list(s.dq))
            rec["O_T_EE"].append(list(s.O_T_EE))
            rec["F_ext"].append(f_ext)
            rec["cmd_socket"].append(P[k].tolist())
            rec["phase"].append(int(plan_data["phase"][k]))
            why = monitor.check(s.q, f_ext, socket, P[k], QN[k])
            if why:
                stop["why"] = why
                break
            panda.raise_error()
            if elapsed >= next_print:
                print(
                    f"  {elapsed:5.1f} / {t_plan[-1]:.0f} s  {plan_data['phase_names'][rec['phase'][-1]]:7s} "
                    f"press {-f_ext[2]:+4.1f} N  |F| {np.linalg.norm(f_ext[:3]):4.1f} N  off cmd {np.linalg.norm(socket - P[k]) * 1e3:4.1f} mm"
                )
                next_print += 5.0
            time.sleep(1.0 / params["rate_hz"])
    except RuntimeError as exc:
        stop["why"] = f"robot error: {exc}"
    finally:
        panda.stop_controller()

    payload = {
        "robot_ip": args.robot_ip,
        "note": args.note,
        "created": dt.datetime.now().isoformat(),
        "mode": "commanded",
        "plan": args.plan.name,
        "centre": plan_data["centre"],
        "bias_mm": params["bias_mm"],
        "press_mm": params["press_mm"],
        "completed": not stop["why"],
        "F_ext_frame": "O",
        "stopped": stop["why"],
        **rec,
    }
    if stop["why"]:
        aborted = (args.out or OUT_ROOT / f"{_stamp()}_pivot.json").with_suffix(".aborted.json")
        _write(aborted, payload)
        print(
            f"STOPPED: {stop['why']} -- {len(rec['t'])} samples kept for diagnosis in {aborted.name}, nothing to fit"
        )
        return 4
    out = args.out or OUT_ROOT / f"{_stamp()}_pivot.json"
    _write(out, payload)
    print(f"wrote {out} ({len(rec['t'])} samples)")
    return 0


def guide_target(c0, bias_m, press_m: float, factor: float) -> np.ndarray:
    """Where the spring pulls the held point: the centre, biased, pressed down onto the ball's stand."""
    return np.asarray(c0) + factor * (np.asarray(bias_m) + press_m * PRESS_DIR)


def guide_socket_cmd(target, R: np.ndarray, anchor_m) -> np.ndarray:
    """The socket-frame origin that puts the held point ``R @ anchor + p`` on ``target`` at rotation R.
    With rotation free, the spring then pulls on the held point's error alone."""
    return np.asarray(target) - R @ np.asarray(anchor_m)


def guide_progress(R_start: np.ndarray, Rs: list[np.ndarray]) -> tuple[float, float]:
    """(largest tilt of the cup's axis off vertical, swing range about the vertical) so far, in
    degrees -- what the fit needs. The cup axis is whatever pointed down when the guide started."""
    if not Rs:
        return 0.0, 0.0
    cup = R_start.T @ PRESS_DIR  # in the socket frame
    tilts, spins = [], []
    for R in Rs:
        tilts.append(np.degrees(np.arccos(np.clip((R @ cup) @ PRESS_DIR, -1, 1))))
        rel = R @ R_start.T  # in fr3_base
        spins.append(np.degrees(np.arctan2(rel[1, 0], rel[0, 0])))
    return float(max(tilts)), float(max(spins) - min(spins))


def guide(args) -> int:
    centre = json.loads(args.centre.read_text(encoding="utf-8"))
    if centre.get("schema") != CENTRE_SCHEMA:
        print(f"ERROR: {args.centre} is not a {CENTRE_SCHEMA}")
        return 2
    lo, hi = GUIDE_DURATION_RANGE_S
    if not lo <= args.duration_s <= hi:
        print(f"ERROR: --duration-s must be within {lo:g}-{hi:g} s")
        return 2
    from panda_py import Panda, controllers

    kin = Kinematics(args.urdf)
    panda = Panda(args.robot_ip)
    state = fresh_state(panda)
    q_now = np.asarray(state.q, dtype=float)
    beyond, close = teaching_wall_problems(q_now, PANDA_WALL_LOWER, PANDA_WALL_UPPER)
    T_sock = kin.fk(q_now)
    anchor = np.asarray(args.anchor_mm, dtype=float) / 1e3
    if np.linalg.norm(anchor) * 1e3 > GUIDE_MAX_ANCHOR_MM:
        print(
            f"ERROR: --anchor-mm {list(args.anchor_mm)} is more than {GUIDE_MAX_ANCHOR_MM:g} mm off the socket frame"
        )
        return 2
    # The held point where the centre was taken: the fitted ball centre, if the socket was seated then.
    R_cap = np.asarray(centre.get("R") or T_sock[:3, :3], dtype=float)
    c0 = np.asarray(centre["centre_m"], dtype=float) + R_cap @ anchor
    off = float(np.linalg.norm(T_sock[:3, 3] + T_sock[:3, :3] @ anchor - c0) * 1e3)
    problems = list(beyond)
    if off > START_TOLERANCE_MM:
        problems.append(f"球窝离取到的球心 {off:.1f} mm：取球心之后插件离开过球，重新坐好、重新取球心")
    if problems:
        print("ERROR: not starting:")
        for line in problems:
            print(f"  {line}")
        return 2
    for line in close:
        print(f"WARNING: {line}")

    X = np.linalg.inv(_ee_matrix(state)) @ T_sock  # Desk's EE -> corenetic_socket_ee, constant
    X_inv = np.linalg.inv(X)
    bias = np.asarray(args.bias_mm, dtype=float) / 1e3
    press = args.press_mm / 1e3
    K = np.diag([args.stiffness] * 3 + [0.0] * 3)  # rotation is the operator's
    ctrl = controllers.CartesianImpedance(
        impedance=K, damping_ratio=1.0, nullspace_stiffness=GUIDE_NULLSPACE_STIFFNESS, filter_coeff=1.0
    )
    flags = {"abort": "", "finish": False}

    def _abort(signum, frame):  # noqa: ARG001
        flags["abort"] = "operator"

    def _finish(signum, frame):  # noqa: ARG001
        flags["finish"] = True

    signal.signal(signal.SIGINT, _abort)
    signal.signal(signal.SIGTERM, _abort)
    signal.signal(signal.SIGUSR1, _finish)
    rec: dict[str, list] = {
        "t": [],
        "q": [],
        "dq": [],
        "O_T_EE": [],
        "F_ext": [],
        "cmd_socket": [],
        "phase": [],
    }
    monitor = Monitor(force_n=GUIDE_STOP_FORCE_N)
    R_start = T_sock[:3, :3]
    seen: list[np.ndarray] = []
    print(
        f"guide: up to {args.duration_s:.0f} s, socket held on the ball by {args.stiffness:.0f} N/m, pressed "
        f"{args.press_mm} mm, bias {list(args.bias_mm)} mm, held at {list(args.anchor_mm)} mm in {SOCKET_FRAME}; "
        "rotation free -- by hand, pitch the gripper and roll it about its own axis +-20-30°, and swing it round the ball"
    )
    panda.start_controller(ctrl)
    release_from = None
    try:
        ee0 = _ee_matrix(state)
        ctrl.set_control(ee0[:3, 3], quat_xyzw(ee0[:3, :3]), q_now)
        t0 = time.monotonic()
        next_print = 0.0
        while not flags["abort"]:
            elapsed = time.monotonic() - t0
            if release_from is None and (flags["finish"] or elapsed >= args.duration_s - GUIDE_RAMP_S):
                release_from = elapsed
            if release_from is not None and elapsed >= release_from + GUIDE_RAMP_S:
                break
            if release_from is not None:
                phase, factor = 2, 1.0 - float(smoothstep((elapsed - release_from) / GUIDE_RAMP_S))
            elif elapsed < GUIDE_RAMP_S:
                phase, factor = 0, float(smoothstep(elapsed / GUIDE_RAMP_S))
            else:
                phase, factor = 1, 1.0
            s = panda.get_state()
            q = np.asarray(s.q, dtype=float)
            T_s = _ee_matrix(s) @ X
            target = guide_target(c0, bias, press, factor)
            cmd = guide_socket_cmd(target, T_s[:3, :3], anchor)
            T_t = T_s.copy()
            T_t[:3, 3] = cmd
            T_ee = T_t @ X_inv
            ctrl.set_control(T_ee[:3, 3], quat_xyzw(T_ee[:3, :3]), q)
            f_ext = list(s.O_F_ext_hat_K)
            rec["t"].append(elapsed)
            rec["q"].append(q.tolist())
            rec["dq"].append(list(s.dq))
            rec["O_T_EE"].append(list(s.O_T_EE))
            rec["F_ext"].append(f_ext)
            rec["cmd_socket"].append(cmd.tolist())
            rec["phase"].append(phase)
            if len(rec["t"]) % 10 == 0:
                seen.append(T_s[:3, :3].copy())
            why = monitor.check(q, f_ext, T_s[:3, 3], cmd)
            if why:
                flags["abort"] = why
                break
            panda.raise_error()
            if elapsed >= next_print:
                tilt, spin = guide_progress(R_start, seen)
                margin = np.degrees(np.minimum(q - PANDA_WALL_LOWER, PANDA_WALL_UPPER - q))
                near = (
                    f"  j{int(margin.argmin()) + 1} 离墙 {margin.min():.0f}°!"
                    if margin.min() < GUIDE_WALL_WARN_DEG
                    else ""
                )
                print(
                    f"  {elapsed:5.1f} / {args.duration_s:.0f} s  {GUIDE_PHASES[phase]:7s} press {-f_ext[2]:+4.1f} N  "
                    f"off {np.linalg.norm(T_s[:3, 3] - cmd) * 1e3:4.1f} mm  tilt max {tilt:4.1f}°  spin {spin:5.1f}°{near}",
                    flush=True,
                )
                next_print += 2.0
            time.sleep(1.0 / GUIDE_RATE_HZ)
    except RuntimeError as exc:
        flags["abort"] = f"robot error: {exc}"
    finally:
        panda.stop_controller()

    tilt, spin = guide_progress(R_start, seen)
    payload = {
        "robot_ip": args.robot_ip,
        "note": args.note,
        "created": dt.datetime.now().isoformat(),
        "mode": "guided",
        "plan": "",
        "centre": args.centre.name,
        "bias_mm": [float(x) for x in args.bias_mm],
        "press_mm": args.press_mm,
        "anchor_mm": [float(x) for x in args.anchor_mm],
        "anchor_from": args.anchor_from,
        "completed": not flags["abort"],
        "F_ext_frame": "O",
        "stopped": flags["abort"],
        "tilt_max_deg": round(tilt, 1),
        "spin_range_deg": round(spin, 1),
        **rec,
    }
    out = args.out or OUT_ROOT / f"{_stamp()}_pivot.json"
    if flags["abort"]:
        aborted = out.with_suffix(".aborted.json")
        _write(aborted, payload)
        print(
            f"STOPPED: {flags['abort']} -- {len(rec['t'])} samples kept for diagnosis in {aborted.name}, nothing to fit"
        )
        return 4
    _write(out, payload)
    print(f"wrote {out} ({len(rec['t'])} samples; tilt max {tilt:.0f}°, spin {spin:.0f}°)")
    return 0


def _quat_matrix(x: float, y: float, z: float, w: float) -> np.ndarray:
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("capture", help="read only: the seated socket's FK is the centre")
    p.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--note", default="")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=capture)
    p = sub.add_parser("plan", help="IK + MuJoCo checks of the pivot about a captured centre")
    p.add_argument("centre", type=Path)
    p.add_argument(
        "--star",
        action="store_true",
        help="the 10-13 fixed star (--tilt-deg / --spin-deg) instead of planned dwells",
    )
    p.add_argument("--tilt-deg", type=float, default=PlanParams.tilt_deg, help="--star only")
    p.add_argument("--spin-deg", type=float, default=PlanParams.spin_deg, help="--star only")
    p.add_argument("--dwells", type=int, default=14, help="dwell poses, the start included")
    p.add_argument("--dwell-s", type=float, default=1.5)
    p.add_argument("--speed-deg-s", type=float, default=PlanParams.speed_deg_s)
    p.add_argument("--press-mm", type=float, default=PlanParams.press_mm)
    p.add_argument(
        "--bias-mm", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="deliberate offset, fr3_base"
    )
    p.add_argument(
        "--anchor-mm",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help="the ball centre in corenetic_socket_ee (an earlier fit's s): the plan turns about it",
    )
    p.add_argument("--anchor-from", default="", help="which fit the anchor came from (recorded in the plan)")
    p.add_argument("--table-z-m", type=float, default=DEFAULT_TABLE_Z_M, help="table top in fr3_base")
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--no-mujoco", action="store_true")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=plan_cmd)
    p = sub.add_parser("guide", help="real robot: socket held on the ball, the operator turns it by hand")
    p.add_argument("centre", type=Path)
    p.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    p.add_argument("--duration-s", type=float, default=GUIDE_DEFAULT_DURATION_S)
    p.add_argument("--press-mm", type=float, default=PlanParams.press_mm)
    p.add_argument(
        "--bias-mm", type=float, nargs=3, default=[0.0, 0.0, 0.0], help="deliberate offset, fr3_base"
    )
    p.add_argument("--stiffness", type=float, default=PlanParams.translational_stiffness, help="N/m")
    p.add_argument(
        "--anchor-mm",
        type=float,
        nargs=3,
        default=[0.0, 0.0, 0.0],
        help=f"the point the spring holds, in {SOCKET_FRAME}: a fit's socket_frame.s_ee_mm from the same grip",
    )
    p.add_argument("--anchor-from", default="", help="which fit the anchor came from (recorded in the run)")
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--note", default="")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=guide)
    p = sub.add_parser("execute", help="real robot: follow a passed plan under soft impedance")
    p.add_argument("plan", type=Path)
    p.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--note", default="")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=execute)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
