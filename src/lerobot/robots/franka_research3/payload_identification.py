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

"""Static identification of the FR3 end-effector load (mass + centre of mass) from joint torques.

The arm is the scale: at rest, the link-side torque sensors read ``tau_J = g_arm(q) + tau_load(q) + b``,
where ``g_arm`` is libfranka's gravity model of the bare arm (``Model.gravity`` with a zero load) and
``b`` a constant per-joint sensor offset. ``tau_load`` is linear in ``theta = [m, m*c_x, m*c_y, m*c_z]``
(``c`` = load CoM in the flange frame ``fr3_link8``), so a set of static poses gives a linear least
squares problem in ``theta`` and ``b``.

What it cannot see: inertia (statics only), and errors of ``g_arm`` itself -- those land in the result.
Two runs over the same poses that differ by one known part (``fit_difference``) cancel both ``b`` and
the arm-model error; that difference is how the method's accuracy is checked on the real robot.

Pure numpy, no robot or simulator imports: the same code fits MuJoCo data and real data.
"""

# ruff: noqa: N806  (matrix names: T, J, R, A follow the maths)
from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import numpy as np

GRAVITY_EARTH = (0.0, 0.0, -9.81)  # libfranka's default; the base is mounted upright

# FR3 kinematics, Craig (modified) DH: (a_{i-1}, d_i, alpha_{i-1}) per joint, then the flange.
# Same numbers as Franka's documentation and the arm chain of fr3_corenetic_gripper.urdf.
_DH = (
    (0.0, 0.333, 0.0),
    (0.0, 0.0, -math.pi / 2),
    (0.0, 0.316, math.pi / 2),
    (0.0825, 0.0, math.pi / 2),
    (-0.0825, 0.384, -math.pi / 2),
    (0.0, 0.0, math.pi / 2),
    (0.088, 0.0, math.pi / 2),
)
_FLANGE_D = 0.107


def _dh(a: float, d: float, alpha: float, theta: float) -> np.ndarray:
    ca, sa, ct, st = math.cos(alpha), math.sin(alpha), math.cos(theta), math.sin(theta)
    return np.array(
        [
            [ct, -st, 0.0, a],
            [st * ca, ct * ca, -sa, -d * sa],
            [st * sa, ct * sa, ca, d * ca],
            [0.0, 0.0, 0.0, 1.0],
        ]
    )


def flange_kinematics(q) -> tuple[np.ndarray, np.ndarray]:
    """Base->flange (fr3_link8) transform and the 6x7 geometric Jacobian [Jv; Jw] at the flange origin."""
    q = np.asarray(q, dtype=np.float64).reshape(7)
    T = np.eye(4)
    axes, origins = [], []
    for (a, d, alpha), theta in zip(_DH, q, strict=True):
        T = T @ _dh(a, d, alpha, theta)
        axes.append(T[:3, 2].copy())
        origins.append(T[:3, 3].copy())
    T = T @ _dh(0.0, _FLANGE_D, 0.0, 0.0)
    p = T[:3, 3]
    J = np.zeros((6, 7))
    for i, (z, o) in enumerate(zip(axes, origins, strict=True)):
        J[:3, i] = np.cross(z, p - o)
        J[3:, i] = z
    return T, J


def _skew(v: np.ndarray) -> np.ndarray:
    return np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])


def load_regressor(q, gravity=GRAVITY_EARTH) -> np.ndarray:
    """7x4 matrix Y with tau_load = Y @ [m, m*cx, m*cy, m*cz] (c in the flange frame, metres).

    tau_load is the joint torque that holds the load: -J^T w, with w the load's weight wrench at the
    flange origin, F = m g and M = (R m c) x g.
    """
    T, J = flange_kinematics(q)
    g = np.asarray(gravity, dtype=np.float64)
    Jv, Jw = J[:3], J[3:]
    return np.column_stack([-Jv.T @ g, Jw.T @ _skew(g) @ T[:3, :3]])


@dataclass
class LoadEstimate:
    mass_kg: float
    com_flange_m: np.ndarray
    mass_std_kg: float
    com_std_m: np.ndarray
    joint_bias_nm: np.ndarray | None  # (groups, 7); None when not estimated (differences)
    residual_rms_nm: np.ndarray  # per joint
    condition_number: float
    n_poses: int

    def as_dict(self) -> dict:
        return {
            "mass_kg": round(self.mass_kg, 5),
            "mass_std_kg": round(self.mass_std_kg, 5),
            "com_flange_mm": np.round(self.com_flange_m * 1000, 2).tolist(),
            "com_std_mm": np.round(self.com_std_m * 1000, 2).tolist(),
            "joint_bias_nm": None if self.joint_bias_nm is None else np.round(self.joint_bias_nm, 4).tolist(),
            "residual_rms_nm": np.round(self.residual_rms_nm, 4).tolist(),
            "condition_number": round(self.condition_number, 1),
            "n_poses": self.n_poses,
        }


def _solve(qs, residuals, gravity, with_bias: bool, groups=None) -> LoadEstimate:
    qs = np.asarray(qs, dtype=np.float64).reshape(-1, 7)
    r = np.asarray(residuals, dtype=np.float64).reshape(-1, 7)
    labels = np.zeros(len(qs), dtype=int) if groups is None else np.unique(groups, return_inverse=True)[1]
    n_groups = int(labels.max()) + 1
    blocks = []
    for q, k in zip(qs, labels, strict=True):
        Y = load_regressor(q, gravity)
        if with_bias:
            B = np.zeros((7, 7 * n_groups))
            B[:, 7 * k : 7 * k + 7] = np.eye(7)
            Y = np.hstack([Y, B])
        blocks.append(Y)
    A = np.vstack(blocks)
    y = r.reshape(-1)
    if A.shape[0] <= A.shape[1]:
        raise ValueError(f"{len(qs)} poses give {A.shape[0]} equations for {A.shape[1]} unknowns")
    x, *_ = np.linalg.lstsq(A, y, rcond=None)
    e = (y - A @ x).reshape(-1, 7)
    dof = A.shape[0] - A.shape[1]
    cov = (e.ravel() @ e.ravel() / dof) * np.linalg.inv(A.T @ A)
    m, mc = x[0], x[1:4]
    c = mc / m
    # delta method for c = mc / m
    G = np.zeros((3, 4))
    G[:, 0] = -mc / m**2
    G[:, 1:] = np.eye(3) / m
    c_cov = G @ cov[:4, :4] @ G.T
    return LoadEstimate(
        mass_kg=float(m),
        com_flange_m=c,
        mass_std_kg=float(math.sqrt(cov[0, 0])),
        com_std_m=np.sqrt(np.diag(c_cov)),
        joint_bias_nm=x[4:].reshape(n_groups, 7) if with_bias else None,
        residual_rms_nm=np.sqrt((e**2).mean(axis=0)),
        condition_number=float(np.linalg.cond(A)),
        n_poses=len(qs),
    )


def fit_load(qs, tau_j, g_arm, gravity=GRAVITY_EARTH, with_bias: bool = True, groups=None) -> LoadEstimate:
    """Total load hanging on the flange from static poses (q, measured tau_J, bare-arm gravity).

    ``groups`` (one label per pose) gives each group its own joint offsets. With the upper arm (j1-j4)
    fixed inside a group, the bare-arm model's error there is a constant too and goes into those offsets
    instead of into the load -- at the price of a less certain mass (see ``default_poses``).
    """
    return _solve(qs, np.asarray(tau_j) - np.asarray(g_arm), gravity, with_bias, groups)


def fit_difference(qs, tau_j_a, tau_j_b, gravity=GRAVITY_EARTH) -> LoadEstimate:
    """Load present in run A but not in run B, over the same poses (bias and arm-model error cancel)."""
    return _solve(qs, np.asarray(tau_j_a) - np.asarray(tau_j_b), gravity, with_bias=False)


def _wrist_poses(j2: float, j4: float, tilts, spins) -> list[np.ndarray]:
    """j1 = j3 = 0; j6 = j2 - j4 points the tool straight down, then (j5, j6) tilt it and j7 spins it."""
    out = []
    for k, (d5, d6) in enumerate(tilts):
        for s in spins if k % 2 == 0 else spins[::-1]:
            out.append(np.array([0.0, j2, 0.0, j4, d5, j2 - j4 + d6, s]))
    return out


def default_poses() -> np.ndarray:
    """32 static poses that tilt the flange in every direction from three arm reaches.

    Wrist (20 poses at panda_py's start pose): j5/j6 tilt the tool up to 0.8 rad off vertical and j7
    spins it, which separates m*cz from the horizontal CoM components. j6 tilts only 0.5 rad towards the
    forearm: at 0.8 the SMR/camera adaptor passes 10 mm from fr3_link5 (MuJoCo surface check,
    `fr3_payload_identification_runtime.py check-poses`).

    Reach (6 poses each, flange 465 mm and 230 mm out instead of 307 mm): with the upper arm fixed, the
    load's weight loads j2/j4 almost identically in every pose and the mass is confounded with those
    joints' sensor offsets (1-sigma 27 g at 0.02 Nm noise); changing the reach brings it to ~5 g.
    But it also lets the bare-arm model's error into the result (MuJoCo: link masses off by 1 % ->
    load -55 g, with residuals that look like noise). Fit both ways -- one set of offsets, and offsets
    per reach (``reach_groups``) -- and only trust the precise one when the two agree.
    """
    start_j2, start_j4 = -math.pi / 4, -3 * math.pi / 4
    poses = _wrist_poses(
        start_j2,
        start_j4,
        ((0.0, 0.0), (0.0, 0.8), (0.8, 0.0), (0.0, -0.5), (-0.8, 0.0)),
        (-2.0, -0.6, 0.8, 2.2),
    )
    for j2, j4 in ((-0.3, -2.0), (-1.1, -2.7)):
        poses += _wrist_poses(j2, j4, ((0.0, 0.0), (0.0, 0.8), (0.8, 0.0)), (-0.6, 2.2))
    return np.array(poses)


def visit_sequence(poses, approach_rad: float, lower, upper) -> list[tuple[np.ndarray, str | None, int]]:
    """Moves for a run: ``(target q, measurement tag or None, pose index)``.

    Static friction in the joints holds part of the load, so tau_J at rest depends on the direction each
    joint last moved in (10-12 on the FR3: +-0.1..0.4 Nm per joint, as large as the SMR's whole signal).
    Each pose is therefore measured twice -- reached with every joint moving up (from q - approach_rad)
    and with every joint moving down (from q + approach_rad) -- and the mean cancels that hysteresis.
    ``approach_rad = 0`` gives the plain one-measurement-per-pose sequence.
    """
    lower, upper = np.asarray(lower, dtype=np.float64), np.asarray(upper, dtype=np.float64)
    out: list[tuple[np.ndarray, str | None, int]] = []
    for i, q in enumerate(np.asarray(poses, dtype=np.float64)):
        if approach_rad <= 0:
            out.append((q, "single", i))
            continue
        for tag, sign in (("up", -1.0), ("down", 1.0)):
            pre = np.clip(q + sign * approach_rad, lower, upper)
            if np.any(np.abs(pre - q) < 0.5 * approach_rad):
                raise ValueError(f"pose {i} is within {approach_rad} rad of a joint limit")
            out.append((pre, None, i))
            out.append((q, tag, i))
    return out


def reach_groups(qs) -> np.ndarray:
    """Label poses by their upper-arm configuration (j1-j4), for per-reach joint offsets."""
    keys = np.round(np.asarray(qs)[:, :4], 2)
    return np.unique(keys, axis=0, return_inverse=True)[1].ravel()


def _rpy(r: float, p: float, y: float) -> np.ndarray:
    cr, sr, cp, sp, cy, sy = (math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y))
    return np.array(
        [
            [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
            [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
            [-sp, cp * sr, cp * cr],
        ]
    )


def urdf_link_load(urdf_path: str | Path, link: str, flange: str = "fr3_link8") -> tuple[float, np.ndarray]:
    """Mass and CoM (flange frame, m) of a URDF link rigidly attached below ``flange``."""
    root = ET.parse(urdf_path).getroot()
    parent_joint = {j.find("child").get("link"): j for j in root.findall("joint")}
    R, t, cur = np.eye(3), np.zeros(3), link
    while cur != flange:
        j = parent_joint.get(cur)
        if j is None:
            raise ValueError(f"{link} does not hang below {flange}")
        if j.get("type") != "fixed":
            raise ValueError(f"{link} is not rigidly attached to {flange} (joint {j.get('name')})")
        o = j.find("origin")
        Rj = _rpy(*map(float, o.get("rpy", "0 0 0").split()))
        tj = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
        R, t = Rj @ R, Rj @ t + tj
        cur = j.find("parent").get("link")
    inertial = root.find(f"link[@name='{link}']/inertial")
    if inertial is None:
        raise ValueError(f"{link} has no inertial")
    com = np.array([float(v) for v in inertial.find("origin").get("xyz", "0 0 0").split()])
    return float(inertial.find("mass").get("value")), R @ com + t


def urdf_subtree_inertial(
    urdf_path: str | Path, root: str = "fr3_link8", exclude: tuple[str, ...] = ()
) -> tuple[float, np.ndarray, np.ndarray]:
    """Mass, CoM and inertia about the CoM of everything below ``root``, in ``root``'s frame.

    Movable joints are taken at 0 (the gripper fingers). Links in ``exclude`` are left out with their
    children.
    """
    root_el = ET.parse(urdf_path).getroot()
    children: dict[str, list] = {}
    for j in root_el.findall("joint"):
        children.setdefault(j.find("parent").get("link"), []).append(j)
    bodies = []
    stack = [(root, np.eye(3), np.zeros(3))]
    while stack:
        link, R, t = stack.pop()
        inertial = root_el.find(f"link[@name='{link}']/inertial")
        if link != root and inertial is not None:
            o = inertial.find("origin")
            Ro = _rpy(*map(float, o.get("rpy", "0 0 0").split())) if o is not None else np.eye(3)
            co = np.array([float(v) for v in o.get("xyz", "0 0 0").split()]) if o is not None else np.zeros(3)
            i = {k: float(v) for k, v in inertial.find("inertia").attrib.items()}
            I_local = np.array(
                [
                    [i["ixx"], i["ixy"], i["ixz"]],
                    [i["ixy"], i["iyy"], i["iyz"]],
                    [i["ixz"], i["iyz"], i["izz"]],
                ]
            )
            Rw = R @ Ro
            bodies.append((float(inertial.find("mass").get("value")), R @ co + t, Rw @ I_local @ Rw.T))
        for j in children.get(link, []):
            child = j.find("child").get("link")
            if child in exclude:
                continue
            o = j.find("origin")
            Rj = _rpy(*map(float, o.get("rpy", "0 0 0").split()))
            tj = np.array([float(v) for v in o.get("xyz", "0 0 0").split()])
            stack.append((child, R @ Rj, R @ tj + t))
    m = sum(b[0] for b in bodies)
    c = sum(b[0] * b[1] for b in bodies) / m
    inertia = np.zeros((3, 3))
    for mi, ci, Ii in bodies:
        d = ci - c
        inertia += Ii + mi * (d @ d * np.eye(3) - np.outer(d, d))
    return m, c, inertia


def subtract_known(estimate: LoadEstimate, parts: list[tuple[float, np.ndarray]]) -> tuple[float, np.ndarray]:
    """Mass and CoM (flange frame) of what is left after removing known parts from an estimate."""
    m = estimate.mass_kg - sum(p[0] for p in parts)
    if m <= 0:
        raise ValueError(f"known parts ({sum(p[0] for p in parts):.3f} kg) outweigh the estimate")
    mc = estimate.mass_kg * estimate.com_flange_m - sum(p[0] * np.asarray(p[1]) for p in parts)
    return m, mc / m
