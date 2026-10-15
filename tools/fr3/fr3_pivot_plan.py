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
The FR3 pivot planned as dwells, for ``fr3_pivot_auto.py plan``.

  survey  a grid of socket orientations about the ball -- heading of the tool axis, pitch (tool tip up +),
          roll about the tool axis -- each solved by IK by continuation from the captured pose and checked:
          FR3 limits and panda_py's walls, the table, MuJoCo self-contact, and the insert's socket and the
          gripper against the ball's stand (stand and socket from the insert CAD, the gripper's own meshes)
  dwells  from the cells that pass, the N whose fit of (s, c) has the smallest worst 1-sigma (greedy, then
          exchange), as a multiple of the per-dwell error
  route   the dwells in the order that moves the joints least (j6 counted twice), each move through cells
          that passed, timed so that neither the tool nor any joint exceeds its speed

Why dwells and not the 10-13 star: a fixed ±tilt/±spin ignores where the stand and the walls are (j2 past
its wall, link4 under the table), and a sweep only ever samples the edge of what the arm can reach. The
grid finds what is reachable from the captured posture; the dwells then spend the time where it pins s.

The stand model was checked on the 10-13 / 10-14 guide pivots (21 runs, 9.8k samples): no recorded pose is
in a predicted collision; the hands stopped where the model leaves 0.8-1.6 mm (roll -44° / +39°: the
fingers on the neck); and the samples where the model puts the socket on the neck fit twice as badly
(residual p50 1.9 mm against 1.0). The real fingertips stay above the socket's bottom edge (-13 vs -16 mm).
"""

# ruff: noqa: N802, N803, N806  (matrix names: R, T, F follow the maths)
from __future__ import annotations

import heapq
import json
import math
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tools.fr3.fr3_pivot_auto import (
    DEFAULT_TABLE_Z_M,
    IK_LOWER,
    IK_UPPER,
    LIMIT_MARGIN_DEG,
    MAX_J6_SPEED_DEG_S,
    MAX_J6_TRAVEL_DEG,
    MAX_JOINT_SPEED_DEG_S,
    MIN_TABLE_CLEARANCE_MM,
    PANDA_WALL_LOWER,
    PANDA_WALL_UPPER,
    PRESS_DIR,
    Kinematics,
    SceneCheck,
    rot_angle,
    smoothstep,
)
from tools.fr3.fr3_tcp_pivot import FR3_JOINT_LIMITS

REPO = Path(__file__).resolve().parents[2]
ASSET_DIR = REPO / "src/lerobot/robots/franka_research3/assets/franka_fr3/pivot_insert_v2"
STAND_SCHEMA = "fr3_pivot_stand/v1"

# The insert frame of the CAD (ball centre at the origin, z up the cup axis) in corenetic_socket_ee axes:
# z = EE x (the cup axis, up when seated), x = -EE z (back towards the flange), y = EE y. Checked on the
# recorded pivots: EE x points up within the tilt the hand gave it.
INSERT_FROM_EE = np.array([[0.0, 0.0, -1.0], [0.0, 1.0, 0.0], [1.0, 0.0, 0.0]])
# The insert is held by the fingertips 35 mm apart (pad faces).
INSERT_PAD_GAP_MM = 35.0

# Margins. The socket model reads true to well under a millimetre (socket on the neck where it says so);
# the gripper meshes are vendor meshes, and the hands stopped where they left about 1 mm -- so more there.
SOCKET_CLEARANCE_MM = 1.0
SOCKET_DWELL_CLEARANCE_MM = 1.5
GRIPPER_CLEARANCE_MM = 2.0
GRIPPER_DWELL_CLEARANCE_MM = 3.0
DWELL_LIMIT_MARGIN_DEG = 6.0
# Clearances are only needed up to this: anything further from the stand reads as this.
CLEARANCE_CAP_MM = 20.0
SDF_STEP_MM = 0.1
IK_TOL_MM = 0.05
IK_TOL_DEG = 0.05
# Pulled back towards the middle of their range by the IK nullspace: joints this close to a limit.
IK_KEEP_OFF_DEG = 30.0
J6_WEIGHT = 2.0
# j6 over all dwells; the moves between them may take it this much further either way.
J6_ROUTE_SLACK_DEG = 5.0
J6_DWELL_SPAN_DEG = MAX_J6_TRAVEL_DEG - 2 * J6_ROUTE_SLACK_DEG
# Two neighbouring cells (7.5° apart at most per axis) whose solutions differ by more than this in any
# joint are on different posture branches (a wrist flip): never a move. Near the wrist singularity a
# 7.5° step honestly turns j5/j7 by 20-35° (10-14, pitch -20°); the densified move checks the rest.
EDGE_MAX_DQ_DEG = 45.0
# A captured pose already closer to the stand than the margins (10-14 16:21: the socket 0.6 mm into the
# neck) is where the arm sits anyway: moves may come that close again, not closer. Not so the table or
# the limits: a capture too near those is taken again.
START_SLACK_MM = 0.2
# Moves are solved at this much tool turn per sample; a joint stepping more than DENSE_MAX_DQ_DEG
# between two samples has jumped branch. Every DENSE_FULL_EVERY-th sample gets the MuJoCo checks too.
DENSE_STEP_DEG = 0.5
DENSE_MAX_DQ_DEG = 5.0
DENSE_FULL_EVERY = 4
# fit_pivot needs six orientations; fewer dwells than this is no plan.
MIN_DWELLS = 8


# --- orientations about the ball -------------------------------------------------------------------


def _rx(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1.0, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1.0, 0], [-s, 0, c]])


def _rz(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


def socket_rotation(heading: float, pitch: float, roll: float) -> np.ndarray:
    """corenetic_socket_ee in fr3_base (radians): the tool axis heads ``heading`` (azimuth in fr3_base),
    its tip ``pitch`` above level, the gripper rolled ``roll`` about the tool axis (0: the cup axis as
    near vertical as the pitch allows). The stand is turned about the vertical, so only pitch and roll
    bring the insert nearer to it."""
    return _rz(heading + math.pi) @ _ry(pitch) @ _rx(roll) @ INSERT_FROM_EE


def socket_angles(R) -> np.ndarray:
    """(heading, pitch, roll) in radians of corenetic_socket_ee rotations (n, 3, 3) -> (n, 3)."""
    R = np.asarray(R, dtype=float).reshape(-1, 3, 3)
    up = np.einsum("ij,nkj,k->ni", INSERT_FROM_EE, R, np.array([0.0, 0.0, 1.0]))  # the vertical, insert frame
    pitch = -np.arcsin(np.clip(up[:, 0], -1.0, 1.0))
    roll = np.arctan2(up[:, 1], up[:, 2])
    out = np.zeros((len(R), 3))
    for i in range(len(R)):
        A = R[i] @ INSERT_FROM_EE.T @ (_ry(pitch[i]) @ _rx(roll[i])).T
        out[i] = [math.remainder(math.atan2(A[1, 0], A[0, 0]) - math.pi, 2 * math.pi), pitch[i], roll[i]]
    return out


# --- the ball's stand ------------------------------------------------------------------------------


def _read_stl(path: Path) -> np.ndarray:
    """Triangles (n, 3, 3) of a binary STL."""
    raw = path.read_bytes()
    n = int(np.frombuffer(raw, dtype="<u4", count=1, offset=80)[0])
    rec = np.dtype([("normal", "<f4", 3), ("v", "<f4", (3, 3)), ("attr", "<u2")])
    return np.frombuffer(raw, dtype=rec, count=n, offset=84)["v"].astype(float)


def _surface_points(tris: np.ndarray, spacing: float) -> np.ndarray:
    """Points on every triangle no further apart than about ``spacing`` (barycentric grid)."""
    out = []
    for a, b, c in tris:
        k = max(
            1,
            int(
                math.ceil(max(np.linalg.norm(b - a), np.linalg.norm(c - a), np.linalg.norm(c - b)) / spacing)
            ),
        )
        i, j = np.meshgrid(np.arange(k + 1), np.arange(k + 1), indexing="ij")
        keep = i + j <= k
        u, v = i[keep] / k, j[keep] / k
        out.append(a + u[:, None] * (b - a) + v[:, None] * (c - a))
    return np.unique(np.round(np.concatenate(out), 3), axis=0)


def _profile_sd(rho: np.ndarray, z: np.ndarray, poly: np.ndarray) -> np.ndarray:
    """Signed distance (inside < 0) of (rho, z) points to a closed polygon in the (rho, z) half-plane."""
    P = np.stack([rho, z], axis=1)[:, None, :]
    A = poly[None]
    AB = (np.roll(poly, -1, axis=0) - poly)[None]
    t = np.clip(((P - A) * AB).sum(-1) / np.maximum((AB * AB).sum(-1), 1e-12), 0.0, 1.0)
    d = np.sqrt(((P - A - t[..., None] * AB) ** 2).sum(-1).min(axis=1))
    crosses = (A[..., 1] > P[..., 1]) != ((A + AB)[..., 1] > P[..., 1])
    x_at = A[..., 0] + (P[..., 1] - A[..., 1]) * AB[..., 0] / np.where(AB[..., 1] == 0, 1e-12, AB[..., 1])
    inside = (crosses & (P[..., 0] < x_at)).sum(axis=1) % 2 == 1
    return np.where(inside, -d, d)


@dataclass
class Stand:
    """The ball's stand (axisymmetric about the vertical through the ball) and the insert's socket, mm."""

    profile_rz: np.ndarray
    plate_radius: float
    plate_top: float
    plate_bottom: float
    socket_points: np.ndarray  # insert frame

    @classmethod
    def load(cls, asset_dir: Path = ASSET_DIR, spacing_mm: float = 1.5) -> Stand:
        spec = json.loads((asset_dir / "stand.json").read_text(encoding="utf-8"))
        if spec.get("schema") != STAND_SCHEMA:
            raise ValueError(f"{asset_dir / 'stand.json'} is not a {STAND_SCHEMA}")
        plate = spec["plate"]
        stand = cls(
            profile_rz=np.asarray(spec["profile_rz"], dtype=float),
            plate_radius=float(plate["half_xy"]) * math.sqrt(2.0),  # its heading is not known
            plate_top=float(plate["z_top"]),
            plate_bottom=float(plate["z_bottom"]),
            socket_points=_surface_points(_read_stl(asset_dir / "socket.stl"), spacing_mm),
        )
        stand._build_field()
        stand.socket_points = stand._closest_socket_points()
        return stand

    def _build_field(self, step_mm: float = SDF_STEP_MM) -> None:
        """The profile's signed distance tabulated on a (rho, z) grid, read back bilinearly: the polygon
        itself is ~2 ms per few thousand points, the table ~0.1 ms."""
        cap = CLEARANCE_CAP_MM
        self._rho = np.arange(0.0, self.profile_rz[:, 0].max() + cap + 2 * step_mm, step_mm)
        self._z = np.arange(
            self.profile_rz[:, 1].min() - cap - step_mm,
            self.profile_rz[:, 1].max() + cap + 2 * step_mm,
            step_mm,
        )
        rr, zz = np.meshgrid(self._rho, self._z, indexing="ij")
        flat = np.empty(rr.size)
        for i in range(0, rr.size, 50000):
            flat[i : i + 50000] = _profile_sd(
                rr.reshape(-1)[i : i + 50000], zz.reshape(-1)[i : i + 50000], self.profile_rz
            )
        self._field = np.minimum(flat.reshape(rr.shape), cap)
        self._step = step_mm

    def _profile_clearance(self, rho: np.ndarray, z: np.ndarray) -> np.ndarray:
        fi = np.clip((rho - self._rho[0]) / self._step, 0.0, len(self._rho) - 1.001)
        fj = np.clip((z - self._z[0]) / self._step, 0.0, len(self._z) - 1.001)
        i, j = fi.astype(int), fj.astype(int)
        u, v = fi - i, fj - j
        F = self._field
        return (1 - u) * ((1 - v) * F[i, j] + v * F[i, j + 1]) + u * (
            (1 - v) * F[i + 1, j] + v * F[i + 1, j + 1]
        )

    def _closest_socket_points(self, keep_mm: float = 3.0) -> np.ndarray:
        """Only the socket points that come within ``keep_mm`` of the closest one at some tilt of a 5°
        grid (pitch ±40°, roll ±60°): between grid tilts a point moves at most ~1.1 mm, so the closest
        point anywhere in that range is among them. Saves most of the work in every clearance."""
        keep = np.zeros(len(self.socket_points), dtype=bool)
        for pitch in np.radians(np.arange(-40, 41, 5)):
            for roll in np.radians(np.arange(-60, 61, 5)):
                c = self.clearance(self.socket_points @ (_ry(pitch) @ _rx(roll)).T)
                if c.min() < CLEARANCE_CAP_MM:
                    keep |= c <= c.min() + keep_mm
        return self.socket_points[keep]

    def clearance(self, pts_mm: np.ndarray, floor_z_mm: float | None = None) -> np.ndarray:
        """Per point, the distance to the stand (negative: inside), at most CLEARANCE_CAP_MM. Points in
        the stand frame: the ball centre at the origin, z up. Below the plate, down to ``floor_z_mm`` (the
        table, relative to the ball), whatever holds the stand up is taken to be no wider than the plate."""
        cap = CLEARANCE_CAP_MM
        pts = np.asarray(pts_mm, dtype=float).reshape(-1, 3)
        z = pts[:, 2]
        out = np.full(len(pts), cap)
        near = z < max(self.profile_rz[:, 1].max(), self.plate_top) + cap
        if not near.any():
            return out
        rho = np.full(len(pts), np.inf)
        rho[near] = np.linalg.norm(pts[near, :2], axis=1)
        prof = near & (z < self.profile_rz[:, 1].max() + cap) & (rho < self.profile_rz[:, 0].max() + cap)
        if prof.any():
            out[prof] = np.minimum(out[prof], self._profile_clearance(rho[prof], z[prof]))
        plate = near & (rho < self.plate_radius + cap)
        if plate.any():
            bottom = self.plate_bottom if floor_z_mm is None else min(self.plate_bottom, floor_z_mm)
            dr = rho[plate] - self.plate_radius
            dz = np.maximum(bottom - z[plate], z[plate] - self.plate_top)
            d = np.where(
                (dr > 0) | (dz > 0), np.hypot(np.maximum(dr, 0.0), np.maximum(dz, 0.0)), np.maximum(dr, dz)
            )
            out[plate] = np.minimum(out[plate], d)
        return out

    def socket_clearance(self, R_socket: np.ndarray) -> float:
        """The socket seated on the ball, turned to ``R_socket`` (corenetic_socket_ee in fr3_base)."""
        pts = self.socket_points @ (np.asarray(R_socket) @ INSERT_FROM_EE.T).T
        return float(self.clearance(pts).min())


class StandCheck:
    """The gripper and the arm's last links against the stand, from the MuJoCo meshes, fingers on the insert."""

    def __init__(self, scene: SceneCheck, stand: Stand, max_vertices: int = 2000):
        import mujoco

        self.mj = mujoco
        self.scene = scene
        self.stand = stand
        m = scene.model
        self._gripper_open(INSERT_PAD_GAP_MM)
        moving = self._descendants("fr3_link5")
        self.geoms = []  # (geom id, body name, local vertices)
        seen = set()
        for g in range(m.ngeom):
            b = int(m.geom_bodyid[g])
            if b not in moving or m.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            mid = int(m.geom_dataid[g])
            if (b, mid) in seen:  # the visual and the collision copy of one mesh
                continue
            seen.add((b, mid))
            v = m.mesh_vert[m.mesh_vertadr[mid] : m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            if len(v) > max_vertices:
                v = v[np.linspace(0, len(v) - 1, max_vertices).astype(int)]
            self.geoms.append((g, m.body(b).name, np.asarray(v, dtype=float)))
        self._ends = np.cumsum([len(v) for _, _, v in self.geoms])

    def _descendants(self, root: str) -> set[int]:
        m = self.scene.model
        r = m.body(root).id
        out = set()
        for b in range(m.nbody):
            a = b
            while a > 0 and a != r:
                a = int(m.body_parentid[a])
            if a == r:
                out.add(b)
        return out

    def _gripper_open(self, gap_mm: float) -> None:
        """Set the gripper joints (all mimic the drive) so the pads are ``gap_mm`` apart, in the scene's qpos."""
        m, d = self.scene.model, self.scene.data
        names = [m.joint(j).name for j in range(m.njnt) if m.joint(j).name.startswith("joint_gripper")]
        if not names:
            return
        adr = [m.jnt_qposadr[m.joint(n).id] for n in names]
        sites = [m.site(n).id for n in ("link_gripper_contact_left_site", "link_gripper_contact_right_site")]
        lo, hi = 0.0, 1.4

        def gap(a: float) -> float:
            for i in adr:
                d.qpos[i] = a
            self.mj.mj_kinematics(m, d)
            return float(np.linalg.norm(d.site_xpos[sites[0]] - d.site_xpos[sites[1]]) * 1e3)

        for _ in range(40):
            mid = 0.5 * (lo + hi)
            lo, hi = (mid, hi) if gap(mid) < gap_mm else (lo, mid)
        gap(lo)
        self.gripper_qpos = dict.fromkeys(names, lo)

    def clearance(self, q, ball_m, floor_z_mm: float | None = None) -> tuple[float, str]:
        """Smallest distance (mm) from the gripper and links 5-7 to the stand under the ball, and whose."""
        sc = self.scene
        for adr, x in zip(sc.qadr, q, strict=True):
            sc.data.qpos[adr] = x
        self.mj.mj_kinematics(sc.model, sc.data)
        d = sc.data
        ball = np.asarray(ball_m, dtype=float)
        pts = np.concatenate([v @ d.geom_xmat[g].reshape(3, 3).T + d.geom_xpos[g] for g, _, v in self.geoms])
        pts = (pts - ball) * 1e3
        c = self.stand.clearance(pts, floor_z_mm)
        i = int(np.argmin(c))
        return float(c[i]), self.geoms[int(np.searchsorted(self._ends, i, side="right"))][1]


# --- one pose, every check -------------------------------------------------------------------------


class PoseCheck:
    """All the checks of one arm posture holding the socket on the ball, as reasons (empty: it passes)."""

    def __init__(
        self,
        q0: np.ndarray,
        R0: np.ndarray,
        ball_m: np.ndarray,
        scene: SceneCheck | None,
        stand: Stand | None,
        table_z_m: float = DEFAULT_TABLE_Z_M,
    ):
        self.q0 = np.asarray(q0, dtype=float)
        self.ball = np.asarray(ball_m, dtype=float)
        self.scene = scene
        self.stand = stand
        # the fingers opened onto the insert first: the contact baseline is taken with them so
        self.stand_check = StandCheck(scene, stand) if scene is not None and stand is not None else None
        if scene is not None:
            scene.baseline = scene.contacts(self.q0)
        self.floor_mm = (table_z_m - self.ball[2]) * 1e3
        self._fr3 = np.asarray(FR3_JOINT_LIMITS)
        self.min_mm = {
            "socket": SOCKET_CLEARANCE_MM,
            "gripper": GRIPPER_CLEARANCE_MM,
            "table": MIN_TABLE_CLEARANCE_MM,
        }
        self.start_why, start = self.check(R0, self.q0)
        self.start = start
        for key in ("socket", "gripper"):
            if f"{key}_mm" in start:
                self.min_mm[key] = min(self.min_mm[key], start[f"{key}_mm"] - START_SLACK_MM)

    def margin_deg(self, q: np.ndarray) -> np.ndarray:
        f = self._fr3
        return np.degrees(
            np.minimum(
                np.minimum(q - f[:, 0], f[:, 1] - q), np.minimum(q - PANDA_WALL_LOWER, PANDA_WALL_UPPER - q)
            )
        )

    def check(self, R: np.ndarray, q: np.ndarray, *, full: bool = True) -> tuple[list[str], dict[str, Any]]:
        """Reasons the pose fails (transit margins) and its numbers. ``full=False``: the cheap ones only
        (limits, j6 travel, the socket on the stand)."""
        why: list[str] = []
        m = self.margin_deg(q)
        out: dict[str, Any] = {"margin_deg": float(m.min())}
        if m.min() < LIMIT_MARGIN_DEG:
            why.append(f"limit: j{int(m.argmin()) + 1} {m.min():.1f}°")
        if abs(math.degrees(q[5] - self.q0[5])) > MAX_J6_TRAVEL_DEG:
            why.append(f"j6 travel: {math.degrees(q[5] - self.q0[5]):+.0f}° from the start")
        if self.stand is not None:
            out["socket_mm"] = self.stand.socket_clearance(R)
            if out["socket_mm"] < self.min_mm["socket"]:
                why.append(f"socket on the stand: {out['socket_mm']:.1f} mm")
        if not full:
            return why, out
        if self.scene is not None:
            new = self.scene.new_contacts(q)
            if new:
                out["contact"] = "/".join(sorted(new)[0])
                why.append(f"self-contact: {out['contact']}")
            out["table_mm"] = self.scene.clearance(q)[0] * 1e3
            if out["table_mm"] < self.min_mm["table"]:
                why.append(f"table: {out['table_mm']:.0f} mm")
        if self.stand_check is not None:
            out["gripper_mm"], out["gripper_body"] = self.stand_check.clearance(q, self.ball, self.floor_mm)
            if out["gripper_mm"] < self.min_mm["gripper"]:
                why.append(f"gripper on the stand: {out['gripper_body']} {out['gripper_mm']:.1f} mm")
        return why, out

    def dwell_ok(self, out: dict[str, Any]) -> bool:
        """The stricter margins of a pose held while pressed (``out`` from a passing ``check``)."""
        return (
            out["margin_deg"] >= DWELL_LIMIT_MARGIN_DEG
            and out.get("socket_mm", math.inf) >= SOCKET_DWELL_CLEARANCE_MM
            and out.get("gripper_mm", math.inf) >= GRIPPER_DWELL_CLEARANCE_MM
        )


# --- the survey ------------------------------------------------------------------------------------


@dataclass
class SurveyParams:
    heading_span_deg: float = 90.0  # either side of the captured heading
    heading_step_deg: float = 7.5
    pitch_deg: tuple[float, float] = (-20.0, 10.0)
    pitch_step_deg: float = 5.0
    roll_deg: tuple[float, float] = (-40.0, 40.0)
    roll_step_deg: float = 5.0
    table_z_m: float = DEFAULT_TABLE_Z_M

    def axes(self, heading0_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        n = int(round(self.heading_span_deg / self.heading_step_deg))
        h = heading0_deg + self.heading_step_deg * np.arange(-n, n + 1)
        p = np.arange(self.pitch_deg[0], self.pitch_deg[1] + 1e-9, self.pitch_step_deg)
        r = np.arange(self.roll_deg[0], self.roll_deg[1] + 1e-9, self.roll_step_deg)
        return h, p, r


@dataclass
class Survey:
    heading_deg: np.ndarray
    pitch_deg: np.ndarray
    roll_deg: np.ndarray
    centre_m: np.ndarray  # the ball (pivot point) in fr3_base
    anchor_m: np.ndarray  # the ball in the socket frame
    q0: np.ndarray
    start: tuple[int, int, int]  # the cell nearest the captured pose
    checker: PoseCheck
    solved: np.ndarray  # (H, P, R) bool: IK reached the cell
    q: np.ndarray  # (H, P, R, 7)
    transit: np.ndarray  # passes the checks for moving through
    dwell: np.ndarray  # passes the stricter ones for a dwell
    why: np.ndarray  # object: why a cell fails ("" if it passes)

    def R(self, idx) -> np.ndarray:
        i, j, k = idx
        return socket_rotation(*np.radians([self.heading_deg[i], self.pitch_deg[j], self.roll_deg[k]]))

    def angles_deg(self, idx) -> list[float]:
        i, j, k = idx
        return [float(self.heading_deg[i]), float(self.pitch_deg[j]), float(self.roll_deg[k])]

    def summary(self) -> dict[str, Any]:
        why: dict[str, int] = {}
        for w in self.why.reshape(-1):
            for part in str(w).split("; ") if w else []:
                key = part.split(":")[0]
                why[key] = why.get(key, 0) + 1
        span = {}
        for name, axis, vals in (
            ("heading", 0, self.heading_deg),
            ("pitch", 1, self.pitch_deg),
            ("roll", 2, self.roll_deg),
        ):
            hit = self.transit.any(axis=tuple(a for a in range(3) if a != axis))
            span[name] = [float(vals[hit].min()), float(vals[hit].max())] if hit.any() else None
        return {
            "cells": int(self.solved.size),
            "reached": int(self.solved.sum()),
            "transit": int(self.transit.sum()),
            "dwell": int(self.dwell.sum()),
            "why": dict(sorted(why.items(), key=lambda kv: -kv[1])),
            "span_deg": span,
            "start_cell_deg": self.angles_deg(self.start),
        }


def _keep_off_limits(q: np.ndarray) -> np.ndarray:
    lo, hi = IK_LOWER + math.radians(IK_KEEP_OFF_DEG), IK_UPPER - math.radians(IK_KEEP_OFF_DEG)
    return np.clip(q, lo, hi)


def _socket_target(R: np.ndarray, ball: np.ndarray, anchor: np.ndarray, offset=0.0) -> np.ndarray:
    """corenetic_socket_ee's pose that puts the ball (``anchor`` in it) on ``ball`` + ``offset``."""
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, ball - R @ anchor + offset
    return T


def solve_cell(kin: Kinematics, T: np.ndarray, seed) -> tuple[np.ndarray, bool]:
    """IK from ``seed``; the nullspace keeps joints off their limits."""
    q, ep, er = kin.ik(T, seed, _keep_off_limits(np.asarray(seed)), iters=80)
    return q, ep * 1e3 < IK_TOL_MM and math.degrees(er) < IK_TOL_DEG


def _neighbours(idx, shape, faces_only: bool = False):
    for d in np.ndindex(3, 3, 3):
        step = tuple(x - 1 for x in d)
        if step == (0, 0, 0) or (faces_only and sum(map(abs, step)) != 1):
            continue
        nb = tuple(a + b for a, b in zip(idx, step, strict=True))
        if all(0 <= x < n for x, n in zip(nb, shape, strict=True)):
            yield nb


def survey(
    centre: dict[str, Any],
    kin: Kinematics,
    scene: SceneCheck | None,
    stand: Stand | None,
    params: SurveyParams | None = None,
    anchor_mm=(0.0, 0.0, 0.0),
) -> Survey:
    """Every grid cell reachable from the captured pose by continuation, and what each cell runs into."""
    params = params or SurveyParams()
    q0 = np.asarray(centre["q"], dtype=float)
    T0 = kin.fk(q0)
    anchor = np.asarray(anchor_mm, dtype=float) / 1e3
    c = T0[:3, 3] + T0[:3, :3] @ anchor  # the ball
    h0, p0, r0 = np.degrees(socket_angles(T0[:3, :3])[0])
    H, P, Rr = params.axes(h0)
    shape = (len(H), len(P), len(Rr))
    S = Survey(
        heading_deg=H,
        pitch_deg=P,
        roll_deg=Rr,
        centre_m=c,
        anchor_m=anchor,
        q0=q0,
        start=(
            int(np.argmin(np.abs(H - h0))),
            int(np.argmin(np.abs(P - p0))),
            int(np.argmin(np.abs(Rr - r0))),
        ),
        checker=PoseCheck(q0, T0[:3, :3], c, scene, stand, params.table_z_m),
        solved=np.zeros(shape, dtype=bool),
        q=np.full(shape + (7,), np.nan),
        transit=np.zeros(shape, dtype=bool),
        dwell=np.zeros(shape, dtype=bool),
        why=np.full(shape, "", dtype=object),
    )

    def evaluate(idx, q) -> None:
        S.solved[idx], S.q[idx] = True, q
        why, out = S.checker.check(S.R(idx), q)
        S.transit[idx] = not why
        S.why[idx] = "; ".join(why)
        S.dwell[idx] = S.transit[idx] and S.checker.dwell_ok(out)

    # Breadth first from the captured pose: each cell seeded by the neighbour that reached it, so all
    # solutions sit on the posture branch the arm is in. Only cells that pass are expanded.
    q_start, ok = solve_cell(kin, _socket_target(S.R(S.start), c, anchor), q0)
    queue: deque[tuple[int, int, int]] = deque()
    if ok:
        evaluate(S.start, q_start)
        queue.append(S.start)
    else:
        S.why[S.start] = "unreachable: IK"
    while queue:
        idx = queue.popleft()
        if not S.transit[idx]:
            continue
        for nb in _neighbours(idx, shape, faces_only=True):
            if S.solved[nb] or S.why[nb]:
                continue
            T = _socket_target(S.R(nb), c, anchor)
            q, ok = solve_cell(kin, T, S.q[idx])
            if not ok:  # from the other solved cells around it, nearest posture first
                seeds = [S.q[x] for x in _neighbours(nb, shape) if S.transit[x] and x != idx]
                for seed in sorted(seeds, key=lambda x: float(np.abs(x - S.q[idx]).sum())):
                    q, ok = solve_cell(kin, T, seed)
                    if ok:
                        break
            if not ok:
                S.why[nb] = "unreachable: IK"
                continue
            evaluate(nb, q)
            queue.append(nb)
    return S


# --- dwells ----------------------------------------------------------------------------------------


def fit_sigma(Rs) -> np.ndarray:
    """1-sigma of (s, c) per unit per-dwell error, for dwells at socket rotations ``Rs``: the linear model
    ``R_i s - c = -t_i`` the fit solves (s in the socket frame, c in fr3_base), each dwell one independent
    3-vector of error. Infinite when the dwells do not turn about two axes."""
    F = np.zeros((6, 6))
    for R in Rs:
        B = np.hstack([np.asarray(R), -np.eye(3)])
        F += B.T @ B
    if np.linalg.cond(F) > 1e12:
        return np.full(6, np.inf)
    return np.sqrt(np.clip(np.diag(np.linalg.inv(F)), 0.0, None))


def choose_dwells(
    S: Survey, n: int = 14, *, j6_span_deg: float = J6_DWELL_SPAN_DEG, exclude=(), exchange_rounds: int = 4
) -> list[tuple[int, int, int]]:
    """The start cell and n-1 more dwell cells, minimising the worst 1-sigma of s (plus a tenth of c's),
    with j6 over all of them inside ``j6_span_deg`` (the moves between need some more)."""
    cand = [tuple(int(x) for x in i) for i in np.argwhere(S.dwell) if tuple(int(x) for x in i) not in exclude]
    info = {}
    for i in set(cand) | {S.start}:
        B = np.hstack([S.R(i), -np.eye(3)])
        info[i] = B.T @ B
    j6 = {i: math.degrees(S.q[i][5]) for i in info}

    def cost(cells, ridge=0.0):
        F = ridge * np.eye(6) + sum(info[i] for i in cells)
        if np.linalg.cond(F) > 1e12:
            return math.inf
        sig = np.sqrt(np.clip(np.diag(np.linalg.inv(F)), 0.0, None))
        return float(sig[:3].max()) + 0.1 * float(sig[3:].max())

    def fits(cells):
        v = [j6[i] for i in cells]
        return max(v) - min(v) <= j6_span_deg

    chosen = [S.start]
    while len(chosen) < n:
        options = [i for i in cand if i not in chosen and fits(chosen + [i])]
        if not options:
            break
        chosen.append(min(options, key=lambda i: cost(chosen + [i], ridge=1e-3)))
    for _ in range(exchange_rounds):
        improved = False
        for k in range(1, len(chosen)):
            current = cost(chosen)
            for i in cand:
                if i in chosen:
                    continue
                trial = chosen[:k] + [i] + chosen[k + 1 :]
                if fits(trial) and cost(trial) < current - 1e-6:
                    chosen, current, improved = trial, cost(trial), True
        if not improved:
            break
    return chosen


# --- route -----------------------------------------------------------------------------------------


def _q_cost(a: np.ndarray, b: np.ndarray) -> float:
    w = np.ones(7)
    w[5] = J6_WEIGHT
    return float(np.degrees(np.abs(a - b)) @ w)


class Router:
    """Shortest moves (in weighted joint travel) between cells, through cells that pass, over edges whose
    two solutions are on one posture branch and that no densified move has found bad."""

    def __init__(self, S: Survey, j6_window_deg: tuple[float, float]):
        self.S = S
        self.lo, self.hi = j6_window_deg
        self.forbidden: set[frozenset] = set()
        self._trees: dict = {}

    def _ok(self, v) -> bool:
        return bool(self.S.transit[v]) and self.lo <= math.degrees(self.S.q[v][5]) <= self.hi

    def edge_ok(self, u, v) -> bool:
        if frozenset((u, v)) in self.forbidden:
            return False
        return float(np.degrees(np.abs(self.S.q[u] - self.S.q[v])).max()) <= EDGE_MAX_DQ_DEG

    def forbid(self, u, v) -> None:
        self.forbidden.add(frozenset((u, v)))
        self._trees.clear()

    def tree(self, src) -> tuple[dict, dict]:
        if src in self._trees:
            return self._trees[src]
        S = self.S
        dist, prev = {src: 0.0}, {}
        heap = [(0.0, src)]
        while heap:
            d, u = heapq.heappop(heap)
            if d > dist.get(u, math.inf):
                continue
            for v in _neighbours(u, S.transit.shape):
                if not self._ok(v) or not self.edge_ok(u, v):
                    continue
                nd = d + _q_cost(S.q[u], S.q[v])
                if nd < dist.get(v, math.inf):
                    dist[v], prev[v] = nd, u
                    heapq.heappush(heap, (nd, v))
        self._trees[src] = (dist, prev)
        return dist, prev

    def cost(self, a, b) -> float:
        return self.tree(a)[0].get(b, math.inf)

    def path(self, a, b) -> list | None:
        dist, prev = self.tree(a)
        if b not in dist:
            return None
        out = [b]
        while out[-1] != a:
            out.append(prev[out[-1]])
        return out[::-1]


def order_tour(router: Router, dwells: list) -> list:
    """Start first and last; nearest neighbour then 2-opt on the move costs. Dwells not reachable are left out."""
    start = dwells[0]
    reach = [d for d in dwells[1:] if math.isfinite(router.cost(start, d))]
    tour = [start]
    rest = set(reach)
    while rest:
        nxt = min(rest, key=lambda b: router.cost(tour[-1], b))
        tour.append(nxt)
        rest.remove(nxt)
    tour.append(start)

    def length(t):
        return sum(router.cost(a, b) for a, b in zip(t[:-1], t[1:], strict=True))

    improved = True
    while improved:
        improved = False
        for i in range(1, len(tour) - 2):
            for j in range(i + 1, len(tour) - 1):
                new = tour[:i] + tour[i : j + 1][::-1] + tour[j + 1 :]
                if length(new) < length(tour) - 1e-9:
                    tour, improved = new, True
    return tour


# --- the timed path --------------------------------------------------------------------------------


@dataclass
class DwellParams:
    dwells: int = 14
    dwell_s: float = 1.5
    speed_deg_s: float = 5.0  # peak angular speed of the tool
    press_mm: float = 3.0
    bias_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    anchor_mm: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    ramp_s: float = 2.0
    rate_hz: float = 100.0
    translational_stiffness: float = 600.0
    rotational_stiffness: float = 40.0


def _slerp(Ra: np.ndarray, Rb: np.ndarray, u: float) -> np.ndarray:
    import pinocchio as pin

    return Ra @ pin.exp3(u * pin.log3(Ra.T @ Rb))


@dataclass
class Move:
    R: list  # dense socket rotations, about DENSE_STEP_DEG apart, the first where the arm is
    q: np.ndarray  # IK of each (pressed)
    bad_edge: tuple | None = None  # the cell edge a failed check lies on
    why: str = ""


def densify(
    S: Survey,
    kin: Kinematics,
    cells: list,
    R_from: np.ndarray,
    q_from: np.ndarray,
    R_to=None,
    q_to=None,
    offset=0.0,
) -> Move:
    """The move along ``cells`` from the arm's pose, IK by continuation at about DENSE_STEP_DEG of tool
    turn, each sample checked; stops at the first bad one and names the cell edge it lies on."""
    way_R = [R_from] + [S.R(i) for i in cells[1:]]
    way_q = [q_from] + [S.q[i] for i in cells[1:]]
    if R_to is not None:  # home to the captured pose itself, not its cell
        way_R[-1], way_q[-1] = R_to, q_to
    seg = [rot_angle(a.T @ b) for a, b in zip(way_R[:-1], way_R[1:], strict=True)]
    arc = np.concatenate([[0.0], np.cumsum(seg)])
    Rs, qs = [R_from], [q_from]
    if arc[-1] < 1e-9:
        return Move(Rs, np.asarray(qs))
    n = max(2, int(math.ceil(math.degrees(arc[-1]) / DENSE_STEP_DEG)))
    q = q_from
    for k, s_arc in enumerate(np.linspace(0.0, arc[-1], n + 1)[1:], start=1):
        j = min(int(np.searchsorted(arc, s_arc, side="right")) - 1, len(seg) - 1)
        u = 0.0 if seg[j] < 1e-12 else (s_arc - arc[j]) / seg[j]
        R = _slerp(way_R[j], way_R[j + 1], u)
        q_new, ep, er = kin.ik(
            _socket_target(R, S.centre_m, S.anchor_m, offset),
            q,
            (1 - u) * way_q[j] + u * way_q[j + 1],
            iters=40,
        )
        edge = (cells[j], cells[j + 1]) if j + 1 < len(cells) else None
        jump = float(np.degrees(np.abs(q_new - q)).max())
        why = []
        if ep * 1e3 > IK_TOL_MM or math.degrees(er) > IK_TOL_DEG:
            why.append("IK")
        if jump > DENSE_MAX_DQ_DEG:
            why.append(f"joint jump {jump:.0f}°")
        if not why:
            why, _ = S.checker.check(R, q_new, full=k % DENSE_FULL_EVERY == 0 or k == n)
        if why:
            return Move(Rs, np.asarray(qs), edge, "; ".join(why))
        q = q_new
        Rs.append(R)
        qs.append(q)
    return Move(Rs, np.asarray(qs))


def route_moves(
    S: Survey, kin: Kinematics, router: Router, tour: list, offset, max_repairs: int = 40
) -> tuple[list, list[Move], list[str]]:
    """Each move of the tour densified and checked; a bad edge is forbidden and the move routed again.
    A dwell no move can reach is dropped. Returns the tour actually kept, its moves, and notes."""
    T0 = kin.fk(S.q0)
    R0 = T0[:3, :3]
    kept, moves, notes = [tour[0]], [], []
    R_now = R0
    q_now, _, _ = kin.ik(_socket_target(R0, S.centre_m, S.anchor_m, offset), S.q0, S.q0)  # pressed in
    targets = tour[1:]
    i = 0
    while i < len(targets):
        dst = targets[i]
        home = i == len(targets) - 1
        move = None
        for _ in range(max_repairs):
            cells = router.path(kept[-1], dst)
            if cells is None:
                break
            move = densify(S, kin, cells, R_now, q_now, R0 if home else None, S.q0 if home else None, offset)
            if not move.why:
                break
            if move.bad_edge is None:  # the last stretch home: nothing to route around
                break
            router.forbid(*move.bad_edge)
        if move is None or move.why:
            reason = "no route" if move is None else move.why
            if home:
                notes.append(f"回不到起点：{reason}")
                moves.append(move or Move([R_now], np.asarray([q_now])))
                break
            notes.append(f"驻点 {S.angles_deg(dst)} 去不了（{reason}），跳过")
            i += 1
            continue
        moves.append(move)
        kept.append(dst)
        R_now, q_now = move.R[-1], move.q[-1]
        i += 1
    return kept, moves, notes


def timed_path(
    S: Survey, kin: Kinematics, moves: list[Move], p: DwellParams
) -> tuple[dict, np.ndarray, list[str]]:
    """Press in, then each move timed by smoothstep (the tool peaks at ``speed_deg_s``, no joint past its
    limit) followed by a dwell, the last move home, release."""
    dt_s = 1.0 / p.rate_hz
    offset = np.asarray(p.bias_mm, dtype=float) / 1e3 + p.press_mm / 1e3 * PRESS_DIR
    R0 = kin.fk(S.q0)[:3, :3]
    limits = np.full(7, MAX_JOINT_SPEED_DEG_S)
    limits[5] = MAX_J6_SPEED_DEG_S
    names: list[str] = ["press"]
    ts, Rs, ps, qs, phase, factor = [], [], [], [], [], []

    def emit(R, q, f):
        ts.append(len(ts) * dt_s)
        Rs.append(R)
        ps.append(S.centre_m - R @ S.anchor_m + f * offset)
        qs.append(q)
        phase.append(len(names) - 1)
        factor.append(f)

    n_ramp = max(2, int(round(p.ramp_s / dt_s)))
    for i in range(n_ramp):
        f = float(smoothstep(i / n_ramp))
        q, _, _ = kin.ik(
            _socket_target(R0, S.centre_m, S.anchor_m, f * offset), qs[-1] if qs else S.q0, S.q0, iters=20
        )
        emit(R0, q, f)
    for k, mv in enumerate(moves, start=1):
        last = k == len(moves)
        if len(mv.R) > 1:
            # Progress in seconds at full speed: each step costs what its slowest part needs (the tool
            # at speed_deg_s, each joint at its limit), so a joint-heavy stretch near the wrist
            # singularity is crossed slowly. Smoothstep over it peaks at 1.5x, hence the 1.5.
            tool = np.degrees([rot_angle(a.T @ b) for a, b in zip(mv.R[:-1], mv.R[1:], strict=True)])
            joints = np.degrees(np.abs(np.diff(mv.q, axis=0)))
            step_s = np.maximum(tool / p.speed_deg_s, (joints / limits).max(axis=1))
            progress = np.concatenate([[0.0], np.cumsum(step_s)])
            n = max(2, int(round(1.5 * progress[-1] / dt_s)))
            names.append("return" if last else f"move{k}")
            for i in range(1, n + 1):
                s = float(smoothstep(i / n)) * progress[-1]
                j = min(int(np.searchsorted(progress, s, side="right")) - 1, len(mv.R) - 2)
                span = progress[j + 1] - progress[j]
                u = 0.0 if span < 1e-12 else (s - progress[j]) / span
                emit(_slerp(mv.R[j], mv.R[j + 1], u), (1 - u) * mv.q[j] + u * mv.q[j + 1], 1.0)
        if not last:
            names.append(f"dwell{k}")
            for _ in range(max(1, int(round(p.dwell_s / dt_s)))):
                emit(mv.R[-1], mv.q[-1], 1.0)
    names.append("release")
    for i in range(1, n_ramp + 1):
        f = 1.0 - float(smoothstep(i / n_ramp))
        q, _, _ = kin.ik(_socket_target(R0, S.centre_m, S.anchor_m, f * offset), qs[-1], S.q0, iters=20)
        emit(R0, q, f)
    path = {
        "t": np.asarray(ts),
        "R": np.asarray(Rs),
        "p": np.asarray(ps),
        "phase": np.asarray(phase),
        "factor": np.asarray(factor),
    }
    return path, np.asarray(qs), names


def plan_dwells(
    centre: dict[str, Any],
    kin: Kinematics,
    scene: SceneCheck | None,
    stand: Stand | None,
    params: DwellParams | None = None,
    survey_params: SurveyParams | None = None,
) -> tuple[dict, np.ndarray, list[str], dict[str, Any], Survey]:
    """survey -> dwells -> tour -> checked moves -> timed path. The returned info has the survey, the
    dwells and their predicted 1-sigma; ``problems`` there are the planner's own (the timed path still
    goes through ``fr3_pivot_auto.check_path`` and ``stand_problems``)."""
    params = params or DwellParams()
    S = survey(centre, kin, scene, stand, survey_params, params.anchor_mm)
    info: dict[str, Any] = {"survey": S.summary(), "problems": [], "notes": []}
    if not S.dwell[S.start]:
        why = "; ".join(S.checker.start_why) or S.why[S.start] or "余量不够做驻点"
        info["problems"].append(f"起始姿态本身不过检查：{why}（换个姿态重新取球心）")
        return {}, np.zeros((0, 7)), [], info, S
    offset = np.asarray(params.bias_mm, dtype=float) / 1e3 + params.press_mm / 1e3 * PRESS_DIR
    dropped: set = set()
    for _ in range(3):  # dwells no move reaches are replaced by others
        dwells = choose_dwells(S, params.dwells, exclude=dropped)
        j6 = [math.degrees(S.q[d][5]) for d in dwells]
        router = Router(S, (min(j6) - J6_ROUTE_SLACK_DEG, max(j6) + J6_ROUTE_SLACK_DEG))
        tour = order_tour(router, dwells)
        kept, moves, notes = route_moves(S, kin, router, tour, offset)
        lost = set(dwells) - set(kept)
        if not lost or len(kept) - 1 >= params.dwells:
            break
        dropped |= lost
    info["notes"] = notes
    if notes and notes[-1].startswith("回不到起点"):
        info["problems"].append(notes[-1])
    path, qs, names = timed_path(S, kin, moves, params)
    sig = fit_sigma([S.R(i) for i in kept[:-1]])
    info.update(
        {
            "dwells_deg": [S.angles_deg(i) for i in kept[:-1]],
            "dwell_q": [np.round(S.q[i], 5).tolist() for i in kept[:-1]],
            "sigma_per_unit": {"s": np.round(sig[:3], 3).tolist(), "c": np.round(sig[3:], 3).tolist()},
            "stand_margins_mm": {
                "socket": SOCKET_CLEARANCE_MM,
                "socket_dwell": SOCKET_DWELL_CLEARANCE_MM,
                "gripper": GRIPPER_CLEARANCE_MM,
                "gripper_dwell": GRIPPER_DWELL_CLEARANCE_MM,
            },
        }
    )
    if len(kept) - 1 < MIN_DWELLS:
        info["problems"].append(
            f"只排进 {len(kept) - 1} 个驻点（拟合至少要 {MIN_DWELLS} 个）：这个球位附近转不开"
        )
    return path, qs, names, info, S


def stand_problems(
    S: Survey, path: dict, qs: np.ndarray, every: int = 10
) -> tuple[list[str], dict[str, Any]]:
    """The stand along the whole timed path, as the last word on it (the moves were checked as they were built)."""
    chk = S.checker
    if chk.stand is None:
        return [], {"socket_stand_mm": None, "gripper_stand_mm": None, "gripper_stand_body": ""}
    idx = list(range(0, len(qs), every)) + [len(qs) - 1]
    sock = min(chk.stand.socket_clearance(path["R"][i]) for i in idx)
    grip, who = math.inf, ""
    if chk.stand_check is not None:
        for i in idx:
            g, w = chk.stand_check.clearance(qs[i], S.centre_m, chk.floor_mm)
            if g < grip:
                grip, who = g, w
    out = []
    need = chk.min_mm
    if sock < need["socket"]:
        out.append(f"球窝离支架最近 {sock:.1f} mm（要 ≥ {need['socket']:.1f} mm）")
    if grip < need["gripper"]:
        out.append(f"{who} 离支架最近 {grip:.1f} mm（要 ≥ {need['gripper']:.1f} mm）")
    return out, {
        "socket_stand_mm": round(sock, 2),
        "gripper_stand_mm": None if chk.stand_check is None else round(grip, 2),
        "gripper_stand_body": who,
    }
