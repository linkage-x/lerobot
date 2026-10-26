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
How well two FR3 bases can be registered to each other, before there is a second arm: a synthetic study.

Roadmap docs/dual_fr3_bimanual_replay_roadmap.html, G3. Two arms side by side (left = +y of right, both
facing +x), each with its own kinematic error, and three ways to get ``T_BL_BR`` (right base -> left base):

  arm_arm_ball   both flanges carry the same cup and pivot about one set of balls on the table
                 (fr3_pivot_auto's dwells); each pivot puts the ball centre in that arm's base, and a rigid
                 Kabsch of the pairs is the edge. No camera, no laser: FK + pivot seating only.
  laser_station  an SMR on each flange, the tracker left where it stands; each arm visits poses in the shared
                 workspace and fits "station x FK" (station pose and the SMR offset), and the two fits compose.
  via_world      each arm registered to the camera world on its own with the ball check (FR3 pivot vs the
                 handheld UMI pivot at the same balls), then composed through the world. What two independent
                 world edges do to the distance between the hands; also the closure the B4b check will see.

What is scored is what a bimanual replay sees, not how close ``T`` is to a truth nobody can command: at poses
in the shared workspace, the right arm's reading of a physical point, mapped by the estimated edge, against
the left arm's reading of the same point (``tlr``). Part of that no rigid edge can remove -- the two arms'
kinematic errors differ from pose to pose -- and is reported as ``floor`` (the best rigid fit, from the
truth). ``reg`` is the edge alone, over the workspace positions.

The error models are the measured ones, scaled, not invented:

  kinematics  each arm's DH parameters perturbed (lengths 0.1 mm, angles 0.02 deg, then scaled together) so
              that FK vs truth after a rigid fit over the workspace has the rms the laser saw (0.47 mm,
              10-12/13; 0.27 mm on 10-10). Independent per arm.
  seating     the cup's centre is off the ball's by an isotropic per-dwell error, 0.55 mm/axis, which with the
              kinematics reproduces the 10-20 pivot's 0.95 mm rms (check ``pivot_rms_mm`` in the report).
  world       the handheld pivot's ball centre in the world, 1.2 mm/axis, which reproduces the 10-15 edge's
              leave-one-out 2.63 mm.
  laser       0.02 mm/axis.

What this can and cannot say: it ranks the methods and layouts and says how sensitive each is (ball count,
height step, spread, dwells, arm spacing), and it fixes the B3 protocol and the closure threshold for G2. It
cannot certify an accuracy: data generated from these models cannot reveal what the models leave out (base
compliance, the ball moving under the press, gravity sag that differs with load) -- the FR3 frame-selection
change of 2026-09 looked right in reconstruction and was overturned on the robot.

Pure numpy; the pivot fit is ``lerobot.robots.franka_research3.tcp_pivot``, loaded by file so the lerobot
package (torch, ...) is not imported.
"""

# ruff: noqa: N802, N803, N806, N815  (matrix names: R, T, J follow the maths)
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3.fr3_kinematics import (  # noqa: E402
    FLANGE_D,
    NOMINAL,
    NOMINAL_DH,
    Q_READY,
    Arm,
    _rotvec,
    ik_retry,
)
from tools.thor.gmsl2 import world_provenance as wp  # noqa: E402

SCHEMA = "dual_fr3_registration_sim/v1"


def _load_tcp_pivot():
    path = REPO / "src" / "lerobot" / "robots" / "franka_research3" / "tcp_pivot.py"
    spec = importlib.util.spec_from_file_location("_fr3_tcp_pivot", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses resolve their module by name
    spec.loader.exec_module(module)
    return module


fit_pivot_joint = _load_tcp_pivot().fit_pivot_joint

MIN_DWELLS = 6  # tcp_pivot keeps a run only with 6+ orientations
#: A ball counts only with this many reachable dwells: at 6 the fit's outlier rejection can drop below its own floor.
MIN_USABLE_DWELLS = 8
#: Errors put into the world -> left base edge to see whether the loop check (world_provenance) catches them.
INJECTED_EDGE_ERRORS_MM = (5.0, 10.0)

# The shapes of the kinematic error before scaling: what a joint/link calibration typically leaves.
UNIT_LENGTH_M = 1e-4
UNIT_ANGLE_RAD = math.radians(0.02)


# ------------------------------------------------------------------ kinematics ---


def perturbed(rng: np.random.Generator, scale: float) -> Arm:
    """An arm whose DH lengths and angles are off by ``scale`` x the unit shape."""
    dh = NOMINAL_DH.copy()
    dh[:, 0:2] += rng.normal(0.0, UNIT_LENGTH_M * scale, (7, 2))
    dh[:, 2:4] += rng.normal(0.0, UNIT_ANGLE_RAD * scale, (7, 2))
    return Arm(dh, FLANGE_D + float(rng.normal(0.0, UNIT_LENGTH_M * scale)))


def seat(arm: Arm, q: np.ndarray, point: np.ndarray, target: np.ndarray, *, iters: int = 3) -> np.ndarray:
    """The joints near ``q`` that put ``arm``'s tool ``point`` on ``target``: the least joint motion, as when
    the press loop slides the cup onto the ball, or the real arm sits where the nominal model said it would."""
    q = np.array(q, dtype=np.float64)
    for _ in range(iters):
        T, J = arm.kinematics(q, point)
        r = target - (T[:, :3, 3] + T[:, :3, :3] @ point)
        Jp = J[:, :3]
        q += (Jp.transpose(0, 2, 1) @ np.linalg.solve(Jp @ Jp.transpose(0, 2, 1), r[:, :, None]))[:, :, 0]
    return q


def kabsch(P: np.ndarray, Q: np.ndarray) -> np.ndarray:
    """Rigid ``T`` (4x4) minimising sum |T p - q|^2."""
    mp, mq = P.mean(axis=0), Q.mean(axis=0)
    U, _, Vt = np.linalg.svd((P - mp).T @ (Q - mq))
    D = np.diag([1.0, 1.0, np.sign(np.linalg.det(Vt.T @ U.T))])
    R = Vt.T @ D @ U.T
    T = np.eye(4)
    T[:3, :3], T[:3, 3] = R, mq - R @ mp
    return T


def apply(T: np.ndarray, P: np.ndarray) -> np.ndarray:
    return P @ T[:3, :3].T + T[:3, 3]


def rigid_inverse(T: np.ndarray) -> np.ndarray:
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def _rot(axis: str, angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return {
        "x": np.array([[1, 0, 0], [0, c, -s], [0, s, c]]),
        "y": np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]]),
        "z": np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]]),
    }[axis]


def rigid(yaw: float = 0.0, t=(0.0, 0.0, 0.0), roll: float = 0.0, pitch: float = 0.0) -> np.ndarray:
    T = np.eye(4)
    T[:3, :3] = _rot("z", yaw) @ _rot("y", pitch) @ _rot("x", roll)
    T[:3, 3] = t
    return T


def kinematic_rms(nominal: Arm, true: Arm, q: np.ndarray) -> float:
    """FK vs truth over ``q``, after the best rigid fit: what the laser reads as the arm's FK error."""
    p_nom, p_true = nominal.fk(q)[:, :3, 3], true.fk(q)[:, :3, 3]
    return float(np.sqrt(np.mean(np.sum((apply(kabsch(p_true, p_nom), p_true) - p_nom) ** 2, axis=1))))


# -------------------------------------------------------------------- the setup ---


@dataclass(frozen=True)
class Config:
    name: str = "baseline"
    spacing_m: float = 0.8  # left base at +y of the right one
    left_yaw_deg: float = 0.0
    layout: str = "height5"
    centre_x_m: float = 0.40  # shared workspace centre in the right base, y = spacing / 2
    ball_z_m: float = 0.20  # ball centres of the lower layer
    span_m: float = 0.30
    height_m: float = 0.15
    dwells: int = 14
    tilt_deg: float = 30.0
    cup_mm: float = 100.0  # cup centre on the flange axis
    seat_sigma_mm: float = 0.55
    fk_rms_mm: float = 0.47
    world_sigma_mm: float = 1.2
    laser_sigma_mm: float = 0.02
    laser_poses: int = 16


def ball_layout(config: Config) -> np.ndarray:
    """Ball centres (k,3) in the right base."""
    cx, cy, z0 = config.centre_x_m, config.spacing_m / 2.0, config.ball_z_m
    h, half = config.height_m, config.span_m / 2.0
    corners = np.array([[cx - half, cy - half], [cx + half, cy - half], [cx + half, cy + half], [cx - half, cy + half]])
    layouts = {
        "planar4": [(*xy, z0) for xy in corners],
        "planar6": [(cx + dx, cy + dy, z0) for dx in (-half, 0.0, half) for dy in (-half, half)],
        "height4": [(*xy, z0) for xy in corners[:3]] + [(cx, cy, z0 + h)],
        "height5": [(*xy, z0) for xy in corners] + [(cx, cy, z0 + h)],
        "height6": [(*xy, z0) for xy in corners] + [(cx - half / 2, cy, z0 + h), (cx + half / 2, cy, z0 + h)],
        "height8": [(*xy, z0) for xy in corners] + [(*xy, z0 + h) for xy in corners],
    }
    if config.layout not in layouts:
        raise ValueError(f"unknown layout {config.layout!r}; one of {', '.join(layouts)}")
    return np.asarray(layouts[config.layout], dtype=np.float64)


def _down(arm: Arm, toward: np.ndarray) -> np.ndarray:
    """The flange orientation of the ready posture turned to face ``toward``: what the dwells tilt about."""
    seed = Q_READY.copy()
    seed[0] = math.atan2(toward[1], toward[0])
    return arm.fk(seed)[0, :3, :3], seed


def dwell_rotations(R_down: np.ndarray, n: int, tilt_deg: float) -> np.ndarray:
    """``n`` cup orientations about the ball: two spins at the vertical, then tilts at headings spread round.

    Tilting is what pins ``s`` across the cup axis and spinning what pins it along, so both are always in.
    """
    tilt = math.radians(tilt_deg)
    out = [R_down @ _rot("z", math.radians(a)) for a in (-60.0, 60.0)]
    rest = n - len(out)
    for i in range(rest):
        heading = 2 * math.pi * i * 0.618034
        t = tilt if i % 2 == 0 else tilt / 2
        axis = np.array([-math.sin(heading), math.cos(heading), 0.0])
        K = np.cross(np.eye(3), axis)
        R_tilt = np.eye(3) + math.sin(t) * K + (1 - math.cos(t)) * K @ K
        roll = math.radians(30.0 if i % 3 == 0 else -30.0 if i % 3 == 1 else 0.0)
        out.append(R_tilt @ R_down @ _rot("z", roll))
    return np.asarray(out[:n])


def _poses_at(R: np.ndarray, point: np.ndarray, tool: np.ndarray) -> np.ndarray:
    """Flange poses whose tool point is at ``point`` with flange rotations ``R``."""
    T = np.broadcast_to(np.eye(4), (len(R), 4, 4)).copy()
    T[:, :3, :3] = R
    T[:, :3, 3] = point - R @ tool
    return T


@dataclass
class Setup:
    """Everything that depends on the layout but not on the trial: the commanded joint angles."""

    config: Config
    T_BL_BR: np.ndarray
    balls_R: np.ndarray  # (k,3) right base
    cup: np.ndarray
    smr: np.ndarray
    pivot_q: dict[str, list[np.ndarray]]  # arm -> per ball, (m,7) nominal joints of the reachable dwells
    usable_balls: list[int]
    laser_q: dict[str, np.ndarray]
    grid_R: np.ndarray  # (g,3) tool points, right base
    grid_q: dict[str, np.ndarray]
    unit_rms_m: float
    notes: list[str]


def _grid(config: Config) -> np.ndarray:
    cx, cy = config.centre_x_m, config.spacing_m / 2.0
    xs = cx + np.array([-0.12, 0.0, 0.12])
    ys = cy + np.array([-0.15, 0.0, 0.15])
    zs = config.ball_z_m + np.array([0.0, 0.10, 0.20])
    return np.array([(x, y, z) for x in xs for y in ys for z in zs])


def build_setup(config: Config, *, seed: int = 0) -> Setup:
    rng = np.random.default_rng(seed)
    T_BL_BR = rigid(math.radians(config.left_yaw_deg), (0.0, config.spacing_m, 0.0))
    T_BL_BR = rigid_inverse(T_BL_BR)  # left base at +y: a right-base point is at -spacing in the left base
    cup = np.array([0.0, 0.0, config.cup_mm * 1e-3])
    smr = np.array([0.03, 0.0, 0.04])
    balls_R = ball_layout(config)
    notes: list[str] = []
    frames = {"right": np.eye(4), "left": T_BL_BR}

    pivot_q: dict[str, list[np.ndarray]] = {}
    for name, T_arm_R in frames.items():
        per_ball = []
        for k, ball in enumerate(apply(T_arm_R, balls_R)):
            R_down, seed_q = _down(NOMINAL, ball)
            R = dwell_rotations(R_down, config.dwells, config.tilt_deg)
            q, ok = ik_retry(NOMINAL, _poses_at(R, ball, cup), seed_q)
            if not ok.all():
                notes.append(f"{name} arm: ball {k} reaches {int(ok.sum())}/{len(ok)} dwells")
            per_ball.append(q[ok])
        pivot_q[name] = per_ball
    usable = [k for k in range(len(balls_R)) if all(len(pivot_q[a][k]) >= MIN_USABLE_DWELLS for a in frames)]
    if len(usable) < len(balls_R):
        notes.append(f"balls usable by both arms: {usable} of {len(balls_R)}")

    # the laser poses: tool points spread over the shared workspace, orientations varied
    grid_R = _grid(config)
    laser_q: dict[str, np.ndarray] = {}
    pick = rng.choice(len(grid_R), size=min(config.laser_poses, len(grid_R)), replace=False)
    for name, T_arm_R in frames.items():
        points = apply(T_arm_R, grid_R[pick])
        Ts = []
        for i, p in enumerate(points):
            R_down, _ = _down(NOMINAL, p)
            Ts.append(_poses_at(dwell_rotations(R_down, 8, config.tilt_deg)[i % 8][None], p, smr)[0])
        seeds = np.array([_down(NOMINAL, p)[1] for p in points])
        q, ok = ik_retry(NOMINAL, np.asarray(Ts), seeds)
        if not ok.all():
            notes.append(f"{name} arm: laser poses reached {int(ok.sum())}/{len(ok)}")
        laser_q[name] = q[ok]

    # the evaluation grid: each tool point at three orientations, kept where both arms reach it
    grid_q: dict[str, list[np.ndarray]] = {"right": [], "left": []}
    keep_points = []
    for p in grid_R:
        qs, ok_all = {}, True
        for name, T_arm_R in frames.items():
            pa = apply(T_arm_R, p[None])[0]
            R_down, seed_q = _down(NOMINAL, pa)
            R = dwell_rotations(R_down, 5, 20.0)[[0, 2, 3]]
            q, ok = ik_retry(NOMINAL, _poses_at(R, pa, cup), seed_q)
            ok_all &= bool(ok.all())
            qs[name] = q
        if ok_all:
            keep_points.extend([p] * 3)
            for name in frames:
                grid_q[name].append(qs[name])
    if not keep_points:
        raise RuntimeError(f"{config.name}: no workspace point is reachable by both arms")
    if len(keep_points) < 3 * len(grid_R):
        notes.append(f"workspace points reachable by both arms: {len(keep_points) // 3} of {len(grid_R)}")
    grid = {name: np.concatenate(v) for name, v in grid_q.items()}

    unit = [kinematic_rms(NOMINAL, perturbed(rng, 1.0), grid["right"]) for _ in range(64)]
    return Setup(
        config=config,
        T_BL_BR=T_BL_BR,
        balls_R=balls_R,
        cup=cup,
        smr=smr,
        pivot_q=pivot_q,
        usable_balls=usable,
        laser_q=laser_q,
        grid_R=np.asarray(keep_points),
        grid_q=grid,
        unit_rms_m=float(np.mean(unit)),
        notes=notes,
    )


# ------------------------------------------------------------------ the methods ---


def pivot_centres(
    setup: Setup, arm: Arm, name: str, balls: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, float]:
    """Each usable ball's centre as this arm's pivot fit puts it (its own base, nominal FK); residual rms."""
    sigma = setup.config.seat_sigma_mm * 1e-3
    Rs, ts = [], []
    for k in setup.usable_balls:
        q = setup.pivot_q[name][k]
        target = balls[k] + rng.normal(0.0, sigma, (len(q), 3))
        T = NOMINAL.fk(seat(arm, q, setup.cup, target))
        Rs.append(T[:, :3, :3])
        ts.append(T[:, :3, 3])
    fit = fit_pivot_joint(Rs, ts, shared_centre=False)
    return fit.c_base_m, fit.residual_rms_m


def laser_station_fit(T_m: np.ndarray, p_S: np.ndarray, *, iters: int = 30) -> tuple[np.ndarray, np.ndarray]:
    """``(T_S_B, o)`` with ``T_S_B (R_i o + t_i) = p_i``: Gauss-Newton on (rotation, translation, o).

    Started from o = 0 Kabsch; alternating the two halves instead converges slowly enough to stop early and
    misreport (10-10: 2.9 mm that was 0.27).
    """
    R_m, t_m = T_m[:, :3, :3], T_m[:, :3, 3]
    T = kabsch(t_m, p_S)
    o = np.zeros(3)
    for _ in range(iters):
        x = t_m + R_m @ o
        pred = apply(T, x)
        r = (p_S - pred).reshape(-1)
        R = T[:3, :3]
        n = len(x)
        J = np.zeros((3 * n, 9))
        Rx = x @ R.T
        for i in range(n):
            J[3 * i : 3 * i + 3, 0:3] = -np.cross(np.eye(3), Rx[i])  # d(pred)/d(rotvec), left perturbation
            J[3 * i : 3 * i + 3, 3:6] = np.eye(3)
            J[3 * i : 3 * i + 3, 6:9] = R @ R_m[i]
        dx, *_ = np.linalg.lstsq(J, r, rcond=None)
        w = dx[:3]
        theta = np.linalg.norm(w)
        K = np.cross(np.eye(3), w / theta) if theta > 0 else np.zeros((3, 3))
        dR = np.eye(3) + math.sin(theta) * K + (1 - math.cos(theta)) * K @ K
        T[:3, :3] = dR @ R
        T[:3, 3] = dR @ T[:3, 3] + dx[3:6]
        o = o + dx[6:9]
        if np.linalg.norm(dx) < 1e-12:
            break
    return T, o


def loo_rms_mm(P: np.ndarray, Q: np.ndarray) -> float | None:
    """Leave-one-out rms of a rigid fit P -> Q, mm: each pair predicted from the others. None below 4 pairs."""
    if len(P) < 4:
        return None
    errors = []
    for j in range(len(P)):
        others = [i for i in range(len(P)) if i != j]
        errors.append(np.linalg.norm(apply(kabsch(P[others], Q[others]), P[j : j + 1])[0] - Q[j]))
    return float(np.sqrt(np.mean(np.square(errors))) * 1e3)


def _graph_edge(src: str, dst: str, T_dst_src: np.ndarray, sigma_mm: float | None) -> dict[str, Any]:
    return {"from_world_frame_id": src, "to_world_frame_id": dst, "T_to_from": T_dst_src.tolist(), "sigma_mm": sigma_mm}


def closure_rows(
    E: dict[str, np.ndarray], T_ball: np.ndarray, sigmas: dict[str, float | None], rng: np.random.Generator
) -> dict[str, Any]:
    """The loop world -> right -> left against world -> left, through world_provenance's own check.

    ``E`` are the world edges ``T_W_B``; sigmas as each edge would carry them (leave-one-out rms). Then the
    same loop with the left world edge moved by INJECTED_EDGE_ERRORS_MM in a random direction.
    """

    def check(T_L_W: np.ndarray) -> dict[str, Any]:
        graph = {
            "edges": [
                _graph_edge("world", "right", rigid_inverse(E["right"]), sigmas["world_right"]),
                _graph_edge("right", "left", T_ball, sigmas["arm_arm"]),
                _graph_edge("world", "left", T_L_W, sigmas["world_left"]),
            ]
        }
        (row,) = wp.world_closures(graph, "world", "left")
        return row

    clean = check(rigid_inverse(E["left"]))
    out = {
        "disagreement_mm": clean["disagreement_mm"],
        "sigma_mm": clean["sigma_mm"],
        "threshold_mm": clean["threshold_mm"],
        "false_alarm": not clean["ok"],
    }
    for error_mm in INJECTED_EDGE_ERRORS_MM:
        direction = rng.normal(size=3)
        moved = rigid_inverse(E["left"])
        moved[:3, 3] += direction / np.linalg.norm(direction) * error_mm * 1e-3
        out[f"caught_{error_mm:g}mm"] = not check(moved)["ok"]
    return out


def _readings(setup: Setup, arms: dict[str, Arm]) -> dict[str, np.ndarray]:
    """Each arm's nominal-FK reading of the grid's physical tool points (its own base)."""
    out = {}
    for name, T_arm_R in (("right", np.eye(4)), ("left", setup.T_BL_BR)):
        target = apply(T_arm_R, setup.grid_R)
        q = seat(arms[name], setup.grid_q[name], setup.cup, target)
        out[name] = NOMINAL.point(q, setup.cup)
    return out


def _scores(setup: Setup, T: np.ndarray, readings: dict[str, np.ndarray], best: np.ndarray) -> dict[str, float]:
    """``reg`` is against ``best``, the rigid edge that fits the two arms' readings best, not the true
    base-to-base transform: a joint-1 offset turns an arm's whole FK frame, and an edge that follows it is
    right, not wrong."""
    tlr = np.linalg.norm(apply(T, readings["right"]) - readings["left"], axis=1)
    reg = np.linalg.norm(apply(T, setup.grid_R) - apply(best, setup.grid_R), axis=1)
    dR = T[:3, :3] @ best[:3, :3].T
    return {
        "tlr_p95_mm": float(np.percentile(tlr, 95) * 1e3),
        "tlr_max_mm": float(tlr.max() * 1e3),
        "reg_max_mm": float(reg.max() * 1e3),
        "rot_deg": float(np.degrees(np.linalg.norm(_rotvec(dR[None])[0]))),
    }


def run_trial(setup: Setup, rng: np.random.Generator) -> dict[str, Any]:
    c = setup.config
    scale = c.fk_rms_mm * 1e-3 / setup.unit_rms_m
    arms = {"right": perturbed(rng, scale), "left": perturbed(rng, scale)}
    readings = _readings(setup, arms)
    floor_T = kabsch(readings["right"], readings["left"])
    floor = np.linalg.norm(apply(floor_T, readings["right"]) - readings["left"], axis=1)
    out: dict[str, Any] = {"floor_p95_mm": float(np.percentile(floor, 95) * 1e3)}

    balls = {"right": setup.balls_R, "left": apply(setup.T_BL_BR, setup.balls_R)}
    centres, rms = {}, []
    for name in ("right", "left"):
        centres[name], r = pivot_centres(setup, arms[name], name, balls[name], rng)
        rms.append(r * 1e3)
    out["pivot_rms_mm"] = float(np.mean(rms))
    T_ball = kabsch(centres["right"], centres["left"])
    out["arm_arm_ball"] = _scores(setup, T_ball, readings, floor_T)
    loo_ball = loo_rms_mm(centres["right"], centres["left"])
    if loo_ball is not None:
        out["arm_arm_ball"]["loo_rms_mm"] = loo_ball

    # one station for both arms, ~2 m in front
    T_S_R = rigid(math.radians(170.0), (2.0, -0.3, -0.2))
    T_S = {"right": T_S_R, "left": T_S_R @ rigid_inverse(setup.T_BL_BR)}
    fits = {}
    for name in ("right", "left"):
        q = setup.laser_q[name]
        smr_true = arms[name].point(q, setup.smr)
        p_S = apply(T_S[name], smr_true) + rng.normal(0.0, c.laser_sigma_mm * 1e-3, smr_true.shape)
        fits[name], _ = laser_station_fit(NOMINAL.fk(q), p_S)
    out["laser_station"] = _scores(setup, rigid_inverse(fits["left"]) @ fits["right"], readings, floor_T)

    # world edges, one per arm, at the same balls but from independent handheld pivots
    T_W_R = rigid(math.radians(90.0), (1.2, 0.0, 0.7), roll=math.radians(-90.0))
    edges, sigmas = {}, {"arm_arm": loo_ball}
    for name in ("right", "left"):
        b_world = apply(T_W_R, setup.balls_R[setup.usable_balls])
        b_world = b_world + rng.normal(0.0, c.world_sigma_mm * 1e-3, b_world.shape)
        edges[name] = kabsch(centres[name], b_world)  # T_W_B
        sigmas[f"world_{name}"] = loo_rms_mm(centres[name], b_world)
    T_world = rigid_inverse(edges["left"]) @ edges["right"]
    out["via_world"] = _scores(setup, T_world, readings, floor_T)
    closure = np.linalg.norm(apply(T_world, setup.grid_R) - apply(T_ball, setup.grid_R), axis=1)
    out["closure_max_mm"] = float(closure.max() * 1e3)
    out["closure_check"] = closure_rows(edges, T_ball, sigmas, rng)
    return out


METHODS = ("arm_arm_ball", "laser_station", "via_world")


def _q(values: list[float], p: float) -> float:
    return round(float(np.percentile(values, p)), 3)


def summarise(trials: list[dict[str, Any]]) -> dict[str, Any]:
    """Medians and 95th percentiles over trials."""
    out: dict[str, Any] = {
        "floor_p95_mm": {"p50": _q([t["floor_p95_mm"] for t in trials], 50)},
        "pivot_rms_mm": {"p50": _q([t["pivot_rms_mm"] for t in trials], 50)},
        "closure_max_mm": {
            p: _q([t["closure_max_mm"] for t in trials], q) for p, q in (("p50", 50), ("p95", 95), ("p99", 99))
        },
    }
    for method in METHODS:
        rows = [t[method] for t in trials]
        out[method] = {
            key: {"p50": _q([r[key] for r in rows], 50), "p95": _q([r[key] for r in rows], 95)}
            for key in rows[0]
        }
    checks = [t["closure_check"] for t in trials]
    out["closure_check"] = {
        "disagreement_mm_p95": _q([c["disagreement_mm"] for c in checks], 95),
        # what CLOSURE_K would have to be for 1 % false alarms with these edges
        "k_for_1pct": _q([c["disagreement_mm"] / c["sigma_mm"] for c in checks], 99),
        "threshold_mm_p50": _q([c["threshold_mm"] for c in checks], 50),
        "false_alarm_rate": round(float(np.mean([c["false_alarm"] for c in checks])), 4),
        **{
            f"caught_{e:g}mm_rate": round(float(np.mean([c[f"caught_{e:g}mm"] for c in checks])), 4)
            for e in INJECTED_EDGE_ERRORS_MM
        },
    }
    rows = [t["arm_arm_ball"] for t in trials if "loo_rms_mm" in t["arm_arm_ball"]]
    if len(rows) > 2:
        loo = np.array([r["loo_rms_mm"] for r in rows])
        reg = np.array([r["reg_max_mm"] for r in rows])
        # can the on-site leave-one-out tell a bad trial from a good one?
        out["arm_arm_ball"]["loo_vs_reg_corr"] = round(float(np.corrcoef(loo, reg)[0, 1]), 3)
    return out


def simulate(config: Config, *, trials: int, seed: int = 0) -> dict[str, Any]:
    setup = build_setup(config, seed=seed)
    if len(setup.usable_balls) < 3:
        return {"config": asdict(config), "error": "fewer than 3 balls usable by both arms", "notes": setup.notes}
    rng = np.random.default_rng(seed + 1)
    rows, failed = [], 0
    for _ in range(trials):
        try:
            rows.append(run_trial(setup, rng))
        except ValueError:  # a pivot fit that rejected itself below 6 orientations: on the robot, a re-run
            failed += 1
    if not rows:
        return {"config": asdict(config), "error": f"all {trials} trials failed a pivot fit", "notes": setup.notes}
    return {
        "config": asdict(config),
        "trials": len(rows),
        "failed_pivot_trials": failed,
        "usable_balls": len(setup.usable_balls),
        "grid_poses": len(setup.grid_R),
        "kinematic_scale": round(config.fk_rms_mm * 1e-3 / setup.unit_rms_m, 3),
        "notes": setup.notes,
        "summary": summarise(rows),
    }


def default_sweep() -> list[Config]:
    """One change at a time from the baseline."""
    base = Config()
    sweep = [base]
    for layout in ("planar4", "planar6", "height4", "height6", "height8"):
        sweep.append(replace(base, name=f"layout={layout}", layout=layout))
    for span in (0.15, 0.45):
        sweep.append(replace(base, name=f"span={span}", span_m=span))
    for height in (0.0, 0.08, 0.25):
        sweep.append(replace(base, name=f"height={height}", height_m=height))
    sweep.append(replace(base, name="dwells=8", dwells=8))
    sweep.append(replace(base, name="tilt=20", tilt_deg=20.0))
    for spacing in (0.6, 1.0):
        sweep.append(replace(base, name=f"spacing={spacing}", spacing_m=spacing))
    for sigma in (0.3, 1.0):
        sweep.append(replace(base, name=f"seat={sigma}", seat_sigma_mm=sigma))
    for fk in (0.27, 1.0):
        sweep.append(replace(base, name=f"fk={fk}", fk_rms_mm=fk))
    sweep.append(replace(base, name="laser_poses=8", laser_poses=8))
    return sweep


def table(results: list[dict[str, Any]]) -> list[str]:
    head = (
        f"{'config':<16} {'balls':>5} {'grid':>4} {'pivot':>6} {'floor':>6} | "
        f"{'ball tLR p95':>12} {'reg max':>8} {'loo':>5} | {'laser tLR':>9} {'reg':>6} | "
        f"{'world tLR':>9} {'reg':>6} | {'loop thr':>8} {'k@1%':>5} {'false':>5} {'5mm':>5} {'10mm':>5}"
    )
    lines = [head, "-" * len(head)]
    for r in results:
        name = r["config"]["name"]
        if "error" in r:
            lines.append(f"{name:<16} {r['error']}")
            continue
        s = r["summary"]

        def cell(method: str, key: str, p: str = "p95", s: dict = s) -> str:
            return f"{s[method][key][p]:.2f}"

        loo = s["arm_arm_ball"].get("loo_rms_mm", {}).get("p50")
        lines.append(
            f"{name:<16} {r['usable_balls']:>5} {r['grid_poses']:>4} {s['pivot_rms_mm']['p50']:>6.2f} "
            f"{s['floor_p95_mm']['p50']:>6.2f} | {cell('arm_arm_ball', 'tlr_p95_mm'):>12} "
            f"{cell('arm_arm_ball', 'reg_max_mm'):>8} {loo if loo is not None else '-':>5} | "
            f"{cell('laser_station', 'tlr_p95_mm'):>9} {cell('laser_station', 'reg_max_mm'):>6} | "
            f"{cell('via_world', 'tlr_p95_mm'):>9} {cell('via_world', 'reg_max_mm'):>6} | "
            f"{s['closure_check']['threshold_mm_p50']:>8.1f} {s['closure_check']['k_for_1pct']:>5.2f} "
            f"{s['closure_check']['false_alarm_rate']:>5.2f} "
            f"{s['closure_check']['caught_5mm_rate']:>5.2f} {s['closure_check']['caught_10mm_rate']:>5.2f}"
        )
    lines.append(
        "mm; p95 over trials of: tLR = p95 over workspace poses of |T p_R - p_L| (incl. floor), reg = max over "
        "workspace of the edge against the best rigid one; pivot/floor/loo = median over trials; loop: "
        "world_provenance's closure threshold (median) and how often it fires with no error / catches an error "
        "put into the left world edge"
    )
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--trials", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--only", nargs="*", help="run only the sweep entries with these names")
    parser.add_argument(
        "--out",
        type=Path,
        default=REPO / "outputs" / "fr3" / "dual_fr3_registration_sim",
        help="directory for the json report",
    )
    args = parser.parse_args(argv)

    configs = [c for c in default_sweep() if not args.only or c.name in args.only]
    results = []
    for config in configs:
        started = time.monotonic()
        results.append(simulate(config, trials=args.trials, seed=args.seed))
        print(f"{config.name}: {time.monotonic() - started:.1f} s", file=sys.stderr, flush=True)
    for line in table(results):
        print(line)
    for r in results:
        for note in r.get("notes") or []:
            print(f"  {r['config']['name']}: {note}")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    args.out.mkdir(parents=True, exist_ok=True)
    path = args.out / f"sim_{stamp}.json"
    path.write_text(
        json.dumps({"schema": SCHEMA, "trials": args.trials, "seed": args.seed, "results": results}, indent=2),
        encoding="utf-8",
    )
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
