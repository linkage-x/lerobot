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
Weigh the FR3 end-effector load with the arm's own joint torque sensors (mass + CoM, flange frame).

Subcommands (math in lerobot.robots.franka_research3.payload_identification):
  check-poses  MuJoCo: clearance between the tool and the arm along the pose sequence, lowest tool point,
               and (with --table-edge-m/--table-top-m) the gap to a table in front of the base
  sim          MuJoCo: synthetic run (optional sensor noise/offsets, bare-arm model error, parts removed)
  collect      real robot: visit the poses, average tau_J at rest, store with libfranka's bare-arm gravity
  fit          one run -> total load with sensor offsets (physical estimate); optionally minus weighed parts
  diff         two runs over the same poses -> the part that differs (e.g. SMR in vs out)
  desk-config  runs -> Franka Desk end-effector JSONs, one per tool configuration (fit without offsets)
  move-waypoint  real robot: one move of the arm-held 6D zero route (start / six_d_zero / check_xplus /
               check_yminus), only between waypoints next to each other on the route `check-poses
               --poses six_d_zero` checks; the calibration page's zero job strings the moves together

Tool configurations (``--tool-config``; what is bolted below the flange):
  gripper_cam_smr  gripper + camera/SMR adaptor + SMR in its nest (10-12 "with SMR")
  gripper_cam      gripper + camera/SMR adaptor, SMR out (10-12 "without SMR")
  gripper          gripper alone, adaptor and camera taken off (the right arm from 11-09)
Runs from before 11-09 carry only ``smr_present`` and read as one of the first two.

Pose sets (``--poses``): ``payload`` (32, the load fit), ``six_d`` (13 wrist poses that turn every
axis of the tool up once, for the BOX 6D payload fit, which the payload poses cannot serve: they keep
the tool within 45 deg of pointing down), or ``payload+six_d`` (both in one visit). The six_d poses
are ordinary static poses for the load fit too.

Desk wants the fit *without* joint offsets, not `fit`'s: the robot subtracts no offsets before its gravity
compensation and its O_F_ext_hat_K. 10-12, the with-offset fit (1.056 kg) loaded into Desk gave
Fz -0.94 N; the no-offset one (0.976 kg) gave -0.17 N. The physical load is less certain than either:
with/without offsets and per reach spread over ~110 g.

Two checks on the real robot:
- SMR difference: collect with the SMR in its nest and without it, `diff` must return ~140 g at the SMR
  centre. The difference cancels sensor offsets and the bare-arm model error, so it checks the sensors'
  linearity and noise, not the absolute load.
- Desk closed loop: load the desk-config JSON into Desk and collect again (`collect` records the load Desk
  had, `desk-config` prints the robot's O_F_ext_hat_K next to its reproduction). Mean Fz should be within
  ~0.2 N of 0. What stays (10-12: Fx -0.8, Fy -0.36 N in every run) is sensor offsets; no load removes it.

On Thor: PYTHONPATH=src with the .venv-fr3 interpreter (see the FR3 control notes); `collect` moves
the arm -- someone must be at the robot with the user stop in reach.
"""

# ruff: noqa: N806  (matrix names: T, J, R, A follow the maths)
from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import math
import sys
import time
import uuid
from pathlib import Path

import numpy as np

from lerobot.robots.franka_research3.backends import (
    PANDA_PY_JOINT_LIMITS_LOWER,
    PANDA_PY_JOINT_LIMITS_UPPER,
    PANDA_PY_JOINT_POSITION_START,
)
from lerobot.robots.franka_research3.payload_identification import (
    GRAVITY_EARTH,
    POSE_SETS,
    default_poses,
    fit_difference,
    fit_load,
    franka_external_wrench,
    load_regressor,
    pose_set,
    reach_groups,
    six_d_zero_moves,
    six_d_zero_waypoints,
    subtract_known,
    urdf_fixed_transform,
    urdf_link_load,
    urdf_subtree_inertial,
    visit_sequence,
)

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))
from tools.fr3 import fr3_arms  # noqa: E402

ASSETS = REPO / "src/lerobot/robots/franka_research3/assets/franka_fr3"
DEFAULT_XML = ASSETS / "fr3_corenetic_gripper.xml"
DEFAULT_URDF = ASSETS / "fr3_corenetic_gripper.urdf"
# Desk's "Corenetic v2.0 BOX" end effector as exported (0.7496 kg, TCP 171.99 mm = the old fingertips)
DEFAULT_DESK_TEMPLATE = REPO / "tools/fr3/desk_templates/corenetic_v2_box.json"
# Desk TCP of the 44 mm fingertips (since 11-06) until an FR3 pivot measures it: the nov4 socket centre
# (the camera labels' TCP: the fingertip ends' centre at the insert's 34.6 mm clamp width), 167.91 mm along
# link_sensor_ft z by the handheld rig's pivot, + the URDF's 22.25 mm flange -> link_sensor_ft. Not
# measured on the arm: on the handheld the gripper sits ~12 mm nearer the rig than the CAD stack says,
# and which layer (rig mount or sensor -> gripper) is not known, so the arm's may be up to ~12 mm off.
# x/y (-0.35/-0.36 mm) are below that and left at 0. Rotation stays the template's (yaw -45 deg).
NEW_TIP_TCP_Z_M = 0.19016
TCP_FLANGE_LINK = "fr3_link8"
SMR_BODY = "link_corenetic_smr"
MOUNT_BODY = "link_corenetic_mount"  # the camera/SMR adaptor; the camera case and the SMR nest hang off it
# Parts weighed on a scale (URDF links); what is left of a fit is the gripper + Part 1 + anything unmodelled.
WEIGHED_LINKS = ("link_corenetic_smr_cam", SMR_BODY)
DEFAULT_APPROACH_RAD = 0.03
# check-poses: path sampling. One joint moving 0.02 rad moves a point 0.85 m out by 17 mm at most,
# about the 15 mm clearance asked; the 10-12 check took 16 samples per move (0.14 rad steps on the
# longest moves), so this is finer everywhere.
PATH_STEP_RAD = 0.02

#: tool configuration -> URDF subtrees that are NOT mounted in it (left out of the CAD inertia / sim mass)
TOOL_CONFIGS: dict[str, tuple[str, ...]] = {
    "gripper_cam_smr": (),
    "gripper_cam": (SMR_BODY,),
    "gripper": (MOUNT_BODY,),
}
TOOL_CONFIG_LABELS = {
    "gripper_cam_smr": "gripper + camera + SMR",
    "gripper_cam": "gripper + camera",
    "gripper": "gripper",
}
# desk-config output names; the first two are the 10-12 files' names
_DESK_SUFFIX = {"gripper_cam_smr": "smr", "gripper_cam": "no_smr", "gripper": "gripper"}
_RUN_SUFFIX = {"gripper_cam_smr": "smr_in", "gripper_cam": "smr_out", "gripper": "gripper"}


def run_tool_config(run: dict) -> str:
    """The tool configuration a run was collected with (runs before 11-09 only say whether the SMR was in)."""
    config = run.get("tool_config")
    if config:
        if config not in TOOL_CONFIGS:
            raise ValueError(f"unknown tool_config {config!r}")
        return config
    return "gripper_cam_smr" if run.get("smr_present", True) else "gripper_cam"


# The FR3's own joint limits (libfranka / the MJCF). panda_py's walls are the Panda's: its j6 runs from
# -0.0175, the FR3's from 0.5445, so a pose has to sit inside both.
FR3_JOINT_LIMITS_LOWER = (-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159)
FR3_JOINT_LIMITS_UPPER = (2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159)
JOINT_LOWER = np.maximum(PANDA_PY_JOINT_LIMITS_LOWER, FR3_JOINT_LIMITS_LOWER)
JOINT_UPPER = np.minimum(PANDA_PY_JOINT_LIMITS_UPPER, FR3_JOINT_LIMITS_UPPER)


def _sequence(approach_rad: float, poses: str = "payload"):
    return visit_sequence(pose_set(poses), approach_rad, JOINT_LOWER, JOINT_UPPER)


# ---------------------------------------------------------------- MuJoCo helpers (check-poses, sim)
def _load_mujoco(xml: Path):
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml))
    return mujoco, model, mujoco.MjData(model)


def _subtree(mujoco, model, root: str) -> list[int]:
    rid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, root)
    out = []
    for b in range(model.nbody):
        a = b
        while a not in (rid, 0):
            a = model.body_parentid[a]
        if a == rid:
            out.append(b)
    return out


def _set_q(mujoco, model, data, q) -> None:
    data.qpos[:] = 0.0
    data.qpos[:7] = q
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def _surface_points(mujoco, model, bodies: set[int], spacing_m: float) -> dict[int, np.ndarray]:
    """Points on the visual surfaces of each body, in the body frame (~spacing_m apart, area-weighted)."""
    rng = np.random.default_rng(0)
    out: dict[int, list[np.ndarray]] = {}
    for g in range(model.ngeom):
        b = int(model.geom_bodyid[g])
        if b not in bodies or model.geom_group[g] not in (
            1,
            2,
        ):  # visual geoms; collision ones duplicate them
            continue
        if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
            mid = model.geom_dataid[g]
            v = model.mesh_vert[model.mesh_vertadr[mid] : model.mesh_vertadr[mid] + model.mesh_vertnum[mid]]
            f = model.mesh_face[model.mesh_faceadr[mid] : model.mesh_faceadr[mid] + model.mesh_facenum[mid]]
            tri = v[f]
            area = 0.5 * np.linalg.norm(np.cross(tri[:, 1] - tri[:, 0], tri[:, 2] - tri[:, 0]), axis=1)
            n = int(area.sum() / spacing_m**2) + 1
            k = rng.choice(len(tri), size=n, p=area / area.sum())
            u, w = rng.random((2, n, 1))
            flip = (u + w) > 1
            u, w = np.where(flip, 1 - u, u), np.where(flip, 1 - w, w)
            local = tri[k, 0] + u * (tri[k, 1] - tri[k, 0]) + w * (tri[k, 2] - tri[k, 0])
            local = np.vstack([local, v])
        elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
            r = model.geom_size[g][0]
            n = int(4 * np.pi * r**2 / spacing_m**2) + 1
            x = rng.standard_normal((n, 3))
            local = r * x / np.linalg.norm(x, axis=1, keepdims=True)
        else:
            continue
        # geom frame -> body frame
        R = np.zeros(9)
        mujoco.mju_quat2Mat(R, model.geom_quat[g])
        out.setdefault(b, []).append(local @ R.reshape(3, 3).T + model.geom_pos[g])
    return {b: np.vstack(p) for b, p in out.items()}


def _world(data, clouds: dict[int, np.ndarray]) -> np.ndarray:
    return np.vstack([p @ data.xmat[b].reshape(3, 3).T + data.xpos[b] for b, p in clouds.items()])


class _NearField:
    """Point cloud hashed into cells: exact nearest distance below ``cell`` m, ``inf`` beyond."""

    _OFFSETS = np.array([(i, j, k) for i in (-1, 0, 1) for j in (-1, 0, 1) for k in (-1, 0, 1)])

    def __init__(self, points: np.ndarray, cell: float = 0.02) -> None:
        self.cell = cell
        idx = np.floor(points / cell).astype(np.int64)
        keys = self._key(idx)
        order = np.argsort(keys)
        self.points, self.keys = points[order], keys[order]
        # queries farther than one cell outside the cloud's box cannot be within a cell of it
        self.lo_box, self.hi_box = points.min(0) - cell, points.max(0) + cell

    @staticmethod
    def _key(idx: np.ndarray) -> np.ndarray:
        idx = idx + 2**20
        return (idx[..., 0] << 42) | (idx[..., 1] << 21) | idx[..., 2]

    def min_distance(self, query: np.ndarray) -> float:
        query = query[np.all((query >= self.lo_box) & (query <= self.hi_box), axis=1)]
        if not len(query):
            return np.inf
        cells = np.floor(query / self.cell).astype(np.int64)[:, None, :] + self._OFFSETS[None]
        keys = self._key(cells).reshape(-1)
        lo = np.searchsorted(self.keys, keys, "left")
        hi = np.searchsorted(self.keys, keys, "right")
        hit = hi > lo
        if not hit.any():
            return np.inf
        q = np.repeat(query, len(self._OFFSETS), axis=0)[hit]
        lo, n = lo[hit], (hi - lo)[hit]
        best = np.inf
        # every (query point, cloud point in a neighbouring cell) pair, in chunks of ~2M pairs
        bounds = np.searchsorted(np.cumsum(n), np.arange(1, int(n.sum()) // 2_000_000 + 2) * 2_000_000)
        start = 0
        for stop in np.unique(np.r_[bounds, len(n)]):
            if stop <= start:
                continue
            ln = n[start:stop]
            first = np.repeat(np.cumsum(ln) - ln, ln)
            idx = np.repeat(lo[start:stop], ln) + (np.arange(int(ln.sum())) - first)
            d2 = ((self.points[idx] - np.repeat(q[start:stop], ln, axis=0)) ** 2).sum(1)
            best = min(best, float(d2.min()))
            start = stop
        # beyond one cell the search above is no longer complete: a nearer point may sit two cells away
        d = float(np.sqrt(best))
        return d if d <= self.cell else np.inf


def check_poses(args) -> int:
    mujoco, model, data = _load_mujoco(args.xml)
    tool_bodies = set(_subtree(mujoco, model, "base_link"))
    # fr3_link7 carries the flange the tool is bolted to -> rigid with the tool, left out
    arm_bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"fr3_link{i}") for i in range(7)}
    tool_pts = _surface_points(mujoco, model, tool_bodies, args.spacing_mm / 1000)
    arm_clouds = _surface_points(mujoco, model, arm_bodies, args.spacing_mm / 1000)
    arm_fields = {b: _NearField(p) for b, p in arm_clouds.items()}
    # what can reach a table in front of the base: the tool and the links beyond the shoulder
    # (10 mm sampling is plenty against a 50 mm margin and keeps the check to seconds)
    table = args.table_edge_m is not None and args.table_top_m is not None
    reach_bodies = tool_bodies | {
        mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"fr3_link{i}") for i in range(3, 8)
    }
    reach_clouds = _surface_points(mujoco, model, reach_bodies, max(args.spacing_mm, 10.0) / 1000) if table else {}
    poses = pose_set(args.poses)
    sequence = _sequence(args.approach_rad, args.poses)
    targets = [np.asarray(PANDA_PY_JOINT_POSITION_START)] + [t for t, _, _ in sequence]
    owner = [0] + [i for _, _, i in sequence]
    # samples along each move, one per PATH_STEP_RAD of the largest joint change (the 0.03 rad
    # approach moves need two, not the 16 a long move does)
    path = []
    for a, b, i in zip(targets[:-1], targets[1:], owner[1:], strict=True):
        n = max(2, int(np.ceil(np.abs(b - a).max() / PATH_STEP_RAD)))
        path.extend((i, a + (b - a) * s) for s in np.linspace(0, 1, n + 1)[1:])
    per_pose = np.full(len(poses), np.inf)
    worst = (np.inf, None, None)
    lowest = np.inf
    table_gap = (np.inf, None)
    for t, q in path:
        _set_q(mujoco, model, data, q)
        tool = _world(data, tool_pts)
        lowest = min(lowest, float(tool[:, 2].min()))
        if table:
            pts = _world(data, reach_clouds)
            over = pts[pts[:, 0] > args.table_edge_m]
            if len(over):
                gap = float(over[:, 2].min()) - args.table_top_m
                if gap < table_gap[0]:
                    table_gap = (gap, t)
        for b, field in arm_fields.items():
            # tool points into the arm body's frame, where its cloud is hashed once
            dist = field.min_distance((tool - data.xpos[b]) @ data.xmat[b].reshape(3, 3))
            per_pose[t] = min(per_pose[t], dist)
            if dist < worst[0]:
                worst = (dist, t, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b))
    ft = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "link_sensor_ft")
    print(
        f"pose set {args.poses}: poses={len(poses)} path samples={len(path)} surface spacing {args.spacing_mm} mm "
        f"({sum(len(p) for p in tool_pts.values())} tool points)"
    )
    for i, q in enumerate(poses):
        _set_q(mujoco, model, data, q)
        up = data.xmat[ft].reshape(3, 3).T @ np.array([0.0, 0.0, 1.0])
        k = int(np.argmax(np.abs(up)))
        print(
            f"  pose {i:2d} q = {np.round(q, 3).tolist()}  min clearance getting there "
            f"{'  > 20' if np.isinf(per_pose[i]) else f'{per_pose[i] * 1000:6.1f}'} mm  "
            f"up: tool {'xyz'[k]}{'+' if up[k] > 0 else '-'} ({abs(up[k]):.2f})"
        )
    dist, t, body = worst
    if body is None:
        print(
            "min tool-arm clearance along the path: > 20 mm everywhere (nothing of the arm within one hash cell)"
        )
    else:
        print(
            f"min tool-arm clearance along the path: {dist * 1000:.1f} mm (vs {body}, on the way to pose {t}; "
            "inf in the table = nothing within 20 mm)"
        )
    print(f"lowest tool point: z = {lowest * 1000:.0f} mm above the robot base plane")
    ok = dist >= args.min_clearance_mm / 1000
    if table:
        gap, t = table_gap
        if t is None:
            print(f"table (edge x > {args.table_edge_m:.3f} m, top z = {args.table_top_m:.3f} m): nothing over it")
        else:
            table_ok = gap >= args.table_margin_mm / 1000
            ok = ok and table_ok
            print(
                f"table (edge x > {args.table_edge_m:.3f} m, top z = {args.table_top_m:.3f} m): lowest arm/tool point "
                f"over it {gap * 1000:.0f} mm above the top (on the way to pose {t}; need >= {args.table_margin_mm:.0f})"
            )
    print(
        f"check-poses={'PASS' if ok else 'FAIL'} (need >= {args.min_clearance_mm:.0f} mm incl. the point "
        "spacing; " + ("the table is a plane, " if table else "the table, ")
        + "laser tracker, cables and anything on the table are NOT modelled -- check those by eye)"
    )
    return 0 if ok else 1


def simulate(args) -> int:
    mujoco, model, data = _load_mujoco(args.xml)
    rng = np.random.default_rng(args.seed)
    tool = _subtree(mujoco, model, "base_link")
    config = getattr(args, "tool_config", None) or ("gripper_cam" if args.without_smr else "gripper_cam_smr")
    poses = getattr(args, "poses", None) or "payload"
    for root in TOOL_CONFIGS[config]:
        for b in _subtree(mujoco, model, root):
            model.body_mass[b] = 0.0
            model.body_inertia[b] = 0.0
    # "libfranka" bare-arm model: same arm, no tool, arm link masses off by --arm-model-error (relative)
    arm_model = mujoco.MjModel.from_xml_path(str(args.xml))
    arm_data = mujoco.MjData(arm_model)
    for b in tool:
        arm_model.body_mass[b] = 0.0
        arm_model.body_inertia[b] = 0.0
    model_rng = np.random.default_rng(1000)  # the arm-model error is a property of the arm, not of the run
    for i in range(1, 8):
        b = mujoco.mj_name2id(arm_model, mujoco.mjtObj.mjOBJ_BODY, f"fr3_link{i}")
        arm_model.body_mass[b] *= 1.0 + args.arm_model_error * model_rng.standard_normal()
    bias = args.bias_nm * np.random.default_rng(2000).standard_normal(7)  # a sensor property, same every run
    # static friction: tau_J reads +h_j off when joint j last moved up, -h_j when it last moved down
    hyst = args.hysteresis_nm * np.abs(np.random.default_rng(3000).standard_normal(7))
    samples: dict[int, dict] = {}
    prev, last_dir = np.asarray(PANDA_PY_JOINT_POSITION_START, dtype=np.float64), np.zeros(7)
    for target, tag, i in _sequence(args.approach_rad, poses):
        step = target - prev
        last_dir = np.where(np.abs(step) > 1e-9, np.sign(step), last_dir)
        prev = target
        if tag is None:
            continue
        _set_q(mujoco, model, data, target)
        tau = data.qfrc_bias[:7].copy() + bias + hyst * last_dir + args.noise_nm * rng.standard_normal(7)
        _set_q(mujoco, arm_model, arm_data, target)
        rec = samples.setdefault(i, {"q": target.tolist(), "g_arm": arm_data.qfrc_bias[:7].tolist()})
        rec[f"tau_J_{tag}"] = tau.tolist()
    for rec in samples.values():
        parts = [v for k, v in rec.items() if k.startswith("tau_J_")]
        rec["tau_J"] = np.mean(parts, axis=0).tolist()
    samples = [samples[i] for i in sorted(samples)]
    # ground truth: everything below the flange, in the flange frame
    _set_q(mujoco, model, data, default_poses()[0])
    f8 = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "fr3_link8")
    # summed by hand: body_subtreemass (and subtree_com, which divides by it) is fixed at compile time
    # and would still count a zeroed SMR
    mass = model.body_mass[tool]
    com_world = (mass[:, None] * data.xipos[tool]).sum(0) / mass.sum()
    com = data.xmat[f8].reshape(3, 3).T @ (com_world - data.xpos[f8])
    truth = {"mass_kg": float(mass.sum()), "com_flange_mm": (com * 1000).round(3).tolist()}
    run = {
        "source": "mujoco",
        "xml": str(args.xml),
        "tool_config": config,
        "poses_set": poses,
        "smr_present": config == "gripper_cam_smr",
        "sim": {
            "noise_nm": args.noise_nm,
            "bias_nm": args.bias_nm,
            "arm_model_error": args.arm_model_error,
            "hysteresis_nm": hyst.round(4).tolist(),
            "approach_rad": args.approach_rad,
            "seed": args.seed,
            "truth": truth,
        },
        "gravity": list(GRAVITY_EARTH),
        "poses": samples,
    }
    _write_run(run, args.out)
    print(f"truth: {truth}")
    return 0


# ---------------------------------------------------------------- real robot
def collect(args) -> int:
    from panda_py import Panda

    poses = pose_set(args.poses)
    config = args.tool_config or ("gripper_cam_smr" if args.smr_present else "gripper_cam")
    print(
        f"collect: {len(poses)} static poses ({args.poses}) x "
        f"{'2 (from below and above)' if args.approach_rad > 0 else '1'}, "
        f"speed_factor={args.speed_factor}, hold {args.hold_s}s after {args.settle_s}s settle; "
        f"tool: {TOOL_CONFIG_LABELS[config]}"
    )
    print(
        "Run `check-poses` first and look at the workspace: the table, laser tracker and cables are not modelled."
    )
    panda = Panda(args.robot_ip)
    model = panda.get_model()
    # Panda.get_state() is the last state of the last control loop -- frozen while the arm stands still
    # (10-12: unique=1 over 400 samples). Robot.read_once() waits for a fresh 1 kHz state instead.
    robot = panda.get_robot()
    first = robot.read_once()
    # check-poses only covers the path between the poses: start from panda_py's start pose (j1-j4)
    off = np.abs(np.asarray(first.q)[:4] - poses[0][:4]).max()
    if off > 0.1:
        print(f"the arm is {off:.2f} rad away from the start pose in j1-j4; run fr3_move_to_start first")
        return 1
    if not args.yes and input("The arm will move. Type yes to start: ").strip().lower() != "yes":
        print("aborted")
        return 1
    desk_load = {
        k: list(np.ravel(getattr(first, k))) if np.ndim(getattr(first, k)) else float(getattr(first, k))
        for k in ("m_ee", "F_x_Cee", "I_ee", "m_load", "F_x_Cload", "m_total", "F_x_Ctotal")
        if hasattr(first, k)
    }
    run = {
        "source": "fr3",
        "robot_ip": args.robot_ip,
        "arm": args.arm,
        "tool_config": config,
        "poses_set": args.poses,
        "smr_present": config == "gripper_cam_smr",
        "note": args.note,
        "desk_load_at_start": desk_load,
        "gravity": list(GRAVITY_EARTH),
        "complete": False,
        "poses": [],
    }
    run["approach_rad"] = args.approach_rad
    out = _write_run(run, args.out)  # rewritten after every measurement: an abort keeps what was measured
    box = None
    if args.six_d_device_id:
        try:
            box = _BoxSixDLog(int(args.six_d_device_id), six_d_log_path(out))
        except RuntimeError as exc:
            print(f"BOX 6D logging refused: {exc}")
            return 1
        run["six_d"] = {
            "device_id": int(args.six_d_device_id),
            "log": box.path.name,
            "uptime_at_start_s": box.uptime_s,  # BOX clock: s since its power-up
        }
        _write_run(run, out, quiet=True)
        print(f"BOX {args.six_d_device_id}: 6D + IMU of every hold -> {box.path}")
        warning = six_d_warmup_warning(box.uptime_s)
        if warning:
            print(f"WARNING: {warning}")  # the page lifts "WARNING: " lines into the job's warnings
    pending: dict | None = None
    try:
        for target, tag, i in _sequence(args.approach_rad, args.poses):
            if not _move(panda, robot, target, args):
                print(f"  stopped on the way to pose {i}; {len(run['poses'])} complete poses are in {out}")
                return 1
            if tag is None:  # approach point: only sets the direction every joint arrives from
                continue
            m = _measure(robot, model, args, box, (i, tag))
            if pending is None or pending["index"] != i:
                pending = {"index": i, "q_target": target.tolist(), "measurements": {}}
            pending["measurements"][tag] = m
            print(
                f"  pose {i:2d}/{len(poses)} {tag:6s}: n={m['n']} unique={m['n_unique']} |dq|max={m['dq_abs_max']:.4f} "
                f"tau_J-g_arm={np.round(np.subtract(m['tau_J'], m['g_arm']), 3).tolist()}"
            )
            expected = {"single"} if args.approach_rad <= 0 else {"up", "down"}
            if set(pending["measurements"]) == expected:
                run["poses"].append(_combine(pending))
                _write_run(run, out, quiet=True)
                if "up" in pending["measurements"]:
                    h = (
                        np.subtract(
                            pending["measurements"]["up"]["tau_J"], pending["measurements"]["down"]["tau_J"]
                        )
                    ) / 2
                    print(f"          (up - down)/2 = {np.round(h, 3).tolist()} Nm")
        _move(panda, robot, poses[0], args)
        run["complete"] = True
        _write_run(run, out)
    finally:
        with contextlib.suppress(Exception):
            panda.stop_controller()
        if box is not None:
            box.close()
            print(f"BOX 6D: {box.rows} samples during the holds in {box.path}")
    return 0


#: The BOX vendor's warm-up before a 6D calibration (数采Box传感器标定与操作说明 v1.1).
SIX_D_WARMUP_S = 30 * 60


def six_d_warmup_warning(uptime_s: float | None) -> str:
    """The reminder for a 6D log started on a BOX on for less than the warm-up, or ""; never a refusal."""
    if uptime_s is None or uptime_s >= SIX_D_WARMUP_S:
        return ""
    return (
        f"BOX 才上电 {uptime_s / 60:.0f} 分钟；6D 拟合要求预热 30 分钟，读数会随温度漂，"
        "拟合可能不达标（照常继续）"
    )


def six_d_log_path(run: Path) -> Path:
    """Where a run's BOX 6D log goes: ``<run>.six_d.log`` (rows in box_six_d's ``fit`` format)."""
    return Path(run).with_name(Path(run).name.removesuffix(".json") + ".six_d.log")


class _BoxSixDLog:
    """The FR3 gripper BOX's 6D force + IMU, written while the arm holds still.

    Opens its own BOX client on :15000 (as fr3_gripper_teleop does), so the recorder must not be
    connected; the BOX's mode is never touched (keep_mode_device_ids). Rows are box_six_d.py
    ``fit`` input, plus the pose index and approach tag they were taken at.
    """

    def __init__(self, device_id: int, path: Path) -> None:
        from tools.thor.box_sdk import box_client as bc

        template = bc.BoxClientConfig(poll_interval_s=0.002, stale_threshold_s=0.5)
        self.pool = bc.BoxPool(bc.BoxFleetConfig(enabled=True, boxes=[template], keep_mode_device_ids=[device_id]))
        if not self.pool.start():
            raise RuntimeError("BOX client did not start: box_sdk missing, or :15000 taken (recorder connected?)")
        try:
            ids = [int(d.device_id) for d in self.pool.discovered()]
            if device_id not in ids:
                raise RuntimeError(f"BOX {device_id} not found; online: {ids}")
            self.client = self.pool.client_for_device(device_id)
            deadline = time.monotonic() + 3.0
            while not self._sample() and time.monotonic() < deadline:
                time.sleep(0.02)
            first = self._sample()
            if not first:
                raise RuntimeError(f"BOX {device_id}: no 6D/IMU sample within 3 s")
            ts = first[0].get("timestamp")
            #: seconds since the BOX powered up (its 6D timestamp is µs on the MCU clock), None if unknown
            self.uptime_s = float(ts) / 1e6 if isinstance(ts, (int, float)) and ts > 0 else None
        except Exception:
            self.pool.stop()
            raise
        self.path = path
        self.fh = path.open("a", buffering=1)
        self.rows = 0
        self.last_ts = None

    def _sample(self):
        sensors = (self.client.read() if self.client is not None else {}).get("sensors") or {}
        f, imu = sensors.get("box_six_d_force") or {}, sensors.get("box_imu") or {}
        if not f.get("fxyz_mxyz") or not imu.get("acc_g"):
            return None
        return f, imu, sensors.get("box_gripper") or {}

    def poll(self, where: tuple[int, str]) -> None:
        got = self._sample()
        if got is None:
            return
        f, imu, grip = got
        if f.get("timestamp") == self.last_ts:
            return
        self.last_ts = f.get("timestamp")
        row = {
            "t": round(time.time(), 4),
            "k": "box_six_d_force",
            "ts": self.last_ts,
            "f": f["fxyz_mxyz"],
            "raw": f.get("fxyz_mxyz_raw"),
            "acc": imu["acc_g"],
            "gyr": imu.get("gyr_deg_s") or [0.0, 0.0, 0.0],
            "dist": grip.get("distance_m"),
            "pose": where[0],
            "tag": where[1],
        }
        self.fh.write(json.dumps(row) + "\n")
        self.rows += 1

    def close(self) -> None:
        with contextlib.suppress(Exception):
            self.fh.close()
        with contextlib.suppress(Exception):
            self.pool.stop()


def _measure(robot, model, args, box: _BoxSixDLog | None = None, where: tuple[int, str] = (-1, "")) -> dict:
    time.sleep(args.settle_s)
    qs, taus, dqs, ext = [], [], [], []
    t_end = time.monotonic() + args.hold_s
    while time.monotonic() < t_end:
        s = robot.read_once()
        qs.append(s.q)
        taus.append(s.tau_J)
        dqs.append(s.dq)
        ext.append(s.O_F_ext_hat_K)
        if box is not None:
            box.poll(where)
        time.sleep(1.0 / args.rate_hz)
    q_mean = np.mean(qs, axis=0)
    g_arm = np.asarray(model.gravity(q_mean.tolist(), 0.0, [0.0, 0.0, 0.0], list(GRAVITY_EARTH)))
    tau = np.asarray(taus)
    return {
        "q": q_mean.tolist(),
        "tau_J": tau.mean(axis=0).tolist(),
        "tau_J_std": tau.std(axis=0).tolist(),
        "g_arm": g_arm.tolist(),
        "dq_abs_max": float(np.abs(dqs).max()),
        "O_F_ext_hat_K": np.mean(ext, axis=0).tolist(),
        "n": len(taus),
        "n_unique": len(np.unique(tau, axis=0)),  # a cached state would repeat itself
    }


def _combine(pending: dict) -> dict:
    """One pose record: means over the up/down measurements (hysteresis cancels), each kept as well."""
    ms = pending["measurements"]
    rec = {
        k: np.mean([m[k] for m in ms.values()], axis=0).tolist()
        for k in ("q", "tau_J", "g_arm", "O_F_ext_hat_K")
    }
    rec["q_target"] = pending["q_target"]
    rec["dq_abs_max"] = max(m["dq_abs_max"] for m in ms.values())
    rec["n_unique"] = min(m["n_unique"] for m in ms.values())
    for tag, m in ms.items():
        rec[f"tau_J_{tag}"] = m["tau_J"]
        rec[f"q_{tag}"] = m["q"]
        rec[f"tau_J_std_{tag}"] = m["tau_J_std"]
    return rec


def active_errors(errors) -> list[str]:
    return [name for name in dir(errors) if not name.startswith("_") and getattr(errors, name) is True]


# Thor's link to the FR3 drops ~30 % of the 1 kHz command packets (100 Mbps segment, 2373-byte state
# packets): a move can die of this reflex any time. Only this one is retried -- anything else (a
# collision reflex above all) stops the run.
_RETRYABLE = {"communication_constraints_violation"}


def _move(panda, robot, q: np.ndarray, args) -> bool:
    for attempt in range(args.max_retries + 1):
        if panda.move_to_joint_position(q.tolist(), speed_factor=args.speed_factor) is not False:
            return True
        errs = active_errors(robot.read_once().last_motion_errors)
        if not errs or not set(errs) <= _RETRYABLE or attempt == args.max_retries:
            print(f"  move failed ({errs or 'no error reported'}); stopping after {attempt + 1} attempt(s)")
            return False
        print(f"  move aborted by {errs}; recover and retry ({attempt + 1}/{args.max_retries})")
        panda.recover()
        time.sleep(0.5)
    return False


# A waypoint counts as "here" when every joint is this close to it: panda_py ends a move within a few
# mrad, and an arm that was pushed or left elsewhere must not start a move check-poses never sampled.
WAYPOINT_TOL_RAD = 0.05


def nearest_waypoint(q) -> tuple[str | None, float]:
    """The zero-route waypoint the arm stands at (within WAYPOINT_TOL_RAD on every joint), and how far off."""
    q = np.asarray(q, dtype=float)
    name, off = min(
        ((n, float(np.abs(q - p).max())) for n, p in six_d_zero_waypoints().items()), key=lambda item: item[1]
    )
    return (name if off <= WAYPOINT_TOL_RAD else None), off


def move_waypoint(args) -> int:
    from panda_py import Panda

    points = six_d_zero_waypoints()
    panda = Panda(args.robot_ip)
    robot = panda.get_robot()
    here, off = nearest_waypoint(robot.read_once().q)
    if here is None:
        print(
            f"REFUSED the arm is at no waypoint of the zero route (nearest {off:.3f} rad off); "
            "move it to the start pose first"
        )
        return 1
    if here == args.to:
        print(f"REACHED {args.to} (already there, {off:.4f} rad)")
        return 0
    if frozenset((here, args.to)) not in six_d_zero_moves():
        print(f"REFUSED {here} -> {args.to} is not a move of the checked route")
        return 1
    print(f"WAYPOINT {here} -> {args.to} speed_factor={args.speed_factor}")
    try:
        ok = _move(panda, robot, points[args.to], args)
    finally:
        with contextlib.suppress(Exception):
            panda.stop_controller()
    err = float(np.abs(np.asarray(robot.read_once().q) - points[args.to]).max())
    if not ok:
        print(f"FAILED on the way to {args.to} ({err:.3f} rad off it)")
        return 1
    print(f"REACHED {args.to} ({err:.4f} rad)")
    return 0


# ---------------------------------------------------------------- fitting
def _write_run(run: dict, out: Path | None, quiet: bool = False) -> Path:
    run.setdefault("created", dt.datetime.now().isoformat(timespec="seconds"))
    if out is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = (
            REPO
            / "outputs/fr3_payload"
            / f"{stamp}_{run['source']}_{_RUN_SUFFIX[run_tool_config(run)]}.json"
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(run, indent=1))
    if not quiet:
        print(f"run written: {out}")
    return out


def _read_run(path: Path) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    run = json.loads(Path(path).read_text())
    p = run["poses"]
    return (
        run,
        np.array([s["q"] for s in p]),
        np.array([s["tau_J"] for s in p]),
        np.array([s["g_arm"] for s in p]),
    )


def _fmt(m: float, c: np.ndarray) -> str:
    return f"{m * 1000:.1f} g at ({', '.join(f'{v * 1000:.1f}' for v in c)}) mm"


def fit(args) -> int:
    run, qs, tau, g_arm = _read_run(args.run)
    if not run.get("complete", True):
        print(f"WARNING: incomplete run ({len(qs)} of {len(pose_set(run.get('poses_set') or 'payload'))} poses)")
    fits = {"global": fit_load(qs, tau, g_arm, run["gravity"], with_bias=not args.no_bias)}
    if not args.no_bias:
        fits["per_reach"] = fit_load(qs, tau, g_arm, run["gravity"], groups=reach_groups(qs))
    report = {"run": str(args.run), "fits": {k: v.as_dict() for k, v in fits.items()}}
    if "tau_J_up" in run["poses"][0]:
        h = np.array([np.subtract(p["tau_J_up"], p["tau_J_down"]) / 2 for p in run["poses"]])
        report["hysteresis_nm"] = {
            "mean": np.round(h.mean(0), 4).tolist(),
            "std": np.round(h.std(0), 4).tolist(),
        }
        print(
            f"joint friction hysteresis (up - down)/2, cancelled by the mean: {report['hysteresis_nm']['mean']} Nm "
            f"(spread over poses {report['hysteresis_nm']['std']})"
        )
    elif run.get("source") == "fr3":
        print("WARNING: one measurement per pose -- joint friction hysteresis (0.1-0.4 Nm) is in the result")
    for name, est in fits.items():
        print(
            f"[{name}] total load: {_fmt(est.mass_kg, est.com_flange_m)} (flange frame)  "
            f"+-{est.mass_std_kg * 1000:.1f} g, CoM +-{np.round(est.com_std_m * 1000, 1).tolist()} mm "
            f"(1 sigma, noise only)"
        )
        print(
            f"    residual rms per joint (Nm) {report['fits'][name]['residual_rms_nm']}  cond {est.condition_number:.0f}"
        )
    use = "global"
    if "per_reach" in fits:
        g, r = fits["global"], fits["per_reach"]
        # the per-reach fit absorbs the bare-arm model error of j1-j4 into its offsets, the global one cannot
        dm, dc = abs(g.mass_kg - r.mass_kg), np.abs(g.com_flange_m - r.com_flange_m)
        agree = dm <= 2 * r.mass_std_kg and np.all(dc <= 2 * np.maximum(r.com_std_m, 1e-3))
        use = "global" if agree else "per_reach"
        report["agreement"] = {
            "agree": bool(agree),
            "dmass_g": round(dm * 1000, 1),
            "dcom_mm": np.round(dc * 1000, 1).tolist(),
            "use": use,
        }
        print(
            f"global vs per-reach: dmass {dm * 1000:.1f} g, dCoM {np.round(dc * 1000, 1).tolist()} mm -> "
            + (
                "agree, use [global]"
                if agree
                else "DISAGREE: the bare-arm gravity model is off between reaches; use [per_reach] (less precise)"
            )
        )
    est = fits[use]
    report["load"] = {
        "fit": use,
        "mass_kg": round(est.mass_kg, 4),
        "com_flange_m": np.round(est.com_flange_m, 4).tolist(),
    }
    print(
        f"[{use}] end-effector load ({TOOL_CONFIG_LABELS[run_tool_config(run)]}): "
        f"mass {est.mass_kg:.3f} kg, CoM {np.round(est.com_flange_m, 4).tolist()} m in the flange frame "
        "(inertia is not identifiable from statics)"
    )
    if not args.no_bias:
        print("  not the load for Desk (the robot subtracts no offsets): use desk-config, or fit --no-bias")
    if args.subtract_weighed:
        config = run_tool_config(run)
        mounted = {"gripper_cam_smr": WEIGHED_LINKS, "gripper_cam": WEIGHED_LINKS[:1], "gripper": ()}[config]
        links = list(mounted)
    else:
        links = [k for k in (args.subtract or "").split(",") if k]
    if links:
        parts = [urdf_link_load(args.urdf, k) for k in links]
        m, c = subtract_known(est, parts)
        report["minus"] = {
            "fit": use,
            "links": links,
            "mass_kg": round(m, 5),
            "com_flange_mm": np.round(c * 1000, 2).tolist(),
        }
        print(f"[{use}] minus weighed {links}: {_fmt(m, c)}")
    if "sim" in run:
        print(f"  sim truth (total): {run['sim']['truth']}")
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    return 0


def diff(args) -> int:
    run_a, qa, tau_a, _ = _read_run(args.run_a)
    run_b, qb, tau_b, _ = _read_run(args.run_b)
    if qa.shape != qb.shape or np.abs(qa - qb).max() > args.max_q_mismatch:
        raise SystemExit(f"runs are not over the same poses (max |dq| = {np.abs(qa - qb).max():.4f} rad)")
    est = fit_difference((qa + qb) / 2, tau_a, tau_b, run_a["gravity"])
    print(
        f"A - B: {_fmt(est.mass_kg, est.com_flange_m)}  +-{est.mass_std_kg * 1000:.1f} g, "
        f"CoM +-{np.round(est.com_std_m * 1000, 1).tolist()} mm; residual rms {est.as_dict()['residual_rms_nm']}"
    )
    if args.expect_link:
        m, c = urdf_link_load(args.urdf, args.expect_link)
        print(f"expected ({args.expect_link}): {_fmt(m, c)}")
        print(
            f"  error: mass {(est.mass_kg - m) * 1000:+.1f} g, CoM "
            f"{np.round((est.com_flange_m - c) * 1000, 1).tolist()} mm (|{np.linalg.norm(est.com_flange_m - c) * 1000:.1f}|)"
        )
    return 0


def _wrench_with_load(qs, tau, g_arm, mass: float, com: np.ndarray, gravity) -> np.ndarray:
    """What the robot would report as O_F_ext_hat_K at each pose with this load configured in Desk."""
    theta = np.r_[mass, mass * np.asarray(com)]
    return np.array(
        [
            franka_external_wrench(q, t - g - load_regressor(q, gravity) @ theta)
            for q, t, g in zip(qs, tau, g_arm, strict=True)
        ]
    )


def _fmt_force(w: np.ndarray) -> str:
    f = w[:, :3]
    return f"mean F {np.round(f.mean(0), 2).tolist()} N, max |F| {np.linalg.norm(f, axis=1).max():.2f} N"


def _desk_names(args) -> dict[str, str]:
    """Desk entry name per tool configuration: ``--name CONFIG=NAME``, plus the 10-12 two-state flags."""
    names = {}
    if getattr(args, "name_smr", None):
        names["gripper_cam_smr"] = args.name_smr
    if getattr(args, "name_no_smr", None):
        names["gripper_cam"] = args.name_no_smr
    for item in getattr(args, "name", None) or []:
        config, sep, name = item.partition("=")
        if not sep or config not in TOOL_CONFIGS or not name.strip():
            raise SystemExit(f"--name {item!r}: expected CONFIG=NAME with CONFIG one of {sorted(TOOL_CONFIGS)}")
        names[config] = name.strip()
    return names


def desk_tcp(
    template_tcp: dict,
    *,
    tcp_z_m: float | None = None,
    pivot: Path | None = None,
    keep_template: bool = False,
    urdf: Path = DEFAULT_URDF,
) -> tuple[dict, dict]:
    """Desk's TCP (flange frame, translation in m + rpy) and where it came from.

    Precedence: an FR3 pivot (``fr3_tcp_pivot.py fit`` / ``fit-joint``: the socket centre the arm itself
    turned about, ``fk.s_ee_mm`` in the URDF frame it names, taken to the flange through the URDF), a
    manual z, the template as is, else the new tips' estimate NEW_TIP_TCP_Z_M. Rotation is the template's.
    """
    tcp = json.loads(json.dumps(template_tcp or {}))
    tcp.setdefault("rotation", {"roll": 0.0, "pitch": 0.0, "yaw": -math.pi / 4})
    tr = tcp.setdefault("translation", {"x": 0.0, "y": 0.0, "z": 0.0})
    if pivot is not None:
        rep = json.loads(Path(pivot).read_text(encoding="utf-8"))
        fk = rep.get("fk") or {}
        frame = rep.get("frame") or ""
        if "s_ee_mm" not in fk or not frame:
            raise SystemExit(f"{pivot}: no fk.s_ee_mm / frame -- not a fr3_tcp_pivot.py fit report")
        T = urdf_fixed_transform(urdf, TCP_FLANGE_LINK, frame)
        p = T[:3, :3] @ (np.asarray(fk["s_ee_mm"], dtype=float) / 1000.0) + T[:3, 3]
        tr.update(dict(zip("xyz", (round(float(v), 5) for v in p), strict=True)))
        sigma = fk.get("s_sigma_mm")
        return tcp, {
            "kind": "pivot",
            "pivot": str(pivot),
            "frame": frame,
            "s_ee_mm": fk["s_ee_mm"],
            "s_sigma_mm": sigma,
            "residual_rms_mm": fk.get("residual_rms_mm"),
            "robot_ip": rep.get("robot_ip", ""),
            "text": f"FR3 pivot {Path(pivot).name} (socket centre in {frame} {fk['s_ee_mm']} mm, sigma {sigma})",
        }
    if tcp_z_m is not None:
        tr.update({"x": 0.0, "y": 0.0, "z": round(float(tcp_z_m), 5)})
        return tcp, {"kind": "manual", "text": f"manual z {tcp_z_m} m"}
    if keep_template:
        return tcp, {"kind": "template", "text": "the template's (171.99 mm = the old fingertips)"}
    tr.update({"x": 0.0, "y": 0.0, "z": NEW_TIP_TCP_Z_M})
    return tcp, {
        "kind": "estimate",
        "text": "44 mm tips, estimated: handheld rig pivot 167.91 + flange->sensor 22.25 mm; "
        "not measured on the arm (up to ~12 mm), an FR3 pivot measures it",
    }


def desk_config(args) -> int:
    """Franka Desk end-effector configs, one per tool configuration: identified mass + CoM, CAD inertia
    scaled to that mass.

    The load is fitted *without* joint offsets: the robot subtracts none before its gravity compensation and
    its external wrench estimate (O_F_ext_hat_K, reproduced by ``franka_external_wrench``), so this is the
    load that leaves the least torque unexplained there. Fits with offsets (global, per reach) are printed
    alongside; the spread of the three is the uncertainty of the *physical* load, not a Desk error. Runs of
    the same tool configuration are pooled.
    """
    names = _desk_names(args)
    by_config: dict[str, list] = {}
    for path in args.runs:
        path = Path(path)
        run, qs, tau, g_arm = _read_run(path)
        if run.get("source") == "fr3" and "tau_J_up" not in run["poses"][0]:
            raise SystemExit(
                f"{path}: one measurement per pose, friction hysteresis is in it (10-12); collect again"
            )
        if not run.get("complete", True):
            raise SystemExit(f"{path}: incomplete run")
        by_config.setdefault(run_tool_config(run), []).append((path, run, qs, tau, g_arm))
    missing = sorted(set(by_config) - set(names))
    if missing:
        raise SystemExit(f"no Desk name for {missing}: pass --name CONFIG=NAME")
    template = json.loads(Path(args.template).read_text())
    tcp, tcp_source = desk_tcp(
        template.get("tcp") or {},
        tcp_z_m=getattr(args, "tcp_z_m", None),
        pivot=getattr(args, "tcp_from_pivot", None),
        keep_template=getattr(args, "tcp_template", False),
        urdf=args.urdf,
    )
    print(f"TCP: {tcp['translation']} m, {tcp_source['text']}")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    masses = {}
    report: dict = {"template": str(args.template), "tcp_source": tcp_source, "configs": {}}
    for config in [c for c in TOOL_CONFIGS if c in by_config]:
        runs = by_config[config]
        gravity = runs[0][1]["gravity"]
        qs, tau, g_arm = (np.concatenate([r[i] for r in runs]) for i in (2, 3, 4))
        est = fit_load(qs, tau, g_arm, gravity, with_bias=False)
        ref = {
            "with offsets, global": fit_load(qs, tau, g_arm, gravity),
            "with offsets, per reach": fit_load(qs, tau, g_arm, gravity, groups=reach_groups(qs)),
        }
        m_cad, c_cad, inertia_cad = urdf_subtree_inertial(args.urdf, exclude=TOOL_CONFIGS[config])
        inertia = inertia_cad * (est.mass_kg / m_cad)  # shape from CAD; statics cannot see inertia
        cfg = dict(template)
        cfg["id"] = str(uuid.uuid4())  # a new Desk entry, not an overwrite of the template's
        cfg["name"] = names[config]
        cfg["tcp"] = tcp
        cfg["inertial"] = {
            "mass": round(est.mass_kg, 4),
            "centerOfMass": dict(zip("xyz", (round(float(v), 5) for v in est.com_flange_m), strict=True)),
            "inertia": {
                f"x{i + 1}{j + 1}": float(f"{inertia[i, j]:.6g}") for i in range(3) for j in range(i, 3)
            },
        }
        out = args.out_dir / f"{args.stem}_{_DESK_SUFFIX[config]}.json"
        out.write_text(json.dumps(cfg, ensure_ascii=False, separators=(",", ":")))
        masses[config] = est.mass_kg
        print(f"{cfg['name']} ({TOOL_CONFIG_LABELS[config]}) <- {', '.join(r[0].name for r in runs)}")
        print(
            f"  Desk load (no joint offsets): {_fmt(est.mass_kg, est.com_flange_m)}  "
            f"(fit noise +-{est.mass_std_kg * 1000:.1f} g)"
        )
        w = _wrench_with_load(qs, tau, g_arm, est.mass_kg, est.com_flange_m, gravity)
        print(f"    robot's O_F_ext_hat_K with it (predicted): {_fmt_force(w)}")
        entry = {
            "label": TOOL_CONFIG_LABELS[config],
            "desk_name": cfg["name"],
            "runs": [r[0].name for r in runs],
            "n_poses": int(len(qs)),
            "mass_kg": round(est.mass_kg, 4),
            "com_flange_m": [round(float(v), 5) for v in est.com_flange_m],
            "mass_std_g": round(est.mass_std_kg * 1000, 1),
            "residual_rms_nm": est.as_dict()["residual_rms_nm"],
            "predicted_force_mean_n": np.round(w[:, :3].mean(0), 3).tolist(),
            "predicted_force_max_n": round(float(np.linalg.norm(w[:, :3], axis=1).max()), 3),
            "desk_at_start": [],
            "references": {},
            "cad": {"mass_kg": round(m_cad, 4), "com_flange_m": [round(float(v), 5) for v in c_cad]},
            "desk_json": str(out),
            "tcp": cfg.get("tcp"),
            "device_id": cfg.get("deviceId"),
        }
        for path, run, q1, t1, g1 in runs:
            loaded = run.get("desk_load_at_start")
            if loaded and "O_F_ext_hat_K" in run["poses"][0]:
                # what the robot reported with the load configured then, and our reproduction of it
                w = np.array([p["O_F_ext_hat_K"] for p in run["poses"]])
                e = w - _wrench_with_load(q1, t1, g1, loaded["m_ee"], loaded["F_x_Cee"], gravity)
                print(
                    f"    {path.name}, Desk {loaded['m_ee']:.4f} kg: measured {_fmt_force(w)}; "
                    f"reproduced to rms {np.sqrt((e**2).mean()):.3f}"
                )
                entry["desk_at_start"].append(
                    {
                        "run": path.name,
                        "m_ee": loaded["m_ee"],
                        "measured_force_mean_n": np.round(w[:, :3].mean(0), 3).tolist(),
                        "measured_force_max_n": round(float(np.linalg.norm(w[:, :3], axis=1).max()), 3),
                        "reproduction_rms": round(float(np.sqrt((e**2).mean())), 4),
                    }
                )
        for name, r in ref.items():
            print(f"  {name}: {_fmt(r.mass_kg, r.com_flange_m)}")
            entry["references"][name] = {
                "mass_kg": round(r.mass_kg, 4),
                "com_flange_m": [round(float(v), 5) for v in r.com_flange_m],
            }
        print(f"  CAD: {_fmt(m_cad, c_cad)}")
        print(f"  tcp {cfg.get('tcp')} ({tcp_source['kind']}), deviceId from the template: {cfg.get('deviceId')}")
        print(f"  written: {out}")
        report["configs"][config] = entry
    if "gripper_cam_smr" in masses and "gripper_cam" in masses:
        print(f"difference between the two loads: {(masses['gripper_cam_smr'] - masses['gripper_cam']) * 1000:.1f} g")
    if getattr(args, "report", None):
        Path(args.report).write_text(json.dumps(report, indent=1, ensure_ascii=False))
        print(f"report written: {args.report}")
    return 0


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("check-poses", help="MuJoCo clearance check of the pose sequence")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--spacing-mm", type=float, default=3.0, help="surface sampling of the meshes")
    p.add_argument(
        "--approach-rad", type=float, default=DEFAULT_APPROACH_RAD, help="0 = one-way (old) sequence"
    )
    p.add_argument("--min-clearance-mm", type=float, default=15.0)
    p.add_argument("--poses", choices=POSE_SETS, default="payload")
    p.add_argument("--table-edge-m", type=float, help="table edge, x in the base frame (in front of the base)")
    p.add_argument("--table-top-m", type=float, help="table top, z above the base plane")
    p.add_argument("--table-margin-mm", type=float, default=50.0)
    p.set_defaults(func=check_poses)

    p = sub.add_parser("sim", help="synthetic run from MuJoCo")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--without-smr", action="store_true", help="SMR taken out of the nest")
    p.add_argument("--tool-config", choices=sorted(TOOL_CONFIGS), help="overrides --without-smr")
    p.add_argument("--poses", choices=POSE_SETS, default="payload")
    p.add_argument(
        "--noise-nm", type=float, default=0.0, help="noise of each pose's averaged tau_J (Nm, 1 sigma)"
    )
    p.add_argument(
        "--bias-nm", type=float, default=0.0, help="constant per-joint sensor offsets (Nm, 1 sigma)"
    )
    p.add_argument(
        "--arm-model-error",
        type=float,
        default=0.0,
        help="relative 1-sigma error of the bare-arm link masses in the gravity model",
    )
    p.add_argument(
        "--hysteresis-nm",
        type=float,
        default=0.0,
        help="static-friction hysteresis scale (Nm; 10-12 real: 0.1-0.4)",
    )
    p.add_argument(
        "--approach-rad", type=float, default=DEFAULT_APPROACH_RAD, help="0 = one-way (old) sequence"
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", type=Path)
    p.set_defaults(func=simulate)

    p = sub.add_parser("collect", help="real robot run (moves the arm)")
    fr3_arms.add_arguments(p)
    p.add_argument("--smr-out", dest="smr_present", action="store_false", help="SMR taken out of the nest")
    p.add_argument(
        "--tool-config", choices=sorted(TOOL_CONFIGS), help="what is mounted (overrides --smr-out); see above"
    )
    p.add_argument("--poses", choices=POSE_SETS, default="payload")
    p.add_argument(
        "--six-d-device-id",
        help="also log this BOX's 6D force + IMU during every hold (<run>.six_d.log, for box_six_d fit); "
        "the recorder must be disconnected",
    )
    p.add_argument("--speed-factor", type=float, default=0.1)
    p.add_argument("--settle-s", type=float, default=1.5)
    p.add_argument("--hold-s", type=float, default=2.0)
    p.add_argument("--rate-hz", type=float, default=200.0)
    p.add_argument(
        "--approach-rad",
        type=float,
        default=DEFAULT_APPROACH_RAD,
        help="measure every pose reached from below and from above by this much (cancels joint friction "
        "hysteresis); 0 = once per pose, as on 10-12 -- not recommended",
    )
    p.add_argument(
        "--max-retries",
        type=int,
        default=3,
        help="retries of a move aborted by communication_constraints_violation",
    )
    p.add_argument("--note", default="")
    p.add_argument("--yes", action="store_true", help="skip the interactive confirmation")
    p.add_argument("--out", type=Path)
    p.set_defaults(func=collect)

    p = sub.add_parser("move-waypoint", help="real robot: one move of the arm-held 6D zero route")
    fr3_arms.add_arguments(p)
    p.add_argument("--to", choices=sorted(six_d_zero_waypoints()), required=True)
    p.add_argument("--speed-factor", type=float, default=0.1)
    p.add_argument("--max-retries", type=int, default=3)
    p.set_defaults(func=move_waypoint)

    p = sub.add_parser("fit", help="total load from one run")
    p.add_argument("run", type=Path)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument("--subtract", help="comma-separated URDF links to subtract from the total load")
    p.add_argument(
        "--subtract-weighed",
        action="store_true",
        help="subtract the parts weighed on a scale (adaptor, and the SMR if the run had it): "
        "what is left is the gripper + Part 1 + anything the URDF does not model",
    )
    p.add_argument(
        "--no-bias",
        action="store_true",
        help="do not estimate per-joint sensor offsets (the load Desk needs)",
    )
    p.add_argument("--json", type=Path, help="also write the report here")
    p.set_defaults(func=fit)

    p = sub.add_parser("desk-config", help="Franka Desk end-effector JSONs, one per tool configuration")
    p.add_argument("runs", type=Path, nargs="+", help="runs; runs of one tool configuration are pooled")
    p.add_argument(
        "--template", type=Path, default=DEFAULT_DESK_TEMPLATE, help="an exported Desk end-effector JSON"
    )
    tcp = p.add_mutually_exclusive_group()
    tcp.add_argument(
        "--tcp-from-pivot",
        type=Path,
        help="fr3_tcp_pivot.py fit / fit-joint report: the TCP is its socket centre, measured on the arm",
    )
    tcp.add_argument("--tcp-z-m", type=float, help="TCP z (flange frame, x = y = 0)")
    tcp.add_argument("--tcp-template", action="store_true", help="the template's TCP (171.99 mm, the old tips)")
    p.add_argument("--name", action="append", help="CONFIG=NAME, the Desk name for that tool configuration")
    p.add_argument("--name-smr", help="= --name gripper_cam_smr=NAME")
    p.add_argument("--name-no-smr", help="= --name gripper_cam=NAME")
    p.add_argument("--report", type=Path, help="also write a JSON summary here")
    p.add_argument("--out-dir", type=Path, required=True)
    p.add_argument("--stem", default="fr3_corenetic_desk_load")
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.set_defaults(func=desk_config)

    p = sub.add_parser("diff", help="load difference between two runs over the same poses")
    p.add_argument("run_a", type=Path)
    p.add_argument("run_b", type=Path)
    p.add_argument("--urdf", type=Path, default=DEFAULT_URDF)
    p.add_argument(
        "--expect-link", default=SMR_BODY, help="URDF link the difference should equal ('' = none)"
    )
    p.add_argument("--max-q-mismatch", type=float, default=0.01)
    p.set_defaults(func=diff)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if hasattr(args, "robot_ip"):  # drives an arm: which one is never a default (roadmap B2)
        try:
            fr3_arms.resolve_args(args)
        except fr3_arms.ArmConfigError as exc:
            print(f"ERROR: {exc}")
            return 2
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
