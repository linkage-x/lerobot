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
  check-poses  MuJoCo: clearance between the tool and the arm along the pose sequence, lowest tool point
  sim          MuJoCo: synthetic run (optional sensor noise/offsets, bare-arm model error, SMR removed)
  collect      real robot: visit the poses, average tau_J at rest, store with libfranka's bare-arm gravity
  fit          one run -> total load with sensor offsets (physical estimate); optionally minus weighed parts
  diff         two runs over the same poses -> the part that differs (e.g. SMR in vs out)
  desk-config  runs (SMR in/out) -> Franka Desk end-effector JSONs (fit without offsets, see below)

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
    default_poses,
    fit_difference,
    fit_load,
    franka_external_wrench,
    load_regressor,
    reach_groups,
    subtract_known,
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
SMR_BODY = "link_corenetic_smr"
# Parts weighed on a scale (URDF links); what is left of a fit is the gripper + Part 1 + anything unmodelled.
WEIGHED_LINKS = ("link_corenetic_smr_cam", SMR_BODY)
DEFAULT_APPROACH_RAD = 0.03


def _sequence(approach_rad: float):
    return visit_sequence(
        default_poses(), approach_rad, PANDA_PY_JOINT_LIMITS_LOWER, PANDA_PY_JOINT_LIMITS_UPPER
    )


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

    @staticmethod
    def _key(idx: np.ndarray) -> np.ndarray:
        idx = idx + 2**20
        return (idx[..., 0] << 42) | (idx[..., 1] << 21) | idx[..., 2]

    def min_distance(self, query: np.ndarray) -> float:
        cells = np.floor(query / self.cell).astype(np.int64)[:, None, :] + self._OFFSETS[None]
        keys = self._key(cells).reshape(-1)
        lo = np.searchsorted(self.keys, keys, "left")
        hi = np.searchsorted(self.keys, keys, "right")
        hit = hi > lo
        if not hit.any():
            return np.inf
        q = np.repeat(query, len(self._OFFSETS), axis=0)[hit]
        best = np.inf
        for qi, a, b in zip(q, lo[hit], hi[hit], strict=True):
            best = min(best, float(((self.points[a:b] - qi) ** 2).sum(1).min()))
        return min(float(np.sqrt(best)), np.inf)


def check_poses(args) -> int:
    mujoco, model, data = _load_mujoco(args.xml)
    tool_bodies = set(_subtree(mujoco, model, "base_link"))
    # fr3_link7 carries the flange the tool is bolted to -> rigid with the tool, left out
    arm_bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"fr3_link{i}") for i in range(7)}
    tool_pts = _surface_points(mujoco, model, tool_bodies, args.spacing_mm / 1000)
    arm_fields = {
        b: _NearField(p)
        for b, p in _surface_points(mujoco, model, arm_bodies, args.spacing_mm / 1000).items()
    }
    poses = default_poses()
    targets = [np.asarray(PANDA_PY_JOINT_POSITION_START)] + [t for t, _, _ in _sequence(args.approach_rad)]
    owner = [0] + [i for _, _, i in _sequence(args.approach_rad)]
    path = []
    for a, b, i in zip(targets[:-1], targets[1:], owner[1:], strict=True):
        path.extend((i, a + (b - a) * s) for s in np.linspace(0, 1, 17)[1:])
    per_pose = np.full(len(poses), np.inf)
    worst = (np.inf, None, None)
    lowest = np.inf
    for t, q in path:
        _set_q(mujoco, model, data, q)
        tool = _world(data, tool_pts)
        lowest = min(lowest, float(tool[:, 2].min()))
        for b, field in arm_fields.items():
            # tool points into the arm body's frame, where its cloud is hashed once
            dist = field.min_distance((tool - data.xpos[b]) @ data.xmat[b].reshape(3, 3))
            per_pose[t] = min(per_pose[t], dist)
            if dist < worst[0]:
                worst = (dist, t, mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b))
    print(
        f"poses={len(poses)} path samples={len(path)} surface spacing {args.spacing_mm} mm "
        f"({sum(len(p) for p in tool_pts.values())} tool points)"
    )
    for i, q in enumerate(poses):
        print(
            f"  pose {i:2d} q = {np.round(q, 3).tolist()}  min clearance getting there {per_pose[i] * 1000:5.1f} mm"
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
    print(
        f"check-poses={'PASS' if ok else 'FAIL'} (need >= {args.min_clearance_mm:.0f} mm incl. the point "
        "spacing; the table, laser tracker and cables are NOT modelled -- check those by eye)"
    )
    return 0 if ok else 1


def simulate(args) -> int:
    mujoco, model, data = _load_mujoco(args.xml)
    rng = np.random.default_rng(args.seed)
    tool = _subtree(mujoco, model, "base_link")
    if args.without_smr:
        for b in _subtree(mujoco, model, SMR_BODY):
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
    for target, tag, i in _sequence(args.approach_rad):
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
        "smr_present": not args.without_smr,
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

    poses = default_poses()
    print(
        f"collect: {len(poses)} static poses x {'2 (from below and above)' if args.approach_rad > 0 else '1'}, "
        f"speed_factor={args.speed_factor}, hold {args.hold_s}s after {args.settle_s}s settle; "
        f"SMR {'IN' if args.smr_present else 'OUT'} the nest"
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
        "smr_present": args.smr_present,
        "note": args.note,
        "desk_load_at_start": desk_load,
        "gravity": list(GRAVITY_EARTH),
        "complete": False,
        "poses": [],
    }
    run["approach_rad"] = args.approach_rad
    out = _write_run(run, args.out)  # rewritten after every measurement: an abort keeps what was measured
    pending: dict | None = None
    try:
        for target, tag, i in _sequence(args.approach_rad):
            if not _move(panda, robot, target, args):
                print(f"  stopped on the way to pose {i}; {len(run['poses'])} complete poses are in {out}")
                return 1
            if tag is None:  # approach point: only sets the direction every joint arrives from
                continue
            m = _measure(robot, model, args)
            if pending is None or pending["index"] != i:
                pending = {"index": i, "q_target": target.tolist(), "measurements": {}}
            pending["measurements"][tag] = m
            print(
                f"  pose {i:2d} {tag:6s}: n={m['n']} unique={m['n_unique']} |dq|max={m['dq_abs_max']:.4f} "
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
        _move(panda, robot, default_poses()[0], args)
        run["complete"] = True
        _write_run(run, out)
    finally:
        with contextlib.suppress(Exception):
            panda.stop_controller()
    return 0


def _measure(robot, model, args) -> dict:
    time.sleep(args.settle_s)
    qs, taus, dqs, ext = [], [], [], []
    t_end = time.monotonic() + args.hold_s
    while time.monotonic() < t_end:
        s = robot.read_once()
        qs.append(s.q)
        taus.append(s.tau_J)
        dqs.append(s.dq)
        ext.append(s.O_F_ext_hat_K)
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


# ---------------------------------------------------------------- fitting
def _write_run(run: dict, out: Path | None, quiet: bool = False) -> Path:
    run.setdefault("created", dt.datetime.now().isoformat(timespec="seconds"))
    if out is None:
        stamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        out = (
            REPO
            / "outputs/fr3_payload"
            / f"{stamp}_{run['source']}_smr_{'in' if run['smr_present'] else 'out'}.json"
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
        print(f"WARNING: incomplete run ({len(qs)} of {len(default_poses())} poses)")
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
        f"[{use}] end-effector load (SMR {'in' if run.get('smr_present', True) else 'out'}): "
        f"mass {est.mass_kg:.3f} kg, CoM {np.round(est.com_flange_m, 4).tolist()} m in the flange frame "
        "(inertia is not identifiable from statics)"
    )
    if not args.no_bias:
        print("  not the load for Desk (the robot subtracts no offsets): use desk-config, or fit --no-bias")
    if args.subtract_weighed:
        links = [k for k in WEIGHED_LINKS if k != SMR_BODY or run.get("smr_present", True)]
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


def desk_config(args) -> int:
    """Franka Desk end-effector configs, one per SMR state: identified mass + CoM, CAD inertia scaled to that mass.

    The load is fitted *without* joint offsets: the robot subtracts none before its gravity compensation and
    its external wrench estimate (O_F_ext_hat_K, reproduced by ``franka_external_wrench``), so this is the
    load that leaves the least torque unexplained there. Fits with offsets (global, per reach) are printed
    alongside; the spread of the three is the uncertainty of the *physical* load, not a Desk error. Runs of
    the same SMR state are pooled.
    """
    by_state: dict[bool, list] = {}
    for path in args.runs:
        run, qs, tau, g_arm = _read_run(path)
        if run.get("source") == "fr3" and "tau_J_up" not in run["poses"][0]:
            raise SystemExit(
                f"{path}: one measurement per pose, friction hysteresis is in it (10-12); collect again"
            )
        if not run.get("complete", True):
            raise SystemExit(f"{path}: incomplete run")
        by_state.setdefault(run.get("smr_present", True), []).append((path, run, qs, tau, g_arm))
    template = json.loads(Path(args.template).read_text())
    args.out_dir.mkdir(parents=True, exist_ok=True)
    masses = {}
    for smr, runs in sorted(by_state.items(), reverse=True):
        gravity = runs[0][1]["gravity"]
        qs, tau, g_arm = (np.concatenate([r[i] for r in runs]) for i in (2, 3, 4))
        est = fit_load(qs, tau, g_arm, gravity, with_bias=False)
        ref = {
            "with offsets, global": fit_load(qs, tau, g_arm, gravity),
            "with offsets, per reach": fit_load(qs, tau, g_arm, gravity, groups=reach_groups(qs)),
        }
        m_cad, c_cad, inertia_cad = urdf_subtree_inertial(args.urdf, exclude=() if smr else (SMR_BODY,))
        inertia = inertia_cad * (est.mass_kg / m_cad)  # shape from CAD; statics cannot see inertia
        cfg = dict(template)
        cfg["id"] = str(uuid.uuid4())  # a new Desk entry, not an overwrite of the template's
        cfg["name"] = args.name_smr if smr else args.name_no_smr
        cfg["inertial"] = {
            "mass": round(est.mass_kg, 4),
            "centerOfMass": dict(zip("xyz", (round(float(v), 5) for v in est.com_flange_m), strict=True)),
            "inertia": {
                f"x{i + 1}{j + 1}": float(f"{inertia[i, j]:.6g}") for i in range(3) for j in range(i, 3)
            },
        }
        out = args.out_dir / f"{args.stem}_{'smr' if smr else 'no_smr'}.json"
        out.write_text(json.dumps(cfg, ensure_ascii=False, separators=(",", ":")))
        masses[smr] = est.mass_kg
        print(f"{cfg['name']} <- {', '.join(r[0].name for r in runs)}")
        print(
            f"  Desk load (no joint offsets): {_fmt(est.mass_kg, est.com_flange_m)}  "
            f"(fit noise +-{est.mass_std_kg * 1000:.1f} g)"
        )
        w = _wrench_with_load(qs, tau, g_arm, est.mass_kg, est.com_flange_m, gravity)
        print(f"    robot's O_F_ext_hat_K with it (predicted): {_fmt_force(w)}")
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
        for name, r in ref.items():
            print(f"  {name}: {_fmt(r.mass_kg, r.com_flange_m)}")
        print(f"  CAD: {_fmt(m_cad, c_cad)}")
        print(f"  tcp / deviceId from the template: {cfg.get('tcp')} / {cfg.get('deviceId')}")
        print(f"  written: {out}")
    if len(masses) == 2:
        print(f"difference between the two loads: {(masses[True] - masses[False]) * 1000:.1f} g")
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
    p.set_defaults(func=check_poses)

    p = sub.add_parser("sim", help="synthetic run from MuJoCo")
    p.add_argument("--xml", type=Path, default=DEFAULT_XML)
    p.add_argument("--without-smr", action="store_true", help="SMR taken out of the nest")
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

    p = sub.add_parser("desk-config", help="Franka Desk end-effector JSONs, one per SMR state")
    p.add_argument("runs", type=Path, nargs="+", help="runs, SMR in and/or out; runs of one state are pooled")
    p.add_argument("--template", type=Path, required=True, help="an exported Desk end-effector JSON")
    p.add_argument("--name-smr", required=True, help="Desk name for the run with the SMR in its nest")
    p.add_argument("--name-no-smr", required=True, help="Desk name for the run without the SMR")
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
