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
Monte02 for its left arm: the session posture, the arm's chain, the MuJoCo model and the capsule model of the
arm against the rest of the robot (roadmap docs/monte02_left_arm_roadmap.html, M1.4).

Only the left arm moves in a session. The trunk (knee, hip, waist) is locked (D4: anti-tip; set only while
parked, braked, arms stowed), and so are the head and the right arm (parked: hanging). A
``Posture`` holds those twelve values; everything here is built for one posture:

* ``left_chain(posture)``: the 7-joint ``urdf_kinematics`` chain from the base to the TCP;
* ``mujoco_model(posture)``: the MJCF (``build_mjcf.py`` in the description) with those joints folded into
  the bodies' frames, as the chain folds them; its fingerprint binds a MuJoCo / IK QC approval to the model
  *and* the posture;
* ``CapsuleModel``: capsules round every part (``fit``, run once with MuJoCo, enclose every vertex of the part's
  meshes, the gripper at every opening), hung on URDF link frames, so ``clearance`` needs numpy only. A capsule
  gap is a lower bound on the mesh gap. Checked: the left arm against the base, trunk, chest, head and the
  parked right arm, and its hand against its own upper arm.

The description (URDF, MJCF, meshes, the fitted capsules) is in the private ``opencv_kalibr`` submodule.
Nothing here talks to a robot.
"""

# ruff: noqa: N803, N806  (matrix names: R, T follow the maths)
from __future__ import annotations

import functools
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import dual_fr3_collision as fr3col, urdf_kinematics as uk  # noqa: E402

DESCRIPTION = REPO / "third_party/opencv_kalibr/monte02_h_v1_description"
URDF = DESCRIPTION / "urdf/monte02_h_v1_gmsl2cam.urdf"
MJCF = DESCRIPTION / "mjcf/monte02_h_v1_gmsl2cam.xml"
CAPSULE_FILE = DESCRIPTION / "mjcf/monte02_capsules.json"
SCHEMA = "monte02_capsules/v1"

BASE_LINK = "link_b0_base"
TCP_LINK = "lt_corenetic_socket_ee"
_ARM = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow_pitch", "wrist_yaw", "wrist_pitch", "wrist_roll")
LEFT_ARM_JOINTS = tuple(f"joint_l{i}_{n}" for i, n in enumerate(_ARM, 1))
RIGHT_ARM_JOINTS = tuple(f"joint_r{i}_{n}" for i, n in enumerate(_ARM, 1))
TRUNK_JOINTS = ("joint_t1_knee_pitch", "joint_t2_hip_pitch", "joint_t3_waist_yaw")
HEAD_JOINTS = ("joint_h1_pitch", "joint_h2_yaw")
#: The right arm parked: hanging by the side, the elbow just inside its lower limit (0.087 rad; the vendor's
#: zero pose has it at 0, outside).
RIGHT_ARM_PARKED = (0.0, 0.0, 0.0, 0.1, 0.0, 0.0, 0.0)


@dataclass(frozen=True)
class Posture:
    """What is locked for a session. Default: upright, head straight, right arm parked."""

    trunk: tuple[float, float, float] = (0.0, 0.0, 0.0)
    head: tuple[float, float] = (0.0, 0.0)
    right_arm: tuple[float, ...] = RIGHT_ARM_PARKED

    def __post_init__(self) -> None:
        for name, values, n in (("trunk", self.trunk, 3), ("head", self.head, 2), ("right_arm", self.right_arm, 7)):
            if len(values) != n:
                raise ValueError(f"posture.{name}: {n} values, got {len(values)}")
            object.__setattr__(self, name, tuple(float(v) for v in values))

    def locked(self) -> dict[str, float]:
        return dict(
            zip(TRUNK_JOINTS + HEAD_JOINTS + RIGHT_ARM_JOINTS, self.trunk + self.head + self.right_arm, strict=True)
        )


@functools.lru_cache(maxsize=1)
def _urdf_root() -> ET.Element:
    return ET.parse(URDF).getroot()


def _chain_to(link: str, posture: Posture) -> uk.Chain:
    """``base -> link`` with every posture joint on the way locked; its free joints are a prefix of the left arm."""
    on_path = {j.get("name") for j in uk._path(_urdf_root(), BASE_LINK, link)}
    chain = uk.load_chain(URDF, BASE_LINK, link, locked={k: v for k, v in posture.locked().items() if k in on_path})
    if chain.joint_names != LEFT_ARM_JOINTS[: chain.dof]:
        raise ValueError(f"{link}: free joints {chain.joint_names} are not a prefix of the left arm")
    return chain


@functools.lru_cache(maxsize=16)
def left_chain(posture: Posture = Posture(), tip_link: str = TCP_LINK) -> uk.Chain:
    chain = _chain_to(tip_link, posture)
    if chain.dof != 7:
        raise ValueError(f"{tip_link} is not on the left arm's end")
    return chain


# ------------------------------------------------------------------ MuJoCo ---


def model_fingerprint(posture: Posture = Posture()) -> str:
    """The MJCF, its meshes and the posture: what a MuJoCo / IK QC approval is for."""
    files = fr3col.model_fingerprint(MJCF)["sha256"]
    locked = json.dumps(sorted(posture.locked().items()))
    return hashlib.sha256(f"{files}|{locked}".encode()).hexdigest()


def mujoco_model(posture: Posture = Posture()) -> Any:
    """The MJCF with the posture's joints folded into their bodies (deleted, the body turned by the value):
    the left arm, the grippers and the wheels stay joints. ``mujoco.MjModel``."""
    import mujoco

    spec = mujoco.MjSpec.from_file(str(MJCF))
    for name, value in posture.locked().items():
        joint = spec.joint(name)
        if joint.type != mujoco.mjtJoint.mjJNT_HINGE or np.linalg.norm(joint.pos) > 1e-12:
            raise ValueError(f"{name}: only a hinge at its body's origin can be folded")
        lo, hi = joint.range
        if not lo - 1e-9 <= value <= hi + 1e-9:
            raise ValueError(f"{name} = {value} is outside its limits [{lo}, {hi}]")
        axis = np.asarray(joint.axis, dtype=np.float64)
        axis /= np.linalg.norm(axis)
        turn = np.concatenate([[math.cos(value / 2)], math.sin(value / 2) * axis])
        quat = np.zeros(4)
        mujoco.mju_mulQuat(quat, np.asarray(joint.parent.quat, dtype=np.float64), turn)
        joint.parent.quat = quat
        spec.delete(joint)
    return spec.compile()


# ------------------------------------------------------------------ capsules ---

_HAND = ("gripper", "fingers", "camera")
_LINKAGE = ("knuckle_left_outer", "knuckle_left_inner", "knuckle_right_outer", "knuckle_right_inner",
            "finger_left", "finger_right", "tip_left", "tip_right")
#: part -> (the link it hangs on, how many capsules, the bodies whose meshes it holds). A part is rigid with
#: its link, except ``*_fingers`` (the gripper linkage), fitted over its whole opening. Two capsules where one
#: would swallow a corner. The chest is a box (200 x 320 x 510 mm): one capsule round it is 165 mm thick
#: and bulges 65 mm in front, so it is cut in ``SLABS`` of equal height, one sideways capsule each (~100 mm).
PARTS: dict[str, tuple[str, int, tuple[str, ...]]] = {
    "trunk_base": ("link_t0_trunk_base", 2, ("link_t0_trunk_base",)),
    "leg": ("link_t1_leg", 2, ("link_t1_leg",)),
    "hip": ("link_t2_hip", 1, ("link_t2_hip",)),
    "chest": ("link_t3_chest", 4, ("link_t3_chest", "link_h0_head_base")),
    "neck": ("link_h1_neck", 1, ("link_h1_neck",)),
    "head": ("link_h2_head", 1, ("link_h2_head", "link_h_sensor_rgbd")),
}
for _s in "lr":
    for _i, _n in enumerate(_ARM, 1):
        PARTS[f"{_s}{_i}"] = (f"link_{_s}{_i}_{_n}", 1 if _i in (1, 7) else 2, (f"link_{_s}{_i}_{_n}",))
    PARTS[f"{_s}_gripper"] = (f"link_{_s}_sensor_ft", 2, (f"link_{_s}_sensor_ft", f"link_{_s}t_gripper_base"))
    PARTS[f"{_s}_fingers"] = (f"link_{_s}_sensor_ft", 1, tuple(f"link_{_s}t_gripper_{n}" for n in _LINKAGE))
PARTS["l_camera"] = ("link_l_sensor_ft", 1, ("link_lt_corenetic_mount", "link_lt_corenetic_smr_cam"))
PARTS["r_camera"] = ("link_r_sensor_ft", 1, ("link_rt_sensor_rgbd",))
#: The chassis (wheeled base and the trunk base's foot) is a plate 600 x 590 mm, 150 mm high: a capsule round
#: it is a 330 mm ball. It is the half-space below ``CHASSIS_TOP`` (base frame, z up, the floor at -0.085)
#: instead -- the left arm is kept above that everywhere, not only over the plate (it has no work down there)
#: -- and the trunk base's capsules hold only what stands above it (the column up to the knee).
CHASSIS_TOP = 0.20
CHASSIS = {"chassis": ("link_b0_base", 0, ("link_b0_base", "link_b_wheel_drive_left", "link_b_wheel_drive_right",
                                           "link_b_wheel_idler_left", "link_b_wheel_idler_right",
                                           "link_b_sensor_lidar_front", "link_b_sensor_lidar_rear",
                                           "link_b_sensor_ultrasonic", "link_t0_trunk_base"))}
#: Gripper openings the linkage is fitted at (fractions of the drive's range).
OPENINGS = np.linspace(0.0, 1.0, 9)
#: Joint index of each left part along the arm (0: not on it); hand parts ride on joint 7.
LEFT_INDEX = {f"l{i}": i for i in range(1, 8)} | {f"l_{h}": 7 for h in _HAND}
#: Left parts are checked against these, and the hand against the upper arm (``self_pair_gap`` joints apart).
OTHERS = ("chassis", "trunk_base", "leg", "hip", "chest", "neck", "head", *(p for p in PARTS if p.startswith("r")))
SELF_PAIR_GAP = 4
#: Pairs in contact by construction: the shoulder is bolted to the chest.
ADJACENT = (("chest", "l1"),)
#: part -> the axis of its link frame it is cut along into ``count`` slabs of equal height.
SLABS = {"chest": 2}
#: Pairs whose relative pose depends only on the first n left joints, so the meshes can be checked over a grid
#: of those joints' whole ranges (MuJoCo, convex hulls); if they never touch, the pair is excluded with the
#: smallest gap found. The upper arm runs a few mm along the chest's side (the shoulder roll's -0.157 limit
#: keeps it off): as capsules they would always look in contact.
GRID_PAIRS = {("chest", "l2"): (2, (121, 81)), ("chest", "l3"): (3, (41, 31, 41))}


def candidate_pairs() -> list[tuple[str, str]]:
    """Part pairs ``clearance`` checks, before the fitted exclusions."""
    pairs = [(o, left) for left in LEFT_INDEX for o in OTHERS]
    pairs += [
        (a, b)
        for a in LEFT_INDEX
        for b in LEFT_INDEX
        if LEFT_INDEX[b] - LEFT_INDEX[a] >= SELF_PAIR_GAP
    ]
    return [p for p in pairs if p not in ADJACENT]


def part_vertices(xml_path: Path | str = MJCF, parts: dict = PARTS) -> dict[str, np.ndarray]:
    """Every mesh vertex of each part, in its link's frame, all joints at 0 (the linkage at every opening)."""
    import mujoco
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    owner = {model.body(body).id: part for part, (_, _, bodies) in parts.items() for body in bodies}
    frame = {part: model.body(link).id for part, (link, _, _) in parts.items()}
    gripper = [j for j in range(model.njnt) if "_gripper_" in model.joint(j).name]
    out: dict[str, list[np.ndarray]] = {part: [] for part in parts}
    for k, opening in enumerate(OPENINGS):
        data.qpos[:] = 0.0
        for j in gripper:
            low, high = model.jnt_range[j]
            data.qpos[model.jnt_qposadr[j]] = low + opening * (high - low)
        mujoco.mj_kinematics(model, data)
        seen: set[tuple[int, int]] = set()
        for g in range(model.ngeom):
            body = int(model.geom_bodyid[g])
            part = owner.get(body)
            if part is None or model.geom_type[g] != mujoco.mjtGeom.mjGEOM_MESH:
                continue
            if k > 0 and not part.endswith("_fingers"):
                continue
            mesh = int(model.geom_dataid[g])
            if (mesh, body) in seen:  # a visual and a collision geom on one mesh
                continue
            seen.add((mesh, body))
            start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
            local = model.mesh_vert[start : start + count].astype(np.float64)
            world = local @ data.geom_xmat[g].reshape(3, 3).T + data.geom_xpos[g]
            R_f, p_f = data.xmat[frame[part]].reshape(3, 3), data.xpos[frame[part]]
            out[part].append((world - p_f) @ R_f)
    missing = [part for part, chunks in out.items() if not chunks]
    if missing:
        raise ValueError(f"parts without meshes: {missing}")
    return {part: np.concatenate(chunks) for part, chunks in out.items()}


@dataclass(frozen=True)
class CapsuleModel:
    names: tuple[str, ...]
    parts: tuple[str, ...]
    links: tuple[str, ...]  # the link each capsule hangs on
    a: np.ndarray  # (K,3) in its link's frame
    b: np.ndarray  # (K,3)
    r: np.ndarray  # (K,)
    pairs: np.ndarray  # (P,2) capsule indices checked
    floor: np.ndarray  # (F,) capsule indices kept above CHASSIS_TOP
    excluded: tuple[tuple[str, str, str], ...]  # part pairs left out by the fit, and why
    fingerprint: str

    def place(self, q_left: np.ndarray, posture: Posture = Posture()) -> tuple[np.ndarray, np.ndarray]:
        """Capsule end points (n,K,3), (n,K,3) in the base frame (``link_b0_base``)."""
        q = np.atleast_2d(np.asarray(q_left, dtype=np.float64))
        A = np.empty((len(q), len(self.names), 3))
        B = np.empty_like(A)
        for link in dict.fromkeys(self.links):
            chain = _chain_to(link, posture)
            T = chain.fk(q[:, : chain.dof])
            k = [i for i, name in enumerate(self.links) if name == link]
            A[:, k] = np.einsum("nij,kj->nki", T[:, :3, :3], self.a[k]) + T[:, None, :3, 3]
            B[:, k] = np.einsum("nij,kj->nki", T[:, :3, :3], self.b[k]) + T[:, None, :3, 3]
        return A, B

    def pair_gaps(self, q_left: np.ndarray, posture: Posture = Posture()) -> np.ndarray:
        """(n, P+F) surface gap of each checked capsule pair, then of each ``floor`` capsule over the chassis
        top, metres (negative: overlap)."""
        A, B = self.place(q_left, posture)
        i, j, f = self.pairs[:, 0], self.pairs[:, 1], self.floor
        pairs = fr3col.segment_distance(A[:, i], B[:, i], A[:, j], B[:, j]) - self.r[i] - self.r[j]
        floor = np.minimum(A[:, f, 2], B[:, f, 2]) - CHASSIS_TOP - self.r[f]
        return np.concatenate([pairs, floor], axis=1)

    def gap_names(self) -> list[tuple[str, str]]:
        """What each column of ``pair_gaps`` is between."""
        return [(self.names[i], self.names[j]) for i, j in self.pairs] + [("chassis", self.names[f]) for f in self.floor]

    def clearance(self, q_left: np.ndarray, posture: Posture = Posture()) -> dict[str, Any]:
        """Per pose: the smallest gap and the two capsules it is between."""
        gaps = self.pair_gaps(q_left, posture)
        k = gaps.argmin(axis=1)
        names = self.gap_names()
        return {"self_m": gaps[np.arange(len(gaps)), k], "self_pair": [names[i] for i in k]}


def _pair_indices(parts: tuple[str, ...], part_pairs: list[tuple[str, str]]) -> np.ndarray:
    index = {}
    for k, part in enumerate(parts):
        index.setdefault(part, []).append(k)
    rows = [(i, j) for a, b in part_pairs if a != "chassis" for i in index[a] for j in index[b]]
    return np.array(rows, dtype=np.int64).reshape(-1, 2)


def _model(data: dict[str, Any]) -> CapsuleModel:
    caps = data["capsules"]
    parts = tuple(c["part"] for c in caps)
    excluded = tuple((e["a"], e["b"], e["why"]) for e in data["excluded"])
    skip = {(a, b) for a, b, _ in excluded}
    floor_pairs = {p for p in candidate_pairs() if p[0] == "chassis" and p not in skip}
    return CapsuleModel(
        names=tuple(c["name"] for c in caps),
        parts=parts,
        links=tuple(c["link"] for c in caps),
        a=np.array([c["a"] for c in caps], dtype=np.float64),
        b=np.array([c["b"] for c in caps], dtype=np.float64),
        r=np.array([c["r"] for c in caps], dtype=np.float64),
        pairs=_pair_indices(parts, [p for p in candidate_pairs() if p not in skip]),
        floor=np.array([k for k, part in enumerate(parts) if ("chassis", part) in floor_pairs], dtype=np.int64),
        excluded=excluded,
        fingerprint=str(data["model"]["sha256"]),
    )


def load(path: Path | str = CAPSULE_FILE) -> CapsuleModel:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {data.get('schema')!r}, expected {SCHEMA!r}")
    return _model(data)


#: A candidate pair overlapping in at least this share of random arm poses is in contact by construction
#: (two capsules round one joint), not a collision the planner could avoid; it is excluded, with the share.
ALWAYS = 0.99
FIT_SAMPLES = 4000


def _fit_slabs(
    points: np.ndarray, count: int, axis: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    edges = np.linspace(points[:, axis].min(), points[:, axis].max(), count + 1)
    out = []
    for low, high in zip(edges[:-1], edges[1:], strict=True):
        part = points[(points[:, axis] >= low - 1e-9) & (points[:, axis] <= high + 1e-9)]
        out.append(fr3col._fit_one(part, fr3col._hull_points(part, rng, 800), rng))
    return out


def grid_gap(part_a: str, part_b: str, n_joints: int, steps: tuple[int, ...]) -> tuple[float, np.ndarray]:
    """Smallest convex-hull gap between two parts' meshes over a grid of the first ``n_joints`` left joints
    (whole ranges, ends included), refined round the smallest: the gap and where."""
    import itertools

    import mujoco

    model = mujoco_model(Posture())
    model.geom_margin[:] = 0.05  # report pairs up to 50 mm apart, with their distance
    data = mujoco.MjData(model)
    adr = [model.jnt_qposadr[model.joint(n).id] for n in LEFT_ARM_JOINTS[:n_joints]]
    ranges = [model.jnt_range[model.joint(n).id] for n in LEFT_ARM_JOINTS[:n_joints]]
    bodies = [{model.body(b).id for b in PARTS[p][2]} for p in (part_a, part_b)]

    def gap(q: tuple[float, ...]) -> float:
        data.qpos[:] = 0.0
        data.qpos[adr] = q
        mujoco.mj_kinematics(model, data)
        mujoco.mj_collision(model, data)
        best = 0.05
        for c in data.contact[: data.ncon]:
            b1, b2 = model.geom_bodyid[c.geom1], model.geom_bodyid[c.geom2]
            if (b1 in bodies[0] and b2 in bodies[1]) or (b2 in bodies[0] and b1 in bodies[1]):
                best = min(best, float(c.dist))
        return best

    grids = [np.linspace(lo, hi, n) for (lo, hi), n in zip(ranges, steps, strict=True)]
    best_gap, best_q = min((gap(q), q) for q in itertools.product(*grids))
    step = [g[1] - g[0] for g in grids]
    fine = [
        np.clip(np.linspace(v - s, v + s, 11), lo, hi) for v, s, (lo, hi) in zip(best_q, step, ranges, strict=True)
    ]
    best_gap, best_q = min([(best_gap, best_q)] + [(gap(q), q) for q in itertools.product(*fine)])
    return best_gap, np.asarray(best_q)


def _base_z(part: str, points: np.ndarray) -> np.ndarray:
    """Height in the base frame of points in a part's link frame (posture zero)."""
    T = _chain_to(PARTS.get(part, CHASSIS.get(part))[0], Posture()).fk(np.zeros((1, 0)))[0]
    return points @ T[2, :3] + T[2, 3]


def fit(xml_path: Path | str = MJCF, seed: int = 0) -> dict[str, Any]:
    """Capsules for every part and the pairs found always overlapping, as ``CAPSULE_FILE`` holds them."""
    rng = np.random.default_rng(seed)
    vertices = part_vertices(xml_path)
    above = _base_z("trunk_base", vertices["trunk_base"]) > CHASSIS_TOP
    vertices["trunk_base"] = vertices["trunk_base"][above]
    capsules = []
    for part, (link, count, _) in PARTS.items():
        fitted = _fit_slabs(vertices[part], count, SLABS[part], rng) if part in SLABS else fr3col._fit_part(
            vertices[part], count, rng
        )
        for k, (a, b, r) in enumerate(fitted):
            capsules.append({"name": f"{part}.{k}", "part": part, "link": link, "a": a.tolist(), "b": b.tolist(),
                             "r": float(r)})
    data = {
        "schema": SCHEMA,
        "model": fr3col.model_fingerprint(xml_path),
        "note": "Each part's capsules enclose every vertex of its meshes (the fingers at every opening), in the "
        "frame of the URDF link they hang on. 'excluded': candidate part pairs overlapping in nearly every "
        "random pose of the left arm (posture zero).",
        "capsules": capsules,
        "excluded": [{"a": a, "b": b, "why": "bolted together"} for a, b in ADJACENT],
    }
    for (a, b), (n_joints, steps) in GRID_PAIRS.items():
        gap, q = grid_gap(a, b, n_joints, steps)
        if gap > 0.0:
            data["excluded"].append({"a": a, "b": b, "why": f"meshes never touch over the range of the first "
                                     f"{n_joints} joints: smallest hull gap {gap * 1e3:.1f} mm at {np.round(q, 3).tolist()}"})
    model = _model(data)
    lo, hi = left_chain().q_limits(margin_deg=0.0)
    gaps = model.pair_gaps(rng.uniform(lo, hi, size=(FIT_SAMPLES, 7)))
    part_of = dict(zip(model.names, model.parts, strict=True)) | {"chassis": "chassis"}
    columns = [(part_of[x], part_of[y]) for x, y in model.gap_names()]
    for a, b in candidate_pairs():
        k = [m for m, pair in enumerate(columns) if pair == (a, b)]
        if not k:  # excluded above
            continue
        share = float((gaps[:, k].min(axis=1) < 0.0).mean())
        if share >= ALWAYS:
            data["excluded"].append({"a": a, "b": b, "why": f"capsules overlap in {share:.1%} of random poses"})
    return data


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fit Monte02's capsule model to its MJCF meshes")
    parser.add_argument("--xml", type=Path, default=MJCF)
    parser.add_argument("--out", type=Path, default=CAPSULE_FILE)
    args = parser.parse_args(argv)
    fitted = fit(args.xml)
    args.out.write_text(json.dumps(fitted, indent=2) + "\n", encoding="utf-8")
    for c in fitted["capsules"]:
        length = float(np.linalg.norm(np.subtract(c["b"], c["a"])))
        print(f"{c['name']:<14} on {c['link']:<26} r {c['r'] * 1e3:6.1f} mm  length {length * 1e3:6.1f} mm")
    for e in fitted["excluded"]:
        print(f"excluded {e['a']} - {e['b']}: {e['why']}")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
