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
Monte02 tip-over margins: the centre of mass model and the trunk posture whitelist (roadmap
docs/monte02_left_arm_roadmap.html, T1-T2).

Monte01 fell over forwards. Monte02 stands on two drive wheels at the front and two casters behind, so the
front edge of its support polygon is the drive axle, and a trunk bent forward with an arm reaching forward
puts the centre of mass over it. Here:

* ``MassModel``: every link's mass at its URDF ``<inertial>`` origin, walked down the whole tree (trunk, head,
  both arms, the grippers), plus ``Load`` s (a payload in the hand; the weighed hand, ``measured``; later the
  corrections the weighing fits, T1'); ``com(posture, q_left)`` in the base frame, batched over left-arm poses;
* ``Support``: the polygon of the wheel contact points on the floor, read from the wheels' collision shapes;
* ``margins``: per edge, how far inside the polygon the centre of mass is, and the base acceleration that
  tips the robot over that edge (static, level floor: a = g * d / h, h the height of the centre of mass);
* ``worst``: per edge, the arm pose (and head turn) within the joint limits that brings the centre of mass
  nearest to that edge: a trunk posture is safe only if it is safe whatever the arm does, because a policy
  drives the arm;
* the whitelist (``WHITELIST``, in the private submodule beside the capsules): the trunk postures a session
  may lock (D4), each verified against ``worst`` with the payload, bound to the mass model's fingerprint.
  The margin an edge needs is a ``Requirement`` derived from measurements (the weighed model's error, the
  chassis e-stop deceleration, ...), not a chosen number; until those exist the entries are only screened
  (``SCREENING_MARGIN_M``) and the whitelist is not in force. ``check`` re-verifies every entry;
  ``refusals(posture)`` is what a session start and the T5 gate ask.

The URDF's masses are the vendor's THEORY values (the chassis is only ~4.5 kg in it); every number here is
provisional until the robot is weighed (T1'). Nothing here talks to a robot.
"""

# ruff: noqa: N803, N806  (matrix names: R, T follow the maths)
from __future__ import annotations

import functools
import hashlib
import json
import math
import sys
import xml.etree.ElementTree as ET
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import monte02_model as mm, urdf_kinematics as uk  # noqa: E402

G = 9.81
WHITELIST = mm.DESCRIPTION / "balance/trunk_whitelist.json"
SCHEMA = "monte02_trunk_whitelist/v1"
#: Joints whose value cannot move the centre of mass: the wheels spin about their axles (their mass is on
#: them), and the gripper fingers weigh 32 g in all; they are taken at zero.
_STILL = ("joint_b_wheel_drive_left", "joint_b_wheel_drive_right", "joint_lt_gripper_drive", "joint_rt_gripper_drive")
#: Wheels whose collision shape stands on the floor.
WHEELS = ("link_b_wheel_drive_left", "link_b_wheel_drive_right", "link_b_wheel_idler_left", "link_b_wheel_idler_right")
SAMPLES = 20000
REFINE_SEEDS = 12
REFINE_ITERS = 150


@dataclass(frozen=True)
class Load:
    """A point mass on a link: a payload in the hand, or a correction the weighing fits."""

    name: str
    mass: float
    link: str
    point: tuple[float, float, float] = (0.0, 0.0, 0.0)


def payload(mass_kg: float) -> Load:
    """An object held in the left hand, at the TCP (between the fingertips)."""
    return Load("payload", float(mass_kg), mm.TCP_LINK)


#: 11-24, weighed: the left hand as it goes onto Monte02 (D2: the gripper with its BOX and the GMSL2 camera),
#: everything below the wrist flange, i.e. the URDF's links from ``HAND_ROOT`` down.
HAND_MEASURED_KG = 1.182
HAND_ROOT = "link_l_sensor_ft"
#: What a session's hand may hold (provisional, 11-24): the payload every whitelist entry is verified with.
PAYLOAD_KG = 0.2


# ------------------------------------------------------------------ the tree ---


@dataclass(frozen=True)
class _Joint:
    name: str
    kind: str
    parent: str
    child: str
    origin: np.ndarray
    axis: np.ndarray
    mimic: tuple[str, float, float] | None


@functools.lru_cache(maxsize=1)
def _tree() -> tuple[tuple[_Joint, ...], dict[str, tuple[float, np.ndarray]], str]:
    """Joints parents-first from the base, each link's (mass, centre of mass in its frame), the URDF's sha256."""
    data = mm.URDF.read_bytes()
    root = ET.fromstring(data)
    by_parent: dict[str, list[ET.Element]] = {}
    for j in root.findall("joint"):
        by_parent.setdefault(j.find("parent").get("link"), []).append(j)
    joints, todo = [], [mm.BASE_LINK]
    while todo:
        link = todo.pop(0)
        for j in by_parent.get(link, []):
            kind = j.get("type")
            mimic = j.find("mimic")
            joints.append(_Joint(
                name=j.get("name"), kind=kind, parent=link, child=j.find("child").get("link"),
                origin=uk._origin(j), axis=uk._axis(j) if kind in uk.MOVABLE else np.zeros(3),
                mimic=None if mimic is None else (mimic.get("joint"), float(mimic.get("multiplier", "1")),
                                                  float(mimic.get("offset", "0"))),
            ))
            todo.append(joints[-1].child)
    masses = {}
    for link in root.findall("link"):
        inertial = link.find("inertial")
        if inertial is None:
            continue
        mass = float(inertial.find("mass").get("value"))
        origin = inertial.find("origin")
        com = [float(v) for v in (origin.get("xyz", "0 0 0") if origin is not None else "0 0 0").split()]
        masses[link.get("name")] = (mass, np.array(com))
    reached = {mm.BASE_LINK} | {j.child for j in joints}
    if set(masses) - reached:
        raise ValueError(f"links with mass not below {mm.BASE_LINK}: {sorted(set(masses) - reached)}")
    return tuple(joints), masses, hashlib.sha256(data).hexdigest()


def link_frames(values: dict[str, np.ndarray | float], n: int) -> dict[str, np.ndarray]:
    """(n,4,4) base->link of every link, for joint ``values`` (scalars or (n,) arrays). Every moving joint must
    have a value, except the ones that cannot move the centre of mass (``_STILL``, zero) and mimic joints."""
    joints, _, _ = _tree()
    frames = {mm.BASE_LINK: np.broadcast_to(np.eye(4), (n, 4, 4))}
    resolved: dict[str, np.ndarray] = {}
    for j in joints:
        T = frames[j.parent] @ j.origin
        if j.kind in uk.MOVABLE:
            if j.mimic is not None:
                leader, mul, off = j.mimic
                q = mul * resolved[leader] + off
            elif j.name in values:
                q = np.broadcast_to(np.asarray(values[j.name], dtype=np.float64), (n,))
            elif j.name in _STILL:
                q = np.zeros(n)
            else:
                raise ValueError(f"no value for joint {j.name}")
            resolved[j.name] = q
            T = T @ uk._motion(j.kind, j.axis, q)
        frames[j.child] = T
    return frames


# ------------------------------------------------------------------ mass model ---


@dataclass(frozen=True)
class MassModel:
    """The URDF's link masses plus ``extra`` point masses (corrections, T1')."""

    extra: tuple[Load, ...] = ()

    @property
    def total(self) -> float:
        _, masses, _ = _tree()
        return sum(m for m, _ in masses.values()) + sum(load.mass for load in self.extra)

    @property
    def fingerprint(self) -> str:
        """The URDF bytes and the extra masses: what a whitelist was verified against."""
        _, _, sha = _tree()
        extra = json.dumps([[e.name, e.mass, e.link, list(e.point)] for e in self.extra])
        return hashlib.sha256(f"{sha}|{extra}".encode()).hexdigest()

    def com(
        self,
        posture: mm.Posture,
        q_left: np.ndarray,
        loads: Sequence[Load] = (),
        head: np.ndarray | None = None,
    ) -> np.ndarray:
        """(n,3) centre of mass in the base frame. ``head`` (n,2), if given, overrides the posture's head."""
        q = np.atleast_2d(np.asarray(q_left, dtype=np.float64))
        values: dict[str, Any] = posture.locked() | dict(zip(mm.LEFT_ARM_JOINTS, q.T, strict=True))
        if head is not None:
            values |= dict(zip(mm.HEAD_JOINTS, np.atleast_2d(head).T, strict=True))
        frames = link_frames(values, len(q))
        _, masses, _ = _tree()
        points = [(m, link, c) for link, (m, c) in masses.items() if m > 0.0]
        points += [(e.mass, e.link, np.asarray(e.point, dtype=np.float64)) for e in (*self.extra, *loads)]
        moment, total = np.zeros((len(q), 3)), 0.0
        for m, link, c in points:
            T = frames[link]
            moment += m * (T[:, :3, :3] @ c + T[:, :3, 3])
            total += m
        return moment / total


def urdf_hand_kg() -> float:
    """The URDF's mass from ``HAND_ROOT`` down."""
    joints, masses, _ = _tree()
    below, total = {HAND_ROOT}, masses.get(HAND_ROOT, (0.0, None))[0]
    for j in joints:  # parents first
        if j.parent in below:
            below.add(j.child)
            total += masses.get(j.child, (0.0, None))[0]
    return total


def hand_correction(measured_kg: float = HAND_MEASURED_KG) -> Load:
    """The weighed hand's mass beyond the URDF's, put at the TCP: the hand's centre of mass was not measured,
    and the TCP is further out than it, so the margins err on the safe side. A hand lighter than the URDF's
    is not subtracted (that would err the other way)."""
    urdf = urdf_hand_kg()
    return Load(f"left hand weighed {measured_kg:.3f} kg, URDF {urdf:.3f} kg", max(measured_kg - urdf, 0.0), mm.TCP_LINK)


def measured() -> MassModel:
    """The mass model the whitelist is verified against: the URDF with the weighed hand (T1' adds more)."""
    return MassModel(extra=(hand_correction(),))


# ------------------------------------------------------------------ support ---


@dataclass(frozen=True)
class Support:
    """The support polygon on the floor (base frame, counter-clockwise) and the floor's height."""

    corners: np.ndarray  # (k,2)
    wheels: tuple[str, ...]  # the wheel at each corner
    floor_z: float

    def edges(self) -> list[tuple[str, np.ndarray, float]]:
        """Per edge: its name (the side it faces), outward unit normal n and offset c (n.p = c on the edge)."""
        out = []
        k = len(self.corners)
        for i in range(k):
            a, b = self.corners[i], self.corners[(i + 1) % k]
            d = (b - a) / np.linalg.norm(b - a)
            n = np.array([d[1], -d[0]])  # outward for a counter-clockwise polygon
            out.append((_facing(n), n, float(n @ a)))
        return out


def _facing(n: np.ndarray) -> str:
    angle = math.degrees(math.atan2(n[1], n[0]))
    return "front" if -45 <= angle < 45 else "left" if 45 <= angle < 135 else "right" if -135 <= angle < -45 else "rear"


@functools.lru_cache(maxsize=1)
def support() -> Support:
    """From the wheels' collision shapes: a drive wheel (a cylinder about its axle) touches the floor straight
    below its centre, a caster (a sphere) too. The floor is the lowest contact."""
    root = ET.parse(mm.URDF).getroot()
    joints, _, _ = _tree()
    frames = link_frames({j.name: 0.0 for j in joints if j.kind in uk.MOVABLE and j.mimic is None}, 1)
    contacts = []
    for name in WHEELS:
        link = root.find(f"link[@name='{name}']")
        geom = link.find("collision/geometry")
        shape = geom.find("cylinder") if geom.find("cylinder") is not None else geom.find("sphere")
        if shape is None:
            raise ValueError(f"{name}: collision is neither a cylinder nor a sphere")
        centre = frames[name][0, :3, 3]
        contacts.append((name, centre[:2], centre[2] - float(shape.get("radius"))))
    floor = min(z for _, _, z in contacts)
    if max(z for _, _, z in contacts) - floor > 0.003:
        raise ValueError(f"wheel contacts not on one floor: {[(n, round(z, 4)) for n, _, z in contacts]}")
    centre = np.mean([p for _, p, _ in contacts], axis=0)
    order = sorted(contacts, key=lambda c: math.atan2(c[1][1] - centre[1], c[1][0] - centre[0]))
    return Support(np.array([p for _, p, _ in order]), tuple(n for n, _, _ in order), floor)


def margins(com: np.ndarray, base: Support | None = None) -> dict[str, Any]:
    """Per pose (rows of ``com``) and edge: how far inside the polygon the centre of mass is (m, negative:
    outside, it falls) and the base acceleration away from that edge that tips it over the edge (m/s^2:
    braking while driving towards the edge, or accelerating away from it)."""
    base = base or support()
    com = np.atleast_2d(com)
    edges = base.edges()
    d = np.stack([c - com[:, :2] @ n for _, n, c in edges], axis=1)
    h = com[:, 2] - base.floor_z
    k = d.argmin(axis=1)
    rows = np.arange(len(com))
    return {
        "edges": [name for name, _, _ in edges],
        "margin_m": d,
        "accel_mps2": G * d / h[:, None],
        "height_m": h,
        "min_m": d[rows, k],
        "min_edge": [edges[i][0] for i in k],
    }


# ------------------------------------------------------------------ worst case ---


def _box() -> tuple[np.ndarray, np.ndarray]:
    """Limits of the left arm then the head (9 values)."""
    lo, hi = mm.left_chain().q_limits(margin_deg=0.0)
    root = ET.parse(mm.URDF).getroot()
    head = [root.find(f"joint[@name='{j}']/limit") for j in mm.HEAD_JOINTS]
    return (np.concatenate([lo, [float(h.get("lower")) for h in head]]),
            np.concatenate([hi, [float(h.get("upper")) for h in head]]))


def worst(
    posture: mm.Posture,
    loads: Sequence[Load] = (),
    model: MassModel | None = None,
    free_head: bool = True,
    height_coeff: float = 0.0,
    seed: int = 0,
) -> dict[str, dict[str, Any]]:
    """Per edge, the left-arm pose (and, with ``free_head``, head turn) within the joint limits that leaves the
    least spare margin there: margin - ``height_coeff`` * height of the centre of mass (0: plain margin; a
    ``Requirement``'s coefficient: what a deceleration or a slope takes, which grows with the height). Random
    poses, then the worst few climbed by projected gradient. Self-collision is ignored, so the answer can be
    slightly pessimistic, never optimistic for a pose the arm can reach (up to the search finding the extreme)."""
    model = model or measured()
    lo, hi = _box()
    if not free_head:
        lo[7:] = hi[7:] = posture.head
    base = support()
    edges = base.edges()
    rng = np.random.default_rng(seed)

    def evaluate(x: np.ndarray) -> np.ndarray:
        com = model.com(posture, x[:, :7], loads, head=x[:, 7:])
        h = com[:, 2] - base.floor_z
        return np.stack([c - com[:, :2] @ n - height_coeff * h for _, n, c in edges], axis=1)

    x = rng.uniform(lo, hi, size=(SAMPLES, 9))
    d = evaluate(x)
    out = {}
    for e, (name, _, _) in enumerate(edges):
        best = x[np.argsort(d[:, e])[:REFINE_SEEDS]].copy()
        step = np.full(len(best), 0.2)
        value = evaluate(best)[:, e]
        eps = 1e-5
        for _ in range(REFINE_ITERS):
            probe = np.repeat(best, 10, axis=0)
            for i in range(9):
                probe[i + 1 :: 10, i] += eps
            vals = evaluate(probe)[:, e].reshape(len(best), 10)
            grad = (vals[:, 1:] - vals[:, :1]) / eps
            norm = np.linalg.norm(grad, axis=1, keepdims=True)
            trial = np.clip(best - step[:, None] * grad / np.maximum(norm, 1e-12), lo, hi)
            trial_value = evaluate(trial)[:, e]
            better = trial_value < value
            best[better], value[better] = trial[better], trial_value[better]
            step = np.where(better, step * 1.2, step * 0.5)
            if step.max() < 1e-4:
                break
        i = int(value.argmin())
        com = model.com(posture, best[i : i + 1, :7], loads, head=best[i : i + 1, 7:])
        m = margins(com, base)
        out[name] = {"margin_m": float(m["margin_m"][0, e]), "height_m": float(m["height_m"][0]),
                     "spare_m": float(value[i]), "accel_mps2": float(m["accel_mps2"][0, e]),
                     "q_left": best[i, :7].tolist(), "head": best[i, 7:].tolist(), "com": com[0].tolist()}
    return out


# ------------------------------------------------------------------ requirement ---


@dataclass(frozen=True)
class Requirement:
    """The margin an edge needs, derived from measurements, not chosen:

    * ``model_error_m``: how far the weighed model's centre of mass may be off horizontally (from the weighing
      residuals, T1');
    * ``decel_mps2``: the base deceleration a session must survive: the chassis e-stop deceleration, measured
      (T3). Applied in every direction with the arm anywhere -- conservative; if that fails every useful
      posture, T3 splits driving off (arm stowed) as its own case;
    * ``tilt_rad``: the floor slope allowed;
    * ``arm_dynamic_m``: how far the arm's own accelerations (its stop included) move the zero-moment point.

    Needed at an edge with the centre of mass at height h: model_error + arm_dynamic + h (tan tilt + decel / g).
    """

    model_error_m: float
    decel_mps2: float
    tilt_rad: float = 0.0
    arm_dynamic_m: float = 0.0

    def __post_init__(self) -> None:
        values = (self.model_error_m, self.decel_mps2, self.tilt_rad, self.arm_dynamic_m)
        if any(not math.isfinite(v) or v < 0.0 for v in values):
            raise ValueError(f"requirement terms must be finite and >= 0: {values}")

    @property
    def height_coeff(self) -> float:
        return math.tan(self.tilt_rad) + self.decel_mps2 / G

    @property
    def offset_m(self) -> float:
        return self.model_error_m + self.arm_dynamic_m

    def margin_m(self, height_m: float) -> float:
        return self.offset_m + height_m * self.height_coeff

    def as_dict(self) -> dict[str, float]:
        return {"model_error_m": self.model_error_m, "decel_mps2": self.decel_mps2, "tilt_rad": self.tilt_rad,
                "arm_dynamic_m": self.arm_dynamic_m}


#: Until the requirement is measured (weighing error, e-stop deceleration), candidates are screened offline
#: (M2.2') against this round number. It is not a requirement: a whitelist screened with it is not in force.
SCREENING_MARGIN_M = 0.08


# ------------------------------------------------------------------ whitelist ---


def _posture(trunk: Sequence[float]) -> mm.Posture:
    return mm.Posture(trunk=tuple(trunk))


def verify(
    trunk: Sequence[float],
    requirement: Requirement | None = None,
    payload_kg: float = PAYLOAD_KG,
    model: MassModel | None = None,
    screening_m: float = SCREENING_MARGIN_M,
) -> dict[str, Any]:
    """One whitelist entry, the arm and the head anywhere, the payload in the hand, the right arm parked: per
    edge the worst margin and, against ``requirement`` (or, without one, ``screening_m``), the spare left."""
    model = model or measured()
    posture = _posture(trunk)
    mm.left_chain(posture)  # the trunk values inside their limits
    coeff, offset = (requirement.height_coeff, requirement.offset_m) if requirement else (0.0, screening_m)
    w = worst(posture, (payload(payload_kg),), model, height_coeff=coeff)
    spare = {name: v["spare_m"] - offset for name, v in w.items()}
    edge = min(spare, key=spare.get)
    return {
        "trunk": [float(v) for v in trunk],
        "basis": "requirement" if requirement else "screening",
        "worst_edge": edge,
        "worst_spare_m": round(spare[edge], 4),
        "worst_margin_m": round(w[edge]["margin_m"], 4),
        "worst_height_m": round(w[edge]["height_m"], 4),
        "worst_q_left": [round(v, 4) for v in w[edge]["q_left"]],
        "worst_head": [round(v, 4) for v in w[edge]["head"]],
        "per_edge_margin_m": {name: round(v["margin_m"], 4) for name, v in w.items()},
        "per_edge_spare_m": {name: round(v, 4) for name, v in spare.items()},
        "per_edge_accel_mps2": {name: round(v["accel_mps2"], 2) for name, v in w.items()},
        "passes": bool(spare[edge] >= 0.0),
    }


def load_whitelist(path: Path | str = WHITELIST) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {data.get('schema')!r}, expected {SCHEMA!r}")
    return data


def _requirement(data: dict[str, Any]) -> Requirement | None:
    return Requirement(**data["requirement"]) if data.get("requirement") else None


def check(path: Path | str = WHITELIST, model: MassModel | None = None) -> list[str]:
    """Everything wrong with the whitelist under the current mass model, against the requirement it records (or
    its screening margin): [] when every entry still holds. Whether it is in force is ``refusals``' business."""
    model = model or measured()
    data = load_whitelist(path)
    requirement = _requirement(data)
    problems = []
    if data["mass_model"] != model.fingerprint:
        problems.append("the mass model changed since the whitelist was verified (URDF or corrections); "
                        "re-verify every entry")
    names = [p["name"] for p in data["postures"]]
    if len(set(names)) != len(names):
        problems.append(f"duplicate names: {names}")
    for entry in data["postures"]:
        got = verify(entry["trunk"], requirement, data["payload_kg"], model, data["screening_margin_m"])
        if not got["passes"]:
            problems.append(f"{entry['name']}: {-got['worst_spare_m'] * 1e3:.0f} mm short at the {got['worst_edge']} "
                            f"edge ({got['basis']})")
        if abs(got["worst_spare_m"] - entry["worst_spare_m"]) > 0.002:
            problems.append(f"{entry['name']}: recorded worst spare {entry['worst_spare_m']} m, now "
                            f"{got['worst_spare_m']} m")
    return problems


def refusals(posture: mm.Posture, path: Path | str = WHITELIST, model: MassModel | None = None) -> list[str]:
    """Why ``posture`` may not be locked for a session ([] when it may): the whitelist must be verified against
    a measured requirement and this mass model, the trunk must be one of its entries (exactly), the right arm
    parked. The head is free: every entry was verified over the head's whole range."""
    model = model or measured()
    data = load_whitelist(path)
    out = []
    if _requirement(data) is None:
        out.append("the whitelist was only screened (no measured requirement yet: the weighed model's error, T1', "
                   "and the chassis e-stop deceleration, T3); it is not in force")
    if data["mass_model"] != model.fingerprint:
        out.append("the whitelist was verified against another mass model (URDF or weighing corrections changed)")
    if tuple(posture.right_arm) != mm.RIGHT_ARM_PARKED:
        out.append(f"right arm {list(posture.right_arm)} is not parked {list(mm.RIGHT_ARM_PARKED)}: the whitelist "
                   "was verified with it parked")
    if not any(np.allclose(posture.trunk, p["trunk"], atol=1e-9, rtol=0.0) for p in data["postures"]):
        out.append(f"trunk {list(posture.trunk)} is not in the whitelist "
                   f"({', '.join(p['name'] + '=' + str(p['trunk']) for p in data['postures'])})")
    return out


def build(
    entries: Sequence[tuple[str, Sequence[float]]],
    requirement: Requirement | None = None,
    payload_kg: float = PAYLOAD_KG,
    screening_m: float = SCREENING_MARGIN_M,
) -> dict[str, Any]:
    """A whitelist from (name, trunk) candidates; one that fails is an error, not left out silently."""
    model = measured()
    postures = []
    for name, trunk in entries:
        got = verify(trunk, requirement, payload_kg, model, screening_m)
        if not got["passes"]:
            raise ValueError(f"{name} {list(trunk)}: {-got['worst_spare_m'] * 1e3:.0f} mm short at the "
                             f"{got['worst_edge']} edge ({got['basis']})")
        postures.append({"name": name, **{k: v for k, v in got.items() if k != "passes"}})
    return {
        "schema": SCHEMA,
        "mass_model": model.fingerprint,
        "total_mass_kg": round(model.total, 3),
        "corrections": [{"name": e.name, "mass_kg": round(e.mass, 4), "link": e.link, "point": list(e.point)}
                        for e in model.extra],
        "requirement": requirement.as_dict() if requirement else None,
        "screening_margin_m": screening_m,
        "payload_kg": payload_kg,
        "right_arm": list(mm.RIGHT_ARM_PARKED),
        "note": "Trunk postures (knee, hip, waist; rad) a session may lock. Each was verified with the payload at "
        "the left TCP, the right arm parked and the left arm and head anywhere within their limits (level "
        "floor, static centre of mass). With a 'requirement' (measured: the weighed model's error, the "
        "chassis e-stop deceleration, the floor slope, the arm's own dynamics) every edge keeps what it needs "
        "at that height; without one the entries were only screened against screening_margin_m, and the "
        "whitelist is not in force. Mass model: the vendor's URDF with the weighed left hand ('corrections').",
        "postures": postures,
    }


# ------------------------------------------------------------------ CLI ---


def _requirement_args(args: Any) -> Requirement | None:
    given = [args.model_error, args.decel]
    if all(v is None for v in given):
        return None
    if any(v is None for v in given):
        raise SystemExit("--model-error and --decel come together (both measured)")
    return Requirement(args.model_error, args.decel, math.radians(args.tilt_deg), args.arm_dynamic)


def _scan(args: Any) -> None:
    base = support()
    requirement = _requirement_args(args)
    print(f"support polygon (base frame, m): {np.round(base.corners, 4).tolist()}  floor z {base.floor_z:.4f}")
    model = measured()
    print(f"total mass {model.total:.2f} kg (hand: {model.extra[0].name}), payload {args.payload} kg, "
          f"{requirement or f'screening {args.screening * 1e3:.0f} mm'}")
    for knee in args.knee:
        for hip in args.hip:
            for waist in args.waist:
                try:
                    got = verify((knee, hip, waist), requirement, args.payload, model, args.screening)
                except ValueError as exc:
                    print(f"knee {knee:+.2f} hip {hip:+.2f} waist {waist:+.2f}: {exc}")
                    continue
                edges = "  ".join(f"{k} {v * 1e3:+5.0f}" for k, v in got["per_edge_margin_m"].items())
                print(f"knee {knee:+.2f} hip {hip:+.2f} waist {waist:+.2f}: worst margin "
                      f"{got['worst_margin_m'] * 1e3:+5.0f} mm, spare {got['worst_spare_m'] * 1e3:+5.0f} mm "
                      f"({got['worst_edge']:<5}) {'ok ' if got['passes'] else 'NO '}| {edges}")


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Monte02 tip-over margins and the trunk posture whitelist")
    sub = parser.add_subparsers(dest="cmd", required=True)
    scan = sub.add_parser("scan", help="worst-case margins over a grid of trunk postures")
    scan.add_argument("--knee", type=float, nargs="+", default=[0.0])
    scan.add_argument("--hip", type=float, nargs="+", default=[0.0])
    scan.add_argument("--waist", type=float, nargs="+", default=[0.0])
    write = sub.add_parser("write", help="verify trunk postures and write them as the whole whitelist")
    write.add_argument("--entry", nargs=4, action="append", required=True, metavar=("NAME", "KNEE", "HIP", "WAIST"))
    write.add_argument("--out", type=Path, default=WHITELIST)
    for p in (scan, write):
        p.add_argument("--payload", type=float, default=PAYLOAD_KG, help="kg held in the left hand")
        p.add_argument("--screening", type=float, default=SCREENING_MARGIN_M, help="m, without a requirement")
        p.add_argument("--model-error", type=float, help="m: the weighed model's centre-of-mass error (T1')")
        p.add_argument("--decel", type=float, help="m/s^2: the measured chassis e-stop deceleration (T3)")
        p.add_argument("--tilt-deg", type=float, default=0.0, help="floor slope allowed")
        p.add_argument("--arm-dynamic", type=float, default=0.0, help="m: ZMP shift by the arm's accelerations")
    sub.add_parser("check", help="re-verify the whitelist under the current mass model")
    args = parser.parse_args(argv)
    if args.cmd == "scan":
        _scan(args)
        return 0
    if args.cmd == "write":
        data = build([(e[0], [float(v) for v in e[1:]]) for e in args.entry], _requirement_args(args), args.payload,
                     args.screening)
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        for e in data["postures"]:
            print(f"{e['name']}: trunk {e['trunk']} worst margin {e['worst_margin_m'] * 1e3:.0f} mm, spare "
                  f"{e['worst_spare_m'] * 1e3:.0f} mm ({e['worst_edge']}, {e['basis']})")
        print(f"wrote {args.out}" + ("" if data["requirement"] else " (screened only: not in force)"))
        return 0
    problems = check()
    for p in problems:
        print(p)
    data = load_whitelist()
    print(("whitelist consistent" if not problems else f"{len(problems)} problem(s)")
          + ("" if data["requirement"] else "; screened only, no measured requirement: not in force"))
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
