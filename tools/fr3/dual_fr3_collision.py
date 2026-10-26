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
The collision model two FR3s are planned (C3) and gated (D3) with: capsules around the MJCF's meshes.

Roadmap docs/dual_fr3_bimanual_replay_roadmap.html, C3/D3. One model for both, so that the online gate checks
what the retarget planned against. Each link of ``fr3_corenetic_gripper.xml`` (link0..link7, and everything
on the flange -- gripper, F/T sensor, SMR camera mount -- as ``hand``) gets one or two capsules that enclose
every vertex of its meshes (``fit``, run once with MuJoCo; the result is ``fr3_corenetic_capsules.json``,
which records the MJCF and mesh hashes it was fitted to). The capsules hang on the DH frames of
``fr3_kinematics`` -- the MJCF body frames are those frames (tested) -- so using them needs numpy only, which
is what the arm runtime has.

Enclosing makes a capsule gap a lower bound on the mesh gap (also tested, against MuJoCo's own distance), so
``d_safe`` on top of it is margin for what the model does not know -- the placement of the bases and the arms'
tracking error -- not for the model's shape.

The environment is a list of primitives in the right arm's base: half-spaces (the table), spheres and capsules
(cameras at their registered poses), each saying whether its pose and its size are measured or assumed.
"""

# ruff: noqa: N802, N803, N806  (matrix names: R, T, J follow the maths)
from __future__ import annotations

import hashlib
import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import fr3_kinematics as kin  # noqa: E402

SCHEMA = "fr3_capsules/v1"
FR3_MJCF = kin.FR3_URDF.with_suffix(".xml")
CAPSULE_FILE = Path(__file__).resolve().parent / "fr3_corenetic_capsules.json"
#: Link frames, in the order ``link_frames`` returns them: the base (link0), the seven DH frames, the flange.
FRAMES = ("link0", "link1", "link2", "link3", "link4", "link5", "link6", "link7", "hand")
#: What the capsules are fitted to: part -> (the frame it hangs on, how many capsules, the MJCF bodies whose
#: meshes it holds -- each with its subtree, down to a body another part names). Two capsules where one would
#: swallow a corner (the base's cable plate, the elbows); the flange carries three parts, the gripper's body,
#: its fingers and the SMR camera mount sticking out at 45 deg, because one capsule round all of them is 80 mm
#: fat and reaches 20 mm past the finger tips.
PARTS = {
    "link0": ("link0", 2, ("fr3_link0",)),
    "link1": ("link1", 1, ("fr3_link1",)),
    "link2": ("link2", 1, ("fr3_link2",)),
    "link3": ("link3", 1, ("fr3_link3",)),
    "link4": ("link4", 2, ("fr3_link4",)),
    "link5": ("link5", 2, ("fr3_link5",)),
    "link6": ("link6", 2, ("fr3_link6",)),
    "link7": ("link7", 1, ("fr3_link7",)),
    "gripper": ("hand", 1, ("fr3_link8",)),
    "fingers": (
        "hand",
        1,
        (
            "link_gripper_knuckle_left_outer",
            "link_gripper_knuckle_left_inner",
            "link_gripper_knuckle_right_outer",
            "link_gripper_knuckle_right_inner",
        ),
    ),
    "smr_mount": ("hand", 1, ("link_corenetic_mount",)),
}
#: Parts kept off the table only by not being planned against it: the base and shoulder are bolted to it, and
#: the fingers are meant to reach it -- a grasp from the table top is in the demonstrations, and keeping the
#: fingers a margin above an assumed table would lift every grasp. They still count against the other arm and
#: the cameras.
TABLE_EXEMPT = ("link0", "link1", "fingers")
_FRAME_BODY = {name: f"fr3_{name}" for name in FRAMES[:-1]} | {"hand": "fr3_link8"}


def link_frames(arm: kin.Arm, q: np.ndarray) -> np.ndarray:
    """(n, 9, 4, 4): base, DH frames 1..7, flange -- the frames ``FRAMES`` names, in the arm's base."""
    q = np.atleast_2d(np.asarray(q, dtype=np.float64))
    T = np.broadcast_to(np.eye(4), (len(q), 4, 4)).copy()
    frames = [T.copy()]
    for i, (a, d, alpha, offset) in enumerate(arm.dh):
        T = T @ kin._dh_batch(a, d, alpha, q[:, i] + offset)
        frames.append(T.copy())
    flange = T.copy()
    flange[:, :3, 3] += T[:, :3, 2] * arm.flange_d
    frames.append(flange)
    return np.stack(frames, axis=1)


# ------------------------------------------------------------------ geometry ---


def segment_distance(p1: np.ndarray, q1: np.ndarray, p2: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Closest distance between segments p1-q1 and p2-q2, broadcast over leading axes (Ericson 5.1.9)."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a = np.einsum("...i,...i", d1, d1)
    e = np.einsum("...i,...i", d2, d2)
    f = np.einsum("...i,...i", d2, r)
    c = np.einsum("...i,...i", d1, r)
    b = np.einsum("...i,...i", d1, d2)
    tiny = 1e-12
    a_safe, e_safe = np.where(a > tiny, a, 1.0), np.where(e > tiny, e, 1.0)
    denom = a * e - b * b
    s = np.where(denom > tiny, np.clip((b * f - c * e) / np.where(denom > tiny, denom, 1.0), 0.0, 1.0), 0.0)
    t = (b * s + f) / e_safe
    s = np.where(
        t < 0.0, np.clip(-c / a_safe, 0.0, 1.0), np.where(t > 1.0, np.clip((b - c) / a_safe, 0.0, 1.0), s)
    )
    t = np.clip(t, 0.0, 1.0)
    # degenerate segments (points)
    s = np.where(a <= tiny, 0.0, s)
    t = np.where(a <= tiny, np.clip(f / e_safe, 0.0, 1.0), t)
    s = np.where((a > tiny) & (e <= tiny), np.clip(-c / a_safe, 0.0, 1.0), s)
    t = np.where(e <= tiny, 0.0, t)
    closest = (p1 + d1 * s[..., None]) - (p2 + d2 * t[..., None])
    return np.linalg.norm(closest, axis=-1)


def _point_segment_distance(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    d = b - a
    dd = float(d @ d)
    t = np.zeros(len(points)) if dd < 1e-18 else np.clip((points - a) @ d / dd, 0.0, 1.0)
    return np.linalg.norm(points - (a + t[:, None] * d), axis=1)


# ------------------------------------------------------------------ fitting ---


def _circle_from(points: list[np.ndarray]) -> tuple[np.ndarray, float]:
    if not points:
        return np.zeros(2), -1.0
    if len(points) == 1:
        return points[0], 0.0
    if len(points) == 2:
        c = (points[0] + points[1]) / 2.0
        return c, float(np.linalg.norm(points[0] - c))
    (ax, ay), (bx, by), (cx, cy) = points
    d = 2.0 * (ax * (by - cy) + bx * (cy - ay) + cx * (ay - by))
    if abs(d) < 1e-15:  # collinear: the widest pair
        pairs = [(points[i], points[j]) for i in range(3) for j in range(i + 1, 3)]
        p, q = max(pairs, key=lambda pq: float(np.linalg.norm(pq[0] - pq[1])))
        return _circle_from([p, q])
    ux = (
        (ax * ax + ay * ay) * (by - cy) + (bx * bx + by * by) * (cy - ay) + (cx * cx + cy * cy) * (ay - by)
    ) / d
    uy = (
        (ax * ax + ay * ay) * (cx - bx) + (bx * bx + by * by) * (ax - cx) + (cx * cx + cy * cy) * (bx - ax)
    ) / d
    c = np.array([ux, uy])
    return c, float(np.linalg.norm(points[0] - c))


def _min_enclosing_circle(xy: np.ndarray, rng: np.random.Generator) -> tuple[np.ndarray, float]:
    """Welzl, iterative (move-to-front), on a few hundred hull points."""
    pts = list(xy[rng.permutation(len(xy))])
    c, r = _circle_from([pts[0]])
    for i in range(1, len(pts)):
        if np.linalg.norm(pts[i] - c) <= r + 1e-12:
            continue
        c, r = _circle_from([pts[i]])
        for j in range(i):
            if np.linalg.norm(pts[j] - c) <= r + 1e-12:
                continue
            c, r = _circle_from([pts[i], pts[j]])
            for k in range(j):
                if np.linalg.norm(pts[k] - c) > r + 1e-12:
                    c, r = _circle_from([pts[i], pts[j], pts[k]])
    return c, r


def _hull_points(points: np.ndarray, rng: np.random.Generator, directions: int = 3000) -> np.ndarray:
    """The points extreme along some of many directions: a superset of most hull vertices, small enough for
    the exact fits. Anything it misses is caught by the final check against every point."""
    u = rng.normal(size=(directions, 3))
    u /= np.linalg.norm(u, axis=1, keepdims=True)
    proj = points @ u.T
    keep = np.unique(np.concatenate([proj.argmax(axis=0), proj.argmin(axis=0)]))
    return points[keep]


def _fit_one(
    points: np.ndarray, hull: np.ndarray, rng: np.random.Generator
) -> tuple[np.ndarray, np.ndarray, float]:
    """The smallest-volume capsule over a cone of axes around the principal one that encloses ``hull``; then
    grown to enclose every point of ``points``."""
    centre = hull.mean(axis=0)
    _, _, Vt = np.linalg.svd(hull - centre, full_matrices=False)
    principal = Vt[0]
    best: tuple[float, np.ndarray, np.ndarray, float] | None = None
    axes = [principal]
    for tilt in np.radians([5.0, 10.0, 20.0]):
        for turn in np.linspace(0.0, 2 * math.pi, 8, endpoint=False):
            axes.append(
                math.cos(tilt) * principal
                + math.sin(tilt) * (math.cos(turn) * Vt[1] + math.sin(turn) * Vt[2])
            )
    for u in axes:
        u = u / np.linalg.norm(u)
        v = np.cross(u, Vt[2] if abs(u @ Vt[2]) < 0.9 else Vt[1])
        v /= np.linalg.norm(v)
        w = np.cross(u, v)
        t = hull @ u
        xy = np.stack([hull @ v, hull @ w], axis=1)
        c2, _ = _min_enclosing_circle(xy, rng)
        radial = np.linalg.norm(xy - c2, axis=1)
        r = float(radial.max())
        h = np.sqrt(np.maximum(r * r - radial * radial, 0.0))
        s0, s1 = float((t + h).min()), float((t - h).max())
        if s0 > s1:
            s0 = s1 = (s0 + s1) / 2.0
        volume = math.pi * r * r * (s1 - s0) + 4.0 / 3.0 * math.pi * r**3
        if best is None or volume < best[0]:
            base = c2[0] * v + c2[1] * w
            best = (volume, base + s0 * u, base + s1 * u, r)
    _, a, b, r = best
    r = max(r, float(_point_segment_distance(points, a, b).max()))
    return a, b, r


def _fit_part(
    points: np.ndarray, count: int, rng: np.random.Generator
) -> list[tuple[np.ndarray, np.ndarray, float]]:
    """``count`` capsules (1-3) that together enclose ``points``: split along the principal axis where the
    total volume is least."""
    hull = _hull_points(points, rng)
    if count == 1:
        return [_fit_one(points, hull, rng)]
    centre = points.mean(axis=0)
    axis = np.linalg.svd(hull - centre, full_matrices=False)[2][0]
    t = points @ axis
    best: tuple[float, list] | None = None
    for cut in np.quantile(t, np.linspace(0.15, 0.85, 11)):
        low, high = points[t <= cut], points[t > cut]
        if len(low) < 10 or len(high) < 10:
            continue
        parts = [_fit_one(low, _hull_points(low, rng, 800), rng)]
        parts += (
            _fit_part(high, count - 1, rng)
            if count > 2
            else [_fit_one(high, _hull_points(high, rng, 800), rng)]
        )
        volume = sum(
            math.pi * r * r * float(np.linalg.norm(b - a)) + 4.0 / 3.0 * math.pi * r**3 for a, b, r in parts
        )
        if best is None or volume < best[0]:
            best = (volume, parts)
    return best[1]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def model_fingerprint(xml_path: Path | str = FR3_MJCF) -> dict[str, Any]:
    """sha256 of the MJCF and of every mesh file it names, and one digest over all of them."""
    import xml.etree.ElementTree as ET

    xml_path = Path(xml_path)
    root = ET.parse(str(xml_path)).getroot()
    compiler = root.find("compiler")
    meshdir = xml_path.parent / (compiler.get("meshdir", "") if compiler is not None else "")
    files = {xml_path.name: _sha256(xml_path)}
    for mesh in root.iter("mesh"):
        if mesh.get("file"):
            path = (meshdir / mesh.get("file")).resolve()
            files[str(path.relative_to(xml_path.parent.resolve(), walk_up=True))] = _sha256(path)
    combined = hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest()
    return {
        "xml": str(xml_path.relative_to(REPO)) if xml_path.is_relative_to(REPO) else str(xml_path),
        "sha256": combined,
        "files": files,
    }


def part_vertices(xml_path: Path | str = FR3_MJCF) -> dict[str, np.ndarray]:
    """Every mesh vertex of each part (``PARTS``), in its frame. The flange's parts are taken at the gripper's
    two extremes, open and closed, so their capsules hold whatever opening the replay commands."""
    import mujoco

    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    named = {model.body(body).id: part for part, (_, _, bodies) in PARTS.items() for body in bodies}
    owner: dict[int, str] = {}
    for b in range(model.nbody):
        walk = b
        while walk not in named and walk != 0:
            walk = int(model.body_parentid[walk])
        if walk in named:
            owner[b] = named[walk]
    frame_body = {part: model.body(_FRAME_BODY[frame]).id for part, (frame, _, _) in PARTS.items()}

    gripper_joints = [j for j in range(model.njnt) if not model.joint(j).name.startswith("fr3_joint")]
    out: dict[str, list[np.ndarray]] = {part: [] for part in PARTS}
    for opening in (0.0, 1.0):
        data.qpos[:] = 0.0
        for j in gripper_joints:
            low, high = model.jnt_range[j]
            data.qpos[model.jnt_qposadr[j]] = low + opening * (high - low)
        mujoco.mj_forward(model, data)
        seen: set[tuple[int, int]] = set()
        for g in range(model.ngeom):
            body = int(model.geom_bodyid[g])
            part = owner.get(body)
            if part is None or (PARTS[part][0] != "hand" and opening > 0.0):
                continue
            R_f, p_f = data.xmat[frame_body[part]].reshape(3, 3), data.xpos[frame_body[part]]
            R_g, p_g = data.geom_xmat[g].reshape(3, 3), data.geom_xpos[g]
            if model.geom_type[g] == mujoco.mjtGeom.mjGEOM_MESH:
                mesh = int(model.geom_dataid[g])
                key = (mesh, body)
                if key in seen:  # a visual and a collision geom on one mesh
                    continue
                seen.add(key)
                start, count = model.mesh_vertadr[mesh], model.mesh_vertnum[mesh]
                local = model.mesh_vert[start : start + count].astype(np.float64)
            elif model.geom_type[g] == mujoco.mjtGeom.mjGEOM_SPHERE:
                radius = float(model.geom_size[g][0])
                u = np.random.default_rng(0).normal(size=(200, 3))
                local = radius * u / np.linalg.norm(u, axis=1, keepdims=True)
            else:
                continue
            world = local @ R_g.T + p_g
            out[part].append((world - p_f) @ R_f)
    return {name: np.concatenate(chunks) for name, chunks in out.items() if chunks}


def fit(xml_path: Path | str = FR3_MJCF, seed: int = 0) -> dict[str, Any]:
    """Capsules for every part, as ``CAPSULE_FILE`` holds them."""
    rng = np.random.default_rng(seed)
    vertices = part_vertices(xml_path)
    capsules = []
    for part, (frame, count, _) in PARTS.items():
        for k, (a, b, r) in enumerate(_fit_part(vertices[part], count, rng)):
            capsules.append(
                {
                    "name": f"{part}.{k}",
                    "part": part,
                    "frame": frame,
                    "a": a.tolist(),
                    "b": b.tolist(),
                    "r": float(r),
                }
            )
    return {
        "schema": SCHEMA,
        "model": model_fingerprint(xml_path),
        "note": "Each part's capsules enclose every vertex of its meshes (the flange's parts at the gripper fully "
        "open and fully closed). Frames: link0 = base, linkN = DH frame N, hand = flange.",
        "capsules": capsules,
    }


# ------------------------------------------------------------------ the model ---


@dataclass(frozen=True)
class CapsuleModel:
    names: tuple[str, ...]
    parts: tuple[str, ...]
    frame_index: np.ndarray  # (K,) index into link_frames' second axis
    a: np.ndarray  # (K,3) in their frame
    b: np.ndarray  # (K,3)
    r: np.ndarray  # (K,)
    fingerprint: str

    def place(
        self, arm: kin.Arm, q: np.ndarray, T_world_base: np.ndarray | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Capsule end points (n,K,3), (n,K,3) in ``T_world_base``'s world (the arm's base when None)."""
        F = link_frames(arm, q)[:, self.frame_index]
        if T_world_base is not None:
            F = np.asarray(T_world_base) @ F
        A = np.einsum("nkij,kj->nki", F[..., :3, :3], self.a) + F[..., :3, 3]
        B = np.einsum("nkij,kj->nki", F[..., :3, :3], self.b) + F[..., :3, 3]
        return A, B

    def table_mask(self) -> np.ndarray:
        return np.array([part not in TABLE_EXEMPT for part in self.parts])


def load(path: Path | str = CAPSULE_FILE) -> CapsuleModel:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if data.get("schema") != SCHEMA:
        raise ValueError(f"{path}: schema {data.get('schema')!r}, expected {SCHEMA!r}")
    caps = data["capsules"]
    return CapsuleModel(
        names=tuple(c["name"] for c in caps),
        parts=tuple(c["part"] for c in caps),
        frame_index=np.array([FRAMES.index(c["frame"]) for c in caps]),
        a=np.array([c["a"] for c in caps], dtype=np.float64),
        b=np.array([c["b"] for c in caps], dtype=np.float64),
        r=np.array([c["r"] for c in caps], dtype=np.float64),
        fingerprint=str(data["model"]["sha256"]),
    )


# ------------------------------------------------------------------ environment ---

OBSTACLE_KINDS = ("halfspace", "sphere", "capsule")


def check_obstacles(obstacles: Sequence[Mapping[str, Any]]) -> list[str]:
    problems = []
    for i, ob in enumerate(obstacles):
        label = ob.get("name") or f"#{i}"
        if ob.get("kind") not in OBSTACLE_KINDS:
            problems.append(f"obstacle {label}: kind {ob.get('kind')!r} not one of {OBSTACLE_KINDS}")
        for key in ("pose", "size"):
            if ob.get(key) not in ("measured", "assumed"):
                problems.append(f"obstacle {label}: {key} must say measured or assumed")
    return problems


def obstacle_gaps(
    obstacles: Sequence[Mapping[str, Any]],
    A: np.ndarray,
    B: np.ndarray,
    r: np.ndarray,
    table_mask: np.ndarray,
) -> np.ndarray:
    """(n, K, M) surface gap of each capsule to each obstacle, metres. Capsules exempt from the table get
    +inf against half-spaces."""
    gaps = []
    for ob in obstacles:
        kind = ob["kind"]
        if kind == "halfspace":
            normal = np.asarray(ob["normal"], dtype=np.float64)
            normal = normal / np.linalg.norm(normal)
            offset = float(ob["offset"])
            g = np.minimum(A @ normal, B @ normal) - offset - r
            g = np.where(table_mask, g, np.inf)
        else:
            a = np.asarray(ob["a"] if kind == "capsule" else ob["centre"], dtype=np.float64)
            b = np.asarray(ob["b"] if kind == "capsule" else ob["centre"], dtype=np.float64)
            g = segment_distance(A, B, a, b) - r - float(ob["r"])
        gaps.append(g)
    if not gaps:
        return np.full((*A.shape[:2], 0), np.inf)
    return np.stack(gaps, axis=-1)


# ------------------------------------------------------------------ two arms ---


def arm_arm_gaps(
    model: CapsuleModel, A_r: np.ndarray, B_r: np.ndarray, A_l: np.ndarray, B_l: np.ndarray
) -> np.ndarray:
    """(n, K, K) surface gaps between every right capsule and every left capsule (same world)."""
    d = segment_distance(A_r[:, :, None, :], B_r[:, :, None, :], A_l[:, None, :, :], B_l[:, None, :, :])
    return d - model.r[None, :, None] - model.r[None, None, :]


def clearance(
    model: CapsuleModel,
    q_right: np.ndarray,
    q_left: np.ndarray,
    T_right_left: np.ndarray,
    obstacles: Sequence[Mapping[str, Any]] = (),
    arm: kin.Arm = kin.NOMINAL,
) -> dict[str, Any]:
    """Per frame: the arms' smallest gap and the capsules it is between, and each arm's smallest gap to the
    environment. Everything in the right arm's base."""
    A_r, B_r = model.place(arm, q_right)
    A_l, B_l = model.place(arm, q_left, T_right_left)
    gaps = arm_arm_gaps(model, A_r, B_r, A_l, B_l)
    flat = gaps.reshape(len(gaps), -1)
    k = flat.argmin(axis=1)
    out: dict[str, Any] = {
        "arm_arm_m": flat[np.arange(len(flat)), k],
        "arm_arm_pair": [(model.names[i // len(model.names)], model.names[i % len(model.names)]) for i in k],
    }
    mask = model.table_mask()
    for name, (A, B) in (("right", (A_r, B_r)), ("left", (A_l, B_l))):
        env = obstacle_gaps(obstacles, A, B, model.r, mask)
        if env.shape[-1] == 0:
            out[f"env_{name}_m"] = np.full(len(A), np.inf)
            out[f"env_{name}_with"] = [None] * len(A)
            continue
        flat_env = env.reshape(len(env), -1)
        j = flat_env.argmin(axis=1)
        out[f"env_{name}_m"] = flat_env[np.arange(len(flat_env)), j]
        out[f"env_{name}_with"] = [
            (model.names[i // len(obstacles)], str(obstacles[i % len(obstacles)].get("name"))) for i in j
        ]
    return out


def main(argv: list[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser(description="Fit the FR3 + gripper capsule model to the MJCF meshes")
    parser.add_argument("--xml", type=Path, default=FR3_MJCF)
    parser.add_argument("--out", type=Path, default=CAPSULE_FILE)
    args = parser.parse_args(argv)
    fitted = fit(args.xml)
    args.out.write_text(json.dumps(fitted, indent=2) + "\n", encoding="utf-8")
    for c in fitted["capsules"]:
        length = float(np.linalg.norm(np.subtract(c["b"], c["a"])))
        print(f"{c['name']:<8} r {c['r'] * 1e3:6.1f} mm  length {length * 1e3:6.1f} mm")
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
