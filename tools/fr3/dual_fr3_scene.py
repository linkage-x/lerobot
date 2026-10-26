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
Two FR3s in one MuJoCo scene, placed by the measured base-to-base transform (roadmap C1).

The single-arm preview's ``--cube both`` draws two independent copies of the model side by side: nothing in
it can touch anything else. This builds one model: ``fr3_corenetic_gripper.xml`` attached twice (``right/``,
``left/``), the left at ``T_right_left`` -- whatever B0/B3 measured, or the assumed placement until then --
the gripper on both whether or not the real one is back, and the environment the arms can hit: the table
top, and the cameras at their registered poses. What is not modelled (the camera rig's posts, the laser
tracker) is listed in the scene record, not silently left out.

The scene record carries a fingerprint over the MJCF and every mesh, the placement and the obstacles; a
report made in a scene names that fingerprint, so a changed placement or model shows up as a different scene.

Also here: the exact-mesh distance between the two arms (MuJoCo's convex-hull distance), the independent check
on the capsule model the retarget plans with (``dual_fr3_collision``), and a video of both arms in one view.
"""

# ruff: noqa: N802, N803, N806  (matrix names: R, T, J follow the maths)
from __future__ import annotations

import hashlib
import json
import math
import subprocess
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import dual_fr3_collision as col  # noqa: E402
from tools.thor.gmsl2 import world_provenance as wp  # noqa: E402

SCHEMA = "dual_fr3_scene/v1"
ARM_PREFIX = {"right": "right/", "left": "left/"}
#: A GMSL2 camera module with its lens and housing, as a ball around the optical centre. Assumed size; the
#: position is the registered one.
CAMERA_RADIUS_M = 0.06
#: Not in the scene, and said so in every record made with it.
UNMODELLED = (
    "camera rig posts and cross bars (no geometry measured; only the cameras' optical centres are known)",
    "laser tracker head and tripod (no station edge in world_graph.json)",
    "objects on the table",
)


def table_obstacle(height_m: float = 0.0) -> dict[str, Any]:
    """The table top as a half-space below ``z = height_m`` of the right base. Assumed: the bases are bolted
    to the table top, so its surface is each base's z = 0."""
    return {
        "name": "table",
        "kind": "halfspace",
        "normal": [0.0, 0.0, 1.0],
        "offset": float(height_m),
        "pose": "assumed",
        "size": "assumed",
        "note": "the arms stand on the table top (z = 0 of the right base); its edges are not modelled",
    }


def camera_obstacles(
    repo_root: Path | str = REPO, graph: Mapping[str, Any] | None = None
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Each registered camera as a ball at its optical centre in the right base, and where that came from.

    Positions come from ``world_registration.json`` (``camera_poses_in_world``, T_world_cam) through the
    world graph's edge into ``fr3_base``; nothing when either is missing, with the reason in the source.
    """
    repo_root = Path(repo_root)
    path = repo_root / wp.WORLD_SUBDIR / wp.WORLD_REGISTRATION_FILE
    try:
        registration = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        return [], {"cameras": f"not placed: {path} unreadable ({exc})"}
    graph = graph if graph is not None else wp.read_world_graph(repo_root)
    world = str(registration.get("world_frame_id") or "")
    found = wp.world_transform(graph, world, wp.FR3_BASE_WORLD_ID) if world else None
    if found is None:
        return [], {
            "cameras": f"not placed: no edge from {world or '(no world id)'} to {wp.FR3_BASE_WORLD_ID}"
        }
    T_base_world = np.asarray(found[0], dtype=np.float64)
    obstacles = []
    for name, pose in sorted((registration.get("camera_poses_in_world") or {}).items()):
        centre = (T_base_world @ np.asarray(pose, dtype=np.float64))[:3, 3]
        obstacles.append(
            {
                "name": name,
                "kind": "sphere",
                "centre": centre.tolist(),
                "r": CAMERA_RADIUS_M,
                "pose": "measured",
                "size": "assumed",
            }
        )
    source = {
        "cameras": str(path.relative_to(repo_root)),
        "calibration_id": registration.get("calibration_id"),
        "world_frame_id": world,
        "edges": found[1],
    }
    return obstacles, source


def scene_record(
    T_right_left: np.ndarray,
    placement: Mapping[str, Any],
    obstacles: Sequence[Mapping[str, Any]],
    obstacle_source: Mapping[str, Any] | None = None,
    xml_path: Path | str = col.FR3_MJCF,
) -> dict[str, Any]:
    """What a scene is, with one fingerprint over all of it. ``placement`` says where ``T_right_left`` came
    from (``{"source": "edge"|"assumed", "edges": [...], ...}``)."""
    problems = col.check_obstacles(obstacles)
    if problems:
        raise ValueError("; ".join(problems))
    model = col.model_fingerprint(xml_path)
    capsules = col.load()
    body = {
        "model_sha256": model["sha256"],
        "capsule_model_sha256": capsules.fingerprint,
        "T_right_left": np.asarray(T_right_left, dtype=np.float64).round(9).tolist(),
        "obstacles": list(obstacles),
    }
    digest = hashlib.sha256(json.dumps(body, sort_keys=True).encode()).hexdigest()
    if capsules.fingerprint != model["sha256"]:
        raise ValueError(
            f"{col.CAPSULE_FILE.name} was fitted to another model ({capsules.fingerprint[:12]} vs "
            f"{model['sha256'][:12]}); refit with tools/fr3/dual_fr3_collision.py"
        )
    return {
        "schema": SCHEMA,
        "fingerprint": digest,
        "model": model,
        "T_right_left": body["T_right_left"],
        "placement": dict(placement),
        "obstacles": list(obstacles),
        "obstacle_source": dict(obstacle_source or {}),
        "unmodelled": list(UNMODELLED),
    }


# ------------------------------------------------------------------ MuJoCo ---


def _quat_wxyz(R: np.ndarray) -> list[float]:
    x, y, z, w = wp._quat_from_matrix(np.asarray(R, dtype=np.float64).tolist())
    return [w, x, y, z]


def build_spec(
    record: Mapping[str, Any], *, xml_path: Path | str = col.FR3_MJCF, show_capsules: bool = False
):
    """The MjSpec of ``record``: both arms with their grippers, the obstacles as geoms (collision group 3),
    and optionally the planning capsules (group 4, not colliding)."""
    import mujoco

    world = mujoco.MjSpec()
    world.compiler.degree = False
    world.visual.global_.offwidth = 1920
    world.visual.global_.offheight = 1080
    T_right_left = np.asarray(record["T_right_left"], dtype=np.float64)
    placements = {"right": np.eye(4), "left": T_right_left}
    for arm, T in placements.items():
        child = mujoco.MjSpec.from_file(str(xml_path))
        frame = world.worldbody.add_frame(pos=T[:3, 3].tolist(), quat=_quat_wxyz(T[:3, :3]))
        frame.attach_body(child.body("base"), ARM_PREFIX[arm], "")
    # Every arm mesh collidable, so the compiler builds its convex hull: on a visual-only mesh (links 3-7 have
    # no collision geom of their own) MuJoCo's distance runs on raw vertices and can come back 0.0 for two
    # meshes 140 mm apart. Nothing here is stepped, so the contacts this enables never act.
    for geom in world.geoms:
        if geom.type == mujoco.mjtGeom.mjGEOM_MESH:
            geom.contype, geom.conaffinity = 1, 1
    world.worldbody.add_light(pos=[0.4, 0.4, 2.5], dir=[0.0, 0.0, -1.0], diffuse=[0.8, 0.8, 0.8])
    for ob in record["obstacles"]:
        rgba = [0.55, 0.45, 0.35, 0.35] if ob["kind"] == "halfspace" else [0.9, 0.6, 0.1, 0.6]
        if ob["kind"] == "halfspace":
            normal = np.asarray(ob["normal"], dtype=np.float64)
            normal /= np.linalg.norm(normal)
            z = np.array([0.0, 0.0, 1.0])
            axis = np.cross(z, normal)
            angle = math.atan2(np.linalg.norm(axis), float(z @ normal))
            R = np.eye(3) if np.linalg.norm(axis) < 1e-12 else _axis_angle(axis / np.linalg.norm(axis), angle)
            world.worldbody.add_geom(
                name=f"env/{ob['name']}",
                type=mujoco.mjtGeom.mjGEOM_PLANE,
                size=[1.5, 1.5, 0.01],
                pos=(normal * float(ob["offset"])).tolist(),
                quat=_quat_wxyz(R),
                rgba=rgba,
                group=3,
            )
        elif ob["kind"] == "sphere":
            world.worldbody.add_geom(
                name=f"env/{ob['name']}", type=mujoco.mjtGeom.mjGEOM_SPHERE, size=[float(ob["r"]), 0, 0],
                pos=list(ob["centre"]), rgba=rgba, group=3,
            )  # fmt: skip
        else:
            world.worldbody.add_geom(
                name=f"env/{ob['name']}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[float(ob["r"]), 0, 0],
                fromto=[*ob["a"], *ob["b"]], rgba=rgba, group=3,
            )  # fmt: skip
    if show_capsules:
        capsules = col.load()
        for arm in placements:
            for k, name in enumerate(capsules.names):
                frame = col.FRAMES[capsules.frame_index[k]]
                body = "base" if frame == "link0" else ("fr3_link8" if frame == "hand" else f"fr3_{frame}")
                world.body(ARM_PREFIX[arm] + body).add_geom(
                    name=f"{arm}/capsule/{name}", type=mujoco.mjtGeom.mjGEOM_CAPSULE, size=[float(capsules.r[k]), 0, 0],
                    fromto=[*capsules.a[k], *capsules.b[k]], rgba=[0.2, 0.6, 1.0, 0.25], group=4,
                    contype=0, conaffinity=0,
                )  # fmt: skip
    return world


def _axis_angle(axis: np.ndarray, angle: float) -> np.ndarray:
    K = np.array([[0.0, -axis[2], axis[1]], [axis[2], 0.0, -axis[0]], [-axis[1], axis[0], 0.0]])
    return np.eye(3) + math.sin(angle) * K + (1.0 - math.cos(angle)) * (K @ K)


class DualScene:
    """A compiled scene: set both arms' joints (and gripper openings) and read distances or render."""

    def __init__(self, record: Mapping[str, Any], *, show_capsules: bool = False):
        import mujoco

        self.mujoco = mujoco
        self.record = record
        self.spec = build_spec(record, show_capsules=show_capsules)
        self.model = self.spec.compile()
        self.data = mujoco.MjData(self.model)
        self.qpos = {
            arm: [self.model.joint(f"{prefix}fr3_joint{i}").qposadr[0] for i in range(1, 8)]
            for arm, prefix in ARM_PREFIX.items()
        }
        self.gripper = {
            arm: [
                (self.model.jnt_qposadr[j], *self.model.jnt_range[j])
                for j in range(self.model.njnt)
                if self.model.joint(j).name.startswith(prefix) and "gripper" in self.model.joint(j).name
            ]
            for arm, prefix in ARM_PREFIX.items()
        }
        # one geom per (body, mesh) of each arm: the visual and collision copies are the same hull
        self.arm_geoms: dict[str, list[int]] = {arm: [] for arm in ARM_PREFIX}
        seen: set[tuple[int, int, int]] = set()
        for g in range(self.model.ngeom):
            body_name = self.model.body(self.model.geom_bodyid[g]).name
            arm = next((a for a, p in ARM_PREFIX.items() if body_name.startswith(p)), None)
            gtype = self.model.geom_type[g]
            if arm is None or self.model.geom_group[g] == 4:
                continue
            if gtype not in (mujoco.mjtGeom.mjGEOM_MESH, mujoco.mjtGeom.mjGEOM_SPHERE):
                continue
            key = (int(self.model.geom_bodyid[g]), int(gtype), int(self.model.geom_dataid[g]))
            if key in seen:
                continue
            seen.add(key)
            self.arm_geoms[arm].append(g)
        self.env_geoms = [
            g for g in range(self.model.ngeom) if (self.model.geom(g).name or "").startswith("env/")
        ]

    def set(self, q_right: np.ndarray, q_left: np.ndarray, opening: tuple[float, float] = (1.0, 1.0)) -> None:
        self.data.qpos[:] = 0.0
        for arm, q, o in (("right", q_right, opening[0]), ("left", q_left, opening[1])):
            self.data.qpos[self.qpos[arm]] = np.asarray(q, dtype=np.float64)
            for adr, low, high in self.gripper[arm]:
                self.data.qpos[adr] = low + float(np.clip(o, 0.0, 1.0)) * (high - low)
        self.mujoco.mj_kinematics(self.model, self.data)

    def arm_distance(self, distmax: float = 0.3) -> tuple[float, tuple[str, str], int]:
        """Smallest hull distance between a right and a left geom (``distmax`` when none is nearer), the bodies,
        and how many pairs MuJoCo got wrong.

        ``mj_geomDistance`` (MuJoCo 3.6) returns 0.0 for some mesh pairs well apart -- witness points 120-190
        mm apart, 140 mm by brute force over the vertices -- with or without hulls. A result whose witness
        points disagree with it is not a distance; it is left out and counted.
        """
        best, pair, wrong = distmax, ("", ""), 0
        fromto = np.zeros(6)
        for g1 in self.arm_geoms["right"]:
            for g2 in self.arm_geoms["left"]:
                d = self.mujoco.mj_geomDistance(self.model, self.data, g1, g2, distmax, fromto)
                if d >= distmax:
                    continue
                if abs(float(np.linalg.norm(fromto[:3] - fromto[3:])) - abs(d)) > 1e-3:
                    wrong += 1
                    continue
                if d < best:
                    best = d
                    pair = (
                        self.model.body(self.model.geom_bodyid[g1]).name,
                        self.model.body(self.model.geom_bodyid[g2]).name,
                    )
        return float(best), pair, wrong

    def render(
        self,
        q_right: np.ndarray,
        q_left: np.ndarray,
        video_path: Path | str,
        *,
        fps: float,
        openings: np.ndarray | None = None,
        width: int = 960,
        height: int = 540,
    ) -> Path:
        """Both arms through their joint paths in one view, as an H.264 MP4."""
        mujoco = self.mujoco
        width -= width % 2
        height -= height % 2
        video_path = Path(video_path)
        video_path.parent.mkdir(parents=True, exist_ok=True)
        renderer = mujoco.Renderer(self.model, height=height, width=width)
        camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(camera)
        mid = np.asarray(self.record["T_right_left"])[:3, 3] / 2.0
        camera.lookat[:] = [0.35 + mid[0], mid[1], 0.35]
        camera.distance = 2.2
        camera.azimuth = 180.0
        camera.elevation = -25.0
        options = mujoco.MjvOption()
        options.geomgroup[:] = [1, 1, 1, 1, 1, 0]
        ffmpeg = subprocess.Popen(
            [
                "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s", f"{width}x{height}",
                "-r", str(max(float(fps), 1.0)), "-i", "-", "-an",
                "-c:v", "libx264", "-preset", "veryfast", "-crf", "23",
                "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video_path),
            ],
            stdin=subprocess.PIPE,
        )  # fmt: skip
        try:
            for k in range(len(q_right)):
                opening = (1.0, 1.0) if openings is None else (float(openings[k, 0]), float(openings[k, 1]))
                self.set(q_right[k], q_left[k], opening)
                renderer.update_scene(self.data, camera=camera, scene_option=options)
                ffmpeg.stdin.write(np.asarray(renderer.render(), dtype=np.uint8).tobytes())
        finally:
            if ffmpeg.stdin is not None:
                ffmpeg.stdin.close()
            code = ffmpeg.wait()
            renderer.close()
        if code != 0:
            raise RuntimeError(f"ffmpeg exited with code {code}")
        return video_path


def mesh_check(
    scene: DualScene,
    q_right: np.ndarray,
    q_left: np.ndarray,
    capsule_gaps: np.ndarray,
    *,
    stride: int = 1,
    within_m: float = 0.25,
) -> dict[str, Any]:
    """MuJoCo's hull distance between the arms on the frames where the capsules come within ``within_m``
    (beyond that the hulls are farther still: the capsules enclose them). The capsule gap must never exceed
    the hull distance where the hulls are apart -- it cannot, the capsules contain every vertex (tested) -- and
    ``capsule_excess_mm`` > 0 would mean exactly that. What this adds is how far apart the real shapes are."""
    frames = [k for k in range(0, len(q_right), max(1, stride)) if capsule_gaps[k] < within_m]
    rows, unreliable = [], 0
    for k in frames:
        scene.set(q_right[k], q_left[k])
        d, pair, wrong = scene.arm_distance(distmax=within_m + 0.05)
        rows.append((k, d, pair, float(capsule_gaps[k])))
        unreliable += wrong
    if not rows:
        return {"frames_checked": 0, "min_mesh_gap_mm": None, "capsule_excess_mm": None, "within_m": within_m}
    k, d, pair, _ = min(rows, key=lambda row: row[1])
    # where the hulls overlap both numbers are penetration measures of different shapes: not comparable
    apart = [row[3] - row[1] for row in rows if row[1] > 0.0]
    excess = max(apart) if apart else None
    return {
        "frames_checked": len(rows),
        "within_m": within_m,
        "min_mesh_gap_mm": d * 1e3,
        "min_mesh_gap_frame": k,
        "min_mesh_gap_bodies": list(pair),
        "capsule_excess_mm": None if excess is None else excess * 1e3,
        "mujoco_pairs_left_out": unreliable,
    }
