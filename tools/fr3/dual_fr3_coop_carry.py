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
A scripted two-hand carry of one tray: both hands on its handles, lift, carry and turn, put down (roadmap D6).

Not a demonstration -- the tray's path is keyframed here. Both TCP targets come from the one tray pose and a
fixed tray->TCP grasp per hand, so T_LR is rigid while the tray is held: the input the closed-chain budget
(``dual_fr3_closed_chain``) and the internal-force guard of the runtime need, and that no recording has yet
(A5's pairs are two independent hands). It goes through C3 as a recording would (``dual_fr3_retarget``: one
joint path for both arms, the arm-arm and environment clearances, the joint limits) and is written as a held
plan -- each arm's joints on the command clock and which frames both hands hold the tray -- that
``dual_fr3_replay_runtime --plan`` runs on fake arms, or on the real ones without the grippers (the held
frames then measure how far the two arms' execution bends T_LR, with nothing to squeeze).

The tray (``Tray``) rests on the table of C2's scene, in front of the gap between the arms, in the right base.
It is the one ``dual_fr3_compliant_handle`` draws: a plate with a handle bar each side. The grip opening is
the one that puts the gripper's pads on a handle bar (from the MJCF), and the TCP sits that far below the bar.
``--render`` adds the tray to the scene as a mocap body and writes a video (the cameras stay obstacles in the
retarget but are not drawn); ``--meshes`` draws it from the handle's CAD instead of boxes.
"""

# ruff: noqa: N802, N803, N806  (matrix names: R, T follow the maths)
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import (  # noqa: E402
    dual_fr3_closed_chain as cc,
    dual_fr3_scene as sc,
    fr3_kinematics as kin,
)

TOOL_FRAME = "corenetic_gripper_ee"
#: the ready pose's TCP orientation: z down, the fingers closing along world y -- across a bar along x
R_DOWN = np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, -1.0]])


@dataclass(frozen=True)
class Tray:
    """A tray with a handle bar along x on each y side, in metres (half sizes): ``dual_fr3_compliant_handle``'s
    plate and grip bars (the bar centres 0.105 m over the table, as the first plan's 0.04 + 0.065)."""

    body_half: tuple[float, float, float] = (0.10, 0.22, 0.0015)
    bar_half: tuple[float, float, float] = (0.055, 0.0175, 0.016)
    #: bar centre from the tray centre, sideways and up
    bar_y: float = 0.195
    bar_z: float = 0.1035


@dataclass(frozen=True)
class Carry:
    """The scripted motion: tray start and goal (x, y, yaw deg) in the right base, the lift, the approach
    height, and each phase's duration in seconds."""

    spacing_m: float = 0.60
    start: tuple[float, float, float] = (0.62, 0.30, 0.0)
    goal: tuple[float, float, float] = (0.56, 0.23, 22.0)
    lift_m: float = 0.15
    hover_m: float = 0.12
    fps: float = 20.0
    approach_s: float = 2.5
    descend_s: float = 1.5
    close_s: float = 1.0
    lift_s: float = 2.0
    carry_s: float = 3.0
    settle_s: float = 0.6
    lower_s: float = 2.0
    open_s: float = 1.0
    retreat_s: float = 1.5
    home_s: float = 2.5


def tray_pose(x: float, y: float, z: float, yaw: float) -> np.ndarray:
    c, s = np.cos(yaw), np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = [x, y, z]
    return T


def grasps(tray: Tray, pad_drop_m: float) -> dict[str, np.ndarray]:
    """Tray -> TCP per hand: on its bar (right hand on the -y bar), the TCP ``pad_drop_m`` below the pads."""
    out = {}
    for arm, sign in (("right", -1.0), ("left", 1.0)):
        T = np.eye(4)
        T[:3, :3] = R_DOWN
        T[:3, 3] = [0.0, sign * tray.bar_y, tray.bar_z - pad_drop_m]
        out[arm] = T
    return out


def grip(scene: sc.DualScene, bar_width_m: float, clearance_m: float = 0.001) -> tuple[float, float]:
    """``(opening, pad drop)``: the gripper opening whose contact pads are ``bar_width_m`` + clearance apart,
    and how far above the TCP the pads then are (the MJCF's fingers, on the right arm at the ready pose)."""
    tool = kin.urdf_fixed_transform(TOOL_FRAME)
    F_inv = np.linalg.inv(kin.NOMINAL.fk(kin.Q_READY[None])[0])

    def pads(o: float) -> tuple[float, float]:
        scene.set(kin.Q_READY, kin.Q_READY, (o, o))
        p = [
            (F_inv @ np.r_[scene.data.xpos[scene.model.body(f"right/link_gripper_contact_{s}").id], 1.0])[:3]
            for s in ("left", "right")
        ]
        return float(np.linalg.norm(p[0] - p[1])), float((p[0][2] + p[1][2]) / 2.0)

    grid = np.linspace(0.0, 1.0, 201)
    widths = np.array([pads(o)[0] for o in grid])
    opening = float(np.interp(bar_width_m + clearance_m, widths, grid))
    return opening, float(tool[2, 3] - pads(opening)[1])


def _minjerk(s: np.ndarray | float) -> np.ndarray:
    s = np.clip(s, 0.0, 1.0)
    return 10 * s**3 - 15 * s**4 + 6 * s**5


def _expm(w: np.ndarray) -> np.ndarray:
    return cc._rotation(np.asarray(w)[None])[0]


def carry_targets(carry: Carry, tray: Tray, opening: float, pad_drop_m: float) -> dict[str, Any]:
    """Per frame: each hand's TCP target in its own base, the tray pose (x, y, z, yaw rad) in the right base,
    the gripper opening, and ``held`` -- both hands closed on the tray (the frames the chain is closed)."""
    T_RL = np.eye(4)
    T_RL[1, 3] = carry.spacing_m
    tool = kin.urdf_fixed_transform(TOOL_FRAME)
    G = grasps(tray, pad_drop_m)
    ready = {"right": kin.NOMINAL.fk(kin.Q_READY[None])[0] @ tool}
    ready["left"] = T_RL @ ready["right"]
    z = sc.TABLE_TOP_M + tray.body_half[2]  # resting on the table
    (x0, y0, w0), (x1, y1, w1) = carry.start, carry.goal
    at, to = (x0, y0, z, np.radians(w0)), (x1, y1, z, np.radians(w1))
    up0, up1 = (x0, y0, z + carry.lift_m, np.radians(w0)), (x1, y1, z + carry.lift_m, np.radians(w1))
    # (duration, tray pose, hand above its grasp: metres or "ready", opening)
    keys = [
        (0.0, at, "ready", 1.0),
        (carry.approach_s, at, carry.hover_m, 1.0),
        (carry.descend_s, at, 0.0, 1.0),
        (carry.close_s, at, 0.0, opening),
        (carry.lift_s, up0, 0.0, opening),
        (carry.carry_s, up1, 0.0, opening),
        (carry.settle_s, up1, 0.0, opening),
        (carry.lower_s, to, 0.0, opening),
        (carry.open_s, to, 0.0, 1.0),
        (carry.retreat_s, to, carry.hover_m, 1.0),
        (carry.home_s, to, "ready", 1.0),
    ]

    def hand(arm: str, pose: tuple, above: float | str) -> np.ndarray:
        if above == "ready":
            return ready[arm]
        T = tray_pose(*pose) @ G[arm]
        T[2, 3] += above
        return T

    frames: dict[str, list[np.ndarray]] = {"right": [], "left": []}
    poses, openings = [], []
    for (_, p0, a0, o0), (d1, p1, a1, o1) in zip(keys[:-1], keys[1:], strict=True):
        n = int(round(d1 * carry.fps))
        for k in range(n):
            s = float(_minjerk((k + 1) / n))
            pose = (1 - s) * np.asarray(p0) + s * np.asarray(p1)
            for arm in frames:
                if a0 != "ready" and a1 != "ready":
                    # on the tray: the hand from the interpolated tray pose, so that it turns on the tray's
                    # arc -- interpolating each hand's own position cuts the chord, and a 22 deg turn then
                    # pulls two hands 0.39 m apart 7 mm together mid-way
                    frames[arm].append(hand(arm, tuple(pose), (1 - s) * a0 + s * a1))
                    continue
                A, B = hand(arm, p0, a0), hand(arm, p1, a1)
                T = np.eye(4)
                T[:3, 3] = (1 - s) * A[:3, 3] + s * B[:3, 3]
                T[:3, :3] = _expm(kin._rotvec((B[:3, :3] @ A[:3, :3].T)[None])[0] * s) @ A[:3, :3]
                frames[arm].append(T)
            poses.append(pose)
            openings.append((1 - s) * o0 + s * o1)
    openings = np.asarray(openings)
    return {
        "targets": {
            "right": np.asarray(frames["right"]),
            "left": np.einsum("ij,njk->nik", np.linalg.inv(T_RL), np.asarray(frames["left"])),
        },
        "tray": np.asarray(poses),
        "opening": openings,
        "held": openings <= opening + 1e-9,
        "T_right_left": T_RL,
        "tool": tool,
    }


def plan(carry: Carry, tray: Tray, *, d_safe: dict[str, Any] | None = None) -> dict[str, Any]:
    """The carry through C3, on the command clock: ``(held plan arrays, summary)``."""
    from tools.fr3 import dual_fr3_collision as col, dual_fr3_retarget as rg

    T_RL = np.eye(4)
    T_RL[1, 3] = carry.spacing_m
    cams, src = sc.camera_obstacles()
    obstacles = [*sc.environment_obstacles(), *cams]
    placement = {"source": "assumed", "note": f"{carry.spacing_m} m along +y, parallel; B0 nominal"}
    record = sc.scene_record(T_RL, placement, obstacles, src)
    scene = sc.DualScene(record)
    opening, pad_drop = grip(scene, 2 * tray.bar_half[1])
    path = carry_targets(carry, tray, opening, pad_drop)
    n = len(path["held"])
    d = d_safe or rg.derive_d_safe(None)
    result = rg.retarget(
        path["targets"],
        {a: np.ones(n, bool) for a in path["targets"]},
        tool=path["tool"],
        T_right_left=T_RL,
        obstacles=obstacles,
        t_raw=np.arange(n) / carry.fps,
        speed=1.0,
        settings=rg.Settings(d_safe_m=d["d_safe_m"], d_env_m=d["d_env_m"]),
    )
    arrays = result.pop("arrays")
    after = result["retarget"]
    gaps = col.clearance(col.load(), arrays["q"][:, :7], arrays["q"][:, 7:], T_RL)["arm_arm_m"]
    mesh = sc.mesh_check(scene, arrays["q"][:, :7], arrays["q"][:, 7:], gaps, stride=2)
    q_cmd, index = rg.resample(arrays["q"], arrays["t_ref"], 1.0 / carry.fps)
    frames = np.arange(n)
    # held on the command clock only where both path frames around it are held
    held = (
        path["held"][np.floor(index).astype(int)]
        & path["held"][np.minimum(np.ceil(index).astype(int), n - 1)]
    )
    arrays_out = {
        "q_right": q_cmd[:, :7],
        "q_left": q_cmd[:, 7:],
        "held": held,
        "T_right_left": T_RL,
        "tool": path["tool"],
        "fps": np.float64(carry.fps),
        "opening": np.interp(index, frames, path["opening"]),
        "tray": np.stack([np.interp(index, frames, path["tray"][:, j]) for j in range(4)], axis=1),
    }
    summary = {
        "what": "scripted two-hand tray carry (not a demonstration; the tray is not simulated)",
        "carry": asdict(carry),
        "tray": asdict(tray),
        "tool_frame": TOOL_FRAME,
        "grip_opening": round(opening, 4),
        "pad_drop_mm": round(pad_drop * 1e3, 2),
        "executable": result["executable"],
        "reasons": result["reasons"],
        "d_safe_mm": round(d["d_safe_m"] * 1e3, 1),
        "change_position_mm": {a: after["arms"][a]["change_position_mm"] for a in ("right", "left")},
        "change_rotation_deg": {a: after["arms"][a]["change_rotation_deg"] for a in ("right", "left")},
        "t_lr_change_position_mm": after["t_lr_change_position_mm"],
        "min_arm_arm_gap_mm": after["min_arm_arm_gap_mm"],
        "min_env_gap_mm": {a: after["arms"][a]["min_env_gap_mm"] for a in ("right", "left")},
        "mesh_min_gap_mm": mesh.get("min_mesh_gap_mm"),
        "timing": result["timing"],
        "frames": len(q_cmd),
        "held_frames": int(held.sum()),
        "held_segments": [list(s) for s in cc.held_segments(held)],
        "scene_fingerprint": record.get("fingerprint"),
    }
    return arrays_out, summary


def write_plan(path: Path, arrays: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, **arrays)


#: (stl path, rgba): tray parts in the tray frame of ``dual_fr3_compliant_handle`` (mm, origin on the table
#: under the plate centre)
Meshes = list[tuple[Path, list[float]]]


def _tray_scene(T_RL: np.ndarray, tray: Tray, meshes: Meshes | None = None) -> tuple[Any, int]:
    """C2's scene with the tray as a mocap body (drawn, not simulated): ``(scene, mocap id)``."""
    import mujoco

    record = sc.scene_record(T_RL, {"source": "assumed", "note": "render"}, sc.environment_obstacles())
    base_build = sc.build_spec
    yellow, blue = [0.96, 0.74, 0.18, 1.0], [0.32, 0.52, 0.82, 1.0]

    def with_tray(rec: dict[str, Any], **kw: Any) -> Any:
        spec = base_build(rec, **kw)
        body = spec.worldbody.add_body(name="tray", mocap=True)
        box = mujoco.mjtGeom.mjGEOM_BOX
        if meshes:
            for k, (path, rgba) in enumerate(meshes):
                spec.add_mesh(name=f"tray_mesh_{k}", file=str(Path(path).resolve()), scale=[1e-3] * 3)
                body.add_geom(
                    type=mujoco.mjtGeom.mjGEOM_MESH, meshname=f"tray_mesh_{k}", pos=[0, 0, -tray.body_half[2]],
                    rgba=rgba, contype=0, conaffinity=0,
                )  # fmt: skip
            return spec
        body.add_geom(
            type=box, size=list(tray.body_half), rgba=[0.78, 0.80, 0.83, 1.0], contype=0, conaffinity=0
        )
        # a block from the plate to each bar, where the cradle is
        bottom = tray.body_half[2]
        top = tray.bar_z - tray.bar_half[2]
        for sign in (-1.0, 1.0):
            body.add_geom(
                type=box, size=list(tray.bar_half), pos=[0, sign * tray.bar_y, tray.bar_z], rgba=yellow,
                contype=0, conaffinity=0,
            )  # fmt: skip
            body.add_geom(
                type=box, size=[tray.bar_half[0] + 0.014, 0.025, (top - bottom) / 2],
                pos=[0, sign * tray.bar_y, (top + bottom) / 2], rgba=blue, contype=0, conaffinity=0,
            )  # fmt: skip
        return spec

    sc.build_spec = with_tray
    try:
        scene = sc.DualScene(record)
    finally:
        sc.build_spec = base_build
    return scene, int(scene.model.body("tray").mocapid[0])


def _pose_frame(scene: Any, mocap: int, arrays: dict[str, Any], k: int) -> None:
    import mujoco

    o = float(arrays["opening"][k])
    scene.set(arrays["q_right"][k], arrays["q_left"][k], (o, o))
    x, y, z, yaw = arrays["tray"][k]
    scene.data.mocap_pos[mocap] = [x, y, z]
    scene.data.mocap_quat[mocap] = [np.cos(yaw / 2), 0.0, 0.0, np.sin(yaw / 2)]
    mujoco.mj_kinematics(scene.model, scene.data)


def _camera(lookat: Any, distance: float, elevation: float, azimuth: float) -> Any:
    import mujoco

    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.lookat[:] = lookat
    camera.distance, camera.elevation, camera.azimuth = distance, elevation, azimuth
    return camera


def _options() -> Any:
    import mujoco

    options = mujoco.MjvOption()
    options.geomgroup[:] = [1, 1, 1, 1, 1, 0]
    return options


def render(
    arrays: dict[str, Any],
    tray: Tray,
    video: Path,
    *,
    width: int = 1280,
    height: int = 720,
    meshes: Meshes | None = None,
) -> Path:
    """The held plan with the tray following its scripted pose."""
    import mujoco

    T_RL = np.asarray(arrays["T_right_left"])
    scene, mocap = _tray_scene(T_RL, tray, meshes)
    renderer = mujoco.Renderer(scene.model, height=height, width=width)
    camera = _camera([0.45, T_RL[1, 3] / 2.0, 0.28], 1.55, -26.0, 150.0)
    options = _options()
    fps = float(arrays["fps"])
    video.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg = subprocess.Popen(
        ["ffmpeg", "-hide_banner", "-loglevel", "error", "-y", "-f", "rawvideo", "-pix_fmt", "rgb24",
         "-s", f"{width}x{height}", "-r", f"{fps:g}", "-i", "-", "-an", "-c:v", "libx264", "-preset", "veryfast",
         "-crf", "20", "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(video)],
        stdin=subprocess.PIPE,
    )  # fmt: skip
    n = len(arrays["q_right"])
    for k in range(n):
        _pose_frame(scene, mocap, arrays, k)
        camera.azimuth = 150.0 + 60.0 * k / max(n - 1, 1)  # a slow orbit
        renderer.update_scene(scene.data, camera=camera, scene_option=options)
        ffmpeg.stdin.write(np.asarray(renderer.render(), dtype=np.uint8).tobytes())
    ffmpeg.stdin.close()
    renderer.close()
    if ffmpeg.wait() != 0:
        raise RuntimeError("ffmpeg failed")
    return video


def still(
    arrays: dict[str, Any],
    tray: Tray,
    frame: int,
    png: Path,
    *,
    view: tuple[Any, float, float, float],
    width: int = 1600,
    height: int = 1000,
    meshes: Meshes | None = None,
) -> Path:
    """One frame of the held plan as a PNG; ``view`` is the camera's (lookat, distance, elevation, azimuth)."""
    import mujoco
    from PIL import Image

    scene, mocap = _tray_scene(np.asarray(arrays["T_right_left"]), tray, meshes)
    _pose_frame(scene, mocap, arrays, frame)
    renderer = mujoco.Renderer(scene.model, height=height, width=width)
    renderer.update_scene(scene.data, camera=_camera(*view), scene_option=_options())
    png.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(np.asarray(renderer.render(), dtype=np.uint8)).save(png)
    renderer.close()
    return png


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Scripted two-hand tray carry through C3, as a held plan (D6)"
    )
    parser.add_argument("--out", type=Path, default=REPO / "outputs" / "fr3" / "dual_fr3_coop_carry")
    parser.add_argument("--spacing", type=float, default=Carry.spacing_m, help="left base along +y, m")
    parser.add_argument("--render", action="store_true", help="also write the video")
    parser.add_argument(
        "--meshes",
        type=Path,
        default=None,
        help="draw the tray from dual_fr3_compliant_handle's CAD (its --out)",
    )
    args = parser.parse_args(argv)
    carry = Carry(
        spacing_m=args.spacing,
        start=(0.62, args.spacing / 2, 0.0),
        goal=(0.56, args.spacing / 2 - 0.07, 22.0),
    )
    tray = Tray()
    arrays, summary = plan(carry, tray)
    plan_path = args.out / "coop_carry_plan.npz"
    write_plan(plan_path, arrays)
    budget = cc.budget(cc.read_plan(plan_path))
    budget["plan"] = str(plan_path)
    (args.out / "coop_carry_plan_closed_chain.json").write_text(
        json.dumps(budget, indent=2) + "\n", encoding="utf-8"
    )
    summary["plan"] = str(plan_path)
    (args.out / "coop_carry.json").write_text(
        json.dumps(summary, indent=2, default=float) + "\n", encoding="utf-8"
    )
    print(json.dumps({k: summary[k] for k in ("executable", "reasons", "t_lr_change_position_mm",
                                              "min_arm_arm_gap_mm", "mesh_min_gap_mm", "frames", "held_frames")},
                     indent=2, default=float))  # fmt: skip
    print("\n".join(cc._summary_lines(budget)))
    if args.render:
        meshes = None
        if args.meshes is not None:
            from tools.fr3 import dual_fr3_compliant_handle as ch

            meshes = ch.mesh_list(args.meshes)
        print(render(arrays, tray, args.out / "coop_carry.mp4", meshes=meshes))
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
