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
Can two FR3s replay this episode, and why not: the per-episode verdict a dataset carries (roadmap C4).

For every episode of a two-hand session: the replay mode's targets (C2, ``dual_fr3_replay_modes``), the joint
retarget against the scene (C3, ``dual_fr3_retarget``, in C1's scene), MuJoCo's exact-mesh distance between
the arms as a check on the capsules (``dual_fr3_scene.mesh_check``), and the verdict:

* ``meta/dual_fr3_executable.json`` -- per mode, per episode: ``dual_fr3_executable``, the reasons, the C3
  figures (retarget change, T_LR change, gaps, margins, timing), the scene fingerprint and the edges the mode
  rests on. A file of its own: ``info.json`` and the labels are not touched, so nothing the exporter or the
  label-chain guard reads changes.
* ``derived/dual_fr3_retarget/episode_NNNNNN.<mode>.{json,npz}`` -- the full C3 report, and the reference
  joint paths with their raw and reference time axes, which a replay streams (``plans_from_retarget``).

A verdict is only as good as its scene: until B0/B3 measure the left base it rests on an assumed placement,
and the record says so (``placement.assumed``, the 20 mm sigma that widens d_safe).
"""

# ruff: noqa: N802, N803, N806  (matrix names: R, T, J follow the maths)
from __future__ import annotations

import argparse
import json
import sys
from collections.abc import Mapping, Sequence
from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import (  # noqa: E402
    dual_fr3_replay_modes as modes,
    dual_fr3_retarget as rg,
    dual_fr3_scene as sc,
    fr3_kinematics as kin,
)
from tools.thor.gmsl2 import bimanual_qc, embodiment as em, world_provenance as wp  # noqa: E402

SCHEMA = "dual_fr3_executable/v1"
META_FILE = Path("meta") / "dual_fr3_executable.json"
RETARGET_DIR = Path("derived") / "dual_fr3_retarget"
#: The Thor replay YAML's replay_fps: 60 fps data played three times slower.
DEFAULT_REPLAY_FPS = 20.0


def retarget_paths(session_dir: Path, episode: int, mode: str) -> tuple[Path, Path]:
    stem = f"episode_{int(episode):06d}.{mode}"
    return session_dir / RETARGET_DIR / f"{stem}.json", session_dir / RETARGET_DIR / f"{stem}.npz"


def episodes_of(session_dir: Path) -> list[int]:
    path = session_dir / "derived" / bimanual_qc.SIDECAR / "state_action.right.csv"
    return sorted(bimanual_qc.read_hand(path))


def scene_for(
    plan: modes.ModePlan, obstacles: Sequence[Mapping[str, Any]], obstacle_source: Mapping[str, Any]
) -> dict[str, Any]:
    placement = {k: v for k, v in plan.placement.items() if k != "reason"}
    placement["source"] = "assumed" if plan.placement.get("assumed") else "graph"
    return sc.scene_record(plan.T_right_left, placement, obstacles, obstacle_source)


def evaluate_episode(
    session_dir: Path,
    episode: int,
    *,
    description: Mapping[str, Any],
    graph: Mapping[str, Any],
    mode: str,
    align: str,
    obstacles: Sequence[Mapping[str, Any]],
    obstacle_source: Mapping[str, Any],
    settings: rg.Settings,
    replay_fps: float,
    mesh: bool = True,
    video: bool = False,
) -> dict[str, Any]:
    """One row of the meta file; writes the episode's C3 report and arrays."""
    ep = modes.read_episode(session_dir, episode)
    plan = modes.plan(mode, ep, description, graph, align=align)
    row: dict[str, Any] = {"frames": len(ep.frames), "mode": plan.record()}
    if not plan.ok:
        return {**row, "dual_fr3_executable": False, "reasons": [f"{mode} replay refused: {plan.reason}"]}
    scene = scene_for(plan, obstacles, obstacle_source)
    d = rg.derive_d_safe(plan.placement.get("sigma_mm"))
    g = replace(settings, d_safe_m=d["d_safe_m"], d_env_m=d["d_env_m"])
    tools = {
        arm: kin.urdf_fixed_transform(description["arms"][arm]["tcp_frame"]) for arm in ("right", "left")
    }
    if not np.allclose(tools["right"], tools["left"]):
        raise ValueError("the arms carry different tools; the retarget takes one")
    result = rg.retarget(
        plan.targets,
        plan.seen,
        tool=tools["right"],
        T_right_left=plan.T_right_left,
        obstacles=obstacles,
        t_raw=ep.t,
        speed=replay_fps / ep.fps,
        settings=g,
    )
    arrays = result.pop("arrays")
    result.update(
        session=str(session_dir),
        episode=int(episode),
        frames=ep.frames.tolist(),
        mode=plan.record(),
        scene_fingerprint=scene["fingerprint"],
        d_safe=d,
        replay_fps=replay_fps,
        tool_frame=description["arms"]["right"]["tcp_frame"],
    )
    reasons = list(result["reasons"])
    if mesh:
        scene_model = sc.DualScene(scene)
        from tools.fr3 import dual_fr3_collision as col

        gaps = col.clearance(col.load(), arrays["q"][:, :7], arrays["q"][:, 7:], plan.T_right_left)[
            "arm_arm_m"
        ]
        check = sc.mesh_check(scene_model, arrays["q"][:, :7], arrays["q"][:, 7:], gaps, stride=2)
        result["mesh_check"] = check
        if check["capsule_excess_mm"] is not None and check["capsule_excess_mm"] > 0.5:
            reasons.append(
                f"the capsule model is not conservative here: {check['capsule_excess_mm']:.1f} mm above the mesh gap"
            )
        if video:
            path = session_dir / RETARGET_DIR / f"episode_{int(episode):06d}.{mode}.mp4"
            q_cmd, _ = rg.resample(arrays["q"], arrays["t_ref"], 1.0 / replay_fps)
            scene_model.render(q_cmd[:, :7], q_cmd[:, 7:], path, fps=replay_fps)
            result["video"] = str(path.relative_to(session_dir))
    result["executable"] = not reasons
    result["reasons"] = reasons
    report_path, arrays_path = retarget_paths(session_dir, episode, mode)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(result, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    np.savez_compressed(
        arrays_path, **arrays, targets_right=plan.targets["right"], targets_left=plan.targets["left"]
    )
    after = result["retarget"]
    return {
        **row,
        "dual_fr3_executable": not reasons,
        "reasons": reasons,
        "scene_fingerprint": scene["fingerprint"],
        "placement_assumed": bool(plan.placement.get("assumed")),
        "d_safe_mm": round(d["d_safe_m"] * 1e3, 1),
        "retarget_change_position_mm": {s: after["arms"][s]["change_position_mm"] for s in ("right", "left")},
        "retarget_change_rotation_deg": {
            s: after["arms"][s]["change_rotation_deg"] for s in ("right", "left")
        },
        "t_lr_change_position_mm": after["t_lr_change_position_mm"],
        "min_arm_arm_gap_mm": after["min_arm_arm_gap_mm"],
        "min_env_gap_mm": {s: after["arms"][s]["min_env_gap_mm"] for s in ("right", "left")},
        "min_limit_margin_rad": {s: after["arms"][s]["min_limit_margin_rad"] for s in ("right", "left")},
        "min_singular_m": {s: after["arms"][s]["min_singular_m"] for s in ("right", "left")},
        "frames_tracked_frame_by_frame": {s: result["start"][s]["frames_tracked"] for s in ("right", "left")},
        "reference_duration_s": result["timing"]["reference_duration_s"],
        "slowdown_vs_nominal": result["timing"]["slowdown_vs_nominal"],
        "mesh_min_gap_mm": (result.get("mesh_check") or {}).get("min_mesh_gap_mm"),
        "report": str(report_path.relative_to(session_dir)),
    }


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def evaluate_session(
    session_dir: Path,
    *,
    description: Mapping[str, Any],
    graph: Mapping[str, Any],
    mode: str,
    align: str = "identity",
    episodes: Sequence[int] | None = None,
    settings: rg.Settings | None = None,
    replay_fps: float = DEFAULT_REPLAY_FPS,
    obstacles: Sequence[Mapping[str, Any]] | None = None,
    obstacle_source: Mapping[str, Any] | None = None,
    repo_root: Path = REPO,
    mesh: bool = True,
    video: bool = False,
) -> dict[str, Any]:
    """Every episode's verdict for ``mode``, merged into the session's meta file (other modes kept)."""
    session_dir = Path(session_dir)
    if obstacles is None:
        cameras, obstacle_source = sc.camera_obstacles(repo_root, graph)
        obstacles = [sc.table_obstacle(), *cameras]
    settings = settings or rg.Settings()
    rows = {}
    for episode in episodes if episodes is not None else episodes_of(session_dir):
        try:
            rows[str(episode)] = evaluate_episode(
                session_dir,
                episode,
                description=description,
                graph=graph,
                mode=mode,
                align=align,
                obstacles=obstacles,
                obstacle_source=obstacle_source or {},
                settings=settings,
                replay_fps=replay_fps,
                mesh=mesh,
                video=video,
            )
        except (ValueError, KeyError) as exc:
            rows[str(episode)] = {"dual_fr3_executable": False, "reasons": [f"not evaluated: {exc}"]}
    path = session_dir / META_FILE
    existing = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}
    if existing.get("schema") not in (None, SCHEMA):
        raise ValueError(f"{path}: schema {existing.get('schema')!r}, expected {SCHEMA!r}")
    by_mode = dict(existing.get("modes") or {})
    previous = dict((by_mode.get(mode) or {}).get("episodes") or {})
    previous.update(rows)
    by_mode[mode] = {
        "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "embodiment": description.get("name"),
        "align": align,
        "replay_fps": replay_fps,
        "settings": asdict(settings),
        "episodes": dict(sorted(previous.items(), key=lambda kv: int(kv[0]))),
        "summary": {
            "episodes": len(previous),
            "executable": sum(1 for r in previous.values() if r.get("dual_fr3_executable")),
        },
    }
    out = {
        "schema": SCHEMA,
        "note": "Whether two FR3s can replay each episode (roadmap C4). Labels are not changed; the reference "
        "joint paths are in derived/dual_fr3_retarget/.",
        "modes": by_mode,
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, default=_jsonable) + "\n", encoding="utf-8")
    return out


def read_verdicts(session_dir: Path | str) -> dict[str, Any]:
    """The meta file, or ``{}``. Stdlib-only readers (the gateway) parse the same JSON."""
    path = Path(session_dir) / META_FILE
    return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else {}


def plans_from_retarget(session_dir: Path, episode: int, mode: str) -> dict[str, Any]:
    """What a replay of ``episode`` streams: per arm the reference joint path on the runtime's command clock,
    its TCP targets (FK of the path -- what was checked, not the raw labels), and the time axes."""
    report_path, arrays_path = retarget_paths(Path(session_dir), episode, mode)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    arrays = np.load(arrays_path)
    fps = float(report["replay_fps"])
    q_cmd, index = rg.resample(arrays["q"], arrays["t_ref"], 1.0 / fps)
    tool = kin.urdf_fixed_transform(report["tool_frame"])
    return {
        "report": report,
        "fps": fps,
        "q": {"right": q_cmd[:, :7], "left": q_cmd[:, 7:]},
        "targets": {
            "right": kin.NOMINAL.fk(q_cmd[:, :7]) @ tool,
            "left": kin.NOMINAL.fk(q_cmd[:, 7:]) @ tool,
        },
        "tool": tool,
        "T_right_left": np.asarray(report["mode"]["T_right_left"], dtype=np.float64),
        "path_index": index,
        "t_raw": np.interp(index, np.arange(len(arrays["t_raw"])), arrays["t_raw"]),
        "t_ref": np.arange(len(index)) / fps,
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Dual-FR3 executability of a two-hand session (roadmap C1-C4)"
    )
    parser.add_argument("session", type=Path)
    parser.add_argument("--mode", choices=modes.MODES, default="absolute")
    parser.add_argument("--align", choices=modes.ALIGNS, default="identity", help="relative mode only")
    parser.add_argument("--episode", type=int, action="append", help="default: every episode")
    parser.add_argument("--embodiment", type=Path, default=em.EMBODIMENT_DIR / "fr3_dual.json")
    parser.add_argument(
        "--assume-spacing",
        type=float,
        default=None,
        help="until B3: put the left base this far along +y of the right (parallel), as an assumed edge "
        "(sigma 20 mm); the left arm is treated as in service",
    )
    parser.add_argument("--replay-fps", type=float, default=DEFAULT_REPLAY_FPS)
    parser.add_argument("--no-mesh", action="store_true", help="skip MuJoCo's mesh-distance check")
    parser.add_argument("--video", action="store_true", help="render both arms in the scene")
    args = parser.parse_args(argv)
    description = em.load(args.embodiment)
    graph = wp.read_world_graph(REPO)
    if args.assume_spacing is not None:
        from tools.fr3 import dual_fr3_replay_runtime as rt

        T = np.eye(4)
        T[1, 3] = args.assume_spacing
        graph = rt.assumed_placement_graph(
            graph, T, f"left base {args.assume_spacing} m along +y of the right, parallel"
        )
        description = json.loads(json.dumps(description))
        for arm in description["arms"].values():
            arm["replay_config"] = arm.get("replay_config") or "(assumed in service for an offline check)"
    out = evaluate_session(
        args.session,
        description=description,
        graph=graph,
        mode=args.mode,
        align=args.align,
        episodes=args.episode,
        mesh=not args.no_mesh,
        video=args.video,
    )
    block = out["modes"][args.mode]
    for ep, row in block["episodes"].items():
        mark = "OK " if row.get("dual_fr3_executable") else "NO "
        detail = "; ".join(row.get("reasons") or []) or (
            f"change p95 R {row['retarget_change_position_mm']['right']['p95']:.2f} / "
            f"L {row['retarget_change_position_mm']['left']['p95']:.2f} mm, gap {row['min_arm_arm_gap_mm']:.0f} mm"
        )
        print(f"  ep {ep}: {mark}{detail}")
    print(
        f"{block['summary']['executable']}/{block['summary']['episodes']} executable ({args.mode}); wrote {args.session / META_FILE}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
