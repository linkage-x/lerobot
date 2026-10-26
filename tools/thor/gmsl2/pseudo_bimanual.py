#!/usr/bin/env python3
"""Stand-in two-hand recordings made from one-hand ones, until the cubes are back (roadmap A5 ①).

Roadmap docs/dual_fr3_bimanual_replay_roadmap.html, A5. Thor holds no two-hand data (A1: the left cube was
never detected after 10-03) and the cubes went for repair with the grippers, so A4 cannot record any. The
dual-arm work that only needs *a* pair of hand tracks -- the scene, the replay modes, the runtime and its
gates (C1-C4, D1/D3) -- takes a pair built here:

* the right hand is a recorded right-hand episode, as it is;
* the left hand is another recorded right-hand episode, mirrored through the plane halfway between the two
  bases (``y = spacing / 2`` in ``fr3_base``; the left base is assumed at ``+y`` of the right, parallel), so
  that it sits in front of the left arm as the original sat in front of the right one.

Both are re-expressed from their camera world into ``fr3_base`` through ``world_graph.json`` (the mirror
needs a frame that knows where the bases are) and paired frame by frame, on the right episode's clock.

What is real and what is not: each hand's speed, jitter, gaps and camera strata are recorded ones. Their
*coordination* is not -- the two hands never moved together, so ``T_LR`` has no physical meaning, and the
spacing is an assumption, not B0's. Hence ``not_for`` in the output's ``meta/info.json``: no benchmark
selection (A2), no placement decision (B0), no ``T_LR`` statistics, no training. The output looks like a
session (``derived/<sidecar>/state_action.<hand>.csv``) so ``bimanual_qc`` and the replay tools read it
unchanged; its directory name starts with ``pseudo_bimanual_`` and its meta says what it is.

What is mirrored is the *flange*, not the TCP: ``T_tcp' = mirror(T_tcp inv(T_tool)) T_tool``, with the arm's
own tool transform (the embodiment's ``tcp_frame``, from the URDF). A mirror flips handedness, so a pose's
rotation becomes ``S R S`` with ``S = diag(1, -1, 1)`` -- proper, the mirror image of the motion. Mirroring
the TCP itself would mirror the gripper too, and the Corenetic gripper sits at -45 deg yaw on the flange, so
the mirrored TCP is not where a left arm with the *same* gripper puts it: on 10-26 most such left frames
needed j6 past the panda_py wall (3.75 rad) and were unreachable. Mirrored at the flange, the left arm
reaches every frame with the right arm's joints mirrored, ``(-q1, q2, -q3, q4, -q5, q6, -q7)``, which the
symmetric walls allow -- so the pair is exactly as executable as its right-hand halves. (``tool=None``
mirrors the TCP, for a gripper that is mirror-symmetric on its flange.)

numpy for the 4x4s (``tools.fr3.fr3_kinematics``); the sidecars are copied off Thor and paired here.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import sys
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[3]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from tools.fr3 import fr3_kinematics as kin  # noqa: E402
from tools.thor.gmsl2 import bimanual_qc, world_provenance as wp  # noqa: E402

SCHEMA = "pseudo_bimanual/v1"
NAME_PREFIX = "pseudo_bimanual_"
#: G3's arm spacing. The real one is B0's; this only decides where the mirrored hand lands.
DEFAULT_SPACING_M = 0.8
#: The TCP the sidecars label (replay_cube_pose_in_robot_base.thor.yaml target_frame_name).
DEFAULT_TOOL_FRAME = "corenetic_socket_ee"
NOT_FOR = (
    "benchmark selection (roadmap A2)",
    "arm placement (roadmap B0)",
    "T_LR statistics: the two hands never moved together",
    "training",
)
#: An episode is a candidate when this share of its frames is solved with MIN_CAMERAS or more.
MIN_GOOD_FRACTION = 0.9
MIN_FRAMES = 300
#: A coarse reach filter, in fr3_base: every solved frame's TCP within this of the shoulder (joint 2, 0.333 m
#: up). 10-03/10-05 sessions sit 1.1-1.2 m out -- the hand worked where the cameras see, not where the arm
#: reaches -- and an absolute replay of them has nothing to do. The arm's IK is the real test (the runtime's
#: move to frame 0); this only keeps hopeless episodes out of the pool.
REACH_M = 0.85
SHOULDER_M = (0.0, 0.0, 0.333)
#: Then the real test: the share of frames the arm's IK reaches inside the panda_py walls (j6 <= 3.75 rad,
#: the wall the 10-09 replay was cut at). On 10-26 the 10-21 task demos had 0-18 % of frames out (mostly
#: j6) and the 10-13/14 pivot and ball-check takes 0-92 %. A pair whose halves are not fully reachable
#: tests the reach, not the runtime, so the default asks for all of them.
MIN_IK_REACHABLE = 1.0
DATASET_WORLD = wp.FR3_BASE_WORLD_ID

# Pose columns come in two spellings: ``state_x_m .. state_qw`` and ``ee.x .. ee.qw``.
_SUFFIXES = (
    ("_x_m", "_y_m", "_z_m", "_qx", "_qy", "_qz", "_qw"),
    (".x", ".y", ".z", ".qx", ".qy", ".qz", ".qw"),
)


def pose_groups(fieldnames: list[str]) -> list[list[str]]:
    """Every complete pose7 column group of a sidecar, as its seven column names."""
    names = set(fieldnames)
    groups = []
    for suffixes in _SUFFIXES:
        for name in fieldnames:
            if name.endswith(suffixes[0]):
                prefix = name[: -len(suffixes[0])]
                group = [prefix + s for s in suffixes]
                if all(g in names for g in group):
                    groups.append(group)
    return groups


def mirror_pose7(pose7: list[float], plane_y: float, tool: np.ndarray | None = None) -> list[float]:
    """The pose mirrored through the plane ``y = plane_y`` -- at the flange when ``tool`` (``T_flange_tcp``)
    is given, see the module docstring; a non-finite pose (a gap) stays a gap."""
    if not all(math.isfinite(v) for v in pose7):
        return list(pose7)
    if tool is None:
        x, y, z, qx, qy, qz, qw = pose7
        return [x, 2.0 * plane_y - y, z, -qx, qy, -qz, qw]
    T = np.asarray(_matrix(pose7)) @ np.linalg.inv(tool)
    S = np.diag([1.0, -1.0, 1.0, 1.0])
    M = S.copy()
    M[1, 3] = 2.0 * plane_y
    T = M @ T @ S @ tool
    return [*T[:3, 3].tolist(), *wp._quat_from_matrix(T[:3, :3].tolist())]


def _matrix(pose7: list[float]) -> list[list[float]]:
    x, y, z, qx, qy, qz, qw = pose7
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    return [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw), x],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw), y],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy), z],
        [0.0, 0.0, 0.0, 1.0],
    ]


def _float(value: Any) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return math.nan


def _map_poses(row: dict[str, str], groups: list[list[str]], fn) -> None:
    for group in groups:
        pose = [_float(row.get(k)) for k in group]
        if all(math.isfinite(v) for v in pose):
            row.update({k: repr(v) for k, v in zip(group, fn(pose), strict=True)})


def read_episodes(path: Path) -> tuple[list[str], dict[int, list[dict[str, str]]]]:
    """``(fieldnames, {episode: rows in file order})`` of one sidecar; a v1 sidecar is refused."""
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        if "schema_version" not in fieldnames:
            raise ValueError(f"{path}: no schema_version column (a v1 sidecar)")
        episodes: dict[int, list[dict[str, str]]] = {}
        for row in reader:
            episodes.setdefault(bimanual_qc._int(row.get("episode_index")), []).append(row)
    return fieldnames, episodes


def good_fraction(rows: list[dict[str, str]]) -> float:
    """Share of frames solved with at least ``bimanual_qc.MIN_CAMERAS`` cameras."""
    if not rows:
        return 0.0
    good = sum(
        1
        for row in rows
        if str(row.get("solve_ok")).strip().lower() in ("1", "true", "yes")
        and _float(row.get("num_cameras")) >= bimanual_qc.MIN_CAMERAS
    )
    return good / len(rows)


def reach_fraction(rows: list[dict[str, str]], T: list[list[float]]) -> float:
    """Share of the solved frames whose position, moved by ``T`` into fr3_base, is within REACH_M of the
    shoulder."""
    inside = total = 0
    for row in rows:
        pose = [_float(row.get(f"state_{k}")) for k in ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")]
        if not all(math.isfinite(v) for v in pose):
            continue
        total += 1
        inside += math.dist(wp.transform_pose7(T, pose)[:3], SHOULDER_M) <= REACH_M
    return inside / total if total else 0.0


def session_world(session_dir: Path) -> str:
    info = json.loads((session_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    return wp.world_frame_id_of(info.get("world_frame"))


def session_fps(session_dir: Path) -> float:
    info = json.loads((session_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    return float(info.get("fps") or bimanual_qc.DEFAULT_FPS)


def _sidecar_path(session_dir: Path, hand: str = "right") -> Path:
    return session_dir / "derived" / bimanual_qc.SIDECAR / f"state_action.{hand}.csv"


def ik_reachable_fraction(rows: list[dict[str, str]], T: list[list[float]], tool: np.ndarray) -> float:
    """Share of the solved frames, moved by ``T`` into fr3_base, that the arm tracks inside the walls along
    the best IK branch for the whole episode (``dual_fr3_retarget.choose_start``, as C3 and the single-arm
    replay's trajectory_start pick it).

    Not each frame solved on its own from the ready pose: that lands in a local minimum at the j6 wall for
    frames a continuous path reaches, and read 10-21's task demonstrations as 0-18 % out of reach (10-26).
    """
    from tools.fr3 import dual_fr3_retarget as rg

    poses = []
    for row in rows:
        pose = [_float(row.get(f"state_{k}")) for k in ("x_m", "y_m", "z_m", "qx", "qy", "qz", "qw")]
        if all(math.isfinite(v) for v in pose):
            poses.append(_matrix(wp.transform_pose7(T, pose)))
    if not poses:
        return 0.0
    _, start = rg.choose_start(kin.NOMINAL, np.asarray(poses) @ np.linalg.inv(tool))
    return start["frames_tracked"] / start["frames"]


def candidates(
    sessions_root: Path,
    graph: Mapping[str, Any],
    *,
    tool_frame: str | None = DEFAULT_TOOL_FRAME,
    min_ik_reachable: float = MIN_IK_REACHABLE,
) -> tuple[list[dict[str, Any]], list[str]]:
    """Right-hand episodes good enough to pair, and why the others were skipped. With ``tool_frame`` set
    (default) an episode also has to be reachable by the arm's IK, ``min_ik_reachable`` of its frames."""
    tool = None if tool_frame is None else kin.urdf_fixed_transform(tool_frame)
    found, skipped = [], []
    for session_dir in sorted(p for p in sessions_root.iterdir() if p.is_dir()):
        if session_dir.name.startswith(NAME_PREFIX):
            continue
        path = _sidecar_path(session_dir)
        if not path.is_file() or not (session_dir / "meta" / "info.json").is_file():
            skipped.append(f"{session_dir.name}: no right-hand sidecar or meta")
            continue
        try:
            world = session_world(session_dir)
            _, episodes = read_episodes(path)
        except (ValueError, OSError, json.JSONDecodeError) as exc:
            skipped.append(f"{session_dir.name}: {exc}")
            continue
        to_base = wp.world_transform(graph, world, DATASET_WORLD)
        if to_base is None:
            skipped.append(f"{session_dir.name}: no edge from {world} to {DATASET_WORLD}")
            continue
        for episode, rows in sorted(episodes.items()):
            fraction = good_fraction(rows)
            if len(rows) < MIN_FRAMES or fraction < MIN_GOOD_FRACTION:
                skipped.append(f"{session_dir.name} ep{episode}: {len(rows)} frames, {fraction:.0%} good")
                continue
            reach = reach_fraction(rows, to_base[0])
            if reach < 1.0:
                skipped.append(f"{session_dir.name} ep{episode}: {1 - reach:.0%} of frames out of reach")
                continue
            ik = None if tool is None else ik_reachable_fraction(rows, to_base[0], tool)
            if ik is not None and ik < min_ik_reachable:
                skipped.append(f"{session_dir.name} ep{episode}: {1 - ik:.0%} of frames out of the arm's IK")
                continue
            found.append(
                {
                    "session": session_dir.name,
                    "episode": episode,
                    "frames": len(rows),
                    "good": fraction,
                    "ik_reachable": ik,
                }
            )
    return found, skipped


def auto_pairs(
    found: list[dict[str, Any]], count: int, seed: int
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    """``count`` (right, left-source) pairs, deterministic in ``seed``; the two halves come from different
    sessions where there are any, so a pair is never one take against itself."""
    pool = list(found)
    random.Random(seed).shuffle(pool)
    pairs = []
    while len(pairs) < count and len(pool) >= 2:
        right = pool.pop(0)
        other = next((i for i, c in enumerate(pool) if c["session"] != right["session"]), 0)
        pairs.append((right, pool.pop(other)))
    return pairs


def build(
    sessions_root: Path,
    pairs: list[tuple[dict[str, Any], dict[str, Any]]],
    out_root: Path,
    *,
    graph: Mapping[str, Any],
    spacing_m: float = DEFAULT_SPACING_M,
    tool_frame: str | None = DEFAULT_TOOL_FRAME,
    name: str | None = None,
) -> Path:
    """Write the pairs as one stand-in session under ``out_root`` and return its directory."""
    if not pairs:
        raise ValueError("no pairs to build")
    plane_y = spacing_m / 2.0
    tool = None if tool_frame is None else kin.urdf_fixed_transform(tool_frame)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    session = out_root / (name or f"{NAME_PREFIX}{stamp}")
    if not session.name.startswith(NAME_PREFIX):
        raise ValueError(f"a stand-in session's name must start with {NAME_PREFIX!r}, not {session.name!r}")
    out_rows: dict[str, list[dict[str, str]]] = {"left": [], "right": []}
    fieldnames: list[str] | None = None
    records, edges_by_world, fps_seen = [], {}, set()
    for episode, (right_src, left_src) in enumerate(pairs):
        tracks = {}
        for hand, src in (("right", right_src), ("left", left_src)):
            session_dir = sessions_root / src["session"]
            names, episodes = read_episodes(_sidecar_path(session_dir))
            if fieldnames is None:
                fieldnames = names
            elif names != fieldnames:
                raise ValueError(f"{src['session']}: sidecar columns differ from the first source's")
            world = session_world(session_dir)
            found = wp.world_transform(graph, world, DATASET_WORLD)
            if found is None:
                raise ValueError(f"{src['session']}: no edge from {world} to {DATASET_WORLD}")
            T, edges = found
            edges_by_world[world] = edges
            fps_seen.add(session_fps(session_dir))
            rows = [dict(row) for row in episodes[int(src["episode"])]]
            groups = pose_groups(names)
            for row in rows:
                _map_poses(row, groups, lambda pose, T=T: wp.transform_pose7(T, pose))
                if hand == "left":
                    _map_poses(row, groups, lambda pose: mirror_pose7(pose, plane_y, tool))
            tracks[hand] = rows
        n = min(len(tracks["right"]), len(tracks["left"]))
        for k in range(n):
            clock = tracks["right"][k]
            for hand in ("right", "left"):
                row = dict(tracks[hand][k])
                row["episode_index"] = str(episode)
                row["frame_index"] = str(k)
                row["frame_global_index"] = str(len(out_rows[hand]))
                row["timestamp_s"] = clock.get("timestamp_s", "")
                out_rows[hand].append(row)
        records.append(
            {
                "episode": episode,
                "frames": n,
                "right": {"session": right_src["session"], "episode": int(right_src["episode"])},
                "left_mirrored_from": {"session": left_src["session"], "episode": int(left_src["episode"])},
            }
        )
    if len(fps_seen) != 1:
        raise ValueError(f"sources disagree on fps: {sorted(fps_seen)}")
    sidecar_dir = session / "derived" / bimanual_qc.SIDECAR
    sidecar_dir.mkdir(parents=True, exist_ok=True)
    for hand, rows in out_rows.items():
        with (sidecar_dir / f"state_action.{hand}.csv").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
    info = {
        "fps": fps_seen.pop(),
        "world_frame": {
            "world_frame_id": DATASET_WORLD,
            "status": wp.STATUS_OK,
            "edges_by_source_world": edges_by_world,
        },
        "pseudo_bimanual": {
            "schema": SCHEMA,
            "created_utc": datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "what": (
                "two recorded right-hand episodes per pair: the right as recorded, the other mirrored through "
                f"y = {plane_y:.3f} m of {DATASET_WORLD} to stand in for the left hand"
            ),
            "assumed_spacing_m": spacing_m,
            "mirror_plane_y_m": plane_y,
            "mirrored_at": "tcp"
            if tool is None
            else f"flange ({kin.FLANGE_LINK}), tool {tool_frame} re-attached",
            "T_flange_tcp": None if tool is None else tool.tolist(),
            "not_for": list(NOT_FOR),
            "sessions_root": str(sessions_root),
            "pairs": records,
        },
    }
    (session / "meta").mkdir(parents=True, exist_ok=True)
    (session / "meta" / "info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    return session


def is_pseudo(session_dir: Path) -> bool:
    """True for a stand-in session: by name, or by its meta (a renamed copy is still one)."""
    if Path(session_dir).name.startswith(NAME_PREFIX):
        return True
    try:
        info = json.loads((Path(session_dir) / "meta" / "info.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    return "pseudo_bimanual" in info


def _parse_pair(text: str) -> tuple[dict[str, Any], dict[str, Any]]:
    try:
        right, left = text.split(",")
        rs, re_ = right.rsplit(":", 1)
        ls, le = left.rsplit(":", 1)
        return {"session": rs, "episode": int(re_)}, {"session": ls, "episode": int(le)}
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            f"--pair takes RIGHT_SESSION:EP,LEFT_SESSION:EP, not {text!r}"
        ) from exc


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--sessions-root", type=Path, required=True, help="directory holding the recorded sessions"
    )
    parser.add_argument("--out", type=Path, default=_REPO_ROOT / "outputs" / "pseudo_bimanual")
    parser.add_argument(
        "--pair", type=_parse_pair, action="append", default=[], help="RIGHT_SESSION:EP,LEFT_SESSION:EP"
    )
    parser.add_argument(
        "--auto", type=int, default=0, help="pick this many pairs from the good right-hand episodes"
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--spacing", type=float, default=DEFAULT_SPACING_M, help="assumed base spacing along y, m"
    )
    parser.add_argument("--name", default=None)
    parser.add_argument("--repo-root", type=Path, default=_REPO_ROOT)
    args = parser.parse_args(argv)
    graph = wp.read_world_graph(args.repo_root)
    pairs = list(args.pair)
    if args.auto:
        found, skipped = candidates(args.sessions_root, graph)
        print(f"{len(found)} candidate right-hand episodes, {len(skipped)} skipped")
        pairs += auto_pairs(found, args.auto, args.seed)
        if len(pairs) < args.auto + len(args.pair):
            print(f"only {len(pairs)} pairs available")
    if not pairs:
        parser.error("give --pair or --auto")
    session = build(args.sessions_root, pairs, args.out, graph=graph, spacing_m=args.spacing, name=args.name)
    report = bimanual_qc.session_qc(session)
    for line in bimanual_qc.describe(report):
        print(line)
    print(f"wrote {session}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
