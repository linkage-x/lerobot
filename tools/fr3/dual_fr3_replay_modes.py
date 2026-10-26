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
Where each arm's targets come from in a two-arm replay: the two replay modes (roadmap C2).

* **absolute** -- each hand goes where it was in the world: the dataset's poses through the recorded edges into
  each arm's base (``embodiment.arm_transform``), nothing else. No alignment of any kind, so no per-episode
  offset can be absorbed. Refused when an arm's base is not connected to the dataset's world, or when the
  dataset was re-expressed through an edge the graph no longer holds (``embodiment.export_edges_current``).
* **relative** -- the two hands keep their relation exactly and are placed together by **one** rigid
  ``T_align`` (dataset -> the reference arm's base); the other arm gets its targets through the base-to-base
  transform alone. Aligning each hand to its own first frame is what this mode exists to forbid: it would
  change the distance between the hands, which is the thing a bimanual demonstration is about. ``T_align`` is
  the identity (the dataset is already in the reference base), the dataset's registration (``registered``),
  a given matrix, or ``fit``: the shift and turn about the vertical, within bounds, that keeps the most
  frames of both arms inside the joint walls.

Either way the plan names the mode, every edge it rests on, the base-to-base transform the collision checks
use and its claimed sigma, and ``T_align`` -- the record a replay report carries.
"""

# ruff: noqa: N802, N803, N806  (matrix names: R, T, J follow the maths)
from __future__ import annotations

import json
import math
import sys
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from tools.fr3 import fr3_kinematics as kin  # noqa: E402
from tools.thor.gmsl2 import bimanual_qc, embodiment as em, world_provenance as wp  # noqa: E402

SCHEMA = "dual_fr3_mode/v1"
MODES = ("absolute", "relative")
ALIGNS = ("identity", "registered", "matrix", "fit")
#: ``fit``'s search: shifts per axis and turns about the vertical through the hands' centroid. Wide enough to
#: pull a demonstration recorded at the edge of the reach back in; small enough that the motion is the same
#: motion, a few centimetres over.
FIT_SHIFTS_M = (-0.10, -0.05, 0.0, 0.05, 0.10)
FIT_TURNS_DEG = (-15.0, 0.0, 15.0)
FIT_SAMPLE_FRAMES = 24
#: Candidates that go on to the IK check, best by the geometric reach first.
FIT_IK_CANDIDATES = 12
#: Shoulder (joint 2's axis) to flange at full stretch, less a little: a flange farther out cannot be reached.
FIT_REACH_M = 0.80
_SHOULDER = np.array([0.0, 0.0, 0.333])


def matrix(pose7: Sequence[float]) -> np.ndarray:
    x, y, z, qx, qy, qz, qw = pose7
    n = math.sqrt(qx * qx + qy * qy + qz * qz + qw * qw)
    qx, qy, qz, qw = qx / n, qy / n, qz / n, qw / n
    T = np.eye(4)
    T[:3, :3] = [
        [1 - 2 * (qy * qy + qz * qz), 2 * (qx * qy - qz * qw), 2 * (qx * qz + qy * qw)],
        [2 * (qx * qy + qz * qw), 1 - 2 * (qx * qx + qz * qz), 2 * (qy * qz - qx * qw)],
        [2 * (qx * qz - qy * qw), 2 * (qy * qz + qx * qw), 1 - 2 * (qx * qx + qy * qy)],
    ]
    T[:3, 3] = (x, y, z)
    return T


@dataclass
class Episode:
    """Both hands of one episode, frame-aligned: poses in the dataset's frame (held over lost frames)."""

    session: str
    episode: int
    frames: np.ndarray  # (n,) frame indices both hands have rows for
    t: np.ndarray  # (n,) seconds, as recorded
    poses: dict[str, np.ndarray]  # hand -> (n,4,4)
    seen: dict[str, np.ndarray]  # hand -> (n,) solved this frame
    world_frame: dict[str, Any]
    fps: float

    @property
    def dataset_world(self) -> str:
        return wp.world_frame_id_of(self.world_frame)


def _held(rows: Mapping[int, Mapping[str, Any]], frames: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    seen = np.array([bool(rows[f]["ok"]) for f in frames])
    if not seen.any():
        raise ValueError("the hand is never seen in this episode")
    first = next(f for f, ok in zip(frames, seen, strict=True) if ok)
    last = rows[first]
    out = []
    for f in frames:
        if rows[f]["ok"]:
            last = rows[f]
        out.append(matrix([*last["p"], *last["q"]]))
    return np.stack(out), seen


def read_episode(session_dir: Path | str, episode: int, hands: Sequence[str] = ("right", "left")) -> Episode:
    """Both hands' sidecars of ``episode``, on the frames they share; a lost frame holds the last pose (as the
    replay does), leading gaps the first."""
    session_dir = Path(session_dir)
    info = json.loads((session_dir / "meta" / "info.json").read_text(encoding="utf-8"))
    tracks = {
        hand: bimanual_qc.read_hand(
            session_dir / "derived" / bimanual_qc.SIDECAR / f"state_action.{hand}.csv"
        )[episode]
        for hand in hands
    }
    frames = sorted(set.intersection(*(set(track) for track in tracks.values())))
    if not frames:
        raise ValueError(f"{session_dir} episode {episode}: the hands share no frame")
    fps = float(info.get("fps") or 0.0)
    t = np.array([tracks[hands[0]][f]["t"] for f in frames], dtype=np.float64)
    if not np.isfinite(t).all():
        if fps <= 0:
            raise ValueError(f"{session_dir}: no timestamps and no fps")
        t = np.asarray(frames, dtype=np.float64) / fps
    poses, seen = {}, {}
    for hand, track in tracks.items():
        poses[hand], seen[hand] = _held(track, frames)
    return Episode(
        session=str(session_dir),
        episode=int(episode),
        frames=np.asarray(frames),
        t=t - t[0],
        poses=poses,
        seen=seen,
        world_frame=dict(info.get("world_frame") or {}),
        fps=fps if fps > 0 else 1.0 / float(np.median(np.diff(t))),
    )


def route_sigma_mm(graph: Mapping[str, Any], hops: Sequence[Mapping[str, Any]]) -> float | None:
    """The combined 1-sigma the edges of a route claim (``world_provenance.edge_sigma_mm``, root sum of
    squares); None for an empty route, and an edge that claims nothing counts as UNKNOWN_EDGE_SIGMA_MM."""
    if not hops:
        return None
    total = 0.0
    for hop in hops:
        key = em._hop_key({**hop, "traversed_reversed": False})[:4]
        edge = next(
            (
                e
                for e in graph.get("edges") or []
                if em._hop_key({**e, "traversed_reversed": False})[:4] == key
            ),
            None,
        )
        sigma = None if edge is None else wp.edge_sigma_mm(edge)
        total += (wp.UNKNOWN_EDGE_SIGMA_MM if sigma is None else sigma) ** 2
    return math.sqrt(total)


@dataclass
class ModePlan:
    mode: str
    ok: bool
    reason: str = ""
    arms: dict[str, dict[str, Any]] = field(default_factory=dict)  # arm -> hand, root, T_root_dataset, edges
    T_right_left: np.ndarray | None = None  # the left arm's base in the right arm's base
    placement: dict[str, Any] = field(default_factory=dict)
    align: dict[str, Any] = field(default_factory=dict)
    targets: dict[str, np.ndarray] = field(default_factory=dict)  # arm -> (n,4,4) TCP in its base
    seen: dict[str, np.ndarray] = field(default_factory=dict)

    def record(self) -> dict[str, Any]:
        return {
            "schema": SCHEMA,
            "mode": self.mode,
            "ok": self.ok,
            "reason": self.reason,
            "arms": self.arms,
            "T_right_left": None if self.T_right_left is None else self.T_right_left.tolist(),
            "placement": self.placement,
            "align": self.align,
        }


def _refused(mode: str, reason: str, **extra: Any) -> ModePlan:
    return ModePlan(mode=mode, ok=False, reason=reason, **extra)


def _placement(
    description: Mapping[str, Any], graph: Mapping[str, Any]
) -> tuple[np.ndarray | None, dict[str, Any]]:
    roots = {arm: str(spec["root"]) for arm, spec in description["arms"].items()}
    if set(roots) != {"right", "left"}:
        return None, {"reason": "not a two-arm embodiment"}
    found = wp.world_transform(graph, roots["left"], roots["right"])
    if found is None:
        return None, {"reason": f"no edge connects {roots['left']} to {roots['right']} (B3)"}
    return np.asarray(found[0], dtype=np.float64), {
        "from": roots["left"],
        "to": roots["right"],
        "edges": found[1],
        "sigma_mm": route_sigma_mm(graph, found[1]),
        "assumed": any(str(e.get("method") or "").startswith("assumed_placement") for e in found[1]),
    }


def _turn_about(centre: np.ndarray, shift: np.ndarray, yaw_rad: float) -> np.ndarray:
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    T = np.eye(4)
    T[:3, :3] = [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]
    T[:3, 3] = centre + shift - T[:3, :3] @ centre
    return T


def fit_alignment(
    hand_poses: Mapping[str, np.ndarray],
    T_arm_ref: Mapping[str, np.ndarray],
    tool: np.ndarray,
    *,
    shifts: Sequence[float] = FIT_SHIFTS_M,
    turns_deg: Sequence[float] = FIT_TURNS_DEG,
    sample: int = FIT_SAMPLE_FRAMES,
) -> tuple[np.ndarray, dict[str, Any]]:
    """The offset (in the reference base) that keeps the most sampled frames of every arm inside its joint
    walls; ties go to the smallest move. ``hand_poses``: arm -> (n,4,4) already in the reference base.

    Every candidate is scored by how many flanges fall within FIT_REACH_M of the shoulder, and only the best
    FIT_IK_CANDIDATES (and the unaligned one) are solved by IK -- per-frame IK from the ready pose, a
    screen: the retarget (C3) is what says whether the whole path can be followed."""
    n = len(next(iter(hand_poses.values())))
    pick = np.unique(np.linspace(0, n - 1, min(sample, n)).round().astype(int))
    centre = np.mean([P[pick, :3, 3].mean(axis=0) for P in hand_poses.values()], axis=0)
    candidates = [
        (np.array([dx, dy, dz]), math.radians(yaw))
        for dx in shifts
        for dy in shifts
        for dz in shifts
        for yaw in turns_deg
    ]
    offsets = np.stack([_turn_about(centre, shift, yaw) for shift, yaw in candidates])
    tool_inv = np.linalg.inv(tool)
    move = np.array([np.linalg.norm(shift) + 0.3 * abs(yaw) for shift, yaw in candidates])
    identity = int(np.argmin(move))
    flanges = {
        arm: np.asarray(T_arm_ref[arm]) @ offsets[:, None] @ P[pick][None] @ tool_inv  # (c, s, 4, 4)
        for arm, P in hand_poses.items()
    }
    # geometric reach for every candidate, IK for the best few (and the unaligned one)
    near = sum(
        (np.linalg.norm(F[..., :3, 3] - _SHOULDER, axis=-1) <= FIT_REACH_M).mean(axis=1)
        for F in flanges.values()
    )
    order = [int(i) for i in np.lexsort((move, -near))[:FIT_IK_CANDIDATES]]
    if identity not in order:
        order.append(identity)
    reach = np.full(len(candidates), -1.0)
    reach[order] = 0.0
    for F in flanges.values():
        _, ok = kin.ik_retry(kin.NOMINAL, F[order].reshape(-1, 4, 4), kin.Q_READY)
        reach[order] += ok.reshape(len(order), len(pick)).sum(axis=1)
    reach[order] /= len(hand_poses) * len(pick)
    best = int(np.lexsort((move, -reach))[0])
    shift, yaw = candidates[best]
    return offsets[best], {
        "shift_m": shift.tolist(),
        "yaw_deg": math.degrees(yaw),
        "about_m": centre.tolist(),
        "reachable_fraction": float(reach[best]),
        "reachable_fraction_unaligned": float(reach[identity]),
        "sampled_frames": len(pick),
    }


def plan(
    mode: str,
    episode: Episode,
    description: Mapping[str, Any],
    graph: Mapping[str, Any],
    *,
    align: str = "identity",
    T_align: np.ndarray | None = None,
) -> ModePlan:
    """Each arm's TCP targets in its own base for ``mode``, or the reason the mode is refused."""
    if mode not in MODES:
        raise ValueError(f"mode {mode!r} not one of {MODES}")
    hands = [hand for hand in ("right", "left") if hand in episode.poses]
    modes = em.replay_modes(description, graph, episode.dataset_world, hands, world_frame=episode.world_frame)
    arm_of = modes["arms"]
    T_right_left, placement = _placement(description, graph)
    if not modes[mode]["ok"]:
        return _refused(mode, modes[mode]["reason"], placement=placement)
    if len(hands) == 2 and T_right_left is None:
        return _refused(mode, f"the collision checks need the bases' placement: {placement['reason']}")
    tool = {arm: kin.urdf_fixed_transform(description["arms"][arm]["tcp_frame"]) for arm in arm_of.values()}

    out = ModePlan(mode=mode, ok=True, T_right_left=T_right_left, placement=placement)
    if mode == "absolute":
        if align != "identity" or T_align is not None:
            return _refused(
                mode, "absolute replay takes no alignment; use relative to move the demonstration"
            )
        for hand, arm in arm_of.items():
            T, edges, _ = em.arm_transform(description, graph, episode.dataset_world, arm)
            T = np.asarray(T, dtype=np.float64)
            out.arms[arm] = {
                "hand": hand,
                "root": description["arms"][arm]["root"],
                "T_root_dataset": T.tolist(),
                "edges": edges,
            }
            out.targets[arm] = T @ episode.poses[hand]
            out.seen[arm] = episode.seen[hand]
        out.align = {"kind": "none"}
        return out

    if align not in ALIGNS:
        raise ValueError(f"align {align!r} not one of {ALIGNS}")
    reference_root = str(description["dataset_frame"])
    reference_arm = next(arm for arm, spec in description["arms"].items() if spec["root"] == reference_root)
    if align == "matrix":
        if T_align is None:
            raise ValueError("align=matrix needs T_align")
        base = np.asarray(T_align, dtype=np.float64)
        base_edges: list[dict[str, Any]] = []
    elif episode.dataset_world == reference_root:
        base, base_edges = np.eye(4), []
    elif align in ("registered", "fit"):
        T, base_edges, reason = em.arm_transform(description, graph, episode.dataset_world, reference_arm)
        if T is None:
            return _refused(mode, f"align={align} starts from the dataset's registration: {reason}")
        base = np.asarray(T, dtype=np.float64)
    else:
        return _refused(
            mode,
            f"the dataset is in {episode.dataset_world or '(no world)'}, not {reference_root}: "
            "relative replay needs align=registered, fit or a matrix",
        )
    T_arm_ref = {}
    for arm in arm_of.values():
        root = str(description["arms"][arm]["root"])
        found = wp.world_transform(graph, reference_root, root)
        T_arm_ref[arm] = np.asarray(found[0], dtype=np.float64)
        out.arms[arm] = {"root": root, "edges": found[1]}
    in_reference = {arm: base @ episode.poses[hand] for hand, arm in arm_of.items()}
    fit = None
    if align == "fit":
        offset, fit = fit_alignment(in_reference, T_arm_ref, tool[reference_arm])
        base = offset @ base
    for hand, arm in arm_of.items():
        T = T_arm_ref[arm] @ base
        out.arms[arm].update(hand=hand, T_root_dataset=T.tolist())
        out.targets[arm] = T @ episode.poses[hand]
        out.seen[arm] = episode.seen[hand]
    out.align = {
        "kind": align,
        "T_align": base.tolist(),
        "reference_root": reference_root,
        "edges": base_edges,
    }
    if fit is not None:
        out.align["fit"] = fit
    return out
