# ruff: noqa: N803, N806  -- T, R, P, Q: the usual names of transforms, rotations, positions, quaternions
"""Where the training TCP went, as the execution layer checks it: a policy that sees no absolute pose cannot
know when the arm is somewhere it was never shown, so the arm is kept where the demonstrations were.

Three checks, from the checkpoint's own training export (the episodes it trained on, not the held-out ones):

A  **start**: the arm's TCP, before anything moves, within ``start_pos_m`` of an episode's first second and
   within ``start_rot_deg`` of its orientation there; the gripper's width inside the widths episodes began with.
B  **every target**: within ``pos_m`` of a training TCP position (nearest neighbour, not the bounding box:
   the demonstrations are a thin tube through a box mostly never visited), and within ``rot_deg`` of the
   orientation of a training pose that close.
C  **every chunk**: step k's displacement from the chunk's anchor (translation and rotation) within what
   training's actions did at step k (99.5th percentile).

No threshold is guessed: each is the margin (``MARGIN``) times what a demonstration needs when it is left out
and checked against the others (leave-one-episode-out: the 99th percentile of B's distances, the largest of
A's), with a floor. The held-out episodes (never trained on) are then checked too, and the share of them that
passes is recorded with the thresholds: if held-out demonstrations fail, the thresholds are too tight.

All poses are in the export's camera world (``frame``); the plan check takes targets there through the
registration edge of the day, as it does for the workspace box.
"""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from tools.fr3 import relative_tcp as rt

MARGIN = 1.5
FLOOR_POS_M = 0.02
FLOOR_ROT_DEG = 10.0
FLOOR_START_POS_M = 0.03
FLOOR_START_ROT_DEG = 15.0
FLOOR_STEP_M = 0.005
FLOOR_STEP_DEG = 2.0
START_WIDTH_SLACK_M = 0.010
EARLY_S = 1.0  # an episode's "start": its first second (the hand is still or just leaving)
SUBSAMPLE = 2  # 60 fps exports -> 30 Hz reference points
VERSION = 1  # bump when what is computed changes (the cache key carries it)


class DistributionError(RuntimeError):
    """The training distribution cannot be built here (export missing, not the one trained on, ...)."""


def angle_deg(qa: np.ndarray, qb: np.ndarray) -> np.ndarray:
    """Angle between unit quaternions (xyzw, any broadcastable shapes), degrees."""
    dot = np.abs(np.sum(np.asarray(qa) * np.asarray(qb), axis=-1))
    return np.degrees(2.0 * np.arccos(np.clip(dot, 0.0, 1.0)))


def _nearest(P: np.ndarray, Q: np.ndarray, refP: np.ndarray, refQ: np.ndarray, radius: float) -> tuple[np.ndarray, np.ndarray]:
    """Per point: distance to the nearest reference position, and the smallest angle to a reference pose
    within ``radius`` of it (inf when none is)."""
    dpos = np.empty(len(P))
    drot = np.full(len(P), math.inf)
    for s in range(0, len(P), 256):
        d = np.linalg.norm(P[s : s + 256, None, :] - refP[None], axis=2)
        dpos[s : s + 256] = d.min(axis=1) if d.size else math.inf
        near = d <= radius
        if near.any():
            ang = angle_deg(Q[s : s + 256, None, :], refQ[None])
            ang = np.where(near, ang, math.inf)
            drot[s : s + 256] = ang.min(axis=1)
    return dpos, drot


@dataclass
class Episode:
    index: int
    poses: np.ndarray  # (n, 7) world TCP, NaN where not tracked
    widths: np.ndarray  # (n,)


@dataclass
class Distribution:
    frame: str
    P: np.ndarray  # (N, 3) training TCP positions (30 Hz)
    Q: np.ndarray  # (N, 4) their orientations (xyzw)
    start_p: np.ndarray  # (M, 3) poses of the episodes' first second
    start_q: np.ndarray
    start_width: tuple[float, float]
    pos_m: float
    rot_deg: float
    start_pos_m: float
    start_rot_deg: float
    step_m: np.ndarray  # (H,) step k's largest displacement from the anchor
    step_deg: np.ndarray
    summary: dict[str, Any] = field(default_factory=dict)

    # ------------------------------------------------------------- checks ---

    def start_problem(self, T_world: np.ndarray, width: float | None) -> str:
        """A: "" when the arm may start here."""
        p = T_world[:3, 3][None]
        q = rt.matrix_to_quat_xyzw(T_world[:3, :3])[None]
        dpos, drot = _nearest(p, q, self.start_p, self.start_q, self.start_pos_m)
        if dpos[0] > self.start_pos_m:
            return (f"start: the TCP is {dpos[0] * 1e3:.0f} mm from where any training episode began "
                    f"(limit {self.start_pos_m * 1e3:.0f} mm): move the arm to the start pose")
        if drot[0] > self.start_rot_deg:
            return (f"start: the gripper is turned {drot[0]:.0f}° from how the training episodes began "
                    f"(limit {self.start_rot_deg:.0f}°)")
        lo, hi = self.start_width
        if width is not None and not lo <= width <= hi:
            return (f"start: the gripper is {width * 1e3:.0f} mm open; training episodes began "
                    f"{lo * 1e3:.0f}-{hi * 1e3:.0f} mm (open the gripper)")
        return ""

    def target_problems(self, T_world: np.ndarray, T_rel: np.ndarray) -> list[str]:
        """B and C per step (targets in the world, and relative to the chunk's anchor); "" = fine."""
        n = len(T_world)
        P = T_world[:, :3, 3]
        Q = rt.matrix_to_quat_xyzw(T_world[:, :3, :3])
        dpos, drot = _nearest(P, Q, self.P, self.Q, self.pos_m)
        step_m = np.linalg.norm(T_rel[:, :3, 3], axis=1)
        step_deg = np.degrees(np.arccos(np.clip((np.trace(T_rel[:, :3, :3], axis1=1, axis2=2) - 1) / 2, -1, 1)))
        out = []
        H = len(self.step_m)
        for k in range(n):
            lim_m, lim_deg = self.step_m[min(k, H - 1)], self.step_deg[min(k, H - 1)]
            if dpos[k] > self.pos_m:
                out.append(f"step {k + 1}: {dpos[k] * 1e3:.0f} mm from any training TCP position (limit {self.pos_m * 1e3:.0f})")
            elif drot[k] > self.rot_deg:
                out.append(f"step {k + 1}: turned {drot[k]:.0f}° from the training poses there (limit {self.rot_deg:.0f}°)")
            elif step_m[k] > lim_m:
                out.append(f"step {k + 1}: the chunk moves {step_m[k] * 1e3:.0f} mm by here; training moved at most "
                           f"{lim_m * 1e3:.0f} mm in {k + 1} steps")
            elif step_deg[k] > lim_deg:
                out.append(f"step {k + 1}: the chunk turns {step_deg[k]:.0f}° by here; training turned at most "
                           f"{lim_deg:.0f}° in {k + 1} steps")
            else:
                out.append("")
        return out

    # ------------------------------------------------------------- storage ---

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(path, P=self.P, Q=self.Q, start_p=self.start_p, start_q=self.start_q,
                            step_m=self.step_m, step_deg=self.step_deg,
                            meta=np.array(json.dumps({
                                "frame": self.frame, "start_width": list(self.start_width), "pos_m": self.pos_m,
                                "rot_deg": self.rot_deg, "start_pos_m": self.start_pos_m,
                                "start_rot_deg": self.start_rot_deg, "summary": self.summary})))

    @classmethod
    def load(cls, path: Path) -> Distribution:
        with np.load(path) as z:
            meta = json.loads(str(z["meta"]))
            return cls(meta["frame"], z["P"], z["Q"], z["start_p"], z["start_q"], tuple(meta["start_width"]),
                       meta["pos_m"], meta["rot_deg"], meta["start_pos_m"], meta["start_rot_deg"],
                       z["step_m"], z["step_deg"], meta["summary"])


# ---------------------------------------------------------------- building ---


def _finite(poses: np.ndarray) -> np.ndarray:
    return np.all(np.isfinite(poses), axis=1)


def _early(ep: Episode, fps: float) -> np.ndarray:
    """Indices of the episode's first EARLY_S seconds of tracked frames."""
    ok = np.flatnonzero(_finite(ep.poses))
    return ok[ok <= (ok[0] + EARLY_S * fps)] if ok.size else ok


def _step_envelope(episodes: Sequence[Episode], offsets: Sequence[int]) -> tuple[np.ndarray, np.ndarray]:
    """Per action offset: the 99.5th percentile of |translation| and rotation from the pose at t."""
    dm: list[list[float]] = [[] for _ in offsets]
    dd: list[list[float]] = [[] for _ in offsets]
    for ep in episodes:
        T = rt.pose7_to_matrix(np.where(_finite(ep.poses)[:, None], ep.poses, np.array([0, 0, 0, 0, 0, 0, 1.0])))
        ok = _finite(ep.poses)
        n = len(T)
        Tinv = np.linalg.inv(T)
        for j, o in enumerate(offsets):
            if o >= n:
                continue
            a, b = np.arange(0, n - o), np.arange(o, n)
            keep = ok[a] & ok[b]
            rel = Tinv[a[keep]] @ T[b[keep]]
            dm[j].extend(np.linalg.norm(rel[:, :3, 3], axis=1).tolist())
            tr = np.trace(rel[:, :3, :3], axis1=1, axis2=2)
            dd[j].extend(np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1))).tolist())
    m = np.array([np.percentile(v, 99.5) if v else math.inf for v in dm])
    d = np.array([np.percentile(v, 99.5) if v else math.inf for v in dd])
    # never smaller than at an earlier step: a chunk that got somewhere by step k has been at least as far
    return np.maximum.accumulate(m), np.maximum.accumulate(d)


def build(episodes: Sequence[Episode], *, fps: float, offsets: Sequence[int], frame: str,
          holdout: Sequence[Episode] = (), source: dict[str, Any] | None = None) -> Distribution:
    """The reference from the training episodes; thresholds by leave-one-episode-out; held-out ones checked."""
    eps = [e for e in episodes if _finite(e.poses).any()]
    if len(eps) < 3:
        raise DistributionError(f"{len(eps)} tracked training episodes: too few to say where the demonstrations went")
    pts = []
    for e in eps:
        idx = np.flatnonzero(_finite(e.poses))[::SUBSAMPLE]
        pts.append((e.index, e.poses[idx, :3], e.poses[idx, 3:7]))
    starts = [(e.index, e.poses[_early(e, fps), :3], e.poses[_early(e, fps), 3:7]) for e in eps]
    first_widths = []
    for e in eps:
        ok = np.flatnonzero(_finite(e.poses) & np.isfinite(e.widths))
        if ok.size:
            first_widths.append(float(e.widths[ok[0]]))

    def others(table, i):
        P = np.concatenate([p for j, p, _ in table if j != i])
        Q = np.concatenate([q for j, _, q in table if j != i])
        return P, Q

    # B: positions first (any radius), then orientations among the poses that close
    loo_pos = np.concatenate([_nearest(p, q, *others(pts, i), 0.0)[0] for i, p, q in pts])
    pos_m = max(FLOOR_POS_M, MARGIN * float(np.percentile(loo_pos, 99)))
    loo_rot = np.concatenate([_nearest(p, q, *others(pts, i), pos_m)[1] for i, p, q in pts])
    loo_rot = loo_rot[np.isfinite(loo_rot)]
    rot_deg = max(FLOOR_ROT_DEG, MARGIN * float(np.percentile(loo_rot, 99))) if loo_rot.size else FLOOR_ROT_DEG
    # A: each episode's first frame against the other episodes' first seconds
    loo_sp = [_nearest(p[:1], q[:1], *others(starts, i), 0.0)[0][0] for i, p, q in starts]
    start_pos_m = max(FLOOR_START_POS_M, MARGIN * float(max(loo_sp)))
    loo_sr = [_nearest(p[:1], q[:1], *others(starts, i), start_pos_m)[1][0] for i, p, q in starts]
    loo_sr = [v for v in loo_sr if math.isfinite(v)]
    start_rot_deg = max(FLOOR_START_ROT_DEG, MARGIN * float(max(loo_sr))) if loo_sr else FLOOR_START_ROT_DEG
    step_m, step_deg = _step_envelope(eps, offsets)
    step_m = np.maximum(FLOOR_STEP_M, MARGIN * step_m)
    step_deg = np.maximum(FLOOR_STEP_DEG, MARGIN * step_deg)
    lo, hi = (min(first_widths), max(first_widths)) if first_widths else (0.0, 1.0)

    dist = Distribution(
        frame, np.concatenate([p for _, p, _ in pts]), np.concatenate([q for _, _, q in pts]),
        np.concatenate([p for _, p, _ in starts]), np.concatenate([q for _, _, q in starts]),
        (lo - START_WIDTH_SLACK_M, hi + START_WIDTH_SLACK_M), pos_m, rot_deg, start_pos_m, start_rot_deg,
        step_m, step_deg,
    )
    dist.summary = {
        "version": VERSION,
        "frame": frame,
        "episodes": [e.index for e in eps],
        "points": int(len(dist.P)),
        "margin": MARGIN,
        "pos_mm": round(pos_m * 1e3, 1),
        "rot_deg": round(rot_deg, 1),
        "start_pos_mm": round(start_pos_m * 1e3, 1),
        "start_rot_deg": round(start_rot_deg, 1),
        "start_width_mm": [round(dist.start_width[0] * 1e3, 1), round(dist.start_width[1] * 1e3, 1)],
        "step1_mm": round(float(step_m[0]) * 1e3, 1),
        "step1_deg": round(float(step_deg[0]), 2),
        "step_mm_at_16_32": [round(float(step_m[min(15, len(step_m) - 1)]) * 1e3, 1), round(float(step_m[-1]) * 1e3, 1)],
        "step_deg_at_16_32": [round(float(step_deg[min(15, len(step_deg) - 1)]), 1), round(float(step_deg[-1]), 1)],
        "leave_one_out": {"pos_mm_p99": round(float(np.percentile(loo_pos, 99)) * 1e3, 1),
                          "rot_deg_p99": round(float(np.percentile(loo_rot, 99)), 1) if loo_rot.size else None,
                          "start_pos_mm_max": round(float(max(loo_sp)) * 1e3, 1),
                          "start_rot_deg_max": round(float(max(loo_sr)), 1) if loo_sr else None},
        "holdout": _holdout_check(dist, holdout, fps, offsets),
        "source": source or {},
    }
    return dist


def _holdout_check(dist: Distribution, holdout: Sequence[Episode], fps: float, offsets: Sequence[int]) -> dict[str, Any]:
    """How much of the never-trained-on episodes the thresholds let through (should be nearly all)."""
    held = [e for e in holdout if _finite(e.poses).any()]
    if not held:
        return {"episodes": 0}
    frames = passed = starts_ok = chunks = chunks_ok = 0
    for e in held:
        idx = np.flatnonzero(_finite(e.poses))[::SUBSAMPLE]
        T = rt.pose7_to_matrix(e.poses[idx])
        why = dist.target_problems(T, np.repeat(np.eye(4)[None], len(T), axis=0))
        frames += len(why)
        passed += sum(1 for w in why if not w)
        first = int(_early(e, fps)[0])
        ok = np.flatnonzero(np.isfinite(e.widths))
        width = float(e.widths[ok[0]]) if ok.size else None
        starts_ok += int(not dist.start_problem(rt.pose7_to_matrix(e.poses[first]), width))
        n = len(e.poses)
        for t in range(0, n - offsets[-1], max(1, int(fps // 2))):
            tgt = e.poses[[t + o for o in offsets]]
            if not (_finite(tgt).all() and _finite(e.poses[t : t + 1]).all()):
                continue
            T0 = rt.pose7_to_matrix(e.poses[t])
            Tt = rt.pose7_to_matrix(tgt)
            rel = np.linalg.inv(T0)[None] @ Tt
            chunks += 1
            chunks_ok += int(not any(dist.target_problems(Tt, rel)))
    return {"episodes": len(held), "frames_pass": round(passed / max(frames, 1), 4),
            "starts_pass": f"{starts_ok}/{len(held)}", "chunks_pass": round(chunks_ok / max(chunks, 1), 4)}


# ------------------------------------------------------- from a checkpoint ---


def _export_root(entry: dict[str, Any], repo_root: Path) -> Path:
    for candidate in (Path(str(entry.get("root") or "")), repo_root / "outputs" / "exports" / str(entry.get("name") or "")):
        if candidate.name and (candidate / "meta" / "info.json").is_file():
            return candidate
    raise DistributionError(f"the training export {entry.get('name')!r} is not on this machine "
                            f"(outputs/exports/{entry.get('name')}): the arm cannot be kept where the demonstrations were")


def for_checkpoint(provenance: dict[str, Any], contract: rt.PolicyContract, repo_root: Path,
                   cache_dir: Path | None = None) -> Distribution:
    """The distribution of the episodes this checkpoint trained on, built once per checkpoint and cached."""
    from tools.fr3 import relative_tcp_data as rtd

    exports = provenance.get("exports") or []
    if not exports:
        raise DistributionError("the checkpoint names no training export (fr3_provenance.json)")
    workspace = provenance.get("workspace") or {}
    frame = str(workspace.get("frame") or exports[0].get("world_frame_id") or "")
    held = {int(e) for e in provenance.get("holdout_episodes") or []}
    key_src = json.dumps({"v": VERSION, "margin": MARGIN, "exports": [(e.get("name"), e.get("labels_fingerprint"),
                          e.get("episodes"), e.get("frames")) for e in exports], "holdout": sorted(held),
                          "fps": contract.fps, "offsets": contract.action_offsets, "frame": frame}, sort_keys=True)
    key = hashlib.sha1(key_src.encode()).hexdigest()[:16]
    cache = (cache_dir or repo_root / "outputs" / "rollout" / "distribution") / f"{key}.npz"
    if cache.is_file():
        return Distribution.load(cache)
    train: list[Episode] = []
    test: list[Episode] = []
    names = []
    for entry in exports:
        root = _export_root(entry, repo_root)
        info = json.loads((root / "meta" / "info.json").read_text(encoding="utf-8"))
        want = (entry.get("episodes"), entry.get("frames"))
        have = (info.get("total_episodes"), info.get("total_frames"))
        if None not in want and tuple(want) != tuple(have):
            raise DistributionError(f"{root.name} here has {have[0]} episodes / {have[1]} frames; the checkpoint "
                                    f"trained on {want[0]} / {want[1]}: not the same export")
        if entry.get("world_frame_id") and frame and entry["world_frame_id"] != frame:
            raise DistributionError(f"{root.name} is in {entry['world_frame_id']}, the workspace in {frame}")
        for arrays in rtd.load_episode_arrays(root, contract):
            ep = Episode(int(arrays.episode), np.asarray(arrays.poses, dtype=np.float64), np.asarray(arrays.widths, dtype=np.float64))
            (test if ep.index in held else train).append(ep)
        names.append(root.name)
    dist = build(train, fps=float(contract.fps), offsets=contract.action_offsets, frame=frame, holdout=test,
                 source={"exports": names, "holdout_episodes": sorted(held), "cache": cache.name})
    dist.save(cache)
    return dist
