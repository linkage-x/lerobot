#!/usr/bin/env python3
# ruff: noqa: N806  -- T: the usual name of a transform
"""How far apart two consecutive predictions of a relative-TCP policy are at the same instant -- offline, on an
export (no arm, no cameras): the takeover gap rollout_execute.Limits.switch_mm is set from.

Online a new chunk is predicted every ~0.43 s (13 policy steps at 30 Hz) and takes over the running one about 3
steps after its observation. Here the policy predicts every ``--stride`` frames of each episode, each chunk
anchored at the hand's tracked pose there; the gap of a pair is the distance between the new chunk's step
``--at`` and the old chunk's step for the same instant.

    python tools/fr3/rollout_chunk_consistency.py --checkpoint <pretrained_model> --export <export> --out gaps.json

11-19, stack_act_wrist_v1 030000 on its training export: training episodes p50 8.1, p95 22.7, p99.5 33.8 mm;
held-out episodes p50 31 mm.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _p in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from tools.fr3 import relative_tcp as rt, relative_tcp_data as rd  # noqa: E402

TAKEOVER_STRIDE = 26  # frames at 60 fps: 0.433 s, the takeover interval measured online (p50 411-429 ms)
TAKEOVER_STEP = 3  # the new chunk's step at the switch (k0 2-4 online)


def predictions(checkpoint: Path, export: Path, *, stride: int = TAKEOVER_STRIDE, device: str = "cpu",
                rtc_delay: int | None = None, rtc_horizon: int | None = None, log=print) -> dict[str, Any]:
    """Every ``stride`` frames of every episode: the predicted TCP positions (world), from the full frames,
    and each step's distance to where the hand really was then (``err_mm``, None past the episode's end).

    With ``rtc_delay`` (a flow-matching checkpoint): each prediction is guided by the one before it
    (real-time chunking), as execution would -- the previous chunk's targets from this instant on,
    re-anchored at this frame's TCP, ``rtc_delay`` steps frozen, guidance fading out at ``rtc_horizon``.
    Offline the hand follows the demonstration, not the previous chunk: the prefix is what the arm would
    have been told, the anchor where the demonstrator's hand was."""
    from tools.fr3 import relative_tcp_policy as rtp

    loaded = rtp.load_checkpoint(checkpoint, device=device)
    c = loaded.contract
    if rtc_delay is not None and not getattr(loaded, "rtc", False):
        raise ValueError(f"{checkpoint}: not a flow-matching policy, it has no real-time chunking")
    lag = stride // c.frame_stride
    if rtc_delay is not None and not 0 <= rtc_delay <= (rtc_horizon or 0) <= c.horizon - lag:
        raise ValueError(f"RTC: need 0 <= delay <= horizon <= {c.horizon - lag} (the previous chunk's steps left)")
    held = set(loaded.provenance.get("holdout_episodes") or [])
    full = rd.ViewVideos(export, c.cameras)
    rows = []
    for arrays in rd.load_episode_arrays(export, c):
        n = len(arrays.poses)
        prev: tuple[int, np.ndarray, np.ndarray] | None = None  # frame, world targets, widths
        for t in range(0, n, stride):
            hist_i, _ = rd._clamped(n, t, c.history_offsets)
            poses = arrays.poses[hist_i]
            if not np.all(np.isfinite(poses)):
                prev = None
                continue
            T_hist = rt.pose7_to_matrix(poses)
            wrench = arrays.wrench[t] if c.wrench_names else None
            images = {cam.name: full.frame(arrays.episode, t, cam.name) for cam in c.cameras}
            sample = rtp.observation_sample(c, T_hist, float(arrays.widths[t]), images, wrench)
            guided = rtc_delay is not None and prev is not None and prev[0] == t - stride
            if guided:
                prefix = rtp.rtc_prefix(T_hist[-1], prev[1][lag:], prev[2][lag:])
                sample = rtp.with_rtc_prefix(sample, prefix, delay=rtc_delay, horizon=rtc_horizon)
            pred = loaded.predict(sample)
            world, widths = rt.targets_from_actions(T_hist[-1], pred)
            err = []
            for k, off in enumerate(c.action_offsets):
                gt = arrays.poses[t + off, :3] if t + off < n else None
                err.append(None if gt is None or not np.all(np.isfinite(gt))
                           else round(float(np.linalg.norm(world[k, :3, 3] - gt) * 1e3), 2))
            rows.append({"episode": int(arrays.episode), "frame": t, "held_out": int(arrays.episode) in held,
                         "guided": guided, "pred": world[:, :3, 3].round(5).tolist(), "err_mm": err})
            prev = (t, world, widths)
        log(f"episode {arrays.episode}: {len(rows)} predictions")
    return {"checkpoint": str(checkpoint), "export": str(export), "stride": stride, "frame_stride": c.frame_stride,
            "rtc": None if rtc_delay is None else {"delay": rtc_delay, "horizon": rtc_horizon}, "rows": rows}


def accuracy(result: dict[str, Any], *, steps: tuple[int, int]) -> dict[str, Any]:
    """Mean distance to the hand's real path over steps ``steps[0]..steps[1]`` (1-based), per split: what a
    guided chunk may cost in following the demonstration (guidance pulls it toward the previous chunk)."""
    out: dict[str, list[float]] = {"training": [], "held_out": []}
    for r in result["rows"]:
        e = [v for v in r["err_mm"][steps[0] - 1: steps[1]] if v is not None]
        if e:
            out["held_out" if r["held_out"] else "training"].append(float(np.mean(e)))
    return {k: {"mean": round(float(np.mean(v)), 1), "p95": round(float(np.percentile(v, 95)), 1), "n": len(v)}
            for k, v in out.items() if v}


def gaps(result: dict[str, Any], *, at: int = TAKEOVER_STEP) -> dict[str, np.ndarray]:
    """mm per consecutive pair, training and held-out episodes apart."""
    lag = result["stride"] // result["frame_stride"]
    rows = result["rows"]
    out: dict[str, list[float]] = {"training": [], "held_out": []}
    for a, b in zip(rows, rows[1:], strict=False):
        if a["episode"] != b["episode"] or b["frame"] - a["frame"] != result["stride"] or lag + at > len(a["pred"]):
            continue
        gap = np.linalg.norm(np.asarray(b["pred"][at - 1]) - np.asarray(a["pred"][lag + at - 1])) * 1e3
        out["held_out" if a["held_out"] else "training"].append(float(gap))
    return {k: np.asarray(v) for k, v in out.items()}


def summary(g: dict[str, np.ndarray]) -> dict[str, Any]:
    return {k: ({q: round(float(np.percentile(v, q)), 1) for q in (50, 95, 99, 99.5)} | {"max": round(float(v.max()), 1), "n": int(len(v))})
            for k, v in g.items() if len(v)}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", type=Path, required=True)
    ap.add_argument("--export", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=TAKEOVER_STRIDE)
    ap.add_argument("--at", type=int, default=TAKEOVER_STEP)
    ap.add_argument("--device", default="cpu", help="cpu by default: not to compete with a live run's GPU")
    ap.add_argument("--rtc-delay", type=int, default=None, help="guide each chunk by the previous one (RTC), "
                    "this many steps frozen (execution: the takeover step - 1)")
    ap.add_argument("--rtc-horizon", type=int, default=None, help="where RTC guidance fades to 0 (delay + the "
                    "steps a chunk executes)")
    args = ap.parse_args(argv)
    if (args.rtc_delay is None) != (args.rtc_horizon is None):
        ap.error("--rtc-delay and --rtc-horizon go together")
    result = predictions(args.checkpoint, args.export, stride=args.stride, device=args.device,
                         rtc_delay=args.rtc_delay, rtc_horizon=args.rtc_horizon)
    result["summary_mm"] = summary(gaps(result, at=args.at))
    result["accuracy_mm"] = accuracy(result, steps=(args.at, min(len(result["rows"][0]["pred"]), args.at + 15)))
    args.out.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps({"gap": result["summary_mm"], "accuracy": result["accuracy_mm"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
