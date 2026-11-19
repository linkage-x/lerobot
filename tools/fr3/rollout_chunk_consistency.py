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
                log=print) -> dict[str, Any]:
    """Every ``stride`` frames of every episode: the predicted TCP positions (world), from the full frames."""
    from tools.fr3 import relative_tcp_policy as rtp

    loaded = rtp.load_checkpoint(checkpoint, device=device)
    c = loaded.contract
    held = set(loaded.provenance.get("holdout_episodes") or [])
    full = rd.ViewVideos(export, c.cameras)
    rows = []
    for arrays in rd.load_episode_arrays(export, c):
        for t in range(0, len(arrays.poses), stride):
            hist_i, _ = rd._clamped(len(arrays.poses), t, c.history_offsets)
            poses = arrays.poses[hist_i]
            if not np.all(np.isfinite(poses)):
                continue
            T_hist = rt.pose7_to_matrix(poses)
            wrench = arrays.wrench[t] if c.wrench_names else None
            images = {cam.name: full.frame(arrays.episode, t, cam.name) for cam in c.cameras}
            pred = loaded.predict(rtp.observation_sample(c, T_hist, float(arrays.widths[t]), images, wrench))
            world, _ = rt.targets_from_actions(T_hist[-1], pred)
            rows.append({"episode": int(arrays.episode), "frame": t, "held_out": int(arrays.episode) in held,
                         "pred": world[:, :3, 3].round(5).tolist()})
        log(f"episode {arrays.episode}: {len(rows)} predictions")
    return {"checkpoint": str(checkpoint), "export": str(export), "stride": stride, "frame_stride": c.frame_stride,
            "rows": rows}


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
    args = ap.parse_args(argv)
    result = predictions(args.checkpoint, args.export, stride=args.stride, device=args.device)
    result["summary_mm"] = summary(gaps(result, at=args.at))
    args.out.write_text(json.dumps(result), encoding="utf-8")
    print(json.dumps(result["summary_mm"], indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
