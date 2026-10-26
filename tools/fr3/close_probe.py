#!/usr/bin/env python
"""Close probe: low and off the peg, does a checkpoint lift and correct, or close where it is?

The holdout probe (`holdout_probe.py`) asks about the descent, 40-250 mm up. The robot's misses are
decided lower: 035000 closed with the tool a median 8.8 mm short in x (R3, 171012), and nothing in
training was low and off-centre until R3 was relabelled. This asks about those frames.

Frames: the view's named holdout episodes (`action_overrides.holdout_named_view_episodes`, R1's 16
for R4), policy frames from `--min-dz-mm` to 60 mm above the peg. Each is classed by what the funnel
label does there: `lift` (it rises before anything else -- outside the 8 mm capture) or `descend`
(it goes on down and closes). On each, the checkpoint's chunk is read for:

- `lifts`: its first `--lift-steps` steps rise more than 1 mm;
- `closes`: it commands the fingers shut (< 0.5) anywhere in the chunk;
- `along`: its xy motion over the chunk toward the label's, over the label's (1.0 = as far).

Wanted: on `lift` frames, lifts high and closes low; on `descend` frames, closes about as often as
the label. Medians / rates per checkpoint, class and dz band go to `--out` (json), the draws beside
it (parquet). No robot. Run it as a capped side job, like the holdout probe.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from tools.fr3.funnel_relabel import load_trial_rows, relabel_source
from tools.fr3.holdout_probe import LOAD_SPEC, TASK, observations, ps

DZ_BANDS_MM = (-10, 15, 35, 60)
LIFT_MM = 1.0
CLOSED_BELOW = 0.5


def chunk_behaviour(chunk: np.ndarray, lift_steps: int) -> dict[str, float | bool]:
    """What a (H, 5) delta chunk does: rises first, shuts the fingers, and its xy travel."""
    xyz = chunk[:, :3]
    return {
        "lifts": bool(xyz[:lift_steps, 2].sum() * 1000.0 > LIFT_MM),
        "closes": bool((chunk[:, 4] < CLOSED_BELOW).any()),
        "dx": float(xyz[:, 0].sum()),
        "dy": float(xyz[:, 1].sum()),
    }


def close_frames(
    train_root: Path, view_root: Path, trial_rows: list, min_dz_mm: float, per_episode: int, lift_steps: int
) -> pd.DataFrame:
    man = json.loads((view_root / "meta/il_view_manifest.json").read_text())
    ov = man["action_overrides"]
    named = ov.get("holdout_named_view_episodes") or []
    if not named:
        raise SystemExit(f"{view_root.name} names no holdout episodes; nothing to probe on")
    by_view = {e["episode_index"]: e for e in man["episode_source_index"]}
    roots = {Path(by_view[v]["source_dataset_root"]).name for v in named}
    offsets = {Path(s["source_root"]).name: np.asarray(s["close_tracking_offset_mm"]) / 1000.0 for s in ov["sources"]}
    starts = pd.read_parquet(view_root / "meta/episodes", columns=["episode_index", "dataset_from_index"])
    start = dict(zip(starts.episode_index.astype(int), starts.dataset_from_index.astype(int), strict=True))
    lab = np.load(view_root / "meta/action_overrides.npz")
    rows = []
    for name in sorted(roots):
        # The offset the view was relabelled with, so the peg here is the label's peg.
        rel, _ = relabel_source(train_root / "outputs/datasets" / name, int(ov["horizon"]), trial_rows=trial_rows,
                                funnel_offset=offsets[name])  # fmt: skip
        for vep in named:
            if Path(by_view[vep]["source_dataset_root"]).name != name:
                continue
            r = rel.get(int(by_view[vep]["source_episode_index"]))
            if r is None:
                continue
            for t, masked in zip(r["frames"], r["masked"], strict=True):
                st = r["state"][t]
                dz = (st[2] - r["peg"][2]) * 1000
                if masked or not min_dz_mm <= dz <= DZ_BANDS_MM[-1]:
                    continue
                idx = start[vep] + int(t)
                pos = int(np.searchsorted(lab["index"], idx))
                if pos >= len(lab["index"]) or lab["index"][pos] != idx:
                    raise SystemExit(f"view index {idx} has no override: the view and its relabel source disagree")
                b = chunk_behaviour(lab["action"][pos].astype(np.float64), lift_steps)
                rows.append({
                    "vep": vep, "t": int(t), "idx": idx, "dz": dz,
                    "ox": (st[0] - r["peg"][0]) * 1000, "oy": (st[1] - r["peg"][1]) * 1000,
                    "cls": "lift" if b["lifts"] else "descend",
                    "lab_closes": b["closes"], "lx": b["dx"], "ly": b["dy"],
                })  # fmt: skip
    f = pd.DataFrame(rows)
    if f.empty:
        raise SystemExit("no low policy frame in the named holdout episodes")
    # Per episode and class: R1's low frames are ~88% `lift`, and the probe has to see both.
    return (
        f.groupby(["vep", "cls"], group_keys=False)[f.columns]
        .apply(lambda d: d.sample(min(per_episode, len(d)), random_state=0))
        .reset_index(drop=True)
    )


def summarize(p: pd.DataFrame) -> dict[str, dict]:
    p = p.copy()
    lab = p[["lx", "ly"]].to_numpy(dtype=np.float64)
    pred = p[["mx", "my"]].to_numpy(dtype=np.float64)
    lab_len = np.linalg.norm(lab, axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        p["along"] = np.where(lab_len > 1e-3, (pred * lab).sum(1) / lab_len**2, np.nan)
    p["band"] = pd.cut(p.dz, list(DZ_BANDS_MM)).astype(str)
    out: dict[str, dict] = {}
    for ckpt, g in p.groupby("ckpt", sort=False):
        m: dict[str, dict] = {}
        for cls, c in g.groupby("cls"):
            m[cls] = {
                "n": int(len(c)),
                "lifts": round(float(c.lifts.mean()), 3),
                "closes": round(float(c.closes.mean()), 3),
                "label_closes": round(float(c.lab_closes.mean()), 3),
                "along": round(float(c.along.median()), 3) if c.along.notna().any() else None,
                "bands": {
                    band: {"n": int(len(b)), "lifts": round(float(b.lifts.mean()), 3), "closes": round(float(b.closes.mean()), 3)}
                    for band, b in c.groupby("band")
                },
            }  # fmt: skip
        out[ckpt] = m
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--train-root", type=Path, required=True, help="repo the checkpoints and view live in")
    ap.add_argument("--view", required=True, help="relabelled view, relative to --train-root")
    ap.add_argument("--trial-rows", type=Path, action="append", required=True,
                    help="the grasp-loop JSONL(s) that recorded the held-out episodes (their pegs)")  # fmt: skip
    ap.add_argument("--out", type=Path, required=True, help="summary json; raw draws go to <out>.parquet")
    ap.add_argument("--per-episode", type=int, default=6, help="frames per held-out episode and class")
    ap.add_argument("--draws", type=int, default=4)
    ap.add_argument("--min-dz-mm", type=float, default=DZ_BANDS_MM[0])
    ap.add_argument("--lift-steps", type=int, default=10,
                    help="steps a chunk must rise over to count as lifting (labels and predictions alike)")  # fmt: skip
    ap.add_argument("ckpts", nargs="+", help="name=checkpoint, the checkpoint relative to --train-root (outputs/...)")
    args = ap.parse_args()
    train_root = args.train_root.resolve()
    view_root = train_root / args.view
    ckpts = dict(a.split("=", 1) for a in args.ckpts)

    frames = close_frames(train_root, view_root, load_trial_rows(args.trial_rows), args.min_dz_mm, args.per_episode,
                          args.lift_steps)  # fmt: skip
    print(f"frames {len(frames)} episodes {frames.vep.nunique()} {frames.cls.value_counts().to_dict()}", flush=True)
    obs = observations(frames, view_root)
    dev = torch.device("cuda")
    draws = []
    for name, ckpt in ckpts.items():
        lp = ps.make_loader(train_root, dev)({**LOAD_SPEC, "checkpoint": ckpt, "dataset_root": args.view})
        torch.manual_seed(0)
        t0 = time.time()
        for i, o in enumerate(obs):
            for d in range(args.draws):
                lp.reset()
                _, chunk = lp.predict(
                    dict(o), device=dev, use_amp=lp.use_amp, inference_delay=0, prev_chunk_left_over=None,
                    execution_horizon=LOAD_SPEC["rtc_execution_horizon"], task=TASK,
                )
                a = chunk.detach().float().cpu().numpy().reshape(-1, chunk.shape[-1]).astype(np.float64)
                b = chunk_behaviour(a, args.lift_steps)
                draws.append((name, i, d, b["lifts"], b["closes"], b["dx"], b["dy"]))
        print(f"{name} done {time.time() - t0:.0f}s", flush=True)
        lp.release()
        del lp
        torch.cuda.empty_cache()

    cols = ["ckpt", "fi", "draw", "lifts", "closes", "mx", "my"]
    p = pd.DataFrame(draws, columns=cols).merge(frames, left_on="fi", right_index=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    p.to_parquet(args.out.with_suffix(".parquet"))
    summary = summarize(p)
    tmp = args.out.with_suffix(".tmp")
    tmp.write_text(json.dumps({"view": args.view, "ckpts": ckpts, "summary": summary}, indent=2))
    tmp.replace(args.out)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
