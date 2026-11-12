#!/usr/bin/env python
"""Supervise only the bottom of the bottom teacher's episodes in a training view (11-11).

The bottom teacher (`grasp_funnel` `bottomOffsetMm`, grasp loop `--grasp-loop-funnel-bottom-teacher`)
lines up 3-12 mm beside the peg and descends straight there before it slides across at the bottom.
Everything above the slide height is therefore the one behaviour the teacher exists to undo: a
straight descent beside the peg (and, at the takeover height, a move *away* from the peg to the
offset). Trained on as recorded, those frames teach the dead band once more. So in each teacher
episode every frame above the slide height (plus `margin_mm`) before the close is left
unsupervised -- its chunk all padding, as `funnel_relabel`'s height mask does -- and only the
slide, the last centred descent, the close and its settle steps are learnt. A trial whose teacher
was abandoned (`bottomAbandoned`: a finger on something, a realign) or never slid is unsupervised
whole: its retreat is a lift, the mode R9 failed with.

Heights are read inside each episode, relative to its own close frame, so neither the view's frame
nor the peg record enters: the slide stands `bottomSlideDzMm - closeDzMm` above the close.

The rows are matched to episodes by their `dataset.firstEpisode` on the teacher's source dataset
(funnel-only recordings: one episode per held trial). Other episodes of the view, and any action
overrides it already has (a relabelled view), are carried over unchanged.

Output: a new view beside the source view (meta and data copied, videos linked) with
`meta/action_overrides.npz`, plus `meta/bottom_teacher_mask.json` saying what was masked.
"""

from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Any

import numpy as np

_REPO_ROOT = Path(__file__).resolve().parents[2]
for _import_root in (str(_REPO_ROOT / "src"), str(_REPO_ROOT)):
    if _import_root not in sys.path:
        sys.path.insert(0, _import_root)

from tools.fr3.funnel_relabel import (  # noqa: E402
    ACTION_OVERRIDES_RELATIVE_PATH,
    _close_index,
    _read_frames,
    _view_episode_starts,
)
from tools.fr3.grasp_funnel import FUNNEL_CLOSE_DZ_MM  # noqa: E402

MASK_SUMMARY_RELATIVE_PATH = Path("meta") / "bottom_teacher_mask.json"
# How far above the slide height a frame may stand and still be learnt: the arm reaches the slide
# height ~1-2 mm after its setpoint does, and the frames just above it already carry the slide in
# their chunk.
DEFAULT_MARGIN_MM = 3.0


def teacher_rows(paths: list[Path]) -> dict[int, dict[str, Any]]:
    """Teacher rows that became an episode, by that episode's index in the teacher's dataset."""

    rows: dict[int, dict[str, Any]] = {}
    for path in paths:
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if not line.strip():
                continue
            row = json.loads(line)
            kept = row.get("dataset") if row.get("kind") == "trial" else None
            if not kept or not kept.get("kept") or kept.get("firstEpisode") is None:
                continue
            if "bottomOffsetMm" not in row:
                continue  # a regrip/repick trial: the plain funnel, nothing to mask
            episode = int(kept["firstEpisode"])
            if episode in rows:
                raise ValueError(f"two teacher rows claim episode {episode} ({path})")
            rows[episode] = {**row, "_run": Path(path).stem}
    return rows


def supervised_frames(state_z: np.ndarray, actions: np.ndarray, row: dict[str, Any], margin_mm: float) -> np.ndarray:
    """Per frame of one teacher episode: True where it is learnt."""

    n = len(state_z)
    if row.get("bottomAbandoned") or row.get("slideEndStep") is None:
        return np.zeros(n, dtype=bool)
    close = _close_index(actions)
    if close is None:
        return np.zeros(n, dtype=bool)
    slide_above_close_mm = float(row["bottomSlideDzMm"]) - FUNNEL_CLOSE_DZ_MM
    cut = state_z[close] + (slide_above_close_mm + margin_mm) / 1000.0
    keep = np.ones(n, dtype=bool)
    keep[:close] = state_z[:close] <= cut
    # Only what follows the first frame down there: the arm does not come back up before the
    # close, so a frame under the cut early on would be a glitch, not the bottom.
    low = np.flatnonzero(keep[:close])
    if len(low):
        keep[: low[0]] = False
    return keep


def build_bottom_teacher_view(
    view_root: Path,
    teacher_source: str,
    row_paths: list[Path],
    out_root: Path,
    *,
    margin_mm: float = DEFAULT_MARGIN_MM,
    horizon: int | None = None,
) -> dict[str, Any]:
    if out_root.exists():
        raise ValueError(f"{out_root} exists; pick a new name rather than overwrite a view")
    manifest = json.loads((view_root / "meta" / "il_view_manifest.json").read_text(encoding="utf-8"))
    existing_path = view_root / ACTION_OVERRIDES_RELATIVE_PATH
    existing: dict[str, np.ndarray] | None = None
    if existing_path.is_file():
        with np.load(existing_path) as data:
            existing = {key: data[key] for key in ("index", "action", "action_is_pad")}
        if horizon is not None and existing["action"].shape[1] != horizon:
            raise ValueError(f"--horizon {horizon}, but the view's overrides hold {existing['action'].shape[1]}-step chunks")
        horizon = int(existing["action"].shape[1])
    if horizon is None:
        raise ValueError("the view has no action overrides to read the chunk length from: pass horizon")

    rows = teacher_rows(row_paths)
    starts = _view_episode_starts(view_root)
    frames = _read_frames(view_root)
    state = np.stack(frames["observation.state"].to_numpy())
    action = np.stack(frames["action"].to_numpy())
    index = frames["index"].to_numpy()
    action_dim = action.shape[1]

    masked: dict[int, np.ndarray] = {}
    per_episode = []
    matched = set()
    for entry in manifest["episode_source_index"]:
        if Path(entry["source_dataset_root"]).name != teacher_source:
            continue
        source_ep = int(entry["source_episode_index"])
        view_ep = int(entry["episode_index"])
        start, end = starts[view_ep]
        row = rows.get(source_ep)
        if row is None:
            continue
        matched.add(source_ep)
        keep = supervised_frames(state[start:end, 2], action[start:end], row, margin_mm)
        for t in np.flatnonzero(~keep):
            masked[int(index[start + t])] = np.ones(horizon, dtype=bool)
        per_episode.append(
            {
                "view_episode": view_ep,
                "source_episode": source_ep,
                "run": row["_run"],
                "trial": row.get("trial"),
                "frames": int(end - start),
                "supervised": int(keep.sum()),
                "abandoned": row.get("bottomAbandoned") or ("no_slide" if row.get("slideEndStep") is None else ""),
                "offset_mm": row.get("bottomOffsetMm"),
                "slide_dz_mm": row.get("bottomSlideDzMm"),
            }
        )
    unmatched = sorted(set(rows) - matched)
    if not per_episode:
        raise ValueError(f"no episode of {teacher_source} in {view_root.name} matched a teacher row")

    ids = dict.fromkeys(masked)
    override_index = []
    override_action = []
    override_pad = []
    if existing is not None:
        for i, abs_idx in enumerate(existing["index"]):
            if int(abs_idx) in ids:
                continue  # ours win: a teacher frame is masked, whatever it carried
            override_index.append(int(abs_idx))
            override_action.append(existing["action"][i])
            override_pad.append(existing["action_is_pad"][i])
    for abs_idx, pad in masked.items():
        override_index.append(abs_idx)
        override_action.append(np.zeros((horizon, action_dim), dtype=np.float32))
        override_pad.append(pad)
    order = np.argsort(np.asarray(override_index))
    overrides = {
        "index": np.asarray(override_index, dtype=np.int64)[order],
        "action": np.asarray(override_action, dtype=np.float32)[order],
        "action_is_pad": np.asarray(override_pad, dtype=bool)[order],
    }

    out_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(view_root / "meta", out_root / "meta")
    shutil.copytree(view_root / "data", out_root / "data")
    (out_root / "videos").symlink_to((view_root / "videos").resolve(), target_is_directory=True)
    np.savez(out_root / ACTION_OVERRIDES_RELATIVE_PATH, **overrides)
    summary = {
        "source_view": str(view_root),
        "teacher_source": teacher_source,
        "rows": [str(p) for p in row_paths],
        "margin_mm": margin_mm,
        "horizon": horizon,
        "teacher_episodes": len(per_episode),
        "abandoned_episodes": sum(bool(ep["abandoned"]) for ep in per_episode),
        "teacher_frames": sum(ep["frames"] for ep in per_episode),
        "supervised_teacher_frames": sum(ep["supervised"] for ep in per_episode),
        "masked_frames": len(masked),
        "carried_overrides": 0 if existing is None else len(existing["index"]),
        "teacher_rows_without_episode_in_view": unmatched,
        "episodes": per_episode,
    }
    (out_root / MASK_SUMMARY_RELATIVE_PATH).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--view", type=Path, required=True, help="training view holding the teacher's episodes")
    parser.add_argument("--teacher-source", required=True, help="the teacher dataset's directory name, as in il_view_manifest")
    parser.add_argument("--rows", type=Path, nargs="+", required=True, help="the teacher runs' grasp_*.jsonl")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--margin-mm", type=float, default=DEFAULT_MARGIN_MM)
    parser.add_argument("--horizon", type=int, default=None, help="chunk length, when the view has no overrides yet")
    args = parser.parse_args()
    summary = build_bottom_teacher_view(
        args.view, args.teacher_source, args.rows, args.out, margin_mm=args.margin_mm, horizon=args.horizon
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=1))


if __name__ == "__main__":
    main()
