#!/usr/bin/env python
"""Supervise only the insert teacher's part of whole-trial insert recordings in a training view (11-16).

The insert-teacher recordings (`--grasp-loop-after-close teacher` + record whole trial, datasets
`trial_teacher_insert*_r4`) hold one episode per trial the teacher seated: the policy's own grasp
(R4 045000, `is_intervention` 0) and then the teacher's lift, carry, insert and release
(`is_intervention` 1, contiguous to the end). The flag is read from the *source* dataset: the merge
writes 0 for a scripted expert's frames (`labelsEveryActionDim`) so that training keeps the
gripper in the loss -- the teacher's release is a label we want. Only the teacher part is learnt here:

* the policy part is R4's own uncorrected steps -- learning them is self-imitation, and with the
  teacher sources merged k times it would weigh R4's grasps k-fold (the 10-23 lesson: a policy's own
  steps, fully supervised, teach its errors back);
* every other episode of the view (R4's grasp-only data) ends ~10 frames after its close, standing
  still with the gripper shut. Those frames say "after the close, stay", the opposite of the
  teacher's "after the close, lift", on frames that look the same. So the trailing closed run after
  the close frame is left unsupervised there too (the close frame itself is still learnt).

"Unsupervised" = the frame's chunk all padding, as `funnel_relabel`'s height mask and
`bottom_teacher_view` do. Existing overrides (a relabelled view) are carried over unless the frame
is masked here.

Output: a new view beside the source view (meta and data copied, videos linked) with
`meta/action_overrides.npz`, plus `meta/insert_teacher_mask.json` saying what was masked.
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
    _read_frames,
    _view_episode_starts,
)

MASK_SUMMARY_RELATIVE_PATH = Path("meta") / "insert_teacher_mask.json"
CLOSED_BELOW = 0.5


def teacher_supervised(is_intervention: np.ndarray) -> np.ndarray:
    """Per frame of one teacher episode: True where it is learnt (the teacher's frames)."""

    keep = np.asarray(is_intervention).astype(bool)
    hits = np.flatnonzero(keep)
    if len(hits) and (hits[-1] != len(keep) - 1 or hits[-1] - hits[0] + 1 != len(hits)):
        raise ValueError("the teacher's frames are not one run to the episode's end")
    return keep


def trailing_closed_after_close(actions: np.ndarray) -> np.ndarray:
    """Per frame: True for the frames after the close in an episode that ends closed (its last run)."""

    closed = np.asarray(actions)[:, 4] < CLOSED_BELOW
    out = np.zeros(len(closed), dtype=bool)
    if not len(closed) or not closed[-1]:
        return out
    opened = np.flatnonzero(~closed)
    onset = int(opened[-1]) + 1 if len(opened) else 0
    out[onset + 1 :] = True
    return out


def _source_flags(source_root: Path) -> dict[int, tuple[np.ndarray, np.ndarray]]:
    """Per source episode: its `is_intervention` flags and states, in frame order."""

    frames = _read_frames(source_root)
    out = {}
    for ep, d in frames.groupby("episode_index"):
        d = d.sort_values("frame_index")
        out[int(ep)] = (d["is_intervention"].to_numpy(), np.stack(d["observation.state"].to_numpy()))
    return out


def build_insert_teacher_view(
    view_root: Path,
    teacher_sources: list[Path],
    out_root: Path,
    *,
    mask_after_close: bool = True,
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

    starts = _view_episode_starts(view_root)
    frames = _read_frames(view_root)
    action = np.stack(frames["action"].to_numpy())
    state = np.stack(frames["observation.state"].to_numpy())
    index = frames["index"].to_numpy()
    action_dim = action.shape[1]
    flags = {Path(root).name: _source_flags(Path(root)) for root in teacher_sources}
    teacher_names = set(flags)

    masked: set[int] = set()
    per_teacher_episode = []
    other_masked = 0
    other_episodes_masked = 0
    seen_teacher_sources: set[str] = set()
    for entry in manifest["episode_source_index"]:
        source = Path(entry["source_dataset_root"]).name
        view_ep = int(entry["episode_index"])
        start, end = starts[view_ep]
        if source in teacher_names:
            seen_teacher_sources.add(source)
            source_flags, source_state = flags[source][int(entry["source_episode_index"])]
            if len(source_flags) != end - start or not np.allclose(source_state, state[start:end], atol=1e-6):
                raise ValueError(f"view episode {view_ep} is not {source}:{entry['source_episode_index']} frame for frame")
            keep = teacher_supervised(source_flags)
            drop = ~keep
            per_teacher_episode.append(
                {
                    "view_episode": view_ep,
                    "source": source,
                    "source_episode": int(entry["source_episode_index"]),
                    "frames": int(end - start),
                    "supervised": int(keep.sum()),
                    "ends_open": bool(action[end - 1, 4] >= CLOSED_BELOW),
                }
            )
        elif mask_after_close:
            drop = trailing_closed_after_close(action[start:end])
            if drop.any():
                other_masked += int(drop.sum())
                other_episodes_masked += 1
        else:
            continue
        masked.update(int(i) for i in index[start:end][drop])
    missing = sorted(teacher_names - seen_teacher_sources)
    if missing:
        raise ValueError(f"teacher sources not in {view_root.name}: {missing}")

    override_index: list[int] = []
    override_action: list[np.ndarray] = []
    override_pad: list[np.ndarray] = []
    carried = 0
    if existing is not None:
        for i, abs_idx in enumerate(existing["index"]):
            if int(abs_idx) in masked:
                continue  # ours win: a masked frame is unsupervised, whatever it carried
            carried += 1
            override_index.append(int(abs_idx))
            override_action.append(existing["action"][i])
            override_pad.append(existing["action_is_pad"][i])
    for abs_idx in sorted(masked):
        override_index.append(abs_idx)
        override_action.append(np.zeros((horizon, action_dim), dtype=np.float32))
        override_pad.append(np.ones(horizon, dtype=bool))
    order = np.argsort(np.asarray(override_index))
    overrides = {
        "index": np.asarray(override_index, dtype=np.int64)[order],
        "action": np.asarray(override_action, dtype=np.float32).reshape(-1, horizon, action_dim)[order],
        "action_is_pad": np.asarray(override_pad, dtype=bool).reshape(-1, horizon)[order],
    }

    out_root.parent.mkdir(parents=True, exist_ok=True)
    shutil.copytree(view_root / "meta", out_root / "meta")
    shutil.copytree(view_root / "data", out_root / "data")
    (out_root / "videos").symlink_to((view_root / "videos").resolve(), target_is_directory=True)
    np.savez(out_root / ACTION_OVERRIDES_RELATIVE_PATH, **overrides)
    summary = {
        "source_view": str(view_root),
        "teacher_sources": sorted(teacher_names),
        "mask_after_close": mask_after_close,
        "horizon": horizon,
        "teacher_episodes": len(per_teacher_episode),
        "teacher_frames": sum(ep["frames"] for ep in per_teacher_episode),
        "supervised_teacher_frames": sum(ep["supervised"] for ep in per_teacher_episode),
        "teacher_episodes_ending_open": sum(ep["ends_open"] for ep in per_teacher_episode),
        "other_episodes_masked_after_close": other_episodes_masked,
        "other_frames_masked_after_close": other_masked,
        "masked_frames": len(masked),
        "carried_overrides": carried,
        "episodes": per_teacher_episode,
    }
    (out_root / MASK_SUMMARY_RELATIVE_PATH).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--view", type=Path, required=True, help="training view holding the teacher's episodes")
    parser.add_argument(
        "--teacher-source", action="append", type=Path, required=True,
        help="a teacher dataset's root (its directory name is the one in il_view_manifest)",
    )
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--keep-after-close", action="store_true", help="do not mask the other episodes' trailing closed frames")
    parser.add_argument("--horizon", type=int, default=None, help="chunk length, when the view has no overrides yet")
    args = parser.parse_args()
    summary = build_insert_teacher_view(
        args.view, args.teacher_source, args.out, mask_after_close=not args.keep_after_close, horizon=args.horizon
    )
    print(json.dumps({k: v for k, v in summary.items() if k != "episodes"}, indent=1))


if __name__ == "__main__":
    main()
