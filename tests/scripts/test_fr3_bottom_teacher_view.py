"""The bottom teacher's view: only the slide and what follows it are learnt (11-11)."""

import json

import numpy as np
import pandas as pd
import pytest

from tools.fr3.bottom_teacher_view import build_bottom_teacher_view, supervised_frames

H = 4


def _episode(offset_steps=30, slide_dz=-2.0, close_at=None):
    """z (m) and actions of a teacher episode: 30 steps down 1 mm each at the offset from
    +60 mm to the slide height, 12 slide steps, the last descent to -6, the close, 5 settle steps."""

    z_close = 0.052
    slide = z_close + (slide_dz + 6.0) / 1000.0
    zs = list(np.linspace(z_close + 0.060, slide, offset_steps)) + [slide] * 12
    zs += list(np.arange(slide - 0.001, z_close, -0.001)) + [z_close] * 6
    zs = np.asarray(zs)
    actions = np.zeros((len(zs), 5), dtype=np.float32)
    actions[:, 4] = 1.0
    close = len(zs) - 6 if close_at is None else close_at
    actions[close:, 4] = 0.0
    return zs, actions, close


def test_frames_above_the_slide_height_are_not_learnt_and_the_bottom_is():
    zs, actions, close = _episode(slide_dz=-2.0)
    keep = supervised_frames(zs, actions, {"bottomSlideDzMm": -2.0, "slideEndStep": 50}, margin_mm=3.0)
    cut = zs[close] + 0.007
    assert not keep[zs > cut + 1e-9][: (zs[:close] > cut + 1e-9).sum()].any()
    assert keep[close:].all()
    # The 12 slide frames and the last descent are learnt.
    assert keep[30:close].all()
    assert keep.sum() < len(zs) and keep[:20].sum() == 0


@pytest.mark.parametrize("row", [{"bottomAbandoned": "contact"}, {"slideEndStep": None}])
def test_an_abandoned_or_unslid_trial_is_not_learnt_at_all(row):
    zs, actions, _ = _episode()
    keep = supervised_frames(zs, actions, {"bottomSlideDzMm": -2.0, "slideEndStep": 50, **row}, margin_mm=3.0)
    assert not keep.any()


def _view(tmp_path, episodes, existing=None):
    root = tmp_path / "view"
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "videos").mkdir()
    rows, metas, index, manifest = [], [], 0, []
    for ep, (source, source_ep, zs, actions) in enumerate(episodes):
        start = index
        for t, (z, a) in enumerate(zip(zs, actions, strict=True)):
            rows.append({
                "index": index, "episode_index": ep, "frame_index": t,
                "observation.state": np.asarray([0.4, -0.1, z, 0, 0, 0, 1, 1.0], dtype=np.float32),
                "action": a, "is_intervention": 1.0,
            })
            index += 1
        metas.append({"episode_index": ep, "dataset_from_index": start, "dataset_to_index": index})
        manifest.append({"episode_index": ep, "source_dataset_root": f"/data/{source}", "source_episode_index": source_ep})
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame(metas).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "il_view_manifest.json").write_text(json.dumps({"episode_source_index": manifest}))
    if existing is not None:
        np.savez(root / "meta" / "action_overrides.npz", **existing)
    return root


def _rows(tmp_path, entries):
    path = tmp_path / "grasp_20261112_100000.jsonl"
    lines = [{"kind": "run_start"}]
    for trial, (episode, extra) in enumerate(entries):
        lines.append({
            "kind": "trial", "trial": trial, "verdict": "held",
            "dataset": {"kept": True, "firstEpisode": episode, "root": "/data/teacher"},
            "bottomOffsetMm": [9.0, 1.0], "bottomSlideDzMm": -2.0, "slideEndStep": 50, "bottomAbandoned": "",
            **extra,
        })
    path.write_text("\n".join(json.dumps(line) for line in lines))
    return path


def test_the_view_masks_teacher_frames_keeps_other_episodes_and_carries_existing_overrides(tmp_path):
    zs, actions, _ = _episode()
    old_z = np.linspace(0.3, 0.05, 20)
    old_actions = np.zeros((20, 5), dtype=np.float32)
    existing = {
        "index": np.asarray([3, 5]),
        "action": np.full((2, H, 5), 0.5, dtype=np.float32),
        "action_is_pad": np.zeros((2, H), dtype=bool),
    }
    view = _view(
        tmp_path,
        [("r4_old", 0, old_z, old_actions), ("teacher", 0, zs, actions), ("teacher", 1, zs, actions)],
        existing=existing,
    )
    rows = _rows(tmp_path, [(0, {}), (1, {"bottomAbandoned": "contact"}), (7, {})])
    out = tmp_path / "out"
    summary = build_bottom_teacher_view(view, "teacher", [rows], out, margin_mm=3.0)
    assert summary["teacher_episodes"] == 2 and summary["abandoned_episodes"] == 1
    assert summary["teacher_rows_without_episode_in_view"] == [7]
    with np.load(out / "meta" / "action_overrides.npz") as data:
        index, action, pad = data["index"], data["action"], data["action_is_pad"]
    assert list(index) == sorted(index)
    # The relabelled view's own overrides are still there.
    for i in (3, 5):
        pos = int(np.searchsorted(index, i))
        assert index[pos] == i and not pad[pos].any() and action[pos][0, 0] == pytest.approx(0.5)
    n = len(zs)
    first, second = 20, 20 + n
    masked = set(index[pad.all(axis=1)].tolist())
    keep = supervised_frames(zs, actions, {"bottomSlideDzMm": -2.0, "slideEndStep": 50}, 3.0)
    assert masked == {first + t for t in np.flatnonzero(~keep)} | set(range(second, second + n))
    assert (out / "videos").is_symlink()
    assert json.loads((out / "meta" / "bottom_teacher_mask.json").read_text())["masked_frames"] == len(masked)
    with pytest.raises(ValueError, match="exists"):
        build_bottom_teacher_view(view, "teacher", [rows], out)
