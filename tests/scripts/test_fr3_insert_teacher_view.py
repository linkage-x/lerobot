"""The insert teacher's view: only the teacher's lift/carry/insert/release are learnt, and the other
episodes' "stand still after the close" tail is not (11-16)."""

import json

import numpy as np
import pandas as pd
import pytest

from tools.fr3.insert_teacher_view import (
    build_insert_teacher_view,
    teacher_supervised,
    trailing_closed_after_close,
)

H = 4


def _actions(n, close_at=None, reopen_at=None):
    a = np.zeros((n, 5), dtype=np.float32)
    a[:, 4] = 1.0
    if close_at is not None:
        a[close_at:, 4] = 0.0
    if reopen_at is not None:
        a[reopen_at:, 4] = 1.0
    return a


def test_teacher_frames_are_learnt_policy_frames_are_not():
    flags = np.r_[np.zeros(6), np.ones(4)]
    assert teacher_supervised(flags).tolist() == [False] * 6 + [True] * 4
    with pytest.raises(ValueError, match="one run"):
        teacher_supervised(np.r_[np.zeros(3), np.ones(2), np.zeros(1), np.ones(2)])
    with pytest.raises(ValueError, match="one run"):
        teacher_supervised(np.r_[np.zeros(3), np.ones(2), np.zeros(2)])


def test_only_the_closed_tail_after_the_last_close_is_masked():
    # Ends closed: the close frame is learnt, the 4 frames after it are not.
    assert np.flatnonzero(trailing_closed_after_close(_actions(10, close_at=5))).tolist() == [6, 7, 8, 9]
    # A regrasp (close, reopen, close again): only the tail after the last close.
    a = _actions(12, close_at=3, reopen_at=6)
    a[9:, 4] = 0.0
    assert np.flatnonzero(trailing_closed_after_close(a)).tolist() == [10, 11]
    # Ends open (a teacher-like episode, or no close): nothing.
    assert not trailing_closed_after_close(_actions(10, close_at=3, reopen_at=8)).any()
    assert not trailing_closed_after_close(_actions(10)).any()


def _view(tmp_path, episodes, existing=None, name="view", flags_in_data=False):
    root = tmp_path / name
    root.parent.mkdir(parents=True, exist_ok=True)
    (root / "data" / "chunk-000").mkdir(parents=True)
    (root / "meta" / "episodes" / "chunk-000").mkdir(parents=True)
    (root / "videos").mkdir()
    rows, metas, index, manifest = [], [], 0, []
    for ep, (source, source_ep, actions, flags) in enumerate(episodes):
        start = index
        for t, (a, f) in enumerate(zip(actions, flags, strict=True)):
            rows.append({
                "index": index, "episode_index": ep, "frame_index": t,
                "observation.state": np.asarray([0.4, -0.1, 0.1, 0, 0, 0, 1, 1.0], dtype=np.float32),
                "action": a, "is_intervention": float(f) if flags_in_data else 0.0,
            })
            index += 1
        metas.append({"episode_index": ep, "dataset_from_index": start, "dataset_to_index": index})
        manifest.append({"episode_index": ep, "source_dataset_root": f"outputs/datasets/{source}", "source_episode_index": source_ep})
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    pd.DataFrame(metas).to_parquet(root / "meta" / "episodes" / "chunk-000" / "file-000.parquet")
    (root / "meta" / "il_view_manifest.json").write_text(json.dumps({"episode_source_index": manifest}))
    if existing is not None:
        np.savez(root / "meta" / "action_overrides.npz", **existing)
    return root


def test_the_view_masks_policy_frames_and_closed_tails_and_carries_existing_overrides(tmp_path):
    # R4-like grasp episode: 10 frames, close at 6, stands closed to the end (frames 7-9 masked).
    r4 = (_actions(10, close_at=6), np.zeros(10))
    # Teacher episode: policy 0-7 (close at 5), teacher 8-13 lifts, carries, and opens at 12.
    teach_actions = _actions(14, close_at=5, reopen_at=12)
    teach_flags = np.r_[np.zeros(8), np.ones(6)]
    existing = {
        "index": np.asarray([2, 8, 11]),  # 8 and 11 sit in the masked R4 tail / teacher's policy part
        "action": np.full((3, H, 5), 0.5, dtype=np.float32),
        "action_is_pad": np.zeros((3, H), dtype=bool),
    }
    # The merge writes is_intervention 0 for a scripted expert: the flags come from the source.
    view = _view(
        tmp_path,
        [("r4_old", 0, *r4), ("trial_teacher_insert1_r4", 0, teach_actions, teach_flags),
         ("trial_teacher_insert1_r4", 0, teach_actions, teach_flags)],
        existing=existing,
    )
    source = _view(tmp_path, [("x", 0, teach_actions, teach_flags)], name="trial_teacher_insert1_r4", flags_in_data=True)
    out = tmp_path / "out"
    summary = build_insert_teacher_view(view, [source], out)
    assert summary["teacher_episodes"] == 2 and summary["supervised_teacher_frames"] == 12
    assert summary["teacher_episodes_ending_open"] == 2
    assert summary["other_frames_masked_after_close"] == 3 and summary["carried_overrides"] == 1
    with np.load(out / "meta" / "action_overrides.npz") as data:
        index, action, pad = data["index"], data["action"], data["action_is_pad"]
    assert list(index) == sorted(index)
    masked = set(index[pad.all(axis=1)].tolist())
    assert masked == {7, 8, 9} | set(range(10, 18)) | set(range(24, 32))
    pos = int(np.searchsorted(index, 2))
    assert index[pos] == 2 and not pad[pos].any() and action[pos][0, 0] == pytest.approx(0.5)
    assert (out / "videos").is_symlink()
    assert json.loads((out / "meta" / "insert_teacher_mask.json").read_text())["masked_frames"] == len(masked)
    with pytest.raises(ValueError, match="exists"):
        build_insert_teacher_view(view, [source], out)
    other = _view(tmp_path, [("x", 0, teach_actions, teach_flags)], name="nope", flags_in_data=True)
    with pytest.raises(ValueError, match="not in"):
        build_insert_teacher_view(view, [other], tmp_path / "out2")
    short = _view(tmp_path, [("x", 0, teach_actions[:-1], teach_flags[:-1])], name="s/trial_teacher_insert1_r4", flags_in_data=True)
    with pytest.raises(ValueError, match="frame for frame"):
        build_insert_teacher_view(view, [short], tmp_path / "out3")
