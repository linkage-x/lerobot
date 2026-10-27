"""Offline relabelling of the policy's own steps with the funnel's chunk.

What has to hold for the labels to teach the correction rather than a new mistake: the chunk is
the funnel's path from that frame's *command* (not the recorded future), it reaches the peg's XY
at `centredZ` and never moves sideways below it, it never goes flatter than 45 degrees, and the
fingers close when the funnel's settle test would close them. The dataset side must swap in the
chunk for exactly the listed frames and leave every other frame as recorded.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import torch

from tools.fr3.funnel_relabel import (
    EXPERT_CLOSE_WAIT_STEPS,
    EXPERT_POST_CLOSE_STEPS,
    command_path,
    expert_chunk,
    NoFunnelCloseError,
    load_trial_rows,
    relabel_source,
)
from tools.fr3.grasp_funnel import FUNNEL_CLOSE_DZ_MM, FunnelConfig

PEG = np.array([0.43, -0.15, 0.058])


def _path(start, actions):
    return np.asarray(start) + np.cumsum(actions[:, :3].astype(np.float64), axis=0)


def test_a_high_offset_frame_heads_down_the_diagonal_to_the_peg_and_is_over_it_by_centred_z():
    cfg = FunnelConfig(pegXyz=tuple(PEG))
    start = PEG + [0.030, -0.020, 0.150]
    actions, is_pad = expert_chunk(start, PEG, 200, config=cfg)
    path = _path(start, actions)
    xy_err = np.linalg.norm(path[:, :2] - PEG[:2], axis=1) * 1000.0
    above = path[:, 2] > cfg.centredZ + 1e-9
    # every step above centredZ lies on the line from the start to (peg, centredZ); the chunk
    # is float32, so "on" is to a micrometre
    goal = np.array([PEG[0], PEG[1], cfg.centredZ])
    direction = (goal - start) / np.linalg.norm(goal - start)
    rel = path[above] - start
    off_line = np.linalg.norm(rel - np.outer(rel @ direction, direction), axis=1) * 1000.0
    assert off_line.max() < 1e-3
    assert xy_err[~above].max() < 1e-3, "sideways motion below centredZ"
    assert path[-1] == pytest.approx(cfg.closeXyz, abs=1e-6)
    assert np.all(actions[:, 3] == 0.0)
    assert not is_pad[:50].any()


def test_the_correction_starts_at_once_not_after_a_straight_descent():
    start = PEG + [0.030, 0.0, 0.150]
    actions, _ = expert_chunk(start, PEG, 50)
    first_second = actions[:30, :2].sum(axis=0)
    # 30 steps at 0.05 m/s along a line 30 mm across and 115 mm down: ~12.5 mm toward the peg
    assert first_second[0] == pytest.approx(-0.030 * (30 * 0.05 / 30.0) / math.hypot(0.030, 0.115), rel=1e-6)


def test_speeds_are_the_funnels():
    cfg = FunnelConfig(pegXyz=tuple(PEG))
    start = PEG + [0.0, 0.0, 0.200]
    actions, _ = expert_chunk(start, PEG, 400, config=cfg)
    path = _path(start, actions)
    step_mm = np.linalg.norm(actions[:, :3], axis=1) * 1000.0
    fast = path[:, 2] > cfg.slowBelowZ + 0.002
    slow = (path[:, 2] < cfg.slowBelowZ - 0.002) & (step_mm > 0)
    assert step_mm[fast][:-1] == pytest.approx(cfg.fastDescendSpeedMS * cfg.controlPeriodS * 1000.0)
    assert step_mm[slow][:-1] == pytest.approx(cfg.descendSpeedMS * cfg.controlPeriodS * 1000.0)


def test_flatter_than_45_degrees_goes_across_at_its_height_then_down():
    cfg = FunnelConfig(pegXyz=tuple(PEG))
    start = PEG + [0.050, 0.0, 0.080]  # 45 mm above centredZ, 50 mm off
    actions, _ = expert_chunk(start, PEG, 200, config=cfg)
    path = _path(start, actions)
    across = np.linalg.norm(path[:, :2] - PEG[:2], axis=1) > 1e-6
    assert np.allclose(path[across, 2], start[2], atol=1e-6)
    assert path[-1] == pytest.approx(cfg.closeXyz, abs=1e-6)


def test_low_and_outside_capture_goes_up_first():
    cfg = FunnelConfig(pegXyz=tuple(PEG))
    start = PEG + [0.030, 0.0, 0.030]
    actions, _ = expert_chunk(start, PEG, 200, config=cfg)
    path = _path(start, actions)
    first_across = int(np.argmax(np.abs(path[:, 0] - start[0]) > 1e-6))
    assert path[first_across - 1, 2] == pytest.approx(cfg.alignZ, abs=1e-6)
    assert path[:first_across, 0] == pytest.approx(np.full(first_across, start[0]), abs=1e-6)


def test_the_fingers_close_after_the_settle_wait_and_the_tail_is_padding():
    cfg = FunnelConfig(pegXyz=tuple(PEG))
    start = np.array(cfg.closeXyz) + [0.0, 0.0, 0.003]
    actions, is_pad = expert_chunk(start, PEG, 50, config=cfg)
    moving = np.flatnonzero(np.abs(actions[:, :3]).max(axis=1) > 0)
    last_move = int(moving[-1])
    closed = np.flatnonzero(actions[:, 4] < 0.5)
    assert int(closed[0]) - 1 - last_move == EXPERT_CLOSE_WAIT_STEPS
    assert np.all(actions[closed[0]:, 4] == cfg.closedGripper)
    assert np.flatnonzero(is_pad)[0] == closed[0] + EXPERT_POST_CLOSE_STEPS


def test_command_path_is_the_command_each_delta_was_measured_from():
    start = np.array([0.1, 0.2, 0.3])
    actions = np.zeros((4, 5))
    actions[:, 0] = [0.001, 0.002, 0.003, 0.004]
    prev = command_path(actions, start)
    assert prev[0] == pytest.approx(start)
    assert prev[1:, 0] == pytest.approx([0.101, 0.103, 0.106])


def _bare_dataset(tmp_path: Path, chunk: int = 4):
    from lerobot.datasets.lerobot_dataset import LeRobotDataset

    ds = LeRobotDataset.__new__(LeRobotDataset)
    ds.root = tmp_path
    ds.delta_indices = {"action": list(range(chunk))}
    return ds


def test_the_dataset_swaps_in_the_chunk_for_listed_frames_only(tmp_path):
    ds = _bare_dataset(tmp_path)
    (tmp_path / "meta").mkdir()
    override = np.arange(2 * 4 * 5, dtype=np.float32).reshape(2, 4, 5)
    pad = np.zeros((2, 4), dtype=bool)
    pad[1, 3] = True
    np.savez(tmp_path / "meta" / "action_overrides.npz", index=np.array([7, 9]), action=override, action_is_pad=pad)
    recorded = {"action": torch.ones(4, 5), "action_is_pad": torch.tensor([False, False, True, True])}

    untouched = ds._apply_action_override(dict(recorded), 8)
    assert untouched["action"] is recorded["action"]
    swapped = ds._apply_action_override(dict(recorded), 9)
    assert torch.equal(swapped["action"], torch.from_numpy(override[1]))
    assert swapped["action_is_pad"].tolist() == [False, False, False, True]


def test_a_view_without_overrides_is_upstream_behaviour(tmp_path):
    ds = _bare_dataset(tmp_path)
    recorded = {"action": torch.ones(4, 5)}
    assert ds._apply_action_override(recorded, 0) is recorded


def test_a_chunk_length_mismatch_is_refused(tmp_path):
    ds = _bare_dataset(tmp_path, chunk=50)
    (tmp_path / "meta").mkdir()
    np.savez(
        tmp_path / "meta" / "action_overrides.npz",
        index=np.array([0]),
        action=np.zeros((1, 4, 5), np.float32),
        action_is_pad=np.zeros((1, 4), bool),
    )
    with pytest.raises(ValueError, match="4-step chunks"):
        ds._apply_action_override({"action": torch.ones(50, 5)}, 0)


def test_the_training_script_reports_a_views_overrides(tmp_path):
    from tools.fr3 import fr3_train_il_policy

    (tmp_path / "meta").mkdir()
    assert fr3_train_il_policy.report_action_overrides(tmp_path) == 0
    np.savez(
        tmp_path / "meta" / "action_overrides.npz",
        index=np.array([0, 3]),
        action=np.zeros((2, 4, 5), np.float32),
        action_is_pad=np.zeros((2, 4), bool),
    )
    assert fr3_train_il_policy.report_action_overrides(tmp_path) == 2


# ---- a whole-trial source with an arm-A (policy-only) episode -------------------------------

_YAW = math.radians(2.4)
_T_B_WS = np.array(
    [
        [math.cos(_YAW), -math.sin(_YAW), 0.0, -0.065],
        [math.sin(_YAW), math.cos(_YAW), 0.0, 0.148],
        [0.0, 0.0, 1.0, 0.294],
        [0.0, 0.0, 0.0, 1.0],
    ]
)
_PEG_BASE = np.array([0.43, -0.08, 0.058])
_TRACKING = np.array([0.0038, -0.0018, -0.0032])  # command minus measured, as on the rig
_CLOSE_DZ = np.array([0.0, 0.0, FUNNEL_CLOSE_DZ_MM / 1000.0])


def _to_base(xyz):
    return _T_B_WS[:3, :3] @ np.asarray(xyz) + _T_B_WS[:3, 3]


def _episode(commands, close_at, intervention_from=None, tail=10, tracking=_TRACKING):
    """Frames whose measured pose trails the command by `tracking`; closes at `close_at` (None:
    never, `tail` still steps past the last move)."""

    commands = [np.asarray(c, dtype=np.float64) for c in commands]
    commands += [commands[-1]] * ((len(commands) - 1 if close_at is None else close_at) + 1 + tail - len(commands))
    actions = np.zeros((len(commands) - 1, 5))
    actions[:, :3] = np.diff(np.asarray(commands), axis=0)
    actions[:, 4] = 1.0
    if close_at is not None:
        actions[close_at:, 4] = 0.0
    prev = np.asarray(commands[:-1])
    state = np.zeros((len(actions), 8))
    state[:, :3] = prev - tracking
    state[:, 6] = 1.0
    flags = np.zeros(len(actions))
    if intervention_from is not None:
        flags[intervention_from:] = 1.0
    return commands[0], state, actions, flags


def _walk_to(start, goal, step=0.001):
    start, goal = np.asarray(start, float), np.asarray(goal, float)
    n = max(1, int(math.ceil(np.linalg.norm(goal - start) / step)))
    return [start + (goal - start) * k / n for k in range(n + 1)]


def _source(tmp_path: Path, misses: bool = False, arm_a_only: bool = False, policy_tracking=_TRACKING):
    """Episode 0: arm B (policy above, the funnel from frame 40). Episode 1: arm A, 15 mm off in -x,
    closing 20 mm deep. Returns the root and the two grasp-loop rows that recorded them.

    With `misses`, two kept arm-A misses as well: episode 2 comes down from 330 mm above the peg
    and closes 12 mm off in +y (empty), episode 3 hovers 25 mm up and 10 mm off without closing
    (no_close).

    With `arm_a_only`, episode 0 is dropped, and the arm-A episodes trail their command by
    `policy_tracking`: a source with no funnel close to measure the offset on."""

    import pandas as pd

    peg_cmd = _T_B_WS[:3, :3].T @ (_PEG_BASE - _T_B_WS[:3, 3]) + _TRACKING
    funnel = _walk_to(peg_cmd + [0.02, 0.01, 0.15], peg_cmd + [0.02, 0.01, 0.11])
    funnel = funnel[:40] + _walk_to(funnel[39], peg_cmd + _CLOSE_DZ)[1:]
    close_b = len(funnel) - 1 + 16
    policy = _walk_to(peg_cmd + [-0.015, 0.0, 0.12], peg_cmd + [-0.015, 0.0, -0.020])
    close_a = len(policy) - 1 + 5
    plan = [(funnel, close_b, 40, "B", "held"), (policy, close_a, None, "A", "held")]
    if misses:
        empty = _walk_to(peg_cmd + [0.0, 0.012, 0.33], peg_cmd + [0.0, 0.012, -0.005])
        hover = _walk_to(peg_cmd + [0.01, 0.0, 0.12], peg_cmd + [0.01, 0.0, 0.025])
        plan += [(empty, len(empty) - 1 + 5, None, "A", "empty"), (hover, None, None, "A", "no_close")]
    if arm_a_only:
        plan = plan[1:]
    root = tmp_path / "trial_src"
    (root / "data" / "chunk-000").mkdir(parents=True)
    rows, records, index = [], [], 0
    for ep, (cmds, close_at, from_, _arm, _verdict) in enumerate(plan):
        tracking = policy_tracking if _arm == "A" else _TRACKING
        start, state, actions, flags = _episode(cmds, close_at, from_, tracking=tracking)
        for t in range(len(actions)):
            rows.append(
                {
                    "index": index,
                    "episode_index": ep,
                    "frame_index": t,
                    "observation.state": state[t].astype(np.float32),
                    "action": actions[t].astype(np.float32),
                    "is_intervention": np.float32(flags[t]),
                }
            )
            index += 1
        records.append(
            {
                "episode_index": ep,
                "rollout": 3 + ep,
                "T_B_Ws": _T_B_WS.tolist(),
                "start_contract_xyzquat": list(start) + [1.0, 0.0, 0.0, 0.0],
            }
        )
        # float32 storage costs a few micrometres; the match tolerance is millimetres
        records[-1]["_close_base"] = None if close_at is None else _to_base(state[close_at, :3]).tolist()
        records[-1]["_arm"], records[-1]["_verdict"] = _arm, _verdict
    pd.DataFrame(rows).to_parquet(root / "data" / "chunk-000" / "file-000.parquet")
    (root / "meta").mkdir()
    (root / "meta" / "episode_frames.jsonl").write_text(
        "".join(json.dumps({k: v for k, v in r.items() if not k.startswith("_")}) + "\n" for r in records)
    )
    trial_rows = [
        {
            "kind": "trial",
            "trial": r["rollout"],
            "arm": r["_arm"],
            "verdict": r["_verdict"],
            "pegXyz": _PEG_BASE.tolist(),
            "closeXyz": r["_close_base"],
            # The runtime records the root since --grasp-loop-keep-misses; the held rows are older.
            "dataset": {"kept": True, "firstEpisode": r["episode_index"], "frames": 0}
            | ({} if r["_verdict"] == "held" else {"root": f"/data/lerobot/outputs/datasets/{root.name}"}),
        }
        for r in records
    ]
    return root, trial_rows, peg_cmd, close_a


def test_a_policy_only_episode_is_refused_without_its_grasp_loop_row(tmp_path):
    root, _rows, _peg, _close = _source(tmp_path)
    with pytest.raises(ValueError, match="episode 1 is policy-only"):
        relabel_source(root, 50)


def test_a_policy_only_episode_takes_its_peg_from_the_row_not_from_where_it_closed(tmp_path):
    root, rows, peg_cmd, close_a = _source(tmp_path)
    out, summary = relabel_source(root, 50, trial_rows=rows)
    assert out[1]["peg"] == pytest.approx(peg_cmd, abs=2e-6)
    # The funnel episode's own close-derived peg is the same point.
    assert out[0]["peg"] == pytest.approx(peg_cmd, abs=2e-6)
    assert summary["policy_only_episodes"] == 1 and summary["trial_row_matched_episodes"] == 2
    assert summary["close_tracking_offset_from"] == "funnel closes"
    assert max(abs(v) for v in summary["row_peg_minus_close_peg_mm"]["median"]) < 0.01
    # A frame low and 15 mm off in x is told to lift and realign, not to close there.
    rec = out[1]
    low = int(np.flatnonzero(rec["frames"] == close_a - 6)[0])
    assert not rec["masked"][low] and rec["action"][low][0, 2] > 0.0
    cfg = FunnelConfig(pegXyz=tuple(peg_cmd))
    actions, _ = expert_chunk(rec["state"][close_a - 6, :3] + _TRACKING, peg_cmd, 300, config=cfg)
    path = rec["state"][close_a - 6, :3] + _TRACKING + np.cumsum(actions[:, :3].astype(np.float64), axis=0)
    first_across = int(np.argmax(np.abs(path[:, 0] - path[0, 0]) > 1e-6))
    assert path[first_across - 1, 2] == pytest.approx(cfg.alignZ, abs=1e-5)
    assert path[-1] == pytest.approx(cfg.closeXyz, abs=1e-5)
    assert rec["action"][low] == pytest.approx(actions[:50], abs=1e-7)


def test_an_arm_a_only_source_aims_with_the_funnels_offset_not_its_own_closes(tmp_path):
    """R3 (10-26): every episode policy-only, closing on the move ~0.7 mm off its command, where
    the funnel's settled closes sit 3.8 mm off. The peg the labels aim at is the funnel's."""

    moving = np.array([0.0007, -0.0006, -0.0003])
    root, rows, peg_cmd, _close = _source(tmp_path, arm_a_only=True, policy_tracking=moving)
    with pytest.raises(NoFunnelCloseError, match="no funnel close"):
        relabel_source(root, 50, trial_rows=rows)
    out, summary = relabel_source(root, 50, trial_rows=rows, funnel_offset=_TRACKING)
    assert out[0]["peg"] == pytest.approx(peg_cmd, abs=2e-6)
    assert summary["close_tracking_offset_from"] == "other sources' funnel closes"
    assert summary["close_tracking_offset_mm"] == pytest.approx([3.8, -1.8, -3.2], abs=0.01)
    assert summary["policy_close_offset_mm"] == pytest.approx([0.7, -0.6, -0.3], abs=0.01)


def test_a_policy_only_episodes_close_and_after_get_no_supervision(tmp_path):
    root, rows, _peg, close_a = _source(tmp_path)
    out, summary = relabel_source(root, 50, trial_rows=rows)
    rec = out[1]
    after = rec["frames"] >= close_a
    assert after.sum() == len(rec["state"]) - close_a == summary["masked_after_policy_close_frames"]
    assert rec["masked"][after].all() and rec["is_pad"][after].all()
    assert not rec["masked"][~after].any() and not rec["is_pad"][~after][:, 0].any()
    # The funnel episode keeps its own frames from the takeover on.
    assert out[0]["frames"].max() < 40 and not out[0]["masked"].any()


def test_a_row_from_another_run_is_not_taken_for_the_episode(tmp_path):
    root, rows, _peg, _close = _source(tmp_path)
    other = dict(rows[1], closeXyz=(np.asarray(rows[1]["closeXyz"]) + [0.01, 0.0, 0.0]).tolist())
    with pytest.raises(ValueError, match="episode 1 is policy-only"):
        relabel_source(root, 50, trial_rows=[rows[0], other])
    with pytest.raises(ValueError, match="matches 2 grasp-loop rows"):
        relabel_source(root, 50, trial_rows=rows + [rows[1]])


def test_the_row_matches_a_reclose_after_a_short_close_pulse(tmp_path):
    """R6 (10-27): R4 045000 closes for a few steps, lifts and closes again; the row's close is the
    one that settled, tens of mm from the first onset. The episode is still its row's, and its
    frames from the first pulse on stay unsupervised."""

    import pandas as pd

    root, rows, _peg, close_a = _source(tmp_path)
    path = root / "data" / "chunk-000" / "file-000.parquet"
    frames = pd.read_parquet(path)
    pulse = close_a - 20
    at = frames.index[(frames.episode_index == 1) & frames.frame_index.between(pulse, pulse + 3)]
    for i in at:
        action = frames.at[i, "action"].copy()
        action[4] = 0.0
        frames.at[i, "action"] = action
    frames.to_parquet(path)
    out, summary = relabel_source(root, 50, trial_rows=rows)
    assert summary["trial_row_matched_episodes"] == 2
    rec = out[1]
    after = rec["frames"] >= pulse
    assert rec["masked"][after].all() and not rec["masked"][~after].any()


def test_trial_rows_load_only_kept_trials(tmp_path):
    path = tmp_path / "grasp_x.jsonl"
    kept = {"kind": "trial", "trial": 0, "closeXyz": [0, 0, 0], "pegXyz": [0, 0, 0],
            "dataset": {"kept": True, "firstEpisode": 4}}
    lines = [
        {"kind": "run_start"},
        kept,
        dict(kept, trial=1, dataset={"kept": False, "frames": 9}),
        dict(kept, trial=2, dataset=None),
    ]
    path.write_text("".join(json.dumps(r) + "\n" for r in lines))
    assert [r["trial"] for r in load_trial_rows([path])] == [0]
    # A kept miss that never closed has no close, and is loaded all the same.
    no_close = dict(kept, trial=3, verdict="no_close", closeXyz=None, dataset={"kept": True, "firstEpisode": 5, "root": "/r"})
    path.write_text("".join(json.dumps(r) + "\n" for r in [*lines, no_close]))
    assert [r["trial"] for r in load_trial_rows([path])] == [0, 3]


def test_a_kept_miss_is_supervised_only_where_it_is_relabelled(tmp_path):
    root, rows, peg_cmd, _close = _source(tmp_path, misses=True)
    out, summary = relabel_source(root, 50, trial_rows=rows)
    assert summary["missed_episodes"] == 2 and summary["never_closed_episodes"] == 1
    assert summary["trial_row_matched_episodes"] == 4
    rec = out[2]
    assert rec["peg"] == pytest.approx(peg_cmd, abs=2e-6)
    # Every frame has an override: above the band and from the close on, none of the policy's
    # own steps keeps its label.
    assert list(rec["frames"]) == list(range(len(rec["state"])))
    close = len(rec["state"]) - 10  # the fixture's episodes end 10 frames after the close
    high = (rec["state"][:, 2] - rec["peg"][2]) * 1000.0 > 300.0
    n_high = int(high.sum())
    assert n_high > 20 and high[:n_high].all()
    assert rec["masked"][:n_high].all() and rec["masked"][close:].all()
    assert not rec["masked"][n_high:close].any()
    assert summary["masked_unrelabelled_miss_frames"] == int(high.sum())
    # A held arm-A episode's frames above the band would have kept their labels; there are none
    # in this fixture, so nothing was left as recorded.
    assert summary["kept_above_max_height_frames"] == 0


def test_a_miss_that_never_closed_is_relabelled_to_its_last_frame(tmp_path):
    root, rows, peg_cmd, _close = _source(tmp_path, misses=True)
    out, _summary = relabel_source(root, 50, trial_rows=rows)
    rec = out[3]
    assert list(rec["frames"]) == list(range(len(rec["state"])))
    assert not rec["masked"].any()
    # Hovering 10 mm off at 25 mm: across at alignZ first, never a close on the spot.
    last = rec["action"][-1]
    assert last[0, 2] > 0.0 and (last[:, 4] == 1.0).all()


def test_a_miss_that_never_closed_needs_a_row_from_its_own_dataset(tmp_path):
    root, rows, _peg, _close = _source(tmp_path, misses=True)
    elsewhere = dict(rows[3], dataset=dict(rows[3]["dataset"], root="/data/lerobot/outputs/datasets/trial_other"))
    with pytest.raises(ValueError, match="episode 3 is policy-only"):
        relabel_source(root, 50, trial_rows=[*rows[:3], elsewhere])
    # Without the root a closeless row could be any dataset's: not taken.
    unrooted = dict(rows[3], dataset={k: v for k, v in rows[3]["dataset"].items() if k != "root"})
    with pytest.raises(ValueError, match="episode 3 is policy-only"):
        relabel_source(root, 50, trial_rows=[*rows[:3], unrooted])


def test_named_holdout_episodes_take_ranges():
    from tools.fr3.funnel_relabel import _episode_list

    assert _episode_list("466-469,490") == [466, 467, 468, 469, 490]
