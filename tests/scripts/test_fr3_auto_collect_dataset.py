"""E6-data: which frames of a night become training samples, and which are refused and why.

This is the pass where data is thrown away, so every test here is about a rule that decides it.
The rules are not interchangeable with plausible alternatives: keeping a failed cycle would train
behaviour cloning on the failure, deleting it would cost a later value pass the only trajectories
it needs, and trimming stillness with a position-only test would delete the frames where the
fingers close on the peg -- the single most valuable frame of a grasp.
"""

import sys

import pytest

from tools.fr3.auto_collect_dataset import (
    DEFAULT_MAX_STILL_FRAMES,
    STILL_RULE_BUFFER,
    STILL_RULE_FALLBACK,
    STILL_RULE_OFF,
    AutoCollectEpisodeWriter,
    SelectedEpisode,
    SelectionPolicy,
    describe_selection,
    select_episodes,
    selection_summary,
)


def _frame(t, episode, phase="descend_to_peg", *, x=0.36, z=0.10, gripper=1.0, step=2.0):
    return {
        "kind": "frame",
        "t": t,
        "phase": phase,
        "episode": episode,
        "state": {"ee.x": x, "ee.y": -0.14, "ee.z": z},
        "sent_action": {
            "ee.x": x, "ee.y": -0.14, "ee.z": z,
            "ee.wx": 0.0, "ee.wy": 0.0, "ee.wz": 0.0,
            "gripper.pos": gripper,
        },
        "stepMm": step,
    }


def _episode(index, *, verdict="held", kind="recovery", audit_ok=True, frames=None, t0=100.0):
    rows = [{"kind": "marker", "t": t0, "marker": "episode_start", "episode": index,
             "cycle": index, "cycleKind": kind, "offsetMm": 12.0}]
    if frames is None:
        frames = [
            _frame(t0 + 0.03 * (i + 1), index, x=0.36 + 0.002 * i)
            for i in range(10)
        ]
    rows.extend(frames)
    rows.append({"kind": "marker", "t": t0 + 10.0, "marker": "episode_end", "episode": index,
                 "cycle": index, "verdict": verdict, "auditOk": audit_ok})
    return rows


def test_a_failed_cycle_is_refused_from_training_and_says_it_is_still_on_disk():
    selected, rejections = select_episodes(_episode(0, verdict="empty"))
    assert selected == []
    assert rejections[0].reason == "verdict_empty"
    assert "kept on disk" in rejections[0].detail


def test_a_failed_cycle_can_be_asked_for_when_something_other_than_bc_needs_it():
    """The exclusion is a policy, not a deletion: a value pass needs exactly these trajectories."""

    policy = SelectionPolicy(keepVerdicts=("held", "empty"))
    selected, _ = select_episodes(_episode(0, verdict="empty"), policy)
    assert [episode.verdict for episode in selected] == ["empty"]


def test_an_episode_the_step_audit_invalidated_is_refused():
    """One command the rollout guard would clip is one command the policy must not be taught."""

    rows = _episode(0, frames=[_frame(100.0 + i, 0, x=0.36 + 0.01 * i, step=10.0) for i in range(10)])
    selected, rejections = select_episodes(rows)
    assert selected == []
    assert rejections[0].reason == "step_over_guard"
    assert "10 commands above 5.0 mm" in rejections[0].detail


def test_an_episode_with_no_end_marker_is_refused_rather_than_assumed_to_have_ended_well():
    rows = [row for row in _episode(0) if row.get("marker") != "episode_end"]
    selected, rejections = select_episodes(rows)
    assert selected == []
    assert rejections[0].reason == "no_end_marker"


def test_a_frame_from_an_unrecorded_leg_reaching_the_recorder_refuses_the_whole_episode():
    """A retrieval frame in a dataset teaches the policy to pull the peg back out of the hole."""

    rows = _episode(0, frames=[_frame(100.0 + i, 0, phase="move_to_place_above") for i in range(10)])
    selected, rejections = select_episodes(rows)
    assert selected == []
    assert rejections[0].reason == "unrecorded_phase_present"
    assert "move_to_place_above" in rejections[0].detail


def test_frames_outside_any_episode_are_never_selected():
    rows = [_frame(1.0, 99)] + _episode(0) + [_frame(500.0, 99)]
    selected, _ = select_episodes(rows)
    assert len(selected) == 1 and len(selected[0].frames) == 10


def test_the_dead_time_trim_keeps_the_run_a_demonstration_would_also_contain():
    """19 identical commands is the grasp settle at 30 Hz; the demonstrations' own runs mean 17."""

    frames = [_frame(100.0, 0, x=0.36)]
    frames += [_frame(100.0 + 0.03 * (i + 1), 0, x=0.40, gripper=0.0) for i in range(19)]
    selected, rejections = select_episodes(_episode(0, frames=frames))
    assert selected, rejections
    episode = selected[0]
    # The first of the run is motion (the pose and the gripper both changed), so 1 + 17 survive.
    assert episode.stillDropped == 19 - DEFAULT_MAX_STILL_FRAMES - 1
    assert len(episode.frames) == len(frames) - episode.stillDropped


def test_closing_the_fingers_while_the_arm_holds_still_is_not_dead_time():
    """The most valuable frame of a grasp is the one a position-only rule would delete."""

    frames = [_frame(100.0, 0, x=0.36, gripper=1.0)]
    # The arm does not move at all; only the gripper command changes, one step at a time.
    frames += [
        _frame(100.0 + 0.03 * (i + 1), 0, x=0.36, gripper=1.0 - 0.05 * (i + 1))
        for i in range(25)
    ]
    selected, _ = select_episodes(_episode(0, frames=frames))
    assert selected[0].stillDropped == 0
    assert len(selected[0].frames) == 26


def test_a_slow_drift_survives_because_stillness_is_measured_against_the_last_kept_frame():
    """A per-step test would throw a drift away one imperceptible step at a time."""

    # 0.03 mm a step: below the 0.05 mm still threshold per step, but it accumulates past it.
    frames = [_frame(100.0 + 0.03 * i, 0, x=0.36 + 0.00003 * i) for i in range(60)]
    selected, _ = select_episodes(_episode(0, frames=frames))
    assert selected[0].stillDropped == 0


def test_the_report_names_the_rule_that_did_the_trimming():
    """Two rules can produce this number, and they do not trim the same frames.

    A reader who cannot tell which one ran cannot tell whether a missing rotation channel
    explains a difference between two nights, so the rule is part of the report and not part of
    the environment it was produced in.
    """

    frames = [_frame(100.0, 0, x=0.36)]
    frames += [_frame(100.0 + 0.03 * (i + 1), 0, x=0.40, gripper=0.0) for i in range(19)]
    selected, rejections = select_episodes(_episode(0, frames=frames))
    summary = selection_summary(selected, rejections)
    assert summary["stillRule"] == STILL_RULE_BUFFER
    assert "still_rule=" + STILL_RULE_BUFFER in describe_selection(summary)
    # Nothing selected means no rule trimmed anything; naming one would describe a pass that did
    # not happen.
    assert selection_summary([], list(rejections))["stillRule"] == "n/a"


def test_turning_the_trim_off_is_reported_as_off_rather_than_as_a_rule_that_kept_everything():
    frames = [_frame(100.0 + 0.03 * i, 0, x=0.36, gripper=0.0) for i in range(40)]
    selected, rejections = select_episodes(
        _episode(0, frames=frames), SelectionPolicy(maxStillFrames=None)
    )
    assert selected[0].stillDropped == 0
    assert selection_summary(selected, rejections)["stillRule"] == STILL_RULE_OFF


def test_the_fallback_trims_exactly_what_the_human_path_trims_on_data_that_does_not_rotate(
    monkeypatch,
):
    """The fallback is only safe because this rig's recorded legs hold a fixed orientation.

    It cannot compare rotation, so it is a superset rule in general. What has to hold is that on
    the data it will actually read -- translation and gripper only -- it decides frame for frame
    what `DaggerFrameBuffer` decides. Left untested, the two drift apart and the difference shows
    up as a night that trimmed differently for no reason anybody can name.
    """

    frames = [_frame(100.0, 0, x=0.36, gripper=1.0)]
    # A settle, a gripper close with the arm still, a slow drift, then motion again: one of each
    # case the rule has to get right.
    frames += [_frame(100.0 + 0.03 * len(frames), 0, x=0.36, gripper=1.0) for _ in range(19)]
    frames += [_frame(100.0 + 0.03 * (len(frames) + i), 0, x=0.36, gripper=1.0 - 0.05 * (i + 1))
               for i in range(10)]
    frames += [_frame(100.0 + 0.03 * (len(frames) + i), 0, x=0.36 + 0.00003 * i, gripper=0.5)
               for i in range(30)]
    frames += [_frame(100.0 + 0.03 * (len(frames) + i), 0, x=0.38 + 0.002 * i, gripper=0.5)
               for i in range(10)]
    rows = _episode(0, frames=frames)

    with_buffer, _ = select_episodes(list(rows))
    assert with_buffer[0].stillRule == STILL_RULE_BUFFER

    # `None` in `sys.modules` is what an absent module looks like to the import statement, which
    # is the condition the fallback exists for.
    monkeypatch.setitem(sys.modules, "tools.fr3.dagger_dataset", None)
    fallback, _ = select_episodes(list(rows))
    assert fallback[0].stillRule == STILL_RULE_FALLBACK

    assert fallback[0].stillDropped == with_buffer[0].stillDropped
    assert [frame["t"] for frame in fallback[0].frames] == [
        frame["t"] for frame in with_buffer[0].frames
    ]


def test_an_episode_too_short_to_be_an_approach_is_refused():
    rows = _episode(0, frames=[_frame(100.0 + i, 0, x=0.36 + 0.002 * i) for i in range(3)])
    selected, rejections = select_episodes(rows)
    assert selected == []
    assert rejections[0].reason == "too_short"


def test_the_summary_says_what_would_be_written_and_what_would_not():
    rows = _episode(0) + _episode(1, verdict="empty", t0=200.0) + _episode(2, kind="nominal", t0=300.0)
    selected, rejections = select_episodes(rows)
    summary = selection_summary(selected, rejections)
    assert summary["episodes"] == 2 and summary["rejected"] == 1
    assert summary["byKind"] == {"recovery": 1, "nominal": 1}
    assert summary["rejectedByReason"] == {"verdict_empty": 1}
    text = describe_selection(summary)
    assert "REJECTED verdict_empty" in text


def test_the_writer_saves_one_episode_per_cycle_and_never_merges_them():
    class FakeDataset:
        def __init__(self):
            self.frames = []
            self.saves = 0

        def add_frame(self, frame):
            self.frames.append(frame)

        def save_episode(self, parallel_encoding=True):
            assert parallel_encoding is False
            self.saves += 1

    dataset = FakeDataset()
    episodes = [
        SelectedEpisode(index=0, kind="recovery", verdict="held", frames=[{"a": 1}, {"a": 2}]),
        SelectedEpisode(index=1, kind="nominal", verdict="held", frames=[{"a": 3}]),
    ]
    written = AutoCollectEpisodeWriter(
        dataset, lambda row, episode: {**row, "episode": episode.index}, emit=lambda _: None
    ).write(episodes)
    assert written == {"episodes": 2, "frames": 3}
    assert dataset.saves == 2
    assert [frame["episode"] for frame in dataset.frames] == [0, 0, 1]


def test_a_schema_mismatch_raises_out_of_the_writer_rather_than_filling_a_column():
    class FakeDataset:
        def add_frame(self, frame):
            pass

        def save_episode(self, parallel_encoding=True):
            pass

    def build_frame(row, episode):
        raise KeyError("observation.images.wrist")

    with pytest.raises(KeyError, match="wrist"):
        AutoCollectEpisodeWriter(FakeDataset(), build_frame, emit=lambda _: None).write(
            [SelectedEpisode(index=0, kind="nominal", verdict="held", frames=[{"a": 1}])]
        )
