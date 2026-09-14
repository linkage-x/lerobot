"""E6-data: can a night be told apart from a night that only looks like one?

The checks worth testing are the ones with a measured comparison behind them, and the one that
matters most is the check that a *suspiciously perfect* recording is a fault rather than a
triumph. An arm under impedance control always trails its setpoint -- that gap is the force --
so a night whose stored state equals its stored command every frame did not record a
well-tracking arm, it recorded the same number twice. That failure produces a dataset in which
the action is a copy of the observation, and a policy trained on it learns to predict where it
already is.
"""

import json

import pytest

from tools.fr3.auto_collect_qc import (
    DEPLOYMENT_LEASH_MM,
    build_report,
    describe_report,
    phase_one_verdict,
)


def _frame(t, episode, phase="descend_to_peg", *, state_z=0.10, action_z=0.095, step=2.0, images=True):
    row = {
        "kind": "frame",
        "t": t,
        "phase": phase,
        "episode": episode,
        "state": {"ee.x": 0.36, "ee.y": -0.14, "ee.z": state_z},
        "sent_action": {"ee.x": 0.36, "ee.y": -0.14, "ee.z": action_z, "gripper.pos": 1.0},
        "stepMm": step,
    }
    if images:
        row["images"] = {"ee": "frames/000000_ee.jpg"}
        row["cameraSkewMs"] = {"ee": 3.0}
    return row


def _episode(index, t0, frames=5, *, verdict="held", kind="recovery", audit_ok=True, **frame_kwargs):
    rows = [{"kind": "marker", "t": t0, "marker": "episode_start", "episode": index,
             "cycle": index, "cycleKind": kind, "offsetMm": 12.0}]
    for step in range(frames):
        rows.append(_frame(t0 + 0.03 * (step + 1), index, **frame_kwargs))
    rows.append({"kind": "marker", "t": t0 + 0.03 * (frames + 1), "marker": "episode_end",
                 "episode": index, "cycle": index, "verdict": verdict, "auditOk": audit_ok})
    return rows


def _night(cycles=3, **kwargs):
    rows = []
    for index in range(cycles):
        rows.extend(_episode(index, 100.0 + index * 10.0, **kwargs))
        rows.append({"kind": "marker", "t": 100.0 + index * 10.0 + 1.0, "marker": "leg",
                     "episode": -1, "leg": "place", "recorded": False})
    return rows


def test_a_night_whose_state_equals_its_command_is_a_fault_not_perfect_tracking(tmp_path):
    """The gap between command and pose is what moves the arm. Zero of it means nothing moved it."""

    rows = _night(state_z=0.10, action_z=0.10)
    report = build_report(tmp_path, rows)
    assert report["tracking"]["p50"] == pytest.approx(0.0)
    verdict = phase_one_verdict(report, cycles_required=3)
    following = next(check for check in verdict["checks"] if check["name"] == "the_arm_was_following")
    assert following["ok"] is False
    assert verdict["ok"] is False


def test_a_command_running_away_from_a_stuck_arm_is_caught(tmp_path):
    rows = _night(state_z=0.10, action_z=0.10 - (DEPLOYMENT_LEASH_MM + 5.0) / 1000.0)
    report = build_report(tmp_path, rows)
    assert report["tracking"]["overLeash"] == report["tracking"]["n"] > 0
    verdict = phase_one_verdict(report, cycles_required=3)
    assert verdict["ok"] is False


def test_an_ordinary_lag_reads_as_following(tmp_path):
    report = build_report(tmp_path, _night())
    assert report["tracking"]["p50"] == pytest.approx(5.0)
    assert report["tracking"]["overLeash"] == 0
    verdict = phase_one_verdict(report, cycles_required=3)
    assert verdict["ok"] is True, [check for check in verdict["checks"] if not check["ok"]]


def test_a_gap_with_a_marker_across_it_is_an_environment_operation_and_one_without_is_missing_data(tmp_path):
    """The two are indistinguishable once the process has exited, which is why markers exist."""

    explained = build_report(tmp_path, _night(cycles=2))
    assert explained["continuity"]["unexplainedGaps"] == []

    rows = _episode(0, 100.0) + _episode(1, 400.0)
    # Drop both markers that sit across the 300 s between them, leaving it genuinely unexplained.
    rows = [
        row for row in rows
        if not (row.get("marker") == "episode_start" and row.get("cycle") == 1)
        and not (row.get("marker") == "episode_end" and row.get("cycle") == 0)
    ]
    report = build_report(tmp_path, rows)
    assert len(report["continuity"]["unexplainedGaps"]) == 1
    assert report["continuity"]["unexplainedGaps"][0]["seconds"] == pytest.approx(300.0, abs=1.0)


def test_a_marker_at_the_head_of_a_gap_does_not_excuse_a_gap_of_any_length(tmp_path):
    """A leg that announced itself and then took five minutes is a stall, not an environment operation.

    The first rule written here was "a marker immediately before the frame explains the gap",
    which is how a legitimate unrecorded leg does look -- and it made a hang invisible for as long
    as anything at all had announced itself first.
    """

    rows = _episode(0, 100.0) + _episode(1, 400.0)
    report = build_report(tmp_path, rows)
    assert report["continuity"]["unexplainedGaps"] == []
    assert len(report["continuity"]["stalledLegs"]) == 1
    assert report["continuity"]["stalledLegs"][0]["seconds"] == pytest.approx(300.0, abs=1.0)
    check = next(
        c for c in phase_one_verdict(report, cycles_required=2)["checks"] if c["name"] == "no_stalled_legs"
    )
    assert check["ok"] is False


def test_frames_recorded_outside_any_episode_are_counted(tmp_path):
    rows = _night(cycles=1) + [_frame(200.0, 99)]
    report = build_report(tmp_path, rows)
    assert report["continuity"]["framesOutsideEpisode"] == 1
    assert phase_one_verdict(report, cycles_required=1)["ok"] is False


def test_an_episode_with_no_end_marker_is_reported_rather_than_assumed_complete(tmp_path):
    rows = [row for row in _night(cycles=2) if row.get("marker") != "episode_end" or row.get("cycle") != 1]
    report = build_report(tmp_path, rows)
    assert report["continuity"]["episodesWithoutEnd"] == [1]
    assert phase_one_verdict(report, cycles_required=1)["ok"] is False


def test_a_shard_left_unfooted_fails_the_night_rather_than_being_read_as_data(tmp_path):
    (tmp_path / "shard_0000").mkdir()
    (tmp_path / "shard_0000" / "rows.jsonl").write_text("", encoding="utf-8")
    report = build_report(tmp_path, _night())
    assert report["shards"]["unfooted"] == ["shard_0000"]
    assert phase_one_verdict(report, cycles_required=3)["ok"] is False


def test_a_step_above_the_guard_fails_the_night(tmp_path):
    report = build_report(tmp_path, _night(step=6.0))
    assert report["steps"]["overLimit"] > 0
    assert phase_one_verdict(report, cycles_required=3)["ok"] is False


def test_the_still_run_is_measured_per_episode_and_compared_to_the_demonstrations(tmp_path):
    rows = _episode(0, 100.0, frames=45, step=0.0)
    report = build_report(tmp_path, rows)
    assert report["steps"]["longestStillRun"] == 45
    assert report["steps"]["stillFraction"] == pytest.approx(1.0)
    check = next(
        c for c in phase_one_verdict(report, cycles_required=1)["checks"]
        if c["name"] == "still_runs_within_the_demonstrations"
    )
    assert check["ok"] is False
    assert "39" in check["detail"], "the demonstrations' own longest run has to be in the message"


def test_only_cycles_that_ended_holding_the_peg_count_towards_the_fifty(tmp_path):
    rows = _night(cycles=2) + _episode(2, 200.0, verdict="empty")
    report = build_report(tmp_path, rows)
    assert report["verdicts"] == {"held": 2, "empty": 1}
    check = next(
        c for c in phase_one_verdict(report, cycles_required=3)["checks"]
        if c["name"] == "uninterrupted_cycles"
    )
    assert check["ok"] is False
    assert "2 of 3" in check["detail"]


def test_the_failed_episode_keeps_its_frames_in_the_report(tmp_path):
    """Kept and labelled: BC filters them, a later value pass needs them, deletion is irreversible."""

    report = build_report(tmp_path, _episode(0, 100.0, frames=5, verdict="empty"))
    episode = report["episodes"][0]
    assert episode["verdict"] == "empty" and episode["frames"] == 5


def test_the_report_names_the_demonstration_numbers_it_is_judging_against(tmp_path):
    text = describe_report(build_report(tmp_path, _night()), phase_one_verdict(build_report(tmp_path, _night()), cycles_required=3))
    assert "demo p50=5.71" in text and "demo p50=1.59" in text
    assert "phase_one=" in text


def test_the_whole_report_survives_json(tmp_path):
    """It is written to a file and read by other things, including a page that has to render it."""

    json.dumps(build_report(tmp_path, _night()))


# -- the verdict is a model, so it gets held-out data rather than trust -----------------------


def test_the_review_sample_is_stratified_because_the_rare_verdict_is_the_one_that_matters(tmp_path):
    """A uniform sample of a good night is fifty `held` cycles and no information about `empty`."""

    from tools.fr3.auto_collect_qc import build_review_sample

    rows = []
    for index in range(20):
        rows.extend(_episode(index, 100.0 + index * 10.0, verdict="held"))
    rows.extend(_episode(20, 400.0, verdict="empty"))
    rows.extend(_episode(21, 410.0, verdict="changed"))
    report = build_report(tmp_path, rows)

    sample = build_review_sample(report, rows, per_verdict=3)
    verdicts = [item["classifier"] for item in sample["items"]]
    assert verdicts.count("held") == 3, "the common verdict was not capped"
    assert verdicts.count("empty") == 1 and verdicts.count("changed") == 1
    assert sample["verdictCounts"] == {"held": 20, "empty": 1, "changed": 1}


def test_the_reviewer_is_handed_the_frames_where_the_fingers_are_around_the_peg_or_not(tmp_path):
    from tools.fr3.auto_collect_qc import build_review_sample

    rows = _episode(0, 100.0, frames=4)
    # Two of the episode's frames are the grasp itself; the approach frames show an empty table.
    rows.insert(5, _frame(100.20, 0, phase="close_gripper"))
    rows.insert(6, _frame(100.23, 0, phase="lift_8cm_after_grasp"))
    report = build_report(tmp_path, rows)
    sample = build_review_sample(report, rows, per_verdict=5)
    item = sample["items"][0]
    assert [frame["phase"] for frame in item["frames"]] == ["close_gripper", "lift_8cm_after_grasp"]
    assert all(frame["images"] for frame in item["frames"]), "the reviewer must get a file to open"


def test_the_label_starts_empty_rather_than_pre_filled_with_the_prediction(tmp_path):
    """A label that starts as the prediction is a label that agrees with it by default."""

    from tools.fr3.auto_collect_qc import build_review_sample

    rows = _episode(0, 100.0)
    sample = build_review_sample(build_report(tmp_path, rows), rows)
    assert sample["items"][0]["humanLabel"] is None
    assert sample["items"][0]["classifier"] == "held"


def test_agreement_is_reported_per_verdict_because_a_pooled_number_hides_the_rare_one():
    from tools.fr3.auto_collect_qc import verdict_agreement

    sample = {
        "items": [
            # Nineteen held cycles the reviewer agrees with...
            *[{"cycle": i, "classifier": "held", "humanLabel": "held"} for i in range(19)],
            # ...and one the classifier got wrong, which is the whole finding.
            {"cycle": 19, "classifier": "held", "humanLabel": "empty"},
        ]
    }
    result = verdict_agreement(sample)
    assert result["labelled"] == 20
    assert result["agreement"] == pytest.approx(0.95)
    assert result["byVerdict"]["held"]["fp"] == 1
    assert result["byVerdict"]["held"]["fpRate"] == pytest.approx(1 / 20)
    assert result["byVerdict"]["empty"]["fn"] == 1
    assert result["byVerdict"]["empty"]["fnRate"] == pytest.approx(1.0), (
        "every empty close there was got missed, which a 95% pooled score reads as a pass"
    )


def test_an_unlabelled_sample_says_so_instead_of_reporting_perfect_agreement():
    from tools.fr3.auto_collect_qc import verdict_agreement

    result = verdict_agreement({"items": [{"cycle": 0, "classifier": "held", "humanLabel": None}]})
    assert result["labelled"] == 0 and result["agreement"] is None
