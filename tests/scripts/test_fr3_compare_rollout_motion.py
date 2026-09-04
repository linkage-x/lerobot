from __future__ import annotations

import json
from pathlib import Path

import pytest

from tools.fr3.fr3_compare_rollout_motion import (
    RECOVERY_CONFIRMED,
    RECOVERY_PARTIAL,
    STILL_MM,
    ComparisonError,
    GroupMotion,
    load_group,
    main,
    motion_from_trace,
    parse_group,
    read_trace,
    recovery,
    summarise,
    trace_path_for,
)

HEADER = "step,x,y,z,gripper_cmd,gripper_raw,status,source"


def _trace(rows: list[tuple], path: Path) -> Path:
    """A trace file shaped like the one the rollout runtime writes."""
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [HEADER]
    for step, x, y, z, gripper, status, source in rows:
        lines.append(f"{step},{x:.6f},{y:.6f},{z:.6f},{gripper:.4f},{gripper:.4f},{status},{source}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def _row(step: int, x: float, *, gripper: float = 1.0, status: str = "pass", source: str = "policy"):
    return (step, x, 0.0, 0.0, gripper, status, source)


def test_a_takeover_contributes_no_displacement(tmp_path):
    """The operator's motion is decisive and large; crediting it to the policy inverts the metric."""
    path = _trace(
        [
            _row(0, 0.000),
            _row(1, 0.001),  # 1 mm, policy
            _row(2, 0.051, source="expert"),  # 50 mm, the operator rescuing it
            _row(3, 0.101, source="expert"),
            _row(4, 0.102),  # first policy row after the takeover
            _row(5, 0.103),  # 1 mm, policy
        ],
        tmp_path / "rollout_001.csv",
    )

    motion = motion_from_trace(read_trace(path), name="t")

    assert motion.policy_steps == 4
    # Two pairs only: 0->1 and 4->5. The 3->4 pair spans the handover and is not the policy's.
    assert motion.steps_mm == pytest.approx([1.0, 1.0])


def test_a_gap_in_the_step_counter_is_not_a_step(tmp_path):
    """Rows can be dropped; differencing across the hole invents a displacement nobody made."""
    path = _trace([_row(0, 0.0), _row(1, 0.001), _row(9, 0.100)], tmp_path / "rollout_001.csv")

    motion = motion_from_trace(read_trace(path), name="t")

    assert motion.steps_mm == pytest.approx([1.0])


def test_leash_limited_pairs_are_dropped_by_default(tmp_path):
    """On a clamped step the recorded motion is the guard's, so a big command reads as a small one."""
    rows = [_row(0, 0.0), _row(1, 0.001, status="leash_limited"), _row(2, 0.002)]
    path = _trace(rows, tmp_path / "rollout_001.csv")

    dropped = motion_from_trace(read_trace(path), name="t")
    kept = motion_from_trace(read_trace(path), name="t", include_clamped=True)

    assert dropped.steps_mm == []
    assert dropped.clamped_pairs == 2
    assert kept.steps_mm == pytest.approx([1.0, 1.0])
    assert kept.clamped_pairs == 0


def test_the_still_fraction_counts_ticks_the_arm_did_nothing_on(tmp_path):
    path = _trace(
        [
            _row(0, 0.0),
            _row(1, 0.0000_10),  # 0.01 mm -- still
            _row(2, 0.0000_20),  # 0.01 mm -- still
            _row(3, 0.0020_20),  # 2.00 mm -- moving
        ],
        tmp_path / "rollout_001.csv",
    )

    motion = motion_from_trace(read_trace(path), name="t")
    group = GroupMotion(label="one", checkpoint="ck")
    group.rollouts.append(motion)
    summary = summarise(group)

    assert summary["still_frac"] == pytest.approx(2 / 3)
    assert summary["median_mm"] < STILL_MM
    assert summary["usable_pairs"] == 3


def test_the_gripper_mid_band_is_measured_on_policy_rows_only(tmp_path):
    path = _trace(
        [
            _row(0, 0.0, gripper=1.0),
            _row(1, 0.001, gripper=0.5),  # policy hesitating
            _row(2, 0.002, gripper=0.5, source="expert"),  # the operator's, not counted
        ],
        tmp_path / "rollout_001.csv",
    )
    group = GroupMotion(label="one", checkpoint="ck")
    group.rollouts.append(motion_from_trace(read_trace(path), name="t"))

    summary = summarise(group)
    # Both columns, because they answer different questions: `gripper_raw` is what the model
    # emitted, `gripper_cmd` is what survived the runtime and reached the hand.
    assert summary["gripper_cmd_mid_frac"] == pytest.approx(0.5)
    assert summary["gripper_raw_mid_frac"] == pytest.approx(0.5)


def test_the_per_rollout_spread_is_reported_next_to_the_pooled_rate(tmp_path):
    """A candidate landing between two groups is only a result if the groups do not overlap."""
    steady = _trace(
        [_row(i, 0.002 * i) for i in range(6)], tmp_path / "a" / "rollout_001.csv"
    )
    frozen = _trace(
        [_row(i, 0.000001 * i) for i in range(6)], tmp_path / "b" / "rollout_001.csv"
    )
    group = GroupMotion(label="mixed", checkpoint="ck")
    group.rollouts.append(motion_from_trace(read_trace(steady), name="steady"))
    group.rollouts.append(motion_from_trace(read_trace(frozen), name="frozen"))

    summary = summarise(group)

    assert summary["still_frac"] == pytest.approx(0.5)  # pooled: half the pairs
    assert summary["still_frac_by_rollout"] == pytest.approx([0.0, 1.0])
    assert summary["still_frac_min"] == pytest.approx(0.0)
    assert summary["still_frac_max"] == pytest.approx(1.0)


def test_a_trace_is_named_by_its_session_stamp_and_index(tmp_path):
    entry = {"logPath": "/x/rollout_ck_real_20260904_024824.log", "rolloutIndex": 2}

    assert trace_path_for(entry, tmp_path) == tmp_path / "session_20260904_024824" / "rollout_002.csv"
    # Entries predating the trace writer carry no index; they are counted, not guessed at.
    assert trace_path_for({"logPath": entry["logPath"]}, tmp_path) is None
    assert trace_path_for({"rolloutIndex": 2}, tmp_path) is None


def test_a_group_reports_logged_rollouts_whose_trace_is_gone(tmp_path):
    traces = tmp_path / "traces"
    _trace([_row(0, 0.0), _row(1, 0.001)], traces / "session_20260904_024824" / "rollout_001.csv")
    entries = [
        {"checkpointId": "ck", "outcome": "failure", "rolloutIndex": 1,
         "logPath": "/x/r_20260904_024824.log"},
        {"checkpointId": "ck", "outcome": "failure", "rolloutIndex": 7,
         "logPath": "/x/r_20260904_024824.log"},
        {"checkpointId": "other", "outcome": "success", "rolloutIndex": 1,
         "logPath": "/x/r_20260904_024824.log"},
    ]

    group = load_group("a", "ck", entries, traces)

    assert len(group.rollouts) == 1
    assert group.missing_traces == 1


def test_a_group_with_no_trace_at_all_is_refused(tmp_path):
    entries = [{"checkpointId": "ck", "rolloutIndex": 9, "logPath": "/x/r_20260904_024824.log"}]

    with pytest.raises(ComparisonError, match="no trace file"):
        load_group("a", "ck", entries, tmp_path)


@pytest.mark.parametrize(
    "candidate_frac, verdict",
    [
        (0.12, "CONFIRMED"),  # 0.47 -> 0.12 closes ~95% of the gap
        (0.30, "PARTIAL"),
        (0.45, "NOT RECOVERED"),
    ],
)
def test_the_verdict_is_a_fraction_of_the_gap_the_regression_opened(candidate_frac, verdict):
    baseline = {"label": "base", "still_frac": 0.10}
    regressed = {"label": "dagger", "still_frac": 0.47}

    result = recovery(baseline, regressed, {"label": "new", "still_frac": candidate_frac})

    assert result["verdict"] == verdict
    assert result["gap"] == pytest.approx(0.37)
    assert 0.0 <= result["recovered"] <= 1.0


def test_the_thresholds_are_ordered_and_fixed_in_the_source():
    """They are the point of the tool: a threshold picked after seeing the number is not one."""
    assert 0.0 < RECOVERY_PARTIAL < RECOVERY_CONFIRMED <= 1.0


def test_recovering_a_regression_that_never_happened_is_refused():
    with pytest.raises(ComparisonError, match="no regression"):
        recovery(
            {"label": "base", "still_frac": 0.30},
            {"label": "dagger", "still_frac": 0.10},
            {"label": "new", "still_frac": 0.10},
        )


def test_group_spelling_is_checked_before_anything_is_read():
    assert parse_group("baseline=L4/030000") == ("baseline", "L4/030000")
    with pytest.raises(Exception, match="LABEL=CHECKPOINT_ID"):
        parse_group("L4/030000")


def test_main_prints_a_table_and_a_verdict(tmp_path, capsys):
    traces = tmp_path / "traces"
    log = tmp_path / "rollout_log.jsonl"
    # base moves 2 mm a step; regressed holds still.
    _trace(
        [_row(i, 0.002 * i) for i in range(6)],
        traces / "session_20260902_014812" / "rollout_001.csv",
    )
    _trace(
        [_row(i, 0.000001 * i) for i in range(6)],
        traces / "session_20260904_024824" / "rollout_001.csv",
    )
    log.write_text(
        "\n".join(
            json.dumps(entry)
            for entry in (
                {"checkpointId": "base", "outcome": "failure", "rolloutIndex": 1,
                 "logPath": "/x/r_20260902_014812.log"},
                {"checkpointId": "dagger", "outcome": "failure", "rolloutIndex": 1,
                 "logPath": "/x/r_20260904_024824.log"},
            )
        )
        + "\n",
        encoding="utf-8",
    )

    code = main(
        [
            "--group", "baseline=base",
            "--group", "regressed=dagger",
            "--rollout-log", str(log),
            "--traces-root", str(traces),
        ]
    )
    out = capsys.readouterr().out

    assert code == 0
    assert "baseline" in out and "regressed" in out
    assert f"frac < {STILL_MM} mm" in out
    # Without --recovery there is no verdict line to mistake for one.
    assert "recovered" not in out
