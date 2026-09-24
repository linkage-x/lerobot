"""The E3 three-arm table: the reduction, the join to the traces, and the arm buckets.

The reduction here is a second copy of the one the roadmap's band numbers came from, so these
tests pin the contract that copy has to keep -- the band bounds, the gripper hysteresis, the
"policy-controlled only" cut, and every reason a rollout gets dropped. The other half of that
guarantee is `--cross-check`, which re-runs the archived script over the same traces; it needs
the workstation's `outputs/` and so cannot be exercised here.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pytest

from tools.fr3 import fr3_e3_arm_table as e3

HOLE = np.array([0.0, 0.0])

# The descent the band statistics are asserted against: eight in-band frames walking straight in
# from 10 mm to 3 mm, one millimetre at a time.
DESCENT = [
    (0.130, 0.020),
    (0.125, 0.020),
    (0.120, 0.020),
    (0.115, 0.010),
    (0.110, 0.009),
    (0.105, 0.008),
    (0.100, 0.007),
    (0.095, 0.006),
    (0.090, 0.005),
    (0.085, 0.004),
    (0.080, 0.003),
    (0.075, 0.003),
    (0.070, 0.003),
]


def build_trace(*, takeover_at: int | None = None, descent=DESCENT, lift: int = 43):
    """A pick-and-insert whose shape the segmentation is supposed to recognise."""
    rows: list[tuple[float, float, float, float, str]] = []
    rows.append((0.020, 0.0, 0.200, 1.0, "policy"))            # open, above the object
    rows.append((0.020, 0.0, 0.100, 0.0, "policy"))            # the grasp: closes below z=0.12
    for step in range(lift):                                   # carry up to the apex
        rows.append((0.020, 0.0, 0.100 + 0.15 * (step + 1) / lift, 0.0, "policy"))
    for z, x in descent:                                       # and down through the band
        rows.append((x, 0.0, z, 0.0, "policy"))
    rows.extend((descent[-1][1], 0.0, descent[-1][0], 0.0, "policy") for _ in range(3))
    rows.extend((descent[-1][1], 0.0, descent[-1][0], 1.0, "policy") for _ in range(10))  # release
    if takeover_at is not None:
        rows[takeover_at] = rows[takeover_at][:4] + ("expert",)
    return rows


def write_trace(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["step,x,y,z,gripper_cmd,gripper_raw,status,source"]
    for step, (x, y, z, grip, source) in enumerate(rows):
        lines.append("%d,%.6f,%.6f,%.6f,%.4f,%.4f,pass,%s" % (step, x, y, z, grip, grip, source))
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ------------------------------------------------------------------ the band ---


def test_band_reads_entry_exit_and_path_off_the_eight_in_band_frames():
    row, why = e3.reduce_trace(build_trace(), HOLE)
    assert why is None
    assert row["n"] == 8
    assert row["e0"] == pytest.approx(10.0, abs=1e-6)
    assert row["e1"] == pytest.approx(3.0, abs=1e-6)
    assert row["conv"] == pytest.approx(7.0, abs=1e-6)
    assert row["path"] == pytest.approx(7.0, abs=1e-6)
    assert row["eff"] == pytest.approx(1.0, abs=1e-6)
    assert row["still"] == pytest.approx(0.0)
    # The grasp happened 20 mm out, which is the difficulty control, not a band number.
    assert row["reach"] == pytest.approx(20.0, abs=1e-6)


def test_the_band_is_half_open_so_a_frame_at_the_ceiling_is_outside_it():
    seg = np.array([[0.0, 0.0, z] for z in (0.120, 0.119, 0.118, 0.117, 0.116, 0.115, 0.114, 0.113, 0.112)])
    stats = e3.band(seg, HOLE)
    assert stats is not None
    assert stats["n"] == 8  # the 0.120 frame is not in the band and does not start the slice


def test_a_descent_that_barely_dips_into_the_band_loses_the_band_not_the_rollout():
    # Four frames in the band, then straight through the floor.
    shallow = [(z, 0.010) for z in (0.130, 0.125, 0.119, 0.118, 0.117, 0.116, 0.070)]
    row, why = e3.reduce_trace(build_trace(descent=shallow), HOLE)
    assert why is None                       # still a usable rollout ...
    assert "e1" not in row                   # ... with no band columns
    assert row["bandWhy"] == "fewer than 8 autonomous frames in the band"
    assert row["endErr"] == pytest.approx(10.0, abs=1e-6)
    assert row["reach"] == pytest.approx(20.0, abs=1e-6)


def test_frames_that_sit_in_the_band_after_the_descent_are_part_of_it():
    """A descent that stops inside the band and holds there is measured, not discarded.

    That is the ordinary shape of a rollout the servo stopped on contact: the peg stands on the
    fixture at z ~ 0.1 and the trace keeps sampling. The exit error is then where it stopped,
    which is the quantity E3 is about -- so the hold has to count.
    """
    stalled = [(z, 0.010) for z in (0.130, 0.125, 0.119, 0.118, 0.117, 0.116)]
    row, why = e3.reduce_trace(build_trace(descent=stalled), HOLE)
    assert why is None
    assert row["n"] == 8  # four descending, three held, and the release frame
    assert row["e0"] == pytest.approx(10.0, abs=1e-6)
    assert row["e1"] == pytest.approx(10.0, abs=1e-6)


# ------------------------------------------------------------------ the handoff ---


def test_a_rollout_that_hands_over_at_the_band_ceiling_is_still_measured():
    """The servo-era shape: the policy drives to z = 0.12 and the runtime leaves the loop.

    No band segment exists, by construction -- 0.12 is the band's ceiling. The handoff error is
    the whole of what the policy is answerable for, so it has to survive when the band does not.
    """
    handoff = [(0.200, 0.020), (0.180, 0.018), (0.160, 0.015), (0.140, 0.012), (0.1201, 0.010)]
    row, why = e3.reduce_trace(build_trace(descent=handoff), HOLE)
    assert why is None
    assert "e1" not in row
    assert row["bandWhy"] == "fewer than 8 autonomous frames in the band"
    assert row["zEnd"] == pytest.approx(0.1201)
    assert row["endErr"] == pytest.approx(10.0, abs=1e-6)
    assert row["stoppedAbove"] is True


def test_a_descent_that_runs_to_the_floor_did_not_stop_above_it():
    row, why = e3.reduce_trace(build_trace(), HOLE)
    assert why is None
    assert row["zEnd"] == pytest.approx(0.070)
    assert row["endErr"] == pytest.approx(3.0, abs=1e-6)
    assert row["stoppedAbove"] is False


# ------------------------------------------------------------------ the cut ---


def test_the_segment_stops_at_the_first_expert_frame():
    # One expert frame early in the carry leaves far fewer than 40 autonomous steps.
    row, why = e3.reduce_trace(build_trace(takeover_at=20), HOLE)
    assert why.startswith("operator took over")
    assert row["reach"] == pytest.approx(20.0, abs=1e-6)  # reported, so the drop is not invisible
    assert "endErr" not in row


def test_an_expert_frame_after_the_release_does_not_shorten_the_segment():
    rows = build_trace()
    before, _ = e3.reduce_trace(rows, HOLE)
    rows[-1] = rows[-1][:4] + ("expert",)
    after, why = e3.reduce_trace(rows, HOLE)
    assert why is None
    assert after == before


def test_a_trace_with_no_grasp_below_the_band_ceiling_is_dropped():
    rows = [(0.02, 0.0, 0.30, 1.0, "policy")] + [(0.02, 0.0, 0.30, 0.0, "policy")] * 60
    row, why = e3.reduce_trace(rows, HOLE)
    assert why == "no grasp below z=0.12"
    assert row == {}


def test_a_short_trace_is_dropped_before_anything_is_inferred_from_it():
    assert e3.reduce_trace([(0.0, 0.0, 0.1, 0.0, "policy")] * 10, HOLE) == ({}, "trace too short")


def test_a_momentary_gripper_dip_is_not_a_grasp():
    rows = build_trace()
    rows.insert(1, (0.020, 0.0, 0.110, 0.0, "policy"))   # one closed frame ...
    rows.insert(2, (0.020, 0.0, 0.110, 1.0, "policy"))   # ... then open again
    row, why = e3.reduce_trace(rows, HOLE)
    assert why is None
    assert row["e0"] == pytest.approx(10.0, abs=1e-6)


# ------------------------------------------------------------------ the log ---


def test_arm_label_never_invents_a_name_for_a_record_that_has_none():
    assert e3.arm_label(None) == "unrecorded"
    assert e3.arm_label({}) == "unrecorded"
    assert e3.arm_label({"actionAggregate": "medoid"}) == "unrecorded"
    assert e3.arm_label({"actionSamples": 1, "actionAggregate": "medoid"}) == "N=1"
    assert e3.arm_label({"actionSamples": 8, "actionAggregate": "medoid"}) == "medoid x8"
    assert e3.arm_label({"actionSamples": 8, "actionAggregate": "mean"}) == "mean x8"


def test_the_session_is_read_off_the_log_path():
    entry = {"logPath": "/x/rollout_ckpt__pi05__20260908_213000_030000_real_20260920_101500.log"}
    assert e3.session_of(entry) == "20260920_101500"
    assert e3.session_of({"logPath": ""}) is None


def test_arms_are_ordered_the_way_the_roadmap_states_them():
    assert e3.order_arms({"N=1", "mean x8", "unrecorded", "medoid x8"}) == [
        "medoid x8",
        "mean x8",
        "N=1",
        "unrecorded",
    ]


def _args(tmp_path: Path, **overrides) -> argparse.Namespace:
    base = dict(
        log=tmp_path / "rollout_log.jsonl",
        traces=tmp_path / "rollout_traces",
        since=None,
        until=None,
        session=None,
        arms_only=True,
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _log(path: Path, entries) -> None:
    path.write_text("\n".join(json.dumps(entry) for entry in entries) + "\n", encoding="utf-8")


def _entry(
    index: int,
    samples: int,
    aggregate: str,
    *,
    recorded: str,
    session: str = "20260920_101500",
    arm_extra: dict | None = None,
    checkpoint: str = "ckpt/030000",
):
    """One graded rollout. `arm_extra` is merged into the arm the way the gateway merges the
    terminal servo's configuration into it, so a test can vary one knob and hold the rest."""
    return {
        "recordedAt": recorded,
        "checkpointId": checkpoint,
        "outcome": "success",
        "rolloutIndex": index,
        "logPath": "/x/rollout_ckpt_real_%s.log" % session,
        "arm": {"actionSamples": samples, "actionAggregate": aggregate, **(arm_extra or {})},
    }


def test_collect_joins_each_record_to_its_own_trace_and_keeps_the_arm(tmp_path):
    entries = [
        _entry(1, 8, "medoid", recorded="2026-09-20T10:15:00+00:00"),
        _entry(2, 1, "medoid", recorded="2026-09-20T10:20:00+00:00"),
    ]
    _log(tmp_path / "rollout_log.jsonl", entries)
    for index in (1, 2):
        write_trace(
            tmp_path / "rollout_traces" / "session_20260920_101500" / ("rollout_%03d.csv" % index),
            build_trace(),
        )
    usable, dropped = e3.collect(_args(tmp_path), HOLE)
    assert dropped == []
    assert [row["arm"] for row in usable] == ["medoid x8", "N=1"]
    assert [row["tag"] for row in usable] == ["101500/001", "101500/002"]
    assert all(row["e1"] == pytest.approx(3.0, abs=1e-6) for row in usable)


def test_a_record_whose_trace_is_gone_is_dropped_loudly_not_skipped(tmp_path):
    _log(tmp_path / "rollout_log.jsonl", [_entry(7, 8, "medoid", recorded="2026-09-20T10:15:00+00:00")])
    usable, dropped = e3.collect(_args(tmp_path), HOLE)
    assert usable == []
    assert len(dropped) == 1
    assert dropped[0]["why"].startswith("no trace at")
    assert dropped[0]["arm"] == "medoid x8"


def test_records_from_before_the_arm_was_recorded_are_left_out_by_default(tmp_path):
    old = _entry(1, 8, "medoid", recorded="2026-09-11T10:00:00+00:00")
    old.pop("arm")
    _log(tmp_path / "rollout_log.jsonl", [old])
    write_trace(
        tmp_path / "rollout_traces" / "session_20260920_101500" / "rollout_001.csv", build_trace()
    )
    assert e3.collect(_args(tmp_path), HOLE) == ([], [])
    usable, dropped = e3.collect(_args(tmp_path, arms_only=False), HOLE)
    assert [row["arm"] for row in usable] == ["unrecorded"]


def test_the_window_bounds_are_applied_to_recordedat(tmp_path):
    entries = [
        _entry(1, 8, "medoid", recorded="2026-09-19T23:00:00+00:00"),
        _entry(2, 8, "medoid", recorded="2026-09-20T10:20:00+00:00"),
    ]
    _log(tmp_path / "rollout_log.jsonl", entries)
    for index in (1, 2):
        write_trace(
            tmp_path / "rollout_traces" / "session_20260920_101500" / ("rollout_%03d.csv" % index),
            build_trace(),
        )
    usable, _ = e3.collect(_args(tmp_path, since="2026-09-20"), HOLE)
    assert [row["rolloutIndex"] for row in usable] == [2]
    usable, _ = e3.collect(_args(tmp_path, until="2026-09-20"), HOLE)
    assert [row["rolloutIndex"] for row in usable] == [1]


def test_the_rig_verdict_rides_along_without_being_merged_into_the_grade(tmp_path):
    entry = _entry(1, 8, "medoid", recorded="2026-09-20T10:15:00+00:00")
    entry["outcome"] = "failure"
    entry["terminalServo"] = {"verdict": "seated", "searchIndex": 3}
    _log(tmp_path / "rollout_log.jsonl", [entry])
    write_trace(
        tmp_path / "rollout_traces" / "session_20260920_101500" / "rollout_001.csv", build_trace()
    )
    usable, _ = e3.collect(_args(tmp_path), HOLE)
    assert usable[0]["outcome"] == "failure"
    assert usable[0]["terminalServo"]["verdict"] == "seated"


def test_a_missing_log_says_so_rather_than_reporting_an_empty_table(tmp_path):
    with pytest.raises(e3.E3Error, match="no rollout log"):
        e3.collect(_args(tmp_path), HOLE)


# ------------------------------------------------------------------ end to end ---


def _run(tmp_path: Path, order, *extra, checkpoints=None) -> str:
    """Write a log whose arms appear in `order`, one trace each, and render the report.

    An entry of `order` is `(samples, aggregate)`, or `(samples, aggregate, arm_extra)` when the
    test needs to vary a field of the arm block that `arm_label` does not name.
    """
    entries = []
    for index, spec in enumerate(order, start=1):
        samples, aggregate, arm_extra = (spec + ({},))[:3] if len(spec) == 2 else spec
        entries.append(
            _entry(
                index,
                samples,
                aggregate,
                recorded="2026-09-20T10:%02d:00+00:00" % index,
                arm_extra=arm_extra,
                checkpoint=(checkpoints[index - 1] if checkpoints else "ckpt/030000"),
            )
        )
        # A per-round offset, the same set for every arm: the arms then differ by nothing, which
        # is the state E3's falsification is supposed to recognise, while the values still vary
        # enough for a rank test to be defined at all.
        offset = 0.001 * ((index - 1) // 3)
        write_trace(
            tmp_path / "rollout_traces" / "session_20260920_101500" / ("rollout_%03d.csv" % index),
            build_trace(descent=[(z, x + offset) for z, x in DESCENT]),
        )
    _log(tmp_path / "rollout_log.jsonl", entries)
    import io
    from contextlib import redirect_stdout

    buffer = io.StringIO()
    with redirect_stdout(buffer):
        code = e3.main(
            [
                "--log", str(tmp_path / "rollout_log.jsonl"),
                "--traces", str(tmp_path / "rollout_traces"),
                "--hole", "0", "0",
                *extra,
            ]
        )
    assert code == 0
    return buffer.getvalue()


BLOCKED = [(8, "medoid")] * 5 + [(8, "mean")] * 5 + [(1, "medoid")] * 5
INTERLEAVED = [(8, "medoid"), (8, "mean"), (1, "medoid")] * 5


def test_running_the_arms_in_blocks_is_called_out_as_unreadable(tmp_path):
    """The fixture creeps within a session, so a blocked run cannot separate arm from drift."""
    out = _run(tmp_path, BLOCKED)
    assert "ARMS ARE BLOCKED IN TIME, NOT INTERLEAVED" in out
    assert "Re-run interleaved" in out


def test_an_interleaved_run_passes_the_audit(tmp_path):
    out = _run(tmp_path, INTERLEAVED)
    assert "ARMS ARE BLOCKED IN TIME" not in out
    assert "no detectable block structure" in out


def test_the_three_arms_appear_in_the_roadmaps_order_with_their_own_counts(tmp_path):
    out = _run(tmp_path, INTERLEAVED)
    table = out.split("=== HANDOFF TABLE")[1].splitlines()
    arms = [line.split()[0] for line in table[2:5]]
    assert arms == ["medoid", "mean", "N=1"]


def test_identical_arms_are_reported_as_indistinguishable(tmp_path):
    """Every arm drives the same synthetic descent, so E3's falsification must trip."""
    out = _run(tmp_path, INTERLEAVED)
    assert "Kruskal-Wallis p=1.0000" in out
    assert "indistinguishable" in out
    assert "Sampling aggregation closes" in out


def test_the_json_dump_carries_the_hole_it_was_reduced_against(tmp_path):
    out = _run(tmp_path, INTERLEAVED, "--json", str(tmp_path / "e3.json"))
    assert "wrote" in out
    payload = json.loads((tmp_path / "e3.json").read_text())
    assert payload["hole"] == [0.0, 0.0]
    assert payload["holeSource"] == "given on the command line"
    assert payload["band"] == [0.08, 0.12]
    assert len(payload["usable"]) == 15


def test_a_half_written_last_line_is_skipped_not_fatal(tmp_path, capsys):
    """The log is append-only and this runs between rollouts, so the tail can be mid-write."""
    good = _entry(1, 8, "medoid", recorded="2026-09-20T10:15:00+00:00")
    (tmp_path / "rollout_log.jsonl").write_text(
        json.dumps(good) + "\n" + '{"recordedAt": "2026-09-20T10:2', encoding="utf-8"
    )
    write_trace(
        tmp_path / "rollout_traces" / "session_20260920_101500" / "rollout_001.csv", build_trace()
    )
    usable, _ = e3.collect(_args(tmp_path), HOLE)
    assert [row["rolloutIndex"] for row in usable] == [1]
    assert "still being written" in capsys.readouterr().err


def test_a_bad_line_in_the_middle_refuses_rather_than_losing_a_rollout(tmp_path):
    good = _entry(1, 8, "medoid", recorded="2026-09-20T10:15:00+00:00")
    (tmp_path / "rollout_log.jsonl").write_text(
        "{not json}\n" + json.dumps(good) + "\n", encoding="utf-8"
    )
    with pytest.raises(e3.E3Error, match="line 1 is not JSON"):
        e3.collect(_args(tmp_path), HOLE)


# ------------------------------------------------------------------ confound audit ---

# The arm as it is actually launched: the window the draws are scored over, and the z the policy
# is taken off the task at. `arm_label` names neither, which is exactly why they are audited.
ARM = {"selectionHorizon": 25, "terminalServoHandoffZ": 0.12}


def _rounds(count: int, extra_for_round) -> list:
    """`count` interleaved rounds of medoid / mean / N=1, each round's arm extras from a callable."""
    order = []
    for index in range(count):
        for samples, aggregate in ((8, "medoid"), (8, "mean"), (1, "medoid")):
            order.append((samples, aggregate, extra_for_round(index)))
    return order


def test_the_arm_block_reaches_the_row_whole_not_just_the_two_fields_of_the_label(tmp_path):
    """`arm_label` has to stay coarse to compare anything; the fields it drops still decide what
    the rollout was, so they must survive to where they can be audited."""
    _log(
        tmp_path / "rollout_log.jsonl",
        [_entry(1, 8, "medoid", recorded="2026-09-20T10:15:00+00:00", arm_extra=ARM)],
    )
    write_trace(
        tmp_path / "rollout_traces" / "session_20260920_101500" / "rollout_001.csv",
        build_trace(),
    )

    usable, _ = e3.collect(_args(tmp_path), HOLE)

    assert usable[0]["arm"] == "medoid x8"
    assert usable[0]["armFields"]["selectionHorizon"] == 25
    assert usable[0]["armFields"]["terminalServoHandoffZ"] == 0.12


def test_one_selection_window_across_the_run_is_reported_and_the_verdict_stands(tmp_path):
    out = _run(tmp_path, _rounds(5, lambda _round: ARM))

    assert "selectionHorizon" in out
    assert "VARIES" not in out
    assert "NO VERDICT" not in out
    assert "indistinguishable" in out


def test_two_selection_windows_in_one_comparison_refuse_a_verdict(tmp_path):
    """The pooling this audit exists for. A medoid x8 scored over 16 steps and one scored over 25
    carry the same label, so they land in one bucket and leave as one median -- and 16 against 25
    is not hypothetical, it is the old default against the corrected one.
    """
    out = _run(
        tmp_path,
        _rounds(5, lambda round_index: {**ARM, "selectionHorizon": 16 if round_index == 4 else 25}),
    )

    assert "*** selectionHorizon" in out
    assert "VARIES: 16 x3  25 x12" in out
    assert "NO VERDICT: selectionHorizon" in out
    # The refusal replaces the verdict rather than sitting above it: a reader who scrolled past
    # the audit would have no way to tell a pooled median from a readable one.
    assert "indistinguishable" not in out
    assert "Kruskal-Wallis p" in out  # the interleaving audit still runs and still reports


def test_pinning_the_offset_for_part_of_a_run_is_a_second_arm(tmp_path):
    """Absent means the offset is re-derived from measured latency; a number means it was pinned.
    Those are two rules, and the draws were scored over two different stretches of the chunk."""
    out = _run(
        tmp_path,
        _rounds(
            5,
            lambda round_index: {**ARM, **({"selectionOffsetSteps": 10} if round_index < 2 else {}),},
        ),
    )

    assert "*** selectionOffsetSteps" in out
    assert "VARIES: (absent) x9  10 x6" in out
    assert "NO VERDICT" in out


def test_moving_the_handoff_mid_comparison_refuses_because_it_moves_the_measurement(tmp_path):
    """Criterion A is the lateral error where the policy stopped driving. Change the z it stops
    at and the two halves of the run are not measuring at the same place."""
    out = _run(
        tmp_path,
        _rounds(5, lambda round_index: {**ARM, "terminalServoHandoffZ": 0.11 if round_index == 0 else 0.12}),
    )

    assert "*** terminalServoHandoffZ" in out
    assert "NO VERDICT" in out


def test_a_second_checkpoint_in_one_comparison_refuses(tmp_path):
    out = _run(
        tmp_path,
        _rounds(2, lambda _round: ARM),
        checkpoints=["ckpt/030000"] * 3 + ["ckpt/040000"] * 3,
    )

    assert "*** checkpointId" in out
    assert "NO VERDICT" in out


def test_opening_the_search_ring_is_noted_but_never_refused_over(tmp_path):
    """E5's argument, enforced: the search runs strictly after the handoff, so it cannot reach a
    distance measured at the handoff. It does move the success column, and that is said."""
    out = _run(
        tmp_path,
        _rounds(
            5,
            lambda round_index: {
                **ARM,
                "terminalServoSearchRingM": 0.007 if round_index < 3 else 0.0,
            },
        ),
    )

    assert "note terminalServoSearchRingM" in out
    assert "*** terminalServoSearchRingM" not in out
    assert "NO VERDICT" not in out
    assert "indistinguishable" in out
    assert "not comparable with one read at a single setting" in out
