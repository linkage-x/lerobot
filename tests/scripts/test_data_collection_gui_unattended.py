"""Runs that outlive the page watching them: can the gateway be wrong about one, and does it say so?

The failure this module exists to prevent is not a crash, it is a confident wrong answer. A run
whose process is gone and whose rows end in a summary finished. One whose process is gone and whose
rows do not *stopped at 2 a.m. for a reason nobody has read*, and on a row count alone the two look
identical. Most of what follows is about keeping them apart, and about the state surviving things
that routinely happen over eight hours: the page closing, the gateway restarting, the connection
dropping.
"""

import json
import os
import time

import pytest

from tools.data_collection_gui.unattended import (
    RUN_KINDS,
    UnattendedError,
    active_run,
    list_runs,
    plan_run,
    process_alive,
    read_run,
    release_brake,
    request_stop,
    runs_root,
)


def _make_run(root, run_id, *, pid=None, rows=(), plan=None, kind="terminal_trials"):
    run_dir = runs_root(root) / run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "plan.json").write_text(
        json.dumps(plan or {"kind": kind, "units": 10}), encoding="utf-8"
    )
    (run_dir / "run.json").write_text(
        json.dumps({"id": run_id, "kind": kind, "pid": pid, "startedAt": time.time(), "argv": []}),
        encoding="utf-8",
    )
    if rows:
        (run_dir / "rows.jsonl").write_text(
            "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
        )
    return run_dir


def _trial(index, verdict="seated"):
    return {"kind": "trial", "index": index, "ok": True, "verdict": verdict}


DEAD_PID = 2 ** 22  # far above any live pid on a normal system


def test_a_run_that_stopped_without_a_summary_is_crashed_and_not_complete(tmp_path):
    """The single most important distinction here, and the one a row count cannot make."""

    _make_run(tmp_path, "terminal_trials_A", pid=DEAD_PID, rows=[_trial(0), _trial(1)])
    assert read_run(tmp_path, "terminal_trials_A")["state"] == "crashed"

    _make_run(
        tmp_path, "terminal_trials_B", pid=DEAD_PID,
        rows=[_trial(0), {"kind": "summary", "ok": True, "haltedOn": "schedule_complete"}],
    )
    assert read_run(tmp_path, "terminal_trials_B")["state"] == "complete"


def test_a_run_that_stopped_on_a_condition_is_halted_rather_than_complete(tmp_path):
    """It ended in an orderly way and it did not finish. Both halves have to show."""

    _make_run(
        tmp_path, "terminal_trials_C", pid=DEAD_PID,
        rows=[_trial(0), {"kind": "summary", "ok": False, "haltedOn": "slip_streak"}],
    )
    run = read_run(tmp_path, "terminal_trials_C")
    assert run["state"] == "halted"
    assert run["summary"]["haltedOn"] == "slip_streak"


def test_a_live_process_is_running_whatever_its_rows_say(tmp_path):
    _make_run(tmp_path, "terminal_trials_D", pid=os.getpid(), rows=[_trial(0)])
    assert read_run(tmp_path, "terminal_trials_D")["state"] == "running"


def test_state_is_read_from_the_directory_so_a_gateway_restart_loses_nothing(tmp_path):
    """Nothing is held in memory, so "re-attach" is not a feature -- it is reading a directory."""

    _make_run(tmp_path, "terminal_trials_E", pid=os.getpid(), rows=[_trial(i) for i in range(4)])
    first = read_run(tmp_path, "terminal_trials_E")
    # A whole new process would do exactly this and get exactly this.
    second = read_run(tmp_path, "terminal_trials_E")
    assert first["unitsDone"] == second["unitsDone"] == 4
    assert first["state"] == second["state"] == "running"


def test_a_half_written_last_row_does_not_stop_the_page_rendering(tmp_path):
    """The file is being appended to while it is read. That is normal, not an error."""

    run_dir = _make_run(tmp_path, "terminal_trials_F", pid=os.getpid(), rows=[_trial(0), _trial(1)])
    with (run_dir / "rows.jsonl").open("a", encoding="utf-8") as handle:
        handle.write('{"kind": "trial", "index": 2, "ok": tr')
    run = read_run(tmp_path, "terminal_trials_F")
    assert run["unitsDone"] == 2
    assert run["rowsTotal"] == 3, "the torn line is counted but not parsed"


def test_the_two_brakes_are_different_actions(tmp_path):
    run_dir = _make_run(tmp_path, "terminal_trials_G", pid=DEAD_PID, rows=[_trial(0)])
    assert read_run(tmp_path, "terminal_trials_G")["stopRequested"] is False

    request_stop(tmp_path, "terminal_trials_G", mode="boundary")
    run = read_run(tmp_path, "terminal_trials_G")
    assert run["stopRequested"] is True and (run_dir / "STOP").exists()
    assert "requested at" in run["stopReason"]

    # The immediate brake needs a live process; asking a dead one is an error, not a no-op.
    with pytest.raises(UnattendedError, match="not running"):
        request_stop(tmp_path, "terminal_trials_G", mode="now")


def test_a_boundary_brake_can_be_released_before_it_is_acted_on(tmp_path):
    _make_run(tmp_path, "terminal_trials_H", pid=DEAD_PID, rows=[_trial(0)])
    request_stop(tmp_path, "terminal_trials_H", mode="boundary")
    assert read_run(tmp_path, "terminal_trials_H")["stopRequested"] is True
    release_brake(tmp_path, "terminal_trials_H")
    assert read_run(tmp_path, "terminal_trials_H")["stopRequested"] is False


def test_an_unknown_stop_mode_is_refused_rather_than_defaulting_to_one_of_them(tmp_path):
    """Against a real run, so the refusal is about the mode and not about the run being missing."""

    _make_run(tmp_path, "terminal_trials_K", pid=DEAD_PID, rows=[_trial(0)])
    with pytest.raises(UnattendedError, match="unknown stop mode"):
        request_stop(tmp_path, "terminal_trials_K", mode="maybe")
    assert read_run(tmp_path, "terminal_trials_K")["stopRequested"] is False, (
        "an unrecognised mode must not fall through to the gentler brake"
    )


def test_there_is_one_arm_so_there_is_one_run(tmp_path):
    _make_run(tmp_path, "terminal_trials_I", pid=os.getpid(), rows=[_trial(0)])
    assert active_run(tmp_path)["id"] == "terminal_trials_I"
    with pytest.raises(UnattendedError, match="one arm"):
        from tools.data_collection_gui.unattended import start_run

        start_run(tmp_path, "terminal_trials", {"holePose": "0.36,-0.13,0.05"})


def test_a_finished_run_does_not_block_the_next_one(tmp_path):
    _make_run(
        tmp_path, "terminal_trials_J", pid=DEAD_PID,
        rows=[{"kind": "summary", "ok": True, "haltedOn": "schedule_complete"}],
    )
    assert active_run(tmp_path) is None


def test_runs_are_listed_newest_first_with_enough_to_draw_a_row(tmp_path):
    for index in range(3):
        _make_run(tmp_path, f"terminal_trials_2026091{index}_000000", pid=DEAD_PID, rows=[_trial(0)])
    listed = list_runs(tmp_path)
    assert all(entry["id"].endswith("_000000") for entry in listed)
    assert [entry["id"] for entry in listed] == sorted(
        (entry["id"] for entry in listed), reverse=True
    ), "newest first"
    assert set(listed[0]) >= {"id", "kind", "state", "unitsDone", "unitsPlanned", "stopRequested"}


def test_a_pid_that_is_gone_reads_as_gone_and_our_own_reads_as_alive():
    assert process_alive(os.getpid()) is True
    assert process_alive(DEAD_PID) is False
    assert process_alive(None) is False
    assert process_alive(0) is False


# -- planning is what "authorise" means, so it happens before anything is written --------------


def test_a_plan_is_expanded_and_fence_checked_without_writing_or_moving_anything(tmp_path):
    planned = plan_run(tmp_path, "terminal_trials", {"holePose": "0.3599,-0.1333,0.0523", "repeats": 2})
    assert planned["kind"] == "terminal_trials" and planned["unit"] == "trial"
    assert planned["plan"]["units"] == len(planned["plan"]["schedule"]) > 0
    assert planned["plan"]["qc"]["ok"] is True
    assert "trials=" in planned["plan"]["text"], "the plan a person reads is in the plan"
    assert not runs_root(tmp_path).exists(), "planning must not create a run directory"


def test_a_plan_that_cannot_pass_the_fence_is_refused_at_planning_time(tmp_path):
    with pytest.raises(Exception):
        plan_run(tmp_path, "terminal_trials", {"holePose": "9.0,-0.1333,0.0523"})


def test_a_missing_required_field_is_named(tmp_path):
    with pytest.raises(UnattendedError, match="holePose is required"):
        plan_run(tmp_path, "terminal_trials", {})


def test_an_unknown_kind_is_refused(tmp_path):
    with pytest.raises(UnattendedError, match="unknown run kind"):
        plan_run(tmp_path, "not_a_kind", {})


def test_both_loops_are_tenants_of_the_same_contract():
    """One page, two tenants. If this drifts, the page has to grow a special case."""

    assert set(RUN_KINDS) == {"terminal_trials", "auto_collect"}
    for kind in RUN_KINDS.values():
        assert kind.label and kind.unit and kind.script.endswith(".py")


def test_reading_a_run_that_does_not_exist_says_so(tmp_path):
    with pytest.raises(UnattendedError, match="no such run"):
        read_run(tmp_path, "nope")
