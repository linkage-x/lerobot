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
    request_continue,
    request_grade,
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


def test_all_loops_are_tenants_of_the_same_contract():
    """One page, two tenants. If this drifts, the page has to grow a special case."""

    assert set(RUN_KINDS) == {"terminal_trials", "auto_collect", "grasp_envelope"}
    for kind in RUN_KINDS.values():
        assert kind.label and kind.unit and kind.script.endswith(".py")


def test_reading_a_run_that_does_not_exist_says_so(tmp_path):
    with pytest.raises(UnattendedError, match="no such run"):
        read_run(tmp_path, "nope")


# -- the first real launch of the envelope sweep died on its argv and read as "running" -----------


def test_a_child_that_exited_reads_as_gone_even_before_anyone_waits_on_it():
    """signal 0 succeeds on a zombie, so a run that died on line one used to read as running."""

    import subprocess
    import sys

    child = subprocess.Popen([sys.executable, "-c", "pass"])
    deadline = time.time() + 10
    while time.time() < deadline:
        try:
            state = open(f"/proc/{child.pid}/stat").read().rsplit(")", 1)[1].split()[0]
        except OSError:
            break
        if state == "Z":
            break
        time.sleep(0.02)
    assert process_alive(child.pid) is False


def test_a_run_that_died_before_writing_a_row_is_crashed_and_shows_why(tmp_path):
    run_dir = _make_run(tmp_path, "grasp_envelope_20260923_144600", pid=DEAD_PID, kind="grasp_envelope")
    (run_dir / "run.log").write_text("usage: ...\nerror: argument --xy-offsets-mm: expected one argument\n")
    run = read_run(tmp_path, run_dir.name)
    assert run["state"] == "crashed", "never 'starting' forever"
    assert run["logTail"][-1].endswith("expected one argument")


@pytest.mark.parametrize(
    ("kind", "module", "request_"),
    [
        ("grasp_envelope", "tools.fr3.fr3_grasp_envelope_runtime", {}),
        ("grasp_envelope", "tools.fr3.fr3_grasp_envelope_runtime", {"xyOffsetsMm": "", "dzOffsetsMm": "", "centreRepeats": 0, "extraPointsMm": "-5,0,-6"}),
        ("terminal_trials", "tools.fr3.fr3_terminal_trials_runtime", {"holePose": "0.3599,-0.1333,0.0523"}),
        ("terminal_trials", "tools.fr3.fr3_terminal_trials_runtime",
         {"holePose": "0.3599,-0.1333,0.0523", "offsetsMm": "0", "searchRingM": "0.007",
          "operatorGrade": "1", "regripInPlace": "1", "graspAttempts": "2"}),
    ],
)
def test_every_planned_argv_is_accepted_by_the_runtime_it_launches(tmp_path, kind, module, request_):
    """Planning and launching are separate processes; the only contract between them is argv."""

    import importlib

    planned = plan_run(tmp_path, kind, request_)
    runtime = importlib.import_module(module)
    args = runtime.parse_args(planned["argv"][1:])
    if kind == "grasp_envelope":
        request = runtime.build_request(args)
        assert request.xyOffsetsMm == tuple(planned["plan"]["request"]["xyOffsetsMm"])
        assert request.dzOffsetsMm == tuple(planned["plan"]["request"]["dzOffsetsMm"])


def test_a_graded_terminal_plan_reaches_the_runtime_with_its_switches(tmp_path):
    import tools.fr3.fr3_terminal_trials_runtime as runtime

    planned = plan_run(tmp_path, "terminal_trials", {
        "holePose": "0.3599,-0.1333,0.0523", "offsetsMm": "0", "searchRingM": "0.007",
        "operatorGrade": "1", "regripInPlace": "yes", "graspAttempts": "2", "refetchEvery": "8",
        "refetchOnTrouble": "1",
    })
    request = runtime.build_request(runtime.parse_args(planned["argv"][1:]))
    assert request.operatorGrade and request.regripInPlace and not request.releaseOnlyWhenSeated
    assert request.graspAttempts == 2
    assert request.refetchEvery == 8 and request.refetchOnTrouble
    assert any(arg.startswith("--grade-file=") and arg.endswith("GRADE") for arg in planned["argv"])
    ungraded = plan_run(tmp_path, "terminal_trials", {"holePose": "0.3599,-0.1333,0.0523"})
    assert "--operator-grade" not in ungraded["argv"]
    assert not any(arg.startswith("--refetch") for arg in ungraded["argv"])
    assert "--home-first" not in ungraded["argv"]
    homed = plan_run(tmp_path, "terminal_trials", {"holePose": "0.3599,-0.1333,0.0523",
                                                   "homeFirst": "1", "fixedHole": "1"})
    assert runtime.parse_args(homed["argv"][1:]).home_first
    assert not runtime.build_request(runtime.parse_args(homed["argv"][1:])).updateReference


def test_a_grade_question_is_answered_with_in_or_out_and_not_with_continue(tmp_path):
    question = {"kind": "needs_operator", "grade": True, "message": "trial 003: 销在孔里吗？"}
    run_dir = _make_run(tmp_path, "terminal_trials_G", pid=os.getpid(), rows=[_trial(0), question])
    run = read_run(tmp_path, "terminal_trials_G")
    assert run["needsGrade"] is True and run["needsOperator"].startswith("trial 003")
    with pytest.raises(UnattendedError):
        request_continue(tmp_path, "terminal_trials_G")
    with pytest.raises(UnattendedError, match="'in', 'out' or 'reset'"):
        request_grade(tmp_path, "terminal_trials_G", "maybe")
    request_grade(tmp_path, "terminal_trials_G", "OUT")
    assert (run_dir / "GRADE").read_text() == "out"


def test_an_answered_or_dead_question_asks_for_nothing(tmp_path):
    question = {"kind": "needs_operator", "grade": True, "message": "q"}
    _make_run(tmp_path, "terminal_trials_H", pid=os.getpid(),
              rows=[question, {"kind": "operator", "trial": 0, "grade": "in"}])
    assert read_run(tmp_path, "terminal_trials_H")["needsGrade"] is False
    with pytest.raises(UnattendedError, match="not asking"):
        request_grade(tmp_path, "terminal_trials_H", "in")
    _make_run(tmp_path, "terminal_trials_I", pid=DEAD_PID, rows=[question])
    assert read_run(tmp_path, "terminal_trials_I")["needsGrade"] is False


def test_a_second_halt_kills_a_run_that_did_not_answer_the_first(tmp_path, monkeypatch):
    import signal
    import tools.data_collection_gui.unattended as unattended

    sent = []
    # Signal 0 is the liveness probe; only the real ones are the brake.
    monkeypatch.setattr(unattended.os, "kill", lambda pid, sig: sig and sent.append(sig))
    run_dir = _make_run(tmp_path, "terminal_trials_K", pid=os.getpid(), rows=[_trial(0)])
    request_stop(tmp_path, "terminal_trials_K", mode="now")
    request_stop(tmp_path, "terminal_trials_K", mode="now")
    assert sent == [signal.SIGINT, signal.SIGKILL]
    assert (run_dir / "HALT").exists()
