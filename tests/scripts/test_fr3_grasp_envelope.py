"""P0 grasp envelope: does each verdict mean what it says, and does the peg stay where the loop thinks?

The sweep is only worth running if its four numbers come from verdicts a person would agree with,
and if every trial starts from a peg standing at the spot. So the fake rig below is a peg with a
known envelope -- fingers within `CAPTURE_M` straddle it, a finger just past that lands on its top,
one further out clips it over, and fingertips below `TABLE_TCP_Z` meet the table -- and the tests
check the loop reads those four regions back out.
"""

import json
import math

import pytest

import tools.fr3.grasp_envelope as grasp_envelope
import tools.fr3.grasp_loop as grasp_loop
import tools.fr3.scene_reset as scene_reset
import tools.fr3.terminal_servo as terminal_servo
from tools.data_collection_gui.unattended import plan_run, read_run, request_continue, runs_root
from tools.fr3.grasp_envelope import (
    FileOperatorGate,
    GraspEnvelopeRequest,
    build_envelope_schedule,
    parse_points_mm,
    run_grasp_envelope,
    summarize_envelope,
    validate_grasp_envelope,
)
from tools.fr3.scene_reset import SceneResetError

from tests.scripts.test_fr3_scene_reset import FakeRobot


PICK = (0.3640, -0.1370, 0.0550)
SPOT = (0.4331, -0.1667)
REF_Z = 0.058
CAPTURE_M = 0.006
TOP_LAND_M = 0.009
KNOCK_M = 0.0115
TABLE_TCP_Z = 0.050
GRASP_Z_BAND = (0.050, 0.066)
PEG_TOP_TCP_Z = 0.075
FENCE = dict(workspace_min=(0.18, -0.45, 0.0), workspace_max=(0.70, 0.45, 0.70))


@pytest.fixture(autouse=True)
def _fast(monkeypatch):
    monkeypatch.setattr(scene_reset, "SCENE_RESET_MAX_SPEED_MS", 50.0)
    for module in (scene_reset, grasp_loop, grasp_envelope, terminal_servo):
        monkeypatch.setattr(module, "precise_sleep", lambda seconds: None)


class EnvelopeRig(FakeRobot):
    """One peg, described by the TCP pose that grips it nominally, and a table."""

    HELD_WIDTH = 0.31
    EMPTY_WIDTH = 0.006

    def __init__(self, peg=(*SPOT, REF_Z)):
        super().__init__()
        self.xyz = (SPOT[0], SPOT[1], 0.20)
        self.peg = tuple(peg)  # None once it has been knocked over
        self.held = False
        self.closed = False
        self.knocks = 0

    def _lateral(self, xyz):
        return math.dist(xyz[:2], self.peg[:2]) if self.peg is not None else math.inf

    def send_action(self, action):
        commanded = float(action["gripper.pos"])
        x, y, z = float(action["ee.x"]), float(action["ee.y"]), float(action["ee.z"])
        if not self.held:
            floor = TABLE_TCP_Z
            lateral = self._lateral((x, y))
            if self.peg is not None and CAPTURE_M < lateral <= TOP_LAND_M:
                floor = max(floor, PEG_TOP_TCP_Z)  # an open finger lands on the peg's top
            elif self.peg is not None and TOP_LAND_M < lateral <= KNOCK_M and z < PEG_TOP_TCP_Z:
                self.peg = None  # clipped on the way past
                self.knocks += 1
            z = max(z, floor)
        result = super().send_action({**action, "ee.z": z})
        if commanded >= 0.5:
            if self.held:
                # Released: it stands on the table wherever the fingers put it down.
                self.peg = (self.xyz[0], self.xyz[1], REF_Z)
            self.held = self.closed = False
            self.gripper = commanded
            return result
        if not self.closed:
            self.closed = True
            self.held = (
                self.peg is not None
                and self._lateral(self.xyz) <= CAPTURE_M
                and GRASP_Z_BAND[0] <= self.xyz[2] <= GRASP_Z_BAND[1]
            )
        if self.held:
            self.peg = (self.xyz[0], self.xyz[1], REF_Z)
        self.gripper = self.HELD_WIDTH if self.held else self.EMPTY_WIDTH
        return result


def _request(**overrides):
    fields = dict(
        spotXy=SPOT,
        pegRefZ=REF_Z,
        xyOffsetsMm=(0.0,),
        dzOffsetsMm=(),
        centreRepeats=0,
        start="spot",
        pickXyz=PICK,
        requestId="t",
    )
    fields.update(overrides)
    return GraspEnvelopeRequest(**fields)


def _run(rig, request, schedule=None, **kwargs):
    rows = []
    summary = run_grasp_envelope(
        rig, request, schedule if schedule is not None else build_envelope_schedule(request), on_row=rows.append, **kwargs
    )
    return summary, rows


def _trials(rows):
    return [r for r in rows if r.get("kind") == "trial"]


# ------------------------------------------------------------------------------ schedule ---


def test_the_default_plan_is_the_grid_the_column_and_the_centre_repeats_shuffled_and_fenced():
    request = GraspEnvelopeRequest(requestId="t")
    schedule = build_envelope_schedule(request)
    assert len(schedule) == 49 + 9 + 5
    assert [p.index for p in schedule] == list(range(len(schedule)))
    # Seeded: the same plan expands to the same order, which is what resuming by index needs.
    assert schedule == build_envelope_schedule(request)
    assert schedule != build_envelope_schedule(GraspEnvelopeRequest(requestId="t", seed=1))
    # Not walked in grid order, so drift in the spot cannot pose as a spatial pattern.
    assert [p.block for p in schedule[:10]] != ["xy"] * 10 or [p.dxMm for p in schedule[:7]] != [-12.0] * 7
    point = next(p for p in schedule if p.block == "xy" and p.dxMm == 8.0 and p.dyMm == -4.0)
    assert point.aimXyz == pytest.approx((0.4331 + 0.008, -0.1667 - 0.004, 0.058 - 0.006))
    qc = validate_grasp_envelope(request, schedule, **FENCE)
    assert qc["ok"] and qc["points"] == 63 and qc["lowestTcpZ"] == pytest.approx(0.046)


def test_an_aim_below_the_floor_is_refused_before_anything_moves():
    request = GraspEnvelopeRequest(dzOffsetsMm=(-20.0,), requestId="t")
    with pytest.raises(SceneResetError, match="below the floor"):
        validate_grasp_envelope(request, build_envelope_schedule(request), **FENCE)


def test_a_descent_faster_than_the_contact_test_was_validated_at_is_refused():
    request = GraspEnvelopeRequest(descentSpeedMs=0.1, requestId="t")
    with pytest.raises(SceneResetError, match="outruns the contact test"):
        validate_grasp_envelope(request, build_envelope_schedule(request), **FENCE)


def test_fine_scan_points_parse_and_malformed_ones_are_named():
    assert parse_points_mm("4,0,-6; 5, 0, -6;") == ((4.0, 0.0, -6.0), (5.0, 0.0, -6.0))
    with pytest.raises(SceneResetError, match="not dx,dy,dz"):
        parse_points_mm("4,0")


# ------------------------------------------------------------------------------ verdicts ---


def test_each_region_of_the_envelope_reads_back_as_its_own_verdict():
    rig = EnvelopeRig()
    request = _request(
        extraPointsMm=(
            (0.0, 0.0, -6.0),  # inside: held
            (7.5, 0.0, -6.0),  # a finger lands on the top: contact
            (0.0, 0.0, 14.0),  # above the grasp band: closes on air
            (20.0, 0.0, -6.0),  # wide of the peg altogether: closes on air, peg untouched
        ),
        xyOffsetsMm=(),
    )
    summary, rows = _run(rig, request)
    by_offset = {(r["dxMm"], r["dzMm"]): r for r in _trials(rows)}
    assert by_offset[(0.0, -6.0)]["verdict"] == "held" and by_offset[(0.0, -6.0)]["success"]
    assert by_offset[(7.5, -6.0)]["verdict"] == "contact" and by_offset[(7.5, -6.0)]["contactBeforeClose"]
    assert by_offset[(7.5, -6.0)]["contactZ"] >= PEG_TOP_TCP_Z - 0.001
    assert by_offset[(0.0, 14.0)]["verdict"] == "empty" and by_offset[(0.0, 14.0)]["emptyGrasp"]
    assert by_offset[(20.0, -6.0)]["verdict"] == "empty"
    # None of these moved the peg, so every verify grasp found it.
    assert not any(r["pegDisturbed"] for r in _trials(rows))
    assert summary["ok"] and summary["haltedOn"] == "schedule_complete"


def test_a_held_grasp_off_centre_is_put_back_and_the_verify_grasp_recentres_the_peg():
    rig = EnvelopeRig()
    request = _request(xyOffsetsMm=(), extraPointsMm=((4.0, 0.0, -6.0), (-4.0, 0.0, -6.0), (0.0, 4.0, -6.0)))
    summary, rows = _run(rig, request)
    assert [r["verdict"] for r in _trials(rows)] == ["held"] * 3
    # After every trial the peg is back at the spot, because the verify grasp closes on it there.
    assert math.dist(rig.peg[:2], SPOT) < 1e-6
    # The close offset is measured, not echoed: it is where the TCP was relative to the peg ref.
    for row in _trials(rows):
        assert row["closeOffsetMm"][0] == pytest.approx(row["dxMm"], abs=0.5)
        assert row["closeOffsetMm"][2] == pytest.approx(-6.0, abs=0.5)


def test_fingertips_below_the_table_read_as_contact_on_the_height_column():
    rig = EnvelopeRig()
    request = _request(xyOffsetsMm=(), dzOffsetsMm=(-14.0, -6.0), minTcpZ=0.040)
    _summary, rows = _run(rig, request)
    by_dz = {r["dzMm"]: r["verdict"] for r in _trials(rows)}
    assert by_dz == {-14.0: "contact", -6.0: "held"}


def test_a_descent_that_stops_a_few_mm_short_is_graded_where_it_stopped_not_called_contact():
    """The real arm's dead-band: short of the setpoint by less than the contact test's 4 mm."""

    rig = EnvelopeRig()
    request = _request(xyOffsetsMm=(), dzOffsetsMm=(-10.0,), minTcpZ=0.040)  # aim 0.048, table 0.050
    _summary, rows = _run(rig, request)
    row = _trials(rows)[0]
    assert row["verdict"] == "held" and row["descentStoppedOn"] == "timeout"
    assert row["descentShortMm"] == pytest.approx(2.0, abs=0.1)
    assert row["closeOffsetMm"][2] == pytest.approx(-8.0, abs=0.1)


# ------------------------------------------------------------------------- lost peg ---


def test_a_knocked_peg_is_flagged_and_restaged_through_a_person_then_the_sweep_goes_on():
    rig = EnvelopeRig(peg=(*SPOT, REF_Z))
    asked = []

    def person(message):
        asked.append(message)
        rig.peg = PICK  # back in the fixture, where the reset stages from
        return True

    request = _request(xyOffsetsMm=(), extraPointsMm=((10.5, 0.0, -6.0), (0.0, 0.0, -6.0)))
    schedule = build_envelope_schedule(request)
    schedule.sort(key=lambda p: -p.dxMm)  # the knock first
    summary, rows = _run(rig, request, schedule, wait_for_operator=person)
    trials = _trials(rows)
    assert trials[0]["pegDisturbed"] and trials[0]["restaged"] and trials[0]["verdict"] == "empty"
    assert "pick fixture" in asked[0]
    assert [r["kind"] for r in rows if r["kind"] in ("needs_operator", "operator")] == ["needs_operator", "operator"]
    assert trials[1]["verdict"] == "held" and not trials[1]["pegDisturbed"]
    assert summary["ok"] and summary["knockOverRadiusMm"] == pytest.approx(10.5)
    assert math.dist(rig.peg[:2], SPOT) < 0.002, "the reset staged the peg back at the spot"


def test_with_nobody_to_ask_a_lost_peg_ends_the_run_named_and_parked():
    rig = EnvelopeRig()
    request = _request(xyOffsetsMm=(), extraPointsMm=((10.5, 0.0, -6.0), (0.0, 0.0, -6.0)))
    schedule = sorted(build_envelope_schedule(request), key=lambda p: -p.dxMm)
    summary, rows = _run(rig, request, schedule)
    assert summary["haltedOn"] == "peg_lost" and not summary["ok"]
    assert len(_trials(rows)) == 1, "the trial that lost it is still written down"
    assert rows[-1]["kind"] == "summary" and rows[-1]["parked"]
    assert rig.move_to_start_calls == 1


def test_a_run_started_from_the_fixture_stages_the_peg_at_the_spot_first():
    rig = EnvelopeRig(peg=PICK)
    request = _request(start="fixture", extraPointsMm=(), xyOffsetsMm=(0.0,), xyDzMm=-6.0)
    summary, rows = _run(rig, request)
    assert rows[0]["kind"] == "staged" and rows[0]["verify"]["verdict"] == "held"
    assert _trials(rows)[0]["verdict"] == "held"
    assert summary["ok"]


def test_the_boundary_stop_is_read_between_trials():
    rig = EnvelopeRig()
    request = _request(xyOffsetsMm=(), extraPointsMm=((0.0, 0.0, -6.0),) * 3)
    calls = iter([False, True, True])
    summary, rows = _run(rig, request, should_stop=lambda: next(calls))
    assert summary["haltedOn"] == "stop_requested" and summary["ok"]
    assert len(_trials(rows)) == 1


# ------------------------------------------------------------------------------- summary ---


def _row(dx, dy, dz, verdict, disturbed=False):
    return {"kind": "trial", "dxMm": dx, "dyMm": dy, "dzMm": dz, "verdict": verdict, "pegDisturbed": disturbed,
            "verifyVerdict": "held" if not disturbed else "empty", "verifyWidth": 0.3}


def test_the_capture_radius_is_a_rate_so_one_outlier_is_named_rather_than_zeroing_it():
    """The 09-23 sweep: one miss at 4 mm, everything out to the 17 mm corners held."""

    request = GraspEnvelopeRequest(requestId="t")
    grid = [(dx, dy) for dx in (-12, -8, -4, 0, 4, 8, 12) for dy in (-12, -8, -4, 0, 4, 8, 12)]
    rows = [_row(dx, dy, -6, "empty" if (dx, dy) == (0, -4) else "held", disturbed=(dx, dy) == (0, -4)) for dx, dy in grid]
    rows += [_row(0, 0, -6, "held")] * 5
    rows.append(_row(-12, -12, -6, "held", disturbed=True))
    summary = summarize_envelope(rows, request)
    assert summary["xyCaptureRadiusMm"] == pytest.approx(16.97)
    assert summary["xyCaptureAtGridEdge"], "the grid ran out before the rate did"
    assert summary["xyMissesInsideCapture"] == [[0, -4]]
    assert summary["knockOverRadiusMm"] is None, "two scattered knocks are not an edge"
    assert summary["xyDisturbed"] == 2


def test_the_capture_radius_stops_where_the_misses_start():
    request = GraspEnvelopeRequest(requestId="t")
    rows = [_row(0, 0, -6, "held")] * 3
    rows += [_row(dx, dy, -6, "held") for dx, dy in ((4, 0), (-4, 0), (0, 4), (0, -4))]
    rows += [_row(dx, dy, -6, "held" if dx < 0 else "empty") for dx, dy in ((8, 0), (-8, 0), (0, 8), (0, -8))]
    rows += [_row(dx, dy, -6, "contact", disturbed=True) for dx, dy in ((12, 0), (-12, 0), (0, 12), (0, -12))]
    summary = summarize_envelope(rows, request)
    assert summary["xyCaptureRadiusMm"] == 4.0, "8 mm is half missed, and the pool drops below the rate"
    assert not summary["xyCaptureAtGridEdge"]
    assert summary["knockOverRadiusMm"] == 12.0


def test_a_peg_tipped_from_above_is_a_height_result_not_a_knock_over_radius():
    request = GraspEnvelopeRequest(requestId="t")
    rows = [_row(0, 0, -6, "held"), _row(4, 0, -6, "held"), _row(0, 0, 20, "empty", disturbed=True)]
    summary = summarize_envelope(rows, request)
    assert summary["knockOverRadiusMm"] is None


def test_a_resumed_run_summarises_the_whole_sweep():
    rig = EnvelopeRig()
    request = _request(xyOffsetsMm=(), extraPointsMm=((0.0, 0.0, -6.0), (4.0, 0.0, -6.0)))
    prior = [_row(8, 0, -6, "held"), _row(12, 0, -6, "held")]
    summary, rows = _run(rig, request, prior_rows=prior)
    assert summary["trials"] == 4 and summary["trialsThisRun"] == 2
    assert rows[-1]["xyCaptureRadiusMm"] == 12.0


def test_the_dz_interval_is_the_contiguous_held_run_around_the_grid_height():
    request = GraspEnvelopeRequest(requestId="t")
    rows = [_row(0, 0, dz, v) for dz, v in ((-12, "contact"), (-8, "held"), (-4, "held"), (0, "held"),
                                            (4, "held"), (8, "empty"), (14, "held"), (20, "empty"))]
    summary = summarize_envelope(rows, request)
    assert summary["graspDzIntervalMm"] == [-8, 4]
    assert summary["graspDzOpen"] == [False, False]
    assert summary["contactBelowDzMm"] == -12


# ------------------------------------------------------------------------------ operator ---


def test_the_file_gate_answers_continue_stop_and_silence(tmp_path):
    path = tmp_path / "CONTINUE"
    ticks = []

    def sleep_then_answer(seconds):
        ticks.append(seconds)
        if len(ticks) == 2:
            path.write_text("go")

    assert FileOperatorGate(path, stop_requested=lambda: False, timeout_s=10, sleep=sleep_then_answer)("put it back")
    assert not path.exists(), "an answer is consumed, so it cannot answer the next question too"
    assert not FileOperatorGate(path, stop_requested=lambda: True, timeout_s=10, sleep=lambda s: None)("x")
    assert not FileOperatorGate(path, stop_requested=lambda: False, timeout_s=1.0, sleep=lambda s: None)("x")


# ---------------------------------------------------------------------- unattended page ---


def test_the_page_plans_it_like_the_other_loops_and_can_resume_a_halted_run(tmp_path):
    planned = plan_run(tmp_path, "grasp_envelope", {})
    assert planned["unit"] == "trial" and planned["plan"]["units"] == 63
    argv = planned["argv"]
    assert argv[0] == "tools/fr3/fr3_grasp_envelope_runtime.py"
    assert any(a.startswith("--continue-file=") for a in argv) and "--home-first" in argv
    assert not runs_root(tmp_path).exists(), "planning must not create a run directory"

    previous = runs_root(tmp_path) / "grasp_envelope_A"
    previous.mkdir(parents=True)
    (previous / "rows.jsonl").write_text(
        "\n".join(json.dumps({"kind": "trial", "index": i}) for i in range(10)) + "\n", encoding="utf-8"
    )
    resumed = plan_run(tmp_path, "grasp_envelope", {"resumeFrom": "grasp_envelope_A", "start": "spot"})
    assert resumed["plan"]["units"] == 53
    assert any(a.startswith("--resume-rows=") for a in resumed["argv"])


def test_a_waiting_run_shows_on_the_page_and_the_page_can_answer_it(tmp_path):
    import os

    run_dir = runs_root(tmp_path) / "grasp_envelope_B"
    run_dir.mkdir(parents=True)
    (run_dir / "plan.json").write_text(json.dumps({"kind": "grasp_envelope", "units": 3}))
    (run_dir / "run.json").write_text(json.dumps({"id": run_dir.name, "kind": "grasp_envelope", "pid": os.getpid()}))
    (run_dir / "rows.jsonl").write_text(
        json.dumps({"kind": "trial", "index": 0}) + "\n"
        + json.dumps({"kind": "needs_operator", "message": "put it back"}) + "\n"
    )
    assert read_run(tmp_path, run_dir.name)["needsOperator"] == "put it back"
    request_continue(tmp_path, run_dir.name)
    assert (run_dir / "CONTINUE").exists()

    with (run_dir / "rows.jsonl").open("a") as handle:
        handle.write(json.dumps({"kind": "operator", "answer": "continued"}) + "\n")
    assert read_run(tmp_path, run_dir.name)["needsOperator"] == ""
    with pytest.raises(Exception, match="not waiting"):
        request_continue(tmp_path, run_dir.name)


def test_a_fine_scan_can_run_its_extra_points_alone(tmp_path):
    planned = plan_run(
        tmp_path,
        "grasp_envelope",
        {"xyOffsetsMm": "", "dzOffsetsMm": "", "centreRepeats": "0", "extraPointsMm": "5,0,-6; 6,0,-6", "start": "spot"},
    )
    assert planned["plan"]["units"] == 2
    assert {p["block"] for p in planned["plan"]["schedule"]} == {"extra"}
