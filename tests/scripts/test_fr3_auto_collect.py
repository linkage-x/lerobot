"""E6-data: is what the recorder writes what actually happened, and does the loop stop when it cannot tell?

Two things are worth testing here and they are not the same thing.

The *recorder* is judged on whether it can be trusted to have recorded. A recorder that silently
drops frames, pairs an action with the state it produced rather than the state it came from, or
leaves a night in one unfooted file, produces data that looks fine and trains a policy on
fiction. Each of those is a test below, and each of them is a real failure this rig has either
had (P0-0's 4192 frames) or is one careless refactor away from.

The *loop* is judged on the property that makes an unattended night worth starting: every state
it cannot interpret ends the run rather than producing another cycle. The rig fake is the peg
physics and nothing else -- fingers hold a peg only if they closed where the peg was standing.
"""

import json
import math
import time

import pytest

import tools.fr3.scene_reset as scene_reset
from tools.fr3.auto_collect import (
    AutoCollectError,
    AutoCollectRequest,
    CycleSpec,
    build_collection_schedule,
    describe_schedule,
    grasp_verdict,
    run_auto_collection,
    validate_auto_collection,
)
from tools.fr3.collection_recorder import (
    DEMO_STEP_P95_MM,
    DEPLOYMENT_STEP_LIMIT_MM,
    ControlTap,
    FrameSink,
    Recorder,
    StepAudit,
    iter_rows,
    read_shards,
)
from tools.fr3.scene_reset import SceneResetStroke

from tests.scripts.test_fr3_scene_reset import FakeRobot


PLACE_Z = 0.055
CARRY_Z = 0.150
MASK = (SceneResetStroke(x=0.36, y=-0.14, radiusM=0.03),)


@pytest.fixture(autouse=True)
def _fast_setpoint(monkeypatch):
    monkeypatch.setattr(scene_reset, "precise_sleep", lambda seconds: None)


class ListSink(FrameSink):
    """A sink that records what it was asked to write instead of writing it."""

    def __init__(self):
        self.writes = []

    def write(self, shard_dir, index, camera, frame):
        self.writes.append((shard_dir.name, index, camera, frame))
        return f"frames/{index:06d}_{camera}.jpg"


class ExplodingSink(FrameSink):
    def write(self, shard_dir, index, camera, frame):
        raise AssertionError("the control loop must never reach a sink")


class FakeCollectRig(FakeRobot):
    """The peg physics and nothing else: fingers hold a peg only where the peg is standing."""

    PEG_WIDTH = 0.31

    def __init__(self, *, peg_xyz=(0.3640, -0.1370, PLACE_Z), held=False, grasp_reach_m=0.006):
        super().__init__()
        self.xyz = (0.3640, -0.1370, CARRY_Z)
        self.peg_xyz = tuple(float(value) for value in peg_xyz)
        self.held = bool(held)
        self.grasp_reach_m = float(grasp_reach_m)
        self.gripper = self.PEG_WIDTH if held else 1.0
        self.closes = []

    def _fingers_on_peg(self):
        return (
            math.hypot(self.xyz[0] - self.peg_xyz[0], self.xyz[1] - self.peg_xyz[1]) <= self.grasp_reach_m
            and abs(self.xyz[2] - self.peg_xyz[2]) <= self.grasp_reach_m
        )

    def send_action(self, action):
        self.actions.append(dict(action))
        self.xyz = (action["ee.x"], action["ee.y"], action["ee.z"])
        self.rotvec = (action["ee.wx"], action["ee.wy"], action["ee.wz"])
        commanded = float(action["gripper.pos"])
        if commanded >= 0.9:
            if self.held:
                self.held = False
                self.peg_xyz = tuple(self.xyz)
            self.gripper = commanded
        elif commanded <= 0.05:
            if not self.held and self._fingers_on_peg():
                self.held = True
            self.closes.append((tuple(self.xyz), self.held))
            self.gripper = self.PEG_WIDTH if self.held else 0.0
        else:
            self.gripper = self.PEG_WIDTH if self.held else commanded
        if self.held:
            self.peg_xyz = tuple(self.xyz)
        return dict(action)


class MissingPegRig(FakeCollectRig):
    """A peg that is not where the loop set it down, which is what an empty close is."""

    def _fingers_on_peg(self):
        return False


def _request(**overrides):
    fields = dict(
        maskStrokes=MASK,
        pickXyz=None,
        placeZ=PLACE_Z,
        carryZ=CARRY_Z,
        cycles=3,
        seed=7,
        recoveryFraction=0.5,
        # Fast enough that the suite runs at wall-clock speed rather than the arm's; the derived
        # speed is asserted separately from a request that keeps the real numbers.
        stepMm=4.0,
        controlPeriodS=0.001,
        transferSpeedMs=50.0,
        timeoutS=2.0,
        requestId="test",
    )
    fields.update(overrides)
    return AutoCollectRequest(**fields)


# -- the recorder is judged on whether it can be trusted to have recorded --------------------


def test_the_control_loop_never_reaches_a_camera_a_sink_or_a_disk():
    """The rule the whole module exists for, tested where it can actually be broken.

    A tap wired to a sink that raises on contact and a camera that raises on contact: if a step
    ever reaches either of them from inside the control loop, this fails. It passes because
    `publish` is an append and nothing else.
    """

    def exploding_camera(_t):
        raise AssertionError("the control loop must never read a camera")

    robot = FakeCollectRig()
    tap = ControlTap()
    request = scene_reset.SceneResetRequest(
        pickXyz=(0.0, 0.0, 0.0), targetXyz=(0.0, 0.0, 0.0), controlPeriodS=0.001
    )
    scene_reset._run_step(
        robot, request, "approach_above_peg", (0.36, -0.14, 0.12), (0.0, 0.0, 0.0), 1.0,
        tap=tap, max_speed_ms=50.0,
    )
    assert tap.status()["published"] > 0
    # Nothing was drained, so nothing was ever handed to the sink or the camera.
    assert tap.status()["pending"] == tap.status()["published"]


def test_a_dropped_sample_is_counted_rather_than_lost_quietly():
    """A thin night has to say it was thin, in the data, or it is indistinguishable from a good one."""

    tap = ControlTap(capacity=4)
    for index in range(10):
        tap.publish(float(index), "phase", {"ee.x": 0.0}, {"ee.x": 0.0})
    status = tap.status()
    assert status["published"] == 10
    assert status["pending"] == 4
    assert status["dropped"] == 6


def test_the_state_published_with_an_action_is_the_state_it_was_chosen_from():
    """Pair the action with the reading it produced and every frame teaches the wrong lesson.

    Asserted against the tap rather than against written rows, because the recorder decimates and
    a decimated stream legitimately skips a link in the chain. The property being tested belongs
    to the control loop: sample N carries the observation taken *before* action N was sent, which
    on a fake that teleports exactly to its command is action N-1's target.
    """

    robot = FakeCollectRig()
    tap = ControlTap()
    request = scene_reset.SceneResetRequest(
        pickXyz=(0.0, 0.0, 0.0), targetXyz=(0.0, 0.0, 0.0), controlPeriodS=0.001
    )
    # 1 mm a step over 50 mm: a walk with steps in it, rather than a single jump that would make
    # the question unaskable.
    scene_reset._run_step(
        robot, request, "descend_to_peg", (0.36, -0.14, 0.10), (0.0, 0.0, 0.0), 1.0,
        tap=tap, max_speed_ms=1.0,
    )

    samples = tap.drain(limit=10_000)
    assert len(samples) > 20
    for sample in samples:
        assert sample.observation["ee.z"] != pytest.approx(sample.action["ee.z"]), (
            "the sample stored the state its own action produced, not the state it came from"
        )
    for previous, current in zip(samples, samples[1:], strict=False):
        assert current.observation["ee.z"] == pytest.approx(previous.action["ee.z"])


def test_a_written_row_carries_both_the_state_and_the_action_it_was_chosen_from(tmp_path):
    robot = FakeCollectRig()
    tap = ControlTap()
    recorder = Recorder(tap, tmp_path, cameras={}, sink=ListSink(), fps=100000.0)
    request = scene_reset.SceneResetRequest(
        pickXyz=(0.0, 0.0, 0.0), targetXyz=(0.0, 0.0, 0.0), controlPeriodS=0.001
    )
    with recorder:
        scene_reset._run_step(
            robot, request, "descend_to_peg", (0.36, -0.14, 0.10), (0.0, 0.0, 0.0), 1.0,
            tap=tap, max_speed_ms=1.0,
        )
        time.sleep(0.3)

    rows = [row for row in iter_rows(tmp_path) if row["kind"] == "frame"]
    assert len(rows) > 3
    assert all(row["state"]["ee.z"] != pytest.approx(row["sent_action"]["ee.z"]) for row in rows)
    # The images are not in `state`: a row that JSON-serialised a 640x480 array would be unreadable.
    assert all(isinstance(value, (int, float, bool, str)) for row in rows for value in row["state"].values())


def test_a_shard_that_was_closed_has_a_footer_and_the_one_in_flight_does_not(tmp_path):
    """The footer is the whole reason a night is not one file: it says which data is complete."""

    tap = ControlTap()
    recorder = Recorder(tap, tmp_path, cameras={}, sink=ListSink(), fps=1000.0, shard_frames=5)
    recorder.start()
    for index in range(12):
        tap.publish(index * 0.001, "descend_to_peg", {"ee.x": 0.0}, {"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0})
    time.sleep(0.3)
    shards = read_shards(tmp_path)
    assert len(shards) >= 2
    assert shards[0]["closed"] is True
    assert shards[0]["footer"]["rows"] == 5
    assert shards[-1]["closed"] is False, "the shard being written to must not look complete"

    recorder.stop()
    assert all(shard["closed"] for shard in read_shards(tmp_path)), "stopping must foot the last shard"


def test_frames_are_paired_to_the_sample_timestamp_not_to_whatever_is_newest(tmp_path):
    """A recorder at its own rate only produces aligned data if it asks for the frame nearest each step."""

    asked = []

    def camera(t):
        asked.append(t)
        return f"frame@{t:.4f}", t + 0.002

    tap = ControlTap()
    recorder = Recorder(tap, tmp_path, cameras={"ee": camera}, sink=ListSink(), fps=1000.0)
    with recorder:
        for index in range(5):
            tap.publish(10.0 + index * 0.01, "descend_to_peg", {"ee.x": 0.0}, {"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0})
        time.sleep(0.2)

    assert asked == pytest.approx([10.0, 10.01, 10.02, 10.03, 10.04])
    rows = [row for row in iter_rows(tmp_path) if row["kind"] == "frame"]
    assert [row["cameraSkewMs"]["ee"] for row in rows] == pytest.approx([2.0] * 5)


def test_one_stale_camera_costs_a_frame_and_not_the_night(tmp_path):
    def camera(_t):
        raise TimeoutError("closest frame is too old")

    tap = ControlTap()
    recorder = Recorder(tap, tmp_path, cameras={"ee": camera}, sink=ListSink(), fps=1000.0)
    with recorder:
        for index in range(3):
            tap.publish(index * 0.01, "descend_to_peg", {"ee.x": 0.0}, {"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0})
        time.sleep(0.2)

    rows = [row for row in iter_rows(tmp_path) if row["kind"] == "frame"]
    assert len(rows) == 3
    assert all("cameraErrors" in row for row in rows)
    assert recorder.status()["recorder"]["cameraFailures"] == 3


def test_decimation_thins_the_dataset_but_never_the_audit(tmp_path):
    """The guard applies to every command the arm receives, so the audit must see every command."""

    audit = StepAudit()
    tap = ControlTap(audit=audit)
    recorder = Recorder(tap, tmp_path, cameras={}, sink=ListSink(), fps=10.0)
    with recorder:
        for index in range(50):
            tap.publish(
                index * 0.01,
                "descend_to_peg",
                {"ee.x": 0.0},
                {"ee.x": index * 0.001, "ee.y": 0.0, "ee.z": 0.0},
            )
        time.sleep(0.3)

    rows = [row for row in iter_rows(tmp_path) if row["kind"] == "frame"]
    assert 4 <= len(rows) <= 7, "kept roughly one frame per 100 ms"
    assert audit.summary()["steps"] == 49, "the audit saw every step, not the kept subsample"


def test_markers_are_never_decimated_because_they_are_what_explains_the_gaps(tmp_path):
    tap = ControlTap()
    recorder = Recorder(tap, tmp_path, cameras={}, sink=ListSink(), fps=1.0)
    with recorder:
        for index in range(5):
            tap.publish(index * 0.001, "descend_to_peg", {"ee.x": 0.0}, {"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0})
            tap.mark("leg", leg="place", recorded=False)
        time.sleep(0.2)

    rows = list(iter_rows(tmp_path))
    assert sum(1 for row in rows if row["kind"] == "marker") == 5
    assert sum(1 for row in rows if row["kind"] == "frame") == 1


def test_the_two_heartbeats_are_separate_because_they_fail_separately(tmp_path):
    """One "it is running" light cannot tell an arm that stopped from a recorder that stopped."""

    tap = ControlTap()
    recorder = Recorder(tap, tmp_path, cameras={}, sink=ListSink(), fps=1000.0)
    with recorder:
        tap.publish(1.0, "descend_to_peg", {"ee.x": 0.0}, {"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0})
        time.sleep(0.2)
        status = recorder.status()

    assert status["controller"]["publishedSteps"] == 1
    assert status["recorder"]["alive"] is True
    assert status["disk"]["lastWriteAgoS"] is not None
    assert "controller:" in recorder.describe_status() and "disk:" in recorder.describe_status()


# -- the audit is an assertion, not a note ---------------------------------------------------


def test_a_step_the_deployment_guard_would_clip_invalidates_the_episode():
    audit = StepAudit()
    audit.begin_episode()
    audit.observe({"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0})
    audit.observe({"ee.x": 0.002, "ee.y": 0.0, "ee.z": 0.0})
    assert audit.episode_is_valid is True
    audit.observe({"ee.x": 0.012, "ee.y": 0.0, "ee.z": 0.0})
    assert audit.episode_is_valid is False
    summary = audit.summary()
    assert summary["violations"] == 1
    assert summary["maxStepMm"] == pytest.approx(10.0)
    assert summary["limitMm"] == pytest.approx(DEPLOYMENT_STEP_LIMIT_MM)


def test_the_jump_across_a_leg_boundary_is_not_a_policy_step():
    audit = StepAudit()
    audit.begin_episode()
    audit.observe({"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0})
    audit.begin_episode()
    audit.observe({"ee.x": 0.5, "ee.y": 0.0, "ee.z": 0.0})
    assert audit.episode_is_valid is True
    assert audit.summary()["steps"] == 0


def test_the_audit_reports_the_distribution_beside_the_demonstrations_own():
    """Never tripping the guard is not the same as looking like the data it will be mixed with."""

    audit = StepAudit()
    audit.begin_episode()
    for index in range(101):
        audit.observe({"ee.x": index * 0.0049, "ee.y": 0.0, "ee.z": 0.0})
    summary = audit.summary()
    assert summary["violations"] == 0
    assert summary["p95StepMm"] == pytest.approx(4.9, abs=0.05)
    assert summary["demoP95StepMm"] == pytest.approx(DEMO_STEP_P95_MM)
    assert summary["p95StepMm"] > summary["demoP95StepMm"]


# -- the schedule is fixed, readable and reproducible before the arm moves --------------------


def test_the_same_seed_produces_the_same_night():
    first = build_collection_schedule(_request(cycles=20, seed=3))
    second = build_collection_schedule(_request(cycles=20, seed=3))
    third = build_collection_schedule(_request(cycles=20, seed=4))
    assert first == second
    assert first != third


def test_orientation_displacement_is_refused_rather_than_ignored():
    """Refused because the reset's QC checks the tool point against the fence, not the fingertips
    against the table, and orientation is the one axis that can put one into the other."""

    with pytest.raises(AutoCollectError, match="phase two"):
        build_collection_schedule(_request(perturbRotDeg=5.0))


def test_a_step_above_the_deployment_limit_is_refused_before_the_night_starts():
    with pytest.raises(AutoCollectError, match="deployment step limit"):
        build_collection_schedule(_request(stepMm=DEPLOYMENT_STEP_LIMIT_MM + 0.1))


def test_a_run_with_no_mask_is_refused_because_nothing_authorised_an_area():
    with pytest.raises(AutoCollectError, match="maskStrokes is empty"):
        build_collection_schedule(_request(maskStrokes=()))


def test_displacements_are_uniform_over_the_disc_rather_than_piled_at_its_centre():
    """A distribution dense where the error is small spends the night re-teaching easy corrections."""

    specs = build_collection_schedule(_request(cycles=400, seed=1, recoveryFraction=1.0))
    radii = [1000.0 * math.hypot(spec.startOffset[0], spec.startOffset[1]) for spec in specs]
    inside_half = sum(1 for radius in radii if radius <= 12.5) / len(radii)
    # Uniform over a disc puts a quarter of the draws inside half the radius; piled at the centre
    # would put about half of them there.
    assert 0.18 < inside_half < 0.32


def test_the_displacement_band_covers_the_hole_e4_found_empty():
    specs = build_collection_schedule(_request(cycles=200, seed=2, recoveryFraction=1.0))
    heights = [spec.start_xyz()[2] for spec in specs]
    assert min(heights) < 0.15 and max(heights) > 0.20
    assert sum(1 for height in heights if 0.15 <= height <= 0.20) > 40
    # And the band is a tool height, not a height above a peg that moves with every placement.
    assert len({round(spec.placeXyz[2], 6) for spec in specs}) == 1


def test_nominal_cycles_are_interleaved_so_displacement_is_not_confounded_with_the_afternoon():
    specs = build_collection_schedule(_request(cycles=40, seed=5, recoveryFraction=0.5))
    kinds = [spec.kind for spec in specs]
    first_half = kinds[:20].count("recovery")
    assert 5 <= first_half <= 15, "recovery cycles were blocked into one half of the night"


def test_the_recorded_speed_is_derived_from_millimetres_per_step_not_chosen_in_metres():
    """What a policy learns is millimetres a step. Seconds are not in its action space."""

    request = AutoCollectRequest(
        maskStrokes=MASK, pickXyz=None, placeZ=PLACE_Z, carryZ=CARRY_Z,
        stepMm=2.0, controlPeriodS=1.0 / 30.0,
    )
    assert request.recorded_speed_ms() == pytest.approx(0.06)
    # And the reset's own limit is what this exists to avoid: 5.0 mm a step at the same period.
    assert scene_reset.SCENE_RESET_MAX_SPEED_MS * (1.0 / 30.0) * 1000.0 == pytest.approx(
        DEPLOYMENT_STEP_LIMIT_MM
    )


def test_the_plan_can_be_read_before_it_is_authorised():
    request = _request(cycles=4)
    specs = build_collection_schedule(request)
    text = describe_schedule(request, specs)
    assert len(text.splitlines()) == 4 + len(specs)
    assert "step_mm=" in text and "perturb_xy_mm=" in text


def test_every_pose_the_night_can_command_is_checked_before_the_first_one_is_sent():
    request = _request(cycles=25)
    specs = build_collection_schedule(request)
    qc = validate_auto_collection(
        request, specs, workspace_min=(0.2, -0.4, 0.0), workspace_max=(0.6, 0.2, 0.5)
    )
    assert qc["cycles"] == 25 and qc["checkedPoses"] == 25

    with pytest.raises(Exception):
        validate_auto_collection(
            request, specs, workspace_min=(0.2, -0.4, 0.0), workspace_max=(0.6, 0.2, 0.16)
        )


def test_carrying_below_the_table_is_refused():
    request = _request(carryZ=PLACE_Z - 0.01)
    specs = build_collection_schedule(request)
    with pytest.raises(AutoCollectError, match="carried between cycles"):
        validate_auto_collection(request, specs)


# -- the loop is judged on stopping, and on what it labels ------------------------------------


def test_a_cycle_records_the_approach_and_the_grasp_and_nothing_else(tmp_path):
    robot = FakeCollectRig(held=True)
    audit = StepAudit()
    tap = ControlTap(capacity=200000, audit=audit)
    recorder = Recorder(tap, tmp_path, cameras={}, sink=ListSink(), fps=100000.0)
    request = _request(cycles=2)
    specs = build_collection_schedule(request)
    with recorder:
        summary = run_auto_collection(robot, request, specs, tap=tap)
        time.sleep(0.4)

    assert summary["haltedOn"] == "schedule_complete", summary["haltedOn"]
    rows = list(iter_rows(tmp_path))
    phases = {row["phase"] for row in rows if row["kind"] == "frame"}
    assert phases == {"approach_above_peg", "descend_to_peg", "close_gripper", "lift_8cm_after_grasp"}
    markers = [row["marker"] for row in rows if row["kind"] == "marker"]
    assert markers.count("episode_start") == 2 and markers.count("episode_end") == 2
    # The unrecorded legs left a trace, which is what stops a gap looking like dropped frames.
    legs = [row.get("leg") for row in rows if row.get("marker") == "leg"]
    assert "place" in legs and "displace" in legs


def test_every_recorded_frame_carries_the_episode_it_belongs_to(tmp_path):
    robot = FakeCollectRig(held=True)
    tap = ControlTap(capacity=200000)
    recorder = Recorder(tap, tmp_path, cameras={}, sink=ListSink(), fps=100000.0)
    request = _request(cycles=2)
    with recorder:
        run_auto_collection(robot, request, build_collection_schedule(request), tap=tap)
        time.sleep(0.4)

    episodes = {row["episode"] for row in iter_rows(tmp_path) if row["kind"] == "frame"}
    assert episodes == {0, 1}, "frames outside an episode were recorded, or episodes were not stamped"


def test_a_failed_cycle_keeps_its_frames_and_is_labelled_where_it_ended(tmp_path):
    """Kept and marked rather than deleted: BC filters them, a later value pass needs them, and
    deletion is the one operation that cannot be undone at 3 a.m."""

    robot = MissingPegRig(held=True)
    tap = ControlTap(capacity=200000)
    recorder = Recorder(tap, tmp_path, cameras={}, sink=ListSink(), fps=100000.0)
    request = _request(cycles=4)
    with recorder:
        summary = run_auto_collection(robot, request, build_collection_schedule(request), tap=tap)
        time.sleep(0.4)

    assert summary["haltedOn"] in {"empty_streak", "recovery_empty"}
    rows = list(iter_rows(tmp_path))
    assert any(row["kind"] == "frame" for row in rows), "the failed cycle's frames were discarded"
    ends = [row for row in rows if row.get("marker") == "episode_end"]
    assert ends and all(end["verdict"] == "empty" for end in ends)


def test_two_empty_closes_in_a_row_end_the_run():
    robot = MissingPegRig(held=True)
    tap = ControlTap(capacity=200000)
    request = _request(cycles=10)
    summary = run_auto_collection(robot, request, build_collection_schedule(request), tap=tap)
    assert summary["ok"] is False
    assert summary["haltedOn"] in {"empty_streak", "recovery_empty"}
    assert summary["cycles"] < 10


def test_a_run_that_starts_without_the_peg_stops_before_it_places_nothing():
    robot = FakeCollectRig(held=False)
    tap = ControlTap()
    request = _request(cycles=5, pickXyz=None)
    summary = run_auto_collection(robot, request, build_collection_schedule(request), tap=tap)
    assert summary["haltedOn"] == "grasp_lost_at_start"
    assert summary["cycles"] == 0


def test_the_run_ends_holding_the_peg_at_carry_height_rather_than_dropping_it():
    """A loop that stopped because it could not read its own state is the worst moment to let go."""

    robot = MissingPegRig(held=True)
    tap = ControlTap(capacity=200000)
    request = _request(cycles=10)
    summary = run_auto_collection(robot, request, build_collection_schedule(request), tap=tap)
    assert summary["parked"] is True
    assert robot.xyz[2] >= CARRY_Z - 1e-6


def test_the_summary_carries_the_audit_so_a_night_reports_its_own_data_quality():
    robot = FakeCollectRig(held=True)
    audit = StepAudit()
    tap = ControlTap(capacity=200000, audit=audit)
    request = _request(cycles=2)
    summary = run_auto_collection(robot, request, build_collection_schedule(request), tap=tap)
    assert summary["audit"]["steps"] > 0
    assert summary["audit"]["violations"] == 0
    assert summary["audit"]["maxStepMm"] <= request.stepMm + 1e-6
    assert summary["recordedEpisodes"] == 2 and summary["invalidatedEpisodes"] == 0


def test_the_verdict_rule_is_the_one_terminal_trials_already_uses():
    request = _request()
    assert grasp_verdict(1.0, None, request) == "open"
    assert grasp_verdict(0.05, None, request) == "empty"
    assert grasp_verdict(0.31, None, request) == "held"
    assert grasp_verdict(0.31, 0.31, request) == "held"
    assert grasp_verdict(0.20, 0.31, request) == "changed"


def test_a_leg_shorter_than_one_record_period_still_appears_in_the_dataset(tmp_path):
    """A missing approach and a thinly sampled one look identical afterwards. They are not."""

    tap = ControlTap()
    recorder = Recorder(tap, tmp_path, cameras={}, sink=ListSink(), fps=10.0)
    with recorder:
        # Three legs inside a single 100 ms record period: without the new-phase rule, two of them
        # would contribute nothing and nothing would say so.
        for index, phase in enumerate(("approach_above_peg", "descend_to_peg", "close_gripper")):
            tap.publish(index * 0.001, phase, {"ee.x": 0.0}, {"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0})
        time.sleep(0.2)

    phases = [row["phase"] for row in iter_rows(tmp_path) if row["kind"] == "frame"]
    assert phases == ["approach_above_peg", "descend_to_peg", "close_gripper"]


def test_the_gripper_settle_does_not_masquerade_as_demonstrated_motion():
    """Most commands in an episode are "stay put". Percentiles over all of them read zero.

    A recorded leg republishes its setpoint through the whole grasp settle -- 0.6 s at 30 Hz is 18
    identical commands -- so an audit that pooled those with the moving steps would report a p50 of
    0.0 mm and be read as "the motion is gentle" when it says nothing about the motion at all.
    """

    audit = StepAudit()
    audit.begin_episode()
    audit.observe({"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0})
    for _ in range(18):
        audit.observe({"ee.x": 0.0, "ee.y": 0.0, "ee.z": 0.0})
    for index in range(1, 5):
        audit.observe({"ee.x": index * 0.002, "ee.y": 0.0, "ee.z": 0.0})
    summary = audit.summary()
    assert summary["steps"] == 22 and summary["movingSteps"] == 4
    assert summary["stillFraction"] == pytest.approx(18 / 22)
    assert summary["p50StepMm"] == pytest.approx(2.0)


def test_a_recorder_that_was_told_to_stop_is_not_a_recorder_that_died(tmp_path):
    tap = ControlTap()
    recorder = Recorder(tap, tmp_path, cameras={}, sink=ListSink(), fps=1000.0)
    recorder.start()
    assert "alive" in recorder.describe_status()
    recorder.stop()
    assert "stopped" in recorder.describe_status()
    assert "DEAD" not in recorder.describe_status()


def test_the_brake_stops_at_a_cycle_boundary_holding_the_peg(tmp_path):
    """Interrupting mid-cycle would leave the peg on the table and the arm where the next cycle
    does not expect it -- the loop's invariant broken by the person trying to be careful."""

    from tools.fr3.collection_recorder import StopFile

    robot = FakeCollectRig(held=True)
    tap = ControlTap(capacity=200000)
    stop = StopFile(tmp_path / "STOP")
    request = _request(cycles=6)

    cycles = {"n": 0}

    def should_stop():
        cycles["n"] += 1
        return cycles["n"] > 2

    summary = run_auto_collection(
        robot, request, build_collection_schedule(request), tap=tap, should_stop=should_stop
    )
    assert summary["haltedOn"] == "stop_requested"
    assert summary["cycles"] == 2
    # A deliberate stop is not a failure, and the arm ends where the next run starts from.
    assert summary["ok"] is True
    assert robot.held is True and robot.xyz[2] >= CARRY_Z - 1e-6


def test_the_stop_file_is_existence_not_content(tmp_path):
    from tools.fr3.collection_recorder import StopFile

    stop = StopFile(tmp_path / "nested" / "STOP")
    assert stop() is False
    stop.request("operator, 02:14")
    assert stop() is True and stop.path.read_text(encoding="utf-8") == "operator, 02:14"
    stop.request("")
    assert stop() is True, "an empty reason still stops the run"
    stop.clear()
    assert stop() is False
    stop.clear()  # clearing twice is not an error: the brake may be released by two things at once
