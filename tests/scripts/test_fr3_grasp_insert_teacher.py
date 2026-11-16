"""The insertion teacher: does it put an off-centre peg in, and does it never press on a miss?

The arm here follows each setpoint exactly, down to a floor: the hole's bottom when the peg is
over the hole, the fixture's face when it is not. That is all the teacher reads -- where the tool
stands, and the finger width -- so it is all the teacher can be tested against.
"""

import math

import pytest

from tools.fr3.grasp_insert_teacher import (
    CARRY,
    DESCEND,
    DONE,
    LIFT,
    RELEASE,
    RETREAT,
    RETRY,
    InsertTeacher,
    InsertTeacherConfig,
    in_hand_offset_xy,
)

HOLE = (0.3597, -0.1328, 0.058)
REGRIP_Z = 0.052
PEG = (0.45, -0.15, 0.058)
HELD_WIDTH = 0.31
EMPTY_WIDTH = 0.006
FACE_ABOVE_M = 0.035
CAPTURE_M = 0.0012


def _held_test(width, gripper):
    return width >= 0.025 and width - gripper >= 0.05


def _config(close_offset_x=0.0, close_dz=0.0, **overrides):
    fields = dict(
        closeXyz=(PEG[0] + close_offset_x, PEG[1], REGRIP_Z + close_dz),
        closeRotvec=None,
        pegXyz=PEG,
        holeXyz=HOLE,
        regripZ=REGRIP_Z,
        carryZ=0.12,
        levelRotvec=(0.0, 0.0, 0.0),
        policyGripper=0.2,
        controlPeriodS=1.0 / 30.0,
    )
    fields.update(overrides)
    return InsertTeacherConfig(**fields)


def _run(config, *, width=HELD_WIDTH, peg_hole_xy=None, max_steps=3000, lag_xy=(0.0, 0.0), edge_push_m=0.0):
    """Drive the teacher on an arm that tracks exactly -- or stands `lag_xy` off every setpoint,
    as the FR3 does sideways -- and the peg is off the tool by the offset. With `edge_push_m`, a
    peg standing on the face within 5 mm of the hole has the tool pushed that far toward the hole
    and stays on the face (092701: blocked pegs skated 6-7 mm along the hub's top)."""

    teacher = InsertTeacher(config, heldTest=_held_test)
    offset = config.offsetXy
    hole_xy = HOLE[:2] if peg_hole_xy is None else peg_hole_xy
    seat_z = config.seatXyz[2]
    xyz, rotvec = config.closeXyz, (0.0, 0.0, 0.0)
    commands = []
    for step in range(max_steps):
        if teacher.done:
            break
        command = teacher.step(step, xyz, rotvec, width)
        commands.append(command)
        tool_xy = (command["ee.x"] + lag_xy[0], command["ee.y"] + lag_xy[1])
        peg_xy = (tool_xy[0] - offset[0], tool_xy[1] - offset[1])
        over_hole = math.dist(peg_xy, hole_xy) <= CAPTURE_M
        floor = seat_z - 0.010 if over_hole else seat_z + FACE_ABOVE_M
        # Only a peg arriving from above meets the face.
        z = command["ee.z"] if xyz[2] < floor - 1e-9 and not over_hole else max(command["ee.z"], floor)
        miss = (hole_xy[0] - peg_xy[0], hole_xy[1] - peg_xy[1])
        if edge_push_m and not over_hole and z <= floor + 1e-9 and math.hypot(*miss) < 0.005:
            norm = math.hypot(*miss)
            tool_xy = (tool_xy[0] + edge_push_m * miss[0] / norm, tool_xy[1] + edge_push_m * miss[1] / norm)
        xyz = (tool_xy[0], tool_xy[1], z)
        rotvec = (command["ee.wx"], command["ee.wy"], command["ee.wz"])
    return teacher, commands


def test_the_in_hand_offset_keeps_only_the_axis_along_the_finger_face():
    close, peg = (0.405, -0.147, 0.06), (0.40, -0.15, 0.058)
    assert in_hand_offset_xy(close, peg) == pytest.approx((0.005, 0.0))
    # Wrist turned 90 deg about z: the fingers now close along base x, so y is what stays.
    assert in_hand_offset_xy(close, peg, (0.0, 0.0, math.pi / 2)) == pytest.approx((0.0, 0.003), abs=1e-9)


@pytest.mark.parametrize("offset_x", [-0.008, -0.002, 0.0, 0.004, 0.008])
def test_an_off_centre_peg_is_seated_by_aiming_the_tool_at_the_hole_plus_the_offset(offset_x):
    config = _config(close_offset_x=offset_x)
    teacher, commands = _run(config)
    assert teacher.outcome == "seated"
    assert [t["to"] for t in teacher.transitions] == [CARRY, DESCEND, RELEASE, RETREAT, DONE]
    assert teacher.stoppedXyz[0] == pytest.approx(HOLE[0] + offset_x, abs=1e-6)
    assert teacher.stoppedXyz[1] == pytest.approx(HOLE[1], abs=1e-6)
    record = teacher.record()
    assert record["offsetXyMm"] == pytest.approx([offset_x * 1000.0, 0.0], abs=0.05)
    assert record["held"] is True and record["aboveSeatMm"] <= 3.0


def test_the_grip_is_the_policys_until_graded_then_the_scripts_and_opens_only_once_seated():
    teacher, commands = _run(_config(close_offset_x=0.003))
    grippers = [c["gripper.pos"] for c in commands]
    first_squeeze = grippers.index(0.0)
    first_open = grippers.index(1.0)
    assert set(grippers[:first_squeeze]) == {0.2}
    assert set(grippers[first_squeeze:first_open]) == {0.0}
    # Graded with the tool at least checkLiftM above the close.
    assert commands[first_squeeze - 1]["ee.z"] >= REGRIP_Z + 0.03 - 1e-9
    # Opened at the bottom, and up only after it opened.
    assert commands[first_open]["ee.z"] == pytest.approx(commands[first_open - 1]["ee.z"])
    assert commands[-1]["ee.z"] > commands[first_open]["ee.z"] + 0.04
    assert set(grippers[first_open:]) == {1.0}


def test_nothing_crosses_sideways_until_the_peg_is_at_carry_height():
    config = _config(close_offset_x=0.002)
    _teacher, commands = _run(config)
    close_xy, seat_xy = config.closeXyz[:2], config.seatXyz[:2]
    for command in commands:
        if command["ee.z"] < 0.12 - 1e-9:
            # Below carry height the tool is straight over where it closed, or over the seat.
            xy = (command["ee.x"], command["ee.y"])
            assert min(math.dist(xy, close_xy), math.dist(xy, seat_xy)) < 1e-9


def test_an_empty_grasp_is_graded_on_the_lift_and_never_squeezed_or_carried():
    teacher, commands = _run(_config(), width=EMPTY_WIDTH)
    assert teacher.outcome == "empty" and teacher.held is False
    assert teacher.state == DONE
    assert {c["gripper.pos"] for c in commands} == {0.2}
    assert all(math.isclose(c["ee.x"], PEG[0]) for c in commands)


def test_a_peg_that_lands_on_the_face_is_held_where_it_stands_and_never_let_go():
    # The hole is 3 mm from where the teacher believes and nothing pushes: the peg lands on the
    # face, and with no push to go by there is no retry either.
    teacher, commands = _run(_config(close_offset_x=0.004), peg_hole_xy=(HOLE[0] + 0.003, HOLE[1]))
    assert teacher.record()["retries"] == 0
    assert teacher.outcome == "blocked"
    assert all(c["gripper.pos"] < 0.5 for c in commands)
    face = _config(close_offset_x=0.004).seatXyz[2] + FACE_ABOVE_M
    # The setpoint never ran more than the contact lag below the tool standing on the face.
    top = max(range(len(commands)), key=lambda i: commands[i]["ee.z"])
    assert min(c["ee.z"] for c in commands[top:]) >= face - 0.004 - 1e-9
    assert teacher.record()["aboveSeatMm"] == pytest.approx(FACE_ABOVE_M * 1000.0, abs=0.1)


def test_a_grip_higher_up_the_peg_seats_that_much_higher():
    config = _config(close_dz=0.006)
    assert config.seatXyz[2] == pytest.approx(HOLE[2] + 0.006)
    teacher, _commands = _run(config)
    assert teacher.outcome == "seated"
    # A lower grip keeps the hole's height: the peg bottoms out on the floor first.
    assert _config(close_dz=-0.004).seatXyz[2] == pytest.approx(HOLE[2])


def test_the_wrist_is_turned_level_on_the_carry_and_not_before():
    tilted = (0.0, 0.0, 0.06)
    config = _config(levelRotvec=(0.0, 0.0, 0.0))
    teacher = InsertTeacher(config, heldTest=_held_test)
    xyz = config.closeXyz
    rotvecs = []
    for step in range(3000):
        if teacher.done:
            break
        command = teacher.step(step, xyz, tilted if step == 0 else rotvecs[-1], HELD_WIDTH)
        rotvecs.append((command["ee.wx"], command["ee.wy"], command["ee.wz"]))
        if teacher.state == LIFT:
            assert rotvecs[-1] == pytest.approx(tilted)
        xyz = (command["ee.x"], command["ee.y"], command["ee.z"])
    assert teacher.outcome == "seated"
    assert rotvecs[-1] == pytest.approx((0.0, 0.0, 0.0), abs=1e-9)
    # Turned gradually: no step turns more than a fraction of the whole.
    steps = [abs(b[2] - a[2]) for a, b in zip(rotvecs, rotvecs[1:])]
    assert max(steps) < 0.01


@pytest.mark.parametrize("lag_xy", [(-0.003, 0.001), (0.002, -0.002), (-0.0045, 0.0)])
def test_an_arm_that_stands_off_its_setpoint_is_steered_onto_the_seat_before_the_peg_reaches_the_hub(lag_xy):
    # 11-13 pilot: the FR3 stood ~2-3 mm -x of the teacher's setpoint, and 3 of 10 pegs landed on
    # the hub's face. The teacher now servos on where the tool stands, above the hub only.
    config = _config(close_offset_x=-0.001)
    teacher, commands = _run(config, lag_xy=lag_xy)
    assert teacher.outcome == "seated"
    seat = config.seatXyz
    assert math.dist(teacher.stoppedXyz[:2], seat[:2]) < 0.0007
    record = teacher.record()
    assert record["alignErrorAtEntryMm"] <= 0.6
    # Where it entered, measured: the next trial's hole estimate learns from this, not the seat.
    assert math.dist(record["entryXyz"][:2], seat[:2]) <= 0.0006 + 1e-6
    assert record["entryXyz"][2] <= seat[2] + config.entryDzM + 0.002
    assert record["alignBiasMm"] == pytest.approx([-v * 1000.0 for v in lag_xy], abs=0.7)


def test_without_the_servo_the_same_lag_puts_the_peg_on_the_face():
    config = _config(close_offset_x=-0.001, alignGain=0.0, alignTimeoutSteps=0)
    teacher, _commands = _run(config, lag_xy=(-0.003, 0.001))
    assert teacher.outcome == "blocked"


@pytest.mark.parametrize("miss_mm,retries", [(1.6, 1), (2.6, 2), (-1.6, 1)])
def test_a_peg_pushed_aside_by_the_hub_is_lifted_and_tried_again_a_millimetre_that_way(miss_mm, retries):
    # 092701: the hole stood ~1 mm from where the teacher believed, and the hub pushed the tool
    # toward it on touching. Each retry moves the aim 1 mm along the push.
    config = _config(close_offset_x=0.003, retries=3)
    teacher, commands = _run(config, peg_hole_xy=(HOLE[0] + miss_mm / 1000.0, HOLE[1]), edge_push_m=0.006)
    assert teacher.outcome == "seated"
    record = teacher.record()
    assert record["retries"] == retries
    assert [t["to"] for t in teacher.transitions] == (
        [CARRY, DESCEND] + [RETRY, DESCEND] * retries + [RELEASE, RETREAT, DONE]
    )
    assert record["aimNudgeMm"] == pytest.approx([math.copysign(retries, miss_mm), 0.0], abs=0.01)
    assert all(c["gripper.pos"] < 0.5 for c in commands[: next(i for i, c in enumerate(commands) if c["gripper.pos"] > 0.5)])
    # Lifted straight off the face before any sideways move, and never higher than the lift.
    for entry in record["retryLog"]:
        assert entry["aboveSeatMm"] == pytest.approx(FACE_ABOVE_M * 1000.0, abs=0.1)
    face = config.seatXyz[2] + FACE_ABOVE_M
    lowest_after_first_block = min(
        c["ee.z"] for c in commands[record["retryLog"][0]["step"]:] if c["ee.z"] > face - 0.0045
    )
    assert lowest_after_first_block >= face - 0.004 - 1e-9


@pytest.mark.parametrize("lag_xy", [(0.0, 0.0), (-0.0025, 0.001)])
def test_a_retry_lifted_to_the_entry_height_keeps_its_move_and_seats(lag_xy):
    # 3.9 mm off: three 1 mm moves leave 0.9 mm, inside this model's 1.2 mm capture. The lift
    # puts the tool back at the servo's height, which must steer onto the moved aim, not the seat.
    config = _config(close_offset_x=0.003, retries=3)
    teacher, _commands = _run(config, peg_hole_xy=(HOLE[0] + 0.0039, HOLE[1]), edge_push_m=0.006, lag_xy=lag_xy)
    assert teacher.outcome == "seated"
    assert teacher.record()["retries"] == 3
    assert teacher.stoppedXyz[0] == pytest.approx(config.seatXyz[0] + 0.003, abs=0.0007)


def test_retries_stop_at_three_and_hand_the_block_on():
    config = _config(close_offset_x=0.003, retries=3)
    teacher, _commands = _run(config, peg_hole_xy=(HOLE[0] + 0.0045, HOLE[1]), edge_push_m=0.006)
    record = teacher.record()
    assert record["retries"] == 3
    assert teacher.outcome == "blocked"
    assert teacher.transitions[-1]["reason"] == "stood_above_seat"


def test_retries_are_off_by_default_and_a_pushed_block_is_handed_on_at_once():
    # 11-16 101911: 0 of 6 retries seated, and each press drove the peg up the fingers.
    config = _config(close_offset_x=0.003)
    teacher, _commands = _run(config, peg_hole_xy=(HOLE[0] + 0.0016, HOLE[1]), edge_push_m=0.006)
    assert teacher.outcome == "blocked" and teacher.record()["retries"] == 0
