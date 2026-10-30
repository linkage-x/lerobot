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


def _run(config, *, width=HELD_WIDTH, peg_hole_xy=None, max_steps=3000):
    """Drive the teacher on an arm that tracks exactly; the peg is off the tool by the offset."""

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
        peg_xy = (command["ee.x"] - offset[0], command["ee.y"] - offset[1])
        over_hole = math.dist(peg_xy, hole_xy) <= CAPTURE_M
        floor = seat_z - 0.010 if over_hole else seat_z + FACE_ABOVE_M
        # Only a peg arriving from above meets the face.
        z = command["ee.z"] if xyz[2] < floor - 1e-9 and not over_hole else max(command["ee.z"], floor)
        xyz = (command["ee.x"], command["ee.y"], z)
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
    # The hole is 3 mm from where the teacher believes: the peg lands on the face.
    teacher, commands = _run(_config(close_offset_x=0.004), peg_hole_xy=(HOLE[0] + 0.003, HOLE[1]))
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
