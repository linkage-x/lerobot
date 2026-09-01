"""The line a running rollout publishes so a browser can draw the arm while it moves.

Both ends of this format are machine-written -- the runtime prints it, the gateway reads it --
so the contract worth testing is that they agree, and that neither can hurt the rollout. A frame
is an observation of a run in progress; nothing here may stop one.
"""

from __future__ import annotations

import json

import pytest

from tools.data_collection_gui import rollout as rollout_backend
from tools.fr3.live_frames import (
    LiveFrameEmitter,
    build_live_frame,
    format_live_frame,
    parse_live_frame,
)


def test_a_frame_survives_the_round_trip_the_gateway_actually_performs():
    # Not `json.loads` on the same dict: the gateway reads these off a log file through the
    # rollout backend, and that is the path that has to agree with the emitter.
    line = format_live_frame(
        build_live_frame(
            frame_index=12,
            joints_rad=[0.1, -0.2, 0.3, -1.4, 0.5, 1.6, -0.7],
            gripper=0.42,
            source="expert",
            status="step_limited",
            rollout_index=3,
            target_position_m=[0.40, -0.10, 0.07],
            actual_position_m=[0.39, -0.10, 0.06],
        )
    )

    parsed = rollout_backend.parse_live_frame(line)

    assert parsed == {
        "frame_index": 12,
        "joints_rad": [0.1, -0.2, 0.3, -1.4, 0.5, 1.6, -0.7],
        "gripper": 0.42,
        "source": "expert",
        "status": "step_limited",
        "rollout_index": 3,
        "target_position_m": [0.4, -0.1, 0.07],
        "actual_position_m": [0.39, -0.1, 0.06],
    }


def test_a_frame_is_one_line():
    """It is read back by a follower that splits on newlines, and written into a log file
    interleaved with everything else the runtime prints."""
    line = format_live_frame(build_live_frame(frame_index=0, joints_rad=[0.0] * 7, gripper=1.0))

    assert "\n" not in line and "\r" not in line


def test_float64_tails_are_not_carried():
    """Rounding is not cosmetic here: this line is printed thirty times a second for the length
    of every rollout, and the digits being dropped describe motion far below what the arm can
    resolve -- 1e-5 rad is 6e-4 degrees."""
    frame = build_live_frame(
        frame_index=1,
        joints_rad=[0.1234567890123, 0, 0, 0, 0, 0, 0],
        gripper=1 / 3,
        actual_position_m=[0.123456789, 0.0, 0.0],
    )

    assert frame["joints_rad"][0] == 0.12346
    assert frame["gripper"] == 0.3333
    assert frame["actual_position_m"][0] == 0.12346


def test_an_ordinary_log_line_is_not_a_frame():
    assert parse_live_frame("[INFO] step=41 status=pass raw_ee=(0.4, -0.0, 0.2)") is None
    assert parse_live_frame("") is None


def test_a_half_written_line_costs_that_frame_and_nothing_else():
    """The follower reads the log file while the runtime is writing it, so it can catch a line
    mid-flush. That must not raise inside the reader thread."""
    complete = format_live_frame(build_live_frame(frame_index=5, joints_rad=[0.0] * 7, gripper=1.0))

    assert parse_live_frame(complete[: len(complete) // 2]) is None


def test_a_json_object_that_is_not_a_frame_is_rejected():
    assert parse_live_frame("live_frame=" + json.dumps({"hello": "world"})) is None
    assert parse_live_frame("live_frame=" + json.dumps([1, 2, 3])) is None


def test_the_stream_is_off_unless_it_was_asked_for():
    """A terminal operator watching the arm itself has no use for these, and the log file is the
    same file either way."""
    printed: list[str] = []
    emitter = LiveFrameEmitter(interval=0, sink=printed.append)

    emitter.emit_step(0, joints_rad=[0.0] * 7, gripper=1.0)

    assert emitter.enabled is False
    assert printed == []


def test_the_interval_selects_steps_rather_than_thinning_afterwards():
    printed: list[str] = []
    emitter = LiveFrameEmitter(interval=3, sink=printed.append)

    for step in range(7):
        emitter.emit_step(step, joints_rad=[0.0] * 7, gripper=1.0)

    assert [parse_live_frame(line)["frame_index"] for line in printed] == [0, 3, 6]


def test_a_frame_that_cannot_be_built_does_not_end_the_rollout():
    """The one property that matters more than the picture: this runs in a loop that may be
    holding an object. A joint that reads NaN costs the view, not the run."""
    printed: list[str] = []
    emitter = LiveFrameEmitter(interval=1, sink=printed.append)

    emitter.emit_step(0, joints_rad=[float("nan")] * 7, gripper=1.0)
    emitter.emit_step(1, joints_rad=[0.0] * 7, gripper=1.0)

    # And the next good frame still goes out: one bad reading is not a broken stream.
    assert [parse_live_frame(line)["frame_index"] for line in printed] == [1]


def test_a_non_finite_value_never_reaches_the_page():
    """Python writes NaN as the bare word `NaN`, which `JSON.parse` refuses. A frame carrying one
    would not cost a frame -- it would break the response that carries every frame after it."""
    with pytest.raises(ValueError):
        format_live_frame(build_live_frame(frame_index=0, joints_rad=[float("inf")] * 7, gripper=1.0))

    assert parse_live_frame('live_frame={"frame_index":0,"joints_rad":[NaN]}') is None


def test_a_field_that_raises_while_being_read_is_caught_too():
    """The runtime passes joint angles as a thunk precisely so the read happens under the guard.
    Evaluated at the call site, a missing key would raise in the control loop instead."""
    printed: list[str] = []
    emitter = LiveFrameEmitter(interval=1, sink=printed.append)

    def missing_joint():
        raise KeyError("joint_4.pos")

    emitter.emit_step(0, joints_rad=missing_joint, gripper=1.0)
    emitter.emit_step(1, joints_rad=lambda: [0.0] * 7, gripper=lambda: 1.0)

    assert [parse_live_frame(line)["frame_index"] for line in printed] == [1]


@pytest.mark.parametrize("source", ["policy", "expert"])
def test_every_frame_says_who_was_driving(source: str):
    """Without this the view cannot show a handoff, which is the reason it exists during a DAgger
    rollout -- and a takeover that is invisible is a takeover that silently enters a success
    rate."""
    frame = build_live_frame(frame_index=0, joints_rad=[0.0] * 7, gripper=1.0, source=source)

    assert frame["source"] == source
