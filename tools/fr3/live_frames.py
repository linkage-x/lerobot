"""What the arm is doing right now, in the smallest form a browser can draw.

The replay path renders MuJoCo to a video and the page plays it afterwards. That is the right
shape for validating a recorded episode and the wrong one for watching something happen: by the
time there is a file to play, the thing you wanted to see -- where the policy went, when the
operator took over, whether the handoff jolted -- is over.

So the running process publishes one line per step and the page draws it as it arrives. The
line carries joint angles rather than a rendered image because the browser already owns an FK
model of this arm (``public/fr3_mujoco_replay/kinematics.json``, used by the replay viewer):
seven floats redraw the arm, and seven floats cost less than a JPEG of it. Nothing here renders,
encodes or blocks -- an emitter that could stall the control loop would be a viewer that changes
the thing it is viewing.

The frame keys are the replay viewer's own (``joints_rad``, ``gripper``, ``target_position_m``,
``actual_position_m``), so the same Three.js component draws a live rollout and a recorded
replay without a second code path.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from typing import Any

# One token, chosen to be greppable in a log file and cheap to reject: every other line the
# runtime prints fails on the first character of this prefix.
LIVE_FRAME_PREFIX = 'live_frame='

# Enough to place the arm inside its own tolerances and no more. A joint rounded to 1e-5 rad is
# 6e-4 degrees, a position to 1e-5 m is 10 microns; carrying float64 tails instead would double
# the line length to describe motion no camera in this rig can see.
_ANGLE_DECIMALS = 5
_POSITION_DECIMALS = 5


def _round_all(values: Iterable[float], decimals: int) -> list[float]:
    return [round(float(value), decimals) for value in values]


def build_live_frame(
    *,
    frame_index: int,
    joints_rad: Iterable[float],
    gripper: float,
    source: str = 'policy',
    status: str = '',
    rollout_index: int | None = None,
    target_position_m: Iterable[float] | None = None,
    actual_position_m: Iterable[float] | None = None,
) -> dict[str, Any]:
    """One step of arm state, ready to be serialised.

    ``source`` is what makes this worth streaming during a DAgger rollout: it says whether the
    policy or the operator produced the command that moved the arm to this pose, so the handoff
    is visible in the drawing rather than only in the log.
    """
    frame: dict[str, Any] = {
        'frame_index': int(frame_index),
        'joints_rad': _round_all(joints_rad, _ANGLE_DECIMALS),
        'gripper': round(float(gripper), 4),
        'source': str(source),
    }
    if status:
        frame['status'] = str(status)
    if rollout_index is not None:
        frame['rollout_index'] = int(rollout_index)
    if target_position_m is not None:
        frame['target_position_m'] = _round_all(target_position_m, _POSITION_DECIMALS)
    if actual_position_m is not None:
        frame['actual_position_m'] = _round_all(actual_position_m, _POSITION_DECIMALS)
    return frame


def _reject_non_finite(constant: str) -> float:
    raise ValueError(f'live frame carried a non-finite value: {constant}')


def format_live_frame(frame: dict[str, Any]) -> str:
    """The line as it appears in the log. One line, no spaces after separators, no newline.

    ``allow_nan=False`` because the reader is a browser. Python writes a non-finite float as the
    bare word ``NaN``, which is not JSON and which ``JSON.parse`` refuses -- so one bad joint
    reading would not cost one frame, it would break the poll that carries every frame after it.
    Raising here loses the frame instead, and the emitter is built to survive that.
    """
    return LIVE_FRAME_PREFIX + json.dumps(
        frame, separators=(',', ':'), sort_keys=True, allow_nan=False
    )


def parse_live_frame(line: str) -> dict[str, Any] | None:
    """The inverse, for whoever is tailing the log. None for every line that is not one.

    Returns None rather than raising on malformed JSON: this runs inside a log follower, and a
    truncated line -- the follower reading a line the writer had not finished -- must cost that
    one frame and nothing else.
    """
    stripped = line.strip()
    if not stripped.startswith(LIVE_FRAME_PREFIX):
        return None
    try:
        # `parse_constant` fires on Infinity/-Infinity/NaN, which Python's decoder accepts by
        # default and every JSON consumer downstream of the gateway does not.
        frame = json.loads(
            stripped[len(LIVE_FRAME_PREFIX) :],
            parse_constant=_reject_non_finite,
        )
    except (json.JSONDecodeError, ValueError):
        return None
    if not isinstance(frame, dict) or 'frame_index' not in frame:
        return None
    return frame


class LiveFrameEmitter:
    """Prints frames at a fixed step interval, and never twice for one step.

    Holds the interval and the "is this even on" decision so the call site in the control loop
    stays one line. Interval 0 disables it: a terminal operator who never opens the page should
    not pay for a channel nobody is reading, and the runtime's log file is the same file either
    way.
    """

    def __init__(self, *, interval: int = 0, sink: Any = None) -> None:
        self.interval = max(0, int(interval))
        self._sink = sink
        self._failed = False

    @property
    def enabled(self) -> bool:
        return self.interval > 0

    def wants(self, step_index: int) -> bool:
        return self.enabled and int(step_index) % self.interval == 0

    def emit(self, frame: dict[str, Any]) -> None:
        if not self.enabled:
            return
        line = format_live_frame(frame)
        if self._sink is not None:
            self._sink(line)
            return
        print(line, flush=True)

    def emit_step(self, step_index: int, **frame_fields: Any) -> None:
        """Build and print in one call, or do nothing on a step that is not due.

        Every failure is swallowed after the first report. A viewer is an observer; a missing
        joint key on some robot config must cost the picture, not the rollout that is currently
        holding an object.

        Any field may be passed as a zero-argument callable, which is resolved *inside* the
        guard. That is what puts the caller's own extraction under it too: reading joint angles
        out of an observation can raise, and an argument evaluated at the call site would raise
        there, outside every protection this method offers.
        """
        if not self.wants(step_index):
            return
        try:
            resolved = {
                key: (value() if callable(value) else value) for key, value in frame_fields.items()
            }
            self.emit(build_live_frame(frame_index=int(step_index), **resolved))
        except Exception as exc:  # noqa: BLE001 - see docstring
            if not self._failed:
                self._failed = True
                print(f'[WARN] live_frame_emit_failed: {exc}', flush=True)
