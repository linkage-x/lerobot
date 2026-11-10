"""The button on the BOX: what the firmware reports, and the gesture an operator made.

Measured on BOX 3969377039 (2026-11-06, firmware after the v4.1 renumbering):

  ========  =============================================================
  event     meaning
  ========  =============================================================
  0         pressed
  1         released
  2         single click, reported 200 ms after the release (the firmware
            waits that long to rule out a second click)
  3         double click, reported 200 ms after the second release
  ========  =============================================================

There is no long-press event: holds of 1.6-2.0 s report 0 and 1 and no click, so a long press is
read here from the time between them, on the BOX's own microsecond stamps. Holds between 0.2 and
1.6 s were not measured; if the firmware reports a click after a hold counted as long, that click
is dropped, so one press never makes two gestures.

Standard library only: the recorder and its tests use it without the SDK.
"""

from __future__ import annotations

from typing import Any

EVENT_DOWN = 0
EVENT_UP = 1
EVENT_SINGLE = 2
EVENT_DOUBLE = 3

SINGLE = "single"
DOUBLE = "double"
LONG = "long"

LONG_PRESS_US = 1_000_000
# A click follows its release by 200 ms; twice that is still the same press.
CLICK_AFTER_LONG_US = 400_000
_STAMP_MODULUS = 1 << 32  # uint32 microseconds: wraps every 71.6 minutes


def _elapsed_us(start: int, end: int) -> int:
    return (end - start) % _STAMP_MODULUS


class ButtonGestures:
    """One BOX's events in, the gestures they make out."""

    def __init__(self) -> None:
        self._down_us: int | None = None
        self._long_us: int | None = None

    def feed(self, event: int, stamp_us: int) -> str | None:
        if event == EVENT_DOWN:
            self._down_us = stamp_us
            return None
        if event == EVENT_UP:
            down, self._down_us = self._down_us, None
            if down is not None and _elapsed_us(down, stamp_us) >= LONG_PRESS_US:
                self._long_us = stamp_us
                return LONG
            return None
        if event in (EVENT_SINGLE, EVENT_DOUBLE):
            long_us, self._long_us = self._long_us, None
            if long_us is not None and _elapsed_us(long_us, stamp_us) <= CLICK_AFTER_LONG_US:
                return None
            return SINGLE if event == EVENT_SINGLE else DOUBLE
        return None


def episode_marks(marks: list[dict[str, Any]], t_start: float, t_end: float) -> list[dict[str, Any]]:
    """The double clicks that fell inside an episode, with their time from its start."""
    return [
        {**mark, "t_rel_s": round(mark["wall_s"] - t_start, 4)}
        for mark in list(marks)
        if t_start <= mark["wall_s"] <= t_end
    ]
