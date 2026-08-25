"""Unit tests for the recorder's stdin command channel.

The gateway drives ``thor_record`` by writing single lines to its stdin. Two
kinds of line travel that pipe: FSM commands (start / save / discard / quit),
which have to be queued in order, and out-of-band side-effects
(``preview_demand``, the calibration triggers, ``episode_time:<seconds>``),
which must NOT enter the queue -- ``_wait_for_command`` would log them as
unexpected and drop them, and worse, a stray "start" would open an episode.
"""

from __future__ import annotations

import io
import threading

from tools.thor.gmsl2 import thor_record as tr


def _drain(text: str, **callbacks) -> list[tr.StdinCommand]:
    """Run the reader over a canned stdin, returning what it queued."""
    queue: list[tr.StdinCommand] = []
    original = tr.sys.stdin
    tr.sys.stdin = io.StringIO(text)
    try:
        tr._read_stdin_loop(queue, threading.Event(), **callbacks)
    finally:
        tr.sys.stdin = original
    return queue


def test_episode_time_is_applied_as_a_side_effect_not_a_queued_command():
    seen: list[float] = []

    # What the calibration wizard sends: the length, then the start newline.
    queue = _drain("episode_time:30\n\n", on_episode_time=seen.append)

    assert seen == [30.0]
    # Only the bare newline is a command; the trailing quit is stdin closing.
    assert [cmd.kind for cmd in queue] == ["start", "quit"]


def test_episode_time_accepts_fractional_seconds():
    seen: list[float] = []

    _drain("episode_time:7.5\n", on_episode_time=seen.append)

    assert seen == [7.5]


def test_episode_time_with_an_unparseable_length_is_ignored():
    seen: list[float] = []

    # A malformed length must not become an episode of some accidental duration,
    # and must not fall through to the "unrecognized command" path either.
    queue = _drain("episode_time:abc\n\n", on_episode_time=seen.append)

    assert seen == []
    assert [cmd.kind for cmd in queue] == ["start", "quit"]


def test_start_save_and_quit_still_queue_in_order():
    queue = _drain("\nsave\nq\n")

    assert [cmd.kind for cmd in queue] == ["start", "save", "quit"]


def test_next_episode_length_prefers_the_gateways_override():
    # A calibration sweep asks for 30 s while the config says 20 s.
    assert tr._next_episode_length_s(30.0, 20.0) == 30.0


def test_next_episode_length_falls_back_to_the_config():
    # No override pending: an ordinary capture is worth what the config says.
    assert tr._next_episode_length_s(0.0, 20.0) == 20.0


def test_next_episode_length_keeps_unlimited_meaning_unlimited():
    assert tr._next_episode_length_s(0.0, 0.0) == 0.0
