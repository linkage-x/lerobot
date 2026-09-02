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


def test_capture_root_is_applied_as_a_side_effect_not_a_queued_command():
    seen: list[str] = []

    # What the calibration wizard sends before a sweep: the destination, the
    # intent, the length, then the start newline.
    queue = _drain("capture_root:/data/calib_1/intrinsics\n\n", on_capture_root=seen.append)

    assert seen == ["/data/calib_1/intrinsics"]
    assert [cmd.kind for cmd in queue] == ["start", "quit"]


def test_capture_root_with_an_empty_value_asks_for_the_session_dataset_back():
    seen: list[str] = []

    _drain("capture_root:\n", on_capture_root=seen.append)

    assert seen == [""]


def test_capture_intent_carries_json_whose_colons_do_not_split_the_command():
    seen: list[str] = []

    # The payload is JSON, so it contains colons of its own -- only the first one
    # separates the command from its argument.
    _drain('capture_intent:{"purpose":"calibration_intrinsics"}\n', on_capture_intent=seen.append)

    assert seen == ['{"purpose":"calibration_intrinsics"}']


def test_capture_intent_keeps_the_case_of_its_payload():
    # The whole line is lowercased to match bare commands; the argument must not
    # be, or a camera named cam_A or a path under /Data would be silently mangled.
    seen: list[str] = []

    _drain('capture_intent:{"target_camera":"CAM_06"}\n', on_capture_intent=seen.append)

    assert seen == ['{"target_camera":"CAM_06"}']


def test_each_capture_root_counts_its_own_episodes(tmp_path):
    # Redirecting a sweep must not renumber it. A single session counter would
    # have the calibration tree's first segment written as episode_000004 just
    # because the task dataset already held four -- or, coming back, overwrite
    # one that is already there.
    dataset = tmp_path / "dataset"
    (dataset / "episodes" / "episode_000000").mkdir(parents=True)
    (dataset / "episodes" / "episode_000001").mkdir(parents=True)
    calibration = tmp_path / "calib_1" / "intrinsics"
    cache: dict = {}

    assert tr._next_episode_index_for_root(dataset, cache) == 2
    assert tr._next_episode_index_for_root(calibration, cache) == 0

    # Advancing one root leaves the other where it was.
    cache[calibration] = 1
    assert tr._next_episode_index_for_root(dataset, cache) == 2
    assert tr._next_episode_index_for_root(calibration, cache) == 1


def test_a_root_is_scanned_once_and_then_the_session_counts(tmp_path):
    # Re-scanning per episode would undo the count whenever a directory is not
    # on disk yet -- the recorder creates it when the episode actually starts.
    root = tmp_path / "dataset"
    (root / "episodes" / "episode_000000").mkdir(parents=True)
    cache: dict = {}

    assert tr._next_episode_index_for_root(root, cache) == 1
    cache[root] = 2
    assert tr._next_episode_index_for_root(root, cache) == 2
