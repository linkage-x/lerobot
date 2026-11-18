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


_NONBLOCKING_CHILD = r"""
import subprocess, sys, threading, time
from tools.thor.gmsl2 import thor_record as tr
queue = []
stop = threading.Event()
reader = threading.Thread(target=tr._read_stdin_loop, args=(queue, stop), daemon=True)
reader.start()
# What ssh does to a stdin it inherits: switch the shared pipe to non-blocking.
subprocess.run([sys.executable, "-c", "import os, time; os.set_blocking(0, False); time.sleep(1.5)"])
reader.join(timeout=5)
print(",".join(c.kind for c in queue), flush=True)
"""


def test_a_child_making_stdin_nonblocking_does_not_end_the_session():
    """2026-10-13: a Discard pressed during the episode-stop ssh quit the whole recorder.

    ssh switched the inherited stdin to non-blocking; the reader's next readline()
    on the empty pipe came back "" and was taken for the GUI closing stdin.
    """
    import subprocess
    import sys
    import time
    from pathlib import Path

    repo = Path(__file__).resolve().parents[2]
    proc = subprocess.Popen(
        [sys.executable, "-c", _NONBLOCKING_CHILD], cwd=repo, stdin=subprocess.PIPE,
        stdout=subprocess.PIPE, text=True,
    )
    time.sleep(0.8)  # the child holds the pipe non-blocking now
    proc.stdin.write("n\n")
    proc.stdin.flush()
    time.sleep(1.5)  # past the child's exit
    proc.stdin.write("\n")
    proc.stdin.flush()
    proc.stdin.close()  # a real EOF still ends the session
    proc.wait(timeout=15)
    assert proc.stdout.read().strip() == "discard,start,quit"


def test_arm_gripper_is_a_side_effect_with_its_request_id():
    seen: list[tuple] = []

    queue = _drain("arm_gripper:3100715645:0.0000:g7\n", on_arm_gripper=lambda *a: seen.append(a))

    assert seen == [(3100715645, 0.0, "g7")]
    assert [cmd.kind for cmd in queue] == ["quit"]


def test_a_malformed_arm_gripper_command_is_dropped_not_guessed():
    seen: list[tuple] = []

    for line in ("arm_gripper:3100715645\n", "arm_gripper:abc:0.01:g1\n", "arm_gripper:3100715645:0.5:g1\n"):
        queue = _drain(line, on_arm_gripper=lambda *a: seen.append(a))
        assert [cmd.kind for cmd in queue] == ["quit"]
    assert seen == []


class _FakeGripperBox:
    def __init__(self, mode: int, width: float | None):
        self.mode, self.width, self.calls = mode, width, []

    def read(self):
        return {"sensors": {"box_gripper": {"distance_m": self.width}}}

    def get_mode(self, timeout_ms: int = 100):
        return 0, self.mode

    def set_mode(self, mode: int) -> int:
        self.calls.append(("set_mode", mode))
        self.mode = mode
        return 0

    def set_clamp_pos(self, pos_m: float) -> int:
        self.calls.append(("set_clamp_pos", pos_m))
        return 0


class _FakePool:
    def __init__(self, client):
        self.client = client

    def client_for_device(self, device_id: int):
        return self.client if device_id == 3100715645 else None


def _gripper_replies(monkeypatch, box, device_id: int = 3100715645, width_m: float = 0.0) -> list[dict]:
    import json

    lines: list[str] = []
    monkeypatch.setattr(tr, "_emit", lines.append)
    tr._run_arm_gripper(_FakePool(box), device_id, width_m, "g1")
    assert all(line.startswith("ARMGRIPPER_DONE ") for line in lines)
    return [json.loads(line.removeprefix("ARMGRIPPER_DONE ")) for line in lines]


def test_arm_gripper_out_of_control_mode_preloads_the_held_width_first(monkeypatch):
    box = _FakeGripperBox(mode=0, width=0.0887)

    (reply,) = _gripper_replies(monkeypatch, box, width_m=0.0)

    # what the jaw holds, then control mode, then the target -- never mode 0
    assert box.calls == [("set_clamp_pos", 0.0887), ("set_mode", 1), ("set_clamp_pos", 0.0)]
    assert reply == {**reply, "ok": True, "request": "g1", "modeBefore": 0, "widthBeforeM": 0.0887}


def test_arm_gripper_in_control_mode_only_sends_the_target(monkeypatch):
    box = _FakeGripperBox(mode=1, width=0.0476)

    (reply,) = _gripper_replies(monkeypatch, box, width_m=0.088)

    assert box.calls == [("set_clamp_pos", 0.088)]
    assert reply["ok"]


def test_arm_gripper_without_a_width_reading_does_not_switch_modes(monkeypatch):
    box = _FakeGripperBox(mode=0, width=None)

    (reply,) = _gripper_replies(monkeypatch, box)

    assert box.calls == []
    assert not reply["ok"] and "没有开度读数" in reply["error"]


def test_arm_gripper_for_a_box_the_recorder_does_not_have_is_an_error_line(monkeypatch):
    (reply,) = _gripper_replies(monkeypatch, _FakeGripperBox(mode=1, width=0.05), device_id=42)

    assert not reply["ok"] and "42" in reply["error"]
