"""The BOX button: the firmware's events read as gestures, and the recorder's side of reporting them.

The sequences are the ones BOX 3969377039 reported on 2026-11-06.
"""

from __future__ import annotations

import io
import json
import threading
from unittest import mock

import pytest

from tools.thor.box_sdk import box_client as bc
from tools.thor.gmsl2 import box_button as bb, thor_record as tr

SINGLE = [(0, 51172837), (1, 51352838), (2, 51552838)]
DOUBLE = [(0, 58012838 - 140000), (1, 57952838), (0, 58012838), (1, 58152838), (3, 58352838)]
LONG = [(0, 65952850), (1, 67572838)]


def _gestures(sequence, decoder=None):
    decoder = decoder or bb.ButtonGestures()
    return [g for g in (decoder.feed(event, stamp) for event, stamp in sequence) if g]


@pytest.mark.parametrize(
    ("sequence", "expected"), [(SINGLE, ["single"]), (DOUBLE, ["double"]), (LONG, ["long"])]
)
def test_the_measured_sequences_make_one_gesture_each(sequence, expected):
    assert _gestures(sequence) == expected


def test_presses_in_a_row_are_read_one_by_one():
    assert _gestures(SINGLE + DOUBLE + LONG + SINGLE) == ["single", "double", "long", "single"]


def test_a_click_reported_after_a_long_press_is_the_same_press():
    assert _gestures([(0, 0), (1, 1_100_000), (2, 1_300_000)]) == ["long"]
    # ... but the next click, seconds later, is its own
    assert _gestures([(0, 0), (1, 1_100_000), (0, 5_000_000), (1, 5_100_000), (2, 5_300_000)]) == [
        "long",
        "single",
    ]


def test_a_hold_across_the_stamp_wrap_is_still_long_and_a_short_one_is_not():
    top = (1 << 32) - 500_000
    assert _gestures([(0, top), (1, 900_000)]) == ["long"]
    assert _gestures([(0, top), (1, 100_000)]) == []


def test_a_release_without_its_press_is_nothing():
    assert _gestures([(1, 5_000_000)]) == []


def test_an_event_the_firmware_may_add_is_ignored():
    assert _gestures([(7, 1)]) == []


class _Box:
    def set_button_callback(self, callback):
        self.callback = callback


def _event(event, stamp, wall=100.0, device=3969377039):
    return {"device_id": device, "event": event, "timestamp": stamp, "received_wall_time_s": wall}


def test_the_recorder_reports_gestures_and_keeps_double_clicks(capsys):
    box = _Box()
    marks = tr._watch_box_button(box)
    for event, stamp in SINGLE:
        box.callback(_event(event, stamp, wall=100.0))
    for event, stamp in DOUBLE:
        box.callback(_event(event, stamp, wall=105.5))
    lines = capsys.readouterr().out.splitlines()
    assert [json.loads(line.removeprefix("BOX_BUTTON "))["gesture"] for line in lines] == ["single", "double"]
    assert [m["wall_s"] for m in marks] == [105.5]


def test_each_box_keeps_its_own_press():
    box = _Box()
    tr._watch_box_button(box)
    with mock.patch.object(tr, "_emit") as emit:
        box.callback(_event(0, 0, device=1))
        box.callback(_event(0, 900_000, device=2))
        box.callback(_event(1, 1_000_000, device=1))
    assert emit.call_count == 1 and '"device_id":1' in emit.call_args.args[0]


def test_only_the_marks_inside_the_episode_go_to_it():
    marks = [{"wall_s": 9.0}, {"wall_s": 10.5, "mcu_us": 7}, {"wall_s": 30.0}]
    assert bb.episode_marks(marks, 10.0, 20.0) == [{"wall_s": 10.5, "mcu_us": 7, "t_rel_s": 0.5}]


def test_protocol_lines_from_two_threads_stay_two_lines(monkeypatch):
    class Chunked(io.StringIO):
        def write(self, text):  # the worst case for a write that is not atomic
            for char in text:
                super().write(char)
            return len(text)

    out = Chunked()
    monkeypatch.setattr(tr.sys, "stdout", out)

    def emit(text):
        for _ in range(200):
            tr._emit(text)

    threads = [threading.Thread(target=emit, args=(text,)) for text in ("Episode 0 ready", "TOUCHCALI_LOG x")]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    assert set(out.getvalue().splitlines()) == {"Episode 0 ready", "TOUCHCALI_LOG x"}


def test_the_pool_hands_the_callback_to_every_box():
    pool = bc.BoxPool(bc.BoxFleetConfig())
    clients = [mock.Mock(), mock.Mock()]
    pool._clients = [("box1", clients[0]), ("box2", clients[1])]
    callback = object()
    pool.set_button_callback(callback)
    for client in clients:
        client.set_button_callback.assert_called_once_with(callback)
