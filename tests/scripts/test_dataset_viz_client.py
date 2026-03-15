#!/usr/bin/env python

import os
import queue
import signal
from unittest.mock import MagicMock

from lerobot.scripts.dataset_viz_client import (
    build_rerun_connect_command,
    can_connect,
    normalize_control_command,
    parse_rerun_connect_url,
    restart_rerun_viewer,
    send_control_command,
    start_rerun_viewer,
    stop_rerun_viewer,
    wait_for_rerun_server_restart,
)
from lerobot.scripts.lerobot_dataset_viz import enqueue_control_command


class FakeSocket:
    def __init__(self):
        self.payloads = []

    def sendall(self, payload: bytes) -> None:
        self.payloads.append(payload)


def test_client_normalize_control_command_accepts_n_and_q():
    assert normalize_control_command("n") == "n"
    assert normalize_control_command(" Next ") == "n"
    assert normalize_control_command("q") == "q"
    assert normalize_control_command(" quit ") == "q"
    assert normalize_control_command("") is None
    assert normalize_control_command("x") is None


def test_send_control_command_writes_newline_delimited_bytes():
    sock = FakeSocket()

    send_control_command(sock, "n")
    send_control_command(sock, "q")

    assert sock.payloads == [b"n\n", b"q\n"]


def test_enqueue_control_command_only_for_supported_commands():
    command_queue: queue.Queue[str] = queue.Queue()

    assert enqueue_control_command(command_queue, "ignored") is None
    assert enqueue_control_command(command_queue, " Next ") == "n"
    assert enqueue_control_command(command_queue, "q") == "q"

    assert command_queue.get_nowait() == "n"
    assert command_queue.get_nowait() == "q"


def test_build_rerun_connect_command_uses_single_cli_invocation():
    assert build_rerun_connect_command("rerun", "rerun+http://127.0.0.1:19876/proxy") == [
        "rerun",
        "--connect",
        "rerun+http://127.0.0.1:19876/proxy",
    ]


def test_parse_rerun_connect_url_extracts_host_and_port():
    assert parse_rerun_connect_url("rerun+http://192.168.1.200:19876/proxy") == ("192.168.1.200", 19876)


def test_start_rerun_viewer_uses_new_process_session(monkeypatch):
    popen = MagicMock(return_value="proc")
    monkeypatch.setattr("lerobot.scripts.dataset_viz_client.subprocess.Popen", popen)

    process = start_rerun_viewer("rerun", "rerun+http://127.0.0.1:19876/proxy")

    assert process == "proc"
    _, kwargs = popen.call_args
    assert kwargs["start_new_session"] is True
    assert kwargs["stdin"] is not None


def test_stop_rerun_viewer_kills_process_group(monkeypatch):
    process = MagicMock()
    process.pid = 1234
    process.poll.return_value = None
    process.wait.return_value = None
    killpg = MagicMock()

    monkeypatch.setattr("lerobot.scripts.dataset_viz_client.os.killpg", killpg)

    stop_rerun_viewer(process)

    killpg.assert_called_once_with(1234, signal.SIGTERM)
    process.wait.assert_called_once()


def test_wait_for_rerun_server_restart_waits_for_outage_then_ready(monkeypatch):
    connectivity = iter([True, True, False, False, True])
    sleeps = []

    monkeypatch.setattr("lerobot.scripts.dataset_viz_client.can_connect", lambda host, port, timeout_s=0.5: next(connectivity))
    monkeypatch.setattr("lerobot.scripts.dataset_viz_client.time.sleep", lambda seconds: sleeps.append(seconds))

    wait_for_rerun_server_restart("rerun+http://127.0.0.1:19876/proxy", poll_interval_s=0.1)

    assert sleeps == [0.1, 0.1, 0.1]


def test_restart_rerun_viewer_stops_old_process_before_starting_new(monkeypatch):
    events = []
    old_process = MagicMock()
    old_process.poll.return_value = None
    old_process.wait.return_value = None

    monkeypatch.setattr(
        "lerobot.scripts.dataset_viz_client.start_rerun_viewer",
        lambda rerun_binary, rerun_connect_url: events.append(("start", rerun_binary, rerun_connect_url)) or "new-proc",
    )
    monkeypatch.setattr("lerobot.scripts.dataset_viz_client.time.sleep", lambda seconds: events.append(("sleep", seconds)))
    monkeypatch.setattr(
        "lerobot.scripts.dataset_viz_client.wait_for_rerun_server_restart",
        lambda rerun_connect_url: events.append(("wait", rerun_connect_url)),
    )

    new_process = restart_rerun_viewer(
        old_process,
        rerun_binary="rerun",
        rerun_connect_url="rerun+http://127.0.0.1:19876/proxy",
        restart_delay_s=0.2,
    )

    old_process.wait.assert_called_once()
    assert events == [
        ("sleep", 0.2),
        ("wait", "rerun+http://127.0.0.1:19876/proxy"),
        ("start", "rerun", "rerun+http://127.0.0.1:19876/proxy"),
    ]
    assert new_process == "new-proc"
