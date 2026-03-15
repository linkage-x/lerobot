#!/usr/bin/env python

import queue

from lerobot.scripts.dataset_viz_client import normalize_control_command, send_control_command
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
