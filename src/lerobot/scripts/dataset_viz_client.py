#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import logging
import select
import socket
import sys
import termios
from contextlib import contextmanager

import tty

from lerobot.utils.utils import init_logging


QUIT_COMMANDS = {"q", "\x03", "\x04"}


def normalize_control_command(raw_command: str) -> str | None:
    normalized = raw_command.strip().lower()
    if not normalized:
        return None
    first = normalized[0]
    if first == "n":
        return "n"
    if first == "q":
        return "q"
    return None


@contextmanager
def raw_terminal_mode():
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def read_control_command() -> str:
    if sys.stdin.isatty():
        with raw_terminal_mode():
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], None)
                if ready:
                    return sys.stdin.read(1).lower()

    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    stripped = line.strip().lower()
    return stripped[:1] if stripped else ""


def send_control_command(sock: socket.socket, command: str) -> None:
    sock.sendall(f"{command}\n".encode("utf-8"))


def run_client(host: str, port: int) -> None:
    with socket.create_connection((host, port)) as sock:
        logging.info("Connected to dataset viz control server at %s:%d.", host, port)
        logging.info("Press 'n' for next episode, 'q' to quit.")
        while True:
            command = normalize_control_command(read_control_command())
            if command is None:
                continue
            send_control_command(sock, command)
            if command in QUIT_COMMANDS:
                return


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True, help="Dataset viz control server host.")
    parser.add_argument("--control-port", type=int, required=True, help="Dataset viz control server TCP port.")
    args = parser.parse_args()

    init_logging()
    try:
        run_client(args.host, args.control_port)
    except (ConnectionError, OSError) as exc:
        raise SystemExit(f"Failed to connect to dataset viz control server at {args.host}:{args.control_port}: {exc}")
    except EOFError:
        logging.info("EOF received. Exiting.")


if __name__ == "__main__":
    main()
