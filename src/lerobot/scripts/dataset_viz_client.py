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
import os
import select
import signal
import socket
import subprocess
import sys
import termios
import time
from contextlib import contextmanager
from urllib.parse import urlparse

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


def build_rerun_connect_command(rerun_binary: str, rerun_connect_url: str) -> list[str]:
    return [rerun_binary, "--connect", rerun_connect_url]


def start_rerun_viewer(rerun_binary: str, rerun_connect_url: str) -> subprocess.Popen:
    return subprocess.Popen(
        build_rerun_connect_command(rerun_binary, rerun_connect_url),
        stdin=subprocess.DEVNULL,
        start_new_session=True,
    )


def stop_rerun_viewer(process: subprocess.Popen | None, timeout_s: float = 5.0) -> None:
    if process is None:
        return
    if process.poll() is not None:
        return

    try:
        os.killpg(process.pid, signal.SIGTERM)
    except ProcessLookupError:
        return
    try:
        process.wait(timeout=timeout_s)
    except subprocess.TimeoutExpired:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            return
        process.wait(timeout=timeout_s)


def parse_rerun_connect_url(rerun_connect_url: str) -> tuple[str, int]:
    parsed = urlparse(rerun_connect_url)
    if parsed.hostname is None or parsed.port is None:
        raise ValueError(f"Invalid rerun connect url: {rerun_connect_url}")
    return parsed.hostname, parsed.port


def can_connect(host: str, port: int, timeout_s: float = 0.5) -> bool:
    try:
        with socket.create_connection((host, port), timeout=timeout_s):
            return True
    except OSError:
        return False


def wait_for_rerun_server_restart(
    rerun_connect_url: str,
    *,
    outage_timeout_s: float = 5.0,
    ready_timeout_s: float = 15.0,
    poll_interval_s: float = 0.1,
) -> None:
    host, port = parse_rerun_connect_url(rerun_connect_url)

    outage_deadline = time.monotonic() + outage_timeout_s
    observed_outage = False
    while time.monotonic() < outage_deadline:
        if not can_connect(host, port):
            observed_outage = True
            break
        time.sleep(poll_interval_s)

    if not observed_outage:
        logging.warning(
            "Did not observe rerun endpoint %s go down after switch request; continuing to wait for readiness.",
            rerun_connect_url,
        )

    ready_deadline = time.monotonic() + ready_timeout_s
    while time.monotonic() < ready_deadline:
        if can_connect(host, port):
            return
        time.sleep(poll_interval_s)

    raise TimeoutError(f"Timed out waiting for rerun endpoint to become ready: {rerun_connect_url}")


def restart_rerun_viewer(
    process: subprocess.Popen | None,
    *,
    rerun_binary: str,
    rerun_connect_url: str,
    restart_delay_s: float,
    wait_for_server_restart: bool = True,
) -> subprocess.Popen:
    stop_rerun_viewer(process)
    if restart_delay_s > 0:
        time.sleep(restart_delay_s)
    if wait_for_server_restart:
        wait_for_rerun_server_restart(rerun_connect_url)
    return start_rerun_viewer(rerun_binary, rerun_connect_url)


def run_client(
    host: str,
    port: int,
    *,
    rerun_connect_url: str | None = None,
    rerun_binary: str = "rerun",
    rerun_restart_delay_s: float = 0.2,
) -> None:
    rerun_process = None
    if rerun_connect_url is not None:
        rerun_process = start_rerun_viewer(rerun_binary, rerun_connect_url)

    with socket.create_connection((host, port)) as sock:
        logging.info("Connected to dataset viz control server at %s:%d.", host, port)
        if rerun_connect_url is not None:
            logging.info("Local rerun viewer started with %s.", rerun_connect_url)
            logging.info("Press 'n' to switch episode and restart the local viewer, 'q' to quit both.")
        else:
            logging.info("Press 'n' for next episode, 'q' to quit.")
        try:
            while True:
                command = normalize_control_command(read_control_command())
                if command is None:
                    continue
                send_control_command(sock, command)
                if command == "n" and rerun_connect_url is not None:
                    rerun_process = restart_rerun_viewer(
                        rerun_process,
                        rerun_binary=rerun_binary,
                        rerun_connect_url=rerun_connect_url,
                        restart_delay_s=rerun_restart_delay_s,
                    )
                if command in QUIT_COMMANDS:
                    return
        finally:
            stop_rerun_viewer(rerun_process)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--host", type=str, required=True, help="Dataset viz control server host.")
    parser.add_argument("--control-port", type=int, required=True, help="Dataset viz control server TCP port.")
    parser.add_argument(
        "--rerun-connect-url",
        type=str,
        default=None,
        help="Optional `rerun+http://HOST:GRPC_PORT/proxy` URL. When set, this script also launches and restarts the local rerun viewer.",
    )
    parser.add_argument(
        "--rerun-binary",
        type=str,
        default="rerun",
        help="Local rerun executable used with `--rerun-connect-url`.",
    )
    parser.add_argument(
        "--rerun-restart-delay-s",
        type=float,
        default=0.2,
        help="Delay before restarting the local rerun viewer after `n`.",
    )
    args = parser.parse_args()

    init_logging()
    try:
        run_client(
            args.host,
            args.control_port,
            rerun_connect_url=args.rerun_connect_url,
            rerun_binary=args.rerun_binary,
            rerun_restart_delay_s=args.rerun_restart_delay_s,
        )
    except (ConnectionError, OSError) as exc:
        raise SystemExit(f"Failed to connect to dataset viz control server at {args.host}:{args.control_port}: {exc}")
    except EOFError:
        logging.info("EOF received. Exiting.")


if __name__ == "__main__":
    main()
