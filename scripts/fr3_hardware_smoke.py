#!/usr/bin/env python3

# Copyright 2026 The HuggingFace Inc. team. All rights reserved.
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

"""
Run a reusable Docker-based FR3 + SpaceMouse hardware smoke test.

This script intentionally skips the Pika gripper path so the robot arm network
and SpaceMouse visibility can be validated independently.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path
import shlex
import subprocess
import sys


DEFAULT_IMAGE = "lerobot-user:local"
DEFAULT_ROBOT_IP = "192.168.1.206"

INNER_SMOKE_SCRIPT = r"""
import glob
import os
from pathlib import Path
import subprocess
import sys
import traceback


def record(results, name, ok, details):
    results.append((name, ok, details))
    status = "PASS" if ok else "FAIL"
    print(f"[{status}] {name}: {details}")


def check_ping(robot_ip, count):
    try:
        completed = subprocess.run(
            ["ping", "-c", str(count), robot_ip],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return True, "ping binary unavailable in container; skipped"

    details = completed.stdout.strip() or completed.stderr.strip() or f"returncode={completed.returncode}"
    return completed.returncode == 0, details


def check_fr3_sdk(robot_ip):
    robot = None
    try:
        from panda_py import Panda

        robot = Panda(robot_ip)
        state = robot.get_state()
        q = getattr(state, "q", None)
        if q is None:
            return False, "connected but state.q missing"
        return True, f"connected, dof={len(q)}, q0={float(q[0]):.4f}"
    except Exception as exc:
        return False, "".join(traceback.format_exception_only(type(exc), exc)).strip()
    finally:
        if robot is not None and hasattr(robot, "stop_controller"):
            try:
                robot.stop_controller()
            except Exception:
                pass


def check_spacemouse_list():
    hidraw_nodes = sorted(glob.glob("/dev/hidraw*"))
    try:
        import pyspacemouse

        if hasattr(pyspacemouse, "list_devices"):
            devices = list(pyspacemouse.list_devices())
        elif hasattr(pyspacemouse, "get_connected_devices"):
            devices = list(pyspacemouse.get_connected_devices())
        else:
            return False, f"hidraw={hidraw_nodes}, unsupported_pyspacemouse_api"
        if not devices:
            return False, f"hidraw={hidraw_nodes}, devices=[]"
        return True, f"hidraw={hidraw_nodes}, devices={devices}"
    except Exception as exc:
        return False, f"hidraw={hidraw_nodes}, error={exc!r}"


def check_spacemouse_open():
    driver = None
    try:
        from lerobot.teleoperators.spacemouse.backend import PySpaceMouseDriver

        driver = PySpaceMouseDriver(device_id=0)
        driver.connect()
        reading = driver.poll()
        return True, f"backend_open_ok, reading={reading}"
    except Exception as exc:
        return False, "".join(traceback.format_exception_only(type(exc), exc)).strip()
    finally:
        if driver is not None:
            try:
                driver.disconnect()
            except Exception:
                pass


def main():
    robot_ip = os.environ.get("FR3_SMOKE_ROBOT_IP", "192.168.1.206")
    ping_count = int(os.environ.get("FR3_SMOKE_PING_COUNT", "2"))
    skip_ping = os.environ.get("FR3_SMOKE_SKIP_PING", "0") == "1"
    skip_fr3_sdk = os.environ.get("FR3_SMOKE_SKIP_FR3_SDK", "0") == "1"
    skip_spacemouse_list = os.environ.get("FR3_SMOKE_SKIP_SPACEMOUSE_LIST", "0") == "1"
    skip_spacemouse_open = os.environ.get("FR3_SMOKE_SKIP_SPACEMOUSE_OPEN", "0") == "1"

    print(f"workspace={Path.cwd()}")
    print(f"robot_ip={robot_ip}")
    print("pika_gripper=mocked_skip")

    results = []

    if not skip_ping:
        ok, details = check_ping(robot_ip, ping_count)
        record(results, "fr3_ping", ok, details)

    if not skip_fr3_sdk:
        ok, details = check_fr3_sdk(robot_ip)
        record(results, "fr3_panda_sdk", ok, details)

    if not skip_spacemouse_list:
        ok, details = check_spacemouse_list()
        record(results, "spacemouse_list_devices", ok, details)

    if not skip_spacemouse_open:
        ok, details = check_spacemouse_open()
        record(results, "spacemouse_backend_open", ok, details)

    failures = [name for name, ok, _ in results if not ok]
    if failures:
        print(f"hardware_smoke=FAIL failed_checks={failures}")
        return 1

    print("hardware_smoke=PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
"""


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run a Docker-based FR3 + SpaceMouse smoke test.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="FR3 controller IP address.")
    parser.add_argument("--image", default=DEFAULT_IMAGE, help="Docker image to use.")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[1],
        help="Repository root to mount into the container.",
    )
    parser.add_argument("--ping-count", type=int, default=2, help="Ping attempts for the FR3 reachability check.")
    parser.add_argument("--skip-ping", action="store_true", help="Skip the FR3 ping reachability check.")
    parser.add_argument("--skip-fr3-sdk", action="store_true", help="Skip the panda_py SDK connectivity check.")
    parser.add_argument(
        "--skip-spacemouse-list",
        action="store_true",
        help="Skip the pyspacemouse.list_devices() visibility check.",
    )
    parser.add_argument(
        "--skip-spacemouse-open",
        action="store_true",
        help="Skip the LeRobot SpaceMouse backend open/poll check.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Print the Docker command without executing it.")
    return parser.parse_args(argv)


def build_docker_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    env_pairs = {
        "FR3_SMOKE_ROBOT_IP": args.robot_ip,
        "FR3_SMOKE_PING_COUNT": str(args.ping_count),
        "FR3_SMOKE_SKIP_PING": "1" if args.skip_ping else "0",
        "FR3_SMOKE_SKIP_FR3_SDK": "1" if args.skip_fr3_sdk else "0",
        "FR3_SMOKE_SKIP_SPACEMOUSE_LIST": "1" if args.skip_spacemouse_list else "0",
        "FR3_SMOKE_SKIP_SPACEMOUSE_OPEN": "1" if args.skip_spacemouse_open else "0",
    }
    command = [
        "docker",
        "run",
        "--rm",
        "--network",
        "host",
        "--ipc",
        "host",
        "--privileged",
        "-u",
        "root",
    ]
    for key, value in env_pairs.items():
        command.extend(["-e", f"{key}={value}"])
    command.extend(
        [
            "-v",
            f"{workspace}:/workspace",
            "-v",
            "/sys/bus/usb:/sys/bus/usb:ro",
            "-v",
            "/sys/class/input:/sys/class/input:ro",
            "-v",
            "/dev:/dev",
            "-v",
            "/dev/bus/usb:/dev/bus/usb",
            "-v",
            "/dev/input:/dev/input",
            "-v",
            "/var/run/dbus:/var/run/dbus",
            "-v",
            "/run/dbus:/run/dbus",
            args.image,
            "bash",
            "-lc",
            (
                "cd /workspace && "
                "PYTHONPATH=/workspace/src /lerobot/.venv/bin/python - <<'PY'\n"
                f"{INNER_SMOKE_SCRIPT}\n"
                "PY"
            ),
        ]
    )
    return command


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    command = build_docker_command(args)
    if args.dry_run:
        print(shlex.join(command))
        return 0

    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
