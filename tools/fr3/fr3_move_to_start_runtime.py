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
Move the FR3 arm to the workstation XML ``home`` keyframe.

The read-back check here answers one question -- *is the arm at the home keyframe, or somewhere
else entirely* -- and deliberately not *how precisely did it arrive*. Precision is checked
downstream by the inference runtime, which compares the live EE pose against the dataset's own
start distribution in the tool frame the policy was trained in. That check knows things this one
cannot: which joints move the EE and which barely do, and how much the recorded episodes
themselves varied.

Getting that boundary wrong is what made this script abort a rollout on a 0.0102 rad wrist-roll
residual -- 0.02 mm of EE motion, well inside the spread of the very episodes it was homing for.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys


DEFAULT_ROBOT_IP = "192.168.1.206"
# First seven qpos values from
# src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper.xml:<key name="home">.
# This is the workstation recording start contract; it is deliberately not Panda.move_to_start().
FR3_PIKA_HOME_JOINTS_RAD = (0.0, -0.785, 0.0, -2.355, 0.0, 1.57079, 0.785)

# panda_py's own ``move_to_joint_position(success_threshold=...)`` default. The motion generator
# stops when every joint is inside this *or* joint velocity drops below ``dq_threshold`` (0.001
# rad/s), so a move that completed normally routinely lands with a residual of very nearly this
# size. Joint 7 lands worst of all: its impedance stiffness is 50 against 600 on joints 1-4, so
# the same stiction torque parks it twelve times further from the target. An acceptance gate set
# to this same number is therefore a coin flip on an identical pose -- it was, and the rollout
# died on 0.0102 rad.
MOTION_SUCCESS_THRESHOLD_RAD = 0.01

# Calibrated against the check that actually protects the rollout: the inference runtime's
# first-frame gate, which compares the live EE pose to the dataset's start distribution and
# allows 20 mm / 8 deg. Perturbing each joint at the home keyframe in
# fr3_pika_gripper_scene.xml, the worst joint for EE displacement is joint 4 at 15.6 mm per
# 0.03 rad, and every joint stays under 1.72 deg -- so no pose this gate admits can slip past the
# EE gate unseen. It is still 3x the controller's own convergence threshold, and one to two
# orders of magnitude below a genuine mis-pose (wrong keyframe, un-homed arm, DAS joint
# configuration), which is the failure this gate exists to catch.
DEFAULT_TOLERANCE_RAD = 0.03
# A residual above the gate is worth one more trajectory before giving up: a fresh motion from
# the current pose re-enters the control loop and usually breaks the stiction that stalled the
# first one. It is not worth many -- if two honest attempts cannot reach the keyframe, something
# is wrong that repeating will not fix.
DEFAULT_ATTEMPTS = 2
TOLERANCE_ENV_VAR = "FR3_MOVE_TO_START_TOLERANCE_RAD"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Move the FR3 arm to the workstation XML home keyframe.")
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help="FR3 controller IP address.")
    parser.add_argument(
        "--tolerance-rad",
        type=float,
        default=_env_tolerance_rad(),
        help=(
            "Maximum allowed read-back joint error after the move. This is a 'the arm is at the "
            "home keyframe rather than somewhere else' gate, not a precision requirement -- the "
            f"pose contract is enforced downstream on the EE. Overridable via {TOLERANCE_ENV_VAR}."
        ),
    )
    parser.add_argument(
        "--attempts",
        type=int,
        default=DEFAULT_ATTEMPTS,
        help="How many times to command the keyframe before accepting the residual as a mis-pose.",
    )
    return parser.parse_args(argv)


def _env_tolerance_rad() -> float:
    """Rig-level override, so tightening the gate does not need a code edit on the workstation."""
    raw = os.environ.get(TOLERANCE_ENV_VAR, "").strip()
    if not raw:
        return DEFAULT_TOLERANCE_RAD
    try:
        value = float(raw)
    except ValueError:
        return DEFAULT_TOLERANCE_RAD
    return value if value > 0.0 else DEFAULT_TOLERANCE_RAD


def format_joint_errors(errors: list[float]) -> str:
    """Every joint, every time.

    The single ``max_joint_error`` number this used to print cannot distinguish a harmless wrist
    residual from an arm parked in the wrong half of the workspace, and those want opposite
    responses from whoever reads the log.
    """
    return " ".join(f"j{index + 1}={error:+.5f}" for index, error in enumerate(errors))


def check_ping(robot_ip: str) -> tuple[str, str]:
    try:
        completed = subprocess.run(
            ["ping", "-c", "1", "-W", "1", robot_ip],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return "SKIP", "ping binary unavailable"

    details = completed.stdout.strip() or completed.stderr.strip() or f"returncode={completed.returncode}"
    if completed.returncode == 0:
        return "PASS", details
    return "FAIL", details


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from panda_py import Panda

    robot = None
    try:
        ping_status, ping_details = check_ping(args.robot_ip)
        print(f"fr3_move_to_start=PING status={ping_status} details={ping_details}")
        print(f"fr3_move_to_start=CONNECT robot_ip={args.robot_ip}")
        try:
            robot = Panda(args.robot_ip)
        except Exception as exc:
            raise RuntimeError(
                f"Failed to connect to FR3 at {args.robot_ip}. "
                "libfranka UDP timeouts usually mean the controller is unreachable from the current machine, "
                "the host NIC is not on the robot subnet, the robot is not ready for FCI, "
                "or another libfranka client already holds the session. "
                f"Ping probe: {ping_status}. "
                "Next steps: run "
                f"`python tools/fr3/fr3_hardware_smoke.py --robot-ip={args.robot_ip} --skip-spacemouse-list --skip-spacemouse-open` "
                "or retry with "
                f"`python tools/fr3/fr3_move_to_start.py --runtime host --robot-ip={args.robot_ip}`."
            ) from exc
        target = list(FR3_PIKA_HOME_JOINTS_RAD)
        tolerance = float(args.tolerance_rad)
        attempts = max(1, int(args.attempts))
        print(f"fr3_move_to_start=TARGET source=fr3_pika_gripper.xml:keyframe/home q={target}")
        for attempt in range(1, attempts + 1):
            converged = robot.move_to_joint_position(target)
            # Read once, right here. ``get_state()`` returns the last state the *control loop*
            # delivered, and the loop stops with the motion -- polling after this point re-reads a
            # frozen snapshot, so a settle-and-wait loop would measure nothing. Another trajectory
            # is the only way to get a newer number.
            state = robot.get_state()
            q = getattr(state, "q", None)
            if q is None:
                raise RuntimeError("FR3 move_to_start completed but state.q is unavailable.")
            current = [float(value) for value in q]
            errors = [actual - desired for actual, desired in zip(current, target, strict=True)]
            worst = max(range(len(errors)), key=lambda index: abs(errors[index]))
            max_error = abs(errors[worst])
            print(
                f"fr3_move_to_start=ARRIVED attempt={attempt}/{attempts} "
                f"controller_converged={bool(converged)} "
                f"max_joint_error={max_error:.5f} rad on joint{worst + 1} "
                f"tolerance={tolerance:.5f} rad"
            )
            print(f"fr3_move_to_start=JOINT_ERRORS rad {format_joint_errors(errors)}")
            if max_error <= tolerance:
                if max_error > MOTION_SUCCESS_THRESHOLD_RAD or not converged:
                    print(
                        f"fr3_move_to_start=NOTE the controller left {max_error:.5f} rad on "
                        f"joint{worst + 1}, at or beyond its own success_threshold of "
                        f"{MOTION_SUCCESS_THRESHOLD_RAD:.5f} rad. Accepted: the start-pose contract "
                        "is checked on the EE against the dataset start distribution once the "
                        "policy runs, which is where a start-pose error actually matters."
                    )
                break
            if attempt < attempts:
                print(
                    f"fr3_move_to_start=RETRY joint{worst + 1} is {max_error:.5f} rad out; "
                    "commanding the keyframe again",
                    file=sys.stderr,
                )
        else:
            raise RuntimeError(
                f"FR3 reached a different start pose after {attempts} attempt(s): "
                f"max_joint_error={max_error:.5f} rad on joint{worst + 1} "
                f"> tolerance={tolerance:.5f} rad. Per-joint errors (rad): "
                f"{format_joint_errors(errors)}. An error this size is a real mis-pose, not "
                "controller residual -- check that nothing is obstructing the arm, that no other "
                "libfranka client is fighting for the session, and that the arm is not still "
                f"holding a teleop pose. Override with {TOLERANCE_ENV_VAR} only if you know why."
            )
        print("fr3_move_to_start=PASS")
        print("Current joint angles (rad):", current)
        print(f"Max joint error (rad): {max_error:.6f}")
        return 0
    except Exception as exc:
        print(f"fr3_move_to_start=FAIL details={exc}", file=sys.stderr)
        raise
    finally:
        if robot is not None and hasattr(robot, "stop_controller"):
            try:
                robot.stop_controller()
            except Exception:
                pass


if __name__ == "__main__":
    raise SystemExit(main())
