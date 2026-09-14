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

"""E6-lite: run the terminal insertion experiment on a loop, without the policy and without a person.

Why this is a script of its own rather than a flag on the rollout runtime: the rollout runtime
exists to fly a checkpoint, and a terminal trial does not need one. E5 measured the approach's
contribution to terminal XY to be zero -- handoff offsets spanning 15.8-61.1 mm all converge to
1.7-2.0 mm before the descent begins -- so a loop that skips the policy runs the same experiment
without loading a model, opening a camera, or writing a dataset. What it needs is an arm, a
gripper, and the fence.

Two runs are worth making, and they differ by one flag:

    --search-ring 0     the bare capture radius. Each trial is E5's descent, aimed deliberately
                        off the hole by a known amount, so the run answers p(seat | offset).
                        This is the number the search pattern was sized against and it is
                        currently known only as 4.2 mm with a 2.5-8.8 mm interval.

    --search-ring 0.007 the same sweep with E7-C switched on, which answers what the covering
                        buys on top of the bare radius at each offset.

Read `--plan-only` output before either. It prints the exact schedule, validates every pose the
run can command against the fence and the arm's own reach, and touches nothing.
"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fr3.terminal_servo import (
    TERMINAL_SERVO_CONTACT_HOLD_S,
    TERMINAL_SERVO_MAX_SPEED_MS,
    TERMINAL_SERVO_CONTACT_LAG_M,
    TERMINAL_SERVO_CONTACT_STALL_M,
    TERMINAL_SERVO_SEARCH_POINTS,
    TERMINAL_SERVO_SEARCH_RING_M,
    TerminalServoRequest,
    parse_terminal_servo_pose,
)
from tools.fr3.terminal_trials import (
    TERMINAL_TRIAL_MAX_TILT_DEG,
    TerminalTrialsRequest,
    build_trial_schedule,
    describe_schedule,
    resume_state,
    run_terminal_trials,
    summarize_by_offset,
    validate_terminal_trials,
)
from tools.fr3.collection_recorder import StopFile
from tools.fr3.workspace_fence import resolve_workspace_fence


DEFAULT_ROBOT_IP = "192.168.1.206"
DEFAULT_GRIPPER_PORT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:9.1.4:1.0-port0"
DEFAULT_RECORD_CONFIG = "tools/fr3/fr3_record_config.yaml"
DEFAULT_URDF = "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper.urdf"
# The staging position the scene reset already picks from, so a run started with the peg on the
# table starts where every other tool on this rig expects to find it.
DEFAULT_PICK_POSE = "0.3640,-0.1370,0.0550"
# The offsets to sweep, in millimetres. Chosen to bracket the disputed number rather than to be
# round: the capture radius is claimed at 4.2 mm with a 2.5-8.8 mm interval, so the sweep has to
# have resolution across that whole span and enough beyond it to see the curve reach zero.
DEFAULT_OFFSETS_MM = "0,2,3,4,5,6,8,10"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT)
    parser.add_argument("--gripper-backend", default="pika")
    parser.add_argument("--gripper-max-width-mm", type=float, default=90.0)
    parser.add_argument("--robot-urdf-path", default=DEFAULT_URDF)
    parser.add_argument("--target-frame-name", default="pika_gripper_ee")
    parser.add_argument(
        "--record-config",
        default=DEFAULT_RECORD_CONFIG,
        help="Where the workspace fence comes from. The same file the rollout uses, for the same reason.",
    )
    parser.add_argument(
        "--hole-pose",
        required=True,
        help="Starting estimate of the seated pose, x,y,z in metres. Reference trials re-read it from contact.",
    )
    parser.add_argument("--handoff-z", type=float, default=0.12)
    parser.add_argument(
        "--step-tolerance-mm",
        type=float,
        default=0.0,
        help=(
            "What the positioning steps (align, search lift/transfer, retreat) must reach. "
            "0 keeps the servo's own 2.0 mm. Raise it past this arm's dead-band -- measured at "
            "about 2 mm along the direction of travel -- rather than raising the servo "
            "tolerance, which the descent also reads for its seated test."
        ),
    )
    parser.add_argument(
        "--pick-pose",
        default=DEFAULT_PICK_POSE,
        help="Where to fetch the peg from before the first trial. Pass '' if the arm already holds it.",
    )
    parser.add_argument(
        "--payload-mass-kg",
        type=float,
        default=0.0,
        help=(
            "Mass of the peg, told to libfranka's gravity model. Belongs here and not in the "
            "shared record config because this loop holds the peg for its whole run by "
            "invariant, while teleop recording spends much of its time empty-handed -- a "
            "constant load is right in one case and wrong by the same amount in the other."
        ),
    )
    parser.add_argument(
        "--max-tilt-deg",
        type=float,
        default=TERMINAL_TRIAL_MAX_TILT_DEG,
        help=(
            "Refuse to start when the tool leans more than this off the table normal. The run "
            "inherits its orientation from wherever the arm was left, and a leaning peg wedges "
            "instead of seating, so a tilted run measures a jam depth and looks normal doing "
            "it. 0 disables the check."
        ),
    )
    parser.add_argument(
        "--grasp-attempts",
        type=int,
        default=1,
        help=(
            "Times to descend and close before calling a grasp lost. Recovers fingers that "
            "closed just off the peg; a peg that fell over is still a halt, because no retry at "
            "the same pose finds it."
        ),
    )
    parser.add_argument(
        "--regrip-in-place",
        action="store_true",
        help=(
            "Close the fingers again where the peg was released, before retreating. Without it "
            "a peg that did not seat is left standing through the retreat and the next descent, "
            "and a peg that falls over ends the run -- which is the half of the sweep the sweep "
            "exists to produce."
        ),
    )
    parser.add_argument(
        "--regrip-drop-mm",
        type=float,
        default=0.0,
        help=(
            "Close this far below the height the peg was released at. It can only have fallen; "
            "closing at the same height is what an empty re-grip after a clean seating looks "
            "like. Clamped at the servo's floor."
        ),
    )
    parser.add_argument(
        "--release-only-when-seated",
        action="store_true",
        help=(
            "Keep hold of a peg that did not seat instead of standing it on the face, where it "
            "falls the instant the fingers open. Nothing measured is lost: the verdict is read "
            "off the descent, before the fingers move."
        ),
    )
    parser.add_argument(
        "--home-first",
        action="store_true",
        help="Move to the home keyframe before the first trial. Home is level to 0.05 deg, "
             "which is the remedy --max-tilt-deg names.",
    )
    parser.add_argument(
        "--resume",
        default="",
        help=(
            "A previous run's JSONL. Its finished trials are skipped and its hole estimate is "
            "carried over, so an interrupted night continues instead of starting again. Rows "
            "are appended to that same file."
        ),
    )
    # How hard the arm is allowed to lean on a peg that stopped. Runtime parameters because
    # they set the axial force a jammed peg sees, and this peg slides in the fingers before the
    # arm gives up: the sweep aims deliberately off the hole, so hitting the rim is not a fault
    # to be avoided, it is half the measurement.
    parser.add_argument(
        "--descent-speed-ms",
        type=float,
        default=TERMINAL_SERVO_MAX_SPEED_MS,
        help=(
            "How fast the descent's setpoint walks down. The only lever on a jammed peg's axial "
            "force that does not touch the contact detector's discrimination: the growth test is "
            "measured over commanded travel and so is speed-free, while the hold is measured in "
            "time and costs half the travel at half the speed. Capped at the module's own limit."
        ),
    )
    parser.add_argument(
        "--contact-lag-mm",
        type=float,
        default=TERMINAL_SERVO_CONTACT_LAG_M * 1000.0,
        help="Growth in follow-behind, over one contact window, that counts as contact.",
    )
    parser.add_argument(
        "--contact-hold-s",
        type=float,
        default=TERMINAL_SERVO_CONTACT_HOLD_S,
        help=(
            "How long that growth must persist before the descent stops. Every second of it is "
            "commanded travel the arm keeps pressing with, and it adds no discrimination once "
            "the growth test has fired."
        ),
    )
    parser.add_argument(
        "--contact-stall-mm",
        type=float,
        default=TERMINAL_SERVO_CONTACT_STALL_M * 1000.0,
        help="Absolute follow-behind that stops the descent outright, with no hold.",
    )
    parser.add_argument("--offsets-mm", default=DEFAULT_OFFSETS_MM)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--control-every", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--search-ring",
        type=float,
        default=0.0,
        help="Ring radius for the offset trials, in metres. 0 measures the bare capture radius.",
    )
    parser.add_argument("--search-points", type=int, default=TERMINAL_SERVO_SEARCH_POINTS)
    parser.add_argument(
        "--reference-search-ring",
        type=float,
        default=TERMINAL_SERVO_SEARCH_RING_M,
        help="Ring for the reference trials, which have to find the hole rather than measure it.",
    )
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 runs the whole schedule.")
    parser.add_argument(
        "--out",
        default="",
        help="JSONL to append rows to. Defaults to outputs/analysis/e6lite/<timestamp>.jsonl.",
    )
    parser.add_argument(
        "--stop-file",
        default="",
        help=(
            "Touch this file to stop after the current trial, holding the peg. Defaults to STOP "
            "beside --out. SIGINT remains the immediate halt."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print and validate the schedule without connecting to the arm.",
    )
    return parser.parse_args(argv)


def build_request(args: argparse.Namespace) -> TerminalTrialsRequest:
    offsets = tuple(float(part) for part in str(args.offsets_mm).split(",") if part.strip())
    servo = TerminalServoRequest(
        xyz=parse_terminal_servo_pose(args.hole_pose),
        stepToleranceM=(float(args.step_tolerance_mm) / 1000.0) or None,
        maxSpeedMs=float(args.descent_speed_ms),
        contactLagM=float(args.contact_lag_mm) / 1000.0,
        contactHoldS=float(args.contact_hold_s),
        contactStallM=float(args.contact_stall_mm) / 1000.0,
        handoffZ=float(args.handoff_z),
        searchPoints=int(args.search_points),
        requestId="",
    )
    pick = str(args.pick_pose).strip()
    return TerminalTrialsRequest(
        servo=servo,
        offsetsMm=offsets,
        repeats=int(args.repeats),
        controlEvery=int(args.control_every),
        seed=int(args.seed),
        searchRingM=float(args.search_ring),
        referenceRingM=float(args.reference_search_ring),
        pickXyz=parse_terminal_servo_pose(pick) if pick else None,
        maxSeconds=float(args.max_seconds),
        maxTiltDeg=float(args.max_tilt_deg),
        graspAttempts=int(args.grasp_attempts),
        regripInPlace=bool(args.regrip_in_place),
        releaseOnlyWhenSeated=bool(args.release_only_when_seated),
        regripDropM=float(args.regrip_drop_mm) / 1000.0,
        requestId=f"terminal_trials_{time.strftime('%Y%m%d_%H%M%S')}",
    )


def _default_out_path(request: TerminalTrialsRequest) -> Path:
    return REPO_ROOT / "outputs" / "analysis" / "e6lite" / f"{request.requestId}.jsonl"


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request = build_request(args)
    schedule = build_trial_schedule(request)

    workspace_min, workspace_max, fence_source = resolve_workspace_fence(
        record_config_path=args.record_config
    )
    print(
        f"[INFO] workspace_fence min=({workspace_min[0]:.3f}, {workspace_min[1]:.3f}, {workspace_min[2]:.3f}) "
        f"max=({workspace_max[0]:.3f}, {workspace_max[1]:.3f}, {workspace_max[2]:.3f}) source={fence_source}",
        flush=True,
    )
    # Without the arm this checks the fence only; the reach probe is the arm's own IK and is
    # added when there is one. A plan that passes here can still be refused on the workstation,
    # which is the right way round.
    qc = validate_terminal_trials(
        request, schedule, workspace_min=workspace_min, workspace_max=workspace_max
    )
    print(describe_schedule(request, schedule), flush=True)
    print(
        f"[INFO] terminal_trials=planned trials={qc['trials']} distinct_aims={qc['distinctAims']} "
        f"widest_offset_mm={qc['widestOffsetMm']:.1f} widest_commanded_mm={qc['widestCommandedMm']:.1f}",
        flush=True,
    )
    if args.plan_only:
        return 0

    resume_done: set[int] = set()
    resume_reference = None
    if args.resume:
        resume_path = Path(args.resume)
        if not resume_path.exists():
            raise SystemExit(f"--resume: no such file: {resume_path}")
        rows = []
        with resume_path.open(encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except json.JSONDecodeError:
                    # The last line of an interrupted run can be half written. Everything before
                    # it is still that run's answer, and refusing the file over its final byte
                    # would throw a night away to save a row.
                    break
        resume_done, resume_reference = resume_state(rows)
        schedule = [spec for spec in schedule if spec.index not in resume_done]
        print(
            f"[INFO] terminal_trials=resume from={resume_path} done={len(resume_done)} "
            f"remaining={len(schedule)} "
            f"reference={'inherited' if resume_reference else 'nominal'}",
            flush=True,
        )
        if not schedule:
            print("[INFO] terminal_trials=resume nothing left to run", flush=True)
            return 0

    out_path = Path(args.resume) if args.resume else (Path(args.out) if args.out else _default_out_path(request))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stop_file = StopFile(Path(args.stop_file) if args.stop_file else out_path.with_suffix(".STOP"))
    # Cleared rather than honoured: a STOP left by the previous run is not a request about this
    # one, and a brake that latched would be a brake nobody could release.
    stop_file.clear()
    print(
        f"[INFO] terminal_trials=brakes boundary_stop=`touch {stop_file.path}` immediate_halt=SIGINT",
        flush=True,
    )

    from lerobot.robots.franka_research3 import FrankaResearch3
    from lerobot.robots.franka_research3.config_franka_research3 import FrankaResearch3Config

    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip=args.robot_ip,
            gripper_port=args.gripper_port,
            gripper_backend=args.gripper_backend,
            allow_mock_gripper=False,
            urdf_path=str(args.robot_urdf_path),
            target_frame_name=str(args.target_frame_name),
            workspace_min=workspace_min,
            workspace_max=workspace_max,
            gripper_max_width_mm=float(args.gripper_max_width_mm),
            payload_mass_kg=float(args.payload_mass_kg),
            cameras={},
        )
    )

    # Every row is flushed as it happens. A loop whose whole purpose is to run unattended must
    # not hold its results in memory until it finishes, because the runs worth reading closely
    # are exactly the ones that did not.
    with out_path.open("a", encoding="utf-8") as handle:
        def write(row: dict) -> None:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

        write({"kind": "header", "request": request.payload(), "schedule": [vars(spec) for spec in schedule]})
        robot.connect()
        if args.home_first:
            # Before anything reads the orientation: the run inherits whatever pose the arm is
            # in, and home is the one pose known to be level.
            print("[INFO] terminal_trials=homing", flush=True)
            robot.move_to_start()
        try:
            summary = run_terminal_trials(
                robot, request, schedule, should_stop=stop_file, on_row=write,
                reference_xyz=resume_reference
            )
        finally:
            robot.disconnect()

    print(f"[INFO] terminal_trials=done halted_on={summary['haltedOn']} "
          f"trials={summary['trials']} seated={summary['seated']} out={out_path}", flush=True)
    # The reference updates are hole readings, and saying so here is what stops them being a
    # column nobody reads for a second time.
    updates = summary.get("referenceUpdates") or []
    print(
        f"[INFO] hole_readings={len(updates)} -- these are the only readings on this rig with the "
        f"grasp offset re-zeroed against the hole itself. Read them with:\n"
        f"       python tools/fr3/hole_stability.py --trials {out_path}",
        flush=True,
    )
    # Read back from the file rather than from this process's rows. A resumed run holds only
    # the trials it ran, so a table built from memory reports a night as the handful of offsets
    # that came after the interruption -- and says nothing about the ones before it, which is
    # worse than saying nothing at all.
    trial_rows: list[dict] = []
    with out_path.open(encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                continue
            if row.get("kind") == "trial":
                trial_rows.append(row)
    for bucket in summarize_by_offset(trial_rows):
        fraction = bucket["seatedFraction"]
        print(
            f"[INFO] offset_mm={bucket['offsetMm']:5.1f} n={bucket['n']:3d} seated={bucket['seated']:3d} "
            f"standing={bucket['standing']:3d} slip={bucket['slip']:3d} ambiguous={bucket['ambiguous']:3d} "
            f"failed={bucket['failed']:3d} "
            f"p_seat={'n/a' if fraction is None else f'{fraction:.2f}'}",
            flush=True,
        )
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
