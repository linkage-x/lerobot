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

"""P0: sweep scripted grasps around a standing peg and write the grasp success envelope.

A script of its own, like E6-lite, because it needs no checkpoint, camera or dataset: an arm, a
gripper and the fence. Normally started from the Unattended Runs page, which passes --out,
--stop-file and --continue-file inside the run's own directory. By hand:

    python tools/fr3/fr3_grasp_envelope_runtime.py --plan-only          # read the schedule
    python tools/fr3/fr3_grasp_envelope_runtime.py --home-first          # run it

See tools/fr3/grasp_envelope.py for what one trial does and what each verdict means.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fr3.collection_recorder import StopFile
from tools.fr3.grasp_envelope import (
    GRASP_ENVELOPE_CENTRE_REPEATS,
    GRASP_ENVELOPE_DESCENT_SPEED_MS,
    GRASP_ENVELOPE_DZ_OFFSETS_MM,
    GRASP_ENVELOPE_MIN_TCP_Z,
    GRASP_ENVELOPE_OPERATOR_WAIT_S,
    GRASP_ENVELOPE_SPOT_XY,
    GRASP_ENVELOPE_XY_DZ_MM,
    GRASP_ENVELOPE_XY_OFFSETS_MM,
    FileOperatorGate,
    GraspEnvelopeRequest,
    build_envelope_schedule,
    describe_schedule,
    done_indices,
    parse_offsets_mm,
    parse_points_mm,
    parse_xy,
    read_rows,
    run_grasp_envelope,
    validate_grasp_envelope,
)
from tools.fr3.grasp_loop import GRASP_LOOP_HELD_WIDTH, GRASP_LOOP_PICK_XYZ, GRASP_LOOP_TARGET_Z
from tools.fr3.terminal_servo import parse_terminal_servo_pose
from tools.fr3.workspace_fence import resolve_workspace_fence

DEFAULT_ROBOT_IP = "192.168.1.206"
DEFAULT_GRIPPER_PORT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:9.1.4:1.0-port0"
DEFAULT_RECORD_CONFIG = "tools/fr3/fr3_record_config.yaml"
DEFAULT_URDF = "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper.urdf"


def _csv(values) -> str:
    return ",".join(f"{v:g}" for v in values)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP)
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT)
    parser.add_argument("--gripper-backend", default="pika")
    parser.add_argument("--gripper-max-width-mm", type=float, default=90.0)
    parser.add_argument("--robot-urdf-path", default=DEFAULT_URDF)
    parser.add_argument("--target-frame-name", default="pika_gripper_ee")
    parser.add_argument("--record-config", default=DEFAULT_RECORD_CONFIG, help="Where the workspace fence comes from.")
    parser.add_argument("--spot", default=_csv(GRASP_ENVELOPE_SPOT_XY), help="x,y of the standing peg, metres.")
    parser.add_argument(
        "--peg-ref-z",
        type=float,
        default=GRASP_LOOP_TARGET_Z,
        help="TCP z of the reset's own grip on a standing peg: relative grasp height zero.",
    )
    parser.add_argument("--xy-offsets-mm", default=_csv(GRASP_ENVELOPE_XY_OFFSETS_MM))
    parser.add_argument("--xy-dz-mm", type=float, default=GRASP_ENVELOPE_XY_DZ_MM)
    parser.add_argument("--dz-offsets-mm", default=_csv(GRASP_ENVELOPE_DZ_OFFSETS_MM))
    parser.add_argument("--centre-repeats", type=int, default=GRASP_ENVELOPE_CENTRE_REPEATS)
    parser.add_argument("--repeats", type=int, default=1)
    parser.add_argument("--extra-points-mm", default="", help="Fine scan: 'dx,dy,dz; dx,dy,dz; ...' in mm.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start", choices=("fixture", "spot"), default="fixture")
    parser.add_argument("--pick-pose", default=_csv(GRASP_LOOP_PICK_XYZ))
    parser.add_argument("--verify-dz-mm", type=float, default=0.0)
    parser.add_argument("--held-width", type=float, default=GRASP_LOOP_HELD_WIDTH)
    parser.add_argument("--descent-speed-ms", type=float, default=GRASP_ENVELOPE_DESCENT_SPEED_MS)
    parser.add_argument("--min-tcp-z", type=float, default=GRASP_ENVELOPE_MIN_TCP_Z)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 runs the whole schedule.")
    parser.add_argument("--operator-wait-s", type=float, default=GRASP_ENVELOPE_OPERATOR_WAIT_S)
    parser.add_argument("--payload-mass-kg", type=float, default=0.0)
    parser.add_argument("--home-first", action="store_true", help="Home before reading the tool orientation.")
    parser.add_argument("--out", default="", help="JSONL to append rows to.")
    parser.add_argument("--stop-file", default="", help="Touch to stop after the current trial. SIGINT halts now.")
    parser.add_argument(
        "--continue-file",
        default="",
        help="Created by the page when a person has put a lost peg back. Defaults to CONTINUE beside --out.",
    )
    parser.add_argument(
        "--resume-rows",
        default="",
        help="rows.jsonl of an earlier run with the same plan: its finished indices are skipped.",
    )
    parser.add_argument("--plan-only", action="store_true", help="Print and validate the schedule; touch nothing.")
    return parser.parse_args(argv)


def build_request(args: argparse.Namespace) -> GraspEnvelopeRequest:
    return GraspEnvelopeRequest(
        spotXy=parse_xy(args.spot),
        pegRefZ=float(args.peg_ref_z),
        xyOffsetsMm=parse_offsets_mm(args.xy_offsets_mm),
        xyDzMm=float(args.xy_dz_mm),
        dzOffsetsMm=parse_offsets_mm(args.dz_offsets_mm),
        centreRepeats=int(args.centre_repeats),
        extraPointsMm=parse_points_mm(args.extra_points_mm),
        repeats=int(args.repeats),
        seed=int(args.seed),
        start=str(args.start),
        pickXyz=parse_terminal_servo_pose(args.pick_pose),
        verifyDzMm=float(args.verify_dz_mm),
        heldWidth=float(args.held_width),
        descentSpeedMs=float(args.descent_speed_ms),
        minTcpZ=float(args.min_tcp_z),
        maxSeconds=float(args.max_seconds),
        operatorWaitS=float(args.operator_wait_s),
        requestId=f"grasp_envelope_{time.strftime('%Y%m%d_%H%M%S')}",
    )


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request = build_request(args)
    schedule = build_envelope_schedule(request)
    workspace_min, workspace_max, fence_source = resolve_workspace_fence(record_config_path=args.record_config)
    qc = validate_grasp_envelope(request, schedule, workspace_min=workspace_min, workspace_max=workspace_max)
    print(describe_schedule(request, schedule), flush=True)
    print(f"[INFO] grasp_envelope=planned {json.dumps(qc, sort_keys=True)} fence={fence_source}", flush=True)
    prior_rows: list[dict] = []
    if args.resume_rows:
        prior_rows = read_rows(Path(args.resume_rows))
        done = done_indices(prior_rows)
        schedule = [point for point in schedule if point.index not in done]
        print(f"[INFO] grasp_envelope=resume from={args.resume_rows} done={len(done)} remaining={len(schedule)}", flush=True)
    if args.plan_only:
        return 0
    if not schedule:
        print("[INFO] grasp_envelope=nothing_left_to_run", flush=True)
        return 0

    out_path = Path(args.out) if args.out else REPO_ROOT / "outputs" / "analysis" / "grasp_envelope" / f"{request.requestId}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    stop_file = StopFile(Path(args.stop_file) if args.stop_file else out_path.with_name("STOP"))
    # A STOP left by a previous run is not a request about this one.
    stop_file.clear()
    gate = FileOperatorGate(
        Path(args.continue_file) if args.continue_file else out_path.with_name("CONTINUE"),
        stop_requested=stop_file.requested,
        timeout_s=request.operatorWaitS,
    )
    gate.clear()

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

    with out_path.open("a", encoding="utf-8") as handle:
        def write(row: dict) -> None:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

        write({"kind": "header", "request": request.payload(), "qc": qc, "schedule": [vars(p) for p in schedule]})
        robot.connect()
        try:
            if args.home_first:
                print("[INFO] grasp_envelope=homing", flush=True)
                robot.move_to_start()
            summary = run_grasp_envelope(
                robot,
                request,
                schedule,
                on_row=write,
                should_stop=stop_file.requested,
                wait_for_operator=gate,
                prior_rows=prior_rows,
            )
        finally:
            robot.disconnect()
    print(f"[INFO] grasp_envelope rows={out_path}", flush=True)
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
