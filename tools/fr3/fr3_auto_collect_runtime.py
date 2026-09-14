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

"""E6-data: collect approach and grasp data all night, from placements we chose and mistakes we made.

Why this is a script of its own rather than a flag on the rollout runtime: the rollout runtime
exists to fly a checkpoint, and none of this needs one. The expert here is a script; the peg's
pose is commanded rather than perceived; the data being collected is precisely the data the
checkpoint does not yet have. Loading a model to collect it would add a GPU, a policy chain and
an action-space conversion to a loop whose whole argument is that it needs an arm, a gripper, two
cameras and the fence.

**Read `--plan-only` before ever running this.** It prints the schedule the night will follow,
validates every pose it can command against the fence and the arm's own reach, and touches
nothing. A collection run is not a button that moves the arm once; it is a plan that moves it
some thousands of times, and the difference between authorising the first and the second is
whether anybody read it.

Two flags decide what the night is worth:

    --cycles 50 --recovery-fraction 0   the acceptance run. Fifty consecutive nominal cycles with
                                        nobody touching it, which is phase one's actual criterion:
                                        MTBF and data quality first, volume afterwards.

    --recovery-fraction 0.7             the collection run. Most cycles begin from a displaced
                                        pose that the script then corrects, which is the data a
                                        demonstration cannot contain because a human demonstrator
                                        does not make those mistakes on purpose.

What it will not do is produce terminal insertion labels. See the module docstring of
`tools/fr3/auto_collect.py` for why that is a boundary rather than an omission.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys
import time

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import yaml

from tools.fr3.auto_collect import (
    AUTO_PERTURB_XY_MM,
    AUTO_PERTURB_Z_RANGE_M,
    AutoCollectRequest,
    build_collection_schedule,
    describe_schedule,
    run_auto_collection,
    validate_auto_collection,
)
from tools.fr3.collection_recorder import (
    DEFAULT_SHARD_FRAMES,
    DEFAULT_STEP_MM,
    ControlTap,
    Recorder,
    StepAudit,
    StopFile,
    cameras_from_robot,
    write_session_header,
)
from tools.fr3.scene_reset import parse_mask_strokes
from tools.fr3.workspace_fence import resolve_workspace_fence


DEFAULT_ROBOT_IP = "192.168.1.206"
DEFAULT_GRIPPER_PORT = "/dev/serial/by-path/pci-0000:00:14.0-usb-0:9.1.4:1.0-port0"
DEFAULT_RECORD_CONFIG = "tools/fr3/fr3_record_config.yaml"
DEFAULT_URDF = "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper.urdf"
# The mask the reset panel already draws and persists. Read rather than re-entered so the region
# this run covers is the region somebody authorised on a map, not a second set of numbers.
DEFAULT_MASK_PATH = "outputs/metrology/scene_reset_mask.json"
# The staging pose every other tool on this rig expects to find the peg at.
DEFAULT_PICK_POSE = "0.3640,-0.1370,0.0550"
DEFAULT_PLACE_Z = 0.0550
# Where the peg is carried between cycles. Above the reset's 8 cm lift off the table, and below
# the top of the displacement band, so a carry never has to pass through a placement.
DEFAULT_CARRY_Z = 0.1500


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
        help="Where the workspace fence and the cameras come from. The same file the rollout uses.",
    )
    parser.add_argument("--mask", default=DEFAULT_MASK_PATH, help="The persisted scene-reset mask.")
    parser.add_argument("--pick-pose", default=DEFAULT_PICK_POSE, help="Pass '' if the arm already holds the peg.")
    parser.add_argument("--place-z", type=float, default=DEFAULT_PLACE_Z)
    parser.add_argument("--carry-z", type=float, default=DEFAULT_CARRY_Z)
    parser.add_argument("--cycles", type=int, default=50)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--recovery-fraction",
        type=float,
        default=0.0,
        help="Fraction of cycles that start from a displaced pose. 0 is the acceptance run.",
    )
    parser.add_argument("--perturb-xy-mm", type=float, default=AUTO_PERTURB_XY_MM)
    parser.add_argument("--perturb-z-low", type=float, default=AUTO_PERTURB_Z_RANGE_M[0])
    parser.add_argument("--perturb-z-high", type=float, default=AUTO_PERTURB_Z_RANGE_M[1])
    parser.add_argument(
        "--step-mm",
        type=float,
        default=DEFAULT_STEP_MM,
        help=(
            "What a recorded step may displace the tool by. The speed is derived from it, because "
            "millimetres per step is what a policy learns and seconds are not in its action space."
        ),
    )
    parser.add_argument("--control-fps", type=float, default=30.0, help="The rate the recorded legs are walked at.")
    parser.add_argument("--record-fps", type=float, default=30.0, help="The rate frames are kept at.")
    parser.add_argument("--shard-frames", type=int, default=DEFAULT_SHARD_FRAMES)
    parser.add_argument("--max-seconds", type=float, default=0.0, help="0 runs the whole schedule.")
    parser.add_argument("--out", default="", help="Defaults to outputs/auto_collect/<timestamp>.")
    parser.add_argument(
        "--stop-file",
        default="",
        help=(
            "Touch this file to stop at the next cycle boundary, holding the peg. Defaults to "
            "STOP inside the session directory. For an immediate halt send SIGINT instead -- that "
            "parks the arm wherever it is, still holding whatever it holds."
        ),
    )
    parser.add_argument(
        "--plan-only",
        action="store_true",
        help="Print and validate the schedule without connecting to the arm. Read this first.",
    )
    return parser.parse_args(argv)


def _parse_xyz(value: str) -> tuple[float, float, float]:
    parts = [part for part in str(value).replace(",", " ").split() if part]
    if len(parts) != 3:
        raise ValueError(f"expected three comma-separated metres, got {value!r}")
    return (float(parts[0]), float(parts[1]), float(parts[2]))


def load_mask(path: str | Path) -> tuple:
    """The drawn region, or a refusal. Deliberately not a default rectangle.

    The gateway degrades an unreadable mask to an empty one, which is right for a page an operator
    is looking at -- they can draw it again. It is wrong here: an empty mask at 2 a.m. would be a
    run with nowhere to put the peg, and a *default* mask would be a run covering a region nobody
    authorised.
    """

    resolved = Path(path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    raw = json.loads(resolved.read_text(encoding="utf-8"))
    strokes = parse_mask_strokes(raw)
    if not strokes:
        raise ValueError(f"{resolved} holds no strokes: draw the target region on the reset panel first.")
    return strokes


def load_realsense_cameras(record_config_path: str | Path) -> dict:
    """The rig's two cameras, built from the config the recording uses.

    Restricted to RealSense on purpose rather than duplicating the rollout runtime's full loader:
    both cameras on this rig are RealSense, that loader lives in a module that pulls in the whole
    policy stack, and a second full copy of it would be a second place for the camera geometry to
    drift. Anything else here fails loudly and names where the general loader is.
    """

    from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig

    resolved = Path(record_config_path)
    if not resolved.is_absolute():
        resolved = REPO_ROOT / resolved
    raw = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    entries = (raw.get("robot") or {}).get("cameras") or {}
    if not entries:
        raise ValueError(f"no robot.cameras entries in {resolved}")
    configs = {}
    for name, cfg in entries.items():
        if cfg.get("type") != "intelrealsense":
            raise ValueError(
                f"camera {name!r} is {cfg.get('type')!r}; this runtime only builds RealSense. Use "
                "fr3_act_infer_real_runtime.load_camera_configs for the general case."
            )
        configs[name] = RealSenseCameraConfig(
            serial_number_or_name=str(cfg["serial_number_or_name"]),
            width=int(cfg["width"]),
            height=int(cfg["height"]),
            fps=int(cfg["fps"]),
        )
    return configs


def build_request(args: argparse.Namespace) -> AutoCollectRequest:
    pick = str(args.pick_pose).strip()
    return AutoCollectRequest(
        maskStrokes=load_mask(args.mask),
        pickXyz=_parse_xyz(pick) if pick else None,
        placeZ=float(args.place_z),
        carryZ=float(args.carry_z),
        cycles=int(args.cycles),
        seed=int(args.seed),
        recoveryFraction=float(args.recovery_fraction),
        perturbXyMm=float(args.perturb_xy_mm),
        perturbZRangeM=(float(args.perturb_z_low), float(args.perturb_z_high)),
        stepMm=float(args.step_mm),
        controlPeriodS=1.0 / float(args.control_fps),
        maxSeconds=float(args.max_seconds),
        requestId=f"auto_collect_{time.strftime('%Y%m%d_%H%M%S')}",
    )


def _git_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except Exception:  # noqa: BLE001 - a missing git is not a reason not to collect
        return ""


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    request = build_request(args)
    schedule = build_collection_schedule(request)

    workspace_min, workspace_max, fence_source = resolve_workspace_fence(record_config_path=args.record_config)
    print(
        f"[INFO] workspace_fence min=({workspace_min[0]:.3f}, {workspace_min[1]:.3f}, {workspace_min[2]:.3f}) "
        f"max=({workspace_max[0]:.3f}, {workspace_max[1]:.3f}, {workspace_max[2]:.3f}) source={fence_source}",
        flush=True,
    )
    # Without the arm this checks the fence only; the reach probe is the arm's own IK and is added
    # when there is one. A plan that passes here can still be refused on the rig, which is the
    # right way round.
    qc = validate_auto_collection(
        request, schedule, workspace_min=workspace_min, workspace_max=workspace_max
    )
    print(describe_schedule(request, schedule), flush=True)
    print(
        f"[INFO] auto_collect=planned cycles={qc['cycles']} recovery={qc['recoveryCycles']} "
        f"nominal={qc['nominalCycles']} widest_start_offset_mm={qc['widestStartOffsetMm']:.1f} "
        f"step_mm={qc['stepMm']:.2f} recorded_speed_ms={qc['recordedSpeedMs']:.4f} "
        f"start_z_m={qc['startZRangeM'][0]:.3f}-{qc['startZRangeM'][1]:.3f}",
        flush=True,
    )
    if args.plan_only:
        return 0

    out_root = Path(args.out) if args.out else REPO_ROOT / "outputs" / "auto_collect" / request.requestId
    out_root.mkdir(parents=True, exist_ok=True)
    write_session_header(
        out_root,
        {
            "requestId": request.requestId,
            "gitRevision": _git_revision(),
            "startedAt": time.time(),
            "request": request.payload(),
            "schedule": [vars(spec) for spec in schedule],
            "qc": qc,
            "fence": {"min": list(workspace_min), "max": list(workspace_max), "source": fence_source},
            "recordFps": float(args.record_fps),
        },
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
            cameras=load_realsense_cameras(args.record_config),
        )
    )

    audit = StepAudit(limit_mm=request.stepLimitMm)
    tap = ControlTap(audit=audit)
    stop_file = StopFile(Path(args.stop_file) if args.stop_file else out_root / "STOP")
    # Cleared at the start rather than honoured: a STOP left behind by the previous night is not
    # a request about this one, and a run that refused to start because of it would be a run whose
    # brake had quietly become a latch.
    stop_file.clear()
    print(
        f"[INFO] auto_collect=brakes boundary_stop=`touch {stop_file.path}` immediate_halt=SIGINT",
        flush=True,
    )
    log_path = out_root / "cycles.jsonl"
    with log_path.open("a", encoding="utf-8") as handle:
        def write(row: dict) -> None:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            handle.flush()

        robot.connect()
        recorder = Recorder(
            tap,
            out_root,
            cameras=cameras_from_robot(robot),
            fps=float(args.record_fps),
            shard_frames=int(args.shard_frames),
        )
        recorder.start()
        try:
            summary = run_auto_collection(
                robot, request, schedule, tap=tap, should_stop=stop_file, on_row=write
            )
        finally:
            # Stopped before the arm is released so the last shard is footed while the process is
            # still healthy. A recorder shut down after a disconnect failure is a recorder whose
            # last shard depends on the disconnect succeeding.
            status = recorder.stop()
            write({"kind": "recorder", **status})
            robot.disconnect()

    print(f"[INFO] auto_collect=done halted_on={summary['haltedOn']} cycles={summary['cycles']} "
          f"held={summary['held']} recorded={summary['recordedEpisodes']} "
          f"invalidated={summary['invalidatedEpisodes']} out={out_root}", flush=True)
    print(f"[INFO] {recorder.describe_status()}", flush=True)
    audit_summary = summary.get("audit") or {}
    if audit_summary:
        print(
            f"[INFO] step_audit steps={audit_summary['steps']} violations={audit_summary['violations']} "
            f"p50_mm={audit_summary['p50StepMm']:.2f} p95_mm={audit_summary['p95StepMm']:.2f} "
            f"max_mm={audit_summary['maxStepMm']:.2f} (demo p50 {audit_summary['demoP50StepMm']:.2f} / "
            f"p95 {audit_summary['demoP95StepMm']:.2f}, limit {audit_summary['limitMm']:.2f})",
            flush=True,
        )
    return 0 if summary["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
