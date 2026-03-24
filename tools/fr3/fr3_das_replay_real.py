#!/usr/bin/env python3
"""
FR3 DAS 数据集真机重播启动器（宿主机运行）

用法：
    python tools/fr3/fr3_das_replay_real.py --episode 0
    python tools/fr3/fr3_das_replay_real.py --episode 5 --dataset outputs/datasets/lerobotv3_0310_100ep
    python tools/fr3/fr3_das_replay_real.py --episode 0 --dry-run   # 只打印 docker 命令
"""

from __future__ import annotations

import argparse
import shlex
import subprocess
import sys
from pathlib import Path

DEFAULT_DATASET = "outputs/datasets/lerobotv3_0310_100ep"
# 真机重播需要访问宿主机 /dev 下的串口设备，因此默认使用带硬件挂载的 teleop 服务。
DEFAULT_SERVICE = "lerobot-fr3-sim-teleop"
DEFAULT_ROBOT_IP = "192.168.1.208"
DEFAULT_GRIPPER_PORT = "/dev/ttyUSB0"
DEFAULT_GRIPPER_BACKEND = "das"
DEFAULT_RESET_GRIPPER_POSITION = 1.0
DEFAULT_RESET_GRIPPER_TIMEOUT_S = 2.0
DEFAULT_TIMING_SOURCE = "timestamp"
DEFAULT_OTG_SCALE = 1.0
DEFAULT_ANALYSIS_OUTPUT_DIR = "outputs/analysis"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Launch FR3 DAS dataset real-robot replay inside the Docker container."
    )
    parser.add_argument("--episode", type=int, default=0, help="Episode index to replay (default: 0)")
    parser.add_argument(
        "--dataset",
        default=DEFAULT_DATASET,
        help=f"Dataset path relative to repo root (default: {DEFAULT_DATASET})",
    )
    parser.add_argument("--fps", type=int, default=30, help="Replay frame rate (default: 30)")
    parser.add_argument(
        "--timing-source",
        choices=["fps", "timestamp"],
        default=DEFAULT_TIMING_SOURCE,
        help="Replay pacing source: fixed fps or dataset timestamps.",
    )
    parser.add_argument("--robot-ip", default=DEFAULT_ROBOT_IP, help=f"FR3 robot IP (default: {DEFAULT_ROBOT_IP})")
    parser.add_argument(
        "--filter-coeff",
        type=float,
        default=None,
        help="Optional panda_py JointPosition filter coefficient.",
    )
    parser.add_argument(
        "--damping",
        default=None,
        help="Optional panda_py JointPosition damping as 7 comma-separated floats.",
    )
    parser.add_argument(
        "--stiffness",
        default=None,
        help="Optional panda_py JointPosition stiffness as 7 comma-separated floats.",
    )
    parser.add_argument(
        "--otg-max-velocity",
        default=None,
        help="Optional OTG max velocity override as 7 comma-separated joint values (rad/s).",
    )
    parser.add_argument(
        "--otg-max-acceleration",
        default=None,
        help="Optional OTG max acceleration override as 7 comma-separated joint values (rad/s^2).",
    )
    parser.add_argument(
        "--otg-max-jerk",
        default=None,
        help="Optional OTG max jerk override as 7 comma-separated joint values (rad/s^3).",
    )
    parser.add_argument(
        "--otg-velocity-scale",
        type=float,
        default=DEFAULT_OTG_SCALE,
        help="Scale factor applied to OTG max velocity limits after optional override (default: 1.0).",
    )
    parser.add_argument(
        "--otg-acceleration-scale",
        type=float,
        default=DEFAULT_OTG_SCALE,
        help="Scale factor applied to OTG max acceleration limits after optional override (default: 1.0).",
    )
    parser.add_argument(
        "--otg-jerk-scale",
        type=float,
        default=DEFAULT_OTG_SCALE,
        help="Scale factor applied to OTG max jerk limits after optional override (default: 1.0).",
    )
    parser.add_argument(
        "--disable-otg",
        action="store_true",
        help="Disable the FR3 joint-space OTG smoother and send IK joint targets directly. Experimental on hardware.",
    )
    parser.add_argument(
        "--joint-targets-csv",
        default=None,
        help="Optional joint target CSV relative to repo root; when set, replay uses these joints instead of online EE->IK. Experimental on hardware.",
    )
    parser.add_argument(
        "--joint-target-column-prefix",
        default="bc_joint",
        help="Column prefix inside the joint target CSV (default: bc_joint).",
    )
    parser.add_argument(
        "--allow-experimental-joint-replay",
        action="store_true",
        help="Acknowledge that joint-target CSV replay is experimental on real hardware.",
    )
    parser.add_argument(
        "--allow-unsafe-otg-bypass",
        action="store_true",
        help="Acknowledge that disabling OTG is unsafe on real hardware.",
    )
    parser.add_argument("--gripper-port", default=DEFAULT_GRIPPER_PORT,
                        help=f"DAS controller serial port (default: {DEFAULT_GRIPPER_PORT})")
    parser.add_argument(
        "--gripper-backend",
        choices=["pika", "das"],
        default=DEFAULT_GRIPPER_BACKEND,
        help=f"Hardware gripper backend to use inside the container (default: {DEFAULT_GRIPPER_BACKEND})",
    )
    parser.add_argument(
        "--reset-gripper-position",
        type=float,
        default=DEFAULT_RESET_GRIPPER_POSITION,
        help="Normalized gripper position commanded before replay starts (default: 1.0, fully open).",
    )
    parser.add_argument(
        "--reset-gripper-timeout-s",
        type=float,
        default=DEFAULT_RESET_GRIPPER_TIMEOUT_S,
        help="Maximum time to wait for the pre-replay gripper reset feedback.",
    )
    parser.add_argument(
        "--analysis-output-dir",
        default=DEFAULT_ANALYSIS_OUTPUT_DIR,
        help=f"Directory for replay analysis artifacts relative to repo root (default: {DEFAULT_ANALYSIS_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Repository root (auto-detected by default)",
    )
    parser.add_argument("--compose-file", type=Path, default=None)
    parser.add_argument("--service", default=DEFAULT_SERVICE)
    parser.add_argument("--dry-run", action="store_true", help="Print docker command without running")
    return parser.parse_args(argv)


def repo_path_to_container_path(path_value: str | Path, workspace: Path) -> str:
    path = Path(path_value)
    if path.is_absolute():
        resolved = path.resolve()
        try:
            relative = resolved.relative_to(workspace)
        except ValueError as exc:
            raise ValueError(f"Path must be inside workspace {workspace}: {resolved}") from exc
    else:
        relative = path
    return f"/lerobot/{relative.as_posix()}"


def build_docker_command(args: argparse.Namespace) -> list[str]:
    workspace = args.workspace.resolve()
    compose_file = args.compose_file.resolve() if args.compose_file else workspace / "docker" / "docker-compose.yml"
    dataset_path = repo_path_to_container_path(args.dataset, workspace)
    analysis_output_dir = repo_path_to_container_path(args.analysis_output_dir, workspace)
    joint_targets_csv = (
        repo_path_to_container_path(args.joint_targets_csv, workspace)
        if args.joint_targets_csv is not None
        else None
    )

    runtime_args = [
        "cd /lerobot &&",
        "PYTHONPATH=/lerobot/src",
        "/lerobot/.venv/bin/python",
        "tools/fr3/fr3_das_replay_real_runtime.py",
        f"--episode={args.episode}",
        f"--dataset={shlex.quote(dataset_path)}",
        f"--fps={args.fps}",
        f"--timing-source={shlex.quote(args.timing_source)}",
        f"--robot-ip={shlex.quote(args.robot_ip)}",
        *( [f"--filter-coeff={args.filter_coeff}"] if args.filter_coeff is not None else [] ),
        *( [f"--damping={shlex.quote(args.damping)}"] if args.damping is not None else [] ),
        *( [f"--stiffness={shlex.quote(args.stiffness)}"] if args.stiffness is not None else [] ),
        *( [f"--otg-max-velocity={shlex.quote(args.otg_max_velocity)}"] if args.otg_max_velocity is not None else [] ),
        *( [f"--otg-max-acceleration={shlex.quote(args.otg_max_acceleration)}"] if args.otg_max_acceleration is not None else [] ),
        *( [f"--otg-max-jerk={shlex.quote(args.otg_max_jerk)}"] if args.otg_max_jerk is not None else [] ),
        f"--otg-velocity-scale={args.otg_velocity_scale}",
        f"--otg-acceleration-scale={args.otg_acceleration_scale}",
        f"--otg-jerk-scale={args.otg_jerk_scale}",
        *( ["--disable-otg"] if args.disable_otg else [] ),
        *( [f"--joint-targets-csv={shlex.quote(joint_targets_csv)}"] if joint_targets_csv is not None else [] ),
        f"--joint-target-column-prefix={shlex.quote(args.joint_target_column_prefix)}",
        *( ["--allow-experimental-joint-replay"] if args.allow_experimental_joint_replay else [] ),
        *( ["--allow-unsafe-otg-bypass"] if args.allow_unsafe_otg_bypass else [] ),
        f"--gripper-port={shlex.quote(args.gripper_port)}",
        f"--gripper-backend={shlex.quote(args.gripper_backend)}",
        f"--reset-gripper-position={args.reset_gripper_position}",
        f"--reset-gripper-timeout-s={args.reset_gripper_timeout_s}",
        f"--analysis-output-dir={shlex.quote(analysis_output_dir)}",
    ]

    return [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "run",
        "--rm",
        args.service,
        "bash",
        "-lc",
        " ".join(runtime_args),
    ]


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.joint_targets_csv is not None and not args.allow_experimental_joint_replay:
        print(
            "[ERROR] 真机默认已回退到已验证路径: action[t] + OTG。\n"
            "如需继续使用 joint-target CSV 实验链路，请显式添加 --allow-experimental-joint-replay。",
            file=sys.stderr,
        )
        return 2
    if args.disable_otg and not args.allow_unsafe_otg_bypass:
        print(
            "[ERROR] 真机默认保持 OTG 开启。\n"
            "如需继续使用 --disable-otg 实验链路，请显式添加 --allow-unsafe-otg-bypass。",
            file=sys.stderr,
        )
        return 2
    cmd = build_docker_command(args)
    if args.dry_run:
        print(shlex.join(cmd))
        return 0
    result = subprocess.run(cmd, check=False)
    if result.returncode != 0:
        print(f"[ERROR] docker compose run 失败，退出码 {result.returncode}", file=sys.stderr)
    return result.returncode


if __name__ == "__main__":
    raise SystemExit(main())
