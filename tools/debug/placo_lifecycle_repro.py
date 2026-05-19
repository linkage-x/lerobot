#!/usr/bin/env python

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
from pathlib import Path
import subprocess
import sys

import numpy as np


def _default_urdf_path() -> str:
    return str(
        Path(__file__).resolve().parents[2]
        / "src"
        / "lerobot"
        / "robots"
        / "franka_research3"
        / "assets"
        / "franka_fr3"
        / "fr3_pika_gripper_ati.urdf"
    )


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal subprocess-based reproducer for placo / RobotKinematics "
            "constructor-destructor lifecycle crashes."
        )
    )
    parser.add_argument("--child", choices=_scenario_order(), help="Run a single scenario in the current process.")
    parser.add_argument("--urdf-path", default=_default_urdf_path(), help="URDF path used for placo RobotWrapper.")
    parser.add_argument("--target-frame", default="pika_gripper_ee", help="End-effector frame name.")
    return parser


def _scenario_order() -> list[str]:
    return [
        "import_placo",
        "import_backends",
        "import_placo_driver_symbol",
        "robot_wrapper_raw",
        "robot_wrapper_suppressed",
        "solver_only",
        "robot_kinematics_ctor",
        "robot_kinematics_fk",
        "robot_kinematics_ik",
        "local_wrapper_ctor",
        "local_wrapper_fk",
        "local_wrapper_ik",
        "local_dataclass_wrapper_ctor",
        "local_dataclass_wrapper_fk",
        "local_dataclass_wrapper_ik",
        "placo_driver_ctor",
        "placo_driver_fk",
        "placo_driver_ik",
        "env_config_plus_placo_driver_ctor",
    ]


def _run_child(mode: str, urdf_path: str, target_frame: str) -> None:
    joint_names = [
        "fr3_joint1",
        "fr3_joint2",
        "fr3_joint3",
        "fr3_joint4",
        "fr3_joint5",
        "fr3_joint6",
        "fr3_joint7",
    ]

    if mode == "import_placo":
        import placo  # type: ignore[import-not-found]

        print(json.dumps({"mode": mode, "status": "ok", "placo_module": placo.__name__}))
        return

    if mode == "import_backends":
        import lerobot.robots.franka_research3.backends as backends

        print(json.dumps({"mode": mode, "status": "ok", "module": backends.__name__}))
        return

    if mode == "import_placo_driver_symbol":
        from lerobot.robots.franka_research3.backends import PlacoKinematicsDriver

        print(json.dumps({"mode": mode, "status": "ok", "symbol": PlacoKinematicsDriver.__name__}))
        return

    if mode == "robot_wrapper_raw":
        import placo  # type: ignore[import-not-found]

        robot = placo.RobotWrapper(urdf_path)
        print(json.dumps({"mode": mode, "status": "ok", "joint_count": len(robot.joint_names())}))
        return

    if mode == "robot_wrapper_suppressed":
        import placo  # type: ignore[import-not-found]
        from lerobot.model.kinematics import _suppress_native_output

        with _suppress_native_output():
            robot = placo.RobotWrapper(urdf_path)
        print(json.dumps({"mode": mode, "status": "ok", "joint_count": len(robot.joint_names())}))
        return

    if mode == "solver_only":
        import placo  # type: ignore[import-not-found]

        robot = placo.RobotWrapper(urdf_path)
        solver = placo.KinematicsSolver(robot)
        solver.mask_fbase(True)
        print(json.dumps({"mode": mode, "status": "ok", "solver_type": type(solver).__name__}))
        return

    if mode == "robot_kinematics_ctor":
        from lerobot.model.kinematics import RobotKinematics

        kin = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=joint_names,
        )
        print(json.dumps({"mode": mode, "status": "ok", "joint_count": len(kin.joint_names)}))
        return

    if mode == "robot_kinematics_fk":
        from lerobot.model.kinematics import RobotKinematics

        kin = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=joint_names,
        )
        joints_deg = np.rad2deg(np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64))
        pose = kin.forward_kinematics(joints_deg)
        print(json.dumps({"mode": mode, "status": "ok", "tcp_z": float(pose[2, 3])}))
        return

    if mode == "robot_kinematics_ik":
        from lerobot.model.kinematics import RobotKinematics

        kin = RobotKinematics(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=joint_names,
        )
        joints_deg = np.rad2deg(np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64))
        pose = kin.forward_kinematics(joints_deg)
        pose[0, 3] += 0.001
        pose[1, 3] -= 0.001
        solution = kin.inverse_kinematics(joints_deg, pose)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "status": "ok",
                    "joint_delta_norm_deg": float(np.linalg.norm(solution - joints_deg)),
                }
            )
        )
        return

    if mode in {"local_wrapper_ctor", "local_wrapper_fk", "local_wrapper_ik"}:
        from lerobot.model.kinematics import RobotKinematics

        class LocalWrapper:
            def __init__(self, urdf_path: str, target_frame_name: str, joint_names: list[str]):
                self._kinematics = RobotKinematics(
                    urdf_path=urdf_path,
                    target_frame_name=target_frame_name,
                    joint_names=joint_names,
                )

            def forward_kinematics(self, joint_positions_rad: np.ndarray) -> np.ndarray:
                joint_positions_deg = np.rad2deg(np.asarray(joint_positions_rad, dtype=np.float64))
                return self._kinematics.forward_kinematics(joint_positions_deg)

            def inverse_kinematics(self, current_joint_positions_rad: np.ndarray, desired_pose: np.ndarray) -> np.ndarray:
                current_joint_positions_deg = np.rad2deg(np.asarray(current_joint_positions_rad, dtype=np.float64))
                solution_deg = self._kinematics.inverse_kinematics(current_joint_positions_deg, desired_pose)
                return np.deg2rad(np.asarray(solution_deg, dtype=np.float64))

        driver = LocalWrapper(urdf_path=urdf_path, target_frame_name=target_frame, joint_names=joint_names)
        if mode == "local_wrapper_ctor":
            print(json.dumps({"mode": mode, "status": "ok", "driver_type": type(driver).__name__}))
            return

        joints_rad = np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
        pose = driver.forward_kinematics(joints_rad)
        if mode == "local_wrapper_fk":
            print(json.dumps({"mode": mode, "status": "ok", "tcp_z": float(pose[2, 3])}))
            return

        pose[0, 3] += 0.001
        pose[1, 3] -= 0.001
        solution = driver.inverse_kinematics(joints_rad, pose)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "status": "ok",
                    "joint_delta_norm_rad": float(np.linalg.norm(solution - joints_rad)),
                }
            )
        )
        return

    if mode in {"local_dataclass_wrapper_ctor", "local_dataclass_wrapper_fk", "local_dataclass_wrapper_ik"}:
        from lerobot.model.kinematics import RobotKinematics

        @dataclass
        class LocalDataclassWrapper:
            urdf_path: str
            target_frame_name: str
            joint_names: list[str]

            def __post_init__(self):
                self._kinematics = RobotKinematics(
                    urdf_path=self.urdf_path,
                    target_frame_name=self.target_frame_name,
                    joint_names=self.joint_names,
                )

            def forward_kinematics(self, joint_positions_rad: np.ndarray) -> np.ndarray:
                joint_positions_deg = np.rad2deg(np.asarray(joint_positions_rad, dtype=np.float64))
                return self._kinematics.forward_kinematics(joint_positions_deg)

            def inverse_kinematics(self, current_joint_positions_rad: np.ndarray, desired_pose: np.ndarray) -> np.ndarray:
                current_joint_positions_deg = np.rad2deg(np.asarray(current_joint_positions_rad, dtype=np.float64))
                solution_deg = self._kinematics.inverse_kinematics(current_joint_positions_deg, desired_pose)
                return np.deg2rad(np.asarray(solution_deg, dtype=np.float64))

        driver = LocalDataclassWrapper(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=joint_names,
        )
        if mode == "local_dataclass_wrapper_ctor":
            print(json.dumps({"mode": mode, "status": "ok", "driver_type": type(driver).__name__}))
            return

        joints_rad = np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
        pose = driver.forward_kinematics(joints_rad)
        if mode == "local_dataclass_wrapper_fk":
            print(json.dumps({"mode": mode, "status": "ok", "tcp_z": float(pose[2, 3])}))
            return

        pose[0, 3] += 0.001
        pose[1, 3] -= 0.001
        solution = driver.inverse_kinematics(joints_rad, pose)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "status": "ok",
                    "joint_delta_norm_rad": float(np.linalg.norm(solution - joints_rad)),
                }
            )
        )
        return

    if mode == "placo_driver_ctor":
        from lerobot.robots.franka_research3.backends import PlacoKinematicsDriver

        driver = PlacoKinematicsDriver(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=joint_names,
        )
        print(json.dumps({"mode": mode, "status": "ok", "driver_type": type(driver).__name__}))
        return

    if mode == "placo_driver_fk":
        from lerobot.robots.franka_research3.backends import PlacoKinematicsDriver

        driver = PlacoKinematicsDriver(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=joint_names,
        )
        joints_rad = np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
        pose = driver.forward_kinematics(joints_rad)
        print(json.dumps({"mode": mode, "status": "ok", "tcp_z": float(pose[2, 3])}))
        return

    if mode == "placo_driver_ik":
        from lerobot.robots.franka_research3.backends import PlacoKinematicsDriver

        driver = PlacoKinematicsDriver(
            urdf_path=urdf_path,
            target_frame_name=target_frame,
            joint_names=joint_names,
        )
        joints_rad = np.asarray([0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785], dtype=np.float64)
        pose = driver.forward_kinematics(joints_rad)
        pose[0, 3] += 0.001
        pose[1, 3] -= 0.001
        solution = driver.inverse_kinematics(joints_rad, pose)
        print(
            json.dumps(
                {
                    "mode": mode,
                    "status": "ok",
                    "joint_delta_norm_rad": float(np.linalg.norm(solution - joints_rad)),
                }
            )
        )
        return

    if mode == "env_config_plus_placo_driver_ctor":
        from lerobot.envs.fr3_mujoco import FR3MujocoEnvConfig
        from lerobot.robots.franka_research3.backends import PlacoKinematicsDriver

        cfg = FR3MujocoEnvConfig()
        driver = PlacoKinematicsDriver(
            urdf_path=cfg.urdf_path,
            target_frame_name=cfg.target_frame_name,
            joint_names=list(cfg.joint_names),
        )
        print(
            json.dumps(
                {
                    "mode": mode,
                    "status": "ok",
                    "driver_type": type(driver).__name__,
                    "target_frame": cfg.target_frame_name,
                }
            )
        )
        return

    raise ValueError(f"Unsupported mode: {mode}")


def _run_parent(urdf_path: str, target_frame: str) -> int:
    script_path = Path(__file__).resolve()
    print(
        json.dumps(
            {
                "script": str(script_path),
                "python": sys.executable,
                "urdf_path": urdf_path,
                "target_frame": target_frame,
            }
        )
    )
    exit_code = 0
    for mode in _scenario_order():
        proc = subprocess.run(
            [
                sys.executable,
                str(script_path),
                "--child",
                mode,
                "--urdf-path",
                urdf_path,
                "--target-frame",
                target_frame,
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )
        result = {
            "mode": mode,
            "returncode": proc.returncode,
            "stdout": proc.stdout.strip(),
            "stderr": proc.stderr.strip(),
        }
        print(json.dumps(result, ensure_ascii=True))
        if proc.returncode != 0:
            exit_code = 1
    return exit_code


def main() -> int:
    args = _build_parser().parse_args()
    if args.child:
        _run_child(args.child, args.urdf_path, args.target_frame)
        return 0
    return _run_parent(args.urdf_path, args.target_frame)


if __name__ == "__main__":
    raise SystemExit(main())
