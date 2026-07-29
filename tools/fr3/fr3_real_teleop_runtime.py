#!/usr/bin/env python3

"""Run SpaceMouse teleoperation against the workstation FR3 and Pika gripper."""

from __future__ import annotations

import logging
import signal
from pathlib import Path

from lerobot.configs import parser
import lerobot.robots.franka_research3  # noqa: F401  # registers the `franka_research3` robot choice
from lerobot.robots import make_robot_from_config
from lerobot.scripts.lerobot_record import RecordConfig, record_loop
import lerobot.teleoperators.spacemouse  # noqa: F401
from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging

from tools.fr3.fr3_record_runtime import make_fr3_ee2ee_processors

_REPO_ROOT = Path(__file__).resolve().parents[2]
_MAX_SESSION_SECONDS = 365 * 24 * 60 * 60


def _resolve_workspace_path(value: str) -> str:
    path = Path(value)
    if str(path).startswith("/lerobot/"):
        return str(_REPO_ROOT / path.relative_to("/lerobot"))
    return value


@parser.wrap()
def teleoperate(cfg: RecordConfig) -> None:
    if cfg.teleop is None:
        raise ValueError("FR3 real teleop requires a teleoperator configuration.")

    init_logging()
    cfg.robot.urdf_path = _resolve_workspace_path(cfg.robot.urdf_path)
    # Camera ownership stays with the gateway preview processes. This keeps
    # both RealSense streams available even when the FR3 FCI connection fails.
    cfg.robot.cameras = {}

    robot = make_robot_from_config(cfg.robot)
    teleop = make_teleoperator_from_config(cfg.teleop)
    teleop_action_processor, robot_action_processor, robot_observation_processor = (
        make_fr3_ee2ee_processors(cfg)
    )
    events = {
        "exit_early": False,
        "rerecord_episode": False,
        "stop_recording": False,
    }

    def request_stop(_signum: int, _frame: object) -> None:
        events["exit_early"] = True
        events["stop_recording"] = True

    signal.signal(signal.SIGINT, request_stop)
    signal.signal(signal.SIGTERM, request_stop)

    try:
        teleop.connect()
        robot.connect()
        observation = robot.get_observation(include_cameras=False)
        sync_gripper = getattr(teleop, "sync_gripper_baseline", None)
        if callable(sync_gripper):
            sync_gripper(float(observation["gripper.pos"]))
        wait_until_idle = getattr(teleop, "wait_until_idle", None)
        if callable(wait_until_idle) and not wait_until_idle(timeout_s=5.0):
            logging.warning("SpaceMouse did not settle to idle within 5 seconds; starting with hold behavior.")

        print("fr3_real_teleop=READY", flush=True)
        record_loop(
            robot=robot,
            events=events,
            fps=cfg.control_fps or cfg.dataset.fps,
            teleop_action_processor=teleop_action_processor,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            teleop=teleop,
            dataset=None,
            policy=None,
            preprocessor=None,
            postprocessor=None,
            control_time_s=_MAX_SESSION_SECONDS,
            single_task=None,
            display_data=False,
            display_compressed_images=False,
        )
    finally:
        if robot.is_connected:
            robot.disconnect()
        if teleop.is_connected:
            teleop.disconnect()


def main() -> None:
    register_third_party_plugins()
    teleoperate()


if __name__ == "__main__":
    main()
