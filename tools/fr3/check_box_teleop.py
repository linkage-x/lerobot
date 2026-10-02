"""Component checks without opening FCI, moving the arm, or commanding a gripper."""
import argparse
import json
from pathlib import Path
import time


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("component", choices=("config", "spacemouse", "rt"))
    parser.add_argument("--config-path", type=Path, default=Path("tools/thor/gmsl2/thor_fr3_teleop.yaml"))
    parser.add_argument("--duration-s", type=float, default=3)
    args = parser.parse_args(argv)
    if args.component == "rt":
        from tools.fr3.box_arm_server import main as arm_main
        return arm_main(["--check", "--config-path", str(args.config_path)])
    if args.component == "config":
        import yaml
        from lerobot.robots.franka_research3.config_franka_research3 import FrankaResearch3Config
        from lerobot.teleoperators.spacemouse.configuration_spacemouse import SpaceMouseTeleopConfig
        from tools.fr3.box_teleop_protocol import Lease
        from xml.etree import ElementTree
        config = yaml.safe_load(args.config_path.read_text())
        robot = dict(config["robot"])
        robot.pop("type", None)
        robot_cfg = FrankaResearch3Config(**robot)
        teleop = dict(config["teleop"])
        teleop.pop("type", None)
        SpaceMouseTeleopConfig(**teleop)
        frames = {item.attrib["name"] for item in ElementTree.parse(robot_cfg.urdf_path).findall("link")}
        if robot_cfg.target_frame_name not in frames:
            raise ValueError(f"Task TCP {robot_cfg.target_frame_name} is absent from the URDF")
        Lease(float(config["fr3_teleop"]["command_timeout_s"]))
        if config["fr3_teleop"]["input_source"] not in ("thor", "host"):
            raise ValueError("input_source must be thor or host")
        print("FR3 config, task TCP, SpaceMouse config, and timeout: OK")
        return 0
    # This lightweight check only needs numpy + pyspacemouse, not torch/FR3.
    from lerobot.teleoperators.spacemouse.backend import PySpaceMouseDriver
    driver = PySpaceMouseDriver(device_id=0)
    try:
        driver.connect()
        print(driver.describe(), flush=True)
        deadline = time.monotonic() + args.duration_s
        while time.monotonic() < deadline:
            reading = driver.poll()
            if reading is not None:
                print(json.dumps({"translation": reading.translation.tolist(), "rotation": reading.rotation.tolist(),
                                  "buttons": reading.buttons}), flush=True)
            time.sleep(0.1)
    finally:
        driver.disconnect()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
