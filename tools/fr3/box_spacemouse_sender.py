"""SpaceMouse on the host -> Thor -> RT workstation, with leased input packets."""
import argparse
from pathlib import Path
import time

from tools.fr3.box_teleop_protocol import BridgeClient, read_token
from tools.thor.fr3_teleop import make_spacemouse


def main(argv=None):
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--host", default="192.168.111.122")
    parser.add_argument("--port", type=int, default=18771)
    parser.add_argument("--config-path", type=Path, default=Path("tools/thor/gmsl2/thor_fr3_teleop.yaml"))
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config_path.read_text())
    bridge = BridgeClient(args.host, args.port, read_token(config["fr3_teleop"]["token_file"]))
    device = None
    try:
        reply = bridge.connect()
        device = make_spacemouse(config["teleop"], reply["gripper"])
        epoch = -1
        interval = 1 / float(config["fr3_teleop"].get("control_hz", 200))
        print("Host SpaceMouse connected; use the UI to start/stop teleoperation", flush=True)
        while True:
            started = time.monotonic()
            if not reply["active"] or reply["epoch"] != epoch:
                device.sync_gripper_baseline(reply["gripper"])
            epoch = reply["epoch"]
            reply = bridge.exchange(device.get_action(), active=reply["active"], epoch=epoch)
            time.sleep(max(0, interval - (time.monotonic() - started)))
    finally:
        bridge.close()
        if device is not None:
            device.disconnect()


if __name__ == "__main__":
    main()
