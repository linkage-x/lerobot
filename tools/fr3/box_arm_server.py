"""Run the native FR3 controller on its dedicated wired, RT control host.

Only high-level deltas and downsampled state cross this TCP channel. Camera,
BOX SDK, disk, and browser work never runs inside the native FCI callback.
"""
from __future__ import annotations

import argparse
import importlib
import math
import os
from pathlib import Path
import signal
import socket
import threading
import time

from tools.fr3.box_teleop_protocol import (
    JsonConnection, Lease, authenticate, neutral_action, read_token, validate_action,
)


def realtime_preflight() -> None:
    marker = Path("/sys/kernel/realtime")
    if not marker.exists() or marker.read_text().strip() != "1":
        raise RuntimeError("FR3 control requires a PREEMPT_RT kernel (/sys/kernel/realtime=1)")
    policy = os.sched_getscheduler(0)
    param = os.sched_getparam(0)
    try:
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(1))
    except PermissionError as exc:
        raise RuntimeError("FR3 control requires permission to use SCHED_FIFO; configure rtprio limits") from exc
    finally:
        os.sched_setscheduler(0, policy, param)


def make_robot(config: dict):
    from lerobot.robots.franka_research3.config_franka_research3 import FrankaResearch3Config
    from lerobot.robots.franka_research3.franka_research3 import FrankaResearch3

    raw = dict(config["robot"])
    raw.pop("type", None)
    raw.update(cameras={}, gripper_backend="mock", allow_mock_gripper=True,
               arm_realtime_enforce=True, arm_start_controller_on_connect=True)
    path = Path(raw["urdf_path"])
    if not path.is_absolute():
        raw["urdf_path"] = str(Path.cwd() / path)
    return FrankaResearch3(FrankaResearch3Config(**raw))


class ArmSession:
    """One latest-command mailbox, one owner, and no queued motion replay."""
    def __init__(self, robot, config: dict):
        self.robot = robot
        self.config = config
        self.timeout_s = float(config.get("command_timeout_s", 0.15))
        self.lock = threading.Lock()
        self.stop = threading.Event()
        self.pending = None
        self.last_received_s = time.monotonic()
        self.telemetry = {}
        self.error = ""
        self.thread = None

    def start(self) -> None:
        from scipy.spatial.transform import Rotation
        self._rotation = Rotation
        try:
            self.robot.connect()
            self.last_received_s = time.monotonic()
            self.thread = threading.Thread(target=self._run, name="fr3-high-level-control", daemon=True)
            self.thread.start()
            deadline = time.monotonic() + 2.0
            while not self.snapshot() and not self.error:
                if time.monotonic() > deadline:
                    raise TimeoutError("FR3 state did not become available")
                time.sleep(0.005)
            if self.error:
                raise RuntimeError(self.error)
        except Exception:
            self.close()
            raise

    def submit(self, action: dict, active: bool) -> None:
        with self.lock:
            self.pending = action if active else neutral_action(action["gripper"])
            self.last_received_s = time.monotonic()

    def snapshot(self) -> dict:
        with self.lock:
            return dict(self.telemetry)

    def _run(self) -> None:
        interval = 1.0 / float(self.config.get("control_hz", 200))
        robot_time_s = None
        robot_advanced_at_s = time.monotonic()
        try:
            while not self.stop.is_set():
                started = time.monotonic()
                with self.lock:
                    action, self.pending = self.pending, None
                    last_received = self.last_received_s
                if started - last_received > self.timeout_s:
                    raise TimeoutError("FR3 command watchdog expired")
                if action is not None:
                    self.robot.send_action(action)
                state = self.robot._arm.get_telemetry()
                validate_state(state)
                if state.get("robot_time_s") != robot_time_s:
                    robot_time_s = state["robot_time_s"]
                    robot_advanced_at_s = time.monotonic()
                elif time.monotonic() - robot_advanced_at_s > self.timeout_s:
                    raise RuntimeError("FR3 native robot clock stopped advancing")
                if started - state["sample_monotonic_s"] > self.timeout_s:
                    raise RuntimeError("FR3 native state is stale")
                if state.get("robot_mode") in ("kReflex", "kUserStopped", "kGuiding"):
                    raise RuntimeError(f"FR3 controller unavailable: {state['robot_mode']}")
                if state["control_command_success_rate"] < float(self.config.get("min_success_rate", 0.995)):
                    raise RuntimeError("FR3 control command success rate below configured limit")
                pose = self.robot._last_command_pose
                measured_pose = self.robot._compute_ee_pose(state["q"])
                state["measured_tcp"] = [*measured_pose[:3, 3].tolist(),
                                          *self._rotation.from_matrix(measured_pose[:3, :3]).as_rotvec().tolist()]
                if pose is not None:
                    state["commanded_ee"] = [*pose[:3, 3].tolist(),
                                              *self._rotation.from_matrix(pose[:3, :3]).as_rotvec().tolist()]
                else:
                    state["commanded_ee"] = None
                state["applied_action"] = action
                with self.lock:
                    self.telemetry = state
                self.stop.wait(max(0.0, interval - (time.monotonic() - started)))
        except Exception as exc:
            self.error = str(exc)
            self.stop.set()

    def close(self) -> None:
        self.stop.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive():
                # Do not leave a native controller running after a hung IK/native call.
                self.robot._arm.disconnect()
                raise RuntimeError("FR3 control worker failed to stop")
        if self.robot.is_connected:
            self.robot.disconnect()


def validate_state(state: dict) -> None:
    for key, size in (("q", 7), ("dq", 7), ("tau_J", 7), ("tau_ext_hat_filtered", 7),
                      ("O_T_EE", 16), ("O_F_ext_hat_K", 6)):
        values = state.get(key)
        if not isinstance(values, list) or len(values) != size or not all(math.isfinite(v) for v in values):
            raise ValueError(f"FR3 state requires finite {key}[{size}]")
    for key in ("sample_monotonic_s", "control_command_success_rate", "robot_time_s"):
        if not math.isfinite(state.get(key, float("nan"))):
            raise ValueError(f"FR3 state requires finite {key}")


def serve_connection(sock: socket.socket, token: str, config: dict, robot_factory=make_robot, shutdown=None) -> None:
    timeout_s = float(config["fr3_teleop"].get("command_timeout_s", 0.15))
    connection = JsonConnection(sock, 5.0)
    session = None
    try:
        authenticate(connection.receive(), token)
        session = ArmSession(robot_factory(config), config["fr3_teleop"])
        session.start()
        lease = Lease(timeout_s)
        connection.timeout_s = timeout_s
        connection.send({"lease": lease.issue(), "state": session.snapshot()})
        while not session.stop.is_set() and not (shutdown and shutdown.is_set()):
            packet = connection.receive()
            received_s = time.monotonic()
            lease.accept(packet)
            if type(packet.get("active")) is not bool:
                raise ValueError("FR3 active must be boolean")
            action = validate_action(packet.get("action"))
            session.submit(action, packet["active"])
            state = session.snapshot()
            connection.send({"sequence": packet["sequence"], "lease": lease.issue(), "state": state,
                             "server_received_s": received_s, "server_sent_s": time.monotonic()})
        if session.error:
            raise RuntimeError(session.error)
    except Exception as exc:
        print(f"FR3 bridge session ended: {exc}", flush=True)
        try:
            connection.send({"error": str(exc)})
        except OSError:
            pass
    finally:
        try:
            if session is not None:
                session.close()
        finally:
            sock.close()


def main(argv=None) -> int:
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", type=Path, default=Path("tools/thor/gmsl2/thor_fr3_teleop.yaml"))
    parser.add_argument("--check", action="store_true", help="Check RT/dependencies without connecting to the robot")
    args = parser.parse_args(argv)
    config = yaml.safe_load(args.config_path.read_text())
    realtime_preflight()
    # FrankaResearch3 constructs its arm driver only during connect(), so
    # constructing the robot alone does not verify the Panda extension's libs.
    importlib.import_module("panda_py")
    if config["robot"].get("use_otg", True):
        importlib.import_module("ruckig")
    robot = make_robot(config)
    robot._make_kinematics_driver()  # load/validate model and task TCP without FCI
    del robot
    if args.check:
        print("FR3 RT kernel, scheduling permission, and native imports: OK")
        return 0
    settings = config["fr3_teleop"]
    token = read_token(settings["token_file"])
    stop = threading.Event()
    for sig in (signal.SIGTERM, signal.SIGINT):
        signal.signal(sig, lambda *_: stop.set())
    with socket.socket() as listener:
        listener.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        listener.bind((settings.get("arm_bind", "127.0.0.1"), int(settings["arm_port"])))
        listener.listen(1)
        listener.settimeout(0.5)
        print("FR3 arm bridge listening (robot connects only when Thor connects)", flush=True)
        while not stop.is_set():
            try:
                peer, _ = listener.accept()
            except socket.timeout:
                continue
            # A second peer can never command an already-owned FCI session.
            serve_connection(peer, token, config, shutdown=stop)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
