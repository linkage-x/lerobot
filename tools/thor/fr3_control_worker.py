#!/usr/bin/env python
"""Isolated, local FR3 controller. Sensor capture never shares its Python process.

The recorder launches this worker on F with an inherited Unix socketpair. The
native panda-py/libfranka controller owns the 1 kHz FCI loop; Python submits
bounded Cartesian increments through the existing FR3 IK and OTG implementation.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import fcntl
import hashlib
import math
import os
from pathlib import Path
import select
import signal
import socket
import sys
import threading
import time
from typing import Any, Callable

from tools.thor.fr3_ipc import JsonChannel


DELTA_KEYS = ("target_x", "target_y", "target_z", "target_wx", "target_wy", "target_wz")
STATE_SHAPES = {"q": 7, "dq": 7, "tau_J": 7, "tau_ext_hat_filtered": 7,
                "O_T_EE": 16, "O_F_ext_hat_K": 6}
# This profile uses the intersection of FR3 limits and the pinned native FER
# virtual walls, staying clear of their combined PD/D zones. A compatible
# libfranka version alone does not change panda-py's native wall constants.
SAFE_JOINT_LOWER = (-2.64, -1.57, -2.70, -2.84, -2.70, 0.60, -2.70)
SAFE_JOINT_UPPER = (2.64, 1.57, 2.70, -0.27, 2.70, 3.65, 2.70)
NATIVE_WALL_MARGIN = (0.25, 0.19, 0.19, 0.19, 0.08, 0.08, 0.08)
OTG_DYNAMIC_LIMITS = {"otg_max_velocity": 0.5, "otg_max_acceleration": 1.0, "otg_max_jerk": 1000.0}
CONTROL_READY_TIMEOUT_S = 1.0
# libfranka reports success over the last 100 native 1 kHz commands.
CONTROL_RATE_WINDOW_S = 0.1


class ArmOwnershipLock:
    """One native owner per configured robot address and Thor user, across UI sessions."""

    def __init__(self, robot_ip: str, lock_dir: Path | None = None):
        if not isinstance(robot_ip, str) or not robot_ip:
            raise ValueError("robot.robot_ip is required for native ownership.")
        key = hashlib.sha256(robot_ip.encode()).hexdigest()[:24]
        directory = lock_dir if lock_dir is not None else Path("/tmp")
        self.path = directory / f"lerobot-fr3-control-{os.getuid()}-{key}.lock"
        self.fd: int | None = None

    def __enter__(self):
        fd = os.open(self.path, os.O_CREAT | os.O_RDWR | os.O_CLOEXEC | os.O_NOFOLLOW, 0o600)
        try:
            if os.fstat(fd).st_uid != os.getuid():
                raise RuntimeError("The FR3 native ownership lock belongs to another user.")
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            os.ftruncate(fd, 0)
            os.write(fd, f"{os.getpid()}\n".encode())
        except BlockingIOError as exc:
            os.close(fd)
            raise RuntimeError(
                "Another local FR3 worker still owns this robot. Stop that process and wait for it to exit before F."
            ) from exc
        except Exception:
            os.close(fd)
            raise
        self.fd = fd
        return self

    def __exit__(self, _type, _value, _traceback):
        if self.fd is not None:
            os.close(self.fd)
            self.fd = None


# The CLI keeps these descriptors open until the operating system exits the
# process, even if a native destructor hangs after run_worker returns.
_PROCESS_OWNERSHIP_LOCKS: list[ArmOwnershipLock] = []


@dataclass(frozen=True)
class Settings:
    realtime_mode: str = "enforce"
    control_hz: float = 200.0
    telemetry_hz: float = 50.0
    command_timeout_s: float = 0.2
    max_state_age_s: float = 0.1
    min_success_rate: float = 0.995
    delta_pos: tuple[float, ...] = (0.001, 0.001, 0.001)
    delta_rot: tuple[float, ...] = (0.01, 0.01, 0.01)
    joint_lower: tuple[float, ...] = SAFE_JOINT_LOWER
    joint_upper: tuple[float, ...] = SAFE_JOINT_UPPER

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> Settings:
        section = config.get("fr3_teleop", {})
        robot = config.get("robot", {})
        result = cls(
            realtime_mode=section.get("realtime_mode", "enforce"),
            control_hz=float(section.get("control_hz", 200)),
            telemetry_hz=float(section.get("telemetry_hz", 50)),
            command_timeout_s=float(section.get("command_timeout_s", 0.2)),
            max_state_age_s=float(section.get("max_state_age_s", 0.1)),
            min_success_rate=float(section.get("min_success_rate", 0.995)),
            delta_pos=tuple(robot.get("max_target_delta_pos") or (0.001,) * 3),
            delta_rot=tuple(robot.get("max_target_delta_rot") or (0.01,) * 3),
            joint_lower=tuple(robot.get("otg_min_position") or SAFE_JOINT_LOWER),
            joint_upper=tuple(robot.get("otg_max_position") or SAFE_JOINT_UPPER),
        )
        if result.realtime_mode not in ("enforce", "ignore"):
            raise ValueError("fr3_teleop.realtime_mode must be 'enforce' or 'ignore'.")
        limits = (("control_hz", 1, 500), ("telemetry_hz", 1, 50),
                  ("command_timeout_s", 0.05, 0.5), ("max_state_age_s", 0.005, 0.5),
                  ("min_success_rate", 0, 1))
        for key, lower, upper in limits:
            value = getattr(result, key)
            if not math.isfinite(value) or not lower < value <= upper:
                raise ValueError(f"fr3_teleop.{key} must be finite and in ({lower}, {upper}].")
        for key in ("delta_pos", "delta_rot"):
            values = getattr(result, key)
            if len(values) != 3 or any(not math.isfinite(v) or v <= 0 for v in values):
                raise ValueError(f"{key} must contain three finite positive bounds.")
        if len(result.joint_lower) != 7 or len(result.joint_upper) != 7:
            raise ValueError("FR3 OTG bounds must contain seven joint positions.")
        for lower, upper, safe_lower, safe_upper in zip(
            result.joint_lower, result.joint_upper, SAFE_JOINT_LOWER, SAFE_JOINT_UPPER, strict=True
        ):
            if not math.isfinite(lower) or not math.isfinite(upper) or not safe_lower <= lower < upper <= safe_upper:
                raise ValueError("FR3 OTG bounds must remain within the conservative native/FR3 envelope.")
        for key, ceiling in OTG_DYNAMIC_LIMITS.items():
            if key in robot:
                validate_dynamic_limits(key, robot[key], ceiling)
        return result


def validate_dynamic_limits(key: str, values, ceiling: float) -> None:
    if not isinstance(values, (list, tuple)) or len(values) != 7 or any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        or not math.isfinite(value) or not 0 < value <= ceiling for value in values
    ):
        raise ValueError(
            f"robot.{key} must contain seven finite positive values no greater than "
            f"the initial Thor preset ({ceiling})."
        )


def load_config(path: str) -> dict[str, Any]:
    import yaml

    with open(path, encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    if not isinstance(config, dict):
        raise ValueError("FR3 config must be a YAML mapping.")
    return config


def check_realtime(mode: str = "enforce") -> None:
    """Apply the selected host scheduling policy before opening FCI."""
    if mode == "ignore":
        print("FR3 realtime_mode=ignore: using libfranka kIgnore on this host; "
              "FCI timing and fault checks remain active.", flush=True)
        return
    if mode != "enforce":
        raise ValueError("Unknown FR3 real-time mode")
    realtime_path = Path("/sys/kernel/realtime")
    if not realtime_path.is_file() or realtime_path.read_text().strip() != "1":
        raise RuntimeError(
            "FR3 requires a PREEMPT_RT kernel on this Thor. /sys/kernel/realtime is not 1; "
            "sensor capture remains available. Install and validate a compatible RT kernel before pressing F."
        )
    previous_policy = os.sched_getscheduler(0)
    previous_parameters = os.sched_getparam(0)
    try:
        priority = os.sched_get_priority_max(os.SCHED_FIFO)
        os.sched_setscheduler(0, os.SCHED_FIFO, os.sched_param(priority))
    except PermissionError as exc:
        raise RuntimeError("FR3 needs SCHED_FIFO permission (rtprio limits or CAP_SYS_NICE) on Thor.") from exc
    finally:
        os.sched_setscheduler(0, previous_policy, previous_parameters)


def build_robot(config: dict[str, Any], config_path: str):
    """Import all native dependencies and validate kinematics before opening FCI."""
    import draccus
    import numpy as np
    import panda_py  # noqa: F401
    import ruckig  # noqa: F401

    from lerobot.robots.franka_research3.config_franka_research3 import FrankaResearch3Config
    from lerobot.robots.franka_research3.backends import require_native_no_automatic_recovery
    from lerobot.robots.franka_research3.franka_research3 import FrankaResearch3

    require_native_no_automatic_recovery()

    raw = dict(config.get("robot", {}))
    raw.pop("type", None)
    if not raw.get("robot_ip") or not raw.get("urdf_path"):
        raise ValueError("robot.robot_ip and robot.urdf_path must be configured for FR3.")
    if raw.get("use_otg") is not True:
        raise ValueError("Direct Thor FR3 teleoperation requires robot.use_otg=true.")
    if "otg_min_position" not in raw or "otg_max_position" not in raw:
        raise ValueError("Configure explicit conservative robot.otg_min_position and otg_max_position.")
    settings = Settings.from_config(config)
    from panda_py import _core

    for key, configured, sign in (
        ("_JOINT_LIMITS_LOWER", settings.joint_lower, 1),
        ("_JOINT_LIMITS_UPPER", settings.joint_upper, -1),
    ):
        native = getattr(_core, key, None)
        if native is None:
            raise RuntimeError(f"The patched panda-py wheel must expose native {key} for joint wall checks.")
        values = np.asarray(native, dtype=np.float64).reshape(-1).tolist()
        if len(values) != 7:
            raise RuntimeError(f"Native {key} must contain seven joint wall bounds.")
        for value, position, margin in zip(values, configured, NATIVE_WALL_MARGIN, strict=True):
            if not math.isfinite(value) or sign * (position - float(value)) < margin - 1e-9:
                raise ValueError("Configured FR3 OTG joint bounds overlap the native virtual wall zones.")
    native_start = getattr(_core, "_JOINT_POSITION_START", None)
    if native_start is None:
        raise RuntimeError("The patched panda-py wheel must expose its native homing start position.")
    start = np.asarray(native_start, dtype=np.float64).reshape(-1).tolist()
    if len(start) != 7 or any(not math.isfinite(q) or not lower <= q <= upper for q, lower, upper in zip(
        start, settings.joint_lower, settings.joint_upper, strict=True
    )):
        raise ValueError("Configured FR3 joint bounds must contain the native move_to_start configuration.")
    urdf = Path(raw["urdf_path"]).expanduser()
    if not urdf.is_absolute():
        repository_path = Path(__file__).resolve().parents[2] / urdf
        urdf = repository_path if repository_path.is_file() else Path(config_path).resolve().parent / urdf
    raw.update(
        urdf_path=str(urdf), cameras={}, gripper_backend="mock", allow_mock_gripper=False,
        arm_start_controller_on_connect=False, arm_realtime_enforce=settings.realtime_mode == "enforce",
        arm_require_no_automatic_recovery=True,
        arm_state_poll_frequency_hz=settings.control_hz,
    )
    robot_config = draccus.decode(FrankaResearch3Config, raw)
    for key, ceiling in OTG_DYNAMIC_LIMITS.items():
        validate_dynamic_limits(key, getattr(robot_config, key), ceiling)
    robot = FrankaResearch3(robot_config)
    # IK/URDF failures must precede the request that moves the robot.
    robot._prepared_kinematics = robot._make_kinematics_driver()
    return robot


class InputMailbox:
    """One latest increment, consumed once, with independent input/parent watchdogs."""

    def __init__(self, settings: Settings, clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self.clock = clock
        self.lock = threading.Lock()
        self.last_parent_at = float("-inf")
        self.last_action_at = clock()
        self.latest_action: dict[str, Any] | None = None
        self.running = False
        self.watchdog_enabled = False

    def activate(self) -> None:
        with self.lock:
            self.running = True
            self.watchdog_enabled = True
            self.latest_action = None
            self.last_action_at = self.clock()

    def accept(self, packet: dict[str, Any]) -> bool:
        now = self.clock()
        op = packet.get("op")
        if op not in {"heartbeat", "action", "stop"}:
            raise ValueError(f"Unknown FR3 IPC operation: {op!r}.")
        if op == "stop":
            return False
        sent = packet.get("sent_monotonic_s")
        if isinstance(sent, bool) or not isinstance(sent, (int, float)) or not math.isfinite(sent):
            raise ValueError("FR3 heartbeat/action must carry a finite sent_monotonic_s.")
        if sent > now + 0.01:
            raise ValueError("FR3 input timestamp is in the future.")
        if now - sent > self.settings.command_timeout_s:
            if op == "heartbeat":
                # Stale heartbeats queued during imports cannot extend a live
                # lease. A following fresh heartbeat is checked before FCI.
                return True
            raise RuntimeError("FR3 input expired in the local IPC channel.")
        if op == "action":
            action = packet.get("action")
            if not isinstance(action, dict) or not isinstance(action.get("enabled"), bool):
                raise ValueError("FR3 action must contain a boolean enabled field.")
            values = (*self.settings.delta_pos, *self.settings.delta_rot, 1.0)
            clean = {"enabled": action["enabled"]}
            for key, bound in zip((*DELTA_KEYS, "gripper"), values, strict=True):
                value = action.get(key)
                if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
                    raise ValueError(f"FR3 action {key} must be finite.")
                if abs(value) > bound + 1e-12 or (key == "gripper" and value < 0):
                    raise ValueError(f"FR3 action {key} exceeds its configured bound.")
                clean[key] = float(value)
            with self.lock:
                self.last_parent_at = max(self.last_parent_at, float(sent))
                if self.running:
                    self.last_action_at = float(sent)
                    self.latest_action = clean
        else:
            with self.lock:
                self.last_parent_at = max(self.last_parent_at, float(sent))
        return True

    def check(self) -> None:
        now = self.clock()
        with self.lock:
            if self.watchdog_enabled and now - self.last_parent_at > self.settings.command_timeout_s:
                raise RuntimeError("FR3 recorder heartbeat expired; arm control stopped.")
            if self.running and now - self.last_action_at > self.settings.command_timeout_s:
                raise RuntimeError("FR3 SpaceMouse input expired; arm control stopped.")

    def consume(self) -> dict[str, Any] | None:
        with self.lock:
            action = self.latest_action
            self.latest_action = None
            return action


class InputReceiver:
    def __init__(self, channel: JsonChannel, mailbox: InputMailbox):
        self.channel = channel
        self.mailbox = mailbox
        self.stopped = threading.Event()
        self.reason = "Stopped by user."
        self.failed = False
        self.stop_callback: Callable[[], None] | None = None
        self.thread = threading.Thread(target=self._run, name="FR3LocalIPC", daemon=True)

    def cancel(self, reason: str, *, failed: bool = False) -> None:
        if self.stopped.is_set():
            return
        self.reason, self.failed = reason, failed
        self.stopped.set()
        callback = self.stop_callback
        if callback is not None:
            try:
                callback()
            except Exception:
                # Teardown still attempts disconnect. The parent bounds native hangs
                # by terminating this isolated worker after its shutdown deadline.
                pass

    def _run(self) -> None:
        while not self.stopped.is_set():
            try:
                # Drain queued heartbeats before checking their timestamp. Native
                # imports may hold the GIL while no FCI connection exists yet.
                for _ in range(256):
                    if not self.mailbox.accept(self.channel.receive()):
                        self.cancel("Stopped by user.")
                        break
                    if b"\n" not in self.channel.buffer and not select.select([self.channel.sock], [], [], 0)[0]:
                        break
                self.mailbox.check()
            except socket.timeout:
                try:
                    self.mailbox.check()
                except Exception as exc:
                    self.cancel(str(exc), failed=True)
            except EOFError:
                self.cancel("FR3 recorder disconnected; arm control stopped.", failed=True)
            except Exception as exc:
                self.cancel(str(exc), failed=True)

    def require_live(self) -> None:
        self.mailbox.check()
        if self.stopped.is_set():
            raise RuntimeError(self.reason)

    def begin_control(self) -> None:
        """Require a fresh stamped parent lease after imports and before opening FCI."""
        deadline = time.monotonic() + self.mailbox.settings.command_timeout_s
        while not self.stopped.is_set():
            now = time.monotonic()
            with self.mailbox.lock:
                if now - self.mailbox.last_parent_at <= self.mailbox.settings.command_timeout_s:
                    self.mailbox.watchdog_enabled = True
                    return
            if now >= deadline:
                raise RuntimeError("No fresh recorder heartbeat arrived before connecting FR3.")
            self.stopped.wait(0.005)
        raise RuntimeError(self.reason)


class StateGuard:
    def __init__(self, settings: Settings, clock: Callable[[], float] = time.monotonic):
        self.settings = settings
        self.clock = clock
        self.last_robot_time: float | None = None
        self.last_advance_at = clock()

    def check(self, state: dict[str, Any], *, active: bool) -> None:
        now = self.clock()
        for key, size in STATE_SHAPES.items():
            value = state.get(key)
            if not isinstance(value, list) or len(value) != size or any(not math.isfinite(v) for v in value):
                raise RuntimeError(f"FR3 state {key} is missing, invalid, or has the wrong shape.")
        if any(q < lower or q > upper for q, lower, upper in zip(
            state["q"], self.settings.joint_lower, self.settings.joint_upper, strict=True
        )):
            raise RuntimeError(
                "FR3 measured joints are outside the conservative control bounds. "
                "Use Desk to place the arm inside the configured range before pressing F."
            )
        sampled = state.get("sample_monotonic_s")
        robot_time = state.get("robot_time_s")
        if sampled is None or not math.isfinite(sampled) or now - sampled > self.settings.max_state_age_s:
            raise RuntimeError("FR3 measured state is stale.")
        if sampled > now + 0.01 or robot_time is None or not math.isfinite(robot_time):
            raise RuntimeError("FR3 state timestamps are invalid.")
        if self.last_robot_time is None or robot_time > self.last_robot_time:
            self.last_robot_time, self.last_advance_at = robot_time, now
        elif robot_time < self.last_robot_time:
            raise RuntimeError("FR3 native robot clock moved backwards.")
        elif active and now - self.last_advance_at > self.settings.max_state_age_s:
            raise RuntimeError("FR3 native robot clock stopped advancing.")
        mode = state.get("robot_mode")
        if mode not in {"kIdle", "kMove"}:
            raise RuntimeError(f"FR3 is in {mode!r}; clear the condition in Desk before pressing F again.")
        if active:
            if mode != "kMove":
                raise RuntimeError("FR3 native control loop is no longer active.")
            success = state.get("control_command_success_rate")
            if success is None or not math.isfinite(success) or not self.settings.min_success_rate <= success <= 1:
                raise RuntimeError(
                    f"FR3 FCI control command success rate {success!r} is below the configured "
                    f"limit {self.settings.min_success_rate}."
                )


def read_telemetry(robot) -> dict[str, Any]:
    """Use the native state's q for FK, avoiding a second, different state sample."""
    import numpy as np
    from lerobot.utils.rotation import Rotation

    state = robot._arm.get_telemetry()
    robot._raise_if_otg_failed()
    measured = robot._compute_ee_pose(np.asarray(state["q"], dtype=np.float64))
    state["measured_tcp"] = [*measured[:3, 3].tolist(), *Rotation.from_matrix(measured[:3, :3]).as_rotvec().tolist()]
    command = robot._last_command_pose
    if command is None:
        command = measured
    state["commanded_ee"] = [*command[:3, 3].tolist(), *Rotation.from_matrix(command[:3, :3]).as_rotvec().tolist()]
    return state


def wait_for_control_ready(robot, settings: Settings, receiver, telemetry_reader):
    """Qualify a fresh native hold window before accepting SpaceMouse increments."""
    guard = StateGuard(settings)
    deadline = time.monotonic() + CONTROL_READY_TIMEOUT_S
    first_active_time = None
    while True:
        receiver.require_live()
        state = telemetry_reader(robot)
        guard.check(state, active=False)
        if time.monotonic() - guard.last_advance_at > settings.max_state_age_s:
            raise RuntimeError("FR3 native robot clock stopped advancing during controller startup.")
        rate = state.get("control_command_success_rate")
        if isinstance(rate, bool) or not isinstance(rate, (int, float)) or not math.isfinite(rate) or not 0 <= rate <= 1:
            raise RuntimeError("FR3 controller startup reported an invalid FCI success rate.")
        if state["robot_mode"] == "kMove":
            if first_active_time is None:
                first_active_time = state["robot_time_s"]
            if state["robot_time_s"] - first_active_time >= CONTROL_RATE_WINDOW_S and rate >= settings.min_success_rate:
                guard.check(state, active=True)
                return state, guard
        else:
            first_active_time = None
        if time.monotonic() >= deadline:
            raise RuntimeError(
                f"FR3 controller did not qualify within {CONTROL_READY_TIMEOUT_S}s: "
                f"mode={state['robot_mode']}, FCI success rate={rate}, required={settings.min_success_rate}."
            )
        receiver.stopped.wait(1 / settings.control_hz)


def run_worker(channel: JsonChannel, config_path: str, *, config: dict[str, Any] | None = None,
               robot_factory: Callable | None = None, preflight: Callable[[], None] | None = None,
               telemetry_reader: Callable = read_telemetry) -> int:
    """One F activation, then exit. A retry is a fresh process after operator recovery."""
    robot = None
    receiver = None
    last_telemetry: dict[str, Any] = {}

    def status(state: str, message: str) -> None:
        channel.send({"event": "status", "state": state, "message": message,
                      "telemetry": last_telemetry if state == "running" else {}})

    def stop_native() -> None:
        if robot is not None:
            robot._otg_running = False
            arm = getattr(robot, "_arm", None)
            if arm is not None:
                arm.stop_motion()

    previous_handlers: dict[int, Any] = {}
    try:
        status("starting", "Checking Thor runtime prerequisites.")
        # Receive heartbeats before any heavyweight native/training imports.
        configuration = config if config is not None else load_config(config_path)
        settings = Settings.from_config(configuration)
        mailbox = InputMailbox(settings)
        receiver = InputReceiver(channel, mailbox)
        receiver.stop_callback = stop_native
        receiver.thread.start()
        if threading.current_thread() is threading.main_thread():
            for signum in (signal.SIGTERM, signal.SIGINT):
                previous_handlers[signum] = signal.getsignal(signum)
                signal.signal(signum, lambda _signum, _frame: receiver.cancel("Stopped by user."))
        if preflight is not None:
            preflight()
        else:
            check_realtime(settings.realtime_mode)
        receiver.require_live()
        robot = (robot_factory or build_robot)(configuration, config_path)
        receiver.begin_control()
        receiver.require_live()
        robot.connect()
        receiver.require_live()
        last_telemetry = telemetry_reader(robot)
        StateGuard(settings).check(last_telemetry, active=False)
        status("moving_to_start", "Moving FR3 to its start configuration.")
        robot.move_to_start()
        receiver.require_live()
        robot.start_arm_controller()
        receiver.require_live()
        last_telemetry, guard = wait_for_control_ready(robot, settings, receiver, telemetry_reader)
        mailbox.activate()
        status("running", "FR3 teleoperation is active.")
        next_report_at = time.monotonic() + 1 / settings.telemetry_hz
        while not receiver.stopped.is_set():
            tick = time.monotonic()
            receiver.require_live()
            last_telemetry = telemetry_reader(robot)
            guard.check(last_telemetry, active=True)
            action = mailbox.consume()
            if action is not None:
                robot.send_action(action)
            if tick >= next_report_at:
                status("running", "FR3 teleoperation is active.")
                next_report_at = tick + 1 / settings.telemetry_hz
            receiver.stopped.wait(max(0, tick + 1 / settings.control_hz - time.monotonic()))
        if receiver.failed:
            raise RuntimeError(receiver.reason)
        return_code = 0
    except Exception as exc:
        # A requested stop during startup/homing is expected. No state from that
        # activation is reused by a later F request.
        requested_stop = receiver is not None and receiver.stopped.is_set() and not receiver.failed
        if requested_stop:
            return_code = 0
        else:
            return_code = 1
            try:
                status("error", str(exc))
            except Exception:
                print(f"FR3 worker: {exc}", file=sys.stderr, flush=True)
    finally:
        if receiver is not None:
            receiver.stopped.set()
        try:
            stop_native()
        except Exception:
            pass
        if robot is not None and robot.is_connected:
            robot.disconnect()
        if receiver is not None:
            receiver.thread.join(timeout=0.2)
        for signum, handler in previous_handlers.items():
            signal.signal(signum, handler)
    if return_code == 0:
        try:
            status("stopped", "FR3 control stopped.")
        except Exception:
            pass
    return return_code


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config-path", required=True)
    parser.add_argument("--ipc-fd", type=int)
    parser.add_argument("--check", action="store_true", help="Validate this Thor without connecting FCI.")
    args = parser.parse_args(argv)
    if args.check:
        try:
            config = load_config(args.config_path)
            settings = Settings.from_config(config)
            check_realtime(settings.realtime_mode)
            build_robot(config, args.config_path)
        except Exception as exc:
            print(f"FR3 preflight failed: {exc}", file=sys.stderr)
            return 1
        print(f"FR3 native/kinematics preflight passed (realtime_mode={settings.realtime_mode}); "
              "no FCI connection was opened.")
        return 0
    if args.ipc_fd is None:
        parser.error("--ipc-fd is required unless --check is used.")
    channel = JsonChannel(socket.socket(fileno=args.ipc_fd), timeout_s=0.02)
    try:
        configuration = load_config(args.config_path)
        ownership = ArmOwnershipLock((configuration.get("robot") or {}).get("robot_ip"))
        ownership.__enter__()
        _PROCESS_OWNERSHIP_LOCKS.append(ownership)
        return run_worker(channel, args.config_path, config=configuration)
    except Exception as exc:
        try:
            channel.send({"event": "status", "state": "error", "message": str(exc), "telemetry": {}})
        except Exception:
            print(f"FR3 worker: {exc}", file=sys.stderr, flush=True)
        return 1
    finally:
        channel.close()


if __name__ == "__main__":
    raise SystemExit(main())
