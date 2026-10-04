"""The real robot is never connected by these worker lifecycle tests."""

import math
import socket
import subprocess
import sys
import threading
import time
import types

import numpy as np
import pytest

from tools.thor import fr3_control_worker as worker
from tools.thor.fr3_ipc import JsonChannel


def action(**overrides):
    return {"enabled": True, **dict.fromkeys(worker.DELTA_KEYS, 0.0), "gripper": 0.5, **overrides}


def packet(**overrides):
    return {"op": "action", "sent_monotonic_s": time.monotonic(), "action": action(), **overrides}


def telemetry(*, robot_time=None, **overrides):
    return {**{key: [0.0] * size for key, size in worker.STATE_SHAPES.items()},
            "q": [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785],
            "sample_monotonic_s": time.monotonic(),
            "robot_time_s": time.monotonic() if robot_time is None else robot_time,
            "robot_mode": "kMove", "control_command_success_rate": 1.0,
            "measured_tcp": [0.0] * 6, "commanded_ee": [0.0] * 6, **overrides}


@pytest.mark.parametrize("overrides", [
    {"control_hz": float("nan")}, {"control_hz": 501}, {"telemetry_hz": 51},
    {"command_timeout_s": 0}, {"min_success_rate": 1.01}, {"max_state_age_s": -1},
    {"realtime_mode": "ignored"}, {"realtime_mode": False}, {"realtime_mode": None},
])
def test_worker_rejects_invalid_safety_settings(overrides):
    with pytest.raises(ValueError):
        worker.Settings.from_config({"fr3_teleop": overrides})


def test_mailbox_keeps_latest_increment_consumes_once_and_discards_startup_input():
    mailbox = worker.InputMailbox(worker.Settings())
    mailbox.accept(packet(action=action(target_x=0.0001)))
    mailbox.activate()
    assert mailbox.consume() is None
    mailbox.accept(packet(action=action(target_x=0.0002)))
    mailbox.accept(packet(action=action(target_x=0.0003)))
    assert mailbox.consume()["target_x"] == pytest.approx(0.0003)
    assert mailbox.consume() is None


@pytest.mark.parametrize("override", [
    {"action": action(target_x=float("nan"))}, {"action": action(target_wx=float("inf"))},
    {"action": action(target_y=0.0011)}, {"action": action(gripper=-0.1)},
    {"action": action(gripper=1.1)}, {"action": action(enabled=1)},
    {"sent_monotonic_s": time.monotonic() - 10}, {"sent_monotonic_s": math.inf},
    {"sent_monotonic_s": None},
])
def test_mailbox_rejects_nonfinite_out_of_bounds_and_expired_packets(override):
    mailbox = worker.InputMailbox(worker.Settings())
    with pytest.raises((ValueError, RuntimeError)):
        mailbox.accept(packet(**override))


def test_heartbeat_cannot_keep_a_missing_spacemouse_input_alive():
    now = [1.0]
    mailbox = worker.InputMailbox(worker.Settings(), clock=lambda: now[0])
    mailbox.activate()
    now[0] = 1.21
    mailbox.accept({"op": "heartbeat", "sent_monotonic_s": now[0]})
    with pytest.raises(RuntimeError, match="SpaceMouse input expired"):
        mailbox.check()


def test_expired_queued_heartbeat_cannot_extend_control_lease():
    now = [1.0]
    mailbox = worker.InputMailbox(worker.Settings(), clock=lambda: now[0])
    mailbox.accept({"op": "heartbeat", "sent_monotonic_s": 0.0})
    mailbox.activate()
    with pytest.raises(RuntimeError, match="heartbeat expired"):
        mailbox.check()
    mailbox.accept({"op": "heartbeat", "sent_monotonic_s": now[0]})
    mailbox.check()


def test_fresh_stamped_parent_heartbeat_is_required_before_fci():
    mailbox = worker.InputMailbox(worker.Settings(command_timeout_s=0.06))
    receiver = worker.InputReceiver(None, mailbox)
    mailbox.accept({"op": "heartbeat", "sent_monotonic_s": time.monotonic() - 1})
    with pytest.raises(RuntimeError, match="No fresh recorder heartbeat"):
        receiver.begin_control()
    assert mailbox.watchdog_enabled is False
    mailbox.accept({"op": "heartbeat", "sent_monotonic_s": time.monotonic()})
    receiver.begin_control()
    assert mailbox.watchdog_enabled is True


def test_state_guard_catches_frozen_native_clock_even_when_python_cache_is_fresh():
    now = [1.0]
    guard = worker.StateGuard(worker.Settings(), clock=lambda: now[0])
    guard.check(telemetry(robot_time=4, sample_monotonic_s=1.0), active=True)
    now[0] = 1.11
    with pytest.raises(RuntimeError, match="clock stopped"):
        guard.check(telemetry(robot_time=4, sample_monotonic_s=1.11), active=True)


@pytest.mark.parametrize("change, message", [
    ({"sample_monotonic_s": 0}, "stale"),
    ({"robot_mode": "kReflex"}, "clear the condition"),
    ({"control_command_success_rate": 0.99}, "success rate"),
    ({"tau_J": [0.0]}, "wrong shape"),
    ({"dq": [float("nan")] * 7}, "invalid"),
])
def test_state_guard_checks_fci_and_coherent_native_state(change, message):
    with pytest.raises(RuntimeError, match=message):
        worker.StateGuard(worker.Settings()).check(telemetry(**change), active=True)


def test_idle_state_has_no_control_success_gate_before_control_starts():
    worker.StateGuard(worker.Settings()).check(
        telemetry(robot_mode="kIdle", control_command_success_rate=0), active=False
    )


class FakeRobot:
    def __init__(self, *, homing_failure=False, blocked_homing=False, frozen_clock=False):
        self.events = []
        self.is_connected = False
        self._otg_running = False
        self._arm = self
        self.homing_failure = homing_failure
        self.blocked_homing = blocked_homing
        self.frozen_clock = frozen_clock
        self.initial_robot_time = time.monotonic()
        self.release_move = threading.Event()
        self.actions = []

    def connect(self):
        self.events.append("connect")
        self.is_connected = True

    def move_to_start(self):
        self.events.append("move_to_start")
        if self.blocked_homing:
            assert self.release_move.wait(1)
        if self.homing_failure:
            raise RuntimeError("Robot reflex during homing")

    def start_arm_controller(self):
        self.events.append("controller")
        self._otg_running = True

    def send_action(self, act):
        self.events.append("action")
        self.actions.append(act)

    def stop_motion(self):
        self.events.append("stop")
        self.release_move.set()

    def disconnect(self):
        self.events.append("disconnect")
        self.is_connected = False

    def state(self):
        return telemetry(robot_time=self.initial_robot_time if self.frozen_clock else None)


class WorkerHarness:
    def __init__(self, robot, *, preflight=lambda: None, configuration=None):
        parent, child = socket.socketpair()
        self.parent = JsonChannel(parent, timeout_s=0.02)
        self.child = JsonChannel(child, timeout_s=0.02)
        self.ending = threading.Event()
        self.keep_input_alive = False
        self.result = []
        self.pump = threading.Thread(target=self._heartbeat, daemon=True)
        self.thread = threading.Thread(
            target=lambda: self.result.append(worker.run_worker(
                self.child, "unused.yaml", config=configuration or {},
                robot_factory=lambda _config, _path: robot, preflight=preflight,
                telemetry_reader=lambda robot: robot.state(),
            )), daemon=True
        )
        self.pump.start()
        self.thread.start()

    def _heartbeat(self):
        while not self.ending.is_set():
            try:
                self.parent.send(packet() if self.keep_input_alive else {
                    "op": "heartbeat", "sent_monotonic_s": time.monotonic()
                })
            except (OSError, EOFError):
                return
            self.ending.wait(0.01)

    def until(self, state):
        deadline = time.monotonic() + 1.5
        seen = []
        while time.monotonic() < deadline:
            try:
                message = self.parent.receive()
            except socket.timeout:
                continue
            seen.append(message)
            if message["state"] == state:
                return message, seen
        raise AssertionError(f"Worker did not emit {state}: {seen}")

    def close(self):
        self.ending.set()
        self.pump.join(0.2)
        if self.thread.is_alive():
            self.parent.send({"op": "stop"})
        self.thread.join(1)
        self.parent.close()
        self.child.close()
        assert not self.thread.is_alive()


def test_f_orders_connect_home_controller_and_consumes_each_delta_once():
    robot = FakeRobot()
    harness = WorkerHarness(robot)
    try:
        _status, statuses = harness.until("running")
        assert [m["state"] for m in statuses] == ["starting", "moving_to_start", "running"]
        assert robot.events[:3] == ["connect", "move_to_start", "controller"]
        harness.parent.send(packet(action=action(target_x=0.0004)))
        deadline = time.monotonic() + 0.15
        while not robot.actions and time.monotonic() < deadline:
            time.sleep(0.005)
        assert len(robot.actions) == 1
        harness.until("running")
        assert len(robot.actions) == 1
    finally:
        harness.close()
    assert harness.result == [0]
    assert robot.events[-1] == "disconnect"


def test_stop_cancels_native_homing_and_never_activates_teleoperation():
    robot = FakeRobot(blocked_homing=True)
    harness = WorkerHarness(robot)
    try:
        harness.until("moving_to_start")
        harness.parent.send({"op": "stop"})
        harness.until("stopped")
    finally:
        harness.close()
    assert harness.result == [0]
    assert "stop" in robot.events and "disconnect" in robot.events
    assert "controller" not in robot.events and not robot.actions


def test_preflight_failure_opens_no_fci_connection():
    robot = FakeRobot()

    def fail():
        raise RuntimeError("Thor needs PREEMPT_RT")

    harness = WorkerHarness(robot, preflight=fail)
    try:
        message, _ = harness.until("error")
        assert "PREEMPT_RT" in message["message"]
    finally:
        harness.close()
    assert not robot.events and harness.result == [1]


def test_homing_error_disconnects_then_fresh_activation_can_succeed():
    first = FakeRobot(homing_failure=True)
    failed = WorkerHarness(first)
    try:
        message, _ = failed.until("error")
        assert "reflex" in message["message"]
    finally:
        failed.close()
    assert "controller" not in first.events and "disconnect" in first.events
    second = FakeRobot()
    retried = WorkerHarness(second)
    try:
        retried.until("running")
        assert not second.actions
    finally:
        retried.close()
    assert failed.result == [1] and retried.result == [0]


def test_peer_loss_stops_native_controller_and_exits():
    robot = FakeRobot()
    harness = WorkerHarness(robot)
    harness.until("running")
    harness.ending.set()
    harness.pump.join(0.2)
    harness.parent.close()
    harness.thread.join(1)
    harness.child.close()
    assert not harness.thread.is_alive()
    assert harness.result == [1]
    assert "stop" in robot.events and robot.events[-1] == "disconnect"


def test_frozen_native_clock_faults_despite_live_heartbeat_and_actions():
    robot = FakeRobot(frozen_clock=True)
    harness = WorkerHarness(robot, configuration={"fr3_teleop": {"max_state_age_s": 0.03}})
    try:
        harness.keep_input_alive = True
        message, seen = harness.until("error")
        assert "clock stopped" in message["message"]
        assert not any(item["state"] == "running" for item in seen)
    finally:
        harness.close()
    assert harness.result == [1] and robot.events[-1] == "disconnect"


def test_startup_waits_for_full_native_window_and_valid_rate_before_input():
    robot = FakeRobot()
    start = [None]

    def state():
        now = time.monotonic()
        if "controller" in robot.events:
            start[0] = start[0] or now
        elapsed = now - start[0] if start[0] is not None else 0
        return telemetry(control_command_success_rate=0.0 if elapsed < 0.02 else 0.99 if elapsed < 0.12 else 1.0)

    robot.state = state
    harness = WorkerHarness(robot)
    try:
        harness.until("running")
        assert time.monotonic() - start[0] >= 0.12
        assert not robot.actions
    finally:
        harness.close()
    assert harness.result == [0]


def test_startup_rate_that_never_qualifies_stops_without_spacemouse_actions(monkeypatch):
    monkeypatch.setattr(worker, "CONTROL_READY_TIMEOUT_S", 0.15)
    robot = FakeRobot()
    robot.state = lambda: telemetry(control_command_success_rate=0.99)
    harness = WorkerHarness(robot)
    try:
        message, seen = harness.until("error")
        assert "success rate=0.99" in message["message"]
        assert not any(item["state"] == "running" for item in seen)
        assert not robot.actions
    finally:
        harness.close()
    assert harness.result == [1]


def test_parent_watchdog_cancels_a_homing_move_before_controller_activation():
    robot = FakeRobot(blocked_homing=True)
    harness = WorkerHarness(robot)
    try:
        harness.until("moving_to_start")
        harness.ending.set()
        harness.pump.join(0.2)
        message, _ = harness.until("error")
        assert "heartbeat expired" in message["message"]
    finally:
        harness.close()
    assert harness.result == [1]
    assert "controller" not in robot.events and robot.events[-1] == "disconnect"


def test_missing_input_stops_control_even_while_parent_heartbeats_continue():
    robot = FakeRobot()
    harness = WorkerHarness(robot)
    try:
        harness.until("running")
        message, _ = harness.until("error")
        assert "SpaceMouse input expired" in message["message"]
    finally:
        harness.close()
    assert harness.result == [1]
    assert "stop" in robot.events and robot.events[-1] == "disconnect"


def test_nonfinite_native_state_reports_specific_fault_through_ipc():
    robot = FakeRobot()
    robot.state = lambda: telemetry(tau_J=[math.nan] * 7)
    harness = WorkerHarness(robot)
    try:
        message, _ = harness.until("error")
        assert "tau_J" in message["message"] and "invalid" in message["message"]
        assert message["telemetry"] == {}
    finally:
        harness.close()
    assert harness.result == [1] and "controller" not in robot.events


@pytest.mark.parametrize("realtime_enforce", [True, False])
def test_native_driver_explicitly_selects_realtime_and_caches_one_state(monkeypatch, realtime_enforce):
    from lerobot.robots.franka_research3.backends import PandaPyArmDriver

    instances = []

    class Panda:
        def __init__(self, ip, **kwargs):
            self.ip, self.kwargs = ip, kwargs
            self.state = types.SimpleNamespace(
                **{key: np.arange(size, dtype=np.float64) for key, size in worker.STATE_SHAPES.items()},
                time=types.SimpleNamespace(to_sec=lambda: 4.2),
                robot_mode=types.SimpleNamespace(name="kIdle"), control_command_success_rate=0.0,
            )
            self.active_error = False
            instances.append(self)

        def get_state(self):
            return self.state

        def raise_error(self):
            if self.active_error:
                raise RuntimeError("native reflex")

    enforce = object()
    ignore = object()
    monkeypatch.setitem(sys.modules, "panda_py", types.SimpleNamespace(
        Panda=Panda, controllers=types.SimpleNamespace(),
        libfranka=types.SimpleNamespace(RealtimeConfig=types.SimpleNamespace(kEnforce=enforce, kIgnore=ignore)),
    ))
    driver = PandaPyArmDriver("192.168.1.206", realtime_enforce=realtime_enforce,
                              start_controller_on_connect=False, state_poll_frequency_hz=200)
    driver._start_state_reader = lambda: None
    driver.connect()
    try:
        assert instances[0].kwargs == {"realtime_config": enforce if realtime_enforce else ignore}
        first = driver.get_telemetry()
        instances[0].state.q[:] = 100
        instances[0].state.tau_J[:] = 200
        assert driver.get_telemetry()["q"] == first["q"]
        assert driver.get_telemetry()["tau_J"] == first["tau_J"]
        first["q"][0] = 900
        assert driver.get_telemetry()["q"][0] == 0
        assert first["robot_time_s"] == 4.2
        instances[0].active_error = True
        with pytest.raises(RuntimeError, match="native reflex"):
            driver.get_telemetry()
    finally:
        driver.disconnect()


@pytest.mark.parametrize("mode", ["enforce", "ignore"])
def test_check_cli_validates_prerequisites_without_calling_connect(monkeypatch, mode):
    events = []
    monkeypatch.setattr(worker, "load_config", lambda _path: {"fr3_teleop": {"realtime_mode": mode}})
    monkeypatch.setattr(worker, "check_realtime", lambda mode: events.append(mode))
    robot = types.SimpleNamespace(connect=lambda: pytest.fail("--check opened FCI"))
    monkeypatch.setattr(worker, "build_robot", lambda _config, _path: events.append("native+IK") or robot)
    assert worker.main(["--config-path", "unused.yaml", "--check"]) == 0
    assert events == [mode, "native+IK"]


def test_explicit_ignore_skips_host_rt_requirements(monkeypatch):
    monkeypatch.setattr(worker, "Path", lambda _path: pytest.fail("ignore checked RT kernel"))
    monkeypatch.setattr(worker.os, "sched_setscheduler", lambda *_args: pytest.fail("ignore changed scheduler"))
    worker.check_realtime("ignore")
    with pytest.raises(ValueError, match="Unknown"):
        worker.check_realtime("ignored")


def test_check_realtime_restores_scheduler_and_refuses_ordinary_kernel(monkeypatch):
    monkeypatch.setattr(worker, "Path", lambda _path: types.SimpleNamespace(
        is_file=lambda: True, read_text=lambda: "0\n"
    ))
    with pytest.raises(RuntimeError, match="PREEMPT_RT"):
        worker.check_realtime()
    monkeypatch.setattr(worker, "Path", lambda _path: types.SimpleNamespace(
        is_file=lambda: True, read_text=lambda: "1\n"
    ))
    monkeypatch.setattr(worker.os, "sched_getscheduler", lambda _pid: 0)
    monkeypatch.setattr(worker.os, "sched_getparam", lambda _pid: "previous")
    calls = []
    monkeypatch.setattr(worker.os, "sched_setscheduler", lambda *args: calls.append(args))
    worker.check_realtime()
    assert calls[0][1] == worker.os.SCHED_FIFO
    assert calls[-1] == (0, 0, "previous")


def test_unpatched_native_wheel_is_rejected_before_constructing_panda(monkeypatch):
    from lerobot.robots.franka_research3.backends import PandaPyArmDriver

    calls = []
    monkeypatch.setitem(sys.modules, "panda_py", types.SimpleNamespace(
        Panda=lambda *_args, **_kwargs: calls.append("FCI"),
        controllers=types.SimpleNamespace(), _core=types.SimpleNamespace(),
    ))
    driver = PandaPyArmDriver("192.168.1.206", require_no_automatic_recovery=True)
    with pytest.raises(RuntimeError, match="patched panda-py wheel"):
        driver.connect()
    assert not calls


@pytest.mark.parametrize("mode", ["enforce", "ignore"])
def test_worker_forces_isolated_gripper_cameras_and_native_safety_flags(monkeypatch, mode):
    from lerobot.robots.franka_research3.franka_research3 import FrankaResearch3

    monkeypatch.setitem(sys.modules, "panda_py", types.SimpleNamespace(
        _core=types.SimpleNamespace(
            FR3_NO_AUTOMATIC_ERROR_RECOVERY=True,
            _JOINT_LIMITS_LOWER=(-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973),
            _JOINT_LIMITS_UPPER=(2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973),
            _JOINT_POSITION_START=(0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4),
        )
    ))
    monkeypatch.setitem(sys.modules, "ruckig", types.SimpleNamespace())
    monkeypatch.setattr(FrankaResearch3, "_make_kinematics_driver", lambda _self: object())
    robot = worker.build_robot({"fr3_teleop": {"realtime_mode": mode}, "robot": {
        "type": "franka_research3", "robot_ip": "192.168.1.206",
        "urdf_path": "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper.urdf",
        "gripper_backend": "franka_hand", "allow_mock_gripper": True,
        "cameras": {"should_never_open": {"type": "realsense"}},
        "use_otg": True, "otg_min_position": list(worker.SAFE_JOINT_LOWER),
        "otg_max_position": list(worker.SAFE_JOINT_UPPER),
        "otg_max_velocity": [0.5] * 7, "otg_max_acceleration": [1.0] * 7,
        "otg_max_jerk": [1000.0] * 7,
    }}, "unused.yaml")
    assert robot.config.gripper_backend == "mock"
    assert robot.config.allow_mock_gripper is False
    assert not robot.cameras
    assert robot.config.arm_realtime_enforce is (mode == "enforce")
    assert robot.config.arm_require_no_automatic_recovery is True
    assert robot.config.arm_start_controller_on_connect is False
    assert robot.is_connected is False
    gripper = robot._make_gripper_driver()
    gripper.connect()
    gripper.set_position(0.3)
    assert gripper.get_position() == pytest.approx(0.3)
    gripper.disconnect()


def test_deferred_native_controller_does_not_run_otg_until_homing_is_complete(monkeypatch):
    from lerobot.robots.franka_research3.config_franka_research3 import FrankaResearch3Config
    from lerobot.robots.franka_research3.franka_research3 import FrankaResearch3
    from tests.robots.test_franka_research3 import DummyArmDriver, DummyKinematicsDriver, DummyOTGDriver

    events = []

    class DeferredArm(DummyArmDriver):
        def start_controller(self):
            events.append("controller")

        def move_to_start(self):
            events.append("home")
            super().move_to_start()

    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DeferredArm)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)

    def start_otg(self, _joints):
        events.append("OTG")
        self._otg_running = True

    monkeypatch.setattr(FrankaResearch3, "_start_otg_loop", start_otg)
    robot = FrankaResearch3(FrankaResearch3Config(
        urdf_path="unused.urdf", gripper_backend="mock", arm_start_controller_on_connect=False,
    ))
    try:
        robot.connect()
        assert events == [] and robot._otg_running is False
        robot.move_to_start()
        assert events == ["home"] and robot._otg_running is False
        robot.start_arm_controller()
        assert events == ["home", "controller", "OTG"] and robot._otg_running is True
    finally:
        robot.disconnect()


def test_active_native_errors_are_rejected_even_when_robot_mode_is_idle(monkeypatch):
    from lerobot.robots.franka_research3.backends import PandaPyArmDriver

    monkeypatch.setitem(sys.modules, "panda_py", types.SimpleNamespace(
        Panda=object, controllers=types.SimpleNamespace(),
    ))
    driver = PandaPyArmDriver("192.168.1.206")
    with pytest.raises(RuntimeError, match="Clear the error in Desk"):
        driver._assert_arm_accepts_control(types.SimpleNamespace(
            current_errors=True, robot_mode=types.SimpleNamespace(name="kIdle")
        ))


def test_realtime_permission_failure_restores_the_original_scheduler(monkeypatch):
    monkeypatch.setattr(worker, "Path", lambda _path: types.SimpleNamespace(
        is_file=lambda: True, read_text=lambda: "1\n"
    ))
    monkeypatch.setattr(worker.os, "sched_getscheduler", lambda _pid: 0)
    monkeypatch.setattr(worker.os, "sched_getparam", lambda _pid: "previous")
    calls = []

    def set_scheduler(pid, policy, parameters):
        calls.append((pid, policy, parameters))
        if policy == worker.os.SCHED_FIFO:
            raise PermissionError("rtprio limit")

    monkeypatch.setattr(worker.os, "sched_setscheduler", set_scheduler)
    with pytest.raises(RuntimeError, match="SCHED_FIFO permission"):
        worker.check_realtime()
    assert calls[-1] == (0, 0, "previous")


def test_native_ownership_lock_blocks_other_process_and_releases_on_owner_exit(tmp_path):
    ownership = worker.ArmOwnershipLock("192.168.1.206", lock_dir=tmp_path)
    script = """
import fcntl, os, sys
fd = os.open(sys.argv[1], os.O_RDWR)
try:
    fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
except BlockingIOError:
    sys.exit(23)
os.close(fd)
"""
    with ownership:
        result = subprocess.run([sys.executable, "-c", script, str(ownership.path)], check=False)
        assert result.returncode == 23
        with pytest.raises(RuntimeError, match="Another local FR3 worker"):
            with worker.ArmOwnershipLock("192.168.1.206", lock_dir=tmp_path):
                pytest.fail("Second owner was allowed")
    result = subprocess.run([sys.executable, "-c", script, str(ownership.path)], check=False)
    assert result.returncode == 0


def test_worker_refuses_disabled_otg_and_bounds_outside_conservative_envelope(monkeypatch):
    monkeypatch.setitem(sys.modules, "panda_py", types.SimpleNamespace(
        _core=types.SimpleNamespace(FR3_NO_AUTOMATIC_ERROR_RECOVERY=True)
    ))
    monkeypatch.setitem(sys.modules, "ruckig", types.SimpleNamespace())
    base = {"robot_ip": "192.168.1.206", "urdf_path": "unused.urdf",
            "otg_min_position": list(worker.SAFE_JOINT_LOWER),
            "otg_max_position": list(worker.SAFE_JOINT_UPPER)}
    with pytest.raises(ValueError, match="use_otg=true"):
        worker.build_robot({"robot": {**base, "use_otg": False}}, "unused.yaml")
    wider = list(worker.SAFE_JOINT_UPPER)
    wider[5] = 4.0
    with pytest.raises(ValueError, match="conservative native/FR3 envelope"):
        worker.build_robot({"robot": {**base, "use_otg": True, "otg_max_position": wider}}, "unused.yaml")


def test_narrowed_joint_bounds_must_include_native_homing_target(monkeypatch):
    monkeypatch.setitem(sys.modules, "panda_py", types.SimpleNamespace(_core=types.SimpleNamespace(
        FR3_NO_AUTOMATIC_ERROR_RECOVERY=True,
        _JOINT_LIMITS_LOWER=(-2.8973, -1.7628, -2.8973, -3.0718, -2.8973, -0.0175, -2.8973),
        _JOINT_LIMITS_UPPER=(2.8973, 1.7628, 2.8973, -0.0698, 2.8973, 3.7525, 2.8973),
        _JOINT_POSITION_START=(0, -math.pi / 4, 0, -3 * math.pi / 4, 0, math.pi / 2, math.pi / 4),
    )))
    monkeypatch.setitem(sys.modules, "ruckig", types.SimpleNamespace())
    lower = list(worker.SAFE_JOINT_LOWER)
    lower[5] = 2.0
    with pytest.raises(ValueError, match="must contain the native move_to_start"):
        worker.build_robot({"robot": {
            "robot_ip": "192.168.1.206", "urdf_path": "unused.urdf", "use_otg": True,
            "otg_min_position": lower, "otg_max_position": list(worker.SAFE_JOINT_UPPER),
        }}, "unused.yaml")


def test_measured_joint_outside_conservative_envelope_stops_before_homing():
    state = telemetry()
    state["q"][5] = 4.0
    with pytest.raises(RuntimeError, match="Use Desk"):
        worker.StateGuard(worker.Settings()).check(state, active=False)


def test_backend_refuses_nonfinite_or_out_of_bounds_native_joint_commands(monkeypatch):
    from lerobot.robots.franka_research3.backends import PandaPyArmDriver

    monkeypatch.setitem(sys.modules, "panda_py", types.SimpleNamespace(Panda=object, controllers=object))
    driver = PandaPyArmDriver("192.168.1.206", joint_position_min=worker.SAFE_JOINT_LOWER,
                              joint_position_max=worker.SAFE_JOINT_UPPER)
    commands = []
    driver._controller = types.SimpleNamespace(set_control=lambda q: commands.append(q))
    valid = np.asarray(telemetry()["q"])
    driver.set_joint_positions(valid)
    assert len(commands) == 1
    invalid = valid.copy()
    invalid[5] = 4.0
    with pytest.raises(RuntimeError, match="conservative bounds"):
        driver.set_joint_positions(invalid)
    invalid[5] = math.nan
    with pytest.raises(ValueError, match="finite"):
        driver.set_joint_positions(invalid)
    assert len(commands) == 1


@pytest.mark.parametrize("key, value", [
    ("otg_max_velocity", 1000.0), ("otg_max_acceleration", 1000.0),
    ("otg_max_jerk", 10000.0), ("otg_max_velocity", -1.0),
    ("otg_max_acceleration", math.nan), ("otg_max_jerk", math.inf),
])
def test_excessive_or_invalid_dynamic_limits_fail_before_native_initialization(key, value):
    with pytest.raises(ValueError, match="initial Thor preset"):
        worker.Settings.from_config({"robot": {key: [value] * 7}})
