#!/usr/bin/env python

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

from pathlib import Path
import threading
import numpy as np
import pytest
import struct
import sys
import time
import types

from lerobot.robots.franka_research3 import FrankaResearch3, FrankaResearch3Config
from lerobot.robots.franka_research3 import backends as fr3_backends
from lerobot.robots.franka_research3.backends import (
    FrankaHandGripperHardwareDriver,
    PandaPyArmDriver,
    PikaGripperHardwareDriver,
)
from lerobot.robots.franka_research3.processor_franka_research3 import (
    AbsoluteEEActionToRobotAction,
    DeltaActionToAbsoluteEEAction,
    KeepAbsoluteEEObservation,
)
from lerobot.processor import RobotProcessorPipeline
from lerobot.processor.converters import (
    observation_to_transition,
    robot_action_observation_to_transition,
    transition_to_observation,
    transition_to_robot_action,
)
from lerobot.utils.rotation import Rotation
import json


class DummyArmDriver:
    instances: list["DummyArmDriver"] = []

    def __init__(self, *args, **kwargs):
        del args, kwargs
        type(self).instances.append(self)
        self.connected = False
        self.joint_positions = np.array([0.1, 0.2, 0.3, -1.0, 0.5, 1.2, -0.7], dtype=np.float64)
        self.set_joint_positions_calls: list[np.ndarray] = []
        self.move_to_start_calls = 0
        self.get_joint_positions_calls = 0

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_joint_positions(self) -> np.ndarray:
        self.get_joint_positions_calls += 1
        return self.joint_positions.copy()

    def get_ee_pose(self) -> np.ndarray | None:
        return None

    def set_joint_positions(self, joint_positions: np.ndarray) -> None:
        self.set_joint_positions_calls.append(np.asarray(joint_positions, dtype=np.float64))

    def move_to_start(self) -> None:
        self.move_to_start_calls += 1
        self.joint_positions = np.array([0.0, -0.5, 0.0, -2.2, 0.0, 1.8, 0.7], dtype=np.float64)


class NoMoveToStartArmDriver(DummyArmDriver):
    move_to_start = None


class UpdatingJointArmDriver(DummyArmDriver):
    def set_joint_positions(self, joint_positions: np.ndarray) -> None:
        super().set_joint_positions(joint_positions)
        self.joint_positions = np.asarray(joint_positions, dtype=np.float64).copy()


class ReportingArmDriver(DummyArmDriver):
    def get_ee_pose(self) -> np.ndarray | None:
        pose = np.eye(4, dtype=np.float64)
        pose[:3, 3] = np.array([9.0, 8.0, 7.0], dtype=np.float64)
        return pose


class DummyGripperDriver:
    instances: list["DummyGripperDriver"] = []

    def __init__(self, *args, **kwargs):
        del args
        type(self).instances.append(self)
        self.connected = False
        self.init_kwargs = kwargs
        self.position = 0.25
        self.set_position_calls: list[float] = []

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def get_position(self) -> float:
        return self.position

    def set_position(self, normalized_position: float) -> None:
        self.set_position_calls.append(normalized_position)
        self.position = normalized_position


class FailingGripperDriver(DummyGripperDriver):
    def connect(self) -> None:
        self.connected = True
        raise RuntimeError("gripper connect failed")


class DummyKinematicsDriver:
    def __init__(self, *args, **kwargs):
        del args, kwargs
        self.forward_pose = np.eye(4, dtype=np.float64)
        self.forward_pose[:3, 3] = np.array([0.4, 0.1, 0.3], dtype=np.float64)
        self.inverse_calls: list[tuple[np.ndarray, np.ndarray]] = []
        self.inverse_kwargs: list[dict[str, float]] = []
        self.inverse_solution = np.array([0.5, 0.4, 0.3, -0.2, 0.1, 0.0, -0.1], dtype=np.float64)

    def forward_kinematics(self, joint_positions_rad: np.ndarray) -> np.ndarray:
        del joint_positions_rad
        return self.forward_pose.copy()

    def inverse_kinematics(
        self,
        current_joint_positions_rad: np.ndarray,
        desired_pose: np.ndarray,
        **kwargs,
    ) -> np.ndarray:
        self.inverse_calls.append((current_joint_positions_rad.copy(), desired_pose.copy()))
        self.inverse_kwargs.append(dict(kwargs))
        return self.inverse_solution.copy()


class DummyOTGDriver:
    instances: list["DummyOTGDriver"] = []

    def __init__(self, *args, **kwargs):
        del args, kwargs
        type(self).instances.append(self)
        self.reset_calls: list[np.ndarray] = []
        self.step_calls: list[tuple[np.ndarray, np.ndarray]] = []

    def reset(self, current_joint_positions: np.ndarray) -> None:
        self.reset_calls.append(np.asarray(current_joint_positions, dtype=np.float64))

    def step(self, current_joint_positions: np.ndarray, target_joint_positions: np.ndarray) -> np.ndarray:
        current = np.asarray(current_joint_positions, dtype=np.float64)
        target = np.asarray(target_joint_positions, dtype=np.float64)
        self.step_calls.append((current.copy(), target.copy()))
        return target.copy()


class FailingOTGDriver(DummyOTGDriver):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        raise RuntimeError("otg init failed")


@pytest.fixture
def robot(monkeypatch):
    DummyArmDriver.instances = []
    DummyGripperDriver.instances = []
    DummyOTGDriver.instances = []
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    cfg = FrankaResearch3Config(
        robot_ip="192.168.1.206",
        gripper_port="/dev/ttyUSB80",
        urdf_path="/tmp/fr3.urdf",
        workspace_min=(0.2, -0.3, 0.2),
        workspace_max=(0.6, 0.3, 0.5),
    )
    device = FrankaResearch3(cfg)
    yield device
    if device.is_connected:
        device.disconnect()


def test_connect_disconnect(robot):
    assert not robot.is_connected
    robot.connect()
    assert robot.is_connected
    assert len(DummyOTGDriver.instances) == 1
    assert len(DummyOTGDriver.instances[-1].reset_calls) == 1
    robot.disconnect()
    assert not robot.is_connected


def test_capture_current_start_joint_positions_uses_cached_observation(robot):
    robot.connect()
    arm = DummyArmDriver.instances[-1]
    cached_joints = arm.joint_positions.copy()

    robot.get_observation(include_cameras=False)
    arm.joint_positions = np.ones(7, dtype=np.float64)

    captured = robot.capture_current_start_joint_positions(require_cached=True)

    assert np.allclose(captured, cached_joints)
    assert robot.config.start_joint_positions == tuple(float(value) for value in cached_joints)


def test_pandapy_arm_driver_connect_seeds_controller_with_current_joints(monkeypatch):
    class DummyJointPositionController:
        def __init__(self):
            self.set_control_calls = []
            self.set_damping_calls = []
            self.set_stiffness_calls = []
            self.set_filter_calls = []

        def set_control(self, joint_positions):
            self.set_control_calls.append(np.asarray(joint_positions, dtype=np.float64))

        def set_damping(self, damping):
            self.set_damping_calls.append(damping)

        def set_stiffness(self, stiffness):
            self.set_stiffness_calls.append(stiffness)

        def set_filter(self, coeff):
            self.set_filter_calls.append(coeff)

    class DummyPanda:
        instances: list["DummyPanda"] = []

        def __init__(self, robot_ip):
            self.robot_ip = robot_ip
            self.started_controllers = []
            self.stopped = 0
            self.state = types.SimpleNamespace(q=np.array([0.1, 0.2, 0.3, -1.0, 0.5, 1.2, -0.7], dtype=np.float64))
            self.get_state_calls = 0
            type(self).instances.append(self)

        def start_controller(self, controller):
            self.started_controllers.append(controller)

        def stop_controller(self):
            self.stopped += 1

        def get_state(self):
            self.get_state_calls += 1
            return self.state

    fake_module = types.SimpleNamespace(
        Panda=DummyPanda,
        controllers=types.SimpleNamespace(JointPosition=DummyJointPositionController),
    )
    monkeypatch.setitem(sys.modules, "panda_py", fake_module)

    driver = PandaPyArmDriver(robot_ip="192.168.1.206")
    driver.connect()

    controller = driver._controller
    assert controller is not None
    assert len(controller.set_control_calls) == 1
    assert np.allclose(controller.set_control_calls[0], DummyPanda.instances[-1].state.q)
    assert DummyPanda.instances[-1].started_controllers == [controller]

    # This driver polls at the 200 Hz default, so its reader thread outlives the test unless it
    # is stopped -- and a stray thread inside `backends` corrupts any later test that patches a
    # module-level global there.
    driver.disconnect()


@pytest.mark.parametrize(
    ("mode_name", "expected_message"),
    [
        ("kUserStopped", "user-stop"),
        ("kReflex", "reflex"),
        ("kGuiding", "guiding"),
    ],
)
def test_pandapy_arm_driver_connect_refuses_modes_that_cannot_start_a_controller(
    monkeypatch, mode_name, expected_message
):
    """`start_controller()` takes no timeout and waits on a loop these modes never start.

    Blocking there is silent and uninterruptible -- the recorder emits nothing and stops reading
    its own stdin -- so the mode has to be refused before the call, with the remedy named.
    """

    class DummyJointPositionController:
        def set_control(self, joint_positions):
            del joint_positions

    class DummyPanda:
        instances: list["DummyPanda"] = []

        def __init__(self, robot_ip):
            self.robot_ip = robot_ip
            self.state = types.SimpleNamespace(
                q=np.zeros(7, dtype=np.float64),
                robot_mode=types.SimpleNamespace(name=mode_name),
            )
            self.started_controllers = []
            type(self).instances.append(self)

        def start_controller(self, controller):
            self.started_controllers.append(controller)

        def stop_controller(self):
            return None

        def get_state(self):
            return self.state

    monkeypatch.setitem(
        sys.modules,
        "panda_py",
        types.SimpleNamespace(
            Panda=DummyPanda,
            controllers=types.SimpleNamespace(JointPosition=DummyJointPositionController),
        ),
    )

    driver = PandaPyArmDriver(robot_ip="192.168.1.206", state_poll_frequency_hz=0.0)
    with pytest.raises(RuntimeError, match=expected_message):
        driver.connect()
    assert DummyPanda.instances[-1].started_controllers == []


def test_pandapy_arm_driver_connect_accepts_a_controllable_mode(monkeypatch):
    """A reported mode that *can* take control must not be turned into a refusal."""

    class DummyJointPositionController:
        def set_control(self, joint_positions):
            del joint_positions

    class DummyPanda:
        instances: list["DummyPanda"] = []

        def __init__(self, robot_ip):
            self.robot_ip = robot_ip
            self.state = types.SimpleNamespace(
                q=np.zeros(7, dtype=np.float64),
                robot_mode=types.SimpleNamespace(name="kIdle"),
            )
            self.started_controllers = []
            type(self).instances.append(self)

        def start_controller(self, controller):
            self.started_controllers.append(controller)

        def stop_controller(self):
            return None

        def get_state(self):
            return self.state

    monkeypatch.setitem(
        sys.modules,
        "panda_py",
        types.SimpleNamespace(
            Panda=DummyPanda,
            controllers=types.SimpleNamespace(JointPosition=DummyJointPositionController),
        ),
    )

    driver = PandaPyArmDriver(robot_ip="192.168.1.206", state_poll_frequency_hz=0.0)
    driver.connect()

    assert DummyPanda.instances[-1].started_controllers == [driver._controller]


def test_pandapy_arm_driver_get_joint_positions_uses_cached_state(monkeypatch):
    class DummyJointPositionController:
        def set_control(self, joint_positions):
            del joint_positions

    class DummyPanda:
        instances: list["DummyPanda"] = []

        def __init__(self, robot_ip):
            self.robot_ip = robot_ip
            self.state = types.SimpleNamespace(q=np.array([0.1, 0.2, 0.3, -1.0, 0.5, 1.2, -0.7], dtype=np.float64))
            self.get_state_calls = 0
            type(self).instances.append(self)

        def start_controller(self, controller):
            del controller

        def stop_controller(self):
            return None

        def get_state(self):
            self.get_state_calls += 1
            return self.state

    fake_module = types.SimpleNamespace(
        Panda=DummyPanda,
        controllers=types.SimpleNamespace(JointPosition=DummyJointPositionController),
    )
    monkeypatch.setitem(sys.modules, "panda_py", fake_module)

    driver = PandaPyArmDriver(robot_ip="192.168.1.206", state_poll_frequency_hz=0.0)
    driver.connect()

    first = driver.get_joint_positions()
    second = driver.get_joint_positions()

    assert np.allclose(first, DummyPanda.instances[-1].state.q)
    assert np.allclose(second, DummyPanda.instances[-1].state.q)
    assert DummyPanda.instances[-1].get_state_calls == 1
    driver.disconnect()


def test_pika_gripper_hardware_driver_deduplicates_and_rate_limits(monkeypatch):
    class FakeSDKGripper:
        instances: list["FakeSDKGripper"] = []

        def __init__(self, port):
            self.port = port
            self.set_gripper_distance_calls: list[float] = []
            type(self).instances.append(self)

        def connect(self):
            return True

        def enable(self):
            return True

        def disable(self):
            return True

        def disconnect(self):
            pass

        def get_gripper_distance(self):
            return 0.0

        def set_gripper_distance(self, width_mm):
            self.set_gripper_distance_calls.append(float(width_mm))

    monkeypatch.setitem(sys.modules, "pika.gripper", types.SimpleNamespace(Gripper=FakeSDKGripper))
    perf_counter_values = iter([0.0, 0.01, 0.05, 0.11])
    monkeypatch.setattr(fr3_backends.time, "perf_counter", lambda: next(perf_counter_values))

    driver = PikaGripperHardwareDriver(
        serial_port="/dev/ttyUSB80",
        max_width_mm=100.0,
        command_rate_limit_hz=10.0,
        command_deadband_mm=0.5,
    )
    driver.connect()

    driver.set_position(0.2)
    driver.set_position(0.2)
    driver.set_position(0.4)
    driver.set_position(0.4)
    driver.set_position(0.4)

    assert FakeSDKGripper.instances[-1].set_gripper_distance_calls == [20.0, 40.0]


def _make_fake_pika_sdk(latest_data: dict, voltage: float = 0.0):
    """Fake Gripper whose connect()/enable() succeed like the real SDK's always do."""

    class FakeSDKGripper:
        instances: list["FakeSDKGripper"] = []

        def __init__(self, port):
            self.port = port
            self.serial_comm = types.SimpleNamespace(latest_data=dict(latest_data))
            self.motor_status = {"Voltage": voltage}
            self.disconnected = False
            type(self).instances.append(self)

        def connect(self):
            return True

        def enable(self):
            return True

        def disable(self):
            return True

        def disconnect(self):
            self.disconnected = True

        def get_gripper_distance(self):
            return 0.0

        def set_gripper_distance(self, width_mm):
            del width_mm

    return FakeSDKGripper


def test_pika_gripper_connect_rejects_a_port_with_no_gripper_telemetry(monkeypatch):
    # Gripper.connect() only opens the serial port and enable() only writes bytes, so
    # both succeed against any serial device. Without a parsed frame the readback stays
    # at the SDK's initial 0.0 and commands are silently dropped, so connect must fail.
    fake_cls = _make_fake_pika_sdk(latest_data={})
    monkeypatch.setitem(sys.modules, "pika.gripper", types.SimpleNamespace(Gripper=fake_cls))

    driver = PikaGripperHardwareDriver(
        serial_port="/dev/ttyUSB80",
        enable_settle_s=0.0,
        telemetry_timeout_s=0.1,
    )

    with pytest.raises(ConnectionError, match="no Pika gripper telemetry"):
        driver.connect()
    assert fake_cls.instances[-1].disconnected is True


def test_pika_gripper_connect_accepts_a_port_streaming_motor_frames(monkeypatch):
    fake_cls = _make_fake_pika_sdk(latest_data={"motor": {"Position": 1.2}})
    monkeypatch.setitem(sys.modules, "pika.gripper", types.SimpleNamespace(Gripper=fake_cls))

    driver = PikaGripperHardwareDriver(
        serial_port="/dev/ttyUSB80",
        enable_settle_s=0.0,
        telemetry_timeout_s=0.1,
    )
    driver.connect()

    assert driver.has_telemetry() is True
    driver.disconnect()


def test_pika_gripper_connect_accepts_bus_voltage_as_proof_of_life(monkeypatch):
    # Some frames carry motorstatus without motor; a non-zero bus voltage still proves
    # the link is really talking to a powered gripper.
    fake_cls = _make_fake_pika_sdk(latest_data={"motorstatus": {}}, voltage=24.0)
    monkeypatch.setitem(sys.modules, "pika.gripper", types.SimpleNamespace(Gripper=fake_cls))

    driver = PikaGripperHardwareDriver(
        serial_port="/dev/ttyUSB80",
        enable_settle_s=0.0,
        telemetry_timeout_s=0.1,
    )
    driver.connect()

    assert driver.has_telemetry() is True
    driver.disconnect()


def test_pika_gripper_skips_the_telemetry_gate_when_it_cannot_introspect(monkeypatch):
    # A stand-in SDK object exposes neither serial_comm nor motor_status; the gate is a
    # diagnostic and must fail open rather than reject a working gripper.
    class BareSDKGripper:
        def __init__(self, port):
            self.port = port

        def connect(self):
            return True

        def enable(self):
            return True

        def disable(self):
            return True

        def disconnect(self):
            pass

        def get_gripper_distance(self):
            return 0.0

        def set_gripper_distance(self, width_mm):
            del width_mm

    monkeypatch.setitem(sys.modules, "pika.gripper", types.SimpleNamespace(Gripper=BareSDKGripper))

    driver = PikaGripperHardwareDriver(
        serial_port="/dev/ttyUSB80",
        enable_settle_s=0.0,
        telemetry_timeout_s=0.1,
    )
    driver.connect()

    assert driver.has_telemetry() is True
    driver.disconnect()


def test_pika_gripper_hardware_driver_skips_small_target_changes(monkeypatch):
    class FakeSDKGripper:
        instances: list["FakeSDKGripper"] = []

        def __init__(self, port):
            self.port = port
            self.set_gripper_distance_calls: list[float] = []
            type(self).instances.append(self)

        def connect(self):
            return True

        def enable(self):
            return True

        def disable(self):
            return True

        def disconnect(self):
            pass

        def get_gripper_distance(self):
            return 0.0

        def set_gripper_distance(self, width_mm):
            self.set_gripper_distance_calls.append(float(width_mm))

    monkeypatch.setitem(sys.modules, "pika.gripper", types.SimpleNamespace(Gripper=FakeSDKGripper))
    perf_counter_values = iter([0.0, 1.0])
    monkeypatch.setattr(fr3_backends.time, "perf_counter", lambda: next(perf_counter_values))

    driver = PikaGripperHardwareDriver(
        serial_port="/dev/ttyUSB80",
        max_width_mm=100.0,
        command_rate_limit_hz=None,
        command_deadband_mm=0.5,
    )
    driver.connect()

    driver.set_position(0.200)
    driver.set_position(0.203)

    assert FakeSDKGripper.instances[-1].set_gripper_distance_calls == [20.0]


def test_das_gripper_hardware_driver_reports_normalized_position_and_rate_limits(monkeypatch):
    class FakeDataBus:
        instances: list["FakeDataBus"] = []

        def __init__(self, tty_port, baudrate, encoder_freq, encoder_callback, tactile_freq=None, tactile_callback=None):
            self.tty_port = tty_port
            self.baudrate = baudrate
            self.encoder_freq = encoder_freq
            self.encoder_callback = encoder_callback
            self.tactile_freq = tactile_freq
            self.tactile_callback = tactile_callback
            self.set_target_distance_calls: list[float] = []
            self.stopped = False
            type(self).instances.append(self)
            self.encoder_callback(struct.pack(">f", 0.0206))

        def set_target_distance(self, distance_m):
            self.set_target_distance_calls.append(float(distance_m))

        def stop(self):
            self.stopped = True

    fake_module = types.ModuleType("gen_controller_sdk_python")
    fake_module.DataBus = FakeDataBus
    monkeypatch.setitem(sys.modules, "gen_controller_sdk_python", fake_module)
    perf_counter_values = iter([0.0, 0.01, 0.05, 0.11])
    monkeypatch.setattr(fr3_backends.time, "perf_counter", lambda: next(perf_counter_values))

    driver = fr3_backends.DasGripperHardwareDriver(
        serial_port="/dev/ttyUSB0",
        baudrate=921600,
        update_frequency_hz=50.0,
        min_distance_m=0.0,
        max_distance_m=0.103,
        initial_position=0.2,
        command_rate_limit_hz=10.0,
        command_deadband_m=1e-4,
    )
    driver.connect()

    assert driver.get_position() == pytest.approx(0.2, abs=1e-3)
    driver.set_position(0.2)
    driver.set_position(0.4)
    driver.set_position(0.4)

    assert FakeDataBus.instances[-1].set_target_distance_calls == pytest.approx([0.0206, 0.0412])

    driver.disconnect()
    assert FakeDataBus.instances[-1].stopped is True


def test_franka_hand_gripper_hardware_driver_reports_normalized_position_and_uses_grasp_when_closing(monkeypatch):
    class FakeFrankaHandState:
        def __init__(self, width=0.08, max_width=0.08):
            self.width = width
            self.max_width = max_width

    class FakeFrankaHand:
        instances: list["FakeFrankaHand"] = []

        def __init__(self, robot_ip):
            self.robot_ip = robot_ip
            self.move_calls: list[tuple[float, float]] = []
            self.grasp_calls: list[tuple[float, float, float, float, float]] = []
            self.stop_calls = 0
            self.homing_calls = 0
            self.state = FakeFrankaHandState()
            type(self).instances.append(self)

        def homing(self):
            self.homing_calls += 1
            return True

        def read_once(self):
            return self.state

        def move(self, width_m, speed_m_s):
            self.move_calls.append((float(width_m), float(speed_m_s)))
            self.state.width = float(width_m)
            return True

        def grasp(self, width_m, speed_m_s, force_n, epsilon_inner_m, epsilon_outer_m):
            self.grasp_calls.append(
                (
                    float(width_m),
                    float(speed_m_s),
                    float(force_n),
                    float(epsilon_inner_m),
                    float(epsilon_outer_m),
                )
            )
            self.state.width = float(width_m)
            return True

        def stop(self):
            self.stop_calls += 1

    fake_module = types.SimpleNamespace(libfranka=types.SimpleNamespace(Gripper=FakeFrankaHand))
    monkeypatch.setitem(sys.modules, "panda_py", fake_module)

    driver = FrankaHandGripperHardwareDriver(
        robot_ip="192.168.1.206",
        command_rate_limit_hz=None,
        command_deadband_m=5e-4,
    )
    driver.connect()

    assert driver.get_position() == pytest.approx(1.0)
    driver.set_position(0.5)

    deadline = time.perf_counter() + 0.2
    while not FakeFrankaHand.instances[-1].move_calls and time.perf_counter() < deadline:
        time.sleep(0.005)

    driver.set_position(0.0)
    driver.set_position(0.0)

    deadline = time.perf_counter() + 0.2
    while not FakeFrankaHand.instances[-1].grasp_calls and time.perf_counter() < deadline:
        time.sleep(0.005)

    assert FakeFrankaHand.instances[-1].move_calls == [(0.08, 0.05)]
    assert FakeFrankaHand.instances[-1].grasp_calls == [(0.0, 0.05, 20.0, 0.005, 0.005)]

    driver.disconnect()
    assert FakeFrankaHand.instances[-1].stop_calls == 1


def test_franka_hand_gripper_hardware_driver_caches_state_for_get_position(monkeypatch):
    class FakeFrankaHandState:
        def __init__(self, width=0.04, max_width=0.08):
            self.width = width
            self.max_width = max_width

    class FakeFrankaHand:
        instances: list["FakeFrankaHand"] = []

        def __init__(self, robot_ip):
            self.robot_ip = robot_ip
            self.homing_calls = 0
            self.stop_calls = 0
            self.read_once_calls = 0
            self.state = FakeFrankaHandState()
            type(self).instances.append(self)

        def homing(self):
            self.homing_calls += 1
            return True

        def read_once(self):
            self.read_once_calls += 1
            return self.state

        def stop(self):
            self.stop_calls += 1

    fake_module = types.SimpleNamespace(libfranka=types.SimpleNamespace(Gripper=FakeFrankaHand))
    monkeypatch.setitem(sys.modules, "panda_py", fake_module)

    driver = FrankaHandGripperHardwareDriver(
        robot_ip="192.168.1.206",
        state_poll_frequency_hz=1.0,
    )
    driver.connect()

    fake_gripper = FakeFrankaHand.instances[-1]
    assert fake_gripper.read_once_calls == 1
    assert driver.get_position() == pytest.approx(0.5)
    assert driver.get_position() == pytest.approx(0.5)
    assert fake_gripper.read_once_calls == 1

    driver.disconnect()
    assert fake_gripper.stop_calls == 1


def test_franka_hand_gripper_hardware_driver_set_position_is_async(monkeypatch):
    command_started = threading.Event()
    command_release = threading.Event()

    class FakeFrankaHandState:
        def __init__(self, width=0.08, max_width=0.08):
            self.width = width
            self.max_width = max_width

    class FakeFrankaHand:
        instances: list["FakeFrankaHand"] = []

        def __init__(self, robot_ip):
            self.robot_ip = robot_ip
            self.state = FakeFrankaHandState()
            self.stop_calls = 0
            type(self).instances.append(self)

        def homing(self):
            return True

        def read_once(self):
            return self.state

        def move(self, width_m, speed_m_s):
            del width_m, speed_m_s
            command_started.set()
            command_release.wait(timeout=1.0)
            return True

        def grasp(self, width_m, speed_m_s, force_n, epsilon_inner_m, epsilon_outer_m):
            del width_m, speed_m_s, force_n, epsilon_inner_m, epsilon_outer_m
            command_started.set()
            command_release.wait(timeout=1.0)
            return True

        def stop(self):
            self.stop_calls += 1
            command_release.set()

    fake_module = types.SimpleNamespace(libfranka=types.SimpleNamespace(Gripper=FakeFrankaHand))
    monkeypatch.setitem(sys.modules, "panda_py", fake_module)

    driver = FrankaHandGripperHardwareDriver(
        robot_ip="192.168.1.206",
        command_rate_limit_hz=None,
        state_poll_frequency_hz=0.0,
    )
    driver.connect()

    start_t = time.perf_counter()
    driver.set_position(1.0)
    elapsed_s = time.perf_counter() - start_t

    assert elapsed_s < 0.05
    assert command_started.wait(timeout=0.2) is True

    command_release.set()
    driver.disconnect()


def test_das_gripper_hardware_driver_decodes_tactile_and_applies_clean_rule(tmp_path, monkeypatch):
    mask_payload = json.loads(
        (Path(__file__).resolve().parents[2] / "docs/tactile/tactile_valid_mask_50x10.json").read_text(encoding="utf-8")
    )
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(json.dumps(mask_payload), encoding="utf-8")

    mask_rows = mask_payload["mask"]
    baseline_flat = [255.0 if int(mask_rows[row][col]) == 0 else 0.0 for row in range(50) for col in range(10)]
    baseline_payload = {
        "data": [
            {
                "tactiles": {
                    "left": baseline_flat,
                    "right": baseline_flat,
                }
            }
        ]
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    left_valid = bytes([index % 256 for index in range(448)])
    right_valid = bytes([(255 - index) % 256 for index in range(448)])

    class FakeDataBus:
        instances: list["FakeDataBus"] = []

        def __init__(self, tty_port, baudrate, encoder_freq, encoder_callback, tactile_freq=None, tactile_callback=None):
            self.tty_port = tty_port
            self.baudrate = baudrate
            self.encoder_freq = encoder_freq
            self.encoder_callback = encoder_callback
            self.tactile_freq = tactile_freq
            self.tactile_callback = tactile_callback
            self.set_target_distance_calls: list[float] = []
            self.stopped = False
            type(self).instances.append(self)
            self.encoder_callback(struct.pack(">f", 0.0206))
            if self.tactile_callback is not None:
                self.tactile_callback(left_valid + right_valid)

        def set_target_distance(self, distance_m):
            self.set_target_distance_calls.append(float(distance_m))

        def stop(self):
            self.stopped = True

    fake_module = types.ModuleType("gen_controller_sdk_python")
    fake_module.DataBus = FakeDataBus
    monkeypatch.setitem(sys.modules, "gen_controller_sdk_python", fake_module)

    driver = fr3_backends.DasGripperHardwareDriver(
        serial_port="/dev/ttyUSB0",
        baudrate=921600,
        update_frequency_hz=50.0,
        tactile_frequency_hz=30.0,
        tactile_valid_mask_path=str(mask_path),
        tactile_baseline_path=str(baseline_path),
        min_distance_m=0.0,
        max_distance_m=0.103,
        initial_position=0.2,
    )
    driver.connect()

    observation = driver.get_tactile_observation()

    assert observation["observation.tactile.left_raw"].shape == (50, 10)
    assert observation["observation.tactile.right_raw"].shape == (50, 10)
    assert observation["observation.tactile.valid_mask"].shape == (50, 10)
    assert int(observation["observation.tactile.valid_mask"].sum()) == 448
    assert float(observation["observation.tactile.left_raw"][0, 0]) == pytest.approx(255.0)
    assert float(observation["observation.tactile.left_raw"][0, 3]) == pytest.approx(0.0)
    assert float(observation["observation.tactile.left_raw"][0, 4]) == pytest.approx(1.0)
    assert float(observation["observation.tactile.left_clean"][0, 0]) == pytest.approx(0.0)
    assert float(observation["observation.tactile.left_clean"][0, 3]) == pytest.approx(0.0)
    assert float(observation["observation.tactile.left_clean"][0, 4]) == pytest.approx(1.0)
    assert float(observation["observation.tactile.right_raw"][0, 3]) == pytest.approx(255.0)
    assert FakeDataBus.instances[-1].tactile_freq == pytest.approx(30.0)

    driver.disconnect()
    assert FakeDataBus.instances[-1].stopped is True


def test_das_gripper_hardware_driver_supports_mask_fill_idle_baseline(tmp_path, monkeypatch):
    mask_payload = json.loads(
        (Path(__file__).resolve().parents[2] / "docs/tactile/tactile_valid_mask_50x10.json").read_text(encoding="utf-8")
    )
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(json.dumps(mask_payload), encoding="utf-8")

    baseline_payload = {
        "encoding": "mask_fill",
        "shape": [50, 10],
        "sides": {
            "left": {"valid_value": 0.0, "invalid_value": 255.0},
            "right": {"valid_value": 0.0, "invalid_value": 255.0},
        },
    }
    baseline_path = tmp_path / "idle_baseline.json"
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    left_valid = bytes([index % 256 for index in range(448)])
    right_valid = bytes([(255 - index) % 256 for index in range(448)])

    class FakeDataBus:
        instances: list["FakeDataBus"] = []

        def __init__(self, tty_port, baudrate, encoder_freq, encoder_callback, tactile_freq=None, tactile_callback=None):
            self.tty_port = tty_port
            self.baudrate = baudrate
            self.encoder_freq = encoder_freq
            self.encoder_callback = encoder_callback
            self.tactile_freq = tactile_freq
            self.tactile_callback = tactile_callback
            self.stopped = False
            type(self).instances.append(self)
            self.encoder_callback(struct.pack(">f", 0.0206))
            if self.tactile_callback is not None:
                self.tactile_callback(left_valid + right_valid)

        def set_target_distance(self, distance_m):
            del distance_m

        def stop(self):
            self.stopped = True

    fake_module = types.ModuleType("gen_controller_sdk_python")
    fake_module.DataBus = FakeDataBus
    monkeypatch.setitem(sys.modules, "gen_controller_sdk_python", fake_module)

    driver = fr3_backends.DasGripperHardwareDriver(
        serial_port="/dev/ttyUSB0",
        baudrate=921600,
        update_frequency_hz=50.0,
        tactile_frequency_hz=30.0,
        tactile_valid_mask_path=str(mask_path),
        tactile_baseline_path=str(baseline_path),
        min_distance_m=0.0,
        max_distance_m=0.103,
        initial_position=0.2,
    )
    driver.connect()

    observation = driver.get_tactile_observation()

    assert float(observation["observation.tactile.left_raw"][0, 0]) == pytest.approx(255.0)
    assert float(observation["observation.tactile.left_raw"][0, 3]) == pytest.approx(0.0)
    assert float(observation["observation.tactile.left_clean"][0, 0]) == pytest.approx(0.0)
    assert float(observation["observation.tactile.left_clean"][0, 4]) == pytest.approx(1.0)

    driver.disconnect()
    assert FakeDataBus.instances[-1].stopped is True


def test_das_gripper_hardware_driver_decodes_448_as_bilateral_horizontal_mirror_expand(tmp_path, monkeypatch):
    mask_payload = json.loads(
        (Path(__file__).resolve().parents[2] / "docs/tactile/tactile_valid_mask_50x10.json").read_text(encoding="utf-8")
    )
    mask_path = tmp_path / "mask.json"
    mask_path.write_text(json.dumps(mask_payload), encoding="utf-8")

    mask_rows = mask_payload["mask"]
    baseline_flat = [255.0 if int(mask_rows[row][col]) == 0 else 0.0 for row in range(50) for col in range(10)]
    baseline_payload = {
        "data": [
            {
                "tactiles": {
                    "left": baseline_flat,
                    "right": baseline_flat,
                }
            }
        ]
    }
    baseline_path = tmp_path / "baseline.json"
    baseline_path.write_text(json.dumps(baseline_payload), encoding="utf-8")

    left_compressed = bytes(range(224))
    right_compressed = bytes((32 + index) % 256 for index in range(224))

    class FakeDataBus:
        instances: list["FakeDataBus"] = []

        def __init__(self, tty_port, baudrate, encoder_freq, encoder_callback, tactile_freq=None, tactile_callback=None):
            self.tty_port = tty_port
            self.baudrate = baudrate
            self.encoder_freq = encoder_freq
            self.encoder_callback = encoder_callback
            self.tactile_freq = tactile_freq
            self.tactile_callback = tactile_callback
            self.stopped = False
            type(self).instances.append(self)
            self.encoder_callback(struct.pack(">f", 0.0206))
            if self.tactile_callback is not None:
                self.tactile_callback(left_compressed + right_compressed)

        def set_target_distance(self, distance_m):
            del distance_m

        def stop(self):
            self.stopped = True

    fake_module = types.ModuleType("gen_controller_sdk_python")
    fake_module.DataBus = FakeDataBus
    monkeypatch.setitem(sys.modules, "gen_controller_sdk_python", fake_module)

    driver = fr3_backends.DasGripperHardwareDriver(
        serial_port="/dev/ttyUSB0",
        baudrate=921600,
        update_frequency_hz=50.0,
        tactile_frequency_hz=30.0,
        tactile_valid_mask_path=str(mask_path),
        tactile_baseline_path=str(baseline_path),
        min_distance_m=0.0,
        max_distance_m=0.103,
        initial_position=0.2,
    )
    driver.connect()

    observation = driver.get_tactile_observation()

    assert observation["observation.tactile.left_raw"].shape == (50, 10)
    assert observation["observation.tactile.right_raw"].shape == (50, 10)
    assert float(observation["observation.tactile.left_raw"][0, 0]) == pytest.approx(255.0)
    assert float(observation["observation.tactile.left_raw"][0, 3]) == pytest.approx(0.0)
    assert float(observation["observation.tactile.left_raw"][0, 6]) == pytest.approx(0.0)
    assert float(observation["observation.tactile.left_raw"][0, 4]) == pytest.approx(1.0)
    assert float(observation["observation.tactile.left_raw"][0, 5]) == pytest.approx(1.0)
    assert float(observation["observation.tactile.left_clean"][0, 6]) == pytest.approx(0.0)

    assert float(observation["observation.tactile.right_raw"][0, 3]) == pytest.approx(3.0)
    assert float(observation["observation.tactile.right_raw"][0, 6]) == pytest.approx(3.0)
    assert float(observation["observation.tactile.right_raw"][0, 4]) == pytest.approx(2.0)
    assert float(observation["observation.tactile.right_raw"][0, 5]) == pytest.approx(2.0)
    assert float(observation["observation.tactile.right_clean"][0, 5]) == pytest.approx(2.0)

    driver.disconnect()
    assert FakeDataBus.instances[-1].stopped is True


def test_connect_uses_das_gripper_backend_when_configured(monkeypatch):
    DummyArmDriver.instances = []
    DummyGripperDriver.instances = []
    DummyOTGDriver.instances = []
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", FailingGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "das_gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)

    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB0",
            gripper_backend="das",
            gen_con_sdk_path="/opt/dependencies/gen_con_sdk_python_release",
            das_baudrate=921600,
            das_update_frequency_hz=60.0,
            das_min_distance_m=0.0,
            das_max_distance_m=0.103,
            das_grasp_threshold_m=0.003,
            das_initial_position=0.6,
            urdf_path="/tmp/fr3.urdf",
        )
    )

    try:
        robot.connect()
        init_kwargs = DummyGripperDriver.instances[-1].init_kwargs
        assert init_kwargs["serial_port"] == "/dev/ttyUSB0"
        assert init_kwargs["gen_con_sdk_path"] == "/opt/dependencies/gen_con_sdk_python_release"
        assert init_kwargs["baudrate"] == pytest.approx(921600)
        assert init_kwargs["update_frequency_hz"] == pytest.approx(60.0)
        assert init_kwargs["tactile_frequency_hz"] is None
        assert init_kwargs["tactile_valid_mask_path"] is None
        assert init_kwargs["tactile_baseline_path"] is None
        assert init_kwargs["max_distance_m"] == pytest.approx(0.103)
        assert init_kwargs["grasp_threshold_m"] == pytest.approx(0.003)
        assert init_kwargs["initial_position"] == pytest.approx(0.6)
        assert init_kwargs["command_deadband_m"] == pytest.approx(robot.config.gripper_command_deadband_mm / 1000.0)
    finally:
        if robot.is_connected:
            robot.disconnect()


def test_connect_uses_franka_hand_gripper_backend_when_configured(monkeypatch):
    DummyArmDriver.instances = []
    DummyGripperDriver.instances = []
    DummyOTGDriver.instances = []
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", FailingGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "franka_hand_gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)

    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_backend="franka_hand",
            urdf_path="/tmp/fr3.urdf",
        )
    )

    try:
        robot.connect()
        init_kwargs = DummyGripperDriver.instances[-1].init_kwargs
        assert init_kwargs["robot_ip"] == "192.168.1.206"
        assert init_kwargs["command_deadband_m"] == pytest.approx(robot.config.gripper_command_deadband_mm / 1000.0)
    finally:
        if robot.is_connected:
            robot.disconnect()


def test_send_action_binarizes_franka_hand_gripper_commands(monkeypatch):
    DummyArmDriver.instances = []
    DummyGripperDriver.instances = []
    DummyOTGDriver.instances = []
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", FailingGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "franka_hand_gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)

    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_backend="franka_hand",
            urdf_path="/tmp/fr3.urdf",
        )
    )

    try:
        robot.connect()

        opened = robot.send_action(
            {
                "enabled": False,
                "target_x": 0.0,
                "target_y": 0.0,
                "target_z": 0.0,
                "target_wx": 0.0,
                "target_wy": 0.0,
                "target_wz": 0.0,
                "gripper": 0.7,
            }
        )
        closed = robot.send_action(
            {
                "enabled": False,
                "target_x": 0.0,
                "target_y": 0.0,
                "target_z": 0.0,
                "target_wx": 0.0,
                "target_wy": 0.0,
                "target_wz": 0.0,
                "gripper": 0.3,
            }
        )

        assert DummyGripperDriver.instances[-1].set_position_calls[-2:] == [1.0, 0.0]
        assert opened["gripper"] == pytest.approx(1.0)
        assert closed["gripper"] == pytest.approx(0.0)
    finally:
        if robot.is_connected:
            robot.disconnect()


def test_connect_passes_gripper_command_throttle_config(monkeypatch):
    DummyArmDriver.instances = []
    DummyGripperDriver.instances = []
    DummyOTGDriver.instances = []
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)

    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
            gripper_command_rate_limit_hz=12.5,
            gripper_command_deadband_mm=0.25,
        )
    )

    try:
        robot.connect()
        assert DummyGripperDriver.instances[-1].init_kwargs["command_rate_limit_hz"] == pytest.approx(12.5)
        assert DummyGripperDriver.instances[-1].init_kwargs["command_deadband_mm"] == pytest.approx(0.25)
    finally:
        if robot.is_connected:
            robot.disconnect()


def test_move_to_start_restarts_otg_and_clears_teleop_state(robot):
    robot.connect()
    robot._reference_pose = np.eye(4, dtype=np.float64)
    robot._last_command_pose = np.eye(4, dtype=np.float64)
    robot._hold_joint_target = np.ones(7, dtype=np.float64)
    robot._prev_enabled = True

    robot.move_to_start()

    assert robot._arm.move_to_start_calls == 1
    assert len(DummyOTGDriver.instances[-1].reset_calls) == 2
    assert np.allclose(
        DummyOTGDriver.instances[-1].reset_calls[-1],
        np.array([0.0, -0.5, 0.0, -2.2, 0.0, 1.8, 0.7], dtype=np.float64),
    )
    assert robot._reference_pose is None
    assert robot._last_command_pose is None
    assert robot._hold_joint_target is None
    assert robot._prev_enabled is False


def test_move_to_start_uses_configured_start_joint_positions(monkeypatch):
    DummyOTGDriver.instances = []
    target = (0.0, -0.785, 0.0, -2.355, 0.0, 1.57079, 0.785)
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", UpdatingJointArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
            start_joint_positions=target,
            start_move_timeout_s=1.0,
        )
    )

    try:
        robot.connect()
        robot.move_to_start()

        assert robot._arm.move_to_start_calls == 0
        assert np.allclose(robot._arm.set_joint_positions_calls[-1], np.asarray(target, dtype=np.float64))
        assert np.allclose(DummyOTGDriver.instances[-1].step_calls[-1][1], np.asarray(target, dtype=np.float64))
        assert np.allclose(DummyOTGDriver.instances[-1].reset_calls[-1], np.asarray(target, dtype=np.float64))
    finally:
        if robot.is_connected:
            robot.disconnect()


def test_move_to_start_raises_when_backend_does_not_support_it(monkeypatch):
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", NoMoveToStartArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
        )
    )

    robot.connect()
    with pytest.raises(RuntimeError, match="does not support move_to_start"):
        robot.move_to_start()
    robot.disconnect()


def test_get_observation(robot):
    robot.connect()
    observation = robot.get_observation()

    expected_ee_keys = {"ee.x", "ee.y", "ee.z", "ee.wx", "ee.wy", "ee.wz", "gripper.pos"}
    expected_prev_cmd_keys = {
        "prev_cmd.ee.x",
        "prev_cmd.ee.y",
        "prev_cmd.ee.z",
        "prev_cmd.ee.wx",
        "prev_cmd.ee.wy",
        "prev_cmd.ee.wz",
        "prev_cmd.gripper.pos",
    }
    expected_joint_keys = {f"joint_{i}.pos" for i in range(1, 8)}
    assert expected_ee_keys.issubset(observation)
    assert expected_prev_cmd_keys.issubset(observation)
    assert expected_joint_keys.issubset(observation)
    assert observation["ee.x"] == pytest.approx(0.4)
    assert observation["ee.y"] == pytest.approx(0.1)
    assert observation["ee.z"] == pytest.approx(0.3)
    assert observation["gripper.pos"] == pytest.approx(0.25)
    assert observation["prev_cmd.ee.x"] == pytest.approx(observation["ee.x"])
    assert observation["prev_cmd.ee.y"] == pytest.approx(observation["ee.y"])
    assert observation["prev_cmd.ee.z"] == pytest.approx(observation["ee.z"])
    assert observation["prev_cmd.gripper.pos"] == pytest.approx(observation["gripper.pos"])


def test_get_observation_includes_tactile_when_gripper_provides_it(robot):
    robot.connect()
    robot._gripper.get_tactile_observation = lambda: {
        "observation.tactile.left_clean": np.ones((50, 10), dtype=np.float32),
        "observation.tactile.valid_mask": np.ones((50, 10), dtype=np.float32),
    }

    observation = robot.get_observation()

    assert observation["observation.tactile.left_clean"].shape == (50, 10)
    assert float(observation["observation.tactile.left_clean"][0, 0]) == pytest.approx(1.0)
    assert float(observation["observation.tactile.valid_mask"][0, 0]) == pytest.approx(1.0)


def test_get_observation_skips_cameras_when_disabled(robot):
    class FakeCamera:
        def __init__(self):
            self.read_latest_calls = 0

        def read_latest(self):
            self.read_latest_calls += 1
            return np.zeros((2, 2, 3), dtype=np.uint8)

    robot.connect()
    camera = FakeCamera()
    robot.cameras = {"front": camera}

    observation = robot.get_observation(include_cameras=False)

    assert "front" not in observation
    assert camera.read_latest_calls == 0

    observation = robot.get_observation()

    assert observation["front"].shape == (2, 2, 3)
    assert camera.read_latest_calls == 1


def test_get_observation_soft_syncs_cameras_from_frame_history(robot):
    class TimestampedCamera:
        def __init__(self, history):
            self.history = history

        def read_latest_with_timestamp(self, max_age_ms):
            del max_age_ms
            return self.history[-1]

        def read_closest(self, timestamp_s, max_age_ms):
            del max_age_ms
            return min(self.history, key=lambda sample: abs(sample[1] - timestamp_s))

    robot.connect()
    robot.reset_capture_timestamp_origin()
    origin = robot._capture_timestamp_origin_s
    robot.cameras = {
        "ee": TimestampedCamera(
            [
                (np.full((2, 2, 3), 66, dtype=np.uint8), origin + 0.066),
                (np.full((2, 2, 3), 100, dtype=np.uint8), origin + 0.100),
            ]
        ),
        "wrist": TimestampedCamera(
            [(np.full((2, 2, 3), 72, dtype=np.uint8), origin + 0.072)]
        ),
    }

    observation = robot.get_observation()

    assert int(observation["ee"][0, 0, 0]) == 66
    assert int(observation["wrist"][0, 0, 0]) == 72
    assert observation["camera.ee.capture_timestamp_s"] == pytest.approx(0.066)
    assert observation["camera.wrist.capture_timestamp_s"] == pytest.approx(0.072)


def test_anchoring_bounds_the_spread_when_one_camera_delivers_late(robot):
    """The property that makes the skew guard satisfiable at all.

    Serving each camera its own newest frame was tried on hardware and aborted an episode after
    21 frames with 25.1 ms of skew: nothing bounds how far apart two cameras' newest frames are
    when their background threads deliver independently and one falls a period behind. Anchoring
    on the oldest of those frames converts that unbounded gap into a bounded one -- here 18 ms of
    divergence between the newest frames becomes 2 ms of recorded skew.
    """

    class TimestampedCamera:
        def __init__(self, history):
            self.history = history

        def read_latest_with_timestamp(self, max_age_ms):
            del max_age_ms
            return self.history[-1]

        def read_closest(self, timestamp_s, max_age_ms):
            del max_age_ms
            return min(self.history, key=lambda sample: abs(sample[1] - timestamp_s))

    robot.connect()
    robot.reset_capture_timestamp_origin()
    origin = robot._capture_timestamp_origin_s
    robot.cameras = {
        # `ee` is a full period ahead; `wrist` has nothing newer than 0.098.
        "ee": TimestampedCamera(
            [
                (np.full((2, 2, 3), 100, dtype=np.uint8), origin + 0.100),
                (np.full((2, 2, 3), 116, dtype=np.uint8), origin + 0.116),
            ]
        ),
        "wrist": TimestampedCamera([(np.full((2, 2, 3), 98, dtype=np.uint8), origin + 0.098)]),
    }

    observation = robot.get_observation()

    # Taking the newest of each would have recorded 0.116 against 0.098 -- 18 ms apart, and past
    # the guard. Anchoring picks ee's 0.100 instead.
    assert observation["camera.ee.capture_timestamp_s"] == pytest.approx(0.100)
    assert observation["camera.wrist.capture_timestamp_s"] == pytest.approx(0.098)
    assert int(observation["ee"][0, 0, 0]) == 100


class _TimestampedTestCamera:
    def __init__(self, timestamp_s, value=0):
        self.timestamp_s = timestamp_s
        self.value = value

    def read_latest_with_timestamp(self, max_age_ms):
        del max_age_ms
        return np.full((2, 2, 3), self.value, dtype=np.uint8), self.timestamp_s

    def read_closest(self, timestamp_s, max_age_ms):
        del timestamp_s, max_age_ms
        return np.full((2, 2, 3), self.value, dtype=np.uint8), self.timestamp_s


def test_get_observation_rejects_camera_skew_above_threshold(robot):
    robot.connect()
    origin = robot._capture_timestamp_origin_s
    robot.cameras = {
        "ee": _TimestampedTestCamera(origin + 0.100),
        "wrist": _TimestampedTestCamera(origin + 0.125),
    }

    with pytest.raises(RuntimeError, match="camera skew 25.0 ms"):
        robot.get_observation()


def test_get_observation_can_record_camera_skew_for_offline_qc(robot):
    robot.config.camera_skew_hard_fail = False
    robot.connect()
    origin = robot._capture_timestamp_origin_s
    robot.cameras = {
        "ee": _TimestampedTestCamera(origin + 0.100, value=11),
        "wrist": _TimestampedTestCamera(origin + 0.133333, value=22),
    }

    observation = robot.get_observation()

    assert int(observation["ee"][0, 0, 0]) == 11
    assert int(observation["wrist"][0, 0, 0]) == 22
    assert observation["camera.ee.capture_timestamp_s"] == pytest.approx(0.100)
    assert observation["camera.wrist.capture_timestamp_s"] == pytest.approx(0.133333)


def test_get_observation_uses_kinematics_target_frame_even_if_arm_reports_pose(monkeypatch):
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", ReportingArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
        )
    )

    robot.connect()
    observation = robot.get_observation()

    assert observation["ee.x"] == pytest.approx(0.4)
    assert observation["ee.y"] == pytest.approx(0.1)
    assert observation["ee.z"] == pytest.approx(0.3)
    robot.disconnect()


def test_connect_falls_back_to_mock_gripper_when_hardware_unavailable(monkeypatch):
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", FailingGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
        )
    )

    robot.connect()
    assert robot.is_connected
    assert robot._gripper_is_mock is True
    assert robot.get_observation()["gripper.pos"] == pytest.approx(1.0)
    robot.disconnect()


def test_connect_uses_explicit_mock_gripper_without_opening_hardware(monkeypatch):
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", FailingGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_backend="mock",
            allow_mock_gripper=False,
            urdf_path="/tmp/fr3.urdf",
            ik_solver="placo",
        )
    )

    robot.connect()
    assert robot.is_connected
    assert robot._gripper_is_mock is True
    assert robot.get_observation()["gripper.pos"] == pytest.approx(1.0)
    robot.disconnect()


def test_connect_raises_when_mock_gripper_disabled(monkeypatch):
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", FailingGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            allow_mock_gripper=False,
            urdf_path="/tmp/fr3.urdf",
        )
    )

    with pytest.raises(RuntimeError, match="FR3 gripper hardware unavailable on /dev/ttyUSB80"):
        robot.connect()

    assert robot.is_connected is False
    assert robot._gripper is None


def test_send_action_clips_workspace_and_sends_joint_targets(robot):
    robot.connect()

    action = {
        "enabled": True,
        "target_x": 0.5,
        "target_y": 0.5,
        "target_z": 0.5,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 1.2,
    }
    returned = robot.send_action(action)
    deadline = time.perf_counter() + 0.2
    while (
        (
            len(robot._arm.set_joint_positions_calls) == 0
            or not np.allclose(robot._arm.set_joint_positions_calls[-1], robot._kinematics.inverse_solution)
        )
        and time.perf_counter() < deadline
    ):
        time.sleep(0.005)

    assert returned["gripper"] == pytest.approx(1.0)
    assert robot._gripper.set_position_calls[-1] == pytest.approx(1.0)
    assert len(robot._arm.set_joint_positions_calls) >= 1
    assert np.allclose(robot._arm.set_joint_positions_calls[-1], robot._kinematics.inverse_solution)

    _, desired_pose = robot._kinematics.inverse_calls[-1]
    assert np.allclose(desired_pose[:3, 3], np.array([0.6, 0.3, 0.5]))


def test_send_action_reuses_observation_joint_snapshot(monkeypatch):
    DummyArmDriver.instances = []
    DummyGripperDriver.instances = []
    DummyOTGDriver.instances = []
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)

    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
            use_otg=False,
        )
    )

    try:
        robot.connect()
        arm = DummyArmDriver.instances[-1]
        assert arm.get_joint_positions_calls == 0

        observation = robot.get_observation(include_cameras=False)
        assert observation["gripper.pos"] == pytest.approx(0.25)
        assert arm.get_joint_positions_calls == 1

        robot.send_action(
            {
                "enabled": True,
                "target_x": 0.01,
                "target_y": -0.02,
                "target_z": 0.03,
                "target_wx": 0.0,
                "target_wy": 0.0,
                "target_wz": 0.0,
                "gripper": 0.5,
            }
        )

        assert arm.get_joint_positions_calls == 1
    finally:
        if robot.is_connected:
            robot.disconnect()


def test_send_action_accepts_absolute_ee_targets(robot):
    robot.connect()

    absolute_action = {
        "ee.x": 0.7,
        "ee.y": 0.5,
        "ee.z": 0.8,
        "ee.wx": 0.0,
        "ee.wy": 0.0,
        "ee.wz": 0.0,
        "gripper.pos": 1.2,
    }
    returned = robot.send_action(absolute_action)
    deadline = time.perf_counter() + 0.2
    while (
        (
            len(robot._arm.set_joint_positions_calls) == 0
            or not np.allclose(robot._arm.set_joint_positions_calls[-1], robot._kinematics.inverse_solution)
        )
        and time.perf_counter() < deadline
    ):
        time.sleep(0.005)

    _, desired_pose = robot._kinematics.inverse_calls[-1]
    assert np.allclose(desired_pose[:3, 3], np.array([0.6, 0.3, 0.5]))
    assert robot._kinematics.inverse_kwargs[-1] == {}
    assert returned["ee.x"] == pytest.approx(0.6)
    assert returned["ee.y"] == pytest.approx(0.3)
    assert returned["ee.z"] == pytest.approx(0.5)
    assert returned["gripper.pos"] == pytest.approx(1.0)
    assert robot._gripper.set_position_calls[-1] == pytest.approx(1.0)


def test_send_action_passes_configured_ik_orientation_weight_for_absolute_ee(monkeypatch):
    DummyArmDriver.instances = []
    DummyGripperDriver.instances = []
    DummyOTGDriver.instances = []
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            gripper_backend="mock",
            urdf_path="/tmp/fr3.urdf",
            ik_orientation_weight=0.012,
            use_otg=False,
        )
    )

    try:
        robot.connect()
        robot.send_action(
            {
                "ee.x": 0.7,
                "ee.y": 0.5,
                "ee.z": 0.8,
                "ee.wx": 0.0,
                "ee.wy": 0.0,
                "ee.wz": 0.0,
                "gripper.pos": 1.0,
            }
        )

        assert robot._kinematics.inverse_kwargs[-1]["orientation_weight"] == pytest.approx(0.012)
    finally:
        if robot.is_connected:
            robot.disconnect()


def test_send_action_disabled_stops_at_current_pose_after_release(robot):
    robot.connect()
    first_action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": -0.02,
        "target_z": 0.03,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }
    robot.send_action(first_action)
    assert robot._last_command_pose is not None
    inverse_call_count = len(robot._kinematics.inverse_calls)
    current_joints = robot._arm.get_joint_positions()

    second_action = {
        "enabled": False,
        "target_x": 0.4,
        "target_y": 0.4,
        "target_z": 0.4,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.1,
    }
    robot.send_action(second_action)

    assert len(robot._kinematics.inverse_calls) == inverse_call_count
    assert np.allclose(robot._last_command_pose, robot._kinematics.forward_pose)
    with robot._otg_target_lock:
        assert robot._otg_target_joints is not None
        assert np.allclose(robot._otg_target_joints, current_joints)


def test_send_action_disabled_after_active_holds_current_otg_command(robot):
    robot.connect()
    first_action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": -0.02,
        "target_z": 0.03,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }
    robot.send_action(first_action)

    latched_otg_command = np.array([0.21, 0.22, 0.23, -0.94, 0.55, 1.18, -0.68], dtype=np.float64)
    with robot._otg_command_lock:
        robot._otg_command_joints = latched_otg_command.copy()

    second_action = {
        "enabled": False,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }
    inverse_call_count = len(robot._kinematics.inverse_calls)
    robot.send_action(second_action)

    assert len(robot._kinematics.inverse_calls) == inverse_call_count
    with robot._otg_target_lock:
        assert robot._otg_target_joints is not None
        assert np.allclose(robot._otg_target_joints, latched_otg_command)

    robot._arm.joint_positions = np.array([0.05, 0.1, 0.15, -1.1, 0.45, 1.05, -0.8], dtype=np.float64)
    robot.send_action(second_action)
    with robot._otg_target_lock:
        assert robot._otg_target_joints is not None
        assert np.allclose(robot._otg_target_joints, latched_otg_command)


def test_send_action_disabled_without_previous_command_holds_current_joints(robot):
    robot.connect()
    current_joints = robot._arm.get_joint_positions()
    action = {
        "enabled": False,
        "target_x": 0.4,
        "target_y": 0.4,
        "target_z": 0.4,
        "target_wx": 0.1,
        "target_wy": -0.1,
        "target_wz": 0.2,
        "gripper": 0.1,
    }

    robot.send_action(action)

    assert len(robot._kinematics.inverse_calls) == 0
    assert np.allclose(robot._last_command_pose, robot._kinematics.forward_pose)
    with robot._otg_target_lock:
        assert robot._otg_target_joints is not None
        assert np.allclose(robot._otg_target_joints, current_joints)


def test_send_action_integrates_relative_pose_while_enabled(robot):
    robot.connect()
    action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": -0.02,
        "target_z": 0.03,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }

    robot.send_action(action)
    robot.send_action(action)

    _, first_desired_pose = robot._kinematics.inverse_calls[-2]
    _, second_desired_pose = robot._kinematics.inverse_calls[-1]
    assert np.allclose(first_desired_pose[:3, 3], np.array([0.41, 0.08, 0.33]))
    assert np.allclose(second_desired_pose[:3, 3], np.array([0.42, 0.06, 0.36]))


def test_delta_action_processor_outputs_absolute_ee_targets(robot):
    robot.connect()
    processor = RobotProcessorPipeline[tuple[dict, dict], dict](
        steps=[
            DeltaActionToAbsoluteEEAction(
                workspace_min=robot.config.workspace_min,
                workspace_max=robot.config.workspace_max,
                max_target_delta_pos=robot.config.max_target_delta_pos,
                max_target_delta_rot=robot.config.max_target_delta_rot,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    observation = robot.get_observation()
    action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": -0.02,
        "target_z": 0.03,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }
    processed_first = processor((action.copy(), observation))
    processed_second = processor((action.copy(), observation))

    assert processed_first["ee.x"] == pytest.approx(0.41)
    assert processed_first["ee.y"] == pytest.approx(0.08)
    assert processed_first["ee.z"] == pytest.approx(0.33)
    assert processed_first["enabled"] is True
    assert processed_first["target_x"] == pytest.approx(0.01)
    assert processed_first["target_y"] == pytest.approx(-0.02)
    assert processed_first["target_z"] == pytest.approx(0.03)
    assert processed_first["gripper"] == pytest.approx(0.5)
    assert processed_first["ee.qx"] == pytest.approx(0.0)
    assert processed_first["ee.qy"] == pytest.approx(0.0)
    assert processed_first["ee.qz"] == pytest.approx(0.0)
    assert processed_first["ee.qw"] == pytest.approx(1.0)
    assert processed_first["gripper.pos"] == pytest.approx(0.5)
    assert processed_second["ee.x"] == pytest.approx(0.42)
    assert processed_second["ee.y"] == pytest.approx(0.06)
    assert processed_second["ee.z"] == pytest.approx(0.36)


def test_delta_action_processor_reset_clears_previous_episode_target(robot):
    robot.connect()
    processor = RobotProcessorPipeline[tuple[dict, dict], dict](
        steps=[
            DeltaActionToAbsoluteEEAction(
                workspace_min=robot.config.workspace_min,
                workspace_max=robot.config.workspace_max,
                max_target_delta_pos=robot.config.max_target_delta_pos,
                max_target_delta_rot=robot.config.max_target_delta_rot,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    episode_one_observation = robot.get_observation()
    active_action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": -0.02,
        "target_z": 0.03,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }
    processor((active_action.copy(), episode_one_observation))

    start_pose_observation = episode_one_observation.copy()
    start_pose_observation["ee.x"] = 0.3
    start_pose_observation["ee.y"] = -0.1
    start_pose_observation["ee.z"] = 0.25
    processor.reset()

    disabled_action = {
        "enabled": False,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }
    processed = processor((disabled_action.copy(), start_pose_observation))

    assert processed["ee.x"] == pytest.approx(0.3)
    assert processed["ee.y"] == pytest.approx(-0.1)
    assert processed["ee.z"] == pytest.approx(0.25)
    assert processed["ee.qx"] == pytest.approx(0.0)
    assert processed["ee.qy"] == pytest.approx(0.0)
    assert processed["ee.qz"] == pytest.approx(0.0)
    assert processed["ee.qw"] == pytest.approx(1.0)
    assert processed["gripper.pos"] == pytest.approx(0.5)


def test_absolute_ee_action_to_robot_action_uses_delta_hold_when_disabled():
    processor = RobotProcessorPipeline[tuple[dict, dict], dict](
        steps=[AbsoluteEEActionToRobotAction()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    observation = {
        "ee.x": 0.4,
        "ee.y": 0.1,
        "ee.z": 0.3,
        "ee.qx": 0.0,
        "ee.qy": 0.0,
        "ee.qz": 0.0,
        "ee.qw": 1.0,
        "gripper.pos": 0.5,
    }
    action = {
        "enabled": False,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
        "ee.x": 0.39,
        "ee.y": 0.09,
        "ee.z": 0.28,
        "ee.qx": 0.0,
        "ee.qy": 0.0,
        "ee.qz": 0.0,
        "ee.qw": 1.0,
        "gripper.pos": 0.5,
    }

    processed = processor((action, observation))

    assert processed == {
        "enabled": False,
        "target_x": 0.0,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }


def test_absolute_ee_action_to_robot_action_converts_quaternion_to_rotvec_when_enabled():
    processor = RobotProcessorPipeline[tuple[dict, dict], dict](
        steps=[AbsoluteEEActionToRobotAction()],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )

    observation = {
        "ee.x": 0.4,
        "ee.y": 0.1,
        "ee.z": 0.3,
        "ee.qx": 0.0,
        "ee.qy": 0.0,
        "ee.qz": 0.0,
        "ee.qw": 1.0,
        "gripper.pos": 0.5,
    }
    quat = Rotation.from_rotvec([0.0, 0.0, np.pi / 2]).as_quat()
    action = {
        "enabled": True,
        "ee.x": 0.39,
        "ee.y": 0.09,
        "ee.z": 0.28,
        "ee.qx": float(quat[0]),
        "ee.qy": float(quat[1]),
        "ee.qz": float(quat[2]),
        "ee.qw": float(quat[3]),
        "gripper.pos": 0.5,
    }

    processed = processor((action, observation))

    assert processed["ee.x"] == pytest.approx(0.39)
    assert processed["ee.y"] == pytest.approx(0.09)
    assert processed["ee.z"] == pytest.approx(0.28)
    assert processed["ee.wx"] == pytest.approx(0.0)
    assert processed["ee.wy"] == pytest.approx(0.0)
    assert processed["ee.wz"] == pytest.approx(np.pi / 2)
    assert processed["gripper.pos"] == pytest.approx(0.5)


def test_keep_absolute_ee_observation_filters_joint_state(robot):
    robot.connect()
    processor = RobotProcessorPipeline[dict, dict](
        steps=[KeepAbsoluteEEObservation()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    filtered = processor(robot.get_observation())

    assert "ee.x" in filtered
    assert "ee.qx" in filtered
    assert "ee.qy" in filtered
    assert "ee.qz" in filtered
    assert "ee.qw" in filtered
    assert "ee.wx" not in filtered
    assert "gripper.pos" in filtered
    assert "prev_cmd.ee.qx" in filtered
    assert "prev_cmd.ee.qw" in filtered
    assert "prev_cmd.ee.wx" not in filtered
    assert "prev_cmd.gripper.pos" in filtered
    assert "joint_1.pos" not in filtered


def test_delta_action_processor_matches_observation_quaternion_sign_on_first_frame(robot):
    robot.connect()
    processor = RobotProcessorPipeline[tuple[dict, dict], dict](
        steps=[
            DeltaActionToAbsoluteEEAction(
                workspace_min=robot.config.workspace_min,
                workspace_max=robot.config.workspace_max,
                max_target_delta_pos=robot.config.max_target_delta_pos,
                max_target_delta_rot=robot.config.max_target_delta_rot,
            )
        ],
        to_transition=robot_action_observation_to_transition,
        to_output=transition_to_robot_action,
    )
    observation_processor = RobotProcessorPipeline[dict, dict](
        steps=[KeepAbsoluteEEObservation()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    observation = robot.get_observation()
    processed_observation = observation_processor(observation)
    processed_action = processor(
        (
            {
                "enabled": False,
                "target_x": 0.0,
                "target_y": 0.0,
                "target_z": 0.0,
                "target_wx": 0.0,
                "target_wy": 0.0,
                "target_wz": 0.0,
                "gripper": observation["gripper.pos"],
            },
            observation,
        )
    )

    observation_quaternion = np.asarray(
        [processed_observation["ee.qx"], processed_observation["ee.qy"], processed_observation["ee.qz"], processed_observation["ee.qw"]],
        dtype=np.float64,
    )
    action_quaternion = np.asarray(
        [processed_action["ee.qx"], processed_action["ee.qy"], processed_action["ee.qz"], processed_action["ee.qw"]],
        dtype=np.float64,
    )

    assert float(np.dot(observation_quaternion, action_quaternion)) > 0.0


def test_keep_absolute_ee_observation_makes_quaternion_sign_continuous():
    processor = RobotProcessorPipeline[dict, dict](
        steps=[KeepAbsoluteEEObservation()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    first = processor(
        {
            "ee.x": 0.4,
            "ee.y": 0.1,
            "ee.z": 0.3,
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": np.pi - 0.01,
            "gripper.pos": 0.5,
        }
    )
    second = processor(
        {
            "ee.x": 0.4,
            "ee.y": 0.1,
            "ee.z": 0.3,
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": -(np.pi - 0.01),
            "gripper.pos": 0.5,
        }
    )

    first_quat = np.array([first["ee.qx"], first["ee.qy"], first["ee.qz"], first["ee.qw"]], dtype=np.float64)
    second_quat = np.array([second["ee.qx"], second["ee.qy"], second["ee.qz"], second["ee.qw"]], dtype=np.float64)
    assert float(np.dot(first_quat, second_quat)) > 0.0


def test_keep_absolute_ee_observation_reset_restarts_quaternion_continuity():
    processor = RobotProcessorPipeline[dict, dict](
        steps=[KeepAbsoluteEEObservation()],
        to_transition=observation_to_transition,
        to_output=transition_to_observation,
    )

    processor(
        {
            "ee.x": 0.4,
            "ee.y": 0.1,
            "ee.z": 0.3,
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": np.pi - 0.01,
            "gripper.pos": 0.5,
        }
    )
    processor.reset()

    after_reset = processor(
        {
            "ee.x": 0.4,
            "ee.y": 0.1,
            "ee.z": 0.3,
            "ee.wx": 0.0,
            "ee.wy": 0.0,
            "ee.wz": -(np.pi - 0.01),
            "gripper.pos": 0.5,
        }
    )
    after_reset_quat = np.array(
        [after_reset["ee.qx"], after_reset["ee.qy"], after_reset["ee.qz"], after_reset["ee.qw"]],
        dtype=np.float64,
    )
    expected_quat = Rotation.from_rotvec([0.0, 0.0, -(np.pi - 0.01)]).as_quat()
    assert np.allclose(after_reset_quat, -expected_quat)


def test_send_action_clamps_target_delta_before_workspace(monkeypatch):
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
            max_target_delta_pos=(0.01, 0.02, 0.03),
            max_target_delta_rot=(0.1, 0.2, 0.3),
        )
    )
    robot.connect()

    action = {
        "enabled": True,
        "target_x": 1.0,
        "target_y": -1.0,
        "target_z": 1.0,
        "target_wx": 1.0,
        "target_wy": -1.0,
        "target_wz": 1.0,
        "gripper": 0.5,
    }
    robot.send_action(action)

    _, desired_pose = robot._kinematics.inverse_calls[-1]
    assert np.allclose(desired_pose[:3, 3], np.array([0.41, 0.08, 0.33]))
    rotvec = Rotation.from_matrix(desired_pose[:3, :3]).as_rotvec()
    assert np.allclose(rotvec, np.array([0.1, -0.2, 0.3]))
    robot.disconnect()


def test_send_action_runs_joint_targets_through_otg(monkeypatch):
    class SmoothingOTGDriver(DummyOTGDriver):
        def step(self, current_joint_positions: np.ndarray, target_joint_positions: np.ndarray) -> np.ndarray:
            current = np.asarray(current_joint_positions, dtype=np.float64)
            target = np.asarray(target_joint_positions, dtype=np.float64)
            self.step_calls.append((current.copy(), target.copy()))
            return target - 0.05

    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", SmoothingOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
            use_otg=True,
        )
    )
    robot.connect()

    action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }
    robot.send_action(action)
    deadline = time.perf_counter() + 0.2
    expected_command = robot._kinematics.inverse_solution - 0.05
    while (
        (
            len(robot._arm.set_joint_positions_calls) == 0
            or not np.allclose(robot._arm.set_joint_positions_calls[-1], expected_command)
        )
        and time.perf_counter() < deadline
    ):
        time.sleep(0.005)

    assert len(robot._otg.step_calls) >= 1
    assert np.allclose(robot._arm.set_joint_positions_calls[-1], expected_command)
    robot.disconnect()


def test_otg_continues_running_after_single_send_action(robot):
    robot.connect()
    action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }

    robot.send_action(action)
    deadline = time.perf_counter() + 0.2
    while len(robot._otg.step_calls) < 2 and time.perf_counter() < deadline:
        time.sleep(0.005)

    assert len(robot._otg.step_calls) >= 2


def test_otg_sender_runs_faster_than_smoother(monkeypatch):
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", DummyOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
            otg_control_frequency=20.0,
            otg_async_control_frequency=200.0,
        )
    )
    robot.connect()
    action = {
        "enabled": True,
        "target_x": 0.01,
        "target_y": 0.0,
        "target_z": 0.0,
        "target_wx": 0.0,
        "target_wy": 0.0,
        "target_wz": 0.0,
        "gripper": 0.5,
    }

    robot.send_action(action)
    deadline = time.perf_counter() + 0.25
    while len(robot._otg.step_calls) < 2 and time.perf_counter() < deadline:
        time.sleep(0.005)

    assert len(robot._otg.step_calls) >= 2
    assert len(robot._arm.set_joint_positions_calls) > len(robot._otg.step_calls)
    robot.disconnect()


def test_connect_cleans_up_partial_backends(monkeypatch):
    DummyArmDriver.instances = []
    DummyGripperDriver.instances = []
    monkeypatch.setattr(FrankaResearch3, "arm_driver_cls", DummyArmDriver)
    monkeypatch.setattr(FrankaResearch3, "gripper_driver_cls", DummyGripperDriver)
    monkeypatch.setattr(FrankaResearch3, "kinematics_driver_cls", DummyKinematicsDriver)
    monkeypatch.setattr(FrankaResearch3, "otg_driver_cls", FailingOTGDriver)
    robot = FrankaResearch3(
        FrankaResearch3Config(
            robot_ip="192.168.1.206",
            gripper_port="/dev/ttyUSB80",
            urdf_path="/tmp/fr3.urdf",
            use_otg=True,
        )
    )

    with pytest.raises(RuntimeError, match="otg init failed"):
        robot.connect()

    assert not robot.is_connected
    assert robot._arm is None
    assert robot._gripper is None
    assert DummyArmDriver.instances[-1].connected is False
    assert DummyGripperDriver.instances[-1].connected is False


def test_arm_timestamp_marks_when_the_state_was_read_not_when_we_picked_it_up(monkeypatch):
    """Two observations served from one cached sample must carry that sample's instant.

    The driver serves a cache refreshed by its own state reader at state_poll_frequency_hz.
    Stamping `perf_counter()` at pickup instead credits the reading with freshness it does not
    have -- up to one poll period, varying frame to frame -- and that error is unrecoverable
    afterwards because it lands inside the arm-vs-camera offset and looks like camera latency.
    """

    class DummyJointPositionController:
        def set_control(self, joint_positions):
            del joint_positions

    class DummyPanda:
        def __init__(self, robot_ip):
            self.robot_ip = robot_ip
            self.state = types.SimpleNamespace(
                q=np.array([0.1, 0.2, 0.3, -1.0, 0.5, 1.2, -0.7], dtype=np.float64),
                robot_mode=types.SimpleNamespace(name="kIdle"),
            )

        def start_controller(self, controller):
            del controller

        def stop_controller(self):
            return None

        def get_state(self):
            return self.state

    monkeypatch.setitem(
        sys.modules,
        "panda_py",
        types.SimpleNamespace(
            Panda=DummyPanda,
            controllers=types.SimpleNamespace(JointPosition=DummyJointPositionController),
        ),
    )

    driver = PandaPyArmDriver(robot_ip="192.168.1.206", state_poll_frequency_hz=0.0)
    driver.connect()

    _, first = driver.get_joint_positions_with_timestamp()
    time.sleep(0.01)
    _, second = driver.get_joint_positions_with_timestamp()

    # Same cached sample -> same instant. A pickup-time stamp would differ by ~10 ms here.
    assert first == second
    # And it is the moment the state arrived, so it precedes any later read.
    assert first <= time.perf_counter()

    driver.disconnect()


def test_arm_capture_timestamp_comes_from_the_driver_when_it_reports_one(robot):
    sampled_at_s = time.perf_counter() - 0.004

    class SamplingArmDriver(DummyArmDriver):
        def get_joint_positions_with_timestamp(self):
            return self.joint_positions, sampled_at_s

    robot._arm = SamplingArmDriver()
    robot.connect()
    robot._arm = SamplingArmDriver()
    robot.reset_capture_timestamp_origin()

    observation = robot.get_observation(include_cameras=False)

    expected = sampled_at_s - robot._capture_timestamp_origin_s
    assert observation["fr3.arm.capture_timestamp_s"] == pytest.approx(expected, abs=1e-9)


def test_arm_capture_timestamp_falls_back_for_a_backend_without_sampling_instants(robot):
    """Older backends keep working; the read instant is the honest upper bound."""
    robot.connect()
    assert not hasattr(robot._arm, "get_joint_positions_with_timestamp")
    robot.reset_capture_timestamp_origin()

    before = time.perf_counter() - robot._capture_timestamp_origin_s
    observation = robot.get_observation(include_cameras=False)
    after = time.perf_counter() - robot._capture_timestamp_origin_s

    assert before <= observation["fr3.arm.capture_timestamp_s"] <= after


def test_franka_hand_gripper_timestamp_marks_when_the_hand_was_read(monkeypatch):
    """The same failure as the arm, and twenty times larger: this driver polls at 10 Hz."""

    class FakeFrankaHandState:
        def __init__(self, width=0.04, max_width=0.08):
            self.width = width
            self.max_width = max_width

    class FakeFrankaHand:
        def __init__(self, robot_ip):
            self.robot_ip = robot_ip
            self.state = FakeFrankaHandState()

        def homing(self):
            return True

        def read_once(self):
            return self.state

        def stop(self):
            return None

    monkeypatch.setitem(
        sys.modules,
        "panda_py",
        types.SimpleNamespace(libfranka=types.SimpleNamespace(Gripper=FakeFrankaHand)),
    )

    driver = FrankaHandGripperHardwareDriver(
        robot_ip="192.168.1.206",
        state_poll_frequency_hz=0.0,
    )
    driver.connect()

    position, first = driver.get_position_with_timestamp()
    time.sleep(0.01)
    _, second = driver.get_position_with_timestamp()

    assert position == pytest.approx(0.5)
    # One cached read behind both calls -> one instant. Stamping at pickup would report the
    # second reading as 10 ms fresher than the first while nothing new was read.
    assert first == second
    assert first <= time.perf_counter()

    driver.disconnect()


def test_das_gripper_timestamp_comes_from_the_encoder_callback(monkeypatch):
    """The databus pushes updates on its own thread, so arrival is the only knowable instant."""

    class FakeDataBus:
        instances: list["FakeDataBus"] = []

        def __init__(self, tty_port, baudrate, encoder_freq, encoder_callback, tactile_freq=None, tactile_callback=None):
            del tty_port, baudrate, encoder_freq, tactile_freq, tactile_callback
            self.encoder_callback = encoder_callback
            type(self).instances.append(self)
            self.encoder_callback(struct.pack(">f", 0.0206))

        def set_target_distance(self, distance_m):
            del distance_m

        def stop(self):
            return None

    fake_module = types.ModuleType("gen_controller_sdk_python")
    fake_module.DataBus = FakeDataBus
    monkeypatch.setitem(sys.modules, "gen_controller_sdk_python", fake_module)

    driver = fr3_backends.DasGripperHardwareDriver(
        serial_port="/dev/ttyUSB0",
        baudrate=921600,
        update_frequency_hz=50.0,
        min_distance_m=0.0,
        max_distance_m=0.103,
        initial_position=0.2,
    )
    driver.connect()
    connect_returned_at_s = time.perf_counter()

    time.sleep(0.01)
    position, first = driver.get_position_with_timestamp()

    assert position == pytest.approx(0.2, abs=1e-3)
    # The update landed during connect(), so its instant is behind us -- not the read we just did.
    assert first <= connect_returned_at_s

    FakeDataBus.instances[-1].encoder_callback(struct.pack(">f", 0.0412))
    moved_position, second = driver.get_position_with_timestamp()

    assert moved_position == pytest.approx(0.4, abs=1e-3)
    assert second > first

    driver.disconnect()


def test_gripper_capture_timestamp_comes_from_the_driver_when_it_reports_one(robot):
    sampled_at_s = time.perf_counter() - 0.05

    class SamplingGripperDriver(DummyGripperDriver):
        def get_position_with_timestamp(self):
            return self.position, sampled_at_s

    robot.connect()
    sampling_gripper = SamplingGripperDriver()
    robot._gripper = sampling_gripper
    robot.reset_capture_timestamp_origin()

    observation = robot.get_observation(include_cameras=False)

    expected = sampled_at_s - robot._capture_timestamp_origin_s
    assert observation["pika_gripper.capture_timestamp_s"] == pytest.approx(expected, abs=1e-9)
    assert observation["gripper.pos"] == pytest.approx(sampling_gripper.position)


def test_gripper_capture_timestamp_falls_back_for_a_backend_without_sampling_instants(robot):
    """`pika` and `corenetic` read on demand, so the read instant is a true upper bound."""
    robot.connect()
    assert not hasattr(robot._gripper, "get_position_with_timestamp")
    robot.reset_capture_timestamp_origin()

    before = time.perf_counter() - robot._capture_timestamp_origin_s
    observation = robot.get_observation(include_cameras=False)
    after = time.perf_counter() - robot._capture_timestamp_origin_s

    assert before <= observation["pika_gripper.capture_timestamp_s"] <= after
