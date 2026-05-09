#!/usr/bin/env python

from __future__ import annotations

import pytest

from lerobot.teleoperators import make_teleoperator_from_config
from lerobot.teleoperators.nintendo import (
    NintendoController,
    NintendoGripperMode,
    NintendoTeleop,
    NintendoTeleopConfig,
)
from lerobot.teleoperators.nintendo.backend import NintendoControllerReading
from tools.fr3.fr3_mujoco_runtime import build_runtime_teleop_config, create_runtime_arg_parser


class DummyNintendoDriver:
    instances: list["DummyNintendoDriver"] = []
    queued_readings: list[NintendoControllerReading | None] = []

    def __init__(self, *args, **kwargs):
        del args, kwargs
        type(self).instances.append(self)
        self.connected = False
        self.readings = list(type(self).queued_readings)

    def connect(self) -> None:
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False

    def poll(self) -> NintendoControllerReading | None:
        if not self.readings:
            return None
        return self.readings.pop(0)


def reading(
    *,
    controller_type: str = "pro",
    left_stick: tuple[float, float] = (0.0, 0.0),
    right_stick: tuple[float, float] = (0.0, 0.0),
    buttons: tuple[str, ...] = (),
) -> NintendoControllerReading:
    return NintendoControllerReading(
        controller_type=controller_type,
        left_stick=left_stick,
        right_stick=right_stick,
        buttons=frozenset(buttons),
        accel_g=(0.0, 0.0, -1.0),
        gyro_dps=(0.0, 0.0, 0.0),
    )


@pytest.fixture
def teleop(monkeypatch):
    DummyNintendoDriver.instances = []
    DummyNintendoDriver.queued_readings = []
    monkeypatch.setattr(NintendoTeleop, "driver_cls", DummyNintendoDriver)
    device = NintendoTeleop(
        NintendoTeleopConfig(
            controller=NintendoController.PRO,
            stick_deadband=0.0,
            translation_scale=0.01,
            vertical_scale=0.02,
            rotation_scale=0.03,
            gripper_cmd_ema_alpha=0.0,
            gripper_cmd_max_rate=0.0,
        )
    )
    yield device
    if device.is_connected:
        device.disconnect()


def test_factory_creates_nintendo_teleop():
    teleop = make_teleoperator_from_config(NintendoTeleopConfig())

    assert isinstance(teleop, NintendoTeleop)


def test_runtime_parser_builds_nintendo_config():
    parser = create_runtime_arg_parser(description="test")
    args = parser.parse_args(
        [
            "--teleop-type",
            "nintendo",
            "--nintendo-controller",
            "pro",
            "--nintendo-device-id",
            "2",
            "--fps",
            "100",
        ]
    )

    cfg = build_runtime_teleop_config(args)

    assert isinstance(cfg, NintendoTeleopConfig)
    assert cfg.type == "nintendo"
    assert cfg.controller == NintendoController.PRO
    assert cfg.device_id == 2
    assert cfg.frequency == 100


def test_get_action_maps_pro_sticks_when_clutch_is_held(teleop):
    teleop.connect()
    teleop._driver.readings.append(
        reading(left_stick=(0.25, 0.5), right_stick=(-0.75, 0.4), buttons=("R",))
    )

    action = teleop.get_action()

    assert action["enabled"] is True
    assert action["target_x"] == pytest.approx(0.5 * 0.01)
    assert action["target_y"] == pytest.approx(0.25 * 0.01)
    assert action["target_z"] == pytest.approx(0.4 * 0.02)
    assert action["target_wx"] == pytest.approx(0.0)
    assert action["target_wy"] == pytest.approx(0.0)
    assert action["target_wz"] == pytest.approx(-0.75 * 0.03)


def test_get_action_requires_clutch_for_motion_but_allows_gripper_update(teleop):
    teleop.connect()
    teleop._driver.readings.append(
        reading(left_stick=(0.0, 1.0), right_stick=(0.0, 0.0), buttons=("ZR",))
    )

    action = teleop.get_action()

    assert action["enabled"] is False
    assert action["target_x"] == pytest.approx(0.0)
    assert action["gripper"] < 1.0


def test_binary_gripper_buttons_close_and_open(monkeypatch):
    monkeypatch.setattr(NintendoTeleop, "driver_cls", DummyNintendoDriver)
    teleop = NintendoTeleop(
        NintendoTeleopConfig(
            gripper_mode=NintendoGripperMode.BINARY,
            gripper_cmd_ema_alpha=0.0,
            gripper_cmd_max_rate=0.0,
        )
    )
    teleop.connect()
    teleop._driver.readings.append(reading(buttons=("ZR",)))
    closed = teleop.get_action()
    teleop._driver.readings.append(reading(buttons=("ZL",)))
    opened = teleop.get_action()

    assert closed["gripper"] == pytest.approx(0.0)
    assert opened["gripper"] == pytest.approx(1.0)
    teleop.disconnect()


def test_stale_reading_times_out_to_zero_action(monkeypatch):
    times = iter([10.0, 10.0, 11.0])
    monkeypatch.setattr(
        "lerobot.teleoperators.nintendo.teleop_nintendo.time.perf_counter",
        lambda: next(times),
    )
    monkeypatch.setattr(NintendoTeleop, "driver_cls", DummyNintendoDriver)
    teleop = NintendoTeleop(NintendoTeleopConfig(stale_timeout_s=0.25, stick_deadband=0.0))
    teleop.connect()
    teleop._driver.readings.append(reading(left_stick=(0.0, 1.0), buttons=("R",)))
    first = teleop.get_action()
    second = teleop.get_action()

    assert first["enabled"] is True
    assert second["enabled"] is False
    teleop.disconnect()
