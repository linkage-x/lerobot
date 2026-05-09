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
    accel_g: tuple[float, float, float] = (0.0, 0.0, -1.0),
    gyro_dps: tuple[float, float, float] = (0.0, 0.0, 0.0),
) -> NintendoControllerReading:
    return NintendoControllerReading(
        controller_type=controller_type,
        left_stick=left_stick,
        right_stick=right_stick,
        buttons=frozenset(buttons),
        accel_g=accel_g,
        gyro_dps=gyro_dps,
    )


@pytest.fixture
def teleop(monkeypatch):
    DummyNintendoDriver.instances = []
    DummyNintendoDriver.queued_readings = []
    monkeypatch.setattr(NintendoTeleop, "driver_cls", DummyNintendoDriver)
    device = NintendoTeleop(
        NintendoTeleopConfig(
            controller=NintendoController.PRO,
            translation_scale=1.0,
            vertical_scale=1.0,
            rotation_scale=1.0,
            stick_deadband=0.0,
            imu_accel_deadband_g=0.0,
            imu_gyro_deadband_dps=0.0,
            imu_stationary_gyro_dps=0.0,
            imu_stationary_accel_norm_tolerance_g=0.0,
            imu_velocity_decay=1.0,
            max_imu_dt_s=1.0,
            max_step_pos_m=1.0,
            max_step_rot_rad=1.0,
            gripper_move_time=0.0,
            gripper_step=0.2,
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
    assert cfg.clutch_buttons == ("ZL",)
    assert cfg.experimental_imu_translation is False


def test_get_action_uses_sticks_for_translation_and_zl_clutch_for_rotation(monkeypatch, teleop):
    now = [10.0]
    monkeypatch.setattr("lerobot.teleoperators.nintendo.teleop_nintendo.time.perf_counter", lambda: now[0])
    teleop.connect()
    teleop._driver.readings.append(
        reading(
            buttons=("ZL",),
            left_stick=(0.25, 0.5),
            right_stick=(0.0, -0.75),
            accel_g=(0.0, 0.0, -1.0),
            gyro_dps=(0.0, 0.0, 0.0),
        )
    )
    first = teleop.get_action()

    now[0] = 10.1
    teleop._driver.readings.append(
        reading(
            buttons=("ZL",),
            left_stick=(0.25, 0.5),
            right_stick=(0.0, -0.75),
            accel_g=(0.2, 0.0, -1.0),
            gyro_dps=(0.0, 30.0, 30.0),
        )
    )
    second = teleop.get_action()

    assert first["enabled"] is True
    assert first["target_x"] == pytest.approx(0.5)
    assert first["target_y"] == pytest.approx(-0.25)
    assert first["target_z"] == pytest.approx(-0.75)
    assert second["enabled"] is True
    assert second["target_x"] == pytest.approx(0.5)
    assert second["target_y"] == pytest.approx(-0.25)
    assert second["target_z"] == pytest.approx(-0.75)
    assert second["target_wy"] == pytest.approx(-30.0 * 0.1 * 3.141592653589793 / 180.0)
    assert second["target_wz"] == pytest.approx(-30.0 * 0.1 * 3.141592653589793 / 180.0)


def test_experimental_imu_translation_can_be_enabled(monkeypatch, teleop):
    teleop.config.experimental_imu_translation = True
    now = [10.0]
    monkeypatch.setattr("lerobot.teleoperators.nintendo.teleop_nintendo.time.perf_counter", lambda: now[0])
    teleop.connect()
    teleop._driver.readings.append(
        reading(buttons=("ZL",), accel_g=(0.0, 0.0, -1.0), gyro_dps=(0.0, 0.0, 0.0))
    )
    assert teleop.get_action()["target_x"] == pytest.approx(0.0)

    now[0] = 10.1
    teleop._driver.readings.append(
        reading(buttons=("ZL",), accel_g=(0.2, 0.0, -1.0), gyro_dps=(0.0, 0.0, 0.0))
    )
    action = teleop.get_action()

    assert action["enabled"] is True
    assert action["target_x"] == pytest.approx(0.2 * 9.80665 * 0.1 * 0.1)


def test_stale_imu_report_does_not_repeat_rotation_delta(monkeypatch, teleop):
    now = [10.0]
    monkeypatch.setattr("lerobot.teleoperators.nintendo.teleop_nintendo.time.perf_counter", lambda: now[0])
    teleop.connect()
    teleop._driver.readings.append(
        reading(buttons=("ZL",), accel_g=(0.0, 0.0, -1.0), gyro_dps=(0.0, 0.0, 0.0))
    )
    assert teleop.get_action()["target_wz"] == pytest.approx(0.0)

    now[0] = 10.1
    teleop._driver.readings.append(
        reading(buttons=("ZL",), accel_g=(0.0, 0.0, -1.0), gyro_dps=(0.0, 0.0, 30.0))
    )
    fresh = teleop.get_action()
    now[0] = 10.105
    stale = teleop.get_action()

    assert fresh["enabled"] is True
    assert fresh["target_wz"] < 0.0
    assert stale["enabled"] is True
    assert stale["target_wz"] == pytest.approx(0.0)


def test_stationary_gyro_after_noisy_clutch_baseline_does_not_drift(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("lerobot.teleoperators.nintendo.teleop_nintendo.time.perf_counter", lambda: now[0])
    monkeypatch.setattr(NintendoTeleop, "driver_cls", DummyNintendoDriver)
    teleop = NintendoTeleop(
        NintendoTeleopConfig(
            imu_gyro_deadband_dps=0.0,
            imu_stationary_gyro_dps=3.0,
            imu_stationary_accel_norm_tolerance_g=0.08,
            gripper_cmd_ema_alpha=0.0,
            gripper_cmd_max_rate=0.0,
        )
    )
    teleop.connect()
    teleop._driver.readings.append(reading(buttons=("ZL",), gyro_dps=(10.0, 0.0, 0.0)))
    assert teleop.get_action()["target_wx"] == pytest.approx(0.0)

    now[0] = 10.1
    teleop._driver.readings.append(reading(buttons=("ZL",), gyro_dps=(0.5, 0.0, 0.0)))
    action = teleop.get_action()

    assert action["enabled"] is True
    assert action["target_wx"] == pytest.approx(0.0)
    assert action["target_wy"] == pytest.approx(0.0)
    assert action["target_wz"] == pytest.approx(0.0)
    teleop.disconnect()


def test_stationary_gravity_vector_change_does_not_integrate_translation(monkeypatch):
    now = [10.0]
    monkeypatch.setattr("lerobot.teleoperators.nintendo.teleop_nintendo.time.perf_counter", lambda: now[0])
    monkeypatch.setattr(NintendoTeleop, "driver_cls", DummyNintendoDriver)
    teleop = NintendoTeleop(
        NintendoTeleopConfig(
            imu_accel_deadband_g=0.0,
            imu_gyro_deadband_dps=0.0,
            experimental_imu_translation=True,
            imu_stationary_gyro_dps=3.0,
            imu_stationary_accel_norm_tolerance_g=0.08,
            imu_velocity_decay=1.0,
            max_imu_dt_s=1.0,
            gripper_cmd_ema_alpha=0.0,
            gripper_cmd_max_rate=0.0,
        )
    )
    teleop.connect()
    teleop._driver.readings.append(reading(buttons=("ZL",), accel_g=(0.0, 0.0, -1.0)))
    assert teleop.get_action()["target_x"] == pytest.approx(0.0)

    now[0] = 10.1
    teleop._driver.readings.append(reading(buttons=("ZL",), accel_g=(0.05, 0.0, -0.99875)))
    action = teleop.get_action()

    assert action["enabled"] is True
    assert action["target_x"] == pytest.approx(0.0)
    assert action["target_y"] == pytest.approx(0.0)
    assert action["target_z"] == pytest.approx(0.0)
    teleop.disconnect()


def test_get_action_requires_zl_button_clutch_and_resets_imu_origin(monkeypatch, teleop):
    now = [10.0]
    monkeypatch.setattr("lerobot.teleoperators.nintendo.teleop_nintendo.time.perf_counter", lambda: now[0])
    teleop.connect()
    teleop._driver.readings.append(reading(buttons=("ZL",), accel_g=(0.0, 0.0, -1.0)))
    assert teleop.get_action()["enabled"] is True
    now[0] = 10.1
    teleop._driver.readings.append(reading(buttons=("ZL",), gyro_dps=(0.0, 0.0, 30.0)))
    moved = teleop.get_action()
    now[0] = 10.2
    teleop._driver.readings.append(reading(buttons=(), gyro_dps=(0.0, 0.0, 30.0)))
    released = teleop.get_action()
    now[0] = 10.3
    teleop._driver.readings.append(reading(buttons=("ZL",), gyro_dps=(0.0, 0.0, 30.0)))
    recentered = teleop.get_action()

    assert moved["target_wz"] < 0.0
    assert released["enabled"] is False
    assert released["target_wz"] == pytest.approx(0.0)
    assert recentered["enabled"] is True
    assert recentered["target_wz"] == pytest.approx(0.0)


def test_r_button_closes_gripper_and_release_opens_it(teleop):
    teleop.connect()
    teleop._driver.readings.append(reading(buttons=("R",)))
    closed = teleop.get_action()
    teleop._driver.readings.append(reading(buttons=()))
    opened = teleop.get_action()

    assert closed["enabled"] is False
    assert closed["gripper"] == pytest.approx(0.8)
    assert opened["gripper"] == pytest.approx(1.0)


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
    teleop._driver.readings.append(reading(buttons=("R",)))
    closed = teleop.get_action()
    teleop._driver.readings.append(reading(buttons=()))
    opened = teleop.get_action()

    assert closed["gripper"] == pytest.approx(0.0)
    assert opened["gripper"] == pytest.approx(1.0)
    teleop.disconnect()


def test_stale_reading_times_out_to_zero_action(monkeypatch):
    now = [10.0]
    monkeypatch.setattr(
        "lerobot.teleoperators.nintendo.teleop_nintendo.time.perf_counter",
        lambda: now[0],
    )
    monkeypatch.setattr(NintendoTeleop, "driver_cls", DummyNintendoDriver)
    teleop = NintendoTeleop(
        NintendoTeleopConfig(
            stale_timeout_s=0.25,
            imu_accel_deadband_g=0.0,
            imu_gyro_deadband_dps=0.0,
        )
    )
    teleop.connect()
    teleop._driver.readings.append(reading(buttons=("ZL",)))
    first = teleop.get_action()
    now[0] = 11.0
    second = teleop.get_action()

    assert first["enabled"] is True
    assert second["enabled"] is False
    teleop.disconnect()
