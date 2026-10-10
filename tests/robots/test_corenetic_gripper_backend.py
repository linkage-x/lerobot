from __future__ import annotations

from types import SimpleNamespace

import pytest

from lerobot.robots.franka_research3 import backends


class FakeBoxClient:
    def __init__(self, config):
        self.config = config
        self.calls: list[tuple[str, float | int] | tuple[str]] = []

    def start(self) -> bool:
        self.calls.append(("start",))
        return True

    def read(self):
        return {"sensors": {"box_gripper": {"distance_m": 0.063}}}

    def set_clamp_pos(self, width_m: float) -> int:
        self.calls.append(("set_clamp_pos", width_m))
        return 0

    def set_mode(self, mode: int) -> int:
        self.calls.append(("set_mode", mode))
        return 0

    def stop(self) -> None:
        self.calls.append(("stop",))


def test_corenetic_connect_reads_real_width_before_control_and_holds_it(monkeypatch) -> None:
    clients: list[FakeBoxClient] = []

    def make_client(config, **kwargs):
        client = FakeBoxClient(config)
        client.kwargs = kwargs
        clients.append(client)
        return client

    module = SimpleNamespace(
        BoxClientConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        BoxClient=make_client,
    )
    monkeypatch.setattr(backends, "_import_box_client", lambda: module)

    driver = backends.CoreneticGripperHardwareDriver(release_mode_on_disconnect=False)
    driver.connect()

    client = clients[-1]
    assert client.config.startup_mode == 0
    assert client.calls == [
        ("start",),
        ("set_clamp_pos", pytest.approx(0.063)),
        ("set_mode", 1),
        ("set_clamp_pos", pytest.approx(0.063)),
    ]
    assert driver.get_position() == pytest.approx(0.7)

    driver.disconnect()


def test_corenetic_driver_pins_the_arm_box_and_says_so_when_it_is_absent(monkeypatch) -> None:
    seen: list[dict] = []

    class _Absent(FakeBoxClient):
        def start(self) -> bool:
            return False

    def make_client(config, **kwargs):
        seen.append(kwargs)
        return _Absent(config)

    module = SimpleNamespace(BoxClientConfig=lambda **kwargs: SimpleNamespace(**kwargs), BoxClient=make_client)
    monkeypatch.setattr(backends, "_import_box_client", lambda: module)

    driver = backends.CoreneticGripperHardwareDriver(device_id=596523097)
    with pytest.raises(ConnectionError, match="device_id=596523097"):
        driver.connect()
    assert seen == [{"device_id": 596523097}]


def test_corenetic_driver_retakes_control_when_the_box_restarts(monkeypatch, caplog) -> None:
    """10-10: the BOX lost power as the wrist swung to the start and came back in collection mode."""

    class _Restarting(FakeBoxClient):
        clock_us = 553_000_000

        def read(self):
            return {"sensors": {"box_gripper": {"distance_m": 0.0887, "timestamp": self.clock_us}}}

    clients: list[_Restarting] = []

    def make_client(config, **kwargs):
        clients.append(_Restarting(config))
        return clients[-1]

    module = SimpleNamespace(BoxClientConfig=lambda **kwargs: SimpleNamespace(**kwargs), BoxClient=make_client)
    monkeypatch.setattr(backends, "_import_box_client", lambda: module)
    driver = backends.CoreneticGripperHardwareDriver(release_mode_on_disconnect=False, command_rate_limit_hz=None)
    driver.connect()
    client = clients[-1]

    client.clock_us += 50_000  # running normally: nothing to do
    driver.set_position(0.986)
    assert driver.reboot_count == 0

    client.calls.clear()
    client.clock_us = 20_500_000  # restarted 20.5 s ago
    driver.set_position(0.0)

    assert driver.reboot_count == 1
    assert "Corenetic BOX restarted" in caplog.text
    # Control mode first and the held target again, then the new command lands in control mode.
    assert client.calls[:2] == [("set_mode", 1), ("set_clamp_pos", pytest.approx(0.0887))]
    assert client.calls[-1] == ("set_clamp_pos", pytest.approx(0.0))
