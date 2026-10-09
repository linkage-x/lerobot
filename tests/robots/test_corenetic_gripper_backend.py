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
