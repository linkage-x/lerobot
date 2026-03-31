#!/usr/bin/env python

from __future__ import annotations

from lerobot.cameras.configs import ColorMode
from lerobot.scripts.lerobot_find_cameras import create_camera_instance


class _FakeHikrobotCamera:
    last_config = None

    def __init__(self, config) -> None:
        type(self).last_config = config
        self.is_connected = False

    def connect(self, warmup: bool = True) -> None:
        self.is_connected = True

    def disconnect(self) -> None:
        self.is_connected = False


def test_create_camera_instance_uses_hikrobot_transport_layer(monkeypatch):
    monkeypatch.setattr("lerobot.scripts.lerobot_find_cameras.HikrobotCamera", _FakeHikrobotCamera)

    cam_dict = create_camera_instance(
        {
            "type": "Hikrobot",
            "id": "DA9342673",
            "transport_layer": "gige",
        }
    )

    assert cam_dict is not None
    assert _FakeHikrobotCamera.last_config is not None
    assert _FakeHikrobotCamera.last_config.serial == "DA9342673"
    assert _FakeHikrobotCamera.last_config.transport_layer == "gige"
    assert _FakeHikrobotCamera.last_config.color_mode == ColorMode.RGB
