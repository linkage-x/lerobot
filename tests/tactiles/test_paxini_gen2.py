#!/usr/bin/env python

from __future__ import annotations

import numpy as np

from lerobot.tactiles.paxini_gen2 import PaxiniGen2OmegaTactile, PaxiniGen2OmegaTactileConfig


class FakePaxiniWrapper:
    def __init__(self, frames_by_connect_id: dict[int, np.ndarray]):
        self.frames_by_connect_id = frames_by_connect_id

    def read_module_sensing_data(self, connect_id: int) -> np.ndarray:
        return self.frames_by_connect_id[connect_id]


def test_paxini_config_sets_total_taxels_from_connect_ids():
    config = PaxiniGen2OmegaTactileConfig(
        serial_port="/dev/ttyACM0",
        baudrate=460800,
        timeout=1.0,
        control_mode=5,
        model_name="GEN2-IP-L5325",
        connect_ids=[6, 10],
        fps=120,
    )

    assert config.num_taxels == 240
    assert config.num_dimensions == 3


def test_paxini_tactile_concatenates_frames_from_all_connect_ids():
    config = PaxiniGen2OmegaTactileConfig(
        serial_port="/dev/ttyACM0",
        baudrate=460800,
        timeout=1.0,
        control_mode=5,
        model_name="GEN2-IP-L5325",
        connect_ids=[6, 10],
        fps=120,
    )
    tactile = PaxiniGen2OmegaTactile(config)

    left = np.full((120, 3), 1, dtype=np.int16)
    right = np.full((120, 3), 2, dtype=np.int16)
    tactile.wrapper = FakePaxiniWrapper({6: left, 10: right})

    frame = tactile._read_from_hardware()

    assert frame.shape == (240, 3)
    np.testing.assert_array_equal(frame[:120], left)
    np.testing.assert_array_equal(frame[120:], right)
