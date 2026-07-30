#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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

# Example of running a specific test:
# ```bash
# pytest tests/cameras/test_opencv.py::test_connect
# ```

import logging
from pathlib import Path
import types
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.cameras.configs import Cv2Rotation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

rs = pytest.importorskip("pyrealsense2")

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig

TEST_ARTIFACTS_DIR = Path(__file__).parent.parent / "artifacts" / "cameras"
BAG_FILE_PATH = TEST_ARTIFACTS_DIR / "test_rs.bag"

# NOTE(Steven): For some reason these tests take ~20sec in macOS but only ~2sec in Linux.


def mock_rs_config_enable_device_from_file(rs_config_instance, _sn):
    return rs_config_instance.enable_device_from_file(str(BAG_FILE_PATH), repeat_playback=True)


def mock_rs_config_enable_device_bad_file(rs_config_instance, _sn):
    return rs_config_instance.enable_device_from_file("non_existent_file.bag", repeat_playback=True)


@pytest.fixture(name="patch_realsense", autouse=True)
def fixture_patch_realsense():
    """Automatically mock pyrealsense2.config.enable_device for all tests."""
    with patch(
        "pyrealsense2.config.enable_device", side_effect=mock_rs_config_enable_device_from_file
    ) as mock:
        yield mock


def test_abc_implementation():
    """Instantiation should raise an error if the class doesn't implement abstract methods/properties."""
    config = RealSenseCameraConfig(serial_number_or_name="042")
    _ = RealSenseCamera(config)


def test_connect():
    config = RealSenseCameraConfig(serial_number_or_name="042", warmup_s=0)

    with RealSenseCamera(config) as camera:
        assert camera.is_connected


def test_connect_already_connected():
    config = RealSenseCameraConfig(serial_number_or_name="042", warmup_s=0)
    with RealSenseCamera(config) as camera, pytest.raises(DeviceAlreadyConnectedError):
        camera.connect(warmup=False)


def test_connect_invalid_camera_path(patch_realsense):
    patch_realsense.side_effect = mock_rs_config_enable_device_bad_file
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)


def test_invalid_width_connect():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=99999, height=480, fps=30)
    camera = RealSenseCamera(config)

    with pytest.raises(ConnectionError):
        camera.connect(warmup=False)


def test_read():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30, warmup_s=0)
    with RealSenseCamera(config) as camera:
        img = camera.read()
        assert isinstance(img, np.ndarray)


# TODO(Steven): Fix this test for the latest version of pyrealsense2.
@pytest.mark.skip("Skipping test: pyrealsense2 version > 2.55.1.6486")
def test_read_depth():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30, use_depth=True)
    camera = RealSenseCamera(config)
    camera.connect(warmup=False)

    img = camera.read_depth(timeout_ms=2000)  # NOTE(Steven): Reading depth takes longer in CI environments.
    assert isinstance(img, np.ndarray)


def test_read_before_connect():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read()


def test_disconnect():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)
    camera.connect(warmup=False)

    camera.disconnect()

    assert not camera.is_connected


def test_disconnect_before_connect():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        camera.disconnect()


def test_async_read():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30, warmup_s=0)

    with RealSenseCamera(config) as camera:
        img = camera.async_read()

        assert camera.thread is not None
        assert camera.thread.is_alive()
        assert isinstance(img, np.ndarray)


def test_async_read_timeout():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30, warmup_s=0)
    with RealSenseCamera(config) as camera, pytest.raises(TimeoutError):
        camera.async_read(timeout_ms=0)  # consumes any available frame by then
        camera.async_read(timeout_ms=0)  # request immediately another one


def test_async_read_before_connect():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.async_read()


def test_read_latest():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30, warmup_s=0)
    with RealSenseCamera(config) as camera:
        img = camera.read()
        latest = camera.read_latest()

        assert isinstance(latest, np.ndarray)
        assert latest.shape == img.shape


def test_read_latest_high_frequency():
    config = RealSenseCameraConfig(serial_number_or_name="042", width=640, height=480, fps=30, warmup_s=0)
    with RealSenseCamera(config) as camera:
        # prime with one read to ensure frames are available
        ref = camera.read()

        for _ in range(20):
            latest = camera.read_latest()
            assert isinstance(latest, np.ndarray)
            assert latest.shape == ref.shape


def test_read_latest_before_connect():
    config = RealSenseCameraConfig(serial_number_or_name="042")
    camera = RealSenseCamera(config)

    with pytest.raises(DeviceNotConnectedError):
        _ = camera.read_latest()


def test_read_latest_too_old():
    config = RealSenseCameraConfig(serial_number_or_name="042")

    with RealSenseCamera(config) as camera:
        # prime to ensure frames are available
        _ = camera.read()

        with pytest.raises(TimeoutError):
            _ = camera.read_latest(max_age_ms=0)  # immediately too old


@pytest.mark.parametrize(
    "rotation",
    [
        Cv2Rotation.NO_ROTATION,
        Cv2Rotation.ROTATE_90,
        Cv2Rotation.ROTATE_180,
        Cv2Rotation.ROTATE_270,
    ],
    ids=["no_rot", "rot90", "rot180", "rot270"],
)
def test_rotation(rotation):
    config = RealSenseCameraConfig(serial_number_or_name="042", rotation=rotation, warmup_s=0)
    with RealSenseCamera(config) as camera:
        img = camera.read()
        assert isinstance(img, np.ndarray)

        if rotation in (Cv2Rotation.ROTATE_90, Cv2Rotation.ROTATE_270):
            assert camera.width == 480
            assert camera.height == 640
            assert img.shape[:2] == (640, 480)
        else:
            assert camera.width == 640
            assert camera.height == 480
            assert img.shape[:2] == (480, 640)


def _frame_stub(domain, timestamp_ms):
    """Minimal stand-in for a pyrealsense2 color frame."""
    return types.SimpleNamespace(
        get_frame_timestamp_domain=lambda: domain,
        get_timestamp=lambda: timestamp_ms,
    )


def _unconnected_camera():
    return RealSenseCamera(RealSenseCameraConfig(serial_number_or_name="042"))


def test_capture_time_reports_acquisition_not_handover():
    """The pipeline delay differs per model and must not end up in the timestamp.

    Measured on the FR3 rig: a D405 hands frames over 4.8 ms after acquisition and a D435i
    29.1 ms, so stamping the handover put 24 ms between two cameras that saw the same instant.
    """
    camera = _unconnected_camera()
    handover_wall_s = 1_700_000_000.0
    handover_perf_s = 5_000.0
    age_s = 0.0291

    capture_time = camera._frame_capture_time_s(
        _frame_stub(rs.timestamp_domain.global_time, (handover_wall_s - age_s) * 1e3),
        handover_perf_s=handover_perf_s,
        handover_wall_s=handover_wall_s,
    )

    assert capture_time == pytest.approx(handover_perf_s - age_s, abs=1e-6)
    # Stays on the monotonic basis the robot's other timestamps use.
    assert capture_time < handover_perf_s


def test_capture_time_refuses_to_splice_a_device_clock_onto_the_host_clock(caplog):
    """HARDWARE_CLOCK has an arbitrary epoch; differencing it against the host clock is meaningless."""
    camera = _unconnected_camera()

    with caplog.at_level(logging.WARNING):
        capture_time = camera._frame_capture_time_s(
            _frame_stub(rs.timestamp_domain.hardware_clock, 12345.0),
            handover_perf_s=5_000.0,
            handover_wall_s=1_700_000_000.0,
        )

    assert capture_time == 5_000.0
    assert "no fixed relation to the host clock" in caplog.text


def test_capture_time_falls_back_when_the_wall_clock_steps(caplog):
    """A negative or absurd age means the two clocks stopped being comparable."""
    camera = _unconnected_camera()
    handover_wall_s = 1_700_000_000.0

    with caplog.at_level(logging.WARNING):
        capture_time = camera._frame_capture_time_s(
            # Frame timestamped 5 s in the future: the host clock stepped backwards.
            _frame_stub(rs.timestamp_domain.global_time, (handover_wall_s + 5.0) * 1e3),
            handover_perf_s=5_000.0,
            handover_wall_s=handover_wall_s,
        )

    assert capture_time == 5_000.0
    assert "not a plausible" in caplog.text


def test_capture_time_warns_once_per_camera(caplog):
    """A per-frame warning at 30 fps would bury the log it is trying to surface."""
    camera = _unconnected_camera()

    with caplog.at_level(logging.WARNING):
        for _ in range(5):
            camera._frame_capture_time_s(
                _frame_stub(rs.timestamp_domain.hardware_clock, 12345.0),
                handover_perf_s=5_000.0,
                handover_wall_s=1_700_000_000.0,
            )

    assert caplog.text.count("no fixed relation to the host clock") == 1
