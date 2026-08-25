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
from threading import Event
import types
from unittest.mock import patch

import numpy as np
import pytest

from lerobot.cameras.configs import Cv2Rotation
from lerobot.utils.errors import DeviceAlreadyConnectedError, DeviceNotConnectedError

rs = pytest.importorskip("pyrealsense2")

from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
from lerobot.cameras.realsense.camera_realsense import _color_stream_sensor, _exposure_step_us

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


def test_read_loop_uses_captured_stop_event(monkeypatch):
    config = RealSenseCameraConfig(serial_number_or_name="042", warmup_s=0)
    camera = RealSenseCamera(config)
    stop_event = Event()
    frame = np.zeros((1, 1, 3), dtype=np.uint8)

    class FakeColorFrame:
        def get_data(self):
            return frame

    class FakeFrameSet:
        def get_color_frame(self):
            return FakeColorFrame()

    camera.stop_event = stop_event

    def fake_read_from_hardware():
        stop_event.set()
        camera.stop_event = None
        return FakeFrameSet()

    monkeypatch.setattr(camera, '_read_from_hardware', fake_read_from_hardware)
    monkeypatch.setattr(camera, '_postprocess_image', lambda image, depth_frame=False: image)
    monkeypatch.setattr(camera, '_frame_capture_time_s', lambda *args, **kwargs: 123.0)

    camera._read_loop()

    assert camera.latest_color_frame is not None
    assert camera.latest_timestamp == 123.0


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


class _FakeRange:
    def __init__(self, maximum: float):
        self.min = 1.0
        self.max = maximum
        self.step = 1.0


class _FakeSensor:
    """A sensor that only knows how to hold option values, which is all the controls path uses."""

    def __init__(self, *, exposure_range_max: float, unsupported: tuple = (), publishes_color=True):
        self.exposure_range_max = exposure_range_max
        self.unsupported = unsupported
        self.publishes_color = publishes_color
        self.values: dict = {}
        self.writes: list = []

    def supports(self, option):
        return option not in self.unsupported

    def get_option_range(self, _option):
        return _FakeRange(self.exposure_range_max)

    def get_option(self, option):
        return self.values.get(option, 0.0)

    def set_option(self, option, value):
        self.values[option] = value
        self.writes.append((option, value))

    def get_stream_profiles(self):
        return [types.SimpleNamespace(stream_type=lambda: rs.stream.color)] if self.publishes_color else []


class _FakeDevice:
    """`has_color_sensor=False` is a D405: colour comes out of the stereo module."""

    def __init__(self, sensor, *, has_color_sensor=True):
        self.sensor = sensor
        self.has_color_sensor = has_color_sensor

    def first_color_sensor(self):
        if not self.has_color_sensor:
            raise RuntimeError("Could not find requested sensor type!")
        return self.sensor

    def query_sensors(self):
        return [_FakeSensor(exposure_range_max=1.0, publishes_color=False), self.sensor]


def _camera_with_sensor(sensor, *, fps=60, has_color_sensor=True, **config_kwargs):
    camera = RealSenseCamera(RealSenseCameraConfig(serial_number_or_name="042", **config_kwargs))
    camera.fps = fps
    device = _FakeDevice(sensor, has_color_sensor=has_color_sensor)
    camera.rs_pipeline = object()
    camera.rs_profile = types.SimpleNamespace(get_device=lambda: device)
    return camera


def test_the_colour_sensor_is_found_on_a_device_that_has_none():
    """A D405 raises from first_color_sensor(); its exposure still has to be reachable."""
    sensor = _FakeSensor(exposure_range_max=165000)

    assert _color_stream_sensor(_FakeDevice(sensor, has_color_sensor=False)) is sensor


@pytest.mark.parametrize(
    ("range_max", "expected_step_us"),
    [(165000, 1.0), (10000, 100.0)],
    ids=["stereo-module-microseconds", "uvc-rgb-hundred-microseconds"],
)
def test_the_exposure_unit_is_read_off_the_option_range(range_max, expected_step_us):
    """librealsense never reports the unit, and the two D400 modules disagree about it."""
    assert _exposure_step_us(_FakeSensor(exposure_range_max=range_max)) == expected_step_us


@pytest.mark.parametrize(
    ("range_max", "expected_raw"),
    [(165000, 15000.0), (10000, 150.0)],
    ids=["stereo-module", "uvc-rgb"],
)
def test_a_fixed_exposure_is_written_in_the_sensors_own_units(range_max, expected_raw):
    """15 ms means 15 ms on both modules, or one of the two cameras is silently wrong."""
    sensor = _FakeSensor(exposure_range_max=range_max)
    camera = _camera_with_sensor(sensor, exposure_us=15000, gain=70)

    camera._apply_sensor_controls()

    assert sensor.values[rs.option.enable_auto_exposure] == 0.0
    assert sensor.values[rs.option.exposure] == expected_raw
    assert sensor.values[rs.option.gain] == 70.0


def test_an_unset_exposure_hands_the_sensor_back_to_auto_exposure():
    """The regression this exists for: controls persist on the device between processes.

    Two days of takes were recorded at 15.0 and 23.6 fps because a manual 36.5/42.3 ms exposure
    left behind by another program was inherited at connect. Leaving the config blank has to
    mean "auto", not "whatever is already there".
    """
    sensor = _FakeSensor(exposure_range_max=165000)
    sensor.values[rs.option.enable_auto_exposure] = 0.0
    sensor.values[rs.option.exposure] = 36481.0
    camera = _camera_with_sensor(sensor)

    camera._apply_sensor_controls()

    assert sensor.values[rs.option.enable_auto_exposure] == 1.0


def test_an_exposure_that_cannot_fit_the_frame_period_is_refused():
    """The sensor would negotiate 60 fps and then deliver 27, with nothing reporting it."""
    sensor = _FakeSensor(exposure_range_max=165000)
    camera = _camera_with_sensor(sensor, fps=60, exposure_us=36481)

    with pytest.raises(ValueError, match="27.4 fps"):
        camera._apply_sensor_controls()

    assert rs.option.exposure not in sensor.values


def test_a_clamped_exposure_is_reported_rather_than_assumed(caplog):
    sensor = _FakeSensor(exposure_range_max=165000)
    camera = _camera_with_sensor(sensor, exposure_us=15000)
    # The sensor accepts the write but holds a different value, as a clamped range does.
    sensor.set_option = lambda option, value: sensor.values.__setitem__(option, 8000.0)

    with caplog.at_level(logging.WARNING):
        camera._apply_sensor_controls()

    assert "the sensor holds 8000 us" in caplog.text


def test_an_unsupported_control_is_logged_and_skipped(caplog):
    """Models differ; a missing gain control must not abort a connect."""
    sensor = _FakeSensor(exposure_range_max=165000, unsupported=(rs.option.gain,))
    camera = _camera_with_sensor(sensor, exposure_us=15000, gain=70)

    with caplog.at_level(logging.WARNING):
        camera._apply_sensor_controls()

    assert sensor.values[rs.option.exposure] == 15000.0
    assert "gain is not supported" in caplog.text


def test_gain_without_exposure_is_rejected_at_config_time():
    """Auto exposure drives gain itself, so the pair would not do what it says."""
    with pytest.raises(ValueError, match="`gain` requires `exposure_us`"):
        RealSenseCameraConfig(serial_number_or_name="042", gain=70)


class _RecordingCamera:
    """A camera whose open either fails or succeeds on cue, to drive `connect`'s retry."""

    def __init__(self, outcomes, device=None):
        self.outcomes = list(outcomes)
        self.attempts = 0
        self.device = device
        self.torn_down = 0

    def open(self):
        self.attempts += 1
        outcome = self.outcomes.pop(0)
        if outcome is not None:
            raise outcome


def _camera_with_open_outcomes(outcomes, *, device_present=True, reset_raises=False):
    camera = RealSenseCamera(RealSenseCameraConfig(serial_number_or_name="042"))
    recorder = _RecordingCamera(outcomes)

    def hardware_reset():
        if reset_raises:
            raise RuntimeError("device busy")
        recorder.reset_calls = getattr(recorder, "reset_calls", 0) + 1

    device = types.SimpleNamespace(hardware_reset=hardware_reset)
    camera._open_and_warm_up = recorder.open
    camera._find_device = lambda: (device if device_present else None)
    camera._tear_down_pipeline = lambda: setattr(recorder, "torn_down", recorder.torn_down + 1)
    return camera, recorder


def test_a_camera_that_delivers_no_frames_is_reset_and_retried(caplog, monkeypatch):
    """Observed on a D405: pipeline starts, read thread runs, no frame ever arrives."""
    monkeypatch.setattr("lerobot.cameras.realsense.camera_realsense._DEVICE_RESET_SETTLE_S", 0.0)
    camera, recorder = _camera_with_open_outcomes([TimeoutError("no frame in 1000 ms"), None])

    with caplog.at_level(logging.WARNING):
        camera.connect()

    assert recorder.attempts == 2
    assert recorder.reset_calls == 1
    assert "Resetting the device" in caplog.text


def test_the_retry_happens_only_once():
    """A camera that is still dead after a reset is dead; looping on it hides that."""
    camera, recorder = _camera_with_open_outcomes(
        [TimeoutError("first"), TimeoutError("second")],
    )

    with pytest.raises(TimeoutError, match="second"):
        camera.connect()

    assert recorder.attempts == 2


def test_an_unplugged_camera_is_not_reset(caplog):
    """`hardware_reset` needs a device; saying "replug it" beats a confusing second failure."""
    camera, recorder = _camera_with_open_outcomes([TimeoutError("no frame")], device_present=False)

    with caplog.at_level(logging.ERROR), pytest.raises(TimeoutError):
        camera.connect()

    assert recorder.attempts == 1
    assert "needs a physical replug" in caplog.text


def test_a_device_that_refuses_the_reset_is_reported_not_retried(caplog):
    camera, recorder = _camera_with_open_outcomes([TimeoutError("no frame")], reset_raises=True)

    with caplog.at_level(logging.ERROR), pytest.raises(TimeoutError):
        camera.connect()

    assert recorder.attempts == 1
    assert "needs a physical replug" in caplog.text


def test_a_misconfigured_exposure_is_not_treated_as_a_wedged_camera():
    """Resetting the hardware cannot fix a number in the YAML, and would hide it for 30 s."""
    camera, recorder = _camera_with_open_outcomes([ValueError("exposure_us=36481 does not fit")])

    with pytest.raises(ValueError, match="does not fit"):
        camera.connect()

    assert recorder.attempts == 1
    assert not hasattr(recorder, "reset_calls")
