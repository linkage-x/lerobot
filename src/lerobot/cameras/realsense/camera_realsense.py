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

"""
Provides the RealSenseCamera class for capturing frames from Intel RealSense cameras.
"""

import logging
import time
from collections import deque
from threading import Event, Lock, Thread
from typing import Any

import cv2  # type: ignore  # TODO: add type stubs for OpenCV
import numpy as np  # type: ignore  # TODO: add type stubs for numpy
from numpy.typing import NDArray  # type: ignore  # TODO: add type stubs for numpy.typing

try:
    import pyrealsense2 as rs  # type: ignore  # TODO: add type stubs for pyrealsense2
except Exception as e:
    logging.info(f"Could not import realsense: {e}")

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from ..configs import ColorMode
from ..utils import get_cv2_rotation
from .configuration_realsense import RealSenseCameraConfig

logger = logging.getLogger(__name__)

# Above this the `rs.option.exposure` range cannot be counting in 100 us steps, because the
# implied ceiling (1 s) is already past anything a D400 exposes. See `_exposure_step_us`.
_UVC_EXPOSURE_RANGE_MAX = 10000

# How long a reset device gets to come back on the bus, and how long after that before it is
# opened. Measured on a D405: re-enumerated in ~2 s, and opening it immediately on reappearance
# is what wedges it again. See `_recover_wedged_device`.
_DEVICE_RESET_TIMEOUT_S = 30.0
_DEVICE_RESET_SETTLE_S = 2.0


def _color_stream_sensor(device: Any) -> Any:
    """The sensor that owns the colour stream, including on models that have no colour sensor.

    `device.first_color_sensor()` raises "Could not find requested sensor type!" on a D405,
    whose colour images come out of the "Stereo Module" -- there is no separate "RGB Camera" to
    find, only a sensor that happens to publish colour profiles. Searching the profiles instead
    of asking for a sensor type keeps exposure control and the settings readback working across
    the whole D400 range, rather than silently skipping the models that need them most.
    """
    try:
        return device.first_color_sensor()
    except RuntimeError:
        pass

    for sensor in device.query_sensors():
        for profile in sensor.get_stream_profiles():
            if profile.stream_type() == rs.stream.color:
                return sensor
    raise RuntimeError("no sensor on this device publishes a colour stream profile")


def _exposure_step_us(sensor: Any) -> float:
    """How many microseconds one step of `rs.option.exposure` is worth on this sensor.

    librealsense reports the range but never the unit, and the two D400 modules disagree: the
    UVC RGB module counts in 100 us steps (measured range 1..10000 on a D435i, so a 1 s
    ceiling), the stereo module counts in microseconds (measured 1..165000 on a D405, so a
    165 ms ceiling). The maximum separates them without a model table, and does so safely: a
    sensor whose ceiling really were 10000 us could not expose past 10 ms, which no D400 is.
    """
    return 100.0 if sensor.get_option_range(rs.option.exposure).max <= _UVC_EXPOSURE_RANGE_MAX else 1.0


class RealSenseCamera(Camera):
    """
    Manages interactions with Intel RealSense cameras for frame and depth recording.

    This class provides an interface similar to `OpenCVCamera` but tailored for
    RealSense devices, leveraging the `pyrealsense2` library. It uses the camera's
    unique serial number for identification, offering more stability than device
    indices, especially on Linux. It also supports capturing depth maps alongside
    color frames.

    Use the provided utility script to find available camera indices and default profiles:
    ```bash
    lerobot-find-cameras realsense
    ```

    A `RealSenseCamera` instance requires a configuration object specifying the
    camera's serial number or a unique device name. If using the name, ensure only
    one camera with that name is connected.

    The camera's default settings (FPS, resolution, color mode) from the stream
    profile are used unless overridden in the configuration.

    Example:
        ```python
        from lerobot.cameras.realsense import RealSenseCamera, RealSenseCameraConfig
        from lerobot.cameras import ColorMode, Cv2Rotation

        # Basic usage with serial number
        config = RealSenseCameraConfig(serial_number_or_name="0123456789") # Replace with actual SN
        camera = RealSenseCamera(config)
        camera.connect()

        # Read 1 frame synchronously (blocking)
        color_image = camera.read()

        # Read 1 frame asynchronously (waits for new frame with a timeout)
        async_image = camera.async_read()

        # Get the latest frame immediately (no wait, returns timestamp)
        latest_image, timestamp = camera.read_latest()

        # Example with depth capture and custom settings
        custom_config = RealSenseCameraConfig(
            serial_number_or_name="0123456789", # Replace with actual SN
            fps=30,
            width=1280,
            height=720,
            color_mode=ColorMode.BGR, # Request BGR output
            rotation=Cv2Rotation.NO_ROTATION,
            use_depth=True
        )
        depth_camera = RealSenseCamera(custom_config)
        depth_camera.connect()

        # Read 1 depth frame
        depth_map = depth_camera.read_depth()

        # Example using a unique camera name
        name_config = RealSenseCameraConfig(serial_number_or_name="Intel RealSense D435") # If unique
        name_camera = RealSenseCamera(name_config)
        # ... connect, read, disconnect ...
        ```
    """

    def __init__(self, config: RealSenseCameraConfig):
        """
        Initializes the RealSenseCamera instance.

        Args:
            config: The configuration settings for the camera.
        """

        super().__init__(config)

        self.config = config

        if config.serial_number_or_name.isdigit():
            self.serial_number = config.serial_number_or_name
        else:
            self.serial_number = self._find_serial_number_from_name(config.serial_number_or_name)

        self.fps = config.fps
        self.color_mode = config.color_mode
        self.use_depth = config.use_depth
        self.warmup_s = config.warmup_s

        self.rs_pipeline: rs.pipeline | None = None
        self.rs_profile: rs.pipeline_profile | None = None

        self.thread: Thread | None = None
        self.stop_event: Event | None = None
        self.frame_lock: Lock = Lock()
        self.latest_color_frame: NDArray[Any] | None = None
        self.latest_depth_frame: NDArray[Any] | None = None
        self.latest_timestamp: float | None = None
        self.frame_history: deque[tuple[float, NDArray[Any]]] = deque(maxlen=8)
        self.new_frame_event: Event = Event()
        # Latched once per session so a camera whose timestamps cannot be placed on the host
        # clock says so once instead of on every frame.
        self._device_clock_unusable_logged: bool = False

        self.rotation: int | None = get_cv2_rotation(config.rotation)

        if self.height and self.width:
            self.capture_width, self.capture_height = self.width, self.height
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.capture_width, self.capture_height = self.height, self.width

    def __str__(self) -> str:
        return f"{self.__class__.__name__}({self.serial_number})"

    @property
    def is_connected(self) -> bool:
        """Checks if the camera pipeline is started and streams are active."""
        return self.rs_pipeline is not None and self.rs_profile is not None

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        """
        Connects to the RealSense camera specified in the configuration.

        Initializes the RealSense pipeline, configures the required streams (color
        and optionally depth), starts the pipeline, and validates the actual stream settings.

        Args:
            warmup (bool): If True, waits at connect() time until at least one valid frame
                           has been captured by the background thread. Defaults to True.

        A camera that opens but never delivers is reset once and retried, rather than taken as
        a dead camera: see `_recover_wedged_device`.

        Raises:
            DeviceAlreadyConnectedError: If the camera is already connected.
            ValueError: If the configuration is invalid (e.g., missing serial/name, name not unique).
            ConnectionError: If the camera is found but fails to start the pipeline or no RealSense devices are detected at all.
            RuntimeError: If the pipeline starts but fails to apply requested settings.
        """
        try:
            self._open_and_warm_up()
        except (ConnectionError, TimeoutError) as first_attempt:
            if not self._recover_wedged_device(first_attempt):
                raise
            self._open_and_warm_up()

        logger.info(f"{self} connected.")

    def _open_and_warm_up(self) -> None:
        """One attempt at a live stream: start the pipeline, apply controls, wait for frames.

        Leaves nothing running behind it. A half-open pipeline holds the USB device against
        every later attempt, including the retry that is supposed to rescue this one, so a
        failure anywhere past `start` tears the pipeline back down on the way out.
        """
        self.rs_pipeline = rs.pipeline()
        rs_config = rs.config()
        self._configure_rs_pipeline_config(rs_config)

        try:
            self.rs_profile = self.rs_pipeline.start(rs_config)
        except RuntimeError as e:
            self.rs_profile = None
            self.rs_pipeline = None
            raise ConnectionError(
                f"Failed to open {self}.Run `lerobot-find-cameras realsense` to find available cameras."
            ) from e

        try:
            self._configure_capture_settings()
            # After `_configure_capture_settings`, which is what fills in `self.fps` when the
            # config left it unset -- the exposure check inside is against the negotiated frame
            # period.
            self._apply_sensor_controls()
            self._start_read_thread()

            # NOTE(Steven/Caroline): Enforcing at least one second of warmup as RS cameras need a bit of time before the first read. If we don't wait, the first read from the warmup will raise.
            self.warmup_s = max(self.warmup_s, 1)

            start_time = time.time()
            while time.time() - start_time < self.warmup_s:
                self.async_read(timeout_ms=self.warmup_s * 1000)
                time.sleep(0.1)
            with self.frame_lock:
                if self.latest_color_frame is None or self.use_depth and self.latest_depth_frame is None:
                    raise ConnectionError(f"{self} failed to capture frames during warmup.")
        except BaseException:
            self._tear_down_pipeline()
            raise

    def _tear_down_pipeline(self) -> None:
        """Return to the disconnected state, best effort. Used on the failure paths."""
        self._stop_read_thread()
        if self.rs_pipeline is not None:
            try:
                self.rs_pipeline.stop()
            except RuntimeError as error:
                logger.debug(f"{self}: pipeline stop during teardown failed ({error}).")
        self.rs_pipeline = None
        self.rs_profile = None

    def _recover_wedged_device(self, cause: BaseException) -> bool:
        """Software-reset the camera and wait for it to come back on the bus.

        A D400 can reach a state where the pipeline starts, the read thread runs and no frame
        ever arrives -- observed on a D405 after a session was killed mid-stream, where a bare
        colour stream with no controls touched went 10 s without a frame while its neighbour on
        the same bus was fine. Nothing in software clears that; the device has to re-enumerate.

        `hardware_reset` is the non-privileged way to do it (measured: back on the bus in ~2 s,
        streaming again 437 ms after the next open), so a wedged camera costs a few seconds
        instead of a failed session and a manual replug.

        Returns:
            True if the device reset and re-enumerated, so a retry is worth making.
        """
        logger.warning(f"{self} opened but delivered no frames ({cause}). Resetting the device.")
        self._tear_down_pipeline()

        device = self._find_device()
        if device is None:
            logger.error(f"{self} is not enumerated; it needs a physical replug.")
            return False

        try:
            device.hardware_reset()
        except RuntimeError as error:
            logger.error(f"{self} could not be reset ({error}); it needs a physical replug.")
            return False

        deadline = time.time() + _DEVICE_RESET_TIMEOUT_S
        while time.time() < deadline:
            time.sleep(1.0)
            if self._find_device() is not None:
                # Enumerated is not the same as ready: the firmware comes back before the
                # streaming interfaces do, and opening into that gap wedges it again.
                time.sleep(_DEVICE_RESET_SETTLE_S)
                logger.warning(f"{self} reset and re-enumerated; retrying the connection.")
                return True

        logger.error(f"{self} did not re-enumerate within {_DEVICE_RESET_TIMEOUT_S:.0f} s.")
        return False

    def _find_device(self) -> Any:
        """This camera's device as the driver currently sees it, or None if it is not there."""
        for device in rs.context().query_devices():
            try:
                if str(device.get_info(rs.camera_info.serial_number)) == self.serial_number:
                    return device
            except RuntimeError:
                continue
        return None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """
        Detects available Intel RealSense cameras connected to the system.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries,
            where each dictionary contains 'type', 'id' (serial number), 'name',
            firmware version, USB type, and other available specs, and the default profile properties (width, height, fps, format).

        Raises:
            OSError: If pyrealsense2 is not installed.
            ImportError: If pyrealsense2 is not installed.
        """
        found_cameras_info = []
        context = rs.context()
        devices = context.query_devices()

        for device in devices:
            camera_info = {
                "name": device.get_info(rs.camera_info.name),
                "type": "RealSense",
                "id": device.get_info(rs.camera_info.serial_number),
                "firmware_version": device.get_info(rs.camera_info.firmware_version),
                "usb_type_descriptor": device.get_info(rs.camera_info.usb_type_descriptor),
                "physical_port": device.get_info(rs.camera_info.physical_port),
                "product_id": device.get_info(rs.camera_info.product_id),
                "product_line": device.get_info(rs.camera_info.product_line),
            }

            # Get stream profiles for each sensor
            sensors = device.query_sensors()
            for sensor in sensors:
                profiles = sensor.get_stream_profiles()

                for profile in profiles:
                    if profile.is_video_stream_profile() and profile.is_default():
                        vprofile = profile.as_video_stream_profile()
                        stream_info = {
                            "stream_type": vprofile.stream_name(),
                            "format": vprofile.format().name,
                            "width": vprofile.width(),
                            "height": vprofile.height(),
                            "fps": vprofile.fps(),
                        }
                        camera_info["default_stream_profile"] = stream_info

            found_cameras_info.append(camera_info)

        return found_cameras_info

    def _find_serial_number_from_name(self, name: str) -> str:
        """Finds the serial number for a given unique camera name."""
        camera_infos = self.find_cameras()
        found_devices = [cam for cam in camera_infos if str(cam["name"]) == name]

        if not found_devices:
            available_names = [cam["name"] for cam in camera_infos]
            raise ValueError(
                f"No RealSense camera found with name '{name}'. Available camera names: {available_names}"
            )

        if len(found_devices) > 1:
            serial_numbers = [dev["serial_number"] for dev in found_devices]
            raise ValueError(
                f"Multiple RealSense cameras found with name '{name}'. "
                f"Please use a unique serial number instead. Found SNs: {serial_numbers}"
            )

        serial_number = str(found_devices[0]["serial_number"])
        return serial_number

    def _configure_rs_pipeline_config(self, rs_config: Any) -> None:
        """Creates and configures the RealSense pipeline configuration object."""
        rs.config.enable_device(rs_config, self.serial_number)

        if self.width and self.height and self.fps:
            rs_config.enable_stream(
                rs.stream.color, self.capture_width, self.capture_height, rs.format.rgb8, self.fps
            )
            if self.use_depth:
                rs_config.enable_stream(
                    rs.stream.depth, self.capture_width, self.capture_height, rs.format.z16, self.fps
                )
        else:
            rs_config.enable_stream(rs.stream.color)
            if self.use_depth:
                rs_config.enable_stream(rs.stream.depth)

    @check_if_not_connected
    def _configure_capture_settings(self) -> None:
        """Sets fps, width, and height from device stream if not already configured.

        Uses the color stream profile to update unset attributes. Handles rotation by
        swapping width/height when needed. Original capture dimensions are always stored.

        Raises:
            DeviceNotConnectedError: If device is not connected.
        """

        if self.rs_profile is None:
            raise RuntimeError(f"{self}: rs_profile must be initialized before use.")

        stream = self.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()

        if self.fps is None:
            self.fps = stream.fps()

        if self.width is None or self.height is None:
            actual_width = int(round(stream.width()))
            actual_height = int(round(stream.height()))
            if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE]:
                self.width, self.height = actual_height, actual_width
                self.capture_width, self.capture_height = actual_width, actual_height
            else:
                self.width, self.height = actual_width, actual_height
                self.capture_width, self.capture_height = actual_width, actual_height

    def _set_sensor_option(self, sensor: Any, option: Any, value: float, label: str) -> bool:
        """Write one control, treating "this model does not have it" as information, not failure."""
        if not sensor.supports(option):
            logger.warning(f"{self}: {label} is not supported by this sensor; leaving it alone.")
            return False
        try:
            sensor.set_option(option, value)
        except RuntimeError as error:
            logger.warning(f"{self}: could not set {label} to {value:g} ({error}).")
            return False
        return True

    @check_if_not_connected
    def _apply_sensor_controls(self) -> None:
        """Put exposure and gain into a known state instead of inheriting the device's.

        RealSense controls live on the device, not in the process: whatever the last program to
        touch this camera left behind -- RealSense Viewer, a preview window, an earlier
        recording -- is what the next `pipeline.start` inherits. That is how a workstation spent
        two days recording 640x480@60 profiles that delivered 15.0 and 23.6 fps: both cameras
        had been left on manual exposure at 36.5 ms and 42.3 ms, each longer than the 16.7 ms
        frame period, so neither sensor could physically produce the rate that had been asked
        for and negotiated. Nothing failed; the images just went stale and the recorded frames
        became 65-75% duplicates.

        Hence an unset `exposure_us` means "hand it back to auto exposure", not "leave it
        alone" -- the inherited state is exactly what must not survive a connect.

        Raises:
            ValueError: If a fixed exposure cannot fit inside the negotiated frame period. That
                combination has no valid outcome: the sensor would silently deliver a slower
                stream than the one it just agreed to.
        """
        if self.rs_profile is None:
            raise RuntimeError(f"{self}: rs_profile must be initialized before setting controls.")

        try:
            sensor = _color_stream_sensor(self.rs_profile.get_device())
        except RuntimeError as error:
            logger.warning(f"{self}: cannot reach the colour sensor to set exposure ({error}).")
            return

        if self.config.exposure_us is None:
            self._set_sensor_option(sensor, rs.option.enable_auto_exposure, 1.0, "auto exposure")
            return

        frame_period_us = 1e6 / self.fps if self.fps else None
        if frame_period_us is not None and self.config.exposure_us >= frame_period_us:
            raise ValueError(
                f"{self}: exposure_us={self.config.exposure_us} does not fit in the "
                f"{frame_period_us:.0f} us frame period of a {self.fps} fps stream. The sensor "
                f"would emit at most {1e6 / self.config.exposure_us:.1f} fps. Shorten the "
                f"exposure (and raise `gain` to keep the brightness) or lower `fps`."
            )

        step_us = _exposure_step_us(sensor)
        self._set_sensor_option(sensor, rs.option.enable_auto_exposure, 0.0, "auto exposure")
        applied = self._set_sensor_option(
            sensor, rs.option.exposure, self.config.exposure_us / step_us, "exposure"
        )
        if self.config.gain is not None:
            self._set_sensor_option(sensor, rs.option.gain, float(self.config.gain), "gain")

        if applied:
            # Read back rather than trust the write: the range is clamped silently, and a
            # clamped exposure is the one failure mode this whole method exists to prevent.
            readback_us = float(sensor.get_option(rs.option.exposure)) * step_us
            if abs(readback_us - self.config.exposure_us) > step_us:
                logger.warning(
                    f"{self}: asked for {self.config.exposure_us} us of exposure, the sensor "
                    f"holds {readback_us:.0f} us."
                )
            logger.info(f"{self}: exposure fixed at {readback_us / 1000:.1f} ms, gain {self.config.gain}.")

    @check_if_not_connected
    def get_session_settings(self) -> dict[str, Any]:
        """Return the active colour-stream and sensor controls as JSON-safe values.

        RealSense Viewer can persist controls directly on the device. The recorder
        therefore reads the sensor after its pipeline has started rather than
        treating the YAML stream request as the effective exposure or white
        balance setting. Options differ between camera models and firmware, so an
        unsupported option is listed separately instead of being invented as zero.
        """
        if self.rs_profile is None:
            raise RuntimeError(f"{self}: rs_profile must be initialized before reading session settings.")

        device = self.rs_profile.get_device()
        # Not `device.first_color_sensor()`: that raises on a D405 and took the readback down
        # with it, so the one camera whose controls were wrong reported nothing at all.
        color_sensor = _color_stream_sensor(device)

        def device_info(name: str) -> str | None:
            info = getattr(rs.camera_info, name, None)
            if info is None:
                return None
            try:
                return str(device.get_info(info)) if device.supports(info) else None
            except RuntimeError:
                return None

        controls: dict[str, float] = {}
        unsupported_controls: list[str] = []
        option_names = (
            "enable_auto_exposure", "auto_exposure_priority", "exposure", "gain",
            "enable_auto_white_balance", "white_balance", "backlight_compensation",
            "brightness", "contrast", "saturation", "sharpness", "gamma", "power_line_frequency",
        )
        for name in option_names:
            option = getattr(rs.option, name, None)
            if option is None:
                unsupported_controls.append(name)
                continue
            try:
                if not color_sensor.supports(option):
                    unsupported_controls.append(name)
                    continue
                controls[name] = float(color_sensor.get_option(option))
            except RuntimeError:
                # A model may advertise a control which the active profile cannot
                # read. Preserve that fact without aborting an otherwise valid take.
                unsupported_controls.append(name)

        stream = self.rs_profile.get_stream(rs.stream.color).as_video_stream_profile()
        return {
            "device": {
                "name": device_info("name"),
                "serial_number": device_info("serial_number") or self.serial_number,
                "firmware_version": device_info("firmware_version"),
                "usb_type_descriptor": device_info("usb_type_descriptor"),
            },
            "stream": {
                "width": int(stream.width()), "height": int(stream.height()),
                "fps": int(stream.fps()), "format": str(stream.format()),
            },
            "controls": controls,
            "unsupported_controls": unsupported_controls,
        }

    @check_if_not_connected
    def read_depth(self, timeout_ms: int = 200) -> NDArray[Any]:
        """
        Reads a single frame (depth) synchronously from the camera.

        This is a blocking call. It waits for a coherent set of frames (depth)
        from the camera hardware via the RealSense pipeline.

        Returns:
            np.ndarray: The depth map as a NumPy array (height, width)
                  of type `np.uint16` (raw depth values in millimeters) and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails or frames are invalid.
        """
        if timeout_ms:
            logger.warning(
                f"{self} read() timeout_ms parameter is deprecated and will be removed in future versions."
            )

        if not self.use_depth:
            raise RuntimeError(
                f"Failed to capture depth frame '.read_depth()'. Depth stream is not enabled for {self}."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()

        _ = self.async_read(timeout_ms=10000)

        with self.frame_lock:
            depth_map = self.latest_depth_frame

        if depth_map is None:
            raise RuntimeError("No depth frame available. Ensure camera is streaming.")

        return depth_map

    def _read_from_hardware(self):
        if self.rs_pipeline is None:
            raise RuntimeError(f"{self}: rs_pipeline must be initialized before use.")

        ret, frame = self.rs_pipeline.try_wait_for_frames(timeout_ms=10000)

        if not ret or frame is None:
            raise RuntimeError(f"{self} read failed (status={ret}).")

        return frame

    @check_if_not_connected
    def read(self, color_mode: ColorMode | None = None, timeout_ms: int = 0) -> NDArray[Any]:
        """
        Reads a single frame (color) synchronously from the camera.

        This is a blocking call. It waits for a coherent set of frames (color)
        from the camera hardware via the RealSense pipeline.

        Returns:
            np.ndarray: The captured color frame as a NumPy array
              (height, width, channels), processed according to `color_mode` and rotation.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If reading frames from the pipeline fails or frames are invalid.
            ValueError: If an invalid `color_mode` is requested.
        """

        start_time = time.perf_counter()

        if color_mode is not None:
            logger.warning(
                f"{self} read() color_mode parameter is deprecated and will be removed in future versions."
            )

        if timeout_ms:
            logger.warning(
                f"{self} read() timeout_ms parameter is deprecated and will be removed in future versions."
            )

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        self.new_frame_event.clear()

        frame = self.async_read(timeout_ms=10000)

        read_duration_ms = (time.perf_counter() - start_time) * 1e3
        logger.debug(f"{self} read took: {read_duration_ms:.1f}ms")

        return frame

    def _postprocess_image(self, image: NDArray[Any], depth_frame: bool = False) -> NDArray[Any]:
        """
        Applies color conversion, dimension validation, and rotation to a raw color frame.

        Args:
            image (np.ndarray): The raw image frame (expected RGB format from RealSense).

        Returns:
            np.ndarray: The processed image frame according to `self.color_mode` and `self.rotation`.

        Raises:
            ValueError: If the requested `color_mode` is invalid.
            RuntimeError: If the raw frame dimensions do not match the configured
                          `width` and `height`.
        """

        if self.color_mode and self.color_mode not in (ColorMode.RGB, ColorMode.BGR):
            raise ValueError(
                f"Invalid requested color mode '{self.color_mode}'. Expected {ColorMode.RGB} or {ColorMode.BGR}."
            )

        if depth_frame:
            h, w = image.shape
        else:
            h, w, c = image.shape

            if c != 3:
                raise RuntimeError(f"{self} frame channels={c} do not match expected 3 channels (RGB/BGR).")

        if h != self.capture_height or w != self.capture_width:
            raise RuntimeError(
                f"{self} frame width={w} or height={h} do not match configured width={self.capture_width} or height={self.capture_height}."
            )

        processed_image = image
        if self.color_mode == ColorMode.BGR:
            processed_image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

        if self.rotation in [cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_90_COUNTERCLOCKWISE, cv2.ROTATE_180]:
            processed_image = cv2.rotate(processed_image, self.rotation)

        return processed_image

    # A frame older than this cannot be a real acquisition delay; it means the two clocks were
    # not comparable after all (a wall-clock step, or a timestamp domain that is not what it
    # claimed). Fall back rather than write a fabricated instant.
    _MAX_PLAUSIBLE_FRAME_AGE_S = 1.0

    def _frame_capture_time_s(
        self, color_frame: Any, *, handover_perf_s: float, handover_wall_s: float
    ) -> float:
        """The instant the frame was *acquired*, on the `perf_counter` basis.

        Stamping `perf_counter()` at handover instead measures when this process got around to
        the frame, which folds in the driver's pipeline delay -- and that delay differs per
        model. Measured on the FR3 rig: a D405 hands frames over 4.8 ms after acquisition, a
        D435i 29.1 ms, so two cameras recording the same instant were stamped 24 ms apart. That
        offset is invisible in the data and would be learned as if it were real.

        RealSense reports the acquisition instant on the device clock. With global time enabled
        the driver maps that onto the host wall clock, which makes it comparable to `time.time()`
        -- so the frame's age is `wall_now - frame_timestamp`, and subtracting it from the
        `perf_counter` read taken at the same moment moves it onto the monotonic basis the
        robot's other timestamps use. Both clocks are read at handover, so the subtraction spans
        microseconds and cannot drift.

        A device clock that is *not* on the host timeline (`HARDWARE_CLOCK`, i.e. global time
        off or unsupported) has an arbitrary epoch. Differencing it against a host clock would
        silently splice two unrelated time bases, so this returns the handover instant and says
        so, once, instead.
        """
        domain = color_frame.get_frame_timestamp_domain()
        if domain not in (rs.timestamp_domain.global_time, rs.timestamp_domain.system_time):
            if not self._device_clock_unusable_logged:
                self._device_clock_unusable_logged = True
                logger.warning(
                    "%s: frame timestamp domain is %s, which has no fixed relation to the host "
                    "clock, so capture timestamps fall back to the handover instant and carry "
                    "this camera's pipeline delay (a D435i measured 29 ms). Enable global time "
                    "on the sensor to timestamp acquisition instead.",
                    self,
                    domain,
                )
            return handover_perf_s

        frame_age_s = handover_wall_s - (color_frame.get_timestamp() / 1000.0)
        if not (0.0 <= frame_age_s <= self._MAX_PLAUSIBLE_FRAME_AGE_S):
            if not self._device_clock_unusable_logged:
                self._device_clock_unusable_logged = True
                logger.warning(
                    "%s: frame timestamp implies an age of %.1f ms, which is not a plausible "
                    "acquisition delay; the host wall clock likely stepped. Falling back to the "
                    "handover instant for this frame.",
                    self,
                    frame_age_s * 1e3,
                )
            return handover_perf_s
        return handover_perf_s - frame_age_s

    def _read_loop(self) -> None:
        """
        Internal loop run by the background thread for asynchronous reading.

        On each iteration:
        1. Reads a color frame with 500ms timeout
        2. Stores result in latest_frame and updates timestamp (thread-safe)
        3. Sets new_frame_event to notify listeners

        Stops on DeviceNotConnectedError, logs other errors and continues.
        """
        if self.stop_event is None:
            raise RuntimeError(f"{self}: stop_event is not initialized before starting read loop.")

        failure_count = 0
        while not self.stop_event.is_set():
            try:
                frame = self._read_from_hardware()
                color_frame_raw = frame.get_color_frame()
                # Read the clocks at handover, before any conversion work, so the frame's age
                # can be subtracted from an instant that is still close to the handover.
                handover_perf_s = time.perf_counter()
                handover_wall_s = time.time()
                color_frame = np.asanyarray(color_frame_raw.get_data())
                processed_color_frame = self._postprocess_image(color_frame)

                if self.use_depth:
                    depth_frame_raw = frame.get_depth_frame()
                    depth_frame = np.asanyarray(depth_frame_raw.get_data())
                    processed_depth_frame = self._postprocess_image(depth_frame, depth_frame=True)

                capture_time = self._frame_capture_time_s(
                    color_frame_raw,
                    handover_perf_s=handover_perf_s,
                    handover_wall_s=handover_wall_s,
                )

                with self.frame_lock:
                    self.latest_color_frame = processed_color_frame
                    if self.use_depth:
                        self.latest_depth_frame = processed_depth_frame
                    self.latest_timestamp = capture_time
                    self.frame_history.append((capture_time, processed_color_frame))
                self.new_frame_event.set()
                failure_count = 0

            except DeviceNotConnectedError:
                break
            except Exception as e:
                if failure_count <= 10:
                    failure_count += 1
                    logger.warning(f"Error reading frame in background thread for {self}: {e}")
                else:
                    raise RuntimeError(f"{self} exceeded maximum consecutive read failures.") from e

    def _start_read_thread(self) -> None:
        """Starts or restarts the background read thread if it's not running."""
        self._stop_read_thread()

        self.stop_event = Event()
        self.thread = Thread(target=self._read_loop, args=(), name=f"{self}_read_loop")
        self.thread.daemon = True
        self.thread.start()

    def _stop_read_thread(self) -> None:
        """Signals the background read thread to stop and waits for it to join."""
        if self.stop_event is not None:
            self.stop_event.set()

        if self.thread is not None and self.thread.is_alive():
            self.thread.join(timeout=2.0)

        self.thread = None
        self.stop_event = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.frame_history.clear()
            self.new_frame_event.clear()

    # NOTE(Steven): Missing implementation for depth for now
    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        """
        Reads the latest available frame data (color) asynchronously.

        This method retrieves the most recent color frame captured by the background
        read thread. It does not block waiting for the camera hardware directly,
        but may wait up to timeout_ms for the background thread to provide a frame.
        It is “best effort” under high FPS.

        Args:
            timeout_ms (float): Maximum time in milliseconds to wait for a frame
                to become available. Defaults to 200ms (0.2 seconds).

        Returns:
            np.ndarray:
            The latest captured frame data (color image), processed according to configuration.

        Raises:
            DeviceNotConnectedError: If the camera is not connected.
            TimeoutError: If no frame data becomes available within the specified timeout.
            RuntimeError: If the background thread died unexpectedly or another error occurs.
        """

        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        if not self.new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"Timed out waiting for frame from camera {self} after {timeout_ms} ms. "
                f"Read thread alive: {self.thread.is_alive()}."
            )

        with self.frame_lock:
            frame = self.latest_color_frame
            self.new_frame_event.clear()

        if frame is None:
            raise RuntimeError(f"Internal error: Event set but no frame available for {self}.")

        return frame

    @check_if_not_connected
    def read_latest_with_timestamp(self, max_age_ms: int = 500) -> tuple[NDArray[Any], float]:
        """Return the latest frame and its host monotonic capture timestamp."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")

        with self.frame_lock:
            frame = self.latest_color_frame
            timestamp = self.latest_timestamp

        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")

        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )
        return frame, timestamp

    @check_if_not_connected
    def read_closest(self, timestamp_s: float, max_age_ms: int = 500) -> tuple[NDArray[Any], float]:
        """Return the buffered frame closest to a host monotonic timestamp."""
        if self.thread is None or not self.thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        with self.frame_lock:
            history = tuple(self.frame_history)
        if not history:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        selected_timestamp, selected_frame = min(history, key=lambda sample: abs(sample[0] - timestamp_s))
        age_ms = (time.perf_counter() - selected_timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} closest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )
        return selected_frame, selected_timestamp

    # NOTE(Steven): Missing implementation for depth for now
    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        """Return the most recent (color) frame captured immediately (Peeking).

        This method is non-blocking and returns whatever is currently in the
        memory buffer. The frame may be stale,
        meaning it could have been captured a while ago (hanging camera scenario e.g.).

        Returns:
            NDArray[Any]: The frame image (numpy array).

        Raises:
            TimeoutError: If the latest frame is older than `max_age_ms`.
            DeviceNotConnectedError: If the camera is not connected.
            RuntimeError: If the camera is connected but has not captured any frames yet.
        """

        frame, _timestamp = self.read_latest_with_timestamp(max_age_ms=max_age_ms)
        return frame

    def disconnect(self) -> None:
        """
        Disconnects from the camera, stops the pipeline, and cleans up resources.

        Stops the background read thread (if running) and stops the RealSense pipeline.

        Raises:
            DeviceNotConnectedError: If the camera is already disconnected (pipeline not running).
        """

        if not self.is_connected and self.thread is None:
            raise DeviceNotConnectedError(
                f"Attempted to disconnect {self}, but it appears already disconnected."
            )

        if self.thread is not None:
            self._stop_read_thread()

        if self.rs_pipeline is not None:
            self.rs_pipeline.stop()
            self.rs_pipeline = None
            self.rs_profile = None

        with self.frame_lock:
            self.latest_color_frame = None
            self.latest_depth_frame = None
            self.latest_timestamp = None
            self.frame_history.clear()
            self.new_frame_event.clear()

        logger.info(f"{self} disconnected.")
