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

"""GMSL2 camera driver for the SENSING SG16A-AGTH-G3Y-A1 adapter on NVIDIA Jetson.

Capture is done through GStreamer + PyGObject (``gi``) rather than OpenCV's optional
GStreamer backend so this driver works on stock JetPack Python without rebuilding cv2.
Hardware-synchronous capture is configured via ``v4l2-ctl``.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
import time
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

import numpy as np
from numpy.typing import NDArray

from lerobot.utils.decorators import check_if_already_connected, check_if_not_connected
from lerobot.utils.errors import DeviceNotConnectedError

from ..camera import Camera
from .configuration_gmsl2 import ColorMode, Cv2Rotation, Gmsl2CameraConfig

logger = logging.getLogger(__name__)


_GST_INIT_LOCK = threading.Lock()
_GST_INITIALIZED = False


def _ensure_gst_initialized() -> Any:
    """Import and initialise GStreamer + PyGObject lazily.

    Returns the ``Gst`` module so callers don't have to re-import.
    """
    global _GST_INITIALIZED
    try:
        import gi  # type: ignore  # noqa: PLC0415

        gi.require_version("Gst", "1.0")
        from gi.repository import GLib, Gst  # type: ignore  # noqa: PLC0415, F401
    except (ImportError, ValueError) as exc:  # pragma: no cover - import-time failure
        raise RuntimeError(
            "PyGObject (gi) with GStreamer 1.0 is required for the GMSL2 camera. "
            "On JetPack: `sudo apt install python3-gi python3-gst-1.0 gstreamer1.0-tools "
            "gstreamer1.0-plugins-good`. "
            "For Argus-based capture you also need `nvidia-l4t-jetson-multimedia-api` "
            "(provides `nvarguscamerasrc`)."
        ) from exc

    with _GST_INIT_LOCK:
        if not _GST_INITIALIZED:
            Gst.init(None)
            _GST_INITIALIZED = True
    return Gst


class Gmsl2Camera(Camera):
    """SENSING GMSL2 camera attached to a Jetson via the SG16A_AGTH_G3Y_A1 adapter.

    The camera is opened through a GStreamer pipeline. The actual capture happens in a
    background thread that pulls samples from an ``appsink``; ``read()`` /
    ``async_read()`` / ``read_latest()`` share one thread-safe latest-frame buffer.

    Hardware-trigger configuration (``sensor_mode``, ``trig_pin``, ``trig_mode``) is
    applied via ``v4l2-ctl`` at ``connect()`` time when ``apply_sync_at_connect`` is
    True. The PWM that drives the trigger pin -- if any -- is *not* set up here; use
    ``tools/gmsl2/setup_sync.sh`` to configure it once per boot.
    """

    def __init__(self, config: Gmsl2CameraConfig):
        super().__init__(config)
        self.config = config
        self.color_mode = config.color_mode
        self.rotation = config.rotation
        self.warmup_s = config.warmup_s
        self.timeout_ms = config.timeout_ms

        # Capture state.
        self._gst: Any = None  # The Gst module after initialisation.
        self._pipeline: Any = None
        self._appsink: Any = None
        self._bus_thread: Thread | None = None
        self._read_thread: Thread | None = None
        self._stop_event: Event | None = None
        self._frame_lock: Lock = Lock()
        self._latest_frame: NDArray[Any] | None = None
        self._latest_timestamp: float | None = None
        self._new_frame_event: Event = Event()
        self._connected = False

    def __str__(self) -> str:
        if self.config.sensor_id is not None:
            return f"Gmsl2Camera(sensor_id={self.config.sensor_id}, pipeline={self.config.pipeline})"
        return f"Gmsl2Camera(device={self.config.device}, pipeline={self.config.pipeline})"

    @property
    def is_connected(self) -> bool:
        return self._connected and self._pipeline is not None

    @staticmethod
    def find_cameras() -> list[dict[str, Any]]:
        """List /dev/videoN nodes that look like GMSL2 cameras.

        The SG16A driver registers up to 16 contiguous V4L2 nodes starting at
        ``/dev/video0``. Returning the raw list lets the user pick the right ones --
        we deliberately do not open the device here, since the kernel driver is not
        always safe to re-open while ``nvargus-daemon`` may already hold it.
        """
        found: list[dict[str, Any]] = []
        for path in sorted(Path("/dev").glob("video*"), key=lambda p: p.name):
            try:
                sensor_id = int(str(path.name).removeprefix("video"))
            except ValueError:
                continue
            found.append(
                {
                    "name": f"GMSL2 video node {path.name}",
                    "type": "GMSL2",
                    "id": str(path),
                    "sensor_id": sensor_id,
                }
            )
        return found

    # ---------------------------------------------------------------- connect ----

    @check_if_already_connected
    def connect(self, warmup: bool = True) -> None:
        self._gst = _ensure_gst_initialized()

        if self.config.apply_sync_at_connect:
            self._apply_v4l2_sync()

        pipeline_str = self._build_pipeline_str()
        logger.info("%s launching GStreamer pipeline: %s", self, pipeline_str)
        pipeline = self._gst.parse_launch(pipeline_str)
        if pipeline is None:  # pragma: no cover
            raise ConnectionError(f"{self} failed to parse pipeline: {pipeline_str}")

        appsink = pipeline.get_by_name("appsink0")
        if appsink is None:  # pragma: no cover
            raise ConnectionError(f"{self} pipeline is missing appsink0.")

        # `try-pull-sample` is exposed as a GObject signal on GstAppSink; the
        # property must be on for the signal to be emitted.
        appsink.set_property("emit-signals", True)
        appsink.set_property("sync", False)
        appsink.set_property("drop", True)
        appsink.set_property("max-buffers", 1)

        self._pipeline = pipeline
        self._appsink = appsink

        ret = pipeline.set_state(self._gst.State.PLAYING)
        if ret == self._gst.StateChangeReturn.FAILURE:
            self._teardown_pipeline()
            raise ConnectionError(f"{self} failed to start pipeline (set_state -> FAILURE).")

        self._stop_event = Event()
        self._new_frame_event.clear()
        self._read_thread = Thread(target=self._read_loop, name=f"{self}_read_loop", daemon=True)
        self._read_thread.start()
        self._bus_thread = Thread(target=self._bus_loop, name=f"{self}_bus_loop", daemon=True)
        self._bus_thread.start()

        self._connected = True

        if warmup and self.warmup_s > 0:
            start = time.time()
            success = False
            while time.time() - start < self.warmup_s + self.timeout_ms / 1000.0:
                try:
                    self.async_read(timeout_ms=self.timeout_ms)
                    success = True
                    break
                except TimeoutError:
                    continue
            if not success:
                self.disconnect()
                raise ConnectionError(f"{self} did not produce frames during warmup.")
            # Continue draining for the rest of the warmup window so AE/AWB stabilise.
            remaining = self.warmup_s - (time.time() - start)
            if remaining > 0:
                time.sleep(remaining)

        logger.info("%s connected.", self)

    # ---------------------------------------------------------------- pipeline ----

    def _build_pipeline_str(self) -> str:
        fps = int(self.fps) if self.fps else 0
        width = int(self.width) if self.width else 0
        height = int(self.height) if self.height else 0
        if not (fps and width and height):
            raise ValueError(
                f"{self} requires explicit fps/width/height in the config "
                f"(got fps={self.fps}, width={self.width}, height={self.height})."
            )

        appsink = "appsink name=appsink0 drop=true max-buffers=1 sync=false"

        if self.config.pipeline == "argus":
            if self.config.sensor_id is None:
                raise ValueError(f"{self} pipeline='argus' requires sensor_id in the config.")
            return (
                f"nvarguscamerasrc sensor-id={self.config.sensor_id} "
                f"sensor-mode={self.config.sensor_mode} do-timestamp=true ! "
                f"video/x-raw(memory:NVMM),format=NV12,width={width},height={height},"
                f"framerate={fps}/1 ! "
                "nvvidconv ! video/x-raw,format=BGRx ! "
                "videoconvert ! video/x-raw,format=BGR ! "
                f"{appsink}"
            )

        device = self.config.resolved_device

        if self.config.pipeline == "v4l2_bayer":
            # Raw Bayer from V4L2 (no ISP). We do NOT use GStreamer's
            # ``bayer2rgb`` because it interprets ``grbg10le`` as MSB-aligned
            # whereas the SG16A driver delivers LSB-aligned 10-bit data inside a
            # 16-bit container -- the result is a near-black frame. We hand the
            # raw bayer buffer to OpenCV instead and debayer there. CPU-only, so
            # effective per-camera fps is well below the sensor's 60 Hz when
            # many channels run concurrently.
            return (
                f"v4l2src device={device} io-mode=mmap do-timestamp=true ! "
                f"video/x-bayer,format={self.config.bayer_format},width={width},"
                f"height={height},framerate={fps}/1 ! "
                f"{appsink}"
            )

        # v4l2 pipeline (YUV cameras like ISX028).
        return (
            f"v4l2src device={device} io-mode=mmap do-timestamp=true ! "
            f"video/x-raw,format={self.config.v4l2_pixel_format},width={width},height={height},"
            f"framerate={fps}/1 ! "
            "videoconvert ! video/x-raw,format=BGR ! "
            f"{appsink}"
        )

    def _apply_v4l2_sync(self) -> None:
        """Apply sensor_mode + trig_pin + trig_mode via ``v4l2-ctl``.

        This is best-effort: if ``v4l2-ctl`` is missing we log a warning and continue,
        so the driver still works for free-running setups during development.
        """
        if shutil.which("v4l2-ctl") is None:
            logger.warning(
                "%s v4l2-ctl not found; skipping hardware-trigger setup. "
                "Install v4l-utils on the Jetson to enable hardware sync.",
                self,
            )
            return

        device = self.config.resolved_device
        ctrls = [
            f"sensor_mode={self.config.sensor_mode}",
            f"trig_pin=0x{self.config.trig_pin:08x}",
            f"trig_mode={self.config.trig_mode}",
        ]
        if self.config.exposure_us is not None:
            ctrls.append(f"exposure={int(self.config.exposure_us)}")
        if self.config.gain is not None:
            ctrls.append(f"gain={int(self.config.gain)}")
        cmd = ["v4l2-ctl", "-d", device, "-c", ",".join(ctrls)]
        logger.info("%s applying trigger controls: %s", self, " ".join(cmd))
        try:
            subprocess.run(cmd, check=True, capture_output=True, text=True, timeout=5.0)
        except subprocess.CalledProcessError as exc:
            raise ConnectionError(
                f"{self} failed to apply trigger controls via {' '.join(cmd)}: "
                f"rc={exc.returncode}, stderr={exc.stderr.strip()!r}"
            ) from exc
        except (FileNotFoundError, subprocess.TimeoutExpired) as exc:
            raise ConnectionError(f"{self} failed to invoke v4l2-ctl: {exc!r}") from exc

    # ---------------------------------------------------------------- frames ----

    def _pull_frame(self) -> NDArray[Any] | None:
        """Pull one BGR frame from appsink or return ``None`` on timeout."""
        if self._appsink is None or self._gst is None:
            return None

        sample = self._appsink.emit("try-pull-sample", self.timeout_ms * self._gst.MSECOND)
        if sample is None:
            return None
        buf = sample.get_buffer()
        caps = sample.get_caps().get_structure(0)
        width = caps.get_value("width")
        height = caps.get_value("height")
        media_name = caps.get_name()
        caps_format = caps.get_string("format") or ""
        ok, map_info = buf.map(self._gst.MapFlags.READ)
        if not ok:  # pragma: no cover
            logger.warning("%s buffer map failed.", self)
            return None
        try:
            data = bytes(map_info.data)
        finally:
            buf.unmap(map_info)

        if media_name == "video/x-bayer":
            # Raw bayer in 16-bit container; debayer in OpenCV.
            return self._debayer_buffer(data, width, height, caps_format)

        # Default path: appsink already delivers BGR packed.
        expected = width * height * 3
        if len(data) < expected:  # pragma: no cover
            logger.warning(
                "%s short BGR buffer: got %d bytes, expected %d (%dx%dx3).",
                self,
                len(data),
                expected,
                width,
                height,
            )
            return None
        frame = np.frombuffer(data, dtype=np.uint8, count=expected).reshape((height, width, 3))
        return frame

    def _debayer_buffer(
        self, data: bytes, width: int, height: int, bayer_format: str
    ) -> NDArray[Any] | None:
        """Debayer a raw Bayer buffer in software, returning an H x W x 3 BGR image."""
        try:
            import cv2  # type: ignore  # noqa: PLC0415
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError(
                "The 'v4l2_bayer' pipeline requires the cv2 module for software debayering. "
                "Install python3-opencv or opencv-python-headless on the Jetson."
            ) from exc

        fmt = (bayer_format or "").lower()
        bits = 8
        big_endian = False
        for tag in ("16", "14", "12", "10"):
            if tag in fmt:
                bits = int(tag)
                break
        if fmt.endswith("be"):
            big_endian = True

        pattern = fmt[:4]
        cv_code_map = {
            "bggr": cv2.COLOR_BayerBG2BGR,
            "gbrg": cv2.COLOR_BayerGB2BGR,
            "grbg": cv2.COLOR_BayerGR2BGR,
            "rggb": cv2.COLOR_BayerRG2BGR,
        }
        if pattern not in cv_code_map:
            raise ValueError(f"{self} unsupported bayer pattern: {bayer_format!r}.")

        if bits == 8:
            expected = width * height
            if len(data) < expected:
                logger.warning("%s short bayer8 buffer: %d/%d.", self, len(data), expected)
                return None
            bayer = np.frombuffer(data, dtype=np.uint8, count=expected).reshape((height, width))
        else:
            expected = width * height * 2
            if len(data) < expected:
                logger.warning("%s short bayer16 buffer: %d/%d.", self, len(data), expected)
                return None
            dtype = ">u2" if big_endian else "<u2"
            bayer16 = np.frombuffer(data, dtype=dtype, count=width * height).reshape(
                (height, width)
            )
            # The SG16A AR0234 driver writes the 10-bit value into the *high*
            # bits of the 16-bit container (observed peak == 0xFFFF), so taking
            # the high byte is the right downcast to uint8 for OpenCV.
            # If we ever meet a sensor that uses LSB alignment (V4L2 spec
            # default, peak == (1 << bits) - 1), `>> (bits - 8)` would be the
            # correct shift instead -- pick by the observed maximum.
            shift = 16 - 8 if bayer16.max() > (1 << bits) - 1 else max(0, bits - 8)
            bayer = (bayer16 >> shift).astype(np.uint8, copy=False)

        bgr = cv2.cvtColor(bayer, cv_code_map[pattern])
        return bgr

    def _postprocess(self, frame: NDArray[Any]) -> NDArray[Any]:
        h, w, _ = frame.shape
        if self.height is not None and self.width is not None and (h != self.height or w != self.width):
            raise RuntimeError(
                f"{self} frame {w}x{h} does not match configured {self.width}x{self.height}."
            )
        out = frame
        if self.color_mode == ColorMode.RGB:
            out = out[:, :, ::-1].copy()
        if self.rotation != Cv2Rotation.NO_ROTATION:
            k = {Cv2Rotation.ROTATE_90: -1, Cv2Rotation.ROTATE_180: 2, Cv2Rotation.ROTATE_270: 1}[
                self.rotation
            ]
            out = np.rot90(out, k=k).copy()
        return out

    def _read_loop(self) -> None:
        stop_event = self._stop_event
        if stop_event is None:
            raise RuntimeError(f"{self}: stop_event not initialised before read loop.")

        failures = 0
        while not stop_event.is_set():
            try:
                raw = self._pull_frame()
                if raw is None:
                    failures += 1
                    if failures % 10 == 1:
                        logger.warning("%s read timeout (%d in a row).", self, failures)
                    continue
                frame = self._postprocess(raw)
                with self._frame_lock:
                    self._latest_frame = frame
                    self._latest_timestamp = time.perf_counter()
                self._new_frame_event.set()
                failures = 0
            except Exception as exc:  # noqa: BLE001
                failures += 1
                if failures > 30:
                    logger.exception("%s read loop aborted after repeated errors.", self)
                    return
                logger.warning("%s read loop error: %s", self, exc)

    def _bus_loop(self) -> None:
        """Drain GStreamer bus errors so a fatal pipeline error doesn't go unseen."""
        if self._pipeline is None or self._gst is None or self._stop_event is None:
            return
        bus = self._pipeline.get_bus()
        timeout = 200 * self._gst.MSECOND  # 200 ms.
        while not self._stop_event.is_set():
            msg = bus.timed_pop_filtered(
                timeout,
                self._gst.MessageType.ERROR | self._gst.MessageType.EOS | self._gst.MessageType.WARNING,
            )
            if msg is None:
                continue
            if msg.type == self._gst.MessageType.ERROR:
                err, debug = msg.parse_error()
                logger.error("%s GStreamer ERROR: %s (%s)", self, err, debug)
            elif msg.type == self._gst.MessageType.WARNING:
                err, debug = msg.parse_warning()
                logger.warning("%s GStreamer WARNING: %s (%s)", self, err, debug)
            elif msg.type == self._gst.MessageType.EOS:
                logger.warning("%s GStreamer EOS received.", self)
                return

    # ---------------------------------------------------------------- API ----

    @check_if_not_connected
    def read(self) -> NDArray[Any]:
        if self._read_thread is None or not self._read_thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        self._new_frame_event.clear()
        return self.async_read(timeout_ms=max(self.timeout_ms, 5000))

    @check_if_not_connected
    def async_read(self, timeout_ms: float = 200) -> NDArray[Any]:
        if self._read_thread is None or not self._read_thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        if not self._new_frame_event.wait(timeout=timeout_ms / 1000.0):
            raise TimeoutError(
                f"{self} timed out waiting for a frame after {timeout_ms} ms."
            )
        with self._frame_lock:
            frame = self._latest_frame
            self._new_frame_event.clear()
        if frame is None:  # pragma: no cover
            raise RuntimeError(f"{self} read thread set the event but no frame is available.")
        return frame

    @check_if_not_connected
    def read_latest(self, max_age_ms: int = 500) -> NDArray[Any]:
        if self._read_thread is None or not self._read_thread.is_alive():
            raise RuntimeError(f"{self} read thread is not running.")
        with self._frame_lock:
            frame = self._latest_frame
            timestamp = self._latest_timestamp
        if frame is None or timestamp is None:
            raise RuntimeError(f"{self} has not captured any frames yet.")
        age_ms = (time.perf_counter() - timestamp) * 1e3
        if age_ms > max_age_ms:
            raise TimeoutError(
                f"{self} latest frame is too old: {age_ms:.1f} ms (max allowed: {max_age_ms} ms)."
            )
        return frame

    def disconnect(self) -> None:
        if not self.is_connected and self._read_thread is None and self._pipeline is None:
            raise DeviceNotConnectedError(f"{self} not connected.")

        if self._stop_event is not None:
            self._stop_event.set()
        for thread in (self._read_thread, self._bus_thread):
            if thread is not None and thread.is_alive():
                thread.join(timeout=2.0)
        self._read_thread = None
        self._bus_thread = None
        self._stop_event = None
        self._teardown_pipeline()
        self._connected = False
        with self._frame_lock:
            self._latest_frame = None
            self._latest_timestamp = None
            self._new_frame_event.clear()
        logger.info("%s disconnected.", self)

    def _teardown_pipeline(self) -> None:
        if self._pipeline is not None and self._gst is not None:
            try:
                self._pipeline.set_state(self._gst.State.NULL)
            except Exception:  # noqa: BLE001
                logger.debug("%s pipeline NULL state failed during teardown.", self, exc_info=True)
        self._pipeline = None
        self._appsink = None
