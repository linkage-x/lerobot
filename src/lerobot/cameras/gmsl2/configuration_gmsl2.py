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

"""Configuration for SENSING GMSL2 cameras on NVIDIA Jetson (SG16A-AGTH-G3Y-A1 adapter)."""

from dataclasses import dataclass

from ..configs import CameraConfig, ColorMode, Cv2Rotation

GMSL2_DEFAULT_COLOR_MODE = ColorMode.BGR
GMSL2_DEFAULT_WARMUP_S = 2
GMSL2_DEFAULT_TIMEOUT_MS = 2000

# Default trigger pin used by the SG16A_AGTH_G3Y_A1 board (matches load_modules.sh).
GMSL2_DEFAULT_TRIG_PIN = 0x00020007

__all__ = [
    "Gmsl2CameraConfig",
    "ColorMode",
    "Cv2Rotation",
    "GMSL2_DEFAULT_COLOR_MODE",
    "GMSL2_DEFAULT_WARMUP_S",
    "GMSL2_DEFAULT_TIMEOUT_MS",
    "GMSL2_DEFAULT_TRIG_PIN",
]


@CameraConfig.register_subclass("gmsl2")
@dataclass
class Gmsl2CameraConfig(CameraConfig):
    """SENSING GMSL2 camera attached to a Jetson via the SG16A_AGTH_G3Y_A1 adapter.

    Frames are captured through GStreamer using either ``nvarguscamerasrc`` (Bayer/RAW
    sensors that go through the Jetson ISP, e.g. SG2-AR0234C-G2F) or ``v4l2src`` for
    sensors that output YUV directly (e.g. SG8-ISX028C-G2G).

    Hardware-synchronous capture is configured via ``v4l2-ctl`` controls
    (``sensor_mode``, ``trig_pin``, ``trig_mode``). When ``sync_role='slave'`` the
    driver puts the sensor in external-trigger mode; one of the cameras must act as
    ``master`` (or, more commonly on this board, a PWM signal from the Jetson is fed
    into the adapter's trigger pin and *all* cameras are slaves -- see
    ``tools/gmsl2/setup_sync.sh``).

    Attributes:
        sensor_id: Argus sensor-id. Maps 1:1 to ``/dev/video{sensor_id}`` in the
            current SG16A driver. Required when ``pipeline='argus'``.
        device: Optional V4L2 device path (e.g. ``/dev/video3``). Required when
            ``pipeline`` is ``'v4l2'`` or ``'v4l2_bayer'``; otherwise defaults to
            ``/dev/video{sensor_id}``.
        pipeline: Capture pipeline kind:
            * ``'argus'`` -- ``nvarguscamerasrc`` (hardware ISP). Recommended on
              JetPack 6 / Orin. JetPack 7 / Thor does *not* ship the plugin yet,
              so prefer ``v4l2_bayer`` there.
            * ``'v4l2_bayer'`` -- ``v4l2src`` + GStreamer ``bayer2rgb`` software
              debayer. Works out of the box on JetPack 7 / Thor with AR0234,
              but ``bayer2rgb`` is CPU-only and saturates well below 60 fps per
              channel at 1920x1080; use lower per-camera fps or accept dropped
              frames when running many channels in parallel.
            * ``'v4l2'`` -- ``v4l2src`` directly. Suitable for sensors that
              already deliver packed YUV (e.g. ISX028).
        sensor_mode: Argus sensor mode index (matches the dtbo's ``mode0/mode1``).
        v4l2_pixel_format: GStreamer caps format string used when ``pipeline='v4l2'``.
        bayer_format: GStreamer ``video/x-bayer`` format string used when
            ``pipeline='v4l2_bayer'`` (e.g. ``'grbg10le'`` for SG2-AR0234C-G2F).
        sync_role: ``'auto'`` keeps the kernel default (``trig_mode=0`` free-run),
            ``'master'`` forces ``trig_mode=0`` and is intended for the camera that
            generates the sync pulse, ``'slave'`` forces ``trig_mode=1`` (external
            trigger). Use ``'slave'`` on every channel when a Jetson PWM is the
            trigger source.
        trig_pin: Trigger-pin selector passed to ``v4l2-ctl`` (default matches
            ``load_modules.sh`` from the SG16A SDK).
        apply_sync_at_connect: If True, run ``v4l2-ctl`` to apply ``sensor_mode``,
            ``trig_pin`` and ``trig_mode`` when ``connect()`` is called.
        color_mode: BGR or RGB output. NV12 → BGR is the cheap default on Jetson.
        rotation: Optional rotation applied in software after the pipeline.
        warmup_s: Seconds to wait for the first stable frame after ``connect()``.
        timeout_ms: Per-frame appsink pull timeout.
    """

    sensor_id: int | None = None
    device: str | None = None
    pipeline: str = "argus"
    sensor_mode: int = 0
    v4l2_pixel_format: str = "UYVY"
    bayer_format: str = "grbg10le"
    sync_role: str = "auto"
    trig_pin: int = GMSL2_DEFAULT_TRIG_PIN
    apply_sync_at_connect: bool = True
    exposure_us: int | None = None
    gain: int | None = None
    color_mode: ColorMode = GMSL2_DEFAULT_COLOR_MODE
    rotation: Cv2Rotation = Cv2Rotation.NO_ROTATION
    warmup_s: int = GMSL2_DEFAULT_WARMUP_S
    timeout_ms: int = GMSL2_DEFAULT_TIMEOUT_MS

    def __post_init__(self) -> None:
        self.color_mode = ColorMode(self.color_mode)
        self.rotation = Cv2Rotation(self.rotation)
        self.pipeline = self.pipeline.lower()
        self.sync_role = self.sync_role.lower()

        if self.pipeline not in {"argus", "v4l2", "v4l2_bayer"}:
            raise ValueError(
                "`pipeline` must be one of ['argus', 'v4l2', 'v4l2_bayer'], "
                f"but {self.pipeline!r} is provided."
            )
        if self.sync_role not in {"auto", "master", "slave"}:
            raise ValueError(
                "`sync_role` must be one of ['auto', 'master', 'slave'], "
                f"but {self.sync_role!r} is provided."
            )
        if self.sensor_id is None and self.device is None:
            raise ValueError("`sensor_id` or `device` must be provided for a GMSL2 camera.")
        if self.sensor_id is not None and self.sensor_id < 0:
            raise ValueError(f"`sensor_id` must be >= 0, but {self.sensor_id} is provided.")
        if self.timeout_ms <= 0:
            raise ValueError(f"`timeout_ms` must be > 0, but {self.timeout_ms} is provided.")
        if self.sensor_mode < 0:
            raise ValueError(f"`sensor_mode` must be >= 0, but {self.sensor_mode} is provided.")
        if self.fps is not None and self.fps <= 0:
            raise ValueError(f"`fps` must be > 0 when provided, but {self.fps} is provided.")
        if self.exposure_us is not None and self.exposure_us <= 0:
            raise ValueError(f"`exposure_us` must be > 0 when provided, but {self.exposure_us} is provided.")
        if self.gain is not None and self.gain <= 0:
            raise ValueError(f"`gain` must be > 0 when provided, but {self.gain} is provided.")

    @property
    def resolved_device(self) -> str:
        if self.device is not None:
            return self.device
        return f"/dev/video{self.sensor_id}"

    @property
    def trig_mode(self) -> int:
        # auto -> 0 (free-run), master -> 0, slave -> 1.
        return 1 if self.sync_role == "slave" else 0
