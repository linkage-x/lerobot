"""Client helpers for the ``argus_online_sync`` live frame bus.

The frame bus is a recorder-owned, read-only interface for online inference.
The recorder publishes one latest synchronized SOF cluster under a tmpfs
directory, and model code reads that cluster without opening Argus or
GStreamer camera sessions itself.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
import time
from typing import Any, Iterator


DEFAULT_FRAME_BUS_DIR = Path("/dev/shm/lerobot_online_sync")


@dataclass(frozen=True)
class OnlineSyncFrame:
    camera: str
    path: Path
    width: int
    height: int
    format: str
    logical_frame_index: int
    local_frame_number: int
    sensor_timestamp_ns: int
    sof_tsc_ns: int
    eof_tsc_ns: int
    internal_frame_count: int

    @property
    def expected_size_bytes(self) -> int:
        if self.format.lower() != "nv12":
            raise ValueError(f"unsupported frame format: {self.format!r}")
        return self.width * self.height * 3 // 2

    def read_nv12(self) -> bytes:
        data = self.path.read_bytes()
        expected = self.expected_size_bytes
        if len(data) != expected:
            raise ValueError(
                f"{self.camera} frame size mismatch: got {len(data)} bytes, "
                f"expected {expected} for {self.width}x{self.height} NV12"
            )
        return data

    def as_rgb(self):
        """Return an RGB numpy array when ``numpy`` and ``cv2`` are installed."""

        try:
            import cv2  # type: ignore
            import numpy as np  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError("as_rgb() requires numpy and opencv-python") from exc

        raw = self.read_nv12()
        yuv = np.frombuffer(raw, dtype=np.uint8).reshape((self.height * 3 // 2, self.width))
        return cv2.cvtColor(yuv, cv2.COLOR_YUV2RGB_NV12)


@dataclass(frozen=True)
class OnlineSyncCluster:
    version: int
    publish_seq: int
    slot: int
    recording: bool
    episode_index: int
    logical_frame_index: int
    sync_source: str
    format: str
    width: int
    height: int
    min_sof_tsc_ns: int
    max_sof_tsc_ns: int
    max_delta_ns: int
    frames: dict[str, OnlineSyncFrame]
    raw: dict[str, Any]

    @property
    def sync_timestamp_ns(self) -> int:
        return (self.min_sof_tsc_ns + self.max_sof_tsc_ns) // 2


class ThorOnlineSyncFrameClient:
    """Read latest full-cluster frames published by ``argus_online_sync``.

    This is intentionally a latest-frame API: if the model is slower than the
    camera, it should skip old clusters rather than back-pressure the recorder.
    """

    def __init__(
        self,
        root: str | Path = DEFAULT_FRAME_BUS_DIR,
        *,
        cameras: list[str] | tuple[str, ...] | None = None,
        validate_files: bool = True,
        poll_interval_s: float = 0.005,
    ):
        self.root = Path(root)
        self.latest_path = self.root / "latest_cluster.json"
        self.cameras = set(cameras) if cameras else None
        self.validate_files = bool(validate_files)
        self.poll_interval_s = max(0.001, float(poll_interval_s))

    def get_latest(
        self,
        *,
        timeout_s: float = 0.0,
        min_publish_seq: int | None = None,
        min_logical_frame_index: int | None = None,
    ) -> OnlineSyncCluster | None:
        deadline = time.monotonic() + max(0.0, float(timeout_s))
        while True:
            cluster = self._try_read_latest()
            if cluster is not None:
                if min_publish_seq is not None and cluster.publish_seq < min_publish_seq:
                    cluster = None
                if (
                    cluster is not None
                    and min_logical_frame_index is not None
                    and cluster.logical_frame_index < min_logical_frame_index
                ):
                    cluster = None
                if cluster is not None:
                    return cluster
            if timeout_s <= 0 or time.monotonic() >= deadline:
                return None
            time.sleep(self.poll_interval_s)

    def iter_clusters(
        self,
        *,
        timeout_s: float = 0.1,
    ) -> Iterator[OnlineSyncCluster]:
        next_publish_seq: int | None = None
        while True:
            cluster = self.get_latest(timeout_s=timeout_s, min_publish_seq=next_publish_seq)
            if cluster is None:
                continue
            next_publish_seq = cluster.publish_seq + 1
            yield cluster

    def _try_read_latest(self) -> OnlineSyncCluster | None:
        try:
            payload = json.loads(self.latest_path.read_text(encoding="utf-8"))
        except (FileNotFoundError, json.JSONDecodeError, OSError):
            return None

        frame_format = str(payload.get("format", "")).lower()
        width = int(payload.get("width", 0))
        height = int(payload.get("height", 0))
        cameras_raw = payload.get("cameras", {}) or {}
        frames: dict[str, OnlineSyncFrame] = {}
        for camera, info in cameras_raw.items():
            if self.cameras is not None and camera not in self.cameras:
                continue
            frame = OnlineSyncFrame(
                camera=str(camera),
                path=Path(str(info["path"])),
                width=width,
                height=height,
                format=frame_format,
                logical_frame_index=int(info.get("logical_frame_index", payload.get("logical_frame_index", 0))),
                local_frame_number=int(info.get("local_frame_number", 0)),
                sensor_timestamp_ns=int(info.get("sensor_timestamp_ns", 0)),
                sof_tsc_ns=int(info.get("sof_tsc_ns", 0)),
                eof_tsc_ns=int(info.get("eof_tsc_ns", 0)),
                internal_frame_count=int(info.get("internal_frame_count", 0)),
            )
            if self.validate_files:
                if not frame.path.exists():
                    return None
                if frame.path.stat().st_size != frame.expected_size_bytes:
                    return None
            frames[frame.camera] = frame

        if self.cameras is not None and frames.keys() != self.cameras:
            return None

        return OnlineSyncCluster(
            version=int(payload.get("version", 0)),
            publish_seq=int(payload.get("publish_seq", 0)),
            slot=int(payload.get("slot", 0)),
            recording=bool(payload.get("recording", False)),
            episode_index=int(payload.get("episode_index", -1)),
            logical_frame_index=int(payload.get("logical_frame_index", 0)),
            sync_source=str(payload.get("sync_source", "")),
            format=frame_format,
            width=width,
            height=height,
            min_sof_tsc_ns=int(payload.get("min_sof_tsc_ns", 0)),
            max_sof_tsc_ns=int(payload.get("max_sof_tsc_ns", 0)),
            max_delta_ns=int(payload.get("max_delta_ns", 0)),
            frames=frames,
            raw=payload,
        )
