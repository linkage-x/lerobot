"""LeRobot-side helper around the vendored Box collection SDK.

The ARM release bundle (``tools/thor/box_sdk/``) ships a ``box_sdk`` wheel that
binds to ``libbox_controller.so`` and pushes/pulls UDP packets to the BOX MCU
on the collection board (gripper, IMU, trigger, six-d force, two Paxini
touch pads). The vendored wheel is the source of truth for the C API; this
module only:

  * decodes the ``SensorCache`` ``ctypes.Structure`` into JSON-friendly
    dicts so the gateway / recorder can pass it through HTTP or write it
    into the dataset metadata sidecar,
  * exposes a small ``BoxClient`` class that wires together
    ``Box.start`` / ``Box.stop`` / ``Box.get_sensor_cache`` per the config
    block in ``thor_gmsl2_11ch_example.yaml``,
  * keeps an ``available()`` probe so calling code can degrade gracefully
    on dev hosts where the wheel is not installed (the gateway still needs
    to import the file to know which device IDs are expected).

The 239-point Paxini frame is flattened into three lists (``fx``, ``fy``,
``fz``); the dataset writer is free to reshape those into the 15-row layout
described in the BOX SDK 需求整理 doc.
"""

from __future__ import annotations

import logging
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from threading import Event, Lock, Thread
from typing import Any

logger = logging.getLogger("box_client")


# Sensor IDs that this module decodes out of ``AllSensor``. The recorder /
# gateway use these strings as device IDs so the frontend can show one row
# per sensor even though the SDK delivers them as a single UDP packet.
KNOWN_SENSOR_IDS = (
    "box_gripper",
    "box_imu",
    "box_trigger",
    "box_six_d_force",
    "box_touch_left",
    "box_touch_right",
)


@dataclass
class SensorSample:
    """One timestamped sample from a single BOX sensor, deduplicated by MCU timestamp."""
    sensor_id: str
    mcu_timestamp: int
    wall_time_s: float
    mono_time_s: float
    data: dict[str, Any]


@dataclass
class BoxClientConfig:
    enabled: bool = True
    bind_ip: str = "0.0.0.0"
    bind_port: int = 15000
    remote_ip: str = "192.168.2.60"
    remote_port: int = 15000
    sdk_dir: str = "tools/thor/box_sdk"
    sdk_setup_script: str = "setup_env.sh"
    urdf_relpath: str = "share/monte_gripper.urdf"
    startup_mode: int = 0  # 0 = collection / trigger-controlled, 1 = control
    poll_interval_s: float = 0.05
    record_poll_interval_s: float = 0.002
    stale_threshold_s: float = 1.0
    expected_devices: list[str] = field(default_factory=lambda: list(KNOWN_SENSOR_IDS))

    def __post_init__(self) -> None:
        if self.bind_port != 15000 or self.remote_port != 15000:
            # SDK 需求整理 §5.1: "本机udp端口(需要固定输入15000)".
            logger.warning(
                "box_collection UDP ports default to 15000; running with %d <-> %d",
                self.bind_port, self.remote_port,
            )
        if self.startup_mode not in (0, 1):
            raise ValueError(f"startup_mode must be 0 or 1, got {self.startup_mode!r}")
        unknown = [d for d in self.expected_devices if d not in KNOWN_SENSOR_IDS]
        if unknown:
            raise ValueError(
                f"unknown box expected_devices: {unknown}; pick from {KNOWN_SENSOR_IDS}"
            )


def from_yaml_dict(raw: dict[str, Any] | None) -> BoxClientConfig:
    if not raw:
        return BoxClientConfig(enabled=False)
    expected = raw.get("expected_devices")
    if expected is None:
        expected = list(KNOWN_SENSOR_IDS)
    return BoxClientConfig(
        enabled=bool(raw.get("enabled", True)),
        bind_ip=str(raw.get("bind_ip", "0.0.0.0")),
        bind_port=int(raw.get("bind_port", 15000)),
        remote_ip=str(raw.get("remote_ip", "192.168.2.60")),
        remote_port=int(raw.get("remote_port", 15000)),
        sdk_dir=str(raw.get("sdk_dir", "tools/thor/box_sdk")),
        sdk_setup_script=str(raw.get("sdk_setup_script", "setup_env.sh")),
        urdf_relpath=str(raw.get("urdf_relpath", "share/monte_gripper.urdf")),
        startup_mode=int(raw.get("startup_mode", 0)),
        poll_interval_s=float(raw.get("poll_interval_s", 0.05)),
        record_poll_interval_s=float(raw.get("record_poll_interval_s", 0.002)),
        stale_threshold_s=float(raw.get("stale_threshold_s", 1.0)),
        expected_devices=[str(x) for x in expected],
    )


def available() -> bool:
    try:
        import box_sdk  # noqa: F401
    except Exception:
        return False
    return True


def _decode_gripper(g) -> dict[str, Any]:
    # GripperData.distance is in meters per SDK header.
    return {"timestamp": int(g.timestamp), "distance_m": float(g.distance)}


def _decode_imu(imu) -> dict[str, Any]:
    return {
        "timestamp": int(imu.timestamp),
        "acc_g": [float(v) for v in imu.acc],
        "gyr_deg_s": [float(v) for v in imu.gyr],
        "roll_deg": float(imu.roll),
        "pitch_deg": float(imu.pitch),
        "yaw_deg": float(imu.yaw),
        "quat_wxyz": [float(v) for v in imu.quat],
    }


def _decode_trigger(t) -> dict[str, Any]:
    # TriggerData.distance is the squeezed travel percentage (0 released, 100 pulled).
    return {"timestamp": int(t.timestamp), "travel_pct": float(t.distance)}


def _decode_six_d_force(f) -> dict[str, Any]:
    return {"timestamp": int(f.timestamp), "fxyz_mxyz": [float(v) for v in f.data]}


def _decode_touch(ts) -> dict[str, Any]:
    # 239 force points, each fx/fy as int8 0.1N, fz as uint8 0.1N (signed for shear).
    fx = [int(p.fx) for p in ts.forces]
    fy = [int(p.fy) for p in ts.forces]
    fz = [int(p.fz) for p in ts.forces]
    return {"timestamp": int(ts.timestamp), "fx_0p1N": fx, "fy_0p1N": fy, "fz_0p1N": fz}


def _touch_sensor_pair(snap) -> tuple[Any, Any]:
    """Return the left/right Paxini frames from the stable SDK cache fields.

    The BOX wheel exposes touch data twice: as legacy flattened members under
    ``snap.data`` and as the explicit two-element ``snap.touch_sensor_data``.
    On current firmware the legacy members can report different effective
    rates after power cycling, so use the explicit array whenever it exists.
    """

    data = snap.data
    touch_array = getattr(snap, "touch_sensor_data", None)
    if touch_array is not None and len(touch_array) >= 2:
        return touch_array[0], touch_array[1]
    if touch_array is not None and len(touch_array) == 1:
        return touch_array[0], data.touch_sensor_data_sec
    return data.touch_sensor_data_first, data.touch_sensor_data_sec


def decode_sensor_cache(snap) -> dict[str, Any]:
    """Turn a ``box_sdk.SensorCache`` into a JSON-friendly dict.

    Includes only those sub-frames whose timestamp is non-zero so the
    consumer can tell which sensors actually checked in this poll.
    """

    out: dict[str, Any] = {
        "valid": bool(getattr(snap, "valid", 0)),
        "liwp_index": int(getattr(snap, "liwp_index", 0)),
        "liwp_timestamp": int(getattr(snap, "liwp_timestemp", 0)),  # SDK typo: 'timestemp'
        "sensors": {},
    }
    data = snap.data
    gripper_distance = float(getattr(data.gripper_data, "distance", 0.0))
    gripper_has_sample = (
        bool(data.gripper_data.timestamp)
        or (out["valid"] and math.isfinite(gripper_distance) and gripper_distance != 0.0)
    )
    if gripper_has_sample:
        out["sensors"]["box_gripper"] = _decode_gripper(data.gripper_data)
    if data.imu_data.timestamp:
        out["sensors"]["box_imu"] = _decode_imu(data.imu_data)
    if data.trigger_data.timestamp:
        out["sensors"]["box_trigger"] = _decode_trigger(data.trigger_data)
    if data.six_d_force_data.timestamp:
        out["sensors"]["box_six_d_force"] = _decode_six_d_force(data.six_d_force_data)
    touch_left, touch_right = _touch_sensor_pair(snap)
    if touch_left.timestamp:
        out["sensors"]["box_touch_left"] = _decode_touch(touch_left)
    if touch_right.timestamp:
        out["sensors"]["box_touch_right"] = _decode_touch(touch_right)
    return out


def _decode_sensor_timestamps(snap) -> dict[str, int]:
    """Return raw per-sensor timestamps, including zeros for absent samples."""

    data = snap.data
    touch_left, touch_right = _touch_sensor_pair(snap)
    return {
        "box_gripper": int(getattr(data.gripper_data, "timestamp", 0)),
        "box_imu": int(getattr(data.imu_data, "timestamp", 0)),
        "box_trigger": int(getattr(data.trigger_data, "timestamp", 0)),
        "box_six_d_force": int(getattr(data.six_d_force_data, "timestamp", 0)),
        "box_touch_left": int(getattr(touch_left, "timestamp", 0)),
        "box_touch_right": int(getattr(touch_right, "timestamp", 0)),
    }


class BoxClient:
    """Lifecycle wrapper around ``box_sdk.Box``.

    Usage::

        cfg = from_yaml_dict(cfg_yaml.get("box_collection"))
        client = BoxClient(cfg)
        client.start()
        snap = client.read()         # dict, see :func:`decode_sensor_cache`
        present = client.detect()    # which expected_devices are publishing
        client.stop()

    All public methods are safe to call when ``box_sdk`` is not installed --
    they just log a warning and return empty/None values so the gateway can
    still bring up the GUI for dev/inspection on non-Thor hosts.
    """

    def __init__(self, cfg: BoxClientConfig, *, so_path: str | None = None):
        self.cfg = cfg
        self._so_path = so_path
        self._box = None
        self._poll_thread: Thread | None = None
        self._stop_event = Event()
        self._lock = Lock()
        self._latest: dict[str, Any] = {"valid": False, "sensors": {}}
        self._latest_at_s: float = 0.0
        self._latest_wall_time_s: float = 0.0
        self._first_seen_at_s: dict[str, float] = {}
        self._last_seen_at_s: dict[str, float] = {}
        self._last_sensor_timestamps: dict[str, int] = {sid: 0 for sid in KNOWN_SENSOR_IDS}
        self._poll_count = 0
        self._valid_poll_count = 0
        self._last_poll_at_s: float = 0.0
        self._last_poll_wall_time_s: float = 0.0
        self._last_rc: int | None = None
        self._last_error: str | None = None
        self._started_at_s: float = 0.0
        self._started_wall_time_s: float = 0.0
        self._sensor_update_count: dict[str, int] = {sid: 0 for sid in KNOWN_SENSOR_IDS}
        self._sensor_rate_window_start_s: float = 0.0
        self._sensor_rate_count: dict[str, int] = {sid: 0 for sid in KNOWN_SENSOR_IDS}
        self._sensor_observed_hz: dict[str, float] = {sid: 0.0 for sid in KNOWN_SENSOR_IDS}
        self._recording = False
        self._record_t0_wall_s = 0.0
        self._record_samples: dict[str, list[SensorSample]] = {}
        self._record_last_ts: dict[str, int] = {}

    # ---- lifecycle ----

    def start(self) -> bool:
        if not self.cfg.enabled:
            logger.info("box_collection disabled in config; skipping start")
            return False

        # The native SDK reads its gripper URDF from $BOX_SDK_URDF; without
        # this it falls back to relative path "thirdpart/monte_gripper.urdf"
        # that doesn't exist in the deploy layout, and the C++ urdf_parser
        # throws std::invalid_argument -> std::terminate -> SIGABRT on the
        # first incoming UDP packet. Resolve to an absolute path here so
        # callers don't have to source tools/thor/box_sdk/setup_env.sh.
        # box_client.py lives at <repo>/tools/thor/box_sdk/, alongside
        # share/monte_gripper.urdf, so we resolve relative to __file__.
        urdf_abs = Path(__file__).resolve().parent / self.cfg.urdf_relpath
        if urdf_abs.exists():
            os.environ.setdefault("BOX_SDK_URDF", str(urdf_abs))
        else:
            logger.warning(
                "box_sdk URDF not found at %s; native SDK will likely abort "
                "(set $BOX_SDK_URDF manually or fix urdf_relpath in "
                "box_collection config)", urdf_abs,
            )

        try:
            from box_sdk import Box  # type: ignore
        except Exception as exc:
            logger.warning(
                "box_sdk wheel not installed; box_client.start() is a no-op (%s)", exc,
            )
            return False

        self._box = Box(so_path=self._so_path) if self._so_path else Box()
        rc = self._box.start(
            self.cfg.bind_ip, self.cfg.bind_port,
            self.cfg.remote_ip, self.cfg.remote_port,
        )
        if rc != 0:
            logger.warning("box.start rc=%d", rc)
            return False
        self._started_at_s = time.monotonic()
        self._started_wall_time_s = time.time()
        # set_mode is best-effort -- on dev boards the BOX MCU may not be
        # alive yet, and we still want the poll loop running so the recorder
        # can decide what to do.
        try:
            rc = self._box.set_mode(int(self.cfg.startup_mode))
            if rc != 0:
                logger.warning("box.set_mode(%d) rc=%d", self.cfg.startup_mode, rc)
        except Exception as exc:
            logger.warning("box.set_mode failed: %s", exc)

        self._stop_event.clear()
        self._poll_thread = Thread(target=self._poll_loop, daemon=True, name="box-poll")
        self._poll_thread.start()
        return True

    def stop(self) -> None:
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.5)
            self._poll_thread = None
        if self._box is not None:
            try:
                self._box.stop()
            except Exception as exc:
                logger.warning("box.stop failed: %s", exc)
            try:
                self._box.close()
            except Exception:
                pass
            self._box = None

    # ---- polling ----

    def _poll_loop(self) -> None:
        assert self._box is not None
        while not self._stop_event.is_set():
            try:
                rc, snap = self._box.get_sensor_cache()
            except Exception as exc:
                logger.error("box.get_sensor_cache raised: %s", exc)
                now = time.monotonic()
                with self._lock:
                    self._poll_count += 1
                    self._last_poll_at_s = now
                    self._last_poll_wall_time_s = time.time()
                    self._last_rc = None
                    self._last_error = str(exc)
                    recording = self._recording
                interval = self.cfg.record_poll_interval_s if recording else self.cfg.poll_interval_s
                self._stop_event.wait(max(0.001, interval))
                continue
            now = time.monotonic()
            wall_now = time.time()
            valid = rc == 0 and bool(getattr(snap, "valid", 0))
            decoded = decode_sensor_cache(snap) if rc == 0 else None
            sensor_timestamps = _decode_sensor_timestamps(snap) if rc == 0 else {}
            err = self._err_str(rc) if rc != 0 else None
            with self._lock:
                self._poll_count += 1
                self._last_poll_at_s = now
                self._last_poll_wall_time_s = wall_now
                self._last_rc = int(rc)
                self._last_error = err
                for sid, sensor_ts in sensor_timestamps.items():
                    if sensor_ts:
                        if sensor_ts != self._last_sensor_timestamps.get(sid, 0):
                            self._sensor_update_count[sid] = self._sensor_update_count.get(sid, 0) + 1
                            self._sensor_rate_count[sid] = self._sensor_rate_count.get(sid, 0) + 1
                        self._last_sensor_timestamps[sid] = sensor_ts
                        self._first_seen_at_s.setdefault(sid, now)
                        self._last_seen_at_s[sid] = now
                if self._sensor_rate_window_start_s == 0.0:
                    self._sensor_rate_window_start_s = now
                window = now - self._sensor_rate_window_start_s
                if window >= 2.0:
                    for sid in self._sensor_rate_count:
                        self._sensor_observed_hz[sid] = self._sensor_rate_count[sid] / window
                        self._sensor_rate_count[sid] = 0
                    self._sensor_rate_window_start_s = now
                if valid and decoded is not None:
                    for sid in decoded.get("sensors", {}):
                        self._first_seen_at_s.setdefault(sid, now)
                        self._last_seen_at_s[sid] = now
                    self._valid_poll_count += 1
                    self._latest = decoded
                    self._latest_at_s = now
                    self._latest_wall_time_s = wall_now
                if self._recording and decoded is not None:
                    sensors = decoded.get("sensors", {})
                    for sid, ts in sensor_timestamps.items():
                        if ts and ts != self._record_last_ts.get(sid, 0) and sid in sensors:
                            self._record_last_ts[sid] = ts
                            self._record_samples.setdefault(sid, []).append(
                                SensorSample(
                                    sensor_id=sid,
                                    mcu_timestamp=ts,
                                    wall_time_s=wall_now,
                                    mono_time_s=now,
                                    data=sensors[sid],
                                )
                            )
                recording = self._recording
            interval = self.cfg.record_poll_interval_s if recording else self.cfg.poll_interval_s
            self._stop_event.wait(max(0.001, interval))

    def _err_str(self, rc: int) -> str:
        if self._box is None:
            return str(rc)
        try:
            return str(self._box.err_str(rc))
        except Exception:
            return str(rc)

    def _status_locked(self) -> dict[str, Any]:
        now = time.monotonic()
        sensor_status = {}
        for sid in self.cfg.expected_devices or list(KNOWN_SENSOR_IDS):
            last_seen = self._last_seen_at_s.get(sid, 0.0)
            fresh = bool(last_seen and now - last_seen <= self.cfg.stale_threshold_s)
            sensor_status[sid] = {
                "seen": bool(last_seen),
                "fresh": fresh,
                "last_seen_age_s": max(0.0, now - last_seen) if last_seen else None,
                "last_timestamp": int(self._last_sensor_timestamps.get(sid, 0)),
                "observed_hz": round(self._sensor_observed_hz.get(sid, 0.0), 1),
                "update_count": self._sensor_update_count.get(sid, 0),
            }
        return {
            "active": self._box is not None and not self._stop_event.is_set(),
            "started_at_s": self._started_at_s,
            "started_wall_time_s": self._started_wall_time_s,
            "poll_count": self._poll_count,
            "valid_poll_count": self._valid_poll_count,
            "last_poll_at_s": self._last_poll_at_s,
            "last_poll_wall_time_s": self._last_poll_wall_time_s,
            "last_rc": self._last_rc,
            "last_error": self._last_error,
            "latest_valid_at_s": self._latest_at_s,
            "latest_valid_wall_time_s": self._latest_wall_time_s,
            "latest_age_s": max(0.0, now - self._latest_at_s) if self._latest_at_s else None,
            "sensor_status": sensor_status,
        }

    def read(self) -> dict[str, Any]:
        with self._lock:
            snap = dict(self._latest)
            snap["received_at_s"] = self._latest_at_s
            snap["received_wall_time_s"] = self._latest_wall_time_s
            snap["status"] = self._status_locked()
        return snap

    def start_recording(self, t0_wall_s: float) -> None:
        """Begin high-frequency per-sensor sample recording for an episode.

        Switches the poll loop to ``record_poll_interval_s`` and starts
        deduplicating sensor samples by MCU timestamp into per-sensor buffers.
        """
        with self._lock:
            self._recording = True
            self._record_t0_wall_s = t0_wall_s
            self._record_samples = {sid: [] for sid in KNOWN_SENSOR_IDS}
            self._record_last_ts = dict(self._last_sensor_timestamps)
        logger.info("recording started (t0=%.3f, poll=%.1fms)",
                    t0_wall_s, self.cfg.record_poll_interval_s * 1000)

    def stop_recording(self) -> dict[str, list[SensorSample]]:
        """Stop recording and return all per-sensor samples collected."""
        with self._lock:
            self._recording = False
            samples = self._record_samples
            self._record_samples = {}
            self._record_last_ts = {}
        total = sum(len(v) for v in samples.values())
        per_sensor = {sid: len(v) for sid, v in samples.items() if v}
        logger.info("recording stopped: %d total samples, per-sensor: %s", total, per_sensor)
        return samples

    @staticmethod
    def serialize_recorded_samples(
        samples: dict[str, list[SensorSample]],
        t0_wall_s: float,
    ) -> list[dict[str, Any]]:
        """Flatten per-sensor samples into a time-sorted list for JSON serialization."""
        out: list[dict[str, Any]] = []
        for sample_list in samples.values():
            for s in sample_list:
                out.append({
                    "sid": s.sensor_id,
                    "mcu_ts": s.mcu_timestamp,
                    "wall_s": s.wall_time_s,
                    "t_rel_s": s.wall_time_s - t0_wall_s,
                    "data": s.data,
                })
        out.sort(key=lambda x: x["wall_s"])
        return out

    def detect(self) -> list[str]:
        """Return which expected sensors have published at least once."""
        with self._lock:
            seen = set(self._first_seen_at_s)
            cutoff = time.monotonic() - self.cfg.stale_threshold_s
            fresh = {
                sid for sid, ts in self._first_seen_at_s.items()
                if ts >= cutoff or ts >= self._latest_at_s - self.cfg.stale_threshold_s
            }
        order = self.cfg.expected_devices or list(KNOWN_SENSOR_IDS)
        present = [sid for sid in order if sid in seen]
        # Anything reporting now but not in the expected list still surfaces
        # so the operator notices unexpected hardware.
        for sid in sorted(seen - set(order)):
            present.append(sid)
        # Filter to fresh sensors when possible; otherwise return all-ever-seen
        # so the caller can decide.
        return [sid for sid in present if sid in fresh] or present

    def connected_devices(self) -> list[str]:
        """Return configured BOX rows whose SDK transport session is active."""

        if not self.is_active():
            return []
        return list(self.cfg.expected_devices or KNOWN_SENSOR_IDS)

    # ---- introspection ----

    def observed_rates(self) -> dict[str, float]:
        """Return per-sensor observed update rates in Hz."""
        with self._lock:
            return dict(self._sensor_observed_hz)

    def is_active(self) -> bool:
        return self._box is not None and not self._stop_event.is_set()

    def latest_age_s(self) -> float | None:
        with self._lock:
            if self._latest_at_s <= 0:
                return None
            return max(0.0, time.monotonic() - self._latest_at_s)

    def urdf_path(self, repo_root: Path | None = None) -> Path:
        base = Path(self.cfg.sdk_dir)
        if not base.is_absolute() and repo_root is not None:
            base = repo_root / base
        return base / self.cfg.urdf_relpath
