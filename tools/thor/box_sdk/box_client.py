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
from dataclasses import dataclass, field, replace
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
    # Stable identity used to namespace this box's sensor IDs when a rig runs
    # more than one BOX (e.g. box_id="box0" -> "box0/box_gripper"). Empty means
    # single-box legacy mode: sensor IDs stay bare ("box_gripper"), wire- and
    # dataset-compatible with every rig recorded before multi-box support.
    box_id: str = ""
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
    # The BOX only streams data (UDP -> :15000) for a short window after each
    # unicast command; the DiscoveryKeepAlive broadcast on :15001 does NOT keep
    # the data channel alive. Without a re-arm it stops streaming within ~1s, so
    # the poll loop re-arms each device with set_mode every rearm_interval_s.
    # Measured on Thor: rearm off -> 0 Hz, 3.0s -> ~0.5 Hz (sporadic bursts),
    # 0.5s -> ~166 Hz (near the ~199 Hz native rate). 0 disables.
    rearm_interval_s: float = 0.5
    stale_threshold_s: float = 1.0
    # The vendored SDK .so unconditionally dumps box_sensor_data_*.csv into CWD
    # at ~35 MB/min with no disable switch; delete this session's file on stop()
    # until the SDK ships an opt-out (see tools/thor/box_sdk/TROUBLESHOOTING.md).
    cleanup_box_csv: bool = True
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
        box_id=str(raw.get("box_id", "")),
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
        rearm_interval_s=float(raw.get("rearm_interval_s", 0.5)),
        stale_threshold_s=float(raw.get("stale_threshold_s", 1.0)),
        cleanup_box_csv=bool(raw.get("cleanup_box_csv", True)),
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


# ---------------------------------------------------------------------------
# v3 multi-device transport plumbing
#
# The v3 BOX SDK implements the protocol drafted in MULTI_BOX_PROTOCOL.md: one
# ``Box`` binds ``:15000`` and demultiplexes N devices by ``device_id``
# (``get_sensor_cache(device_id)`` / ``set_mode(mode, device_id)`` / ...), plus a
# broadcast ``discover()`` enumeration on ``:15001``. The helpers below adapt
# that surface onto the host abstractions: ``_DeviceHandle`` re-exposes the old
# point-to-point ``Box`` interface BoxClient expects, bound to one device_id over
# a shared Box; ``discover_boxes()`` enumerates devices for the GUI roster.
# ---------------------------------------------------------------------------

DATA_PORT_DEFAULT = 15000
DISCOVERY_PORT_DEFAULT = 15001

# Vendor SDK capability bits (box_sdk._internal.ctypes_backend CAP_*). Mirrored
# here so caps→sensor mapping works without importing the ARM-only wheel.
CAP_GRIPPER = 1 << 0
CAP_TRIGGER = 1 << 1
CAP_TOUCH = 1 << 2
CAP_6D_FORCE = 1 << 3
CAP_IMU = 1 << 4


def caps_to_sensor_ids(capabilities: int) -> list[str]:
    """Map a discovered device's ``capabilities`` bitfield onto host sensor IDs.

    The vendor's single ``CAP_TOUCH`` bit covers both Paxini pads, so it expands
    to ``box_touch_left`` + ``box_touch_right``. Returned in canonical
    :data:`KNOWN_SENSOR_IDS` order.
    """
    present: set[str] = set()
    if capabilities & CAP_GRIPPER:
        present.add("box_gripper")
    if capabilities & CAP_IMU:
        present.add("box_imu")
    if capabilities & CAP_TRIGGER:
        present.add("box_trigger")
    if capabilities & CAP_6D_FORCE:
        present.add("box_six_d_force")
    if capabilities & CAP_TOUCH:
        present.update(("box_touch_left", "box_touch_right"))
    return [sid for sid in KNOWN_SENSOR_IDS if sid in present]


@dataclass
class DiscoveredBox:
    """One BOX enumerated by :func:`discover_boxes` over the shared socket."""

    device_id: int
    sn: str = ""
    ip: str = ""
    data_port: int = DATA_PORT_DEFAULT
    fw_version: int = 0
    capabilities: int = 0
    box_id: str = ""
    expected_devices: list[str] = field(default_factory=list)

    def to_public_dict(self) -> dict[str, Any]:
        return {
            "device_id": self.device_id,
            "sn": self.sn,
            "ip": self.ip,
            "data_port": self.data_port,
            "fw_version": self.fw_version,
            "capabilities": self.capabilities,
            "capability_names": caps_to_sensor_ids(self.capabilities),
            "box_id": self.box_id,
            "expected_devices": list(self.expected_devices),
        }


def discover_boxes(
    timeout: float = 2.0,
    broadcast_addr: str = "255.255.255.255",
    *,
    retries: int = 1,
) -> list[DiscoveredBox]:
    """Broadcast-discover BOX devices on the subnet via the vendored SDK.

    Never raises: returns ``[]`` when the wheel is absent (dev hosts) or the
    broadcast finds nothing, so the gateway/recorder can degrade gracefully.
    ``expected_devices`` is filled from each device's self-described
    ``capabilities`` so the host need not hardcode the sensor set.

    The first broadcast after a cold start can miss devices (ARP not warmed /
    boxes mid-cycle), so ``retries`` re-broadcasts until something answers.
    """
    try:
        import box_sdk  # type: ignore
    except Exception as exc:  # noqa: BLE001
        logger.warning("box_sdk wheel not installed; discover_boxes() -> [] (%s)", exc)
        return []
    found = []
    for attempt in range(max(1, retries)):
        try:
            found = box_sdk.discover(timeout=timeout, broadcast_addr=broadcast_addr)
        except Exception as exc:  # noqa: BLE001 - wrong arch / socket errors on dev hosts
            logger.warning("box_sdk.discover() failed: %s", exc)
            return []
        if found:
            break
    out: list[DiscoveredBox] = []
    for d in found:
        caps = int(getattr(d, "capabilities", 0))
        out.append(
            DiscoveredBox(
                device_id=int(d.device_id),
                sn=str(getattr(d, "sn", "") or ""),
                ip=str(getattr(d, "ip", "") or ""),
                data_port=int(getattr(d, "data_port", DATA_PORT_DEFAULT) or DATA_PORT_DEFAULT),
                fw_version=int(getattr(d, "fw_version", 0) or 0),
                capabilities=caps,
                expected_devices=caps_to_sensor_ids(caps),
            )
        )
    out.sort(key=lambda b: (b.ip, b.device_id))
    return out


def _ensure_box_env(cfg: BoxClientConfig) -> None:
    """Point the native SDK at the bundled URDF + conf before constructing ``Box``.

    Without ``$BOX_SDK_URDF`` the C++ urdf_parser aborts (SIGABRT) on the first
    packet; ``$BOX_SDK_CONFIG`` (box_sdk.conf, ``sensor_csv_enabled=0``) disables
    the high-rate CSV dump at the source on the v3 SDK. Both are resolved next to
    this file so callers need not source ``setup_env.sh``.
    """
    here = Path(__file__).resolve().parent
    urdf_abs = here / cfg.urdf_relpath
    if urdf_abs.exists():
        os.environ.setdefault("BOX_SDK_URDF", str(urdf_abs))
    else:
        logger.warning(
            "box_sdk URDF not found at %s; native SDK will likely abort "
            "(set $BOX_SDK_URDF manually or fix urdf_relpath)", urdf_abs,
        )
    conf_abs = here / "box_sdk.conf"
    if conf_abs.exists():
        os.environ.setdefault("BOX_SDK_CONFIG", str(conf_abs))


def _snapshot_session_csv(csv_dir: Path | None) -> set[Path]:
    """Record pre-existing ``box_sensor_data_*.csv`` so cleanup only removes ours."""
    if csv_dir is None:
        return set()
    try:
        return set(csv_dir.glob("box_sensor_data_*.csv"))
    except OSError:
        return set()


def _remove_session_csv(csv_dir: Path | None, pre: set[Path], enabled: bool) -> None:
    """Delete the ``box_sensor_data_*.csv`` the SDK .so dumped this session.

    The vendored ``libbox_controller.so`` appends a high-rate CSV (~35 MB/min) to
    CWD; the v3 ``box_sdk.conf`` disables it at the source, but this remains a
    belt-and-suspenders cleanup for SDKs/configs that still dump. Only files that
    appeared after :func:`_snapshot_session_csv` are removed.
    """
    if not enabled or csv_dir is None:
        return
    try:
        current = set(csv_dir.glob("box_sensor_data_*.csv"))
    except OSError:
        return
    for path in current - pre:
        try:
            path.unlink()
            logger.info("removed box SDK debug CSV %s", path.name)
        except OSError as exc:
            logger.warning("could not remove box CSV %s: %s", path, exc)


def _register_discovered(box, devices: list[DiscoveredBox]) -> None:
    """Register each discovered device's address so commands can be addressed."""
    for d in devices:
        try:
            rc = box.register_device(int(d.device_id), str(d.ip), int(d.data_port))
            if rc != 0:
                logger.warning(
                    "register_device(%d, %s:%d) rc=%d", d.device_id, d.ip, d.data_port, rc
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("register_device(%d) raised: %s", d.device_id, exc)


class _DeviceHandle:
    """Adapter exposing the legacy point-to-point ``Box`` surface BoxClient drives,
    bound to one ``device_id`` over a shared v3 ``Box``.

    ``device_id=None`` (single-box legacy fallback) forwards to the SDK's
    most-recent-device path; control calls lazily learn an id from
    ``get_device_ids()`` once the box reports.
    """

    def __init__(self, box, device_id: int | None):
        self._box = box
        self.device_id = device_id

    def _resolve_id(self) -> int | None:
        if self.device_id is not None:
            return self.device_id
        try:
            ids = self._box.get_device_ids()
        except Exception:  # noqa: BLE001
            ids = []
        if ids:
            self.device_id = int(ids[0])
        return self.device_id

    def get_sensor_cache(self):
        return self._box.get_sensor_cache(self.device_id)

    def set_mode(self, mode: int) -> int:
        did = self._resolve_id()
        if did is None:
            raise RuntimeError("set_mode: device_id unknown (no box reported yet)")
        return int(self._box.set_mode(int(mode), did))

    def set_clamp_pos(self, pos_m: float) -> int:
        did = self._resolve_id()
        if did is None:
            raise RuntimeError("set_clamp_pos: device_id unknown")
        return int(self._box.set_clamp_pos(float(pos_m), device_id=did))

    def set_trigger_zero(self) -> int:
        did = self._resolve_id()
        if did is None:
            raise RuntimeError("set_trigger_zero: device_id unknown")
        return int(self._box.set_trigger_zero(device_id=did))

    def cali_6d_force_sensor(self) -> int:
        did = self._resolve_id()
        if did is None:
            raise RuntimeError("cali_6d_force_sensor: device_id unknown")
        return int(self._box.cali_6d_force_sensor(device_id=did))

    def cali_touch_sensor(self) -> int:
        did = self._resolve_id()
        if did is None:
            raise RuntimeError("cali_touch_sensor: device_id unknown")
        return int(self._box.cali_touch_sensor(device_id=did))

    def err_str(self, rc: int) -> str:
        return self._box.err_str(rc)


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

    def __init__(
        self,
        cfg: BoxClientConfig,
        *,
        so_path: str | None = None,
        shared_box=None,
        device_id: int | None = None,
        box_id: str | None = None,
    ):
        self.cfg = cfg
        self._so_path = so_path
        # When a BoxPool owns the shared multi-device Box, this view binds one
        # device_id over it and does NOT manage the box lifecycle. Standalone
        # (shared_box is None) -> we own a private Box (single-device legacy).
        self._shared_box = shared_box
        self._owns_box = shared_box is None
        self._device_id = device_id
        self._box_id = box_id if box_id is not None else cfg.box_id
        self._box: _DeviceHandle | None = None
        self._raw_box = None  # the owned box_sdk.Box (standalone only)
        self._keepalive = None
        self._csv_dir: Path | None = None
        self._pre_session_csv: set[Path] = set()
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
        self._last_rearm_at_s: float = 0.0

    # ---- lifecycle ----

    def start(self) -> bool:
        if not self.cfg.enabled:
            logger.info("box_collection disabled in config; skipping start")
            return False

        handle = self._start_owned_box() if self._owns_box else self._start_shared_view()
        if handle is None:
            return False
        self._box = handle

        self._started_at_s = time.monotonic()
        self._started_wall_time_s = time.time()
        # set_mode is best-effort -- the BOX MCU may not be alive yet (or the
        # device_id not learned), and we still want the poll loop running so the
        # recorder can decide what to do.
        try:
            rc = self._box.set_mode(int(self.cfg.startup_mode))
            if rc != 0:
                logger.warning("box.set_mode(%d) rc=%d", self.cfg.startup_mode, rc)
        except Exception as exc:  # noqa: BLE001
            logger.warning("box.set_mode failed: %s", exc)
        self._last_rearm_at_s = time.monotonic()

        self._stop_event.clear()
        self._poll_thread = Thread(target=self._poll_loop, daemon=True, name="box-poll")
        self._poll_thread.start()
        return True

    def _start_owned_box(self) -> _DeviceHandle | None:
        """Standalone path: bind a private v3 Box and resolve a single device_id.

        The native SDK reads its gripper URDF from $BOX_SDK_URDF; without it the
        C++ urdf_parser throws -> SIGABRT on the first UDP packet. :func:`_ensure_box_env`
        resolves the bundled URDF (+ conf) so callers need not source setup_env.sh.
        """
        _ensure_box_env(self.cfg)
        try:
            import box_sdk  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "box_sdk wheel not installed; box_client.start() is a no-op (%s)", exc,
            )
            return None
        box = box_sdk.Box(so_path=self._so_path) if self._so_path else box_sdk.Box()
        self._raw_box = box
        # Snapshot pre-existing SDK CSV dumps so stop() removes only the
        # box_sensor_data_*.csv this session creates in CWD.
        self._csv_dir = Path.cwd()
        self._pre_session_csv = _snapshot_session_csv(self._csv_dir)
        try:
            rc = box.start(
                self.cfg.bind_ip, self.cfg.bind_port,
                self.cfg.remote_ip, self.cfg.remote_port,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("box.start raised: %s", exc)
            return None
        if rc != 0:
            logger.warning("box.start rc=%d", rc)
            return None
        # Broadcast-discover so we learn + register the device_id up front; also
        # start a keepalive (the firmware ages out its master after ~15s silence).
        device_id = self._device_id
        try:
            found = box_sdk.discover(timeout=2.0)
        except Exception as exc:  # noqa: BLE001
            found = []
            logger.warning("discover during start failed: %s", exc)
        if found:
            _register_discovered(box, [
                DiscoveredBox(device_id=int(d.device_id), ip=str(getattr(d, "ip", "") or ""),
                              data_port=int(getattr(d, "data_port", DATA_PORT_DEFAULT) or DATA_PORT_DEFAULT))
                for d in found
            ])
            primary = next(
                (d for d in found if self.cfg.remote_ip and str(getattr(d, "ip", "")) == self.cfg.remote_ip),
                found[0],
            )
            device_id = int(primary.device_id)
        try:
            self._keepalive = box_sdk.DiscoveryKeepAlive()
        except Exception as exc:  # noqa: BLE001
            self._keepalive = None
            logger.warning("DiscoveryKeepAlive unavailable: %s", exc)
        self._device_id = device_id
        return _DeviceHandle(box, device_id)

    def _start_shared_view(self) -> _DeviceHandle | None:
        """Pool path: bind one device_id over the BoxPool's shared Box."""
        if self._shared_box is None:
            logger.warning("shared box missing; cannot start view")
            return None
        return _DeviceHandle(self._shared_box, self._device_id)

    def stop(self) -> None:
        self._stop_event.set()
        if self._poll_thread is not None:
            self._poll_thread.join(timeout=1.5)
            self._poll_thread = None
        self._box = None
        # Only a standalone client owns the underlying Box / keepalive / CSV; a
        # pooled view shares the BoxPool's Box and must not tear it down.
        if self._owns_box:
            if self._keepalive is not None:
                try:
                    self._keepalive.close()
                except Exception:  # noqa: BLE001
                    pass
                self._keepalive = None
            if self._raw_box is not None:
                try:
                    self._raw_box.stop()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("box.stop failed: %s", exc)
                try:
                    self._raw_box.close()
                except Exception:  # noqa: BLE001
                    pass
                self._raw_box = None
            self._cleanup_session_csv()

    def _cleanup_session_csv(self) -> None:
        """Delete the box_sensor_data_*.csv the SDK .so dumped this session.

        The v3 ``box_sdk.conf`` disables the dump at the source, but this
        remains a belt-and-suspenders cleanup; only files that appeared after
        :meth:`start` are removed (see tools/thor/box_sdk/TROUBLESHOOTING.md).
        """
        _remove_session_csv(self._csv_dir, self._pre_session_csv, self.cfg.cleanup_box_csv)
        self._pre_session_csv = set()

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
            # Keep the device's data stream alive: re-arm with set_mode on a fixed
            # cadence. The BOX firmware ages out its per-device data master ~10-15s
            # after the last unicast command, and the broadcast DiscoveryKeepAlive
            # on :15001 does NOT refresh it -- without this re-arm the box stops
            # streaming during the Connect->Start gap (or a long episode), so
            # stop_recording() returns 0 samples. Done outside the lock (network
            # round-trip) and best-effort (a transient failure is retried next tick).
            if self.cfg.rearm_interval_s > 0:
                now_rearm = time.monotonic()
                if now_rearm - self._last_rearm_at_s >= self.cfg.rearm_interval_s:
                    self._last_rearm_at_s = now_rearm
                    try:
                        self._box.set_mode(int(self.cfg.startup_mode))
                    except Exception:  # noqa: BLE001 - best-effort keepalive
                        pass
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

    # ---- control ----

    def _six_d_force_vector(self) -> list[float] | None:
        """Latest decoded 6D force/torque vector (Fx,Fy,Fz,Mx,My,Mz) or None."""
        with self._lock:
            force = self._latest.get("sensors", {}).get("box_six_d_force")
        if isinstance(force, dict) and isinstance(force.get("fxyz_mxyz"), list):
            return [float(v) for v in force["fxyz_mxyz"]]
        return None

    def calibrate_six_d_force(self) -> dict[str, Any]:
        """Trigger the native 6D force-sensor calibration (zeroing).

        Calls box_sdk's ``cali_6d_force_sensor()`` (TLV_TYPE_CALI_6D_FORCE_SENSOR)
        on the live handle, sampling the live force vector just before and ~0.5s
        after so the caller can show the effect (matches the vendor demo). The
        native SDK serializes its command channel, so this is safe to call from a
        thread other than the poll loop (same as ``set_clamp_pos`` mid-recording).

        Never raises: when the box never started, ``box_sdk`` is absent, or the
        installed wheel predates ``cali_6d_force_sensor()`` (e.g. 0.1.0), it
        returns ``ok=False`` with an explanatory ``error`` instead.
        """
        if self._box is None:
            return {"ok": False, "rc": None, "error": "box not started"}
        fn = getattr(self._box, "cali_6d_force_sensor", None)
        if not callable(fn):
            return {
                "ok": False, "rc": None,
                "error": "installed box_sdk lacks cali_6d_force_sensor(); "
                         "needs a newer SDK build than the current wheel",
            }
        before = self._six_d_force_vector()
        try:
            rc = int(fn())
        except Exception as exc:  # noqa: BLE001 - report, never crash the recorder
            return {"ok": False, "rc": None, "before": before,
                    "error": f"cali_6d_force_sensor() raised: {exc}"}
        time.sleep(0.5)  # let the post-cal reading settle before sampling
        return {
            "ok": rc == 0, "rc": rc,
            "error": None if rc == 0 else self._err_str(rc),
            "before": before, "after": self._six_d_force_vector(),
        }

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


# ---------------------------------------------------------------------------
# Multi-box scaffolding
#
# The current vendored SDK is point-to-point: one ``Box`` instance binds one
# ``UdpConfig`` (single bind/remote) and ``get_sensor_cache()`` returns a single
# merged ``SensorCache`` with no source attribution. Running several BOXes on
# one subnet therefore needs a vendor SDK change (per-source attribution or an
# enumeration API -- see ``tools/thor/box_sdk/MULTI_BOX_PROTOCOL.md``).
#
# Everything below is the host-side half that we *can* build now without the
# SDK: a fleet config that holds N per-box configs, a pluggable discovery
# strategy (static today, SDK-backed later), and a ``BoxPool`` that owns N
# ``BoxClient``s and aggregates their reads/detection/recording under
# namespaced sensor IDs. With one box and an empty ``box_id`` the pool delegates
# straight through to the single client, so single-box rigs are byte-identical
# to the pre-pool behaviour.
# ---------------------------------------------------------------------------


def namespace_sid(box_id: str, sid: str) -> str:
    """Prefix a bare sensor ID with its owning box's identity.

    Empty ``box_id`` (single-box legacy mode) returns the bare ID unchanged so
    existing datasets, frontend rows and the box_sensors.jsonl sidecar keep
    their historical keys.
    """
    return f"{box_id}/{sid}" if box_id else sid


@dataclass
class BoxFleetConfig:
    """A set of BOX configs the recorder should bring up together.

    Parsed from either the legacy flat ``box_collection`` block (one box, empty
    ``box_id``) or the new ``boxes:`` list form. ``discovery`` selects how the
    pool turns this config into the live box list: ``"static"`` uses ``boxes``
    verbatim; ``"sdk"`` will defer to the vendor enumeration API once it exists
    and falls back to ``boxes`` until then.
    """

    enabled: bool = True
    discovery: str = "static"  # "static" | "sdk"
    boxes: list[BoxClientConfig] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.discovery not in ("static", "sdk"):
            raise ValueError(
                f"discovery must be 'static' or 'sdk', got {self.discovery!r}"
            )
        if len(self.boxes) > 1:
            ids = [b.box_id for b in self.boxes]
            if any(not bid for bid in ids):
                raise ValueError(
                    "every box needs a non-empty box_id when more than one box "
                    "is configured (used to namespace sensor IDs)"
                )
            dupes = sorted({bid for bid in ids if ids.count(bid) > 1})
            if dupes:
                raise ValueError(f"duplicate box_id(s): {dupes}")


def fleet_from_yaml_dict(raw: dict[str, Any] | None) -> BoxFleetConfig:
    """Build a :class:`BoxFleetConfig` from the ``box_collection`` YAML block.

    Accepts both shapes::

        # legacy single-box (unchanged)
        box_collection: {enabled: true, remote_ip: ..., expected_devices: [...]}

        # multi-box
        box_collection:
          enabled: true
          discovery: static
          boxes:
            - {box_id: box0, remote_ip: 192.168.2.60, expected_devices: [...]}
            - {box_id: box1, remote_ip: 192.168.2.61, expected_devices: [...]}
    """
    if not raw:
        return BoxFleetConfig(enabled=False, boxes=[])
    enabled = bool(raw.get("enabled", True))
    discovery = str(raw.get("discovery", "static"))
    raw_boxes = raw.get("boxes")
    if raw_boxes is None:
        # Legacy flat form: the whole block describes a single box.
        single = from_yaml_dict(raw)
        return BoxFleetConfig(enabled=enabled, discovery=discovery, boxes=[single])
    boxes: list[BoxClientConfig] = []
    for entry in raw_boxes:
        if not isinstance(entry, dict):
            raise ValueError(f"each box entry must be a mapping, got {entry!r}")
        # Inherit the fleet-level enabled flag unless the box overrides it.
        per_box = {"enabled": enabled, **entry}
        boxes.append(from_yaml_dict(per_box))
    return BoxFleetConfig(enabled=enabled, discovery=discovery, boxes=boxes)


class BoxDiscovery:
    """Strategy that yields the per-box configs a :class:`BoxPool` should run.

    Subclasses encapsulate *how* the live box list is obtained so the pool
    stays agnostic. Today only :class:`StaticBoxDiscovery` is usable;
    :class:`SdkBoxDiscovery` is the seam the vendor enumeration API will plug
    into without touching the pool.
    """

    def discover(self) -> list[BoxClientConfig]:
        raise NotImplementedError


class StaticBoxDiscovery(BoxDiscovery):
    """Return exactly the boxes named in the config -- no probing."""

    def __init__(self, boxes: list[BoxClientConfig]):
        self._boxes = list(boxes)

    def discover(self) -> list[BoxClientConfig]:
        return list(self._boxes)


# Proposed ``sensor_mask`` bit order for ``box_discover()`` / self-describing
# packets: bit i set => ``KNOWN_SENSOR_IDS[i]`` is present on that box. The
# vendor must match this layout (documented in MULTI_BOX_PROTOCOL.md §2.2);
# if they pick a different order, adjust this single tuple.
SENSOR_MASK_BITS: tuple[str, ...] = KNOWN_SENSOR_IDS


def decode_sensor_mask(mask: int) -> list[str]:
    """Turn a ``sensor_mask`` bitfield into the sensor IDs a box advertises."""
    return [sid for i, sid in enumerate(SENSOR_MASK_BITS) if mask & (1 << i)]


@dataclass
class BoxInfo:
    """Host-side mirror of the SDK ``BoxInfo`` record from ``box_discover()``.

    Field names follow MULTI_BOX_PROTOCOL.md §2.2. ``box_serial`` is the stable
    identity (survives IP changes); ``sensor_mask`` advertises which sensors the
    box carries so the host need not hardcode the full set.
    """

    box_serial: str
    ip: str = ""
    model: str = ""
    fw_version: int = 0
    proto_version: int = 0
    sensor_mask: int = 0


def box_info_to_config(
    info: BoxInfo,
    template: BoxClientConfig,
    *,
    aliases: dict[str, str] | None = None,
) -> BoxClientConfig:
    """Map one discovered :class:`BoxInfo` onto a :class:`BoxClientConfig`.

    Shared transport defaults (bind_ip, ports, poll intervals, sdk_dir, urdf)
    come from ``template``; per-box fields come from the discovery record. The
    namespace ``box_id`` defaults to the stable serial; ``aliases`` lets an
    operator pin a friendly label (``{serial: "box0"}``) so dataset keys stay
    short. An empty ``sensor_mask`` falls back to the template's expected list.
    """
    box_id = (aliases or {}).get(info.box_serial, info.box_serial)
    expected = decode_sensor_mask(info.sensor_mask) or list(template.expected_devices)
    return replace(
        template,
        box_id=box_id,
        remote_ip=info.ip or template.remote_ip,
        expected_devices=expected,
    )


class SdkBoxDiscovery(BoxDiscovery):
    """SDK-backed subnet enumeration (option A in MULTI_BOX_PROTOCOL.md).

    Only the raw enumeration call into the vendor SDK (``box_discover()``) is
    still missing -- everything downstream is implemented and tested: mapping
    ``BoxInfo`` records onto namespaced :class:`BoxClientConfig`s that inherit
    shared transport defaults from a template. When the SDK ships, pass its
    binding as ``enumerate_fn`` (a zero-arg callable returning ``list[BoxInfo]``)
    and the rest works unchanged. Until then ``discover()`` logs a warning and
    returns the statically configured boxes, so ``discovery: sdk`` is safe to
    set in advance.
    """

    def __init__(
        self,
        fallback: list[BoxClientConfig],
        *,
        enumerate_fn: Any = None,
        template: BoxClientConfig | None = None,
        aliases: dict[str, str] | None = None,
    ):
        self._fallback = StaticBoxDiscovery(fallback)
        self._enumerate_fn = enumerate_fn
        # Shared transport defaults for discovered boxes: reuse the first
        # configured box if present, else library defaults.
        self._template = template or (fallback[0] if fallback else BoxClientConfig())
        self._aliases = dict(aliases or {})

    def _enumerate(self) -> list[BoxInfo]:
        if self._enumerate_fn is None:
            raise NotImplementedError(
                "box_discover() not available in the vendored SDK yet; "
                "see tools/thor/box_sdk/MULTI_BOX_PROTOCOL.md"
            )
        return list(self._enumerate_fn())

    def discover(self) -> list[BoxClientConfig]:
        try:
            infos = self._enumerate()
        except NotImplementedError as exc:
            logger.warning("SdkBoxDiscovery: %s; falling back to static boxes", exc)
            return self._fallback.discover()
        except Exception as exc:  # SDK present but the enumeration call failed
            logger.warning(
                "SdkBoxDiscovery enumeration raised (%s); falling back to static boxes", exc
            )
            return self._fallback.discover()
        if not infos:
            logger.warning("SdkBoxDiscovery: no boxes discovered; falling back to static boxes")
            return self._fallback.discover()
        return [box_info_to_config(i, self._template, aliases=self._aliases) for i in infos]


def make_discovery(cfg: BoxFleetConfig) -> BoxDiscovery:
    if cfg.discovery == "sdk":
        return SdkBoxDiscovery(cfg.boxes)
    return StaticBoxDiscovery(cfg.boxes)


class BoxPool:
    """Own and aggregate N :class:`BoxClient`s behind the single-box interface.

    Exposes the same surface ``thor_record`` already drives on one client
    (``start``/``stop``/``read``/``detect``/``connected_devices``/
    ``observed_rates``/``start_recording``/``stop_recording``) so it is a
    drop-in replacement. With a single box whose ``box_id`` is empty the pool
    delegates verbatim to that client; otherwise per-box sensor IDs are
    namespaced via :func:`namespace_sid`.
    """

    def __init__(
        self,
        cfg: BoxFleetConfig,
        discovery: BoxDiscovery | None = None,
        *,
        client_factory: Any = None,
        so_path: str | None = None,
    ):
        self.cfg = cfg
        # Static fallback (config-driven) used only when live discovery finds
        # nothing; the live path is box_sdk.discover() over the shared socket.
        self._discovery = discovery or make_discovery(cfg)
        self._client_factory = client_factory or BoxClient
        self._so_path = so_path
        # list of (box_id, BoxClient); populated at start()
        self._clients: list[tuple[str, BoxClient]] = []
        # The single v3 Box shared by every device view (方案A).
        self._shared_box = None
        self._keepalive = None
        self._template = BoxClientConfig()
        self._csv_dir: Path | None = None
        self._pre_session_csv: set[Path] = set()
        # Roster of devices discovered at start(), surfaced to the GUI/recorder.
        self._devices: list[DiscoveredBox] = []

    @property
    def enabled(self) -> bool:
        return self.cfg.enabled

    def _passthrough(self) -> bool:
        """True when we can delegate straight to a lone, unnamespaced client."""
        return len(self._clients) == 1 and not self._clients[0][0]

    def discovered_devices(self) -> list[dict[str, Any]]:
        """Roster of BOX devices found by broadcast at :meth:`start`.

        Each entry carries device_id / sn / ip / capabilities / box_id so the
        recorder can emit it and the gateway can render one GUI row per
        discovered box (instead of relying on static YAML config).
        """
        return [d.to_public_dict() for d in self._devices]

    # ---- lifecycle ----

    def start(self) -> bool:
        if not self.cfg.enabled:
            logger.info("box fleet disabled in config; skipping start")
            return False
        template = self.cfg.boxes[0] if self.cfg.boxes else BoxClientConfig()
        self._template = template
        _ensure_box_env(template)
        try:
            import box_sdk  # type: ignore
        except Exception as exc:  # noqa: BLE001
            logger.warning("box_sdk wheel not installed; BoxPool.start() no-op (%s)", exc)
            return False

        # One shared Box binds :15000 and demultiplexes every device by id.
        try:
            box = box_sdk.Box(so_path=self._so_path) if self._so_path else box_sdk.Box()
        except Exception as exc:  # noqa: BLE001
            logger.warning("box_create failed: %s", exc)
            return False
        self._shared_box = box
        self._csv_dir = Path.cwd()
        self._pre_session_csv = _snapshot_session_csv(self._csv_dir)
        try:
            rc = box.start(
                template.bind_ip, template.bind_port,
                template.remote_ip, template.remote_port,
            )
        except Exception as exc:  # noqa: BLE001
            logger.warning("shared box.start raised: %s", exc)
            self._teardown_shared_box()
            return False
        if rc != 0:
            logger.warning("shared box.start rc=%d", rc)
            self._teardown_shared_box()
            return False

        # Enumerate BEFORE the keepalive grabs :15001, then register every
        # discovered address so commands can be addressed by id.
        self._devices = self._enumerate_devices()
        if self._devices:
            _register_discovered(box, self._devices)
        else:
            logger.warning(
                "no BOX devices discovered; falling back to a single passthrough view"
            )
        # Keepalive so the firmware doesn't age out its master after ~15s silence.
        try:
            self._keepalive = box_sdk.DiscoveryKeepAlive()
        except Exception as exc:  # noqa: BLE001
            self._keepalive = None
            logger.warning("DiscoveryKeepAlive unavailable: %s", exc)

        any_started = False
        for box_cfg, device_id, box_id in self._plan_views(template, self._devices):
            client = self._client_factory(
                box_cfg, shared_box=box, device_id=device_id, box_id=box_id,
            )
            try:
                started = client.start()
            except Exception as exc:  # one bad box must not sink the others
                logger.warning("box %r start raised: %s", box_id, exc)
                started = False
            self._clients.append((box_id, client))
            any_started = any_started or started
        return any_started

    def _enumerate_devices(self) -> list[DiscoveredBox]:
        """Live broadcast enumeration; assign a namespace box_id per device.

        A lone discovered device on a legacy single-box fleet keeps ``box_id=""``
        (bare sensor IDs, dataset-compatible). With >1 device each gets a unique
        namespace: its serial when that serial is unique, else ``box<device_id>``
        -- because un-personalized firmware ships every unit with the same default
        ``box_serial`` (which would collide and merge two physical boxes into one
        namespace). ``device_id`` is the firmware's per-unit identity: unique, not
        IP-derived, and stable across the session (unlike the DHCP IP or a
        discovery-order index), so it is the safest fallback short of a
        personalized serial.
        """
        found = discover_boxes(timeout=2.0, retries=3)
        legacy_single = len(self.cfg.boxes) <= 1 and (
            not self.cfg.boxes or not self.cfg.boxes[0].box_id
        )
        sns = [d.sn for d in found]

        def _ambiguous(d: DiscoveredBox) -> bool:
            return (not d.sn) or sns.count(d.sn) > 1

        if any(_ambiguous(d) for d in found):
            logger.warning(
                "BOX serials are not unique/personalized (%s); namespacing by "
                "device_id. Personalize box_serial in firmware for shorter, "
                "reboot-stable dataset keys.", sns,
            )
        for d in found:
            if len(found) == 1 and legacy_single:
                d.box_id = ""
            elif not _ambiguous(d):
                d.box_id = d.sn
            else:
                d.box_id = f"box{d.device_id}"
        return found

    def _plan_views(self, template: BoxClientConfig, devices: list[DiscoveredBox]):
        """Yield ``(box_cfg, device_id, box_id)`` for every per-device view."""
        if devices:
            for d in devices:
                box_cfg = replace(
                    template,
                    box_id=d.box_id,
                    remote_ip=d.ip or template.remote_ip,
                    expected_devices=d.expected_devices or list(template.expected_devices),
                )
                yield box_cfg, d.device_id, d.box_id
            return
        # Fallback: no live discovery. Use the static config; a legacy single box
        # becomes a passthrough view whose device_id is learned lazily from
        # get_device_ids() once data flows.
        static = self._discovery.discover()
        cfg0 = static[0] if static else template
        if len(static) > 1:
            logger.warning(
                "multiple boxes configured but discovery found none; only the "
                "first (%r) can be addressed on the shared socket", cfg0.box_id,
            )
        yield cfg0, None, cfg0.box_id

    def _teardown_shared_box(self) -> None:
        if self._keepalive is not None:
            try:
                self._keepalive.close()
            except Exception:  # noqa: BLE001
                pass
            self._keepalive = None
        if self._shared_box is not None:
            try:
                self._shared_box.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("shared box.stop failed: %s", exc)
            try:
                self._shared_box.close()
            except Exception:  # noqa: BLE001
                pass
            self._shared_box = None
        _remove_session_csv(self._csv_dir, self._pre_session_csv, self._template.cleanup_box_csv)
        self._pre_session_csv = set()

    def stop(self) -> None:
        for _, client in self._clients:
            try:
                client.stop()
            except Exception as exc:  # noqa: BLE001
                logger.warning("box stop failed: %s", exc)
        self._clients = []
        self._teardown_shared_box()
        self._devices = []

    def is_active(self) -> bool:
        return any(client.is_active() for _, client in self._clients)

    # ---- aggregation ----

    def read(self) -> dict[str, Any]:
        if self._passthrough():
            return self._clients[0][1].read()
        merged: dict[str, Any] = {
            "valid": False,
            "sensors": {},
            "status": {"active": False, "sensor_status": {}},
            "boxes": {},
        }
        for box_id, client in self._clients:
            snap = client.read()
            merged["valid"] = merged["valid"] or bool(snap.get("valid"))
            for sid, data in snap.get("sensors", {}).items():
                merged["sensors"][namespace_sid(box_id, sid)] = data
            status = snap.get("status", {}) if isinstance(snap.get("status"), dict) else {}
            merged["status"]["active"] = merged["status"]["active"] or bool(status.get("active"))
            for sid, sstat in (status.get("sensor_status") or {}).items():
                merged["status"]["sensor_status"][namespace_sid(box_id, sid)] = sstat
            merged["boxes"][box_id] = snap
        return merged

    def detect(self) -> list[str]:
        if self._passthrough():
            return self._clients[0][1].detect()
        out: list[str] = []
        for box_id, client in self._clients:
            out.extend(namespace_sid(box_id, sid) for sid in client.detect())
        return out

    def connected_devices(self) -> list[str]:
        if self._passthrough():
            return self._clients[0][1].connected_devices()
        out: list[str] = []
        for box_id, client in self._clients:
            out.extend(namespace_sid(box_id, sid) for sid in client.connected_devices())
        return out

    def observed_rates(self) -> dict[str, float]:
        if self._passthrough():
            return self._clients[0][1].observed_rates()
        out: dict[str, float] = {}
        for box_id, client in self._clients:
            for sid, hz in client.observed_rates().items():
                out[namespace_sid(box_id, sid)] = hz
        return out

    # ---- recording ----

    def start_recording(self, t0_wall_s: float) -> None:
        for _, client in self._clients:
            client.start_recording(t0_wall_s)

    def stop_recording(self) -> dict[str, list[SensorSample]]:
        if self._passthrough():
            return self._clients[0][1].stop_recording()
        merged: dict[str, list[SensorSample]] = {}
        for box_id, client in self._clients:
            for sid, samples in client.stop_recording().items():
                nsid = namespace_sid(box_id, sid)
                merged[nsid] = [replace(s, sensor_id=nsid) for s in samples]
        return merged

    # ---- control ----

    def calibrate_six_d_force(self) -> list[dict[str, Any]]:
        """Dispatch a 6D force calibration to each box exposing that sensor.

        Returns one result dict per attempted box, tagged with ``box_id`` (empty
        string for the single-box passthrough rig). Boxes that don't advertise a
        6D force sensor are skipped; if none advertise one, every client is still
        attempted so a not-yet-detected sensor isn't silently ignored.
        """
        targets = [
            (box_id, client) for box_id, client in self._clients
            if "box_six_d_force" in (set(client.detect()) | set(client.cfg.expected_devices or []))
        ]
        if not targets:
            targets = list(self._clients)
        return [
            {"box_id": box_id, **client.calibrate_six_d_force()}
            for box_id, client in targets
        ]

    serialize_recorded_samples = staticmethod(BoxClient.serialize_recorded_samples)
