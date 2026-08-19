#!/usr/bin/env python3

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

"""Why the cameras missed their alignment budget -- measured live, cameras only, no robot.

``fr3_sync_audit.py`` says *that* a recording's cameras disagreed. It cannot say why, because a
dataset only holds the timestamps of the frames that were *selected*: a frame the sensor never
produced, one the host dropped on the way in, and one that arrived late all look identical there.
This runs the cameras alone -- same class-shaped read loop, same anchoring, no arm, no encoder --
and records the three quantities that separate those cases:

``frame_counter`` (sensor-side sequence number)
    Gaps mean the sensor produced frames this host never received: USB bandwidth, a link that
    negotiated USB 2.1, or a kernel/driver stall. The count of missing counters *is* the drop
    count; nothing else measures it.

device timestamp spacing
    Contiguous counters with intervals wider than the nominal period mean the *sensor* slowed
    down, which for a RealSense colour stream is almost always auto-exposure: with
    ``auto_exposure_priority`` on, a dim scene lets the sensor stretch exposure past the frame
    period and silently halve the rate. ``actual_exposure`` is recorded next to it, so the two
    are read together rather than inferred from one another.

handover lag (acquisition -> our thread)
    Contiguous counters at nominal spacing but a late handover means the host is the problem:
    the frames arrived, this process was busy. This is the one the recorder's own load can
    cause, and the one ``--extra-work-ms`` doses deliberately.

The poll thread reproduces ``FrankaResearch3.get_observation``'s selection exactly (anchor on the
oldest of the cameras' latest frames, then each camera's buffered frame closest to that instant),
so the skew it prints is the number the recorder would have recorded, produced by a process doing
nothing else. If the skew is healthy here and unhealthy under the recorder, the cameras are fine
and the load is not.

Protocol (each step is one run; stop at the first step that reproduces)
----------------------------------------------------------------------
1. ``--duration 60 --poll-hz 60``: the baseline. Read the preamble first -- a ``usb_type`` of 2.1,
   ``global_time`` off, or a negotiated profile below the requested fps ends the investigation
   right there.
2. ``--poll-hz 30`` against the same scene: separates "the cameras are unhealthy" from "polling
   at the sensor rate is what exposes it". Delivery and drops must not care about the poll rate;
   if they do, the read path is being starved by the poll thread.
3. ``--extra-work-ms 4 / 8 / 16`` at ``--poll-hz 60``: a dose-response for the host-load
   hypothesis. Per-frame work is what recording at 60 fps doubled, and 16 ms is roughly a full
   60 Hz tick. If drops and handover lag climb with the dose, the recorder's per-frame cost is
   the mechanism and the fix is load, not cameras.
4. ``--lights-on`` is not a flag: rerun step 1 under a brighter scene. If ``actual_exposure``
   falls and the delivered rate rises, auto-exposure was the throttle.

Deliberately does not import ``lerobot`` -- the read loop here is a copy of
``RealSenseCamera._read_loop``'s shape, not the class itself, because the class deliberately
hides the raw frame metadata this tool exists to read.
"""

from __future__ import annotations

import argparse
from collections import deque
from dataclasses import dataclass, field
import json
from pathlib import Path
import statistics
import threading
import time
from typing import Any

import numpy as np

try:
    import pyrealsense2 as rs  # type: ignore
except ImportError as exc:  # pragma: no cover - environment, not logic
    raise SystemExit(f"pyrealsense2 is required for this bench: {exc}") from exc

DEFAULT_CONFIG_PATH = Path(__file__).resolve().with_name("fr3_record_config.yaml")
# Mirrors RealSenseCamera.frame_history, which is what read_closest searches.
FRAME_HISTORY = 8


@dataclass(frozen=True)
class FrameRecord:
    """One delivered frame, as the host saw it.

    ``counter`` is the sensor's own sequence number, read from frame metadata -- the only
    quantity here that can prove a frame existed and never arrived. On a Linux kernel without
    the RealSense metadata patches it is absent, and ``frame_number`` (assigned by librealsense
    on delivery) is all there is: it counts what arrived, so it cannot see a drop. The summary
    says which one it had rather than reporting zero drops from the counter that cannot find any.
    """

    counter: int | None
    frame_number: int
    device_timestamp_ms: float
    domain_name: str
    exposure_us: float | None
    handover_perf_s: float
    acquisition_perf_s: float


@dataclass
class CameraStream:
    """A camera's read loop and the buffer the poll thread selects from."""

    name: str
    serial: str
    width: int
    height: int
    fps: int
    lock: threading.Lock = field(default_factory=threading.Lock)
    history: deque[float] = field(default_factory=lambda: deque(maxlen=FRAME_HISTORY))
    latest: float | None = None
    records: list[FrameRecord] = field(default_factory=list)
    info: dict[str, Any] = field(default_factory=dict)
    errors: list[str] = field(default_factory=list)


def _option(sensor: Any, option: Any) -> float | None:
    """Read a sensor option, or None where this model does not carry it."""
    try:
        if not sensor.supports(option):
            return None
        return float(sensor.get_option(option))
    except Exception:  # noqa: BLE001 - an unsupported option must not end the run
        return None


def _metadata(frame: Any, value: Any) -> float | None:
    try:
        if not frame.supports_frame_metadata(value):
            return None
        return float(frame.get_frame_metadata(value))
    except Exception:  # noqa: BLE001 - same
        return None


def _color_sensor(device: Any) -> Any:
    """The sensor carrying the colour stream, on models that do not expose one by that name.

    A D435i answers ``first_color_sensor()``; a D405 produces colour from its stereo module and
    may not, so fall back to whichever sensor advertises a colour profile.
    """
    try:
        return device.first_color_sensor()
    except Exception:  # noqa: BLE001 - model difference, not a failure
        pass
    for sensor in device.query_sensors():
        for stream_profile in sensor.get_stream_profiles():
            if stream_profile.stream_type() == rs.stream.color:
                return sensor
    return device.query_sensors()[0]


def _acquisition_perf_s(frame: Any, *, handover_perf_s: float, handover_wall_s: float) -> float:
    """The stamping rule from RealSenseCamera._frame_capture_time_s, copied so it is measured.

    A device clock off the host timeline has an arbitrary epoch, so the fallback is the handover
    instant -- which carries this camera's pipeline delay and is exactly the failure mode that
    once put a fabricated 24 ms between two views of one instant.
    """
    domain = frame.get_frame_timestamp_domain()
    if domain not in (rs.timestamp_domain.global_time, rs.timestamp_domain.system_time):
        return handover_perf_s
    frame_age_s = handover_wall_s - (frame.get_timestamp() / 1000.0)
    if not (0.0 <= frame_age_s <= 1.0):
        return handover_perf_s
    return handover_perf_s - frame_age_s


def _read_loop(stream: CameraStream, stop: threading.Event, *, enable_global_time: bool) -> None:
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_device(stream.serial)
    config.enable_stream(rs.stream.color, stream.width, stream.height, rs.format.rgb8, stream.fps)
    profile = pipeline.start(config)

    device = profile.get_device()
    color_sensor = profile.get_stream(rs.stream.color).as_video_stream_profile()
    sensor = _color_sensor(device)
    if enable_global_time:
        try:
            sensor.set_option(rs.option.global_time_enabled, 1)
        except Exception as exc:  # noqa: BLE001
            stream.errors.append(f"could not enable global time: {exc}")

    def _device_info(field_name: str) -> str:
        try:
            return str(device.get_info(getattr(rs.camera_info, field_name)))
        except Exception:  # noqa: BLE001
            return "?"

    stream.info = {
        "model": _device_info("name"),
        "firmware": _device_info("firmware_version"),
        "usb_type": _device_info("usb_type_descriptor"),
        "negotiated": f"{color_sensor.width()}x{color_sensor.height()}@{color_sensor.fps()}",
        "requested": f"{stream.width}x{stream.height}@{stream.fps}",
        "global_time": _option(sensor, rs.option.global_time_enabled),
        "auto_exposure": _option(sensor, rs.option.enable_auto_exposure),
        "auto_exposure_priority": _option(sensor, rs.option.auto_exposure_priority),
    }

    try:
        while not stop.is_set():
            try:
                frames = pipeline.wait_for_frames(timeout_ms=1000)
            except Exception as exc:  # noqa: BLE001 - a timeout is data, not a crash
                stream.errors.append(f"wait_for_frames: {exc}")
                continue
            color = frames.get_color_frame()
            if not color:
                stream.errors.append("frame set carried no colour frame")
                continue
            handover_perf_s = time.perf_counter()
            handover_wall_s = time.time()
            # The same copy the real read loop makes, so its cost is in the measurement.
            np.asanyarray(color.get_data())
            counter = _metadata(color, rs.frame_metadata_value.frame_counter)
            record = FrameRecord(
                counter=None if counter is None else int(counter),
                frame_number=int(color.get_frame_number()),
                device_timestamp_ms=float(color.get_timestamp()),
                domain_name=str(color.get_frame_timestamp_domain()),
                exposure_us=_metadata(color, rs.frame_metadata_value.actual_exposure),
                handover_perf_s=handover_perf_s,
                acquisition_perf_s=_acquisition_perf_s(
                    color, handover_perf_s=handover_perf_s, handover_wall_s=handover_wall_s
                ),
            )
            with stream.lock:
                stream.records.append(record)
                stream.history.append(record.acquisition_perf_s)
                stream.latest = record.acquisition_perf_s
    finally:
        pipeline.stop()


@dataclass(frozen=True)
class PollSample:
    """One tick of the anchored selection the robot performs per frame."""

    tick_perf_s: float
    selected: dict[str, float]


def _poll_loop(
    streams: list[CameraStream],
    stop: threading.Event,
    *,
    poll_hz: float,
    extra_work_ms: float,
    samples: list[PollSample],
) -> None:
    period_s = 1.0 / poll_hz
    burn = np.zeros(4096, dtype=np.float64)
    next_tick = time.perf_counter()
    while not stop.is_set():
        next_tick += period_s
        latest: dict[str, float] = {}
        histories: dict[str, tuple[float, ...]] = {}
        for stream in streams:
            with stream.lock:
                if stream.latest is None:
                    continue
                latest[stream.name] = stream.latest
                histories[stream.name] = tuple(stream.history)
        if len(latest) == len(streams):
            anchor = min(latest.values())
            selected = {
                name: min(history, key=lambda value: abs(value - anchor))
                for name, history in histories.items()
            }
            samples.append(PollSample(tick_perf_s=time.perf_counter(), selected=selected))
        if extra_work_ms > 0:
            # Stand-in for the recorder's per-frame cost (colour conversion, encoder feed,
            # parquet append). Numpy releases the GIL the way that work does.
            deadline = time.perf_counter() + extra_work_ms / 1000.0
            while time.perf_counter() < deadline:
                np.square(burn).sum()
        sleep_s = next_tick - time.perf_counter()
        if sleep_s > 0:
            time.sleep(sleep_s)
        else:
            next_tick = time.perf_counter()


def _ms_stats(values: list[float]) -> dict[str, float]:
    if not values:
        return {"p50": 0.0, "p95": 0.0, "max": 0.0}
    ordered = sorted(values)
    return {
        "p50": statistics.median(ordered) * 1e3,
        "p95": ordered[min(len(ordered) - 1, int(0.95 * len(ordered)))] * 1e3,
        "max": ordered[-1] * 1e3,
    }


def summarize_stream(records: list[FrameRecord], *, nominal_fps: float) -> dict[str, Any]:
    """Split a camera's delivery into sensor-side and host-side failures.

    ``dropped_frames`` comes from the frame counter and nothing else: it is the only quantity
    that distinguishes a frame the sensor never produced from one this host failed to collect.
    """
    if len(records) < 2:
        return {"frames": len(records), "note": "not enough frames to measure"}

    elapsed_s = records[-1].acquisition_perf_s - records[0].acquisition_perf_s
    acquisition_gaps = [
        later.acquisition_perf_s - earlier.acquisition_perf_s
        for earlier, later in zip(records[:-1], records[1:], strict=True)
    ]
    handover_lags = [record.handover_perf_s - record.acquisition_perf_s for record in records]

    counters = [record.counter for record in records if record.counter is not None]
    counter_available = len(counters) == len(records) and len(counters) >= 2
    if not counter_available:
        # librealsense's own delivery counter. It cannot see a drop -- it counts what arrived --
        # but a gap in it still means this process missed a delivery the library made.
        counters = [record.frame_number for record in records]
    dropped = 0
    drop_events = 0
    for earlier, later in zip(counters[:-1], counters[1:], strict=True):
        missing = later - earlier - 1
        if missing > 0:
            dropped += missing
            drop_events += 1

    exposures = [record.exposure_us for record in records if record.exposure_us is not None]
    domains: dict[str, int] = {}
    for record in records:
        domains[record.domain_name] = domains.get(record.domain_name, 0) + 1

    nominal_period_s = 1.0 / nominal_fps if nominal_fps > 0 else 0.0
    return {
        "frames": len(records),
        "elapsed_s": elapsed_s,
        "delivered_fps": (len(records) - 1) / elapsed_s if elapsed_s > 0 else 0.0,
        "frame_counter_available": counter_available,
        "dropped_frames": dropped,
        "drop_events": drop_events,
        "acquisition_gap_ms": _ms_stats(acquisition_gaps),
        # Wider than nominal with contiguous counters = the sensor itself slowed down.
        "slow_sensor_intervals": sum(
            1 for gap in acquisition_gaps if nominal_period_s and gap > 1.5 * nominal_period_s
        ),
        "handover_lag_ms": _ms_stats(handover_lags),
        "exposure_us": {
            "p50": statistics.median(exposures) if exposures else None,
            "max": max(exposures) if exposures else None,
        },
        "timestamp_domains": domains,
    }


def summarize_poll(samples: list[PollSample], *, camera_names: list[str]) -> dict[str, Any]:
    """Cross-camera skew and frame reuse, as the recorder would have recorded them."""
    if len(samples) < 2:
        return {"ticks": len(samples), "note": "not enough ticks to measure"}
    skews = [max(s.selected.values()) - min(s.selected.values()) for s in samples]
    staleness = [s.tick_perf_s - min(s.selected.values()) for s in samples]
    reused = 0
    for earlier, later in zip(samples[:-1], samples[1:], strict=True):
        if all(later.selected[name] == earlier.selected[name] for name in camera_names):
            reused += 1
    return {
        "ticks": len(samples),
        "reused_frame_fraction": reused / (len(samples) - 1),
        "cross_camera_skew_ms": _ms_stats(skews),
        "anchor_staleness_ms": _ms_stats(staleness),
    }


def _cameras_from_config(config_path: Path) -> list[dict[str, Any]]:
    import yaml

    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    cameras = ((config or {}).get("robot") or {}).get("cameras") or {}
    resolved = []
    for name, camera in cameras.items():
        if str(camera.get("type", "")) != "intelrealsense":
            continue
        resolved.append(
            {
                "name": str(name),
                "serial": str(camera["serial_number_or_name"]),
                "width": int(camera.get("width", 640)),
                "height": int(camera.get("height", 480)),
                "fps": int(camera.get("fps", 30)),
            }
        )
    if not resolved:
        raise SystemExit(f"No intelrealsense cameras in {config_path}")
    return resolved


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG_PATH)
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        metavar="NAME=SERIAL[@WIDTHxHEIGHT@FPS]",
        help="Override the config's cameras. Repeat per camera.",
    )
    parser.add_argument("--duration", type=float, default=60.0, help="Seconds to stream.")
    parser.add_argument(
        "--poll-hz",
        type=float,
        default=60.0,
        help="Rate the anchored selection runs at; the recorder's dataset.fps.",
    )
    parser.add_argument(
        "--extra-work-ms",
        type=float,
        default=0.0,
        help="Synthetic per-tick host work, to dose the load hypothesis.",
    )
    parser.add_argument(
        "--no-global-time",
        action="store_true",
        help="Do not force global time on; measure what the driver defaults to.",
    )
    parser.add_argument("--json", type=Path, default=None, help="Also write the report here.")
    return parser.parse_args(argv)


def _parse_camera_flag(value: str) -> dict[str, Any]:
    name, _, rest = value.partition("=")
    serial, _, profile = rest.partition("@")
    width, height, fps = 640, 480, 60
    if profile:
        resolution, _, rate = profile.partition("@")
        if "x" in resolution:
            width, height = (int(part) for part in resolution.split("x", 1))
        if rate:
            fps = int(rate)
    return {"name": name, "serial": serial, "width": width, "height": height, "fps": fps}


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    specs = (
        [_parse_camera_flag(value) for value in args.camera]
        if args.camera
        else _cameras_from_config(args.config)
    )
    streams = [CameraStream(**spec) for spec in specs]

    stop = threading.Event()
    samples: list[PollSample] = []
    threads = [
        threading.Thread(
            target=_read_loop,
            args=(stream, stop),
            kwargs={"enable_global_time": not args.no_global_time},
            daemon=True,
        )
        for stream in streams
    ]
    for thread in threads:
        thread.start()
    # Let every camera publish a first frame before the poll thread starts asking for one.
    time.sleep(2.0)
    poll_thread = threading.Thread(
        target=_poll_loop,
        args=(streams, stop),
        kwargs={
            "poll_hz": args.poll_hz,
            "extra_work_ms": args.extra_work_ms,
            "samples": samples,
        },
        daemon=True,
    )
    poll_thread.start()

    time.sleep(args.duration)
    stop.set()
    poll_thread.join(timeout=5.0)
    for thread in threads:
        thread.join(timeout=5.0)

    report: dict[str, Any] = {
        "poll_hz": args.poll_hz,
        "extra_work_ms": args.extra_work_ms,
        "duration_s": args.duration,
        "cameras": {},
        "poll": summarize_poll(samples, camera_names=[stream.name for stream in streams]),
    }
    for stream in streams:
        with stream.lock:
            records = list(stream.records)
        summary = summarize_stream(records, nominal_fps=stream.fps)
        report["cameras"][stream.name] = {
            "serial": stream.serial,
            "info": stream.info,
            "errors": stream.errors[:20],
            **summary,
        }

    print(f"poll {args.poll_hz:g} Hz   extra work {args.extra_work_ms:g} ms/tick   {args.duration:g} s")
    for name, camera in report["cameras"].items():
        info = camera["info"]
        print(
            f"\n{name}  {camera['serial']}  {info.get('model')}  fw {info.get('firmware')}  "
            f"USB {info.get('usb_type')}  profile {info.get('negotiated')} "
            f"(requested {info.get('requested')})"
        )
        print(
            f"    global_time={info.get('global_time')}  auto_exposure={info.get('auto_exposure')}  "
            f"ae_priority={info.get('auto_exposure_priority')}"
        )
        if "note" in camera:
            print(f"    {camera['note']}")
            continue
        print(
            f"    delivered      {camera['frames']} frames in {camera['elapsed_s']:.1f} s "
            f"-> {camera['delivered_fps']:.1f} fps"
        )
        print(
            f"    missed frames  {camera['dropped_frames']} in {camera['drop_events']} event(s)"
            + (
                "   [from the sensor's frame_counter: these existed and never arrived]"
                if camera["frame_counter_available"]
                else "   [no frame_counter metadata on this kernel; counted from librealsense "
                "delivery numbers, which cannot see a sensor-side drop]"
            )
        )
        gap = camera["acquisition_gap_ms"]
        print(
            f"    acquisition    p50 {gap['p50']:6.1f}  p95 {gap['p95']:6.1f}  max {gap['max']:6.1f} ms"
            f"   ({camera['slow_sensor_intervals']} interval(s) >1.5x nominal)"
        )
        lag = camera["handover_lag_ms"]
        print(f"    handover lag   p50 {lag['p50']:6.1f}  p95 {lag['p95']:6.1f}  max {lag['max']:6.1f} ms")
        exposure = camera["exposure_us"]
        if exposure["p50"] is not None:
            print(
                f"    exposure       p50 {exposure['p50'] / 1000.0:.1f} ms  "
                f"max {exposure['max'] / 1000.0:.1f} ms"
            )
        print(f"    domains        {camera['timestamp_domains']}")
        for error in camera["errors"]:
            print(f"    ERROR          {error}")

    poll = report["poll"]
    print(f"\npoll (anchored selection, {len(streams)} cameras)")
    if "note" in poll:
        print(f"    {poll['note']}")
    else:
        skew, stale = poll["cross_camera_skew_ms"], poll["anchor_staleness_ms"]
        print(
            f"    ticks {poll['ticks']}   reused frame on "
            f"{100 * poll['reused_frame_fraction']:.1f}% of ticks"
        )
        print(
            f"    cross-camera skew  p50 {skew['p50']:6.1f}  p95 {skew['p95']:6.1f}  "
            f"max {skew['max']:6.1f} ms"
        )
        print(
            f"    anchor staleness   p50 {stale['p50']:6.1f}  p95 {stale['p95']:6.1f}  "
            f"max {stale['max']:6.1f} ms"
        )

    if args.json is not None:
        args.json.write_text(json.dumps(report, indent=2), encoding="utf-8")
        print(f"\nreport={args.json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
