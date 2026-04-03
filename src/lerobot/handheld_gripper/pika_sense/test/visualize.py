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

"""
Visualize one or more Pika Sense handheld grippers as width-vs-time curves.

Example:
    python3 src/lerobot/handheld_gripper/pika_sense/test/visualize.py \
        --sensor "name=left,port=/dev/ttyUSB0" \
        --sensor "name=right,port=/dev/ttyUSB1"
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lerobot.handheld_gripper.pika_sense import PikaSense, PikaSenseConfig

DEFAULT_FPS = 120
DEFAULT_WARMUP_S = 1.0
DEFAULT_ASYNC_TIMEOUT_MS = 200.0
DEFAULT_MAX_AGE_MS = 1000
DEFAULT_HISTORY_S = 10.0
DEFAULT_REFRESH_PAUSE_S = 0.001
DEFAULT_LINE_COLOR = "#1f5aa6"


@dataclass
class SensorSpec:
    name: str
    port: str
    fps: int = DEFAULT_FPS
    warmup_s: float = DEFAULT_WARMUP_S


@dataclass
class SensorPanel:
    gripper: PikaSense
    spec: SensorSpec
    ax: Any
    line: Any
    marker: Any
    current_text: Any
    min_text: Any
    max_text: Any
    times: deque[float] = field(default_factory=lambda: deque(maxlen=1))
    widths: deque[float] = field(default_factory=lambda: deque(maxlen=1))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensor",
        action="append",
        default=[],
        help=(
            "Sensor specification. Repeat this flag for multiple sensors. "
            "Format: name=left,port=/dev/ttyUSB0,fps=120,warmup_s=1.0"
        ),
    )
    parser.add_argument(
        "--async-timeout-ms",
        type=float,
        default=DEFAULT_ASYNC_TIMEOUT_MS,
        help="Timeout passed to gripper.async_read().",
    )
    parser.add_argument(
        "--max-age-ms",
        type=int,
        default=DEFAULT_MAX_AGE_MS,
        help="Fallback age threshold for gripper.read_latest() after async timeout.",
    )
    parser.add_argument(
        "--history-s",
        type=float,
        default=DEFAULT_HISTORY_S,
        help="Time window displayed on the x-axis in seconds.",
    )
    parser.add_argument(
        "--width-min",
        type=float,
        default=None,
        help="Fixed lower y-axis bound in mm. If omitted, infer from data.",
    )
    parser.add_argument(
        "--width-max",
        type=float,
        default=None,
        help="Fixed upper y-axis bound in mm. If omitted, infer from data.",
    )
    parser.add_argument(
        "--figure-scale",
        type=float,
        default=1.0,
        help="Multiply the default figure size by this value.",
    )
    parser.add_argument(
        "--list-ports",
        action="store_true",
        help="List visible serial ports and exit.",
    )
    return parser.parse_args()


def parse_sensor_spec(raw_spec: str, index: int) -> SensorSpec:
    items = {}
    for chunk in raw_spec.split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "=" not in chunk:
            raise ValueError(f"Invalid sensor spec entry {chunk!r}. Expected key=value.")
        key, value = chunk.split("=", 1)
        items[key.strip()] = value.strip()

    if "port" not in items:
        raise ValueError(f"Missing `port` in --sensor spec: {raw_spec!r}")

    return SensorSpec(
        name=items.get("name", f"gripper_{index}"),
        port=items["port"],
        fps=int(items.get("fps", DEFAULT_FPS)),
        warmup_s=float(items.get("warmup_s", DEFAULT_WARMUP_S)),
    )


def create_grippers(specs: list[SensorSpec]) -> list[PikaSense]:
    grippers: list[PikaSense] = []
    for spec in specs:
        config = PikaSenseConfig(
            port=spec.port,
            fps=spec.fps,
            warmup_s=spec.warmup_s,
        )
        grippers.append(PikaSense(config))
    return grippers


def build_panels(grippers: list[PikaSense], specs: list[SensorSpec], history_s: float, figure_scale: float) -> tuple[Any, list[SensorPanel]]:
    import matplotlib.pyplot as plt

    sensor_count = len(grippers)
    subplot_cols = math.ceil(math.sqrt(sensor_count))
    subplot_rows = math.ceil(sensor_count / subplot_cols)
    fig, axes = plt.subplots(
        subplot_rows,
        subplot_cols,
        figsize=(7.2 * subplot_cols * figure_scale, 4.8 * subplot_rows * figure_scale),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    panels: list[SensorPanel] = []

    for index, (gripper, spec) in enumerate(zip(grippers, specs, strict=True)):
        ax = axes_flat[index]
        line, = ax.plot([], [], color=DEFAULT_LINE_COLOR, linewidth=2.2)
        marker, = ax.plot([], [], "o", color="#d94841", markersize=6)
        current_text = ax.text(
            0.02,
            0.95,
            "-- mm",
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=18,
            fontweight="bold",
            color="#111111",
        )
        min_text = ax.text(
            0.02,
            0.05,
            "min: -- mm",
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=10,
            color="#4c566a",
        )
        max_text = ax.text(
            0.98,
            0.05,
            "max: -- mm",
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color="#4c566a",
        )
        ax.set_title(f"{spec.name}\nport={spec.port}", fontsize=11)
        ax.set_xlabel(f"Time Window ({history_s:.1f}s)")
        ax.set_ylabel("Width (mm)")
        ax.grid(True, color="#d7dce5", linewidth=0.8)
        ax.set_facecolor("#fbfcfe")

        maxlen = max(int(math.ceil(spec.fps * history_s * 1.5)), 32)
        panels.append(
            SensorPanel(
                gripper=gripper,
                spec=spec,
                ax=ax,
                line=line,
                marker=marker,
                current_text=current_text,
                min_text=min_text,
                max_text=max_text,
                times=deque(maxlen=maxlen),
                widths=deque(maxlen=maxlen),
            )
        )

    for ax in axes_flat[len(panels) :]:
        ax.set_visible(False)

    fig.subplots_adjust(left=0.08, right=0.98, bottom=0.1, top=0.9, wspace=0.24, hspace=0.32)
    return fig, panels


def print_ports() -> None:
    ports = PikaSense.find_handheld_grippers()
    if not ports:
        print("No serial ports detected.")
        return

    print("\n--- Detected Handheld Grippers ---")
    for index, port in enumerate(ports):
        print(f"Device #{index}:")
        for key, value in port.items():
            print(f"  {key}: {value}")
        print("-" * 20)


def fetch_width(gripper: PikaSense, async_timeout_ms: float, max_age_ms: int) -> float | None:
    try:
        return gripper.async_read(timeout_ms=async_timeout_ms)
    except TimeoutError:
        pass

    try:
        return gripper.read_latest(max_age_ms=max_age_ms)
    except Exception:
        return None


def update_panel(panel: SensorPanel, width_mm: float, now_s: float, history_s: float, width_min: float | None, width_max: float | None) -> None:
    panel.times.append(now_s)
    panel.widths.append(width_mm)

    latest_time = panel.times[-1]
    x_values = [timestamp - latest_time for timestamp in panel.times]
    y_values = list(panel.widths)
    panel.line.set_data(x_values, y_values)
    panel.marker.set_data([x_values[-1]], [y_values[-1]])

    panel.current_text.set_text(f"{width_mm:.2f} mm")
    panel.min_text.set_text(f"min: {min(y_values):.2f} mm")
    panel.max_text.set_text(f"max: {max(y_values):.2f} mm")

    panel.ax.set_xlim(-history_s, 0.0)

    if width_min is not None and width_max is not None:
        panel.ax.set_ylim(width_min, width_max)
        return

    data_min = min(y_values) if width_min is None else width_min
    data_max = max(y_values) if width_max is None else width_max
    if math.isclose(data_min, data_max):
        data_min -= 1.0
        data_max += 1.0

    margin = max((data_max - data_min) * 0.15, 1.0)
    panel.ax.set_ylim(
        width_min if width_min is not None else data_min - margin,
        width_max if width_max is not None else data_max + margin,
    )


def main() -> int:
    args = parse_args()

    if args.list_ports:
        print_ports()
        return 0

    if not args.sensor:
        raise ValueError("At least one `--sensor` must be provided unless `--list-ports` is used.")

    specs = [parse_sensor_spec(raw_spec, index) for index, raw_spec in enumerate(args.sensor)]
    grippers = create_grippers(specs)

    connected_grippers: list[PikaSense] = []
    try:
        for gripper in grippers:
            gripper.connect()
            connected_grippers.append(gripper)

        import matplotlib.pyplot as plt

        fig, panels = build_panels(grippers, specs, args.history_s, args.figure_scale)
        start_s = time.perf_counter()

        while plt.fignum_exists(fig.number):
            loop_now = time.perf_counter()
            elapsed_s = loop_now - start_s

            for panel in panels:
                width_mm = fetch_width(panel.gripper, args.async_timeout_ms, args.max_age_ms)
                if width_mm is None:
                    continue
                update_panel(panel, width_mm, elapsed_s, args.history_s, args.width_min, args.width_max)

            fig.canvas.draw_idle()
            plt.pause(DEFAULT_REFRESH_PAUSE_S)

    finally:
        for gripper in reversed(connected_grippers):
            try:
                gripper.disconnect()
            except Exception:
                pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
