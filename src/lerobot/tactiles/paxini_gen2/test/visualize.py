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
Visualize one or more Paxini tactile sensors as annotated taxel grids.

Example:
    python3 src/lerobot/tactiles/paxini_gen2/test/visualize.py \
        --sensor "name=left,serial_port=/dev/ttyUSB0,connect_id=1,model_name=GEN2-IP-L5325,control_mode=5" \
        --sensor "name=right,serial_port=/dev/ttyUSB0,connect_id=2,model_name=GEN2-IP-M3025,control_mode=5"
"""

from __future__ import annotations

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))

from lerobot.tactiles.paxini_gen2 import PaxiniGen2OmegaTactile, PaxiniGen2OmegaTactileConfig

DEFAULT_BAUDRATE = 460800
DEFAULT_TIMEOUT = 1.0
DEFAULT_FPS = 30
DEFAULT_ASYNC_TIMEOUT_MS = 200.0
DEFAULT_MAX_AGE_MS = 2000
DEFAULT_REFRESH_PAUSE_S = 0.001


@dataclass
class SensorSpec:
    name: str
    serial_port: str
    connect_id: int
    model_name: str
    control_mode: int
    baudrate: int = DEFAULT_BAUDRATE
    timeout: float = DEFAULT_TIMEOUT
    fps: int = DEFAULT_FPS


@dataclass
class SensorPanel:
    tactile: PaxiniGen2OmegaTactile
    spec: SensorSpec
    ax: Any
    image: Any
    texts: list[Any]
    rows: int
    cols: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sensor",
        action="append",
        default=[],
        help=(
            "Sensor specification. Repeat this flag for multiple sensors. "
            "Format: name=left,serial_port=/dev/ttyUSB0,connect_id=1,model_name=GEN2-IP-L5325,control_mode=5"
        ),
    )
    parser.add_argument(
        "--async-timeout-ms",
        type=float,
        default=DEFAULT_ASYNC_TIMEOUT_MS,
        help="Timeout passed to tactile.async_read().",
    )
    parser.add_argument(
        "--max-age-ms",
        type=int,
        default=DEFAULT_MAX_AGE_MS,
        help="Fallback age threshold for tactile.read_latest() after async timeout.",
    )
    parser.add_argument(
        "--magnitude-max",
        type=float,
        default=None,
        help="Fixed color scale maximum. If omitted, use an adaptive maximum.",
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

    required_keys = ["serial_port", "connect_id", "model_name", "control_mode"]
    missing_keys = [key for key in required_keys if key not in items]
    if missing_keys:
        raise ValueError(f"Missing keys in --sensor spec: {missing_keys}. Raw spec: {raw_spec!r}")

    name = items.get("name", f"sensor_{index}")
    return SensorSpec(
        name=name,
        serial_port=items["serial_port"],
        connect_id=int(items["connect_id"]),
        model_name=items["model_name"],
        control_mode=int(items["control_mode"]),
        baudrate=int(items.get("baudrate", DEFAULT_BAUDRATE)),
        timeout=float(items.get("timeout", DEFAULT_TIMEOUT)),
        fps=int(items.get("fps", DEFAULT_FPS)),
    )


def infer_grid_shape(num_taxels: int) -> tuple[int, int]:
    best_rows = 1
    best_cols = num_taxels
    best_gap = num_taxels - 1

    for rows in range(1, int(math.sqrt(num_taxels)) + 1):
        cols = math.ceil(num_taxels / rows)
        gap = cols - rows
        if rows * cols >= num_taxels and gap < best_gap:
            best_rows = rows
            best_cols = cols
            best_gap = gap

    return best_rows, best_cols


def infer_font_size(rows: int, cols: int) -> float:
    return max(6.5, min(11.0, 64.0 / max(rows, cols)))


def magnitude_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list(
        "paxini_magnitude",
        ["#ffffff", "#e8f1ff", "#afc8f8", "#5a88d9", "#1f4f9a", "#0d2148"],
    )


def create_tactiles(specs: list[SensorSpec]) -> list[PaxiniGen2OmegaTactile]:
    tactiles: list[PaxiniGen2OmegaTactile] = []
    for spec in specs:
        config = PaxiniGen2OmegaTactileConfig(
            serial_port=spec.serial_port,
            baudrate=spec.baudrate,
            timeout=spec.timeout,
            control_mode=spec.control_mode,
            model_name=spec.model_name,
            connect_id=spec.connect_id,
            fps=spec.fps,
        )
        tactiles.append(PaxiniGen2OmegaTactile(config))
    return tactiles


def build_panels(tactiles: list[PaxiniGen2OmegaTactile], specs: list[SensorSpec], figure_scale: float) -> tuple[Any, list[SensorPanel], Any]:
    import matplotlib.pyplot as plt

    sensor_count = len(tactiles)
    subplot_cols = math.ceil(math.sqrt(sensor_count))
    subplot_rows = math.ceil(sensor_count / subplot_cols)
    fig, axes = plt.subplots(
        subplot_rows,
        subplot_cols,
        figsize=(6.8 * subplot_cols * figure_scale, 6.6 * subplot_rows * figure_scale),
        squeeze=False,
    )
    axes_flat = axes.flatten()
    cmap = magnitude_cmap()
    panels: list[SensorPanel] = []
    shared_image = None

    for i, (tactile, spec) in enumerate(zip(tactiles, specs, strict=True)):
        ax = axes_flat[i]
        rows, cols = infer_grid_shape(tactile.num_taxels or 120)
        empty_grid = np.zeros((rows, cols), dtype=np.float32)
        image = ax.imshow(empty_grid, cmap=cmap, vmin=0.0, vmax=1.0, interpolation="nearest", aspect="equal")
        if shared_image is None:
            shared_image = image

        font_size = infer_font_size(rows, cols)
        texts: list[Any] = []
        for row in range(rows):
            for col in range(cols):
                taxel_index = row * cols + col
                if taxel_index >= (tactile.num_taxels or 120):
                    continue
                text = ax.text(
                    col,
                    row,
                    "x:0\ny:0\nz:0",
                    ha="center",
                    va="center",
                    fontsize=font_size,
                    color="#1f1f1f",
                )
                texts.append(text)

        ax.set_xticks(np.arange(-0.5, cols, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, rows, 1), minor=True)
        ax.grid(which="minor", color="#d7dce5", linewidth=0.7)
        ax.tick_params(which="both", bottom=False, left=False, labelbottom=False, labelleft=False)
        ax.set_title(
            f"{spec.name}\nport={spec.serial_port} connect_id={spec.connect_id}",
            fontsize=11,
        )

        panels.append(
            SensorPanel(
                tactile=tactile,
                spec=spec,
                ax=ax,
                image=image,
                texts=texts,
                rows=rows,
                cols=cols,
            )
        )

    for ax in axes_flat[len(panels) :]:
        ax.axis("off")

    fig.subplots_adjust(left=0.04, right=0.9, bottom=0.04, top=0.9, wspace=0.18, hspace=0.24)
    colorbar_ax = fig.add_axes([0.92, 0.14, 0.018, 0.68])
    colorbar = fig.colorbar(shared_image, cax=colorbar_ax)
    colorbar.set_label("sqrt(x^2 + y^2 + z^2)", rotation=270, labelpad=18)
    fig.suptitle("Paxini Tactile Visualization", fontsize=15)
    return fig, panels, colorbar


def fetch_frame(panel: SensorPanel, async_timeout_ms: float, max_age_ms: int) -> np.ndarray | None:
    try:
        return np.asarray(panel.tactile.async_read(timeout_ms=async_timeout_ms), dtype=np.float32)
    except TimeoutError:
        try:
            return np.asarray(panel.tactile.read_latest(max_age_ms=max_age_ms), dtype=np.float32)
        except Exception:
            return None


def update_panel(panel: SensorPanel, frame: np.ndarray, magnitude_limit: float) -> None:
    magnitudes = np.linalg.norm(frame, axis=1)
    heatmap = np.zeros((panel.rows, panel.cols), dtype=np.float32)
    limit = max(magnitude_limit, 1e-6)

    for taxel_index in range(panel.rows * panel.cols):
        row = taxel_index // panel.cols
        col = taxel_index % panel.cols
        if taxel_index >= frame.shape[0]:
            heatmap[row, col] = np.nan
            continue

        x_value, y_value, z_value = frame[taxel_index]
        magnitude = float(magnitudes[taxel_index])
        heatmap[row, col] = magnitude

        normalized = magnitude / limit
        text_color = "#0f172a" if normalized < 0.55 else "#ffffff"
        panel.texts[taxel_index].set_text(
            f"x:{x_value:>4.0f}\ny:{y_value:>4.0f}\nz:{z_value:>4.0f}"
        )
        panel.texts[taxel_index].set_color(text_color)

    panel.image.set_data(heatmap)
    panel.image.set_clim(0.0, limit)
    panel.ax.set_title(
        f"{panel.spec.name}\n"
        f"port={panel.spec.serial_port} connect_id={panel.spec.connect_id} "
        f"mean={magnitudes.mean():.1f} max={magnitudes.max():.1f}",
        fontsize=11,
    )


def list_ports() -> None:
    ports = PaxiniGen2OmegaTactile.find_tactiles()
    if not ports:
        print("No serial ports found.")
        return

    for port in ports:
        print(f"{port['serial_port']}: {port['description']} ({port['hwid']})")


# python3 src/lerobot/tactiles/paxini_gen2/test/visualize.py \
# --sensor "name=left,serial_port=/dev/ttyACM0,connect_id=6,model_name=GEN2-IP-L5325,control_mode=5" \
# --sensor "name=right,serial_port=/dev/ttyACM0,connect_id=10,model_name=GEN2-IP-L5325,control_mode=5"
def main() -> int:
    args = parse_args()

    if args.list_ports:
        list_ports()
        return 0

    if not args.sensor:
        raise SystemExit(
            "At least one --sensor is required. "
            "Example: --sensor \"name=left,serial_port=/dev/ttyUSB0,connect_id=1,model_name=GEN2-IP-L5325,control_mode=5\""
        )

    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:
        raise SystemExit(
            "matplotlib is required for visualization. Install it with `pip install matplotlib` "
            "or use the project's matplotlib extra."
        ) from exc

    specs = [parse_sensor_spec(raw_spec, i) for i, raw_spec in enumerate(args.sensor, start=1)]
    tactiles = create_tactiles(specs)

    for tactile in tactiles:
        tactile.connect()

    fig = None
    try:
        fig, panels, colorbar = build_panels(tactiles, specs, args.figure_scale)
        adaptive_limit = 1.0

        while plt.fignum_exists(fig.number):
            updated_frames: list[tuple[SensorPanel, np.ndarray]] = []
            observed_max = 0.0

            for panel in panels:
                frame = fetch_frame(panel, args.async_timeout_ms, args.max_age_ms)
                if frame is None:
                    print(f"Warning: No valid frame for panel {panel.spec.name}. Skipping update.")
                    continue

                updated_frames.append((panel, frame))
                observed_max = max(observed_max, float(np.linalg.norm(frame, axis=1).max()))

            if not updated_frames:
                plt.pause(DEFAULT_REFRESH_PAUSE_S)
                continue

            if args.magnitude_max is None:
                adaptive_limit = max(observed_max, adaptive_limit * 0.98, 1.0)
                magnitude_limit = adaptive_limit
            else:
                magnitude_limit = max(args.magnitude_max, 1e-6)

            for panel, frame in updated_frames:
                update_panel(panel, frame, magnitude_limit)

            colorbar.mappable.set_clim(0.0, magnitude_limit)
            fig.canvas.draw_idle()
            plt.pause(DEFAULT_REFRESH_PAUSE_S)

    except KeyboardInterrupt:
        pass
    finally:
        for tactile in tactiles:
            try:
                if tactile.is_connected:
                    tactile.disconnect()
            except Exception:
                pass
        if fig is not None:
            plt.close(fig)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
