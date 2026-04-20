#!/usr/bin/env python

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

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Protocol

import numpy as np


@dataclass(frozen=True)
class MarkerStyle:
    sphere_radius: float = 0.012
    axis_radius: float = 0.003
    axis_length: float = 0.06
    target_rgba: tuple[float, float, float, float] = (0.15, 0.85, 0.35, 0.85)
    tcp_rgba: tuple[float, float, float, float] = (0.95, 0.45, 0.10, 0.85)


class TeleopReader(Protocol):
    def get_action(self) -> dict[str, Any]: ...


class ViewerHandle(Protocol):
    user_scn: Any

    def is_running(self) -> bool: ...

    def lock(self) -> Any: ...

    def sync(self) -> None: ...

    def close(self) -> None: ...


def _axis_segments_from_pose(pose: np.ndarray, axis_length: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    origin = np.asarray(pose[:3, 3], dtype=np.float64)
    rotation = np.asarray(pose[:3, :3], dtype=np.float64)
    colors = (
        np.array([1.0, 0.2, 0.2, 0.95], dtype=np.float32),
        np.array([0.2, 1.0, 0.2, 0.95], dtype=np.float32),
        np.array([0.2, 0.5, 1.0, 0.95], dtype=np.float32),
    )
    segments = []
    for axis_idx, color in enumerate(colors):
        direction = rotation[:, axis_idx] * axis_length
        segments.append((origin, origin + direction, color))
    return segments


def marker_geoms_from_info(info: dict[str, Any], style: MarkerStyle) -> list[dict[str, Any]]:
    target_pose = np.asarray(info["target_pose"], dtype=np.float64)
    tcp_pose = np.asarray(info["tcp_pose"], dtype=np.float64)

    geoms: list[dict[str, Any]] = [
        {
            "kind": "sphere",
            "name": info["target_marker_name"],
            "pos": np.asarray(target_pose[:3, 3], dtype=np.float64),
            "rgba": np.asarray(style.target_rgba, dtype=np.float32),
            "size": np.array([style.sphere_radius, 0.0, 0.0], dtype=np.float64),
        },
        {
            "kind": "sphere",
            "name": info["tcp_marker_name"],
            "pos": np.asarray(tcp_pose[:3, 3], dtype=np.float64),
            "rgba": np.asarray(style.tcp_rgba, dtype=np.float32),
            "size": np.array([style.sphere_radius, 0.0, 0.0], dtype=np.float64),
        },
    ]
    for pose, prefix in ((target_pose, "target"), (tcp_pose, "tcp")):
        for axis_name, (start, end, color) in zip(("x", "y", "z"), _axis_segments_from_pose(pose, style.axis_length), strict=True):
            geoms.append(
                {
                    "kind": "connector",
                    "name": f"{prefix}_{axis_name}",
                    "start": start,
                    "end": end,
                    "rgba": color,
                    "radius": style.axis_radius,
                }
            )
    return geoms


def update_passive_viewer_markers(mujoco, viewer: ViewerHandle, info: dict[str, Any], style: MarkerStyle) -> None:
    geoms = marker_geoms_from_info(info, style)
    scene = viewer.user_scn
    if scene.maxgeom < len(geoms):
        raise RuntimeError(f"MuJoCo viewer user scene supports {scene.maxgeom} geoms, need {len(geoms)}.")

    scene.ngeom = 0
    for geom_index, geom_data in enumerate(geoms):
        geom = scene.geoms[geom_index]
        if geom_data["kind"] == "sphere":
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_SPHERE,
                size=geom_data["size"],
                pos=geom_data["pos"],
                mat=np.eye(3, dtype=np.float64).reshape(-1),
                rgba=geom_data["rgba"],
            )
        else:
            mujoco.mjv_initGeom(
                geom,
                type=mujoco.mjtGeom.mjGEOM_CAPSULE,
                size=np.array([geom_data["radius"], 0.0, 0.0], dtype=np.float64),
                pos=np.zeros(3, dtype=np.float64),
                mat=np.eye(3, dtype=np.float64).reshape(-1),
                rgba=geom_data["rgba"],
            )
            mujoco.mjv_connector(
                geom,
                mujoco.mjtGeom.mjGEOM_CAPSULE,
                float(geom_data["radius"]),
                np.asarray(geom_data["start"], dtype=np.float64),
                np.asarray(geom_data["end"], dtype=np.float64),
            )
            geom.rgba[:] = geom_data["rgba"]
        scene.ngeom += 1


def render_camera_grid(camera_obs: dict[str, np.ndarray], width: int, height: int) -> np.ndarray:
    """Assemble camera observations into a 2x2 grid as an RGB numpy array."""
    keys = ["third_person", "side", "wrist"]
    cells: list[list[np.ndarray]] = [[], []]
    for i, key in enumerate(keys):
        frame = camera_obs.get(key)
        if frame is None:
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        else:
            frame = np.asarray(frame, dtype=np.uint8)
        row = i // 2
        cells[row].append(frame)
    if len(keys) < 4:
        cells[1].append(np.zeros((height, width, 3), dtype=np.uint8))
    top_row = np.hstack(cells[0])
    bottom_row = np.hstack(cells[1])
    return np.vstack([top_row, bottom_row])


def run_sim_teleop_loop(
    *,
    env,
    teleop: TeleopReader,
    fps: int,
    viewer: ViewerHandle | None = None,
    viewer_data: Any | None = None,
    duration_s: float | None = None,
    max_steps: int | None = None,
    marker_style: MarkerStyle | None = None,
    render_cameras: bool = False,
    camera_width: int = 640,
    camera_height: int = 480,
) -> dict[str, Any]:
    marker_style = marker_style or MarkerStyle()
    start = time.perf_counter()
    steps = 0
    _, info = env.reset()
    sync_gripper = getattr(teleop, "sync_gripper_baseline", None)
    if callable(sync_gripper) and "gripper_command" in info:
        sync_gripper(float(info["gripper_command"]))

    if viewer is not None:
        copy_visual_state = getattr(env, "copy_visual_state", None)
        if viewer_data is not None and callable(copy_visual_state):
            copy_visual_state(viewer_data)
        with viewer.lock():
            update_passive_viewer_markers(env._mujoco, viewer, info, marker_style)
        viewer.sync()

    http_server = None
    screen = None

    if render_cameras:
        import http.server
        import os
        import socketserver
        import threading

        out_dir = "/tmp/camera_stream"
        os.makedirs(out_dir, exist_ok=True)

        with open(os.path.join(out_dir, "index.html"), "w") as f:
            f.write(
                """<!DOCTYPE html>
<html>
<head>
  <title>Camera Stream (2x2)</title>
  <style>
    body { margin: 0; background: #000; display: flex; align-items: center; justify-content: center; height: 100vh; overflow: hidden; }
    canvas { max-width: 100vw; max-height: 100vh; }
  </style>
</head>
<body>
  <canvas id="c"></canvas>
  <script>
    var canvas = document.getElementById('c');
    var ctx = canvas.getContext('2d');
    var img = new Image();
    var w = 0, h = 0;

    img.onload = function() {
      if (w !== img.width || h !== img.height) {
        w = img.width;
        h = img.height;
        canvas.width = w;
        canvas.height = h;
      }
      ctx.drawImage(img, 0, 0, w, h);
    };

    function update() {
      img.src = 'grid.jpg?t=' + Date.now();
    }
    update();
    setInterval(update, 50);
  </script>
</body>
</html>"""
            )

        class Handler(http.server.SimpleHTTPRequestHandler):
            def __init__(self, *args, **kwargs):
                super().__init__(directory=out_dir, *args, **kwargs)

            def end_headers(self):
                self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                self.send_header("Expires", "0")
                super().end_headers()

        class UniqueTCPServer(socketserver.TCPServer):
            allow_reuse_address = True

        http_server = UniqueTCPServer(("", 18765), Handler)
        threading.Thread(target=http_server.serve_forever, daemon=True).start()
        print("Camera stream: http://localhost:18765/ (2x2 grid)")

        try:
            import pygame
            pygame.init()
            screen = pygame.display.set_mode((camera_width * 2, camera_height * 2))
            pygame.display.set_caption("Camera Observations (2x2)")
        except Exception:
            screen = None

    while True:
        if viewer is not None and not viewer.is_running():
            break
        if duration_s is not None and time.perf_counter() - start >= duration_s:
            break
        if max_steps is not None and steps >= max_steps:
            break

        loop_start = time.perf_counter()
        action = teleop.get_action()
        _, _, terminated, truncated, info = env.step_teleop_action(action, control_period_s=1.0 / fps)

        if viewer is not None:
            copy_visual_state = getattr(env, "copy_visual_state", None)
            if viewer_data is not None and callable(copy_visual_state):
                copy_visual_state(viewer_data)
            with viewer.lock():
                update_passive_viewer_markers(env._mujoco, viewer, info, marker_style)
            viewer.sync()

        if render_cameras and info.get("camera_obs"):
            import cv2
            import os
            import tempfile

            grid = render_camera_grid(info["camera_obs"], camera_width, camera_height)
            tmp_fd, tmp_path = tempfile.mkstemp(suffix=".jpg", dir="/tmp/camera_stream")
            os.close(tmp_fd)
            cv2.imwrite(tmp_path, cv2.cvtColor(grid, cv2.COLOR_RGB2BGR))
            os.replace(tmp_path, os.path.join("/tmp/camera_stream", "grid.jpg"))

            if screen is not None:
                try:
                    import pygame
                    surf = pygame.surfarray.make_surface(np.transpose(grid, (1, 0, 2)))
                    screen.fill((0, 0, 0))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            screen = None
                except Exception:
                    screen = None

        steps += 1
        if terminated or truncated:
            break

        dt_s = time.perf_counter() - loop_start
        sleep_s = max(1.0 / fps - dt_s, 0.0)
        if sleep_s > 0.0:
            time.sleep(sleep_s)

    if http_server is not None:
        http_server.shutdown()
    if screen is not None:
        import pygame
        pygame.quit()

    info = dict(info)
    info["loop_steps"] = steps
    return info
