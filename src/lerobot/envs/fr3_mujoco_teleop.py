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

import threading
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


@dataclass
class _LatestCameraFrame:
    jpeg_bytes: bytes | None = None

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def set(self, jpeg_bytes: bytes) -> None:
        with self._lock:
            self.jpeg_bytes = jpeg_bytes

    def get(self) -> bytes | None:
        with self._lock:
            return self.jpeg_bytes


@dataclass
class _LatestTeleopInfo:
    info: dict[str, Any]
    loop_steps: int = 0
    terminated: bool = False
    truncated: bool = False

    def __post_init__(self) -> None:
        self._lock = threading.Lock()

    def update(self, *, info: dict[str, Any], loop_steps: int, terminated: bool, truncated: bool) -> None:
        with self._lock:
            self.info = info
            self.loop_steps = loop_steps
            self.terminated = terminated
            self.truncated = truncated

    def snapshot(self) -> tuple[dict[str, Any], int, bool, bool]:
        with self._lock:
            return self.info, self.loop_steps, self.terminated, self.truncated


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


def _corrected_axis_segments_from_pose(pose: np.ndarray, axis_length: float) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    origin = np.asarray(pose[:3, 3], dtype=np.float64)
    rotation = np.asarray(pose[:3, :3], dtype=np.float64)
    theta = -np.pi * 3 / 4
    corr = np.array([[-1, 0, 0], [0, np.cos(theta), -np.sin(theta)], [0, np.sin(theta), np.cos(theta)]], dtype=np.float64)
    corrected = rotation @ corr
    colors = (
        np.array([1.0, 0.2, 0.2, 0.95], dtype=np.float32),
        np.array([0.2, 1.0, 0.2, 0.95], dtype=np.float32),
        np.array([0.2, 0.5, 1.0, 0.95], dtype=np.float32),
    )
    segments = []
    for axis_idx, color in enumerate(colors):
        direction = corrected[:, axis_idx] * axis_length
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
        segments = _corrected_axis_segments_from_pose(pose, style.axis_length)
        for axis_name, (start, end, color) in zip(("x", "y", "z"), segments, strict=True):
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


def _camera_stream_index_html() -> str:
    return """<!DOCTYPE html>
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
    setInterval(update, 33);
  </script>
</body>
</html>"""


def _encode_grid_jpeg(grid: np.ndarray, cv2_module) -> bytes:
    ok, encoded = cv2_module.imencode(".jpg", cv2_module.cvtColor(grid, cv2_module.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Failed to encode camera grid as JPEG.")
    return encoded.tobytes()


def _render_camera_frames(
    *,
    mujoco,
    renderer,
    render_data,
    camera_names: tuple[str, ...],
    camera_name_mapping: dict[str, str],
) -> dict[str, np.ndarray]:
    frames: dict[str, np.ndarray] = {}
    for camera_name in camera_names:
        model_camera_name = camera_name_mapping.get(camera_name, camera_name)
        renderer.update_scene(render_data, camera=model_camera_name)
        frames[camera_name] = np.asarray(renderer.render()).copy()
    return frames


def _start_camera_stream_outputs(
    *,
    camera_width: int,
    camera_height: int,
) -> tuple[Any, _LatestCameraFrame, Any | None]:
    import http.server
    import socketserver

    import cv2

    latest_frame = _LatestCameraFrame()
    blank_grid = np.zeros((camera_height * 2, camera_width * 2, 3), dtype=np.uint8)
    latest_frame.set(_encode_grid_jpeg(blank_grid, cv2))

    try:
        import pygame

        pygame.init()
        screen = pygame.display.set_mode((camera_width * 2, camera_height * 2))
        pygame.display.set_caption("Camera Observations (2x2)")
    except Exception:
        pygame = None
        screen = None

    class Handler(http.server.BaseHTTPRequestHandler):
        def log_message(self, format, *args):
            del format, args

        def do_GET(self):
            if self.path == "/" or self.path.startswith("/index.html"):
                body = _camera_stream_index_html().encode("utf-8")
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.send_header("Expires", "0")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                except BrokenPipeError:
                    pass
                return

            if self.path.startswith("/grid.jpg"):
                payload = latest_frame.get()
                if payload is None:
                    self.send_error(503, "Camera frame not ready")
                    return
                try:
                    self.send_response(200)
                    self.send_header("Content-Type", "image/jpeg")
                    self.send_header("Cache-Control", "no-store, no-cache, must-revalidate")
                    self.send_header("Expires", "0")
                    self.send_header("Content-Length", str(len(payload)))
                    self.end_headers()
                    self.wfile.write(payload)
                except BrokenPipeError:
                    pass
                return

            self.send_error(404)

    class ThreadedTCPServer(socketserver.ThreadingMixIn, socketserver.TCPServer):
        allow_reuse_address = True
        daemon_threads = True

    http_server = ThreadedTCPServer(("", 18765), Handler)
    threading.Thread(target=http_server.serve_forever, daemon=True).start()
    print("Camera stream: http://localhost:18765/ (2x2 grid)")
    return http_server, latest_frame, screen


def _run_control_loop(
    *,
    env,
    teleop: TeleopReader,
    fps: int,
    shared_info: _LatestTeleopInfo,
    stop_event: threading.Event,
    render_cameras: bool,
) -> None:
    loop_steps = 0
    while not stop_event.is_set():
        loop_start = time.perf_counter()
        action = teleop.get_action()
        if render_cameras:
            _, _, terminated, truncated, info = env.step_teleop_action(
                action,
                control_period_s=1.0 / fps,
                include_camera_obs_in_observation=False,
                include_camera_obs_in_info=False,
            )
        else:
            _, _, terminated, truncated, info = env.step_teleop_action(action, control_period_s=1.0 / fps)

        loop_steps += 1
        shared_info.update(
            info=info,
            loop_steps=loop_steps,
            terminated=terminated,
            truncated=truncated,
        )

        if terminated or truncated:
            stop_event.set()
            break

        sleep_s = max(1.0 / fps - (time.perf_counter() - loop_start), 0.0)
        if sleep_s > 0.0:
            stop_event.wait(sleep_s)


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
    camera_fps: float = 30.0,
) -> dict[str, Any]:
    marker_style = marker_style or MarkerStyle()
    start = time.perf_counter()
    if render_cameras:
        _, info = env.reset(
            include_camera_obs_in_observation=False,
            include_camera_obs_in_info=False,
        )
    else:
        _, info = env.reset()
    sync_gripper = getattr(teleop, "sync_gripper_baseline", None)
    if callable(sync_gripper) and "gripper_command" in info:
        sync_gripper(float(info["gripper_command"]))
    shared_info = _LatestTeleopInfo(info=dict(info))
    stop_event = threading.Event()

    if viewer is not None:
        copy_visual_state = getattr(env, "copy_visual_state", None)
        if viewer_data is not None and callable(copy_visual_state):
            copy_visual_state(viewer_data)
        with viewer.lock():
            update_passive_viewer_markers(env._mujoco, viewer, info, marker_style)
        viewer.sync()

    http_server = None
    latest_frame = None
    screen = None
    camera_renderer = None
    camera_render_data = None
    cv2_module = None
    viewer_period_s = 1.0 / 60.0
    camera_period_s = 1.0 / max(float(camera_fps), 1.0)
    next_viewer_sync = time.perf_counter()
    next_camera_render = time.perf_counter()

    if render_cameras:
        import cv2

        cv2_module = cv2
        http_server, latest_frame, screen = _start_camera_stream_outputs(
            camera_width=camera_width,
            camera_height=camera_height,
        )
        camera_renderer = env._mujoco.Renderer(
            env.model,
            height=camera_height,
            width=camera_width,
        )
        camera_render_data = env._mujoco.MjData(env.model)

    control_thread = threading.Thread(
        target=_run_control_loop,
        name="fr3-teleop-control",
        daemon=True,
        kwargs={
            "env": env,
            "teleop": teleop,
            "fps": fps,
            "shared_info": shared_info,
            "stop_event": stop_event,
            "render_cameras": render_cameras,
        },
    )
    control_thread.start()

    while True:
        info, loop_steps, terminated, truncated = shared_info.snapshot()
        if viewer is not None and not viewer.is_running():
            stop_event.set()
            break
        if duration_s is not None and time.perf_counter() - start >= duration_s:
            stop_event.set()
            break
        if max_steps is not None and loop_steps >= max_steps:
            stop_event.set()
            break
        if terminated or truncated:
            break

        now = time.perf_counter()
        if viewer is not None and now >= next_viewer_sync:
            copy_visual_state = getattr(env, "copy_visual_state", None)
            if viewer_data is not None and callable(copy_visual_state):
                copy_visual_state(viewer_data)
            with viewer.lock():
                update_passive_viewer_markers(env._mujoco, viewer, info, marker_style)
            viewer.sync()
            next_viewer_sync = now + viewer_period_s

        if render_cameras and camera_renderer is not None and camera_render_data is not None and latest_frame is not None and now >= next_camera_render:
            env.copy_visual_state(camera_render_data)
            camera_obs = _render_camera_frames(
                mujoco=env._mujoco,
                renderer=camera_renderer,
                render_data=camera_render_data,
                camera_names=tuple(env.cfg.camera_names),
                camera_name_mapping=dict(env.cfg.camera_name_mapping),
            )
            grid = render_camera_grid(camera_obs, camera_width, camera_height)
            latest_frame.set(_encode_grid_jpeg(grid, cv2_module))
            if screen is not None:
                try:
                    import pygame

                    surf = pygame.surfarray.make_surface(np.transpose(grid, (1, 0, 2)))
                    screen.fill((0, 0, 0))
                    screen.blit(surf, (0, 0))
                    pygame.display.flip()
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            stop_event.set()
                except Exception:
                    screen = None
            next_camera_render = now + camera_period_s

        sleep_s = 0.001
        if viewer is not None or render_cameras:
            next_deadline = now + 1.0
            if viewer is not None:
                next_deadline = min(next_deadline, next_viewer_sync)
            if render_cameras:
                next_deadline = min(next_deadline, next_camera_render)
            sleep_s = max(min(next_deadline - time.perf_counter(), 0.01), 0.001)
        time.sleep(sleep_s)

    stop_event.set()
    control_thread.join(timeout=1.0)
    info, loop_steps, _, _ = shared_info.snapshot()

    if viewer is not None and viewer.is_running():
        copy_visual_state = getattr(env, "copy_visual_state", None)
        if viewer_data is not None and callable(copy_visual_state):
            copy_visual_state(viewer_data)
        with viewer.lock():
            update_passive_viewer_markers(env._mujoco, viewer, info, marker_style)
        viewer.sync()

    if render_cameras and camera_renderer is not None and camera_render_data is not None and latest_frame is not None:
        env.copy_visual_state(camera_render_data)
        camera_obs = _render_camera_frames(
            mujoco=env._mujoco,
            renderer=camera_renderer,
            render_data=camera_render_data,
            camera_names=tuple(env.cfg.camera_names),
            camera_name_mapping=dict(env.cfg.camera_name_mapping),
        )
        grid = render_camera_grid(camera_obs, camera_width, camera_height)
        latest_frame.set(_encode_grid_jpeg(grid, cv2_module))
        if screen is not None:
            try:
                import pygame

                surf = pygame.surfarray.make_surface(np.transpose(grid, (1, 0, 2)))
                screen.fill((0, 0, 0))
                screen.blit(surf, (0, 0))
                pygame.display.flip()
            except Exception:
                screen = None

    if http_server is not None:
        http_server.shutdown()
        server_close = getattr(http_server, "server_close", None)
        if callable(server_close):
            server_close()
    if camera_renderer is not None:
        camera_renderer.close()
    if screen is not None:
        import pygame
        pygame.quit()

    info = dict(info)
    info["loop_steps"] = loop_steps
    return info
