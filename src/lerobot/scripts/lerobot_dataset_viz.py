#!/usr/bin/env python

# Copyright 2024 The HuggingFace Inc. team. All rights reserved.
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
""" Visualize data of **all** frames of any episode of a dataset of type LeRobotDataset.

Note: The last frame of the episode doesn't always correspond to a final state.
That's because our datasets are composed of transition from state to state up to
the antepenultimate state associated to the ultimate action to arrive in the final state.
However, there might not be a transition from a final state to another state.

Note: This script aims to visualize the data used to train the neural networks.
~What you see is what you get~. When visualizing image modality, it is often expected to observe
lossy compression artifacts since these images have been decoded from compressed mp4 videos to
save disk space. The compression factor applied has been tuned to not affect success rate.

Examples:

- Visualize data stored on a local machine:
```
local$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0
```

- Visualize data stored on a distant machine with a local viewer:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --save 1 \
    --output-dir path/to/directory

local$ scp distant:path/to/directory/lerobot_pusht_episode_0.rrd .
local$ rerun lerobot_pusht_episode_0.rrd
```

- Visualize data stored on a distant machine through streaming:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --grpc-port 9876

local$ rerun rerun+http://IP:GRPC_PORT/proxy
```

- Visualize data on a distant machine and switch episodes from a local terminal:
```
distant$ lerobot-dataset-viz \
    --repo-id lerobot/pusht \
    --episode-index 0 \
    --mode distant \
    --grpc-port 9876 \
    --control-port 9999

local$ rerun rerun+http://IP:9876/proxy
local$ dataset_viz_client.py --host IP --control-port 9999
```

"""

import argparse
import contextlib
import gc
import logging
import multiprocessing as mp
import os
import queue
import select
import socket
import socketserver
import sys
import termios
import threading
import time
from datetime import datetime, timedelta
from pathlib import Path
import tty
from urllib.parse import quote
from uuid import uuid4

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset, LeRobotDatasetMetadata
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD
from lerobot.utils.rotation import Rotation
from lerobot.utils.state_feature_names import flatten_feature_name_paths, get_ee_pose_state_indices
from lerobot.utils.utils import init_logging


EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
EE_QUAT_KEYS = ("ee.qx", "ee.qy", "ee.qz", "ee.qw")
EE_RULER_AXIS_COLORS = {
    "x": [255, 0, 0, 255],
    "y": [0, 255, 0, 255],
    "z": [0, 0, 255, 255],
}

QUIT_COMMANDS = {"q", "\x03", "\x04"}
MANUALLY_LOGGED_KEYS = {ACTION, OBS_STATE, DONE, REWARD, "next.success"}
AUTO_VIZ_EXCLUDED_KEYS = {"episode_index", "frame_index", "index", "task_index", "timestamp"}


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def tensor_to_rerun_image_numpy(value: torch.Tensor) -> np.ndarray | None:
    if value.ndim == 2:
        image = value.detach().cpu().numpy()
        if image.dtype == np.bool_:
            return image.astype(np.uint8) * 255
        return image

    if value.ndim == 3 and value.shape[0] in (1, 3, 4) and value.shape[-1] not in (1, 3, 4):
        if value.dtype == torch.bool:
            image = value.detach().cpu().to(torch.uint8).mul(255).permute(1, 2, 0).numpy()
        elif value.dtype == torch.float32 and value.min().item() >= 0.0 and value.max().item() <= 1.0:
            image = to_hwc_uint8_numpy(value)
        else:
            image = value.detach().cpu().permute(1, 2, 0).numpy()
        if image.ndim == 3 and image.shape[-1] == 1:
            return image[..., 0]
        return image

    return None


def build_scalar_entity_paths(
    root: str,
    feature_names: list[str] | tuple[str, ...] | dict[str, list[str] | dict] | None,
    width: int,
) -> list[str]:
    flattened_names = flatten_feature_name_paths(feature_names)
    if flattened_names is None or len(flattened_names) != width:
        return [f"{root}/{dim_idx}" for dim_idx in range(width)]
    return [f"{root}/{name}" for name in flattened_names]


def build_device_capture_timestamp_entity_paths(
    feature_names: list[str] | tuple[str, ...] | dict[str, list[str] | dict] | None,
    width: int,
) -> list[str]:
    root = "observation/device_capture_timestamp"
    flattened_names = flatten_feature_name_paths(feature_names)
    if flattened_names is None or len(flattened_names) != width:
        return [f"{root}/{dim_idx}" for dim_idx in range(width)]
    entity_paths = []
    for name in flattened_names:
        normalized_name = name.replace(".", "/")
        if normalized_name.endswith("/capture_timestamp_s"):
            normalized_name = normalized_name.removesuffix("/capture_timestamp_s")
        entity_paths.append(f"{root}/{normalized_name}")
    return entity_paths


def get_auto_visualization_keys(batch: dict, camera_keys: list[str] | tuple[str, ...]) -> list[str]:
    excluded_keys = set(camera_keys) | MANUALLY_LOGGED_KEYS | AUTO_VIZ_EXCLUDED_KEYS
    return [
        key
        for key in batch
        if key not in excluded_keys and not key.endswith("_is_pad") and not isinstance(batch[key], str | bytes)
    ]


def log_feature_value(
    key: str,
    value: torch.Tensor,
    feature_names: list[str] | tuple[str, ...] | dict[str, list[str] | dict] | None,
    display_compressed_images: bool,
) -> None:
    if not isinstance(value, torch.Tensor):
        return

    if key == "observation.device_capture_timestamp":
        value = as_1d_tensor(value)

    if value.ndim == 0:
        rr.log(key, rr.Scalars(value.item()))
        return

    if value.ndim == 1:
        if key == "observation.device_capture_timestamp":
            entity_paths = build_device_capture_timestamp_entity_paths(feature_names, len(value))
        else:
            entity_paths = build_scalar_entity_paths(key, feature_names, len(value))
        for entity_path, val in zip(entity_paths, value, strict=True):
            rr.log(entity_path, rr.Scalars(val.item()))
        return

    image = tensor_to_rerun_image_numpy(value)
    if image is not None:
        entity = rr.Image(image)
        if display_compressed_images and image.dtype == np.uint8:
            entity = entity.compress()
        rr.log(key, entity=entity)
        return

    rr.log(key, rr.Tensor(value.detach().cpu().numpy()))


def as_1d_tensor(value: torch.Tensor) -> torch.Tensor:
    if value.ndim == 0:
        return value.reshape(1)
    return value


def has_ee_pose(batch: dict, ee_pose_state_indices: dict[str, int] | None = None) -> bool:
    return all(key in batch for key in (*EE_POSITION_KEYS, *EE_QUAT_KEYS)) or (
        ee_pose_state_indices is not None and OBS_STATE in batch
    )


def _to_float(value) -> float:
    if hasattr(value, "item"):
        return float(value.item())
    return float(value)


def make_system_time_anchor(first_relative_timestamp_s: float, now: datetime | None = None) -> datetime:
    if now is None:
        now = datetime.now().astimezone()
    return now - timedelta(seconds=float(first_relative_timestamp_s))


def to_system_timestamp(anchor: datetime, relative_timestamp_s: float) -> datetime:
    return anchor + timedelta(seconds=float(relative_timestamp_s))


def extract_ee_pose(
    batch: dict,
    index: int,
    ee_pose_state_indices: dict[str, int] | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    if all(key in batch for key in (*EE_POSITION_KEYS, *EE_QUAT_KEYS)):
        position = np.array([_to_float(batch[key][index]) for key in EE_POSITION_KEYS], dtype=np.float32)
        quaternion = np.array([_to_float(batch[key][index]) for key in EE_QUAT_KEYS], dtype=np.float32)
        rotation = Rotation.from_quat(quaternion).as_matrix().astype(np.float32)
        return position, rotation
    elif ee_pose_state_indices is not None and OBS_STATE in batch:
        state = batch[OBS_STATE][index]
        position = np.array([_to_float(state[ee_pose_state_indices[key]]) for key in EE_POSITION_KEYS], dtype=np.float32)
        quaternion = np.array([_to_float(state[ee_pose_state_indices[key]]) for key in EE_QUAT_KEYS], dtype=np.float32)
        rotation = Rotation.from_quat(quaternion).as_matrix().astype(np.float32)
        return position, rotation
    else:
        raise KeyError("EE pose is unavailable in both top-level batch keys and observation.state.")


def log_ee_pose_3d(
    batch: dict,
    index: int,
    trajectory_positions: list[np.ndarray],
    axis_length: float,
    ee_pose_state_indices: dict[str, int] | None = None,
) -> None:
    position, rotation = extract_ee_pose(batch, index, ee_pose_state_indices=ee_pose_state_indices)
    trajectory_positions.append(position.copy())

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/ee_pose/trajectory",
        rr.LineStrips3D(
            [np.asarray(trajectory_positions, dtype=np.float32)],
            colors=[[255, 176, 0, 255]],
            radii=[0.002],
        ),
    )
    rr.log(
        "world/ee_pose/position",
        rr.Points3D([position], colors=[[255, 176, 0, 255]], radii=[0.008]),
    )
    rr.log(
        "world/ee_pose/frame",
        rr.Transform3D(translation=position, mat3x3=rotation, axis_length=axis_length),
    )


def build_ee_axis_ruler_strips(
    axis: str,
    length: float,
    *,
    minor_tick_step: float = 0.01,
    major_tick_step: float = 0.05,
    minor_tick_height: float = 0.005,
    major_tick_height: float = 0.01,
) -> list[np.ndarray]:
    if length <= 0:
        return []
    if axis not in {"x", "y", "z"}:
        raise ValueError(f"Unsupported ruler axis: {axis}")

    axis_index = {"x": 0, "y": 1, "z": 2}[axis]
    tick_axis_index = {"x": 1, "y": 2, "z": 0}[axis]

    ruler_end = np.zeros(3, dtype=np.float32)
    ruler_end[axis_index] = length
    strips = [
        np.array(
            [
                np.zeros(3, dtype=np.float32),
                ruler_end,
            ],
            dtype=np.float32,
        )
    ]

    tick_positions = np.arange(0.0, length + minor_tick_step * 0.5, minor_tick_step, dtype=np.float32)
    for tick_position in tick_positions:
        is_major_tick = np.isclose(np.mod(tick_position, major_tick_step), 0.0, atol=1e-6) or np.isclose(
            np.mod(tick_position, major_tick_step), major_tick_step, atol=1e-6
        )
        tick_height = major_tick_height if is_major_tick else minor_tick_height
        tick_start = np.zeros(3, dtype=np.float32)
        tick_end = np.zeros(3, dtype=np.float32)
        tick_start[axis_index] = tick_position
        tick_end[axis_index] = tick_position
        tick_end[tick_axis_index] = tick_height
        strips.append(
            np.array(
                [
                    tick_start,
                    tick_end,
                ],
                dtype=np.float32,
            )
        )

    return strips


def log_ee_ruler(length: float) -> None:
    strips = []
    colors = []
    for axis, color in EE_RULER_AXIS_COLORS.items():
        axis_strips = build_ee_axis_ruler_strips(axis, length)
        strips.extend(axis_strips)
        colors.extend([color] * len(axis_strips))

    if not strips:
        return

    rr.log("world", rr.ViewCoordinates.RIGHT_HAND_Z_UP, static=True)
    rr.log(
        "world/ruler",
        rr.LineStrips3D(
            strips,
            colors=colors,
            radii=[0.0015] * len(strips),
        ),
        static=True,
    )


def should_enable_episode_switch(mode: str, save: bool) -> bool:
    return mode == "distant" and not save


def format_host_for_url(host: str) -> str:
    if ":" in host and not host.startswith("["):
        return f"[{host}]"
    return host


def build_public_rerun_connect_url(
    *,
    grpc_port: int,
    server_uri: str,
    public_host: str | None = None,
) -> str:
    if public_host is None:
        return server_uri
    return f"rerun+http://{format_host_for_url(public_host)}:{grpc_port}/proxy"


def build_public_rerun_web_viewer_url(
    *,
    web_port: int,
    connect_url: str,
    public_host: str,
) -> str:
    return f"http://{format_host_for_url(public_host)}:{web_port}?url={quote(connect_url, safe='')}"


def build_episode_switch_visualize_kwargs(cli_kwargs: dict) -> dict:
    visualize_kwargs = cli_kwargs.copy()
    visualize_kwargs.pop("episode_index")
    return visualize_kwargs


def build_episode_process_visualize_kwargs(visualize_kwargs: dict, *, rerun_recording_id: str) -> dict:
    visualize_kwargs = visualize_kwargs.copy()
    visualize_kwargs["rerun_recording_id"] = rerun_recording_id
    return visualize_kwargs


def normalize_control_command(raw_command: str) -> str | None:
    normalized = raw_command.strip().lower()
    if not normalized:
        return None
    first = normalized[0]
    if first == "n":
        return "n"
    if first == "q":
        return "q"
    return None


def enqueue_control_command(command_queue: queue.Queue[str], raw_command: str) -> str | None:
    command = normalize_control_command(raw_command)
    if command is not None:
        command_queue.put(command)
    return command


def get_next_episode_index(current_episode_index: int, total_episodes: int) -> int | None:
    next_episode_index = current_episode_index + 1
    if next_episode_index >= total_episodes:
        return None
    return next_episode_index


@contextlib.contextmanager
def raw_terminal_mode():
    if not sys.stdin.isatty():
        yield
        return

    fd = sys.stdin.fileno()
    old_attrs = termios.tcgetattr(fd)
    try:
        tty.setraw(fd)
        yield
    finally:
        termios.tcsetattr(fd, termios.TCSADRAIN, old_attrs)


def read_episode_switch_command() -> str:
    if sys.stdin.isatty():
        with raw_terminal_mode():
            while True:
                ready, _, _ = select.select([sys.stdin], [], [], None)
                if ready:
                    return sys.stdin.read(1).lower()

    line = sys.stdin.readline()
    if line == "":
        raise EOFError
    stripped = line.strip().lower()
    return stripped[:1] if stripped else ""


class EpisodeControlTCPServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, server_address, request_handler_class, command_queue: queue.Queue[str]):
        self.command_queue = command_queue
        super().__init__(server_address, request_handler_class)


class EpisodeControlTCPHandler(socketserver.StreamRequestHandler):
    def handle(self) -> None:
        server: EpisodeControlTCPServer = self.server  # type: ignore[assignment]
        client = f"{self.client_address[0]}:{self.client_address[1]}"
        logging.info("Dataset viz control client connected: %s", client)
        try:
            while True:
                line = self.rfile.readline()
                if not line:
                    break
                enqueue_control_command(server.command_queue, line.decode("utf-8", errors="ignore"))
        finally:
            logging.info("Dataset viz control client disconnected: %s", client)


def start_control_server(host: str, port: int) -> tuple[callable, callable, int]:
    command_queue: queue.Queue[str] = queue.Queue()
    server = EpisodeControlTCPServer((host, port), EpisodeControlTCPHandler, command_queue)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()

    def read_command() -> str:
        return command_queue.get()

    def shutdown() -> None:
        server.shutdown()
        server.server_close()
        thread.join(timeout=5.0)

    return read_command, shutdown, server.server_address[1]


def terminate_episode_process(process: mp.Process | None, timeout_s: float = 5.0) -> None:
    if process is None:
        return
    if not process.is_alive():
        process.join(timeout=timeout_s)
        return

    process.terminate()
    process.join(timeout=timeout_s)
    if process.is_alive():
        process.kill()
        process.join(timeout=timeout_s)


def is_tcp_port_available(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def wait_for_tcp_port_available(
    host: str,
    port: int,
    *,
    timeout_s: float = 5.0,
    poll_interval_s: float = 0.05,
    label: str | None = None,
) -> None:
    deadline = time.perf_counter() + timeout_s
    while time.perf_counter() < deadline:
        if is_tcp_port_available(host, port):
            return
        time.sleep(poll_interval_s)

    port_label = label or f"{host}:{port}"
    raise RuntimeError(
        f"Port {port_label} is still in use. Stop the previous dataset_viz/rerun process or choose another port."
    )


def create_episode_process(
    ctx: mp.context.BaseContext,
    *,
    repo_id: str,
    root: Path | None,
    episode_index: int,
    tolerance_s: float,
    visualize_kwargs: dict,
) -> mp.Process:
    return ctx.Process(
        target=serve_dataset_episode,
        kwargs={
            "repo_id": repo_id,
            "root": root,
            "episode_index": episode_index,
            "tolerance_s": tolerance_s,
            "visualize_kwargs": visualize_kwargs,
        },
        daemon=False,
    )


def run_episode_switch_loop(
    *,
    start_episode_index: int,
    total_episodes: int,
    launch_episode,
    terminate_episode,
    read_command,
) -> None:
    current_episode_index = start_episode_index
    process = launch_episode(current_episode_index)
    try:
        while True:
            command = read_command()
            if not command:
                continue
            if command in QUIT_COMMANDS:
                break
            if command != "n":
                continue

            next_episode_index = get_next_episode_index(current_episode_index, total_episodes)
            if next_episode_index is None:
                logging.info(
                    "Episode %d is the last available episode (%d total). Ignoring 'n'.",
                    current_episode_index,
                    total_episodes,
                )
                continue

            logging.info("Switching from episode %d to episode %d.", current_episode_index, next_episode_index)
            terminate_episode(process)
            process = launch_episode(next_episode_index)
            current_episode_index = next_episode_index
    finally:
        terminate_episode(process)


def serve_dataset_episode(
    *,
    repo_id: str,
    root: Path | None,
    episode_index: int,
    tolerance_s: float,
    visualize_kwargs: dict,
) -> None:
    init_logging()
    logging.info("Loading dataset for episode %d", episode_index)
    dataset = LeRobotDataset(repo_id, episodes=[episode_index], root=root, tolerance_s=tolerance_s)
    try:
        visualize_dataset(dataset, episode_index=episode_index, **visualize_kwargs)
    finally:
        del dataset
        gc.collect()
        rr.disconnect()


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    grpc_port: int = 9876,
    public_host: str | None = None,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
    ee_axis_length: float = 0.05,
    ee_ruler_length: float = 0.1,
    rerun_recording_id: str | None = None,
    **kwargs,
) -> Path | None:
    if save:
        assert output_dir is not None, (
            "Set an output directory where to write .rrd files with `--output-dir path/to/directory`."
        )

    repo_id = dataset.repo_id

    logging.info("Loading dataloader")
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=num_workers,
        batch_size=batch_size,
    )

    logging.info("Starting Rerun")

    if mode not in ["local", "distant"]:
        raise ValueError(mode)

    spawn_local_viewer = mode == "local" and not save
    rr.init(f"{repo_id}/episode_{episode_index}", recording_id=rerun_recording_id, spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        logging.info(f"Connect to a Rerun Server: rerun rerun+http://IP:{grpc_port}/proxy")
        viewer_connect_uri = build_public_rerun_connect_url(
            grpc_port=grpc_port,
            server_uri=server_uri,
            public_host=public_host,
        )
        if viewer_connect_uri != server_uri:
            logging.info("Using public rerun connect URL: %s", viewer_connect_uri)
        rr.serve_web_viewer(open_browser=False, web_port=web_port)
        if public_host is not None:
            logging.info(
                "Open remote web viewer: %s",
                build_public_rerun_web_viewer_url(
                    web_port=web_port,
                    connect_url=viewer_connect_uri,
                    public_host=public_host,
                ),
            )

    logging.info("Logging to Rerun")
    logging.info("World frame convention: +X forward, +Y left, +Z up (URDF base/world frame).")
    log_ee_ruler(ee_ruler_length)

    first_index = None
    system_time_anchor = None
    ee_trajectory_positions: list[np.ndarray] = []
    state_feature_names = dataset.meta.features.get(OBS_STATE, {}).get("names")
    action_feature_names = dataset.meta.features.get(ACTION, {}).get("names")
    ee_pose_state_indices = get_ee_pose_state_indices(state_feature_names)
    state_name_paths = flatten_feature_name_paths(state_feature_names)
    action_name_paths = flatten_feature_name_paths(action_feature_names)
    action_root = ACTION
    if state_name_paths is not None and action_name_paths == state_name_paths:
        action_root = "action_target"
        logging.info(
            "Action and observation.state share identical feature names; logging actions under '%s/'.",
            action_root,
        )
    action_entity_paths = None
    state_entity_paths = None
    for batch in tqdm.tqdm(dataloader, total=len(dataloader)):
        if first_index is None:
            first_index = batch["index"][0].item()
        if system_time_anchor is None:
            system_time_anchor = make_system_time_anchor(batch["timestamp"][0].item())
            logging.info("Rerun timestamp timeline anchored to system time at %s.", system_time_anchor.isoformat())
        # iterate over the batch
        for i in range(len(batch["index"])):
            relative_timestamp_s = batch["timestamp"][i].item()
            rr.set_time("frame_index", sequence=batch["index"][i].item() - first_index)
            rr.set_time("episode_time", duration=relative_timestamp_s)
            rr.set_time("timestamp", timestamp=to_system_timestamp(system_time_anchor, relative_timestamp_s))

            # display each camera image
            for key in dataset.meta.camera_keys:
                img = to_hwc_uint8_numpy(batch[key][i])
                img_entity = rr.Image(img).compress() if display_compressed_images else rr.Image(img)
                rr.log(key, entity=img_entity)

            # display each dimension of action space (e.g. actuators command)
            if ACTION in batch:
                action_value = as_1d_tensor(batch[ACTION][i])
                if action_entity_paths is None:
                    action_entity_paths = build_scalar_entity_paths(action_root, action_feature_names, len(action_value))
                for entity_path, val in zip(action_entity_paths, action_value, strict=True):
                    rr.log(entity_path, rr.Scalars(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if OBS_STATE in batch:
                state_value = as_1d_tensor(batch[OBS_STATE][i])
                if state_entity_paths is None:
                    state_entity_paths = build_scalar_entity_paths("state", state_feature_names, len(state_value))
                for entity_path, val in zip(state_entity_paths, state_value, strict=True):
                    rr.log(entity_path, rr.Scalars(val.item()))

            if DONE in batch:
                rr.log(DONE, rr.Scalars(batch[DONE][i].item()))

            if REWARD in batch:
                rr.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

            for key in get_auto_visualization_keys(batch, dataset.meta.camera_keys):
                feature_names = dataset.meta.features.get(key, {}).get("names")
                log_feature_value(key, batch[key][i], feature_names, display_compressed_images)

            if has_ee_pose(batch, ee_pose_state_indices=ee_pose_state_indices):
                log_ee_pose_3d(
                    batch,
                    i,
                    ee_trajectory_positions,
                    axis_length=ee_axis_length,
                    ee_pose_state_indices=ee_pose_state_indices,
                )

    if mode == "local" and save:
        # save .rrd locally
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        repo_id_str = repo_id.replace("/", "_")
        rrd_path = output_dir / f"{repo_id_str}_episode_{episode_index}.rrd"
        rr.save(rrd_path)
        return rrd_path

    elif mode == "distant":
        # stop the process from exiting since it is serving the websocket connection
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            print("Ctrl-C received. Exiting.")


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--repo-id",
        type=str,
        required=True,
        help="Name of hugging face repository containing a LeRobotDataset dataset (e.g. `lerobot/pusht`).",
    )
    parser.add_argument(
        "--episode-index",
        type=int,
        required=True,
        help=(
            "Episode to visualize initially. In `--mode distant` without `--save`, "
            "press `n`/`q` in the remote terminal by default, or use `dataset_viz_client.py` "
            "from a local terminal when `--control-port` is set."
        ),
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=None,
        help="Root directory for the dataset stored locally (e.g. `--root data`). By default, the dataset will be loaded from hugging face cache folder, or downloaded from the hub if available.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory path to write a .rrd file when `--save 1` is set.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size loaded by DataLoader.",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of processes of Dataloader for loading the data.",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="local",
        help=(
            "Mode of viewing between 'local' or 'distant'. "
            "'local' requires data to be on a local machine. It spawns a viewer to visualize the data locally. "
            "'distant' creates a server on the distant machine where the data is stored. "
            "Visualize the data by connecting to the server with `rerun rerun+http://IP:GRPC_PORT/proxy` on the local machine. "
            "In distant mode without `--save`, switch episodes either from the remote terminal with `n`/`q`, "
            "or from a local terminal through `dataset_viz_client.py` when `--control-port` is set."
        ),
    )
    parser.add_argument(
        "--web-port",
        type=int,
        default=9090,
        help="Web port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--ws-port",
        type=int,
        help="deprecated, please use --grpc-port instead.",
    )
    parser.add_argument(
        "--grpc-port",
        type=int,
        default=9876,
        help="gRPC port for rerun.io when `--mode distant` is set.",
    )
    parser.add_argument(
        "--public-host",
        type=str,
        default=None,
        help=(
            "Optional public hostname or IP advertised to the remote web viewer. "
            "Use this in `--mode distant` when the browser cannot reach the host through 127.0.0.1. "
            "The script will print a full browser URL with `?url=...` for the remote web viewer. "
            "Example: `--public-host 192.168.1.200`."
        ),
    )
    parser.add_argument(
        "--control-host",
        type=str,
        default="0.0.0.0",
        help=(
            "Host interface for the optional TCP control server. "
            "Use together with `--control-port`, then connect from a local terminal with `dataset_viz_client.py`."
        ),
    )
    parser.add_argument(
        "--control-port",
        type=int,
        default=None,
        help=(
            "Optional TCP control port for remote episode switching. "
            "When set in `--mode distant`, the script listens for `n`/`q` commands from `dataset_viz_client.py` "
            "instead of requiring the remote terminal."
        ),
    )
    parser.add_argument(
        "--save",
        type=int,
        default=0,
        help=(
            "Save a .rrd file in the directory provided by `--output-dir`. "
            "It also deactivates the spawning of a viewer. "
            "Visualize the data by running `rerun path/to/file.rrd` on your local machine."
        ),
    )

    parser.add_argument(
        "--tolerance-s",
        type=float,
        default=1e-4,
        help=(
            "Tolerance in seconds used to ensure data timestamps respect the dataset fps value"
            "This is argument passed to the constructor of LeRobotDataset and maps to its tolerance_s constructor argument"
            "If not given, defaults to 1e-4."
        ),
    )

    parser.add_argument(
        "--display-compressed-images",
        action="store_true",
        help="If set, display compressed images in Rerun instead of uncompressed ones.",
    )
    parser.add_argument(
        "--ee-axis-length",
        type=float,
        default=0.05,
        help="Axis length in meters for the 3D end-effector pose visualization.",
    )
    parser.add_argument(
        "--ee-ruler-length",
        type=float,
        default=0.1,
        help="Length in meters of the static 3D ruler drawn in the world frame. Set to 0 to disable it.",
    )

    args = parser.parse_args()
    kwargs = vars(args)
    repo_id = kwargs.pop("repo_id")
    root = kwargs.pop("root")
    tolerance_s = kwargs.pop("tolerance_s")

    if kwargs["ws_port"] is not None:
        logging.warning(
            "--ws-port is deprecated and will be removed in future versions. Please use --grpc-port instead."
        )
        logging.warning("Setting grpc_port to ws_port value.")
        kwargs["grpc_port"] = kwargs.pop("ws_port")

    init_logging()
    if should_enable_episode_switch(args.mode, bool(args.save)):
        meta = LeRobotDatasetMetadata(repo_id, root=root)
        total_episodes = meta.total_episodes
        del meta
        gc.collect()

        if args.episode_index >= total_episodes:
            raise IndexError(f"Episode index {args.episode_index} out of range for dataset with {total_episodes} episodes.")

        if args.control_port is not None:
            wait_for_tcp_port_available(
                args.control_host,
                args.control_port,
                label=f"control server {args.control_host}:{args.control_port}",
            )

        visualize_kwargs = build_episode_switch_visualize_kwargs(kwargs)

        ctx = mp.get_context("spawn")

        def launch_episode(episode_index: int) -> mp.Process:
            episode_visualize_kwargs = build_episode_process_visualize_kwargs(
                visualize_kwargs,
                rerun_recording_id=str(uuid4()),
            )
            wait_for_tcp_port_available("0.0.0.0", visualize_kwargs["grpc_port"], label=f"gRPC {visualize_kwargs['grpc_port']}")
            wait_for_tcp_port_available("0.0.0.0", visualize_kwargs["web_port"], label=f"web viewer {visualize_kwargs['web_port']}")
            process = create_episode_process(
                ctx,
                repo_id=repo_id,
                root=root,
                episode_index=episode_index,
                tolerance_s=tolerance_s,
                visualize_kwargs=episode_visualize_kwargs,
            )
            process.start()
            return process

        if args.control_port is not None:
            read_command, shutdown_control, control_port = start_control_server(args.control_host, args.control_port)
            logging.info(
                "Interactive episode switching enabled through TCP control server on %s:%d.",
                args.control_host,
                control_port,
            )
            logging.info("Use dataset_viz_client.py locally and send `n` / `q` commands over the control port.")
        else:
            logging.info(
                "Interactive episode switching enabled for distant mode. Press 'n' for next episode, 'q' to quit."
            )
            read_command = read_episode_switch_command
            shutdown_control = lambda: None

        try:
            run_episode_switch_loop(
                start_episode_index=args.episode_index,
                total_episodes=total_episodes,
                launch_episode=launch_episode,
                terminate_episode=terminate_episode_process,
                read_command=read_command,
            )
        finally:
            shutdown_control()
        return

    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, episodes=[args.episode_index], root=root, tolerance_s=tolerance_s)
    try:
        visualize_dataset(dataset, **vars(args))
    finally:
        del dataset
        gc.collect()
        rr.disconnect()


if __name__ == "__main__":
    main()
