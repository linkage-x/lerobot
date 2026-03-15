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

"""

import argparse
import gc
import logging
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import rerun as rr
import torch
import torch.utils.data
import tqdm

from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.utils.rotation import Rotation
from lerobot.utils.constants import ACTION, DONE, OBS_STATE, REWARD
from lerobot.utils.utils import init_logging


EE_POSITION_KEYS = ("ee.x", "ee.y", "ee.z")
EE_QUAT_KEYS = ("ee.qx", "ee.qy", "ee.qz", "ee.qw")
EE_RULER_AXIS_COLORS = {
    "x": [255, 0, 0, 255],
    "y": [0, 255, 0, 255],
    "z": [0, 0, 255, 255],
}


def to_hwc_uint8_numpy(chw_float32_torch: torch.Tensor) -> np.ndarray:
    assert chw_float32_torch.dtype == torch.float32
    assert chw_float32_torch.ndim == 3
    c, h, w = chw_float32_torch.shape
    assert c < h and c < w, f"expect channel first images, but instead {chw_float32_torch.shape}"
    hwc_uint8_numpy = (chw_float32_torch * 255).type(torch.uint8).permute(1, 2, 0).numpy()
    return hwc_uint8_numpy


def get_ee_pose_state_indices(state_feature_names: list[str] | tuple[str, ...] | None) -> dict[str, int] | None:
    if state_feature_names is None:
        return None
    state_name_to_index = {name: idx for idx, name in enumerate(state_feature_names)}
    required_keys = (*EE_POSITION_KEYS, *EE_QUAT_KEYS)
    if not all(key in state_name_to_index for key in required_keys):
        return None
    return {key: state_name_to_index[key] for key in required_keys}


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


def visualize_dataset(
    dataset: LeRobotDataset,
    episode_index: int,
    batch_size: int = 32,
    num_workers: int = 0,
    mode: str = "local",
    web_port: int = 9090,
    grpc_port: int = 9876,
    save: bool = False,
    output_dir: Path | None = None,
    display_compressed_images: bool = False,
    ee_axis_length: float = 0.05,
    ee_ruler_length: float = 0.1,
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
    rr.init(f"{repo_id}/episode_{episode_index}", spawn=spawn_local_viewer)

    # Manually call python garbage collector after `rr.init` to avoid hanging in a blocking flush
    # when iterating on a dataloader with `num_workers` > 0
    # TODO(rcadene): remove `gc.collect` when rerun version 0.16 is out, which includes a fix
    gc.collect()

    if mode == "distant":
        server_uri = rr.serve_grpc(grpc_port=grpc_port)
        logging.info(f"Connect to a Rerun Server: rerun rerun+http://IP:{grpc_port}/proxy")
        rr.serve_web_viewer(open_browser=False, web_port=web_port, connect_to=server_uri)

    logging.info("Logging to Rerun")
    logging.info("World frame convention: +X forward, +Y left, +Z up (URDF base/world frame).")
    log_ee_ruler(ee_ruler_length)

    first_index = None
    system_time_anchor = None
    ee_trajectory_positions: list[np.ndarray] = []
    ee_pose_state_indices = get_ee_pose_state_indices(dataset.meta.features.get(OBS_STATE, {}).get("names"))
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
                for dim_idx, val in enumerate(batch[ACTION][i]):
                    rr.log(f"{ACTION}/{dim_idx}", rr.Scalars(val.item()))

            # display each dimension of observed state space (e.g. agent position in joint space)
            if OBS_STATE in batch:
                for dim_idx, val in enumerate(batch[OBS_STATE][i]):
                    rr.log(f"state/{dim_idx}", rr.Scalars(val.item()))

            if DONE in batch:
                rr.log(DONE, rr.Scalars(batch[DONE][i].item()))

            if REWARD in batch:
                rr.log(REWARD, rr.Scalars(batch[REWARD][i].item()))

            if "next.success" in batch:
                rr.log("next.success", rr.Scalars(batch["next.success"][i].item()))

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
        help="Episode to visualize.",
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
            "Visualize the data by connecting to the server with `rerun rerun+http://IP:GRPC_PORT/proxy` on the local machine."
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
    logging.info("Loading dataset")
    dataset = LeRobotDataset(repo_id, episodes=[args.episode_index], root=root, tolerance_s=tolerance_s)

    visualize_dataset(dataset, **vars(args))


if __name__ == "__main__":
    main()
