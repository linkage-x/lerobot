# FR3 Waypoint Capture (2026-04-15)

## Goal

`tools/fr3/fr3_waypoint_capture.py` captures fixed-point FR3 datasets with this loop:

1. move EE to waypoint `i`
2. wait for settle + dwell
3. read robot EE pose
4. read all configured cameras (for example 6 Hikrobot cameras)
5. save one synchronized frame into `LeRobotDataset`
6. continue with waypoint `i+1`

Each saved frame contains:

- `observation.state`: measured robot EE pose + gripper
- `observation.images.<camera_name>`: image for each configured camera
- `observation.device_capture_timestamp`: robot and per-camera capture times
- `action`: commanded waypoint pose used for this frame

## Files

- Runtime script: `tools/fr3/fr3_waypoint_capture.py`
- Example config: `tools/fr3/fr3_waypoint_capture_example.yaml`

## Config Schema

Top-level keys:

- `robot`: `franka_research3` config, including `cameras`
- `dataset`: output dataset config
- `runtime`: settle/dwell behavior
- `waypoints`: list of target poses

Waypoint fields:

- required position: `ee.x`, `ee.y`, `ee.z`
- required orientation: exactly one format:
  - rotvec: `ee.wx`, `ee.wy`, `ee.wz`
  - quaternion: `ee.qx`, `ee.qy`, `ee.qz`, `ee.qw`
- optional gripper: `gripper.pos` in `[0, 1]`

## Run

For Hikrobot GigE, run on host and keep MVS environment variables available:

```bash
source "$HOME/.local/bin/env"
PYTHONPATH=src:/opt/MVS/Samples/64/Python:/opt/MVS/Samples/32/Python \
LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib:/usr/local/lib \
HIKROBOT_MVS_HOME=/opt/MVS \
MVCAM_COMMON_RUNENV=/opt/MVS/lib \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --python .venv/bin/python python tools/fr3/fr3_waypoint_capture.py \
  --config_path tools/fr3/fr3_waypoint_capture_example.yaml
```

## Notes

- This is software-level synchronization using each device's latest frame and timestamp.
- `timestamp` in `LeRobotDataset` remains canonical (`frame_index / fps`).
- Real capture times are stored in `observation.device_capture_timestamp`.
- With `dataset.num_episodes = 1`, one episode contains all configured waypoints in order.

## Dataset Feature Names

- `observation.state` names:
  - `ee.x`, `ee.y`, `ee.z`, `ee.qx`, `ee.qy`, `ee.qz`, `ee.qw`, `gripper.pos`
- `action` names:
  - `ee.x`, `ee.y`, `ee.z`, `ee.qx`, `ee.qy`, `ee.qz`, `ee.qw`, `gripper.pos`
- `observation.device_capture_timestamp` names:
  - `robot.ee.capture_timestamp_s`
  - `camera.<camera_name>.capture_timestamp_s` for each configured camera
