# Handheld Multimodal Capture Progress

## Current Status

As of 2026-04-10, `tools/handheld/handheld_record.py` can complete a 1-second smoke recording with:

- 6 Hikrobot GigE cameras
- 1 OpenCV USB camera at `/dev/video6`
- 1 Intel RealSense D405 with serial `315122271805`
- 1 Paxini Gen2 Omega tactile controller on `/dev/ttyACM0`
- 1 Pika Sense device on `/dev/ttyUSB0`

The recording was saved successfully to:

- `/tmp/handheld_multimodal_v1_smoke`

## Repo Changes

The following capture-path fixes have been applied in-repo:

- `third_party/pika_sdk/` is vendored from `https://github.com/linkage-x/pika_sdk`
- `tools/fr3/setup_host_env.sh` now installs vendored `pika_sdk` by default
- `tools/handheld/handheld_record.py` registers `opencv` and `intelrealsense` config types
- Hikrobot camera connect failures now emit warnings and do not block the remaining cameras
- Paxini tactile config now uses `connect_ids` only
- `observation.device_capture_timestamp` now logs to stable per-device Rerun paths for both live recording and dataset replay:
  - `observation/device_capture_timestamp/camera/<name>`
  - `observation/device_capture_timestamp/tactile/<name>`
  - `observation/device_capture_timestamp/handheld_gripper/<name>`
- Handheld tactile frames are exported in a viewer-friendly layout:
  - `observation.tactile.paxini.left_xyz`: `(3, 10, 12)`
  - `observation.tactile.paxini.right_xyz`: `(3, 10, 12)`
  - `observation.tactile.paxini.left_magnitude`: `(10, 12)`
  - `observation.tactile.paxini.right_magnitude`: `(10, 12)`
  - `observation.tactile.paxini.raw_xyz`: `(2, 120, 3)`

`raw_xyz` preserves the original per-side taxel ordering, while the `left/right` fields are reshaped for direct Rerun inspection.

## Current Sensor Mapping

Current `tools/handheld/handheld_record_example.yaml` assumptions:

- Hikrobot cameras:
  - `cam_0`: `DA9342471`
  - `cam_1`: `DA9342716`
  - `cam_2`: `DA9342700`
  - `cam_3`: `DA9342583`
  - `cam_4`: `DA9342477`
  - `cam_5`: `DA9342685`
- `pika_opencv.index_or_path`: `/dev/video6`
- `pika_realsense.serial_number_or_name`: `315122271805`
- `paxini.serial_port`: `/dev/ttyACM0`
- `paxini.connect_ids`: `[6, 10]`
- `pika.port`: `/dev/ttyUSB0`

## Working Commands

Use the project `uv` environment and keep the MVS variables split correctly.

### 1. Record a smoke episode

```bash
source "$HOME/.local/bin/env"
PYTHONPATH=src:/opt/MVS/Samples/64/Python:/opt/MVS/Samples/32/Python \
LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib:/usr/local/lib \
HIKROBOT_MVS_HOME=/opt/MVS \
MVCAM_COMMON_RUNENV=/opt/MVS/lib \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --python .venv/bin/python python tools/handheld/handheld_record.py \
  --config_path tools/handheld/handheld_record_example.yaml \
  --display_data false \
  --dataset.root /tmp/handheld_multimodal_v1_smoke \
  --dataset.num_episodes 1 \
  --dataset.episode_time_s 1
```

Notes:

- The script waits for Enter before each episode.
- After recording, confirm `Save current episode? [Y/n]:`.
- `*******XOpenDisplay Fail *******` from MVS is expected in headless mode and did not block recording.

### 2. Inspect the saved dataset in Rerun

```bash
source "$HOME/.local/bin/env"
UV_CACHE_DIR=/tmp/uv-cache \
PYTHONPATH=src \
uv run --python .venv/bin/python python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id local/handheld_multimodal_v1 \
  --root /tmp/handheld_multimodal_v1_smoke \
  --episode-index 0
```

## Validation

Targeted tests covering the current handheld capture path:

```bash
source "$HOME/.local/bin/env"
UV_CACHE_DIR=/tmp/uv-cache \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src \
uv run --python .venv/bin/python python -m pytest \
  tests/tactiles/test_paxini_gen2.py \
  tests/scripts/test_handheld_record.py
```
