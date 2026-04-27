# Handheld Multimodal Capture Progress

## Current Status

As of 2026-04-27, `tools/handheld/handheld_record.py` can complete handheld smoke recordings with:

- 8 Hikrobot GigE cameras
- 1 OpenCV USB camera at `/dev/video6`
- 1 Intel RealSense D405 with serial `315122271805`
- 1 Pika Sense device on `/dev/ttyUSB0`

The example config keeps Paxini tactile disabled by default. Uncomment the `sensors.tactiles.paxini` block when the Paxini Gen2 Omega controller is connected on `/dev/ttyACM0`.

Validated smoke recordings were saved successfully under timestamped roots such as:

- `/tmp/handheld_multimodal_v1_smoke_<YYYYmmdd_HHMMSS>`

## Repo Changes

The following capture-path fixes have been applied in-repo:

- `third_party/pika_sdk/` is vendored from `https://github.com/linkage-x/pika_sdk`
- `tools/fr3/setup_host_env.sh` now installs vendored `pika_sdk` by default
- `tools/handheld/handheld_record.py` registers `opencv` and `intelrealsense` config types
- Hikrobot camera connect failures now emit warnings and do not block the remaining cameras
- Paxini tactile config now uses `connect_ids` only
- The example config now leaves Paxini tactile commented out so camera + Pika smoke runs do not require the tactile controller
- During an interactive TTY recording, press `s` to stop and save the current episode immediately, or `n` to stop and discard it immediately
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
  - `cam_0`: `DA9342700`
  - `cam_1`: `DA9342716`
  - `cam_2`: `DA9342685`
  - `cam_3`: `DA9342471`
  - `cam_4`: `DA9342477`
  - `cam_5`: `DA9342673`
  - `cam_6`: `DA9342615`
  - `cam_7`: `DA9342583`
- `pika_opencv.index_or_path`: `/dev/video6`
- `pika_realsense.serial_number_or_name`: `315122271805`
- Optional Paxini tactile, disabled by default:
  - `paxini.serial_port`: `/dev/ttyACM0`
  - `paxini.connect_ids`: `[6, 10]`
- `pika.port`: `/dev/ttyUSB0`

## Working Commands

Use the project `uv` environment and keep the MVS variables split correctly.

### 1. Record a smoke episode

```bash
source "$HOME/.local/bin/env"
TS=$(date +%Y%m%d_%H%M%S)
DATASET_ROOT="/tmp/handheld_multimodal_v1_smoke_${TS}"

PYTHONPATH=src:/opt/MVS/Samples/64/Python:/opt/MVS/Samples/32/Python \
LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib:/usr/local/lib \
HIKROBOT_MVS_HOME=/opt/MVS \
MVCAM_COMMON_RUNENV=/opt/MVS/lib \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --python .venv/bin/python python tools/handheld/handheld_record.py \
  --config_path tools/handheld/handheld_record_example.yaml \
  --display_data false \
  --dataset.root "${DATASET_ROOT}" \
  --dataset.num_episodes 3 \
  --dataset.episode_time_s 10
```

Notes:

- The script waits for Enter before each episode.
- While an episode is recording in an interactive terminal, press `s` to stop and save that episode immediately.
- While an episode is recording in an interactive terminal, press `n` to stop and discard that episode immediately.
- If the episode reaches `dataset.episode_time_s`, confirm `Save current episode? [Y/n]:`.
- `dataset.num_episodes` counts saved episodes. Discarded attempts do not advance the saved episode count.
- The early-stop shortcuts require a TTY. In non-interactive stdin environments, recording runs to the configured duration and then uses the normal save prompt when input is available.
- Paxini tactile is disabled in `tools/handheld/handheld_record_example.yaml`. Uncomment `sensors.tactiles.paxini` before running if tactile data should be included.
- `*******XOpenDisplay Fail *******` from MVS is expected in headless mode and did not block recording.

### 2. Inspect the saved dataset in Rerun

```bash
source "$HOME/.local/bin/env"
DATASET_ROOT=/tmp/handheld_multimodal_v1_smoke_<YYYYmmdd_HHMMSS>

UV_CACHE_DIR=/tmp/uv-cache \
PYTHONPATH=src \
uv run --python .venv/bin/python python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id local/handheld_multimodal_v1 \
  --root "${DATASET_ROOT}" \
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
