# Handheld Multi-Sensor Soft Sync Plan

## Problem

The handheld multimodal recorder samples each device with `read_latest()` on the dataset clock. It preserves per-device capture timestamps in `observation.device_capture_timestamp`.

The default recording path should preserve raw samples and raw capture metadata first. Soft-sync analysis is now an offline step so that the original data remains auditable and can be reprocessed with different sync thresholds.

## Target Behavior

- Keep the LeRobot dataset clock canonical: dataset rows still use `frame_index / dataset.fps`.
- During recording, collect the latest sample from each connected device and store the raw per-device capture timestamps.
- Write `meta/handheld_raw_capture.json` with device/config metadata, capture timestamp feature names, and the fact that soft sync was or was not applied during recording.
- By default, do not add `observation.soft_sync` to newly recorded datasets.
- Run `tools/handheld/handheld_soft_sync.py` after recording to compute sync diagnostics from the raw capture timestamps.
- Do not move hardware-specific logic into handheld orchestration. Device adapters continue to own background capture and `latest_timestamp` updates.

## Configuration

`sensors.soft_sync` remains in `tools/handheld/handheld_record_example.yaml` for compatibility, but the default is now:

```yaml
sensors:
  soft_sync:
    enabled: false
```

When explicitly enabled, the recorder can still write `observation.soft_sync`, but normal collection should leave this disabled and use the offline script.

Soft-sync fields:

- `enabled`: turn live soft sync waiting on/off.
- `tolerance_ms`: accepted early-arrival margin against the target time.
- `wait_timeout_ms`: maximum wait per dataset row before falling back to latest samples.
- `poll_interval_ms`: polling interval while waiting for timestamps to reach the target.
- `buffer_duration_s`: per-device timestamped sample history retained for nearest-target selection.

Recording session control:

- `dataset.num_episodes = 0`: record unlimited saved episodes.
- `dataset.num_episodes > 0`: stop after that many saved episodes.
- `s`: stop and save the current episode immediately.
- `n`: stop and discard the current episode immediately.
- `Esc`: stop the recording session and discard the current in-progress episode.

Defaults:

- `enabled: false`
- `tolerance_ms: 20.0`
- `wait_timeout_ms: 150.0`
- `poll_interval_ms: 1.0`
- `buffer_duration_s: 0.25`

## Dataset Diagnostics

Raw recording always keeps `observation.device_capture_timestamp` as the per-device source of truth.

The recorder also writes:

```text
meta/handheld_raw_capture.json
```

The offline script writes:

```text
meta/handheld_soft_sync_report.json
```

When live soft sync is explicitly enabled, `observation.soft_sync` contains:

- `target_timestamp_s`: target dataset-relative capture time.
- `max_skew_s`: `max(device_capture_timestamp) - min(device_capture_timestamp)` at collection time.
- `oldest_device_lag_s`: positive lag when the oldest device timestamp is still behind the target.
- `global_lag_s`: `median(device_capture_timestamp) - target_timestamp_s`, used to detect full-pipeline lag.
- `timed_out`: `1.0` when the synchronizer had to fall back after timeout, otherwise `0.0`.

For default raw recordings, the same quantities are computed offline in the JSON report instead of being stored per row in the dataset.

## Commands

Raw smoke recording, no soft sync by default:

```bash
TS=$(date +%Y%m%d_%H%M%S) &&
PYTHONPATH=src:/opt/MVS/Samples/64/Python:/opt/MVS/Samples/32/Python \
LD_LIBRARY_PATH=/opt/MVS/lib/64:/opt/MVS/lib:/usr/local/lib \
HIKROBOT_MVS_HOME=/opt/MVS \
MVCAM_COMMON_RUNENV=/opt/MVS/lib \
UV_CACHE_DIR=/tmp/uv-cache \
uv run --python .venv/bin/python python tools/handheld/handheld_record.py \
  --config_path tools/handheld/handheld_record_example.yaml \
  --display_data false \
  --dataset.root "/tmp/handheld_multimodal_v1_smoke_${TS}" \
  --dataset.episode_time_s 10
```

Offline soft-sync report:

```bash
PYTHONPATH=src uv run --python .venv/bin/python python tools/handheld/handheld_soft_sync.py \
  --dataset "/tmp/handheld_multimodal_v1_smoke_${TS}" \
  --tolerance-ms 20 \
  --global-lag-tolerance-ms 50
```

## Execution Steps

1. Done: Add `HandheldSoftSyncConfig` and validation.
2. Done: Add a small soft-sync helper that only reads `latest_timestamp` from devices and returns diagnostics.
3. Done: Pass soft-sync diagnostics into `collect_dataset_frame`.
4. Done: Add `observation.soft_sync` feature metadata only when soft sync is enabled.
5. Done: Log soft-sync diagnostics to Rerun when present.
6. Done: Update the example YAML and handheld usage doc.
7. Done: Add unit tests for feature metadata, frame diagnostics, ready sync, and timeout fallback.
8. Done: Run targeted handheld tests.
9. Done: Replace latest-only row collection with short timestamped buffers and nearest-target sample selection.
10. Done: Decouple scheduled capture from dataset writes with a bounded episode queue.
11. Done: Add `global_lag_s` to the soft-sync diagnostics.
12. Done: Change the default recorder behavior back to raw latest-sample capture with raw metadata.
13. Done: Add `tools/handheld/handheld_soft_sync.py` for offline soft-sync diagnostics.

## Validation

```bash
source "$HOME/.local/bin/env"
UV_CACHE_DIR=/tmp/uv-cache \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src \
uv run --python .venv/bin/python python -m pytest tests/scripts/test_handheld_record.py
```

If the project virtualenv does not already contain test dependencies, use:

```bash
source "$HOME/.local/bin/env"
UV_CACHE_DIR=/tmp/uv-cache \
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 \
PYTHONPATH=src \
uv run --extra test --python .venv/bin/python python -m pytest tests/scripts/test_handheld_record.py
```

Latest hardware validation:

- Dataset: `/tmp/handheld_multimodal_v1_smoke_20260427_113717`
- Mode: 30 FPS, three saved episodes, 321 total rows.
- Devices: 8 Hikrobot cameras, `pika_opencv`, and Pika gripper. `pika_realsense` was not part of this run.
- Soft-sync timeouts: `0/321`.
- `global_lag_s`: mean `0.001544`, p95 `0.008732`, max `0.012947`.
- `max_skew_s`: mean `0.027073`, p95 `0.031574`, max `0.036192`.
- Conclusion: capture/write decoupling fixed the 30 FPS write-pipeline lag. The remaining skew is consistent
  with 30 FPS multi-camera sample phase and does not currently require another software pipeline change.

## Follow-Ups

- Keep `global_lag_s` p95 and `max_skew_s` p95 as per-dataset quality gates for future collection batches.
- Consider an offline dataset validator that reports per-episode skew percentiles and timeout counts.
- Only pursue hard sync or device-trigger work if the downstream task requires `max_skew_s < 20 ms`.
