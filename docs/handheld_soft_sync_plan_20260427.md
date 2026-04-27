# Handheld Multi-Sensor Soft Sync Plan

## Problem

The handheld multimodal recorder currently samples each device with `read_latest()` on the dataset clock. It preserves per-device capture timestamps, but it does not wait for camera, tactile, and gripper streams to reach the same target capture time before committing a dataset frame.

This means a nominal `timestamp = frame_index / fps` row can contain device observations captured at noticeably different host times.

## Target Behavior

- Keep the LeRobot dataset clock canonical: dataset rows still use `frame_index / dataset.fps`.
- Before each row is collected, compute the target host capture time from `episode_start_time_s + frame_index / fps`.
- Keep a short timestamped sample buffer for each configured device.
- Wait until every configured device has buffered at least one sample with `timestamp >= target_time - tolerance`.
- If all devices enter the window before timeout, select the buffered sample nearest to the target time.
- If timeout expires, keep recording with the latest available samples, but mark the row as timed out so bad sync can be filtered or inspected later.
- Do not move hardware-specific logic into handheld orchestration. Device adapters continue to own background capture and `latest_timestamp` updates.

## Configuration

Add `sensors.soft_sync` to `tools/handheld/handheld_record_example.yaml`:

- `enabled`: turn soft sync waiting on/off.
- `tolerance_ms`: accepted early-arrival margin against the target time.
- `wait_timeout_ms`: maximum wait per dataset row before falling back to latest samples.
- `poll_interval_ms`: polling interval while waiting for timestamps to reach the target.
- `buffer_duration_s`: per-device timestamped sample history retained for nearest-target selection.

Initial defaults:

- `enabled: true`
- `tolerance_ms: 20.0`
- `wait_timeout_ms: 150.0`
- `poll_interval_ms: 1.0`
- `buffer_duration_s: 0.25`

## Dataset Diagnostics

When soft sync is enabled, add `observation.soft_sync` with:

- `target_timestamp_s`: target dataset-relative capture time.
- `max_skew_s`: `max(device_capture_timestamp) - min(device_capture_timestamp)` at collection time.
- `oldest_device_lag_s`: positive lag when the oldest device timestamp is still behind the target.
- `global_lag_s`: `median(device_capture_timestamp) - target_timestamp_s`, used to detect full-pipeline lag.
- `timed_out`: `1.0` when the synchronizer had to fall back after timeout, otherwise `0.0`.

The existing `observation.device_capture_timestamp` remains the per-device source of truth.

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
