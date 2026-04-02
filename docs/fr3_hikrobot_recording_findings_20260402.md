# FR3 Hikrobot Recording Findings 2026-04-02

This note records the FR3 + Franka Hand + Hikrobot recording issues investigated on 2026-04-02.

## Scope

- Robot: `franka_research3`
- Gripper: `franka_hand`
- Cameras: dual Hikrobot GigE
- Teleop: SpaceMouse
- Recording config: `tools/fr3/fr3_record_hikrobot_example.yaml`

## Confirmed Control-Path Findings

### 1. Slow / sticky teleop was not a single SpaceMouse bug

The sluggish feel came from multiple stacked issues in the FR3 recording path:

- the outer record loop was targeting `control_fps: 200` while the real loop often ran much slower
- arm state was being read redundantly in the same tick
- `Franka Hand` state polling was synchronous in the control path
- `Franka Hand` commands were blocking the main control thread
- Hikrobot camera observations were being pulled too often for the dataset capture requirements

Mitigations applied in code:

- camera observations are now only sampled on dataset ticks for recording
- `Franka Hand` state is cached in a background thread
- `Franka Hand` commands run through an async worker
- `Franka Hand` command semantics are binary open/close only
- SpaceMouse is forced to binary gripper mode when `gripper_backend=franka_hand`
- Panda arm state is cached in a background reader
- FR3 reuses one observation snapshot per control tick instead of re-reading arm state

## Remaining Control-Path Risk

The arm can still feel sticky even after the fixes above because:

- IK still runs synchronously in the outer control loop
- `max_target_delta_pos` / `max_target_delta_rot` are per-tick clamps, so effective motion speed collapses when the real loop rate drops below the configured `control_fps`

Those two items remain the next likely root causes if teleop still feels slow.

## Confirmed Color-Pipeline Finding

### Symptom

- Recorded dataset videos could show red/blue channel swaps.
- Example observed by the operator:
  `outputs/datasets/fr3_hikrobot_pick_place_v1_20260402_111738/videos/observation.images.ee/chunk-000/file-000.mp4`
  contained regions that were red in the scene but appeared blue in the saved MP4.

### Root Cause

The Hikrobot backend can legitimately return `BGR` frames when configured with `color_mode: bgr`.

However, the dataset image/video writing path assumes generic 3-channel arrays are already `RGB`:

- `src/lerobot/datasets/image_writer.py`
  uses `PIL.Image.fromarray(image_array)`
- `src/lerobot/datasets/video_utils.py`
  uses `Image.fromarray(frame_data)` before PyAV encoding

Pillow interprets `uint8` HWC 3-channel arrays as `RGB`. If a `BGR` frame reaches this layer unchanged, red and blue are swapped in the saved PNG / MP4.

### Minimal Reproduction

A synthetic reproduction confirmed the mismatch:

- input frame pixel: `BGR = [255, 0, 0]`
- decoded pixel after current streaming video encode path: `RGB ~= [253, 0, 0]`

That means a pure-blue `BGR` frame was encoded as a near-pure-red `RGB` frame.

### Fix Applied

To avoid changing device-side camera behavior, the recording path now normalizes configured `BGR` camera observations to `RGB` before building dataset frames for:

- dataset saving
- policy observation frames during recording

This keeps the Hikrobot capture path unchanged while making recorded dataset images/videos follow the repository-wide `RGB` expectation.

## Note On Trying Device-Side `RGB`

Changing Hikrobot configs from `bgr` to `rgb` is theoretically valid and the backend has test coverage for `RGB888` support checks.

But real hardware validation was not completed in the current execution context on 2026-04-02 because a local `preflight` run could not reach FR3/Hikrobot devices from that session. Because of that, the safer fix was to leave the capture config in `bgr` and normalize only at dataset ingress.

## Files Updated In This Investigation

- `src/lerobot/scripts/lerobot_record.py`
- `tests/test_control_robot.py`
