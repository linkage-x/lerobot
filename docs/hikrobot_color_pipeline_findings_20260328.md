# Hikrobot Color Pipeline Findings 2026-03-28

This note captures the Hikrobot color-cast investigation completed on 2026-03-28 for camera `DA5404760`.

## Symptom

- A recording made with:
  - `white_balance_mode=auto_continuous`
  - `keep_awb_running_during_recording=true`
  - `exposure_us=10000`
  - `gain_db=12`
  - `gamma=1.3`
  showed obvious color cast while the scene was static.
- Earlier recordings showed a brief green cast at startup.
- A later baseline recording (`outputs/hikrobot/hikrobot_DA5404760_1280x720_30fps_20260328_172813.mp4`) showed a much stronger sustained color error: blue was nearly absent in the encoded video statistics.
- Hikrobot's official OpenCV demo, using continuous white balance, did not show the same startup or steady-state color failure on the same camera.

## Initial Hypotheses

- Continuous auto white balance was re-entering a convergence phase at recording start.
- Gamma or mp4 encoding might be distorting the channels.
- Our capture pipeline might not match the official MVS sample's pixel-format and color-conversion path.

## Evidence Collected

### AWB State Handling

- `tools/hikrobot/hikrobot_record_test.py` originally used `auto_continuous` by default and called `camera.get_white_balance_ratios()` immediately after `connect(warmup=True)`.
- `get_white_balance_ratios()` temporarily disabled `BalanceWhiteAuto`, read the ratios, then restored the previous mode.
- This explained the early startup transient: the recording path could restart continuous AWB immediately before the first recorded frames.

### Official Sample Behavior

- The local MVS OpenCV sample at
  `/opt/MVS/Samples/64/OpenCV/Python/GrabImage_Cv/GrabImage_Cv.py`
  does not force the device into a new packed color format and then swap channels in OpenCV.
- Instead, it:
  - reads the device buffer,
  - optionally HB-decodes,
  - converts to `PixelType_Gvsp_RGB8_Packed` with `MV_CC_ConvertPixelTypeEx`,
  - writes that result directly to OpenCV / disk.

### Our Pipeline Behavior

- The Hikrobot camera backend forced device-side `RGB8`:
  - `MV_CC_SetEnumValue("PixelFormat", 0x02180014)`
- Then, in `_postprocess_image()`, it applied:
  - `cv2.cvtColor(processed, cv2.COLOR_RGB2BGR)`
- That means the pipeline effectively requested `RGB8` from the device and then swapped channels again before returning frames to the rest of the stack.

### Direct Comparison

- Official sample saved reference frame statistics were roughly:
  - `B=17.49`
  - `G=11.17`
  - `R=10.06`
- Our pipeline's pre-encoding frame from the same camera and similar settings was roughly:
  - `B=14.56`
  - `G=16.37`
  - `R=25.21`
- The channel relationship was consistent with an `R/B` mismatch rather than an AWB-only problem.

### Live Drift Check

- A direct 180-frame live sample from our pipeline with:
  - `white_balance_auto=continuous`
  - `lock_white_balance_after_warmup=False`
  - `gamma=1.3`
  did not show runaway green drift over time.
- This ruled out the theory that continuous AWB alone was the main root cause of the steady-state color failure.

### Encoding Check

- Re-encoding a captured BMP frame through the same `mp4v` path caused only minor channel movement.
- The severe blue-channel loss seen in the bad recording was therefore not caused primarily by OpenCV's `mp4v` encoder.

## Root Cause

There were two separate problems:

1. Startup transient:
   the recording script could restart continuous white balance right before writing frames by querying white-balance ratios after warmup.

2. Main color bug:
   the Hikrobot capture path requested one packed device color format and then performed an extra channel swap in software.
   This made our effective color pipeline differ from Hikrobot's official sample and produced incorrect channel ordering in recorded frames.

The steady-state "green" failure was not primarily an AWB instability issue. It was a color-pipeline mismatch.

## Fix Applied

### AWB Handling

- The camera backend now tracks the effective white-balance auto mode internally.
- Querying white-balance ratios no longer restarts continuous AWB after it has been locked off.
- The recording tool defaults to:
  - use continuous AWB during warmup,
  - lock white balance before recording,
  - keep a separate explicit flag when continuous AWB should remain active while recording.

### Pixel Format and Channel Order

- The camera backend now requests device-side color format based on `color_mode`:
  - `ColorMode.BGR` -> device `BGR8`
  - `ColorMode.RGB` -> device `RGB8`
- The extra `cv2.COLOR_RGB2BGR` conversion in `_postprocess_image()` was removed.
- In practice, the `BGR` path now stays BGR end-to-end, which matches the rest of the OpenCV recording path.

## Files Changed

- `src/lerobot/cameras/hikrobot/camera_hikrobot.py`
- `src/lerobot/cameras/hikrobot/configuration_hikrobot.py`
- `tools/hikrobot/hikrobot_record_test.py`
- `tests/cameras/test_hikrobot.py`

## Regression Coverage

- Added tests for:
  - restoring continuous AWB only when it is still active,
  - not restarting AWB after lock,
  - requiring `BGR888` support for `BGR` mode,
  - preserving device-side BGR channel order in `BGR` mode.

## Practical Guidance

- For stable recorded video:
  - use warmup and lock white balance before writing frames.
- For preview-style operation:
  - continuous AWB can remain enabled, but do not treat startup frames as color-stable without an explicit pre-roll policy.
- When comparing with official Hikrobot demos:
  - verify the full pixel path, not just exposure, gain, gamma, and white-balance settings.
  - The main discrepancy in this incident was the color-format pipeline, not the AWB setting alone.
