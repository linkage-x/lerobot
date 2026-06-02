# Recorder-Owned Preview Design

## Problem

Thor needs two workflows at the same time:

1. Stable 11-camera GMSL2 recording through Argus, NVMM, NVENC, and splitmuxsink.
2. Live camera preview in Device Manager while the recorder owns the cameras.

The old idle preview opened a second `nvarguscamerasrc` per sensor from the gateway. That is safe only when the recorder is stopped. During recording it competes for the same Argus sensor-id and can leave stale Argus sessions behind.

A naive recorder preview branch added at Connect time also fails the priority model. Connect is the most fragile phase: each worker opens Argus, allocates NVMM buffers, starts VI/ISP flow, and starts NVENC. Adding `tee -> nvvidconv -> jpegenc -> appsink` for 11 cameras during this phase amplifies the known `NvBufSurfaceFromFd Failed` / `dmabuf_fd -1` race.

## First-principles constraints

- One sensor-id should have one Argus owner while recording.
- Recording is the hard real-time path; preview is optional and lossy.
- Preview must never back-pressure the recording branch.
- Preview startup must not overlap the 11-camera Argus open burst.
- Preview failure must not turn into a recording stream failure.
- The gateway must not spawn `nvarguscamerasrc` while the recorder owns cameras.

## Chosen design

Each recorder worker builds the initial Argus pipeline with no raw-surface preview fork. The only tee is after NVENC and parser, where it duplicates H26x bytestream rather than Argus/NVMM buffers:

```text
nvarguscamerasrc
  -> caps
  -> nvv4l2h26xenc
  -> h26xparse
  -> tee
     -> queue -> splitmuxsink
```

No preview conversion or decode elements are present while Connect proves the 11 recording streams can reach PLAYING. After all active recording workers are connected, `thor_record.py` calls `PersistentCameraSession.enable_previews()`. Each worker then dynamically requests one encoded tee source pad and attaches a lossy branch:

```text
tee request pad
  -> leaky queue -> h26xparse -> decoder -> downscale -> videorate -> jpegenc -> appsink
```

After attaching the branch, the worker asks the encoder for an IDR frame so the preview decoder gets parameter sets quickly. Preview branches are created one camera at a time with a configurable stagger. They are removed again on disable, branch error, or recorder shutdown.

Worker preview frames are written atomically to:

```text
/dev/shm/lerobot_preview/<camera_id>.jpg
```

The gateway serves those JPEGs whenever the recorder owns the cameras. It keeps the previous idle-only preview pipeline only for the stopped-recorder case.

## Failure policy

- Preview queue is `leaky=downstream max-size-buffers=1`, so it drops frames instead of blocking recording.
- Preview branch bus errors are logged and ignored; the worker removes that dynamic branch and keeps recording.
- Gateway returns 503 if no fresh recorder-owned JPEG exists yet.
- Leaving Device Manager no longer needs to release recorder-owned Argus resources because no second Argus client was opened.

## Config knobs

Under `sensors.cameras`:

```yaml
recording_preview_enabled: true
recording_preview_stagger_s: 0.5
```

`recording_preview_enabled: false` restores recording-only behavior while preserving idle preview. `recording_preview_stagger_s` controls post-Connect dynamic preview branch creation cadence.

## Validation plan

1. Local unit tests:
   - pipeline without preview has no tee/appsink;
   - pipeline with preview has its tee after encoder/parser and no preview elements at Connect time;
   - proxy/session methods send enable/disable preview commands;
   - gateway serves recorder-owned preview while `state.process` is set.

2. Frontend build:
   - camera tiles poll snapshots both idle and recording states.

3. Thor hardware test (`nvidia@192.168.111.122`):
   - sync repo to Thor;
   - recover Argus if needed;
   - run `thor_record.py` with 11 cameras;
   - verify Connect reaches active pipelines;
   - verify `/dev/shm/lerobot_preview/cam_*.jpg` freshness;
   - record one short episode while preview files are updating;
   - inspect logs for `NvBufSurfaceFromFd Failed`, stream exits, or empty MKVs.
