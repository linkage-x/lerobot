# `argus_online_sync` implementation status

## Current State

The current branch contains a working implementation of the planned
`argus_online_sync` backend, and the shared Thor example YAML now selects it
as the default production camera backend.

- `gmsl2_record.py` accepts `recorder_backend: argus_online_sync` and parses
  `sensors.cameras.online_sync`.
- `argus_online_sync_session.py` is wired as a drop-in session backend for
  `thor_record.py`.
- `thor_record.py` selects `ArgusOnlineSyncCameraSession` for this backend and
  gates saved episodes on `online_sync_manifest.json`.
- `argus_online_sync_video_recorder.cpp` opens all selected cameras in one
  process, reads same-frame Argus buffer + metadata, aligns frames by SOF TSC,
  and pushes only full SOF clusters into per-camera hardware encoders.
- `thor_gmsl2_11ch_example.yaml` now defaults to
  `recorder_backend: argus_online_sync`.
- `argus_metadata` remains available as an explicit fallback for diagnostics
  or rollback.

## UI Compatibility

The UI usage flow is unchanged:

```text
Connect -> Start -> Stop / auto duration -> Save / Discard
```

The GUI still launches `tools/thor/gmsl2/thor_record.py` and uses the same stdin
protocol. The backend change is internal to the recorder session:

- Connect still detects cameras, arms hardware sync, applies camera controls,
  starts BOX if enabled, and reports active cameras.
- Start still creates one episode directory and starts one camera session.
- Stop still ends the episode through the same session API.
- Save now checks `online_sync_manifest.json` instead of running the old
  metadata materialization path.

No frontend or operator workflow change is required.

## Implementation Summary

The successful path now follows NVIDIA's Thor MMAPI sample pattern:

```text
Argus BufferOutputStream
  -> preallocated DmaBuffer pool
  -> IBuffer::getMetadata() for same-buffer SOF TSC
  -> SOF full-cluster alignment
  -> NvVideoEncoder
  -> encoded appsrc muxer
  -> cam_XX.mkv
```

Important details:

- The recorder no longer uses GStreamer raw `appsrc` for NV12 input.
- The recorder no longer creates a new `NvBuffer` per frame from
  `IImageNativeBuffer`.
- Each accepted `FrameBundle` owns one Argus `Buffer*` and one `DmaBuffer*`.
- Startup/warmup frames are released immediately and are not encoded.
- Accepted frames are released back to Argus only after the encoder output plane
  dequeues the corresponding buffer.
- Each encoder keeps a bounded number of in-flight Argus buffers to avoid
  starving the Argus producer.
- The recorder writes `online_sync_manifest.json` and then exits with `_exit`
  after normal completion. This avoids NVIDIA Argus/MMAPI destructor-chain
  crashes after all episode files have already been finalized.
- `ArgusOnlineSyncCameraSession` uses a longer default recorder stop timeout
  (`120 s`) than the metadata backend because multi-camera encoder EOS/mux close
  can exceed the old 10-second timeout.

## SOF Cluster Contract

For every logical frame:

1. each active camera must have at least one acquired Argus buffer;
2. the queue-front SOF TSC values are compared across cameras;
3. if `max_sof - min_sof <= tolerance_ns`, the cluster is accepted;
4. otherwise frames older than `max_sof - tolerance_ns` are released and
   counted as unmatched boundary drops;
5. after recording starts, any unmatched drop means the episode fails instead
   of inserting or duplicating frames.

This means the output frame count is not chosen from one reference camera. The
logical frame count is the number of full SOF clusters accepted by all active
cameras.

## Thor Verification

All tests below used:

```text
fps=60
codec=h265
container=mkv
tolerance_ms=1.0
no ffmpeg materialization
```

### 2 cameras, 1 frame

```text
sids: 6,7
actual_frames: 1
video frames: 1 / 1
sidecar rows: 1 / 1
online_sync_manifest.ok: true
```

### 2 cameras, 60 frames

```text
sids: 6,7
actual_frames: 60
video frames: 60 / 60
sidecar rows: 60 / 60
online_sync_manifest.ok: true
max_abs_delta_ns_by_camera: 5000 ns
```

### 2 cameras, 600 frames

```text
sids: 6,7
actual_frames: 600
video frames: 600 / 600
sidecar rows: 600 / 600
online_sync_manifest.ok: true
max_abs_delta_ns_by_camera: 5000 ns
```

### 7 cameras, 60 frames

```text
sids: 3,6,7,8,9,13,14
actual_frames: 60
video frames: 60 for every camera
sidecar rows: 60 for every camera
online_sync_manifest.ok: true
max_abs_delta_ns_by_camera: <= 119500 ns
```

### 7 cameras, 600 frames

```text
sids: 3,6,7,8,9,13,14
actual_frames: 600
video frames: 600 for every camera
sidecar rows: 600 for every camera
online_sync_manifest.ok: true
max_abs_delta_ns_by_camera: <= 185500 ns
```

### 7 cameras, 3600 frames

```text
sids: 3,6,7,8,9,13,14
actual_frames: 3600
video frames: 3600 for every camera
sidecar rows: 3600 for every camera
online_sync_manifest.ok: true
max_abs_delta_ns_by_camera: <= 8000 ns
elapsed wall time including open/close/ffprobe wrapper: 159 s before ffprobe
```

The 3600-frame test output directory was:

```text
/tmp/lerobot_online_sync_buffer_7cam_3600f_20260706_152028
```

### `thor_record.py` UI-protocol smoke

Temporary YAML changes only:

```text
recorder_backend: argus_online_sync
detect_all: false
sensor_ids: [6, 7]
episode_time_s: 1
num_episodes: 1
recording_preview_enabled: false
```

Result:

```text
Episode saved.
Driver RC: 0
actual_frames: 60
video frames: 60 / 60
sidecar rows: 60 / 60
online_sync_manifest.ok: true
max_abs_delta_ns_by_camera: 4000 ns
meta.json: present
```

The UI smoke output directory was:

```text
/tmp/lerobot_online_sync_ui_smoke_20260706_153739
```

## Local Verification After Default Switch

After switching the default backend from `argus_metadata` to
`argus_online_sync`, the following local checks passed:

```bash
python3 -m py_compile \
  tools/thor/gmsl2/argus_online_sync_session.py \
  tools/thor/gmsl2/gmsl2_record.py \
  tools/thor/gmsl2/thor_record.py \
  tests/scripts/test_thor_argus_metadata_session.py \
  tests/scripts/test_thor_record_meta.py

git diff --check
```

The default configuration also resolves to the intended backend:

```text
CameraDefaults().recorder_backend = argus_online_sync
thor_gmsl2_11ch_example.yaml recorder_backend = argus_online_sync
online_sync = enabled, sync_source=sof_tsc_ns, tolerance_ms=1.0
```

Direct-import smoke checks passed for:

- default backend selection;
- explicit `argus_metadata` fallback acceptance;
- online-sync preflight timeout defaults;
- online-sync bounded preflight success path with fake recorder sidecars;
- online-sync group preflight timeout kill path;
- online-sync group timeout fast-fail path that avoids sequential
  single-camera isolation;
- online-sync named-camera group failure still drops the named camera and
  retries survivors;
- online-sync single-camera preflight shorter timeout path;
- legacy CLI rejection of non-splitmux backends;
- metadata sync stop-marker trimming;
- online-sync manifest accept path;
- online-sync missing-manifest reject path;
- online-sync frame-count-mismatch reject path.

Full project `pytest` is still blocked in this local environment because
`tests/conftest.py` imports `serial`, which is not installed here.

## 2026-07-07 8-Camera Burn-In Attempt

Requested test:

```text
current locked cameras, 10 rounds, 60 s per round, 60 Hz
```

Current locked camera set at test start:

```text
3,6,7,8,9,12,13,14
```

Result:

- the UI-path 10x60 run did not reach `Episode 0 ready`;
- `thor_record.py` timed out during 8-camera online-sync preflight after the
  global 120 s Connect budget;
- direct 8-camera 60-frame recorder smoke accepted 8 synchronized frames, then
  failed with `missing full SOF cluster inside recording window`;
- the accepted 8 frames had equal video/sidecar counts on every camera and max
  SOF delta below 1 ms;
- direct 7-camera smoke excluding `cam_12` passed 60/60 frames with equal
  video/sidecar counts and max SOF delta around 7 us;
- after the 8-camera failure, `recover_argus.sh` left only `12,13` probe-OK;
  the other locked cameras showed driver/Argus errors such as `i2c write
  failed`, `max96726 i2c-w, write failed`, and `Error turning on streaming`.

The local report and logs are preserved at:

```text
outputs/datasets/online_sync_burnin_10x60_20260707_021343/
```

Interpretation:

- the online-sync recorder correctly refused to insert or duplicate frames when
  a full cluster was missing inside the recording window;
- the all-current-camera burn-in is blocked by 8-camera Thor camera-stack
  stability, not by post-recording materialization;
- `argus_online_sync` still performs no ffmpeg materialization/re-encode.

Code follow-up from this attempt:

- `online_sync.preflight_timeout_s` now defaults to 30 s;
- `online_sync.single_preflight_timeout_s` now defaults to 10 s;
- `online_sync.frame_timeout_ms` now defaults to 1000 ms and is passed through
  to the C++ recorder as `--frame-timeout-ms`. This replaces the old hard-coded
  8 s Argus buffer acquire timeout inside full-cluster formation. The value is
  not a sync tolerance; it is the maximum time to wait for a camera to deliver
  the next Argus buffer before failing the episode/preflight.
- Full-cluster acquisition failures now carry the concrete camera/error detail
  into `online_sync_manifest.json.failure`, for example
  `cam_12: timed out waiting for Argus buffer after 1000 ms`. The burn-in
  analyzer surfaces that same manifest failure in the markdown/JSON report, so
  the next all-camera failure should identify the first stream that stopped
  delivering buffers without manually scanning the full driver log.
- online-sync preflight uses `Popen(..., start_new_session=True)` and kills the
  recorder process group on timeout;
- online-sync group preflight timeout now fails fast instead of running
  sequential single-camera isolation. This prevents an 8-camera group timeout
  from turning into a long `30 + N*10` second Connect wait and avoids further
  stressing an already wedged Argus/driver stack;
- if the group error names a camera, online-sync still drops that camera and
  retries the survivors;
- group/single preflight timeouts are separate from episode stop/mux close
  timeout, which remains 120 s.
- `tools/thor/gmsl2/online_sync_burnin.py` now provides a repeatable burn-in
  driver and analyzer. It uses the same stdin/stdout protocol as the UI, writes
  a temporary YAML with `recorder_backend: argus_online_sync`, and produces
  `online_sync_burnin_sync_report.md` plus `online_sync_burnin_summary.json`.
  It does not run ffmpeg materialization or any re-encode; optional `--ffprobe`
  is read-only frame-count QC.
- The burn-in driver uses a reader thread instead of selector-based text I/O so
  protocol lines buffered by Python are not missed. By default it fails fast on
  recorder `ERROR:`, `Episode discarded`, or `Stream exited early:` output to
  avoid repeatedly stressing a wedged all-camera Argus/driver stack. Use
  `--continue-on-failure` only when deliberately collecting multiple failures.

Recommended next clean Thor burn-in command after a reboot or camera power
cycle:

```bash
cd /home/nvidia/lerobot
python3 tools/thor/gmsl2/online_sync_burnin.py run \
  --episodes 10 \
  --episode-time-s 60 \
  --fps 60 \
  --ffprobe
```

For a fast camera-only sanity check on the previously stable 7-camera set:

```bash
cd /home/nvidia/lerobot
python3 tools/thor/gmsl2/online_sync_burnin.py run \
  --episodes 1 \
  --episode-time-s 60 \
  --fps 60 \
  --sensor-ids 3,6,7,8,9,13,14 \
  --ffprobe
```

The analyzer can also be rerun on existing datasets:

```bash
python3 tools/thor/gmsl2/online_sync_burnin.py analyze \
  outputs/datasets/<dataset_root> \
  --expected-episodes 10 \
  --expected-frames 3600 \
  --ffprobe
```

## Remaining Issues

- Argus still logs socket/client errors during controlled shutdown. The recorder
  returns `RC=0` and the saved files/manifest are valid, but the logs are noisy.
  Treat these as shutdown noise only when they occur after the recorder has
  emitted the final manifest and exited successfully.
- 8-camera short UI-path recording is now verified after the recorder shutdown
  fix. Longer 8/10/11-camera production burn-in is still needed.
- UI on-demand preview should still be smoke-tested with the online-sync default
  on a clean Thor session. Episode recording itself stops idle preview before
  capture, so preview cannot compete with the online-sync recorder.

## Prior Failed Paths Kept For Context

- Sequential per-camera acquisition failed because queue starts can differ by
  one trigger period, causing SOF deltas around `16.66 ms`.
- CPU/system-memory GStreamer raw `appsrc` could align frames at microsecond
  scale but was too slow for 1080p60.
- DMABUF/NVMM GStreamer raw `appsrc` variants segfaulted inside
  `libgstnvvideo4linux2.so`.
- A direct `FrameConsumer -> IImageNativeBuffer::createNvBuffer -> NvVideoEncoder`
  path could queue the first frames but was unstable for longer 60Hz runs.

The current `BufferOutputStream + DmaBuffer + NvVideoEncoder` path is the first
path that passed the 7-camera 3600-frame Thor verification without post-recording
re-encode.

## 2026-07-07 Follow-Up Remote Test

Changes verified on Thor:

- updated `argus_online_sync_video_recorder.cpp`,
  `argus_online_sync_session.py`, `online_sync_burnin.py`,
  `gmsl2_record.py`, `thor_record.py`, and the shared YAML were synced to
  `/home/nvidia/lerobot`;
- Python compile on Thor passed;
- C++ online-sync recorder compiled on Thor via
  `ArgusOnlineSyncCameraSession._build_binary()`;
- `/tmp/lerobot_argus_online_sync_video_recorder --help` showed the new
  `--frame-timeout-ms 1000` option.

Test attempt 1:

```bash
python3 tools/thor/gmsl2/online_sync_burnin.py run \
  --episodes 1 \
  --episode-time-s 60 \
  --fps 60 \
  --sensor-ids 3,6,7,8,9,13,14 \
  --ffprobe
```

Result:

- dataset root:
  `outputs/datasets/online_sync_burnin_1x60_20260707_025624`;
- no episode was saved;
- `3,7,8` timed out in the initial nvargus probe, leaving
  `6,9,13,14`;
- online-sync preflight then timed out and the new C++ diagnostic identified
  `cam_06: timed out waiting for Argus buffer after 1000 ms`;
- this exposed a Python matching bug: the session matched all camera names in
  the text `for cam_06,cam_09,cam_13,cam_14` and dropped every survivor.

Follow-up code fix:

- `argus_online_sync_session.py` now only treats `cam_XX:` recorder-error
  prefixes as specific failed cameras;
- the candidate-list context `for cam_06,cam_09,...` no longer causes mass
  camera drops;
- local tests cover this exact timeout text and verify only `cam_06` is
  dropped.

Test attempt 2 after syncing that Python fix:

- dataset root:
  `outputs/datasets/online_sync_burnin_1x60_20260707_025930`;
- no episode was saved;
- all seven requested camera IDs still appeared locked, but every nvargus probe
  timed out or emitted Argus/GStreamer errors;
- recorder failed before Connect with `ERROR: no cameras stream through
  nvargus`.

Manual recovery:

```bash
bash tools/thor/gmsl2/recover_argus.sh \
  --sdk tools/thor/gmsl2/sdk \
  --sids 3,6,7,8,9,13,14 \
  --probe-timeout 8 \
  --probe-buffers 30
```

Result:

- `RECOVER_OK_SIDS=` was empty;
- `RECOVER_FAIL_SIDS=3,6,7,8,9,13,14`;
- probe logs showed `NvBufSurfaceFromFd Failed`, `dmabuf_fd -1 mapped entry
  NOT found`, `Failed to create CaptureSession`, and `Failed to create
  CameraProvider`;
- dmesg showed `ar0234c ... i2c write failed`, `max96726 ... i2c-w, write
  failed`, and `Error turning on streaming`;
- no recorder or `gst-launch-1.0` process was left running afterward.

Local evidence pulled back:

```text
outputs/datasets/online_sync_burnin_1x60_20260707_025624/
outputs/datasets/online_sync_burnin_1x60_20260707_025930/
```

Current conclusion:

- the C++/Python fixes compile and the new diagnostics work;
- the failed run above used a software recovery path that did not reproduce the
  operator's later successful recovery with
  `--sdk ~/Desktop/SG16A_AGTH_G3Y_A1`;
- after that recovery, the remaining blocker was a recorder shutdown-order bug,
  not frame synchronization itself.

## 2026-07-07 8-Camera Shutdown Fix

Operator recovery evidence:

- manual `recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1` reported all
  locked cameras probe-OK:
  `3,6,7,8,9,12,13,14`.

Bug found:

- direct 8-camera, 2-frame online-sync recorder accepted synchronized frames
  and wrote `online_sync_manifest.json` with `ok=true`;
- however the recorder did not exit until the outer `timeout` sent SIGTERM;
- root cause was shutdown ordering in
  `argus_online_sync_video_recorder.cpp`: `waitForIdle()` was called while the
  encoder output plane still held accepted Argus buffers. Argus idle could then
  wait for buffers that were only released by `encoder->stop()`.

Fix:

- stop Argus repeat and mark the buffer stream EOS;
- do not call unbounded `waitForIdle()` before encoder drain;
- let `encoder->stop()` drain EOS and release the held Argus buffers.

Verification after syncing and rebuilding on Thor:

```text
Direct 8 cameras, 2 frames:
  sids: 3,6,7,8,9,12,13,14
  recorder rc: 0
  manifest ok: true
  actual_frames: 2
  frame_count_by_camera: 2 for every camera
  max_abs_delta_ns_by_camera: <= 257500 ns

UI/burn-in path, 8 cameras, 10 s at 60 Hz:
  dataset root:
    /home/nvidia/lerobot/outputs/datasets/online_sync_burnin_1x10_20260707_031902
  driver rc: 0
  manifest ok: true
  actual_frames: 600
  video frames: 600 for every camera
  sidecar rows: 600 for every camera
  max SOF delta: 10000 ns
  cleanup_duration_s: 0.26433092399383895
  postprocessing: no ffmpeg materialization/re-encode
```

Remaining validation:

- rerun longer 8-camera 60 s and 10x60 burn-in after this shutdown fix;
- then expand to any additional locked production camera set.
