# Argus Metadata Sync Implementation

Branch: `lht/box_sync`

## Goal

The UI should keep using the same recorder entrypoint, but camera episodes must be saved only when all camera streams can be aligned by per-frame Libargus timestamps. Local per-camera frame numbers are not treated as a synchronization contract.

The default backend in `thor_gmsl2_11ch_example.yaml` and
`CameraDefaults` is now `argus_metadata`. The old
`gstreamer_splitmux` backend remains available only as an explicit fallback.

## Synchronization Contract

Each camera must produce one sidecar row per encoded frame:

```text
camera,encoded_frame_index,local_frame_number,sensor_timestamp_ns,sof_tsc_ns,eof_tsc_ns,internal_frame_count
```

Episode save uses one reference camera's `sof_tsc_ns` sequence as the target frame sequence. Every other camera is matched by nearest `sof_tsc_ns` under a configured tolerance.

An episode is saveable only if:

1. every active camera has a sidecar;
2. the sidecars contain a non-empty common contiguous frame set;
3. every retained match is monotonic and within `argus_frame_sync.tolerance_ms`;
4. dropped frames are only at the episode boundary, not inside the retained window.

This handles the observed Thor behavior where equal local frame numbers can be one trigger period apart, while timestamp-nearest frames are only microseconds apart.

## Implemented Pieces

### `argus_frame_sync.py`

Pure Python sync core:

- reads/writes `<camera>.argus_frame_metadata.csv`;
- aligns frames by reference `SOF TSC`;
- drops boundary frames that are not common to all cameras;
- rejects interior missing frames and out-of-tolerance retained matches;
- writes `argus_frame_alignment.json`.
- exposes `camera_frame_windows()` for the final video materialization step.

This module is backend-independent and covered by local tests in:

```text
tests/scripts/test_thor_argus_frame_sync.py
```

### `thor_record.py` save gate

The UI still spawns the same recorder script. `thor_record.py` now understands:

```yaml
sensors:
  cameras:
    argus_frame_sync:
      enabled: true
      required: true
      reference_camera: ""
      tolerance_ms: 1.0
      report_name: argus_frame_alignment.json
```

When enabled, the recorder evaluates sidecars before saving. When `required` is true, sync failure changes the episode decision to discard and emits an operator-facing error line.

When `sensors.cameras.defaults.recorder_backend: argus_metadata` is selected,
`thor_record.py` forces `argus_frame_sync.enabled=true` and
`required=true`, because the backend's contract is the metadata sidecar.

For `argus_metadata`, a successful sync gate also rewrites each `cam_XX.mkv`
to the accepted frame window before the episode is saved. After this step the
episode video files themselves have identical frame counts.

When `argus_recording_markers.json` is present, the sync gate starts alignment
at `start_sof_tsc_ns`. This prevents Argus startup/warmup frames from being
included in the saved episode while the UI timer is already running.

The same marker file may also include `stop_sof_tsc_ns_exclusive`, written by
the signal-controlled recorder at UI Stop time before stopping Argus repeat.
The sync gate treats that value as an exclusive upper bound, so any frames
drained after Stop are kept in the raw sidecars/videos for debugging but are
not materialized into the final episode files.

### `argus_video_materialize.py`

Frame-window materialization layer:

- reads `camera_frame_windows(alignment)`;
- uses `ffmpeg` frame selection to rewrite each raw camera video to the exact
  accepted encoded-frame range;
- validates the output frame count with `ffprobe` when available;
- checks that ffmpeg exposes the required software encoder (`libx265` for
  H.265, `libx264` for H.264) before the save gate tries to materialize an
  episode, so missing host codec support fails with a clear error instead of an
  opaque ffmpeg failure;
- rewrites each sidecar to one row per final video frame, with
  `encoded_frame_index` renumbered from zero while preserving the original
  Libargus timestamps.

For example, if the sync window is:

```text
cam_06 encoded_frame_index [1, 19)
cam_07 encoded_frame_index [0, 18)
```

then both final videos are rewritten to exactly 18 frames.

### `argus_metadata_video_recorder.cpp`

Thor-only Libargus + GStreamer recorder. One process owns all selected cameras.
For each camera it creates two Argus streams from the same capture request:

```text
video stream    -> nveglstreamsrc -> nvv4l2h265enc/h264enc -> mux -> cam_XX.mkv
metadata stream -> FrameConsumer -> cam_XX.argus_frame_metadata.csv
```

Verified build command on Thor:

```bash
g++ -std=c++14 -O2 \
  -I/usr/src/jetson_multimedia_api/argus/include \
  -I/usr/src/jetson_multimedia_api/argus/samples/utils \
  $(pkg-config --cflags gstreamer-1.0 glib-2.0) \
  tools/thor/gmsl2/argus_metadata_video_recorder.cpp \
  /usr/src/jetson_multimedia_api/argus/samples/utils/ArgusHelpers.cpp \
  -L/usr/lib/aarch64-linux-gnu/tegra -lnvargus_socketclient \
  $(pkg-config --libs gstreamer-1.0 glib-2.0) \
  -lEGL -lGLESv2 -lpthread \
  -o /tmp/argus_metadata_video_recorder
```

Supported modes:

- `--frames N`: finite capture. The video branch uses `identity eos-after=N+1`
  on this Thor GStreamer stack so the muxed video contains exactly `N` frames.
- `--frames 0`: signal-controlled capture for UI start/stop. The process logs
  `recording started` only after the reference camera has entered a stable
  metadata window, and writes `argus_recording_markers.json` with
  `start_sof_tsc_ns`. The Python session waits for that marker before treating
  the episode as started. SIGINT/SIGTERM then stops the recording and drains EOS.
- A camera that does not deliver frame metadata no longer hangs the recorder:
  each metadata acquire has a finite timeout, and startup fails with a non-zero
  return code before the UI sees `recording started`.
- `--codec` selects H.264/H.265 encoder + parser; `--container` selects
  `qtmux`/`.mp4` or `matroskamux`/`.mkv`. The Python session passes both from
  YAML so the final video path and `meta.json` container field match.

### `argus_metadata_session.py`

Drop-in Python session wrapper with the same public API shape as
`PersistentCameraSession`:

```text
connect()
start_episode(ep_dir, idx)
stop_episode(handle)
discard_episode(handle)
poll_errors()
disconnect()
```

This lets the GUI keep spawning `thor_record.py` and keep using the same stdin
commands. The backend is selected by YAML:

```yaml
sensors:
  cameras:
    defaults:
      recorder_backend: argus_metadata
```

The wrapper auto-builds the C++ recorder on Thor when the binary is missing or
stale, runs a short real-recorder preflight during Connect, waits the YAML
`connect_stable_s` settling interval, then runs one recorder process per
episode. The preflight uses the same Libargus metadata consumer and video
encoder path as real recording, so a link-locked camera that does not deliver
metadata fails before the UI reports Connected.

The preflight is isolation-aware:

1. run the full selected camera set because production records all cameras in
   one Libargus-owned process;
2. if the recorder error names a camera, drop that camera and retry the
   survivors;
3. if the recorder cannot name a camera, probe each camera with the same
   recorder, drop only failing cameras, then retry the survivors together;
4. if every camera fails, or if the group fails but every camera passes alone
   and no failing camera can be isolated, Connect fails instead of hiding a
   resource/contention bug.

Dropped cameras are reported through `poll_errors()`, so `thor_record.py` emits
the existing GUI-facing `WARNING: ... stream(s) failed` and `Cameras (active):`
lines. Saved episode `meta.json` also records the same Connect result in:

- `active_camera_sids`: cameras that actually recorded the episode;
- `connect_stream_errors`: structured `{sid, name, message}` entries from
  Connect-time failures such as metadata-preflight drops;
- `connect_failed_sids`: union of initial Argus probe failures and
  Connect-time stream/preflight failures.

The wrapper also forwards the UI camera `name_prefix` into the C++ recorder.
The C++ output names therefore stay consistent with `thor_record.py`'s
`EpisodeHandle` names instead of assuming `cam_XX` forever.

Preview compatibility:

- the UI still sends `preview_demand` and reads
  `/dev/shm/lerobot_preview/<camera>.jpg`;
- while Connected but idle, `ArgusMetadataCameraSession` can spawn lightweight
  per-camera `gst-launch` preview processes. The preview pipeline uses
  `multifilesink location=/dev/shm/lerobot_preview/<camera>.jpg max-files=1`
  so GStreamer writes the current JPEG directly instead of routing concatenated
  JPEG bytes through Python stdout parsing;
- `start_episode()` always stops idle preview processes before launching the
  Libargus metadata recorder, so preview never competes with production
  episode capture;
- while an episode is recording, preview enable/refresh calls are no-ops; after
  the episode stops, the existing preview-demand loop can re-enable idle
  previews.

### `argus_frame_metadata_capture.cpp`

Thor-only Libargus metadata capture tool. It opens multiple Argus camera sessions from one process and writes the sidecars required by `argus_frame_sync.py`.

Build command verified on Thor:

```bash
g++ -std=c++14 -O2 \
  -I/usr/src/jetson_multimedia_api/argus/include \
  -I/usr/src/jetson_multimedia_api/argus/samples/utils \
  tools/thor/gmsl2/argus_frame_metadata_capture.cpp \
  /usr/src/jetson_multimedia_api/argus/samples/utils/ArgusHelpers.cpp \
  -L/usr/lib/aarch64-linux-gnu/tegra -lnvargus_socketclient -lpthread \
  -o /tmp/argus_frame_metadata_capture
```

Short Thor runtime check:

```bash
/tmp/argus_frame_metadata_capture --sids 6,7 --frames 5 --out-dir /tmp/argus_meta_check
```

It produced:

```text
/tmp/argus_meta_check/cam_06.argus_frame_metadata.csv
/tmp/argus_meta_check/cam_07.argus_frame_metadata.csv
```

The observed output confirmed the expected one-frame local-number offset:

```text
cam_06 encoded_frame_index=1 sof_tsc_ns=594078973576000
cam_07 encoded_frame_index=0 sof_tsc_ns=594078973563000
delta = 13000 ns = 13 us
```

## Remaining Production Work

The metadata-integrated path is now implemented end-to-end for the UI API:
Libargus capture, metadata sidecars, sync gate, and final video
materialization are all wired through `thor_record.py`.

Remaining work is validation and hardening, not a missing architecture piece:

- run longer 10/11-camera burn-in on Thor;
- measure save-time cost of ffmpeg materialization for full-length episodes;
- decide whether production should use H.265 (`libx265`) or switch the
  metadata backend to H.264 to speed up materialization/export;
- optionally replace ffmpeg materialization with a Jetson hardware
  decode/encode path if CPU transcode cost is too high.

## Verification Done In This Branch

Local:

```bash
python3 -m py_compile \
  tools/thor/gmsl2/argus_frame_sync.py \
  tools/thor/gmsl2/argus_video_materialize.py \
  tools/thor/gmsl2/argus_metadata_session.py \
  tools/thor/gmsl2/gmsl2_record.py \
  tools/thor/gmsl2/thor_record.py
```

Direct pure-Python smoke checks for the new sync/session/meta tests passed:
31 direct-import cases across
`test_thor_argus_frame_sync.py`,
`test_thor_argus_metadata_session.py`, and
`test_thor_record_meta.py`.

Project `pytest` was not runnable in the sandbox with system Python because `serial` is missing from the local environment. `uv run` attempted to download a Python runtime but network access is blocked.

Thor:

- `argus_frame_metadata_capture.cpp` compiled successfully.
- Short two-camera metadata run produced correct sidecars.
- `argus_metadata_video_recorder.cpp` compiled successfully.
- After adding the explicit `--container` option, the current
  `argus_metadata_video_recorder.cpp` was copied to Thor `/tmp`, compiled
  successfully, and `--help` showed `--container mkv`.
- Thor ffmpeg exposes both materialization encoders required by this branch:
  `libx264` and `libx265`.
- Finite 10-frame two-camera run produced 10 video frames per camera and 10
  metadata rows per camera.
- Signal-controlled run waited for `recording started`, stopped with SIGINT,
  and produced metadata that aligned to a common 18-frame window:
  - `cam_06` accepted encoded frame indices `[1, 19)`
  - `cam_07` accepted encoded frame indices `[0, 18)`
  - max SOF delta was `311000 ns`
- Thor ffmpeg materialization of that signal-controlled run produced:
  - `cam_06.aligned.mkv`: 18 frames
  - `cam_07.aligned.mkv`: 18 frames
- End-to-end `thor_record.py` UI-protocol smoke on Thor:
  - default YAML path used `recorder_backend: argus_metadata`; the smoke only
    narrowed `sensor_ids` to `[6, 7]`, disabled BOX, and shortened
    `episode_time_s`;
  - Connect built the C++ recorder and ran the real-recorder metadata preflight
    before reporting `Connected`;
  - stdin protocol matched the GUI path: wait for `Episode 0 ready`, send a blank
    line to start, then auto-save after the configured duration;
  - PWM was armed at 60 Hz and both cameras were set to `trig_mode=1`;
  - final saved videos had equal frame counts:
    - `cam_06.mkv`: 69 frames
    - `cam_07.mkv`: 69 frames
  - final sidecars had 69 rows per camera;
  - max cross-camera SOF delta was `11000 ns`;
  - `argus_frame_sync.ok=true`, `failures=[]`, and both videos were rewritten
    by the materialization step.
- Final two-camera UI-protocol smoke after the stop-marker change:
  - `cam_06.mkv` / `cam_07.mkv`: 64 frames each after materialization;
  - final sidecars: 64 rows each;
  - raw videos contained 71 / 75 frames before materialization;
  - `argus_recording_markers.json` included both `start_sof_tsc_ns` and
    `stop_sof_tsc_ns_exclusive`;
  - max cross-camera SOF delta was `11000 ns`;
  - `argus_frame_sync.ok=true`, `failures=[]`.
- Direct `ArgusMetadataCameraSession` idle-preview smoke wrote both JPEGs:
  `/dev/shm/lerobot_preview/cam_06.jpg` and `cam_07.jpg`.
- Multi-camera finite smoke on the currently locked Thor cameras:
  - locked ids at test time: `3,6,7,8,9,12,13,14`;
  - 8-camera run exposed a bad/no-frame `cam_03` despite link lock; after the
    finite-timeout fix the recorder exited with `rc=6` instead of hanging;
  - in that failed 8-camera run, only `cam_03` timed out waiting for metadata;
    the other 7 cameras each produced 60 frames before the episode was rejected;
  - rerun excluding `cam_03` (`6,7,8,9,12,13,14`) completed 60 requested frames;
  - all 7 videos had exactly 60 frames and all 7 sidecars had 60 rows;
  - max nearest-SOF delta against `cam_06` was 12-14 us, with only boundary
    startup drops before the common window.
- Connect-only metadata-filter smoke on Thor:
  - ran default `detect_all=true` with `--skip-argus-probe` to force the locked
    but bad `cam_03` into the metadata-integrated backend;
  - full-set metadata preflight failed and named `cam_03`;
  - `argus_metadata_session.py` dropped `cam_03`, retried the survivors, and
    connected 7 active cameras:
    `cam_06, cam_07, cam_08, cam_09, cam_12, cam_13, cam_14`;
  - UI-facing output included `WARNING: 1 stream(s) failed: cam_03(...)` and
    `Cameras (active): ...`;
  - saved episodes after such a Connect include `connect_failed_sids` /
    `connect_stream_errors`, so downstream checks can tell that the camera was
    intentionally dropped before recording;
  - no episode was saved in this smoke; it only verified Connect behavior.
- Temporary files under `/tmp` were removed after the check.

Not yet verified:

- UI on-demand idle-preview smoke after switching to `multifilesink`.
  The direct session preview path passed, but before the UI preview rerun could
  complete the Thor camera stack entered a lower-level driver failure state.
- Longer full 10/11-camera burn-in with production-length episodes.

Current Thor hardware blocker:

- after `recover_argus.sh --sdk tools/thor/gmsl2/sdk`, all locked camera probes
  failed;
- subsequent standalone `gst-launch-1.0 nvarguscamerasrc sensor-id=6/7 ...`
  produced 0-byte JPEG files and logged `nvbuf_utils: dmabuf_fd -1 mapped entry
  NOT found`;
- sudo `dmesg` showed `ar0234c ... i2c write failed`, `max96726 ... i2c-w,
  write failed`, and `Error turning on streaming`;
- this is below the Python recorder/metadata implementation and currently
  prevents clean UI preview rerun and full multi-camera burn-in until the Thor
  camera stack is reset outside this branch.
