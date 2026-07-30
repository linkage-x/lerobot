# FR3 workstation recording, replay, and timestamp-sync audit

Deploy chain: `bash run/deploy.sh workstation` syncs the repo to `hph@192.168.100.155`,
restarts the gateway with `--profile workstation --config-path tools/fr3/fr3_record_config.yaml`,
and serves the frontend at <http://localhost:5173/>.

## Recording

**Live Record** page → pick a backend → **Connect** → **StartEpisode**.

| Backend | Robot | Cameras | Notes |
| --- | --- | --- | --- |
| `Real FR3` | hardware over FCI | the two RealSense units from the config | RealSense teleop preview is stopped first (the recorder needs those devices) |
| `MuJoCo Sim` | `franka_research3_mujoco` | rendered from the MJCF scene | headless EGL; writes to `<root>_sim` under a `<repo_id>_sim` id |

Both backends run the *same* `record_loop` with the *same* ee2ee processor pipeline, so the
two datasets are byte-identical in schema — feature names, ordering, shapes, and the
`observation.device_capture_timestamp` column. That is the point: a sim-trained policy needs no
translation layer to meet hardware data.

The SpaceMouse is the only device that is physically opened in both backends. In a sim session
the arm/gripper/camera rows stay `idle` ("simulated in MuJoCo") rather than pretending to be
connected.

Episode control maps onto the gateway's existing recorder protocol:

| UI | stdin | recorder behaviour |
| --- | --- | --- |
| StartEpisode | `\n` | begins recording; ends automatically after `dataset.episode_time_s` |
| Save | `save` | ends the episode early and keeps it |
| Discard | `n` | ends the episode early and drops it |
| Exit | `exit` | finalizes the dataset and shuts down |

Pressing **Connect** again on an existing dataset root resumes it (episodes accumulate) after a
schema-compatibility check, instead of failing on the root already existing.

## Action contract

**Recording always stores the absolute EE contract**: `action` = `ee.x/y/z` + `ee.qx/qy/qz/qw` +
`gripper.pos`. Delta contracts are produced later, by the Training View step.

### Why delta is derived offline, not at capture

A delta's magnitude is tied to the interval it spans, so it must span the interval at which the
policy will be queried -- one dataset frame. The teleop pipeline runs at `control_fps` (200 Hz)
while frames are captured at `dataset.fps` (30 Hz), and a processor step in that pipeline cannot
tell which of its ~6.7 invocations per frame is the recorded one. Computing the delta at capture
time therefore stored a *one-control-tick* increment against a per-frame grid -- measured 0.5 mm
recorded where the command had actually advanced 1.0-1.5 mm, i.e. a policy trained on it would
drive the arm ~6.7x too slow. Differencing consecutive dataset frames offline is exact by
construction.

Nothing is lost: the absolute action plus `observation.state.prev_cmd.ee.*` determine the delta
exactly, and the absolute command stream stays the source of truth. A delta definition can also
be fixed or swapped without re-recording on hardware.

### Training View

**Dataset Export** page (workstation profile) → pick a contract → **Build View**. Or:

```bash
python tools/fr3/fr3_train_il_policy.py \
  --dataset-root outputs/datasets/<name> \
  --cameras observation.images.ee,observation.images.side \
  --action-mode delta_ee_from_prev_cmd --prepare-only
```

| contract | action features | rotation | reconstruction |
| --- | --- | --- | --- |
| `absolute_ee` | `ee.x/y/z` + `ee.qx/qy/qz/qw` + `gripper.pos` | quaternion | — |
| `delta_ee_from_prev_cmd` **(default)** | `delta_ee_from_prev_cmd.dx/dy/dz` + `.drx/dry/drz` + `gripper.pos` | rotvec | `prev_cmd_pose ∘ delta` |
| `delta_ee_from_current` | `delta_ee_from_current.dx/…` | rotvec | `current_ee_pose ∘ delta` |

The reference is part of the feature *name*, so a view is self-describing: an offline tool can
tell from the column names alone how to integrate it back. Videos are symlinked (`--copy-videos`
is off by default), so a view costs ~1% of the source's disk — measured: video is 99% of a
dataset.

`prev_cmd` is the default because the reference is the previous *command*: the arm's tracking lag
stays out of the action, and a held frame is an exact zero. `delta_ee_from_current` references the
*measured* pose, so every action carries the rig's tracking residual — on the MuJoCo rig that
showed up as an implied p95 speed of 587 mm/s versus 15 mm/s for `prev_cmd`. The manifest's
`per_frame_scale` block is there to make that visible before you train on it.

Conventions, both delta modes:

- **translation is world-frame**: `delta = desired_pos − reference_pos`
- **rotation is body/tool-frame, right-multiplied**: `desired_R = reference_R @ delta_R`
- **gripper stays absolute** (0..1 opening); a gripper delta would accumulate drift with no
  reference in the observation to correct against.

Episode boundaries are never crossed — a delta spanning the seam would encode a `move_to_start()`
homing move as an operator command.

### Why rotvec and not a quaternion

A quaternion is required for *absolute* orientation (it covers all of SO(3); a rotvec aliases at
θ = π). A per-frame delta is ~1 mm / 0.01 rad, three hundred times away from that singularity, and
there the quaternion is actively worse:

- `qw = cos(θ/2)` spans only `1.25e-5`, about **8 bits** of float32 (a rotvec component gets ~24).
  After training-time normalisation that dimension is mostly quantisation noise.
- Recovering the angle via `acos` near `qw≈1` amplifies error: `1e-3` on `qw` turns 0.010 rad into
  0.090 rad. On a rotvec the same error is 1:1.
- 4 components carry 3 degrees of freedom, and a regressor does not honour ‖q‖=1.
- Action chunking averages actions; rotvec averages linearly at small angles, quaternions need slerp.

(The no-redundancy quaternion variant — store `qx,qy,qz`, recover `qw` from the unit constraint —
is `rotvec/2` to first order, so the quaternion adds no information here.)

### Deployment

`DeltaEEToAbsoluteEEAction` is the single reconstruction, shared by the transform's self-check and
by inference. `fr3_act_infer_real_runtime.py` reads the contract off the checkpoint's dataset action
names (`build_delta_action_reconstructor`) and rebuilds in the **dataset frame**, before the
dataset→base→E conversions. Feeding it a delta checkpoint without the reference observation raises
rather than guessing.

Every transform self-checks by rebuilding the absolute stream and comparing against the source
(tolerance 1e-5 m / 1e-2 deg); the result is recorded in `meta/il_view_manifest.json`. That check
is what would have caught the capture-time rate bug — the MuJoCo replay gate could not, because it
compares commanded vs *achieved*, not commanded vs *recorded-commanded*.

## Timestamp-synchronisation audit

Every saved episode is audited immediately from the in-memory frame buffer — a LeRobot v3
dataset keeps one parquet file open until `finalize()`, so an episode cannot be re-read the
moment it is saved. The verdict appears under the record controls and in the event log. At
session end the same audit runs against the finalized files and is persisted to
`meta/fr3_sync_report.json`.

Standalone: `python tools/fr3/fr3_sync_audit.py --dataset <root> [--fail-on-violation]`.

What the numbers mean:

- `skew_*_ms` — spread *within* one frame across devices. Budget: 20 ms (`--sync-tolerance-ms`).
- `grid_lag_p95_ms` — drift of the real capture times away from the dataset's nominal
  `frame_index / fps` grid. Budget: 50 ms.
- `interval_ms=A/Bnominal` — the cadence the control loop actually delivered vs. the cadence the
  dataset claims. **`A > B` means the recorded frame spacing is a fiction**: the dataset labels
  frames as evenly spaced at `1/fps` while they were captured further apart. Lower
  `dataset.fps`, or cut per-frame work, and re-record. Do not train on that cadence as-is.
  `A` is the episode's elapsed time divided by its frame count, *not* a median of per-frame
  gaps — jitter is asymmetric (a late frame is followed by an early one), so the median gap
  reads high and condemns a cadence that was in fact exact. On the hardware rig the median gap
  was 35.4 ms where the true average was 33.34 ms against a 33.33 ms nominal.
- `bias_vs_arm_ms[...]` — each modality's median offset from the arm read. A *constant* bias is
  a fixed pipeline offset and is reported, never silently subtracted.

`clock_semantics` says which clock produced the timestamps, because the two backends do not
mean the same thing by them:

- `hardware_mixed` — every column is a host `perf_counter` read, but not of the same event.
  Arm and gripper are stamped when their driver read returns, inside `get_observation()`.
  Camera columns are stamped by the camera's background read loop *after* it has converted and
  post-processed the frame — so they are neither exposure midpoint nor driver handover, and
  they carry the conversion cost. They are also older than the arm read by construction: the
  loop hands over the most recent frame it already has. Measured on the rig, cameras sit
  5.7 ms (`ee`) and 17.3 ms (`side`) ahead of the arm, each stable to ~2 ms, with `side`
  drifting against `ee` at ~0.35 ms/s (two free-running 30 fps sensors beating). Cross-device
  skew is therefore real but dominated by a fixed per-camera pipeline offset, not by jitter.
- `sim_extraction_wallclock` — MuJoCo extracts every modality from one physics instant, so
  there is no acquisition skew to measure. These timestamps record extraction cost (state read,
  then one render per camera). Useful for catching a straggling render; **not** comparable to
  hardware sensor timestamps.

The hardware robot additionally fails a frame outright when its cameras disagree by more than
`camera_max_skew_ms` (15 ms); the MuJoCo robot applies the same guard to its render pass.

## Replay

**Episode Replay** page → select dataset and episode → **MuJoCo replay**.

The recorded absolute-EE action stream is fed back through the same
`FrankaResearch3Mujoco` robot and scored against what the simulated arm reached. An unscored
approach phase first drives to the recorded start pose, so frame 0 is not judged against a
homing move. Passing thresholds are 20 mm / 15°, and a pass is what unlocks real-robot replay.

Standalone:

```bash
python tools/fr3/fr3_gui_replay_runtime.py \
  --dataset outputs/datasets/<name> --episode 0 \
  --config-path tools/fr3/fr3_record_config.yaml
```

Reports land in `derived/fr3_mujoco_replay/episode_XXXXXX.json`.

## Known constraints

- `save_episode(parallel_encoding=False)` is deliberate. The multi-camera parallel path forks a
  `ProcessPoolExecutor` from a process already holding the MuJoCo/EGL context, camera driver
  threads, and the stdin reader; that fork deadlocks. Sequential encoding costs a few hundred
  ms between episodes. `streaming_encoding: true` (the shipped config) avoids the path entirely.
- Sim camera resolution is one renderer for all cameras, so `--backend sim` refuses a config
  whose cameras have different resolutions rather than recording one of them at the wrong size.
