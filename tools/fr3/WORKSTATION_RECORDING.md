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

The recorder opens the gripper after Connect and after every automatic `move_to_start`, before it
prints `Episode <n> ready`. On this rig `move_to_start` means the explicit `home` keyframe from
`fr3_pika_gripper.xml`, not the DAS/SDK default start. The operator should still visually confirm
the gripper is open before pressing StartEpisode; starting a take from a closed gripper records a
first frame whose gripper state does not match the task's starting condition, and on a grasp task
the policy learns that opening is something that happens before the data begins.

### Recording into a task

**Task Library** page → **New Task** (name, description, target episodes, `Dataset Repo ID`) →
**Go to Record**. The gateway then deep-copies `fr3_record_config.yaml` and patches only
`dataset.repo_id` / `dataset.root` / `dataset.single_task` from the task, writing
`outputs/.active_task_config.yaml` and spawning the recorder against that. The recorder itself
knows nothing about tasks — it takes a config path — which is why the page works identically on
both profiles and why the task's description becomes the episode's task prompt.

The binding is fixed at Connect (the dataset root is resolved once, when the recorder starts), so
the Live Record banner refuses to switch or unbind a task while a recorder is alive. Progress
counting matches the task's `Dataset Repo ID` trailing name against dataset directory names with
the session stamp stripped, so `local/pick_and_place` counts every
`pick_and_place_<YYYYmmdd_HHMMSS>` session.

Two limits worth knowing before planning a campaign around it:

- A sim session writes `<name>_sim_<stamp>`, which is not that task's name once the stamp is
  stripped, so **sim takes do not count toward a task's progress**.
- Each Connect is its own dataset, and the workstation's Dataset Export page builds a training
  view from *one* recording. Consolidating a task's sessions into a single view is the Thor
  route's "Consolidate a Task" and has no workstation equivalent yet; `fr3_train_il_policy.py`
  will merge a directory of dataset roots if you point `--dataset-root` at one.

Episode control maps onto the gateway's existing recorder protocol:

| UI | stdin | recorder behaviour |
| --- | --- | --- |
| StartEpisode | `\n` | begins recording; ends automatically after `dataset.episode_time_s` |
| Save | `save` | ends the episode early and keeps it |
| Discard | `n` | ends the episode early and drops it |
| Exit | `exit` | finalizes the dataset and shuts down |

### One session, one dataset

`dataset.root` names a *series*; the recorder appends `_<YYYYmmdd_HHMMSS>` to it, so every
**Connect** opens its own root — `fr3_spacemouse_20260731_143012`. Sessions stay separable, which
matters because the things that change between them (camera rate, gripper, lighting, a
firmware) appear nowhere in the schema. The gateway's episode counter strips that suffix back
off, so all of a series' sessions still attribute to one dataset name, and this is what the
docker recorder (`fr3_record.py`) and the Thor recorder have always done.

The consequence: **pressing Connect twice gives you two datasets, not one longer one.** A session
interrupted halfway does not continue into the next Connect. Two ways to extend one specific
dataset instead, both of which skip the stamp:

- set `resume: true` in the config, which names one dataset by definition, or
- point `dataset.root` at an already-stamped root, which is extended rather than stamped again.

Either way the recorder refuses to append episodes whose schema disagrees with the existing ones,
rather than producing a dataset that only fails at training time.

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

**The dataset has to have passed QC first**, the same gate the Thor route puts in front of its v3
export. On this profile the view *is* the export — it is the last step before a policy trains on
these frames, and nothing downstream looks at QC again. The timestamp-sync verdict in particular
only exists inside a QC run, so an ungated build was how a dataset whose modalities disagreed
reached training with its verdict sitting in a file nothing had opened. A `qc_warn` dataset can be
built with the warnings acknowledged in the confirmation window; a dataset whose QC never ran
cannot, because there is nothing to judge. The row on the Training View page carries the QC status
and a **Run QC** shortcut, so a disabled **Build View** says why without being pressed first.

The CLI above is deliberately not gated — it is the escape hatch for a merge across sessions
(`--dataset-root <parent dir>`), which the page cannot express. Run QC on each source first.

| contract | action features | rotation | reconstruction |
| --- | --- | --- | --- |
| `absolute_ee` | `ee.x/y/z` + `ee.qx/qy/qz/qw` + `gripper.pos` | quaternion | — |
| `delta_ee_from_prev_cmd` **(default)** | `delta_ee_from_prev_cmd.dx/dy/dz` + `.drx/dry/drz` + `gripper.pos` | rotvec | `prev_cmd_pose ∘ delta` |
| `delta_ee_from_current` | `delta_ee_from_current.dx/…` | rotvec | `current_ee_pose ∘ delta` |

The reference is part of the feature *name*, so a view is self-describing: an offline tool can
tell from the column names alone how to integrate it back. Videos are symlinked (`--copy-videos`
is off by default), so a view costs ~1% of the source's disk — measured: video is 99% of a
dataset.

#### What the view leaves out

The build drops every episode marked **not for training** in Episode Replay's annotation panel
(`includeInTraining` in `meta/gui_annotations.json`), and the Training View page shows the count
and the episode numbers on the row before you press Build View. Surviving episodes are renumbered
contiguously — LeRobot addresses episodes by position, so a gap would be a dataset claiming
episodes it does not have — and `meta/il_view_manifest.json` records both `excluded_episodes` and
an `episode_source_index` map so a view's episode 4 can still be traced back to the recording.

The recording is never modified: videos are symlinked whole and each surviving episode keeps its
own `from_timestamp`/`to_timestamp` range inside them. Deleting an episode in Episode Replay is
still the destructive option; this one is a filter, and it is reversible by unchecking the box and
rebuilding.

That flag used to be inert. It was written to the annotation store and read by nothing, so an
operator who reviewed a session and marked three bad takes still trained on them — the only
exclusion that worked was deletion. `--exclude-episodes 3,7` does the same thing from the command
line for a single-source build, and `--no-respect-annotations` builds the unfiltered view.

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

#### Rollout on this rig

```bash
FR3_INFER_CHECKPOINT=outputs/train/<job>/checkpoints/last \
  bash tools/fr3/run_pick_place_infer_workstation.sh smoke     # then preview, then real
```

There is no GUI entry point; rollout is a terminal step. `run_pick_place_infer_host.sh` is the
*other* FR3 rig (Hikrobot cameras, DAS/Corenetic gripper, 192.168.11.102) and cannot be pointed
here with environment variables, because three of its settings do not mean the same thing on this
hardware:

- **Tool frame.** The runtime defaults a Pika gripper to `pika_gripper_ee`; this rig records
  against `pika_task_tcp` (`fr3_record_config.yaml`). Both are fixed frames on the same URDF, about
  0.4 m apart. Left at the default, the rollout runs, tracks its targets, and is wrong by that
  offset everywhere.
- **Gripper units.** Here `gripper.pos` is a normalized 0..1 opening; on the Hikrobot rig the same
  column is a Pika width in millimetres. That script's `--gripper-close-below 70` would clamp every
  step of a normalized policy to fully closed.
- **Cameras.** The checkpoint asks for `observation.images.ee` / `.side` by name, so the camera
  config has to key its two RealSense units as `ee` and `side` —
  `tools/fr3/fr3_il_infer_realsense_camera_config.yaml`, matched by serial, at the recorder's 60 Hz
  so the images are as fresh at deployment as they were in training.

`fr3_act_infer_real_runtime.py` now resolves gripper units from the dataset feature name before
falling back to the legacy value heuristic. For this workstation's `gripper.pos` contract, values
like `0.08` stay normalized instead of being misread as `0.08 m`; for unit-bearing legacy features
such as `*.width_mm` the runtime still converts through `gripper_max_width_mm`.

`FR3_GRIPPER_CLOSE_BELOW` is disabled by default in the workstation rollout launcher. Use it only
as a deliberate task-specific binary close guard, not as a unit workaround.

The script never uses `--move-to-das-start` — that homes to a joint configuration belonging to the
DAS rig. `T_B_Ws` is solved from the first observation against the dataset's start pose, so the
arm's start pose is what places the whole trajectory in the workspace: it has to be the pose the
episodes were recorded from. The recording contract is the explicit `home` keyframe in
`fr3_pika_gripper.xml`: `0 -0.785 0 -2.355 0 1.57079 0.785` for the arm joints.

Tuning knobs the host script ships values for — `n_action_steps`, ACT temporal ensembling, command
EMA, controller gains — are left unset here. Those numbers were measured on the other rig's arm,
tool and task; the checkpoint's and driver's own defaults are the honest baseline to tune away
from. The MuJoCo replay gate is still the thing to clear first, but note what it does *not* cover:
it scores the recorded EE stream through IK, never the policy's output.

## Timestamp-synchronisation audit

Every saved episode is audited immediately from the in-memory frame buffer — a LeRobot v3
dataset keeps one parquet file open until `finalize()`, so an episode cannot be re-read the
moment it is saved. The verdict appears under the record controls and in the event log. At
session end the same audit runs against the finalized files and is persisted to
`meta/fr3_sync_report.json`.

Standalone: `python tools/fr3/fr3_sync_audit.py --dataset <root> --camera-fps 60
[--fail-on-violation]`. The verdict is also a QC check (`timestamp_sync` on the Dataset
Processing page), so a dataset whose alignment failed cannot reach `qc_pass` and therefore cannot
be exported. QC recomputes the report when it is missing (an interrupted session never reaches
`finalize()`) or when it predates schema v3.

### What `status` rests on

**Not the spread across all four devices.** That number is 27 ms on a *healthy* episode here,
because the cameras sit a constant 23 ms behind the arm read at 60 fps — a frame already exists
when the loop asks for it, while the arm is read on demand. Judged against the 20 ms budget it
failed **268 of 300 frames** of a take whose two cameras agreed to 3.8 ms and whose cadence was
exact to 0.03%. A constant reported as a defect on every episode is the same as no signal at all,
so the verdict is now the conjunction of four budgeted quantities (all in `skew_evaluation`):

| quantity | what it catches | budget | measured on a healthy hardware take |
| --- | --- | --- | --- |
| `within_group` skew (cameras vs each other) | one camera falling behind; the handover-stamping bug that put a fake 24 ms between two views of one instant | 20 ms (`--tolerance-ms`) | p50 3.8, max 8.3 ms |
| `residual` skew (all devices, each one's median offset removed) | an offset that *drifts* inside a session, which a constant-bias check cannot see | one camera frame period + 20 ms → 36.7 ms at 60 fps (`--camera-fps` / `--residual-tolerance-ms`) | p95 12.8, max 19.1 ms |
| `bias` (each device's median offset from the arm) | a *changed* pipeline: a swapped camera, a stamping regression, a second clock spliced in | 60 ms (`--bias-tolerance-ms`) | cameras −23.3 / −22.8 ms, gripper +2.3 ms |
| `grid_lag` | the control loop missing the dataset's own cadence | 50 ms | p95 3.9 ms |

The residual's floor is one sensor period: nothing triggers these cameras, so the phase between
acquisition and the loop tick wanders over a full frame period and lands entirely in this number.
That period is *not* in the dataset (`dataset.fps` is 30, the sensors run at 60), so the budget
comes from the config at record time or from `--camera-fps`; without either, the residual is
measured and reported with no verdict rather than judged against an invented threshold.

The raw all-device spread is still in the report — `skew_evaluation.raw_all_device` and the
untouched `summary.max_skew_s` — because it is the honest answer to "how far apart are these
columns". It is simply not a pass/fail question on a rig with real pipeline offsets. Report schema
is `3`; a v2 report's `status` was produced by the old rule and is not comparable.

What the other numbers mean:

- `grid_lag_p95_ms` — drift of the **arm read** away from the dataset's nominal
  `frame_index / fps` grid, as p95 of `|lag|`. Budget: 50 ms. The arm is the reference because it
  is the one modality read *for* the frame rather than delivered to it, so it carries no pipeline
  delay. It used to be measured against the median across devices, which charged the cameras'
  honest 25 ms latency to the loop: 13.5 ms of reported grid lag for a cadence that was exact to
  0.03%. Camera latency is not lost, it is reported as `bias_vs_arm_ms` instead.

  It is not the loop tick exactly: the arm column is the instant the 200 Hz state reader sampled,
  so it trails the tick by up to one 5 ms poll period and this metric carries that as noise. That
  is the right trade — the column has to mean when the value was *sampled* for every other number
  here to mean anything.
- `interval_ms=A/Bnominal` — the cadence the control loop actually delivered vs. the cadence the
  dataset claims. **`A > B` means the recorded frame spacing is a fiction**: the dataset labels
  frames as evenly spaced at `1/fps` while they were captured further apart. Lower
  `dataset.fps`, or cut per-frame work, and re-record. Do not train on that cadence as-is.
  `A` is the episode's elapsed time divided by its frame count, *not* a median of per-frame
  gaps — jitter is asymmetric (a late frame is followed by an early one), so the median gap
  reads high and condemns a cadence that was in fact exact. On the hardware rig the median gap
  was 35.4 ms where the true average was 33.34 ms against a 33.33 ms nominal.
- `bias_vs_arm_ms[...]` — each modality's median offset from the arm read, as the median of the
  per-frame differences. A *constant* bias is a fixed pipeline offset: it is budgeted against a
  wide regression tripwire (above) and reported, never silently subtracted from the data.

`clock_semantics` says which clock produced the timestamps, because the two backends do not
mean the same thing by them:

- `hardware_mixed` — every column is host `perf_counter`, and every one means **when the value was
  sampled**, not when `get_observation()` picked it up.

  The arm column is the instant its 200 Hz state reader took the state off the arm — the driver
  serves a cache, so stamping the pickup would have claimed up to a poll period of freshness that
  did not exist. The gripper column depends on the backend, which is why the backend is in the
  column name (`<backend>_gripper.capture_timestamp_s`): `franka_hand` (10 Hz poll, so up to
  100 ms at stake) and `das` (stamped in its databus callback) report their sampling instant;
  `pika` — the shipped backend — and `corenetic` read on demand and expose none, so those columns
  are the read instant, an upper bound rather than a guess. A `corenetic` sample does carry its
  own timestamp, but on the BOX MCU's clock; using it here would splice two time bases with no
  measured offset between them.

  Camera columns are the **acquisition** instant: the sensor reports it on the device clock,
  global time maps that onto the host wall clock, and the camera subtracts the frame's age at
  handover to put it on the same monotonic basis. Not the exposure midpoint — it is what the
  sensor calls acquisition.

  Cameras are older than the arm read by construction: the arm is read on demand, while a frame
  already exists by the time anything asks for it. Measured on the rig at 30 fps, images sit
  **42–45 ms** behind the arm (stable to ~4 ms) and the two cameras are within **~3 ms** of each
  other. That first number is a real image-vs-state offset — it halves at 60 fps — not a clock
  error, so it is reported and never subtracted.

  Camera columns used to be stamped when the background loop finished post-processing the frame.
  That carried each camera's pipeline delay (a D405 hands over 4.8 ms after acquisition, a D435i
  29.1 ms), which put a fabricated 24 ms between two cameras seeing the same instant and made
  the images look 25 ms fresher than they were.
- `sim_extraction_wallclock` — MuJoCo extracts every modality from one physics instant, so
  there is no acquisition skew to measure. These timestamps record extraction cost (state read,
  then one render per camera). Useful for catching a straggling render; **not** comparable to
  hardware sensor timestamps.

### Which frame each camera contributes

Every camera is anchored on the **oldest** of the cameras' latest frames and asked for its frame
closest to that instant. This is what bounds cross-camera skew, and the bound is what makes the
guard below satisfiable.

The cost is staleness: both cameras end up as old as the slowest one, measured 25 ms behind the
arm read at 60 Hz. Serving each camera its own newest frame instead measures 8.5 ms — half a
frame period, the floor for a sensor nothing triggers — and **was tried on hardware, where it
aborted an episode after 21 frames** with 25.1 ms of skew. Nothing bounds the gap between two
cameras' newest frames: their background threads deliver independently, and one falling a whole
period behind opens a gap past any guard worth keeping.

Worth recording why the offline analysis missed that, because the same trap is easy to walk into
again: the comparison reconstructed each camera's newest frame as `selected + k × period` from
already-recorded episodes. That arithmetic is exact, but every `selected` in it came from the
anchored strategy, so the reconstruction inherited the very regularity anchoring imposes and
predicted a 16.2 ms worst case. Thread scheduling and dropped frames are invisible in data that
anchoring already cleaned up — a strategy cannot be evaluated on data produced by its
alternative.

The hardware robot can fail a frame outright when its cameras disagree by more than
`camera_max_skew_ms` (20 ms), and replay/inference keep that strict default. Workstation
recording sets `camera_skew_hard_fail: false`: the capture timestamps are still recorded, the
episode-level SYNC audit prints warnings, and the QC gate decides whether the take may reach
training. This avoids losing the recorder process to one jittery camera frame while still
preventing bad data from passing unnoticed. The MuJoCo robot keeps the hard guard on its render
pass.

## Replay

**Episode Replay** page → select dataset and episode → **MuJoCo replay**.

The recorded absolute-EE action stream is fed back through the same
`FrankaResearch3Mujoco` robot and scored against what the simulated arm reached. An unscored
approach phase first drives to the recorded start pose, so frame 0 is not judged against a
homing move. Passing thresholds are 20 mm / 15°.

There is no cube selection on this profile. The Thor route picks between AprilTag cubes and
replays `state_action.<cube>.csv`; a workstation dataset has no cubes and no such sidecar, so the
gateway drops the cube mode for it and the Episode Replay page hides the picker along with the
saved-report **Pass MuJoCo check** — that button re-reads a `mujoco_preview.<cube>` file this
route never writes. The verdict here is settled automatically: the runtime reports on the
`mujoco_replay_result=` line and the gateway resolves pass/fail when the process exits.

### Real-robot replay

`--backend real` on the *same* runtime: the hardware `FrankaResearch3` from the same config,
driven by the same reconstructed trajectory, scored the same way. Sharing the runtime is the
point — a real replay that rebuilt its trajectory by different code from the run that cleared it
would have validated one thing and executed another. Thor keeps its cube-sidecar path
(`third_party/opencv_kalibr/.../replay_cube_pose_in_robot_base.py`); nothing is shared between
them but the panel.

What differs on hardware, and only what must:

- Nothing is rendered, and the cameras are not opened — the RealSense units belong to the live
  monitor the operator is watching the arm through, and this loop never reads an image.
- The replay is paced at the dataset frame rate, always. Streaming a 30 Hz trajectory as fast as
  the loop can issue it would ask the arm for motion the recording never made.
- **The approach phase refuses instead of warning.** In sim, failing to reach the recorded start
  pose is logged and the episode is scored anyway. On hardware it aborts before a single
  trajectory frame is sent: frame 0 of a trajectory whose start the arm never reached is a step
  of unknown size at full command rate, and the thing that absorbs it is the hardware.
- The result lands in `derived/fr3_real_replay/episode_N.json`, never over the sim report that
  authorized the run.

The MuJoCo pass is still required in front of it, and that is deliberate: it is the only check
that puts the recorded EE stream through IK before an arm executes it, and it is complementary to
the hardware preflight, which vets the arm but never looks at the trajectory. A validation that
ran and *failed* can be overridden in the confirmation window, with its errors in front of you; a
validation that never ran cannot, because there is nothing to judge.

The MuJoCo verdict does not touch `safety`, which describes only whether the hardware path is
authorized — that is the real preflight's answer to give. A failed score therefore leaves the
robot-free controls alone instead of presenting the rig as faulted.

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
