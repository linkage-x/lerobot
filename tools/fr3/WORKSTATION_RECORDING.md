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

### Moving the return pose: Set Home / Reset Home

**Live Record** page, next to the episode controls, on the workstation profile only:

- **Set Home** captures the arm's current joints as the pose `move_to_start` returns to between
  episodes. Use it when a task needs to begin somewhere other than the `home` keyframe.
- **Reset Home** puts it back to the pose the recorder was launched with — the config's
  `robot.start_joint_positions`, which
  `tests/robots/test_fr3_home_keyframe_contract.py` holds equal to the `home` keyframe.

Both only touch the *running recorder's* config object; neither writes
`fr3_record_config.yaml`, so a fresh Connect always starts from the file again. That is what makes
Reset Home an exact undo rather than a second guess at what "home" means: it restores whatever that
process read at startup, so a rig that legitimately declares a different start pose resets to its
own. A config that declares no start pose resets back to *none*, meaning the arm backend's own
start pose applies again — the recorder says so in the log rather than substituting a joint vector.

Neither control does anything on the MuJoCo backend, which has no capture: the recorder emits a
`WARN` and the session continues. And note that changing the return pose mid-collection splits the
corpus the same way any other recording-contract change does — episodes recorded before and after
begin from different places, and `T_B_Ws` is solved against the dataset's start pose at rollout.

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

- **Tool frame.** Two fixed frames on the same URDF, 410.85 mm apart with identical orientation.
  Naming the wrong one does not fail: the rollout runs, tracks its targets, and is wrong by that
  offset everywhere. The frame must follow the *checkpoint's dataset*, which is normally the same
  thing as the record config — and is: both are `pika_gripper_ee` since the switch below. The two
  only diverge if you train on episodes recorded before it, in which case export
  `FR3_TARGET_FRAME_NAME=pika_task_tcp` for that rollout.
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

`--move-to-das-start` is now **off by default** in the runtime — it used to be on, and it homes to a
joint configuration belonging to the DAS rig. `T_B_Ws` is solved from the first observation against
the dataset's start pose, so the arm's start pose is what places the whole trajectory in the
workspace: it has to be the pose the episodes were recorded from. The recording contract is the
explicit `home` keyframe in `fr3_pika_gripper.xml`: `0 -0.785 0 -2.355 0 1.57079 0.785` for the arm
joints, checked against the recorder's and the launcher's copies by
`tests/robots/test_fr3_home_keyframe_contract.py`.

The launchers still pass `--no-move-to-das-start` explicitly. That is now redundant, and kept so:
the flag is what a reader greps for, and leaving it means the launcher does not silently change
behaviour if the default is ever flipped back.

Tuning knobs the host script ships values for — `n_action_steps`, ACT temporal ensembling, command
EMA, controller gains — are left unset here. Those numbers were measured on the other rig's arm,
tool and task; the checkpoint's and driver's own defaults are the honest baseline to tune away
from. The MuJoCo replay gate is still the thing to clear first, but note what it does *not* cover:
it scores the recorded EE stream through IK, never the policy's output.

#### Which frame is the tool, really

`pika_task_tcp` is documented in the URDF as "midpoint between the two finger working points". It
is not. Measured against the finger meshes in the model's own `gripper_base` frame
(`tests/robots/test_fr3_tool_frame_geometry.py`):

| frame | position in `gripper_base` | distance to the finger working-point midpoint |
| --- | --- | --- |
| finger working-point midpoint | `(0.1883, 0, 0.0006)` | — |
| `pika_gripper_ee` | `(0.185, 0, 0)` | **3.4 mm** |
| `pika_task_tcp` | `(0, 0, 0.366842)` | **411.8 mm** |

`pika_gripper_ee` is the tool point. `0.366842` is the fingertip reach measured in
`quest3_pika_gripper_scene.xml` — a free-flying wrist frame whose `gripper_base` carries the pika
mesh 0.1765 m up its own **+z**, so there the tool axis is +z and `0.366842` lands on the far
finger's furthest vertex (measured `0.366492`, 0.35 mm out). Not micron-exact — that scene's two
jaws are not mirror-symmetric about the frame, reaching `0.354676` and `0.366492` — but unambiguous
about which frame it was taken in.
In the arm-mounted model `gripper_base` *is* the pika mesh frame and the tool axis is **+x**, so the
same literal points across the gripper instead of along it. The 45° z-rotation that travelled with
it was already recognised as frame-dependent and removed; the translation is from the same source
and was not.

`pika_task_tcp` is what this rig recorded against until the switch below, and it is a *rigid*
frame — so recording, MuJoCo replay and rollout stayed mutually consistent as long as all three
named it, and the episodes collected under it are not garbage. Do not "fix" the URDF: that silently
reinterprets every episode already collected, in either frame. What the 0.41 m lever cost, measured
at the `home` keyframe:

- **Rotation leaks into translation.** `fr3_joint7` is the tool roll axis: +1° moves the real tool
  **0.03 mm** and moves `pika_task_tcp` **6.43 mm**. Teleop and encoder jitter about that axis
  enters `observation.state` as translation the policy has to learn to ignore.
- **The step clamps are looser than they read.** They bound the *target frame*, so the rotation
  allowance buys extra tool motion through the lever: `--max-step-*` at 3 mm / 2° permits up to
  **17.3 mm** at the tool, and `--first-frame-max-*` at 20 mm / 8° permits up to **77.4 mm**.
- **`workspace_min`/`workspace_max` do not bound the tool**, only the offset frame. This is the
  one the switch below actually fixes rather than merely shrinks: the box is now derived from the
  tabletop and applies to the fingertips.

#### The switch to `pika_gripper_ee` — applied

**Applied, for a task that reorients the tool.** `fr3_record_config.yaml` now records against
`pika_gripper_ee`, `workspace_min`/`workspace_max` are declared there and re-derived at the tool
point, and `scale_wx`/`scale_wy` are no longer held at zero. The three are guarded together by
`tests/robots/test_fr3_recording_workspace_contract.py`, which fails if any one of them moves
without the others — including a revert, which the home-clearance check catches on its own.

The measurement further down said the lever was *dormant* for pick-and-place — 0.62 mm p95 per step
against a 3.9 mm real tool step, 1.02x path inflation — and on that evidence switching bought that
task nothing. A task that reorients the tool is a categorically different case, worked out under "A
task that reorients the tool" below: there the lever is not noise, it is where the command pivots.

What the switch deliberately does **not** do: it does not touch a recorded episode. Datasets
collected before it are anchored to `pika_task_tcp` and are neither replayable nor trainable against
this config until they are converted (exact recipe at the end of this section).

Everything that *launches* something did move with it, including the rollout launcher. That one is
normally the exception — a rollout has to meet the frame its checkpoint was trained in, not the
frame the recorder happens to use today — but nothing had been trained when the switch happened
(no `outputs/train` on either machine), so there was no checkpoint for the old default to protect.
The exception reappears the moment someone trains on pre-switch episodes; `FR3_TARGET_FRAME_NAME`
is the escape hatch and the launcher's own comment says so.

**Why you would switch.** `pika_gripper_ee` *is* the tool point, and the real-robot config class
already defaults to it (`config_franka_research3.py`) — only this rig's YAML overrides it. In its
favour:

- Rotation stops leaking into translation. The processor rotates the target about *its own origin*,
  so today a pure reorientation swings the fingertips through an arc of radius 0.41 m; teleop and
  encoder jitter about the roll axis enter `observation.state` as translation the policy has to
  learn to ignore (`fr3_joint7`: +1° = 0.03 mm at the tool, 6.43 mm at `pika_task_tcp`).
- Every safety number starts bounding the thing it names — the clamps, the workspace box, and the
  rollout's first-frame and per-step gates all stop being ~5–6x looser at the tool than they read.
- The recorded state becomes physically interpretable: "where the fingers are", not "where a point
  0.41 m past the fingers is".

**What it costs.** For pick-and-place alone none of this would have been worth paying:

- Part of it is a hardware-in-the-loop re-tune, not a config edit. The teleop gains were tuned with
  the lever present and cannot be re-tuned from a text editor.
- Operators have to relearn the feel: rotation used to drag the tool through an arc, and now it
  does not.
- Every dataset that is not migrated is unmixable with the new ones — a lasting split in the corpus
  for as long as both exist. This is the real price, and it is why the switch had to happen *before*
  the rotation task was recorded rather than after.
- **Measured on pick-and-place, it buys almost nothing.** The cost scales with the lever *times the
  rotation actually used*, and that task barely rotates the wrist. See below.

**What changed together.** `target_frame_name` reaches further than the recorder, and everything
that names it has to agree — a dataset recorded in one frame and replayed or rolled out in another
is wrong by 411 mm everywhere, with no error message:

| where | value | note |
| --- | --- | --- |
| `fr3_record_config.yaml` `target_frame_name` | **`pika_gripper_ee`** | changed. The recorder, real teleop, sim teleop and both replay paths all read this one key |
| `fr3_record_config.yaml` `workspace_min`/`max` | **`[0.18, -0.45, 0]`–`[0.70, 0.45, 0.70]`** | added. It was falling through to `FrankaResearch3Config`'s `(0.2,-0.6,0.05)`–`(0.9,0.6,0.8)`, which was never derived for this rig and, at the tool point, describes a region 411 mm from where it reads |
| `fr3_record_config.yaml` `scale_wx`/`scale_wy` | **removed** | changed. Roll and pitch follow `rotation_scale` like every other axis; they were pinned to `0.0` only because of the lever |
| `run_pick_place_infer_workstation.sh:46` | **`pika_gripper_ee`** | changed. The rollout frame must match the *checkpoint's* dataset, so this default only moved because there was nothing to protect: no `outputs/train` exists on either machine, so no checkpoint predates the switch. Export `FR3_TARGET_FRAME_NAME=pika_task_tcp` if one is ever trained on pre-switch episodes |
| gateway sim-teleop, real-replay, and both status panels | *from config* | follows automatically through `_fr3_target_frame_name()`. The replay page and the idle teleop status used to print `pika_task_tcp` as a literal / a stale dataclass default; both now report the configured frame, so the label an operator reads before pressing Run comes from where the command does |
| `gateway._fr3_target_frame_name` fallback | **`pika_gripper_ee`** | changed. Only reached by a config that omits the key, which the workstation's does not — but the recorder would build its robot from `FrankaResearch3Config`, whose default is `pika_gripper_ee`, so a `pika_task_tcp` fallback here meant recording in one frame while telling the sim teleop and the replay another. Now pinned to the dataclass rather than to a literal |
| `fr3_gui_record_runtime.py:400` | fallback `pika_task_tcp` | fallback only, and the record config now sets the key explicitly |
| `config_franka_research3_mujoco.py:35`, `envs/fr3_mujoco.py:58` | `pika_task_tcp` | library defaults for a bare CLI run. Every GUI path passes the frame explicitly, so these never apply to this rig; left alone because they are shared with the quest3 and RL envs |

**What was re-derived, and what is still yours to tune.** Every one of these is stated against the
target frame, so the switch re-aims it onto the real tool without anyone editing a number.

*The workspace box — re-derived.* It clips `desired_pose[:3, 3]`, the target frame origin, so at
`pika_task_tcp` it fenced a point 411 mm behind the fingers and could not stop the fingers hitting
anything. At `pika_gripper_ee` it can, so it is now derived from the thing it is protecting. The
workstation table in `fr3_pika_gripper_scene.xml` sits at base-frame x `[0.18, 0.86]`,
y `[-0.46, 0.46]`, with its top at exactly **z = 0**:

| | old, at `pika_task_tcp` | new, at `pika_gripper_ee` | where it comes from |
| --- | --- | --- | --- |
| x | `0.2` … `0.9` | **`0.18` … `0.70`** | the table's near edge; far edge trimmed so most of the box is inside the FR3's 855 mm reach |
| y | `-0.6` … `0.6` | **`-0.45` … `0.45`** | the table's own width, less 10 mm |
| z | `0.05` … `0.8` | **`0.0` … `0.70`** | the tabletop. The fingertips can no longer be commanded into the surface |

`home` puts the tool point at base-frame `(0.309, -0.001, 0.398)`, clearing every wall by at least
129 mm, and the 11 600 recorded pick-and-place frames re-expressed at the tool point span
x `[0.264, 0.464]`, y `[-0.355, 0.122]`, z `[0.028, 0.397]` — the whole existing task fits inside
with margin, so the fence is not what will stop a new one either. Two of the box's eight corners
(far, high) are outside the arm's reach; no axis-aligned box that covers this table avoids that, and
an unreachable command fails loudly in IK rather than silently.

*The per-step clamps — unchanged, and the reason is measurement, not luck.* The geometric worst case
says a step used to allow 1 mm of translation plus 0.01 rad of rotation, which the lever turned into
~5 mm at the fingers, and that the same clamps now allow 1 mm — a 5x tightening that ought to feel
abruptly rate-limited. It does not, because neither clamp was ever reached: at full SpaceMouse
deflection the shipped gains produce at most 0.615 mm and 0.000648 rad per tick, **1.6x and 15x
below** the clamps. What actually changed is that the clamps became honest — 1 mm/step is now 1 mm of fingertip
motion — and the arm got safer, not slower.

*The teleop gains — yours, on hardware.* `translation_scale` / `rotation_scale` were tuned with the
lever present, and `rotation_scale` in particular was tuned for an axis whose two siblings were
switched off. Tune them from **Teleoperation → SpaceMouse 6D Gains** (next section) and write what
you settle on back into the YAML. One interaction worth knowing before you turn the knob:
`max_target_delta_pos` is 0.001, so a `translation_scale` above `0.001` starts being clipped on x
rather than scaled — 0.002 buys 1.63x there, not 2x — while y and z, which the device calibration
scales to 94% and 59%, keep going for a while longer and the motion skews.
`max_target_delta_rot` is 0.01, which is also the panel's cap, so rotation has headroom over its
whole range.

*The rollout gates — untouched, and now due for a look.* `--max-step-*` / `--first-frame-max-*` are
stated against the target frame, so they were ~6x looser at the tool than they read for as long as
the rollout ran in `pika_task_tcp`. With the launcher's default moved they are honest — and
therefore tighter in practice. Nothing has been trained yet, so nothing has been measured against
them; re-check them against the first post-switch rollout rather than assuming the old numbers
transfer.

**What the recorded episodes actually say.** The bullets above are the geometry's worst case. The
lever only injects error when the wrist turns, so the question is how much this task turns it.
Measured on the recorded `observation.state` of both corpora on the workstation —
`fr3_pick_place_ee2ee_v1_merged_20260611_20260612` (50 episodes / 20 999 frames) and
`merged_fr3_pick_place_data_260316` (51 / 15 305) — by re-expressing every frame in
`pika_gripper_ee` through `p_ee = p_tcp + R(q) @ d` and comparing the two streams:

| | merged_20260611_20260612 | merged_260316 |
| --- | --- | --- |
| wrist rotation from episode start, median / max | **4.3° / 11.7°** | 5.9° / 11.2° |
| episode path length, `pika_task_tcp` vs `pika_gripper_ee` | 0.810 m vs 0.795 m (**1.02x**) | 0.774 m vs 0.763 m (1.02x) |
| per-step translation injected by the lever, p95 | **0.62 mm** | 1.09 mm |
| … as a fraction of the real tool's median step | **0.08** | 0.13 |
| steps where the tool is still (<0.2 mm) but `pika_task_tcp` is not | 27.1% of frames, p95 **0.11 mm** | 8.7%, p95 0.24 mm |

The 0.41 m lever is real but dormant: 4° of wrist motion cannot swing it far. The rotation the
teleop actually commands is ~0.086°/step (0.62 mm ÷ 411 mm), **6.6x below** the
`max_target_delta_rot` of 0.01 rad — so the clamp being ~5x looser at the tool than it reads never
binds either. All three metrics are invariant to which fixed frame the episodes are expressed in
(a rigid re-framing rotates `Δ(R @ d)` without changing its norm), so they hold regardless of
whether the recorded columns are base-frame or workspace-frame.

The trigger to revisit was a *task* change, not a code change: a task with real reorientation —
in-hand regrasp, pouring, inserting at an angle, anything swinging the wrist through tens of
degrees — moves these numbers by the ratio of the rotation used, and at 40° the injected motion is
an order of magnitude larger than what is measured here. That is what happened, and the next section
is why the rotation case does not reduce to scaling this table up.

Re-run this measurement on the first post-switch corpus before trusting it: every number here was
taken on episodes whose wrist barely moved, which is exactly the regime the rotation task leaves.

**A task that reorients the tool.** The measurement above says the lever is dormant, not that it is
harmless — it is dormant because pick-and-place turns the wrist a median of 4.3°. The reason it
matters so much more for a rotation task is not noise; it is where the command pivots.
`franka_research3.py:782` composes the target as

```python
desired_pose[:3, :3] = self._reference_pose[:3, :3] @ delta_rot.as_matrix()
desired_pose[:3, 3]  = self._reference_pose[:3, 3] + delta_pos
```

The rotation multiplies the rotation block alone, so a pure rotation command holds the *target
frame origin* fixed and swings everything else around it. With the target frame 411 mm behind the
fingers, "rotate the gripper" means "sweep the fingers along an arc of radius 0.41 m":

| commanded rotation | fingertip arc, `pika_task_tcp` | fingertip arc, `pika_gripper_ee` |
| --- | --- | --- |
| 5° | 36 mm | 0.3 mm |
| 15° | 107 mm | 0.9 mm |
| 30° | 213 mm | 1.8 mm |
| 45° | 314 mm | 2.6 mm |
| 90° | 581 mm | 4.8 mm |

(Chord `2 R sin(θ/2)`; the right column uses the 3.4 mm residual between `pika_gripper_ee` and the
measured finger working-point midpoint.) A 90° reorientation alone sweeps 581 mm through a
workspace box only 700 mm wide in x, so it clipped before it completed. To turn an object *in place*
the operator has to push translation to cancel a two-to-three-hundred-millimetre arc at the same
time, by hand, continuously — and the policy then has to learn that compensation as if it were part
of the task. This is almost certainly why `scale_wx` and `scale_wy` *were* pinned to `0.0` in
`fr3_record_config.yaml`, and why all three rotation axes are still zeroed in the sim teleop's own
flag defaults: roll and pitch are not usable from that frame. Switching the frame is what makes them
usable, and the pins are gone as of the switch above.

The ordering for a rotation task is therefore: switch the frame, re-derive the box, unpin the
rotation axes, confirm the teleop feel on hardware with roll and pitch live, and only then record.
The first three are done and committed; the fourth is a hardware session and has not happened yet.
Recording before it would produce exactly the corpus split the next paragraph warns about.

**Existing datasets do not have to be thrown away.** The two frames share an orientation exactly,
so their separation expressed in the *tool* frame — the frame the recorded rotvec columns describe
— is a rigid constant, independent of arm configuration
(`test_fr3_tool_frame_geometry.py::test_the_offset_is_the_same_constant_in_the_tool_frame_for_every_configuration`):

```
d = (-0.366842, 0, 0.185)   metres, in the tool frame
p_ee = p_tcp + R(rotvec) @ d
```

That converts a `pika_task_tcp` episode into a `pika_gripper_ee` one exactly. Rotations, gripper and
images are frame-independent and stay untouched. Three details decide whether a migration script is
right or merely plausible:

- **Three position triplets, each with its own rotation.** The recorded schema is
  `observation.state = [ee.xyz, prev_cmd.ee.xyz, ee.q, prev_cmd.ee.q, gripper.pos,
  prev_cmd.gripper.pos]` and `action = [ee.xyz, ee.q, gripper.pos]`. So `ee.xyz` converts with
  `ee.q`, `prev_cmd.ee.xyz` with **`prev_cmd.ee.q`**, and the action's `ee.xyz` with the action's
  own `ee.q`. Reusing `ee.q` for the commanded pose is the obvious slip and it is wrong by the
  measured-to-commanded rotation difference — small, never zero, and silent. (The action columns are
  absolute poses named `ee.*`; `target_*` is the teleop runtime's delta naming in
  `delta_action_processor.py` and never reaches a parquet.)
- **Recompute the statistics; do not transform them.** `meta/stats.json` and the per-episode
  `stats/observation.state/*` and `stats/action/*` columns in `meta/episodes/*.parquet` carry
  min/max/mean/std/count and q01…q99, and today they are `pika_task_tcp` numbers (state mean x =
  0.7145 on `fr3_spacemouse_20260813_160401`). The conversion is nonlinear in `q`, so no closed form
  maps the old quantiles to the new ones. `fr3_train_il_policy.py` rebuilds
  `observation.state`/`action` stats from the data when it materialises a training view, so the
  training path is covered either way — but anything reading the source dataset's stats directly is
  not.
- **Nothing in the dataset says which frame it is in.** `ee.x` is `ee.x` in both. A migrated and an
  unmigrated copy are indistinguishable by schema, which is why the rename below is not cosmetic.

What you must *not* do is mix the two frames in one training set — that trains on two different
robots — so migrate the whole set or none of it, and rename the dataset series so the two can never
be globbed together.

`tools/fr3/fr3_migrate_tool_frame.py` does all of this. It matches the pose columns by feature
*name* rather than by offset, recomputes both stats locations with the same estimator
`fr3_train_il_policy.vector_stats` uses, and writes `meta/fr3_tool_frame.json` as the frame marker
the schema lacks. Run it with `.venv-fr3` (it needs `mujoco` for the exit check), one dataset at a
time — the rename is the safety mechanism, so it is never derived:

```
.venv-fr3/bin/python tools/fr3/fr3_migrate_tool_frame.py \
    --src outputs/datasets/fr3_spacemouse_20260813_160401 \
    --dst outputs/datasets/fr3_spacemouse_eeframe_20260813_160401
```

It refuses a source that already reads as `pika_gripper_ee`, so running it twice is an error rather
than an 822 mm dataset. Videos are hardlinked (`--videos copy|symlink` to change); `--dry-run`
identifies the source frame and stops.

Measured on `fr3_spacemouse_20260813_160401` (10 305 frames, 20 episodes): the three converted
triplets land within 1.5e-5 mm of the closed form, quaternion/gripper/index columns come out
bit-identical, and the global and per-episode stats agree with the migrated values. Had `prev_cmd`
been converted with `ee.q` instead of its own quaternion the result would have been **11.4 mm** out —
the bullet above is not hypothetical.

**Two ways to check a migration instead of trusting it.** `fr3_sim_record_replay_runtime.py`
already identifies the frame from the data: it solves the first recorded pose against both bodies
and prints `pos diff to pika_gripper_ee` / `pos diff to pika_task_tcp` before picking the closer
one. On a correctly migrated episode the `pika_gripper_ee` diff collapses and the `pika_task_tcp`
one goes to ~411 mm; if both stay large, the conversion is wrong in a way arithmetic review would
not have caught. Then run the MuJoCo replay gate on one migrated episode before any hardware
replay — scoring the recorded stream through IK is exactly what it is for, and a 411 mm frame error
is the loudest thing it can possibly report.

Until they are migrated, the pre-switch datasets are **not safe to replay against this config**, in
two ways that compound. Both were silent, and both now fail in the replay preflight
(`fr3_gui_replay_runtime._preflight_contract`) before the arm is connected:

- **Frame.** Replay reads recorded poses and commands them. With the config naming
  `pika_gripper_ee` and the poses recorded at `pika_task_tcp`, the arm puts the fingertips where the
  old frame's origin was — 410.85 mm off, running to completion without an error.
- **Workspace.** `_make_pose_from_absolute_action` clips absolute replay commands too, so the box
  applies to a replayed trajectory exactly as it does to a teleop command. The new box was derived
  around the *migrated* footprint (x `[0.264, 0.464]`); the same episodes unmigrated sit at
  x `[0.633, 0.832]`, so **65.2% of `fr3_spacemouse_20260813_160401`'s frames land outside it**,
  displaced by up to 132 mm (mean 22.2 mm) — every one of them flattened against `x = 0.70`. Under
  the old box, 0% were outside. This is the worse of the two: a frame error is a rigid offset the
  arm still tracks, whereas clipping reshapes the trajectory and then the score is computed against
  the reshaped version.

Either convert the dataset, or replay it against an overlay config that names the old frame *and*
restores the old box — `target_frame_name: pika_task_tcp` with
`workspace_min: [0.2, -0.6, 0.05]` / `workspace_max: [0.9, 0.6, 0.8]`. Measured on episode 0 of that
dataset: unmigrated under the old contract scores 590/590 frames at 2.81 mm avg / 7.27 mm max, and
migrated under the new one scores 590/590 at **1.94 mm avg / 4.88 mm max** — better, because the
error is now measured at the tool instead of at the end of a 0.41 m lever.

The Episode Replay page also prints the frame the gateway will actually use, next to the dataset it
will use it on; read them together.

**Replay runs at the dataset's frame rate, not the recorder's.** `ReplayStatus.fps` is seeded from
the recorder config's `dataset.fps`, which describes what the *next* recording will do. The two
diverged the moment the recording rate became adjustable (`Add FR3 recording FPS and crop controls`
set it to 60 while every dataset on disk was recorded at 30), and replaying at the wrong rate fails
in two ways that look like anything but a frame-rate bug:

- the qpos preview report is indexed at that rate, so the arm is compared against the same
  frame grid the browser scrubs;
- `fr3_gui_replay_runtime` derives the sim's `teleop_control_frequency` from it, so each command is
  integrated for a fraction of the frame period the recorder actually had. The comment in
  `_replay_robot_config` predicted exactly this: *"tracking error that is really just an
  under-integrated servo window"*.

Measured on `eeframe_fr3_spacemouse_20260813_160401` episode 0, changing nothing but `--fps`:

| `--fps` | verdict | avg | max pos | max rot |
| --- | --- | --- | --- | --- |
| 30 (the recorded rate) | **passed** | 1.94 mm | 4.88 mm | 0.64 deg |
| 60 (the config's rate) | failed | 2.43 mm | **43.26 mm** | **8.23 deg** |

`gateway._replay_fps()` now takes the rate from the dataset's own `meta/info.json` for both the
MuJoCo gate and the real replay, and selecting a dataset moves `ReplayStatus.fps` onto it so the
timeline agrees. A dataset that declares no rate keeps the status value rather than silently
becoming 30.

One sharp edge the switch exposed: `fr3_gui_replay_runtime.py` run **by hand without
`--ik-orientation-weight`** takes the robot backend default, which core-dumps (SIGFPE, ~frame 250)
against `pika_gripper_ee` where it was stable against `pika_task_tcp`. The gateway always passes
`0.012` (`DEFAULT_WORKSTATION_REPLAY_IK_ORIENTATION_WEIGHT`), so the GUI path is unaffected — but
pass it explicitly if you drive the runtime from a terminal.

### SpaceMouse gains from the Teleoperation page

The six-axis gains that map the SpaceMouse onto the tool are editable in the GUI under
**Teleoperation → SpaceMouse 6D Gains**, because they can only be tuned with a hand on the device
and an arm in front of you. The panel edits the same eight fields the config declares:
`translation_scale` and `rotation_scale` are the global gains, and `scale_x` … `scale_wz` override
them one axis at a time.

- **Blank is not zero.** An empty per-axis field means "follow the global gain"; a `0` disables that
  axis. `fr3_record_config.yaml` now ships all six axes blank, so all six follow the two globals; it
  used to pin `scale_wx: 0.0` and `scale_wy: 0.0`, which left yaw as the only live rotation axis.
  The sim teleop still zeroes all three — the panel shows both columns for exactly this reason.
- **Blank is not the global gain either.** A blank axis follows the global *times* the device's own
  per-axis calibration (`TRANSLATION_AXIS_CALIBRATION` / `ROTATION_AXIS_CALIBRATION` in
  `teleop_spacemouse.py`): `x1.00 / x0.94 / x0.59` on translation and `x1.00 / x0.95 / x0.93` on
  rotation. A *filled* axis replaces the calibrated value instead of scaling it, so typing the
  global's own number into z speeds that axis up 1.7x rather than changing nothing. The panel prints
  each row's factor and resolves both cases in its m/s / rad/s readout; the factors are mirrored from
  the teleoperator and pinned by
  `test_the_mirrored_axis_calibration_matches_the_teleoperator`.
- **Untouched means unchanged.** With no override the rig launches byte-for-byte as before: real
  teleop and the recorder get the literal config file, and the sim teleop gets no gain flags at all.
- **An override reaches teleop *and* recording.** The same teleoperator drives both, so a session
  tuned at one gain and recorded at another would put demonstrations in the dataset that nobody
  ever felt. Real teleop is spawned against an overlay config
  (`outputs/.teleop_gains_config.yaml`), the recorder against its own overlay, and the sim teleop
  gets matching CLI flags.
- **It applies on the next start.** The teleoperator reads its gains once, when it connects; a
  running session keeps the ones it started with.
- **It never writes the YAML.** `fr3_record_config.yaml` defines the recording contract; a session
  of live experimenting should not rewrite it. Once a gain is worth keeping, edit the file.
- **Bounds.** Values are capped at ±0.01 per device tick. At the 200 Hz poll rate that is already
  2 m/s or 2 rad/s at full deflection, so the cap catches a typo'd digit rather than constraining
  tuning. The real per-step safety net is `robot.max_target_delta_pos` / `max_target_delta_rot`,
  which clamp regardless of the gain that produced the command.

**Roll is inverted at the device, on purpose.** `SpaceMouseTeleopConfig.rotation_axis_map` defaults
to `[-raw_wx, raw_wy, raw_wz]`: the device's roll axis opposes the tool's x axis on this rig. It sat
at identity — and wrong — for as long as the rig recorded against `pika_task_tcp`, because
`scale_wx`/`scale_wy` were pinned to `0.0` there and nothing ever exercised roll or pitch. Turning
them back on for the tool-frame switch is what surfaced it. No recorded episode is affected: roll was
switched off in every one of them, so no roll command was ever stored.

Two things follow. The rotation map is *not* the translation map and must not be tidied into it —
`target_{x,y,z}` is added in the robot base frame while `target_{wx,wy,wz}` is right-multiplied onto
the reference orientation, so they align device axes to different frames. And **pitch is the axis
that still has no evidence behind it**: it was pinned off alongside roll, and yaw was the only
rotation axis live during all previous recording. Check `wy` on hardware before trusting it; the
convention is pinned by `tests/teleoperators/test_spacemouse.py`.

Note that `tools/fr3/fr3_mujoco_teleop.py` does **not** read the recorder YAML — it carries its own
flag defaults, and they differ from the hardware on every axis that matters: 3x the translation
gain (`0.001845` vs `0.000615`) and all three rotation axes zeroed. Sim and hardware therefore do
not feel alike until a gain is applied in the panel, which is sent to both. The gateway mirrors
those sim defaults so the UI can say so, pinned to the parser by
`tests/scripts/test_data_collection_gui_gateway.py::test_the_mirrored_sim_gain_defaults_match_the_sim_script`.

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

### When the audit fails: which cause it was

The audit's budgets say alignment broke; they cannot say why, because a dataset only holds the
timestamps of the frames that were *selected*. A frame the sensor never produced, one the host
dropped on the way in, and one that arrived late are indistinguishable there. Two tools split
them apart, and each fault shape has a distinct signature — pinned by
`tests/scripts/test_fr3_camera_delivery.py`, so a shape that reads one way in the test cannot
read the other way on the rig.

`python tools/fr3/fr3_camera_delivery_probe.py <dataset_root> --camera-fps 60` — offline, from
the same column the audit reads. Reports two things the audit does not:

- **Frame reuse**: a repeated capture timestamp means the loop wrote the *same image* into two
  frames because no new one had arrived. Skew cannot see this (a reused frame is perfectly
  aligned with itself) and it is the more damaging failure — the images stand still while state
  and action keep moving.
- **Real delivery rate**, as a histogram in units of the sensor period. `{1x: …}` with a small
  tail is a host that hiccuped; `{2x: …}` throughout is a sensor that ran at half rate. Plus
  whether the bad frames arrived in bursts or evenly, which separates a load fault from a
  standing one.

`python tools/fr3/fr3_camera_delivery_bench.py --duration 60 --poll-hz 60` — live, cameras only,
no arm and no encoder. It reproduces this page's anchoring in a process doing nothing else, so
the skew it prints is what the recorder *would* have recorded had it been free. It also reads
the three things only available live:

| reading | meaning |
| --- | --- |
| `frame_counter` gaps | the sensor produced frames this host never received — USB bandwidth, a link that negotiated 2.1, a driver stall. The only measurement that proves a drop; absent on a kernel without the RealSense metadata patches, and the tool says so instead of reporting zero |
| acquisition spacing wider than nominal, counters contiguous | the *sensor* slowed down. For a colour stream that is nearly always exposure: no sensor emits a frame faster than it exposes one, so an exposure past the frame period caps the rate whatever profile was negotiated. `actual_exposure` is printed next to it — compare it against `1000 / fps` ms |
| handover lag, with counters contiguous at nominal spacing | the frames arrived and this process was busy — the host-side failure, and the one recording load can cause |

The preamble alone ends most investigations: a `usb_type` of 2.1, `global_time` off (capture
timestamps then fall back to the handover instant and carry each camera's pipeline delay), or a
negotiated profile below the requested rate.

Then vary one thing at a time: `--poll-hz 30` against the same scene (delivery must not care how
often it is polled), `--extra-work-ms 4/8/16` as a dose-response for host load, and a rerun under
brighter light to test auto-exposure. Recording the same scene at two `dataset.fps` values and
diffing the two `meta/fr3_sync_report.json` closes the loop on the full pipeline.

### Exposure is a frame-rate control, and it lives on the device

This is what the procedure above caught on 2026-08-19, and it is worth knowing by name because
nothing in the recording path reports it. Both cameras had been left on **manual** exposure —
36.5 ms on `ee`, 42.3 ms on `side` — by some earlier program, most likely RealSense Viewer.
RealSense controls are device state, not process state: they outlive whatever set them, and the
next `pipeline.start` inherits them.

Neither exposure fits in a 16.7 ms frame period, so the sensors emitted at 15.0 and 23.6 fps.
Every other indicator stayed green — profiles negotiated 640x480@60, USB 3.2, `global_time` on,
frame counters contiguous with zero drops — because nothing had failed. The images simply went
stale (max 95.8 ms behind the arm read, against a `camera_max_age_ms` of 100), 65-75% of the
recorded frames were duplicates of their predecessor, and the SYNC audit failed on cross-camera
skew, which is four steps downstream of the cause.

Two things now stand between that and a dataset:

- `exposure_us` and `gain` in the camera config. Set, they pin a rate that cannot drift with the
  light; **unset, they hand the sensor back to auto exposure at connect** — which is the part
  that matters, because "leave it alone" is what let stale device state through. `connect`
  refuses outright if the exposure cannot fit the frame period.
- `fr3_camera_delivery_bench.py` prints `actual_exposure` beside the delivered rate, so the next
  occurrence is one command away rather than two days of recordings.

Fixed exposure is not free: brightness no longer tracks the room, so `gain` has to be
re-measured when the lighting changes. What it buys beyond the rate is constant motion blur and
constant image statistics across a take, which auto exposure does not give.

### A camera that opens but never delivers

```
TimeoutError: Timed out waiting for frame from camera RealSenseCamera(315122271876) after
1000 ms. Read thread alive: True.
```

This is not a configuration problem and not a busy device — the pipeline started, the read
thread is running, and no frame is coming. A D400 can reach that state after a session dies
mid-stream, and nothing in software clears it: observed on the `ee` D405 with a bare colour
stream, no controls touched, going 10 s without a frame while the `side` D435i on the same bus
opened in 383 ms. Only re-enumeration fixes it.

`connect` now does that itself: it resets the device with `hardware_reset`, waits for it to
come back (~2 s), settles, and retries the connection **once**. Measured end to end at 6.8 s,
against a failed session. Once, not in a loop — a camera still dead after a reset is telling you
something, and retrying forever would bury it.

Two cases deliberately skip the reset, because it cannot help and would cost 30 s of confusion:
a device that is not enumerated at all (log says replug it) and a `ValueError` from the exposure
guard, which is a number in this YAML rather than a sick camera.

If it still fails after the automatic reset, replug the camera. `tools/fr3/fr3_realsense_preview.py`
is the quickest way to confirm it is back before starting a session.

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

Reports land in `derived/fr3_mujoco_replay/episode_XXXXXX.json`. During replay the runtime
streams per-frame MuJoCo body poses through gateway memory for the Episode Replay browser
viewer; the GUI no longer asks the runtime to render an MP4 preview.

## Known constraints

- `save_episode(parallel_encoding=False)` is deliberate. The multi-camera parallel path forks a
  `ProcessPoolExecutor` from a process already holding the MuJoCo/EGL context, camera driver
  threads, and the stdin reader; that fork deadlocks. Sequential encoding costs a few hundred
  ms between episodes. `streaming_encoding: true` (the shipped config) avoids the path entirely.
- Sim camera resolution is one renderer for all cameras, so `--backend sim` refuses a config
  whose cameras have different resolutions rather than recording one of them at the wrong size.
