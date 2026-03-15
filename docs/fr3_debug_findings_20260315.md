# FR3 Debug Findings 2026-03-15

This note captures the main conclusions from the FR3 recording and visualization debugging work completed in this round.

## Recording Runtime

- `@parser.wrap()` needed to resolve postponed/string annotations before calling `draccus.parse(...)`; otherwise FR3 record entrypoints could pass a string config type and crash in `dataclasses.fields(...)`.
- The episode-boundary EE jump was caused by a state mismatch:
  - `move_to_start()` reset the robot-side teleop state.
  - `DeltaActionToAbsoluteEEAction` kept its cached `_last_command_pose`.
  - The first frame of the next episode reused the stale absolute EE target.
- Fix: call `teleop_action_processor.reset()` after each saved episode in `tools/fr3/fr3_record_runtime.py`.

## Startup Drift

- Pure teleoperation did not drift, but FR3 recording did.
- Root cause was not the SpaceMouse alone; it was the recording pipeline's idle action semantics.
- Pure teleop sends disabled delta actions on idle frames, which makes `FrankaResearch3.send_action()` hold the current joints.
- FR3 recording originally converted idle frames into absolute `ee.*` targets and resent them every frame.
- That forced repeated IK on idle frames and exposed small FK/IK/TCP inconsistencies as a slow visible drift.
- Fix:
  - `DeltaActionToAbsoluteEEAction` now preserves both absolute EE outputs and the raw teleop metadata.
  - `AbsoluteEEActionToRobotAction` converts idle frames back into disabled delta actions before commands are sent to the robot.
  - Dataset output remains ee2ee; robot idle behavior now matches pure teleop.

## Controller Startup Hold

- `PandaPyArmDriver` was updated to seed the `JointPosition` controller with the current joint state before starting the controller.
- This avoids starting the controller without an explicit hold setpoint.

## Episode Start Settle

- Recorded episode openings were still noisy even after the idle-action fixes because recording began before the arm/gripper had fully settled at the initial command target.
- Fix: FR3 recording now freezes a single initial absolute EE/gripper target before each episode and waits until the robot observation stays within settle thresholds before writing frames.
- Current settle thresholds in `tools/fr3/fr3_record_runtime.py`:
  - position error: `2 mm`
  - orientation error: `1 deg`
  - gripper error: `0.02`
  - consecutive samples: `5`
  - timeout: `3 s`
- Important implementation detail:
  the settle stage must not keep re-running the delta teleop processor, otherwise enabled delta commands would be re-integrated and the target would drift during the wait window.
- Practical result:
  episode openings became noticeably cleaner, especially for the gripper trace.

## Dataset Visualization

- The FR3 ee2ee dataset stores EE pose inside `observation.state`, not as top-level `ee.x`, `ee.y`, `ee.z`, `ee.qx`, `ee.qy`, `ee.qz`, `ee.qw` keys.
- The visualizer originally looked only for top-level EE keys, so the 3D EE visualization never activated.
- Fix: `src/lerobot/scripts/lerobot_dataset_viz.py` now reads packed quaternion EE poses from
  `observation.state` with
  `names=["ee.x", "ee.y", "ee.z", "ee.qx", "ee.qy", "ee.qz", "ee.qw", "gripper.pos"]`.

## Absolute Orientation Representation

- Internal robot control and teleop delta rotations remain rotation-vector based.
- Absolute EE orientation in the ee2ee dataset was switched to quaternion representation.
- Dataset action and observation state are now:
  - `ee.x`, `ee.y`, `ee.z`
  - `ee.qx`, `ee.qy`, `ee.qz`, `ee.qw`
  - `gripper.pos`
- Quaternions are continuous within an episode using hemisphere alignment:
  if `dot(q_t, q_{t-1}) < 0`, the sign is flipped.

## World Frame Interpretation

- The visualized EE pose is produced by placo FK using `RobotKinematics.forward_kinematics(...)`.
- The returned transform is `get_T_world_frame(target_frame_name)`.
- For FR3, `target_frame_name` defaults to `pika_gripper_ee`.
- The placo `world` frame is the URDF root/base frame because:
  - the base is fixed with `mask_fbase(True)`
  - the URDF root link `base` is connected to `fr3_link0` with a zero fixed transform
- Practical axis convention:
  - `+X`: forward, toward the robot workspace
  - `+Y`: left
  - `+Z`: up

## Rerun 3D Additions

- The dataset visualizer now logs:
  - EE trajectory
  - current EE position
  - current EE local frame
  - a static ruler at the world origin
- Ruler convention:
  - `X`: red
  - `Y`: green
  - `Z`: blue
  - configurable with `--ee-ruler-length`

## Rerun Time Axis

- The dataset stores `timestamp` as episode-relative seconds, not wall-clock Unix time.
- Rerun visualization originally logged that relative value directly as a `timestamp` timeline, which made the viewer show dates near `1970-01-01`.
- Fix:
  `src/lerobot/scripts/lerobot_dataset_viz.py` now uses two timelines:
  - `episode_time`: relative duration timeline
  - `timestamp`: absolute wall-clock timeline anchored to the current system time when visualization starts
- This keeps dataset timing semantics intact while making the Rerun absolute-time axis human-readable.

## Gripper Latency Investigation

- In dataset `outputs/datasets/fr3_pick_place_ee2ee_v1_20260315_110655`, gripper `action -> observation.state` aligns best at roughly `31-32` frames.
- At `30 FPS`, this corresponds to about `1.03-1.07 s` of apparent latency.
- The arm EE mismatch is not the same issue; the large lag is specific to the gripper channel.
- Current FR3 control path:
  - every control tick sends a gripper command via `FrankaResearch3.send_action()`
  - the recording config currently uses `control_fps: 200`
  - `PikaGripperHardwareDriver.set_position()` forwards every call directly to `Gripper.set_gripper_distance(...)`
  - `PikaGripperHardwareDriver.get_position()` reads `Gripper.get_gripper_distance()`
- This means the current FR3 runtime is:
  - writing gripper commands at up to `200 Hz` with no deduplication
  - observing gripper state through the SDK's high-level width conversion API
- In contrast, the standalone smoke script checks latency using the lower-level motor position feedback (`motor_position_rad`) rather than only the converted width reading.
- Initial working hypothesis:
  the observed `~1 s` latency is mainly software-side, caused by repeated high-rate command writes plus a slower / more delayed high-level feedback path in the current SDK wrapper, not by the physical gripper alone.

## Pika SDK Frequency Findings

- The dependency source should move from `agx-pypika` to `git@github.com:linkage-x/pika_sdk.git`.
- Source inspection of `pika_sdk` shows:
  - default serial baudrate: `460800`
  - default serial timeout: `1.0 s`
  - host-side serial reading thread sleep: `0.001 s` per loop, i.e. about `1000 Hz` polling of the host serial buffer
- The SDK does not appear to implement a host-side default write scheduler for gripper commands.
  each `set_gripper_distance(...)` call writes immediately to the serial port.
- Therefore the actual feedback update frequency is still partly determined by device-side firmware publish behavior, not only by the SDK's host polling loop.

## Docker Runtime Reproduction

- The FR3 Docker runtime now installs `pika_sdk` from `git@github.com:linkage-x/pika_sdk.git`.
- Docker source inspection confirmed:
  - `SerialComm(..., timeout=1.0)` is the container default.
  - `set_gripper_distance(...)` computes the target motor angle and immediately writes a serial command.
  - no host-side deduplication, coalescing, or write-rate limiting is implemented in the SDK.
- Direct Docker measurements against the real gripper showed:
  - single command path:
    motor-position feedback changed in about `10-80 ms`
  - repeated `200 Hz` writes to the same close target:
    first observable gripper feedback moved only after about `1.06 s`
  - repeated `200 Hz` writes to the same open target:
    first observable feedback moved after about `0.25 s`
- This turned the earlier hypothesis into a confirmed runtime finding:
  the large gripper lag was primarily caused by command spam from the FR3 recording loop, not by Docker device passthrough itself and not by the physical gripper alone.

## Gripper Command Spam Fix

- `FrankaResearch3Config` now exposes:
  - `gripper_command_rate_limit_hz`
  - `gripper_command_deadband_mm`
- `PikaGripperHardwareDriver` now:
  - skips writes when the requested width change is within the deadband
  - rate-limits gripper serial writes
  - keeps only the newest pending target between allowed send slots
- Default values used in code:
  - rate limit: `15 Hz`
  - deadband: `0.5 mm`
- Practical intent:
  preserve the arm control loop at `200 Hz` while reducing the gripper serial path to a rate the Pika runtime can actually absorb.

## Latest Re-measurement

- Latest dataset:
  `outputs/datasets/fr3_pick_place_ee2ee_v1_20260315_120255`
- Recomputed gripper `action -> observation.state` alignment now peaks at roughly `6` frames.
- At `30 FPS`, this corresponds to about `0.20 s`.
- This is a major reduction from the previous `31-32` frame (`1.03-1.07 s`) lag.
- Event-level inspection of the latest episode shows:
  - most complete gripper transitions begin reacting after `4-6` frames
  - one later close transition reacted after about `9` frames
  - the final tiny reopen command at the episode tail did not fully appear before the recording ended
- Current interpretation:
  the primary `~1 s` latency bug has been removed; the remaining gripper lag is now in the same rough regime as the standalone smoke baseline and looks like normal device / control-path latency rather than a runaway software queueing issue.

## Dataset Ownership Fix

- A later visualization attempt on dataset
  `outputs/datasets/fr3_pick_place_ee2ee_v1_20260315_122209`
  failed with:
  `av.error.PermissionError: [Errno 13] Permission denied`
- The failing mp4 files were root-owned Docker outputs, e.g. `root:root` with mode
  `600`, so the normal user could not open them through the dataset loader.
- Manual `chown` fixed the immediate issue, which confirmed the problem was output
  ownership rather than video corruption or a decoder bug.
- The FR3 Docker recording wrapper now follows up a successful recording run with a
  container-side `chown -R <host_uid>:<host_gid> <dataset_root>`.
- This keeps the existing hardware-facing Docker setup unchanged while ensuring
  the resulting dataset is readable by the host user for visualization and
  post-processing.

## Current Best Practice

- Record FR3 datasets through `tools/fr3/fr3_record.py`, not through ad-hoc
  `docker compose run ...` commands.
- Keep the FR3 arm control loop at `200 Hz`, but rate-limit the Pika gripper
  command path with the current defaults:
  - `gripper_command_rate_limit_hz = 15.0`
  - `gripper_command_deadband_mm = 0.5`
- Treat the wrapper's automatic ownership normalization as part of the standard
  recording path. A dataset that is not readable by the host user is considered
  an invalid recording artifact for downstream tooling.
- For remote inspection, use `lerobot_dataset_viz --mode distant` plus a local
  `rerun` viewer instead of trying to forward the remote GUI directly.
- During episode-start settle, freeze the current EE pose from robot
  observation and only inherit the current gripper target from teleop.
- Do not freeze live SpaceMouse translation/rotation deltas into the settle
  target, even for a single frame.

## Settle Target Regression Fix

- After the gripper command-spam fix, recording start showed a new visible
  default motion along `-X`.
- Root cause:
  `_wait_for_episode_start_settle(...)` froze its settle target from the first
  live `teleop.get_action()` sample.
- That meant a transient SpaceMouse sample such as:
  - `enabled=True`
  - `target_x != 0`
  could be converted into an absolute EE target and replayed throughout the
  settle window.
- Why it became visible only now:
  gripper rate limiting did not create the bug, but it lengthened the settle
  window enough for the bad first-frame target to move the arm.
- Fix:
  `tools/fr3/fr3_record_runtime.py` now builds the settle target from:
  - current robot EE pose
  - current robot EE orientation
  - current teleop gripper target
  and forces zero translational / rotational deltas for the settle phase.
- Regression tests now lock this down:
  - the settle helper must not call the teleop delta-to-absolute processor
  - an initial SpaceMouse `target_x` sample must not change the settle EE target

## Next TODO

- Decide whether the `15 Hz` / `0.5 mm` defaults need task-specific tuning after collecting more than one episode.
- After validating the new SDK source path, update any build/run instructions that still assume `agx-pypika`.
- Verify on at least one more freshly recorded episode that host ownership is
  normalized automatically and `lerobot_dataset_viz` can open the videos without
  any manual `chown`.

## Targeted Validation

- `PYTHONPATH=src .venv-codex/bin/pytest tests/configs/test_plugin_loading.py`
- `PYTHONPATH=src .venv-codex/bin/pytest tests/robots/test_franka_research3.py`
- `PYTHONPATH=src .venv-codex/bin/pytest tests/scripts/test_fr3_record_runtime.py`
- `PYTHONPATH=src .venv-codex/bin/pytest tests/scripts/test_fr3_record.py`
- `PYTHONPATH=src .venv-codex/bin/pytest tests/scripts/test_lerobot_dataset_viz.py`
- Dataset latency comparison:
  `outputs/datasets/fr3_pick_place_ee2ee_v1_20260315_110655` vs `outputs/datasets/fr3_pick_place_ee2ee_v1_20260315_120255`
