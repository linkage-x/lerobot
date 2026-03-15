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

## Targeted Validation

- `PYTHONPATH=src .venv-codex/bin/pytest tests/configs/test_plugin_loading.py`
- `PYTHONPATH=src .venv-codex/bin/pytest tests/robots/test_franka_research3.py`
- `PYTHONPATH=src .venv-codex/bin/pytest tests/scripts/test_fr3_record_runtime.py`
- `PYTHONPATH=src .venv-codex/bin/pytest tests/scripts/test_lerobot_dataset_viz.py`
