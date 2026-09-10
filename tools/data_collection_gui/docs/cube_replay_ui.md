# Cube MuJoCo and real-robot replay

The Episode Replay page supports generated AprilTag cube trajectories from:

```text
<dataset>/derived/april_cube_tracking_in_robot_base/state_action.left.csv
<dataset>/derived/april_cube_tracking_in_robot_base/state_action.right.csv
```

The gateway is expected to run directly on Thor, as started by `run/deploy.sh`.
MuJoCo and hardware replay therefore execute locally on Thor; the hardware path
does not SSH from Thor back into itself.

## Prerequisites

1. Select a finalized recorded dataset and episode in Episode Replay.
2. Generate the EE trajectory in Dataset Processing. The selected cube must have
   a non-empty `state_action.<cube>.csv` for that episode.
3. Thor's `/home/nvidia/Code/infer/.venv-fr3/bin/python3` must provide MuJoCo and
   the hardware-equivalent IK dependencies. `ffmpeg` must be available for the
   browser-playable native MP4.
4. Hardware replay additionally needs `panda_py`, the BOX SDK environment used
   by the Corenetic gripper, network access to the selected FR3 IP, and an
   operator ready to stop the robot.
5. The optional live monitor needs `pyrealsense2`, OpenCV, and a connected
   RealSense camera. Failure of the monitor does not substitute for direct
   observation of the robot.

## MuJoCo replay in Episode Replay

Choose `Left cube`, `Right cube`, or `Both cubes`, then click `Run MuJoCo`.
The gateway runs the selected episode headlessly with the hardware-equivalent IK
solver, creates a native MuJoCo video, and embeds it in the same timeline as the
recorded cameras and telemetry.

Outputs are written beside the generated cube trajectories:

```text
derived/april_cube_tracking_in_robot_base/
  mujoco_preview.<left|right|both>.episode_XXXXXX.json
  mujoco_preview.<left|right|both>.episode_XXXXXX.mp4
```

The validation record contains the selected cube mode. A pass for `left` cannot
unlock real replay of `right`, and a `both` result cannot unlock either
single-arm hardware action.

## Single-cube hardware replay

The Real Robot Replay panel only permits `left` or `right`:

1. Run and pass MuJoCo for the same dataset, episode, FPS, thresholds, and cube.
2. Select that cube in the hardware panel.
3. Enter the FR3 IPv4 address.
4. Keep the workspace clear and an operator at the robot.
5. Click `Run real-robot replay`, verify the dataset/episode/cube/IP confirmation,
   and continue only after reviewing the selected trajectory.

The gateway re-runs hardware preflight against the exact IP entered in the
panel, launches the noninteractive selected episode from
`replay_cube_pose_in_robot_base.py`, and starts the RealSense monitor. Exported
datasets remain blocked because their derived action semantics are not approved
as a verified robot command stream.

Use `Abort` to terminate the replay process group. Treat Abort as a software
control, not as a replacement for the robot's physical emergency stop.

## Frozen P0 native arm replay with project-side checks disabled

The Thor Episode Replay page also exposes a separate **P0 Native Arm Replay · Checks Disabled**
panel for the 2026-09-08 handoff package. This is deliberately independent of
the dataset/cube replay above: it cannot accept a
script path or arbitrary command-line arguments. The GUI entry uses the stored
joint knots directly and is intentionally marked high risk. When an operator
explicitly activates a passing P1 base-relocalization result, it instead uses
the hash-pinned joint knots generated from that result.

The two backend endpoints are:

```text
POST /api/replay/p0-native-check?episode=0&gripper_width_mm=88
POST /api/replay/p0-native-execute?episode=0&gripper_width_mm=88&confirmation=YES
```

Only episodes 0 and 1 and widths from 0 through 89.05 mm are accepted. The
execute endpoint launches exactly:

```text
sudo -n bash /home/nvidia/lerobot/tools/thor/run_p0_native_arm_unchecked.sh \
  <0-or-1> --gripper-width-mm <width> --execute
```

The browser requires the operator to type `YES`; the gateway checks the same
token and supplies it through a pseudo-terminal to the script's TTY-only
confirmation prompt. The gateway must run on Thor, this repository must be
deployed at `/home/nvidia/lerobot`, and the frozen package must exist at its
fixed `/home/nvidia/box_api/replay_p0_native_arm_only_20260908` path.

Unlike the frozen package's original `run_native_arm.sh`, this separate entry
does not load the trajectory-audit, robot-state safety, joint-envelope, TCP, or
collision/scene modules. All fixed replay trajectories are constructed before
the FR3 connection. After connection, panda_py reads the measured joints once
and plans one `start` trajectory at speed factor `0.05` from that pose to the stored first pose.
This start path is not audited or collision checked. After it finishes, the
script does not insert measured positions into, replan, or audit any
`replay_*` chunk. The width is recorded only and
the Corenetic gripper is not commanded. FR3/libfranka protections below this
process still apply; the script does not disable firmware/controller-level
protections.

The GUI and process output repeat these risks before execution. `Abort`
terminates the spawned process group but is not a substitute for the physical
emergency stop.

## P1_simple_eye_hand_calibration

The Calibration page contains `P1_simple_eye_hand_calibration` for relocating
the FR3 base while preserving the canonical camera-rig world
`world_20260819_031843`. It uses the same production intrinsics and camera
extrinsics as the current P0 tracking run, not cam13 plus an auxiliary base
marker.

The fixture contract is two rigid wrist/TCP tags:

- family `tag36h11`, IDs 56 and 57;
- detected black tag size 55 mm;
- 70 mm is backing/overall size metadata and is not passed to PnP;
- each tag must remain rigid against `fr3_ee` (the FR3 flange alias) for the whole run;
- P1 uses a mock gripper backend and never commands the physical gripper.

After the operator types `P1_MOVE_FR3`, the runner reads the 211 measured
`observation.state` / `observation.joints` pairs behind
`thor_gmsl2_extrinisics_robot_base_0720`. The operator identified the current
measured TCP Z=0.100469 m as table contact, so selection requires TCP
Z>=0.250469 m (150 mm clearance). It then keeps candidates that were valid in
at least four production cameras during that extrinsics calibration, restricts
selection to the nearest 2.5x candidate pool using the nearest historically
visible camera center, and uses farthest-point TCP translation/orientation
coverage to select 50 while preserving database order. It uses the database
pose and joints exactly: no pitch transform and no regenerated IK. Every camera
must retain at least five selected observations, and all joints must fit the
FR3/Panda-controller common range before hardware motion. Before opening the recorder it uses the same
FR3-native trajectory controller as P0 to move slowly from measured joints to
the first selected pose at speed factor `0.05`; this avoids the legacy Panda
joint-wall mismatch without relaxing FR3 limits. The regular pose controller
gets up to 12 seconds per pose and the run stops if it cannot reach within
0.02 rad. At each pose the FR3 stops, the measured TCP
pose is saved, and Thor captures the production cameras. The robust solve uses

```text
T_world_tag(i) = T_world_base · T_base_tcp(i) · T_tcp_tag
```

and jointly estimates one `T_world_base` plus separate `T_tcp_tag56` and
`T_tcp_tag57`. A candidate needs at least 15 distinct robot poses, at least two
cameras, optimizer convergence, translation residual RMS no more than 5 mm,
and rotation residual RMS no more than 2 degrees. Failure remains visible as a
failed run and cannot be activated.

A passing result writes a versioned calibration JSON and builds an inactive P0
joint plan. The retarget step preserves every original P0 TCP pose in the
canonical world, re-expresses it in the newly located base, and solves
continuous joint IK. Pressing **Activate** atomically writes:

```text
outputs/calibration/p1_simple_eye_hand_calibration/active.json
```

The unchecked P0 executor verifies the active calibration and plan hashes
before loading them. Missing active state uses the frozen original P0 plan;
malformed, missing, or hash-mismatched active state fails closed instead of
silently using the old plan.

P1 capture itself moves a real FR3 through historical taught poses. Because the
base may have moved relative to the room, those paths are not validated by a
project scene/collision model. Clear the cell and keep the physical E-stop in
hand; GUI Cancel remains only a software process stop.

## Testing without robot motion

With a real dataset available, validate in this order:

1. Confirm the selected episode has finite left/right sidecar poses.
2. Run one single-cube MuJoCo replay and inspect its embedded MP4 and metrics.
3. Run `both` MuJoCo replay and confirm the two trajectories remain in separate
   robot-base viewports.
4. Confirm the hardware panel stays locked for a different cube or episode.
5. Verify FR3 connectivity and preflight independently before permitting motion.
6. Start hardware replay at a conservative workspace setup and stop immediately
   if the initial pose or cube-to-EE transform is unexpected.
