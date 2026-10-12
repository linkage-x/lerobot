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

## Laser vs FR3 replay error (one click)

When a real replay ran with *Record the laser tracker* on, its run directory
holds both halves of a ground-truth comparison:

```text
derived/april_cube_tracking_in_robot_base/real_replay/<UTC stamp>_episode_XXXXXX_<cube>/
  replay_pose_traces/*.csv        desired TCP pose + the FR3's own (FK) pose, per frame
  laser_tracker/lt_*/             landed tracker session (rt.csv, meta.json, sync.csv)
```

The **Laser vs FR3 Replay Error** panel under Real Robot Replay (Thor only)
analyses every such run of the selected episode and cube:

1. Select the dataset and episode, pick the cube, click `Analyse laser runs`.
   The button is disabled while a replay is running (its tracker session has not
   landed yet) and when no run of this episode recorded the tracker.
2. The gateway runs `python -m metrology.fr3_replay_laser_error <runs...>` in a
   subprocess (pure numpy; a few seconds) and writes
   `real_replay/laser_replay_error_episode_XXXXXX_<cube>.json`, plus
   `laser_replay_error.json` in each run directory. The panel reloads the report
   by itself; it is also shown again after a gateway restart.

What the numbers are (all in mm, at the SMR point, on the frames the beam saw,
minus 10 frames before each loss):

| Column | Meaning |
|---|---|
| Laser vs desired | the replay error that really happened |
| FK vs desired | what the trace alone says (`target_position_error`) |
| Laser vs FK | how far the FR3's own readback is off; a lower bound, because the tracker-to-base registration is fitted to the FK |
| Offset / Lag / Rest | laser vs desired split into a constant offset, the follow lag (frames the arm trails the target), and what is left |

The registration fits the tracker-to-base transform and the SMR position in the
TCP frame together (Gauss–Newton, 9 parameters). The SMR offset is reported with
its formal σ and must agree across runs if the SMR was not reseated. Run-to-run
repeatability compares the laser points of all analysed runs frame by frame.

The beam-loss line reads the line of sight in the gripper frame. A loss at the
edge of a small swing points at the SMR (aperture turned away from the tracker,
or something on the gripper clipping the beam): face the SMR at the tracker at
the trajectory start. A loss well inside the directions already seen points at
something in the scene crossing the beam.

Reference (2026-10-10, 1009 ep0 right, three runs): laser vs FK 0.27 mm RMS;
laser vs desired 7.8 mm RMS = 5.5 mm downward sag + 150 ms follow lag + 2.5–3 mm.
See W3f in `docs/box_umi_laser_tracker_gt_roadmap.html`.
