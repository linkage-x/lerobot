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
   the hardware-equivalent IK dependencies. The browser preview uses the JSON
   body-pose preview stream plus Three.js/WebGL; it no longer requires `ffmpeg` or an MP4.
4. Hardware replay additionally needs `panda_py`, the BOX SDK environment used
   by the Corenetic gripper, network access to the selected FR3 IP, and an
   operator ready to stop the robot.
5. The optional live monitor needs `pyrealsense2`, OpenCV, and a connected
   RealSense camera. Failure of the monitor does not substitute for direct
   observation of the robot.

## MuJoCo replay in Episode Replay

Choose `Left cube`, `Right cube`, or `Both cubes`, then click `Run MuJoCo`.
The gateway runs the selected episode headlessly with the hardware-equivalent IK
solver, streams per-frame MuJoCo poses through gateway memory, writes structured
metrics at completion, and the browser renders the live pose stream with Three.js/WebGL
in the same timeline as the recorded cameras
and telemetry.

Outputs are written beside the generated cube trajectories:

```text
derived/april_cube_tracking_in_robot_base/
  mujoco_preview.<left|right|both>.episode_XXXXXX.json
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
2. Run one single-cube MuJoCo replay and inspect its Three.js pose preview and metrics.
3. Run `both` MuJoCo replay and confirm the two trajectories remain in separate
   robot-base viewports.
4. Confirm the hardware panel stays locked for a different cube or episode.
5. Verify FR3 connectivity and preflight independently before permitting motion.
6. Start hardware replay at a conservative workspace setup and stop immediately
   if the initial pose or cube-to-EE transform is unexpected.
