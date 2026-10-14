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

Validations are stored in `meta/gui_replay_validations.json`. Selecting a
dataset or episode restores the newest one for it. One for the current cube mode
wins; if there is none, the gateway switches to the stored run's cube. A
restarted gateway is back at `left`, and before this rule it silently dropped a
passed `right` run, which left Move to trajectory start greyed out.

A stored validation is only restored for the robot model it ran on: it records a
digest of `fr3_corenetic_gripper.urdf` and `.xml`. MuJoCo plans the start joints
for the model's TCP, so after any model change (2026-10-12: the TCP moved from
191 to 172 mm below the flange; 2026-10-13: `corenetic_socket_ee` added) run MuJoCo again.

The trajectory is driven by the frame named in `robot.target_frame_name` of
`replay_cube_pose_in_robot_base.thor.yaml`. The gateway passes the same frame to
MuJoCo (`--target-frame-name`), and the IK QC config uses it too. Since 2026-10-13 it is
`corenetic_socket_ee`: the pivot insert's socket centre, (1.45, −0.78, −5.06) mm
in `corenetic_gripper_ee` (5 mm up the tool axis), with the same axes. Until 2026-10-14 it was
z −10.3 mm, fitted end to end with the laser tracker; that fit had absorbed about 5 mm of the
world → fr3_base edge. The FR3 pivot about a fixed ball and a caliper (flange face → socket
166.9 mm) agree on −5.06. The camera trajectories put their
TCP at that socket, while `corenetic_gripper_ee` sits at the fingertip: with the gripper closed,
a tape measured about 175 mm from the flange to the nose. Driving the EE frame put every
replayed pose about 10 mm out along the tool, which was 15.7 mm p95 end to end; on the socket
frame at −10.3 it was 7.7 mm. At −5.06 the ~5 mm of edge error that −10.3 hid is back in the
end-to-end number until the world → fr3_base edge is fixed. The replay refuses a socket frame that is not a pure translation of
the EE frame.

## Single-cube hardware replay

The Real Robot Replay panel only permits `left` or `right`:

1. Run and pass MuJoCo for the same dataset, episode, FPS, thresholds, cube, and
   robot model.
2. Select that cube in the hardware panel.
3. Enter the FR3 IPv4 address.
4. Keep the workspace clear and an operator at the robot.
5. Click `Run real-robot replay`, verify the dataset/episode/cube/IP confirmation,
   and continue only after reviewing the selected trajectory.

Check the end-effector load in Franka Desk before replaying. The robot
compensates gravity with the Desk load, and the replay controller is a soft joint
impedance (`replay_cube_pose_in_robot_base.thor.yaml`: stiffness
[300, 300, 300, 300, 120, 80, 30], half the panda_py default). A wrong load
shows up as a constant offset: on 2026-10-10 a load 0.36 kg too light dropped the
TCP by 5.7 mm. Use **Corenetic v2.0 BOX + cam + SMR** (1.1108 kg) while the
SMR sits on the gripper, and **Corenetic v2.0 BOX + cam** (0.9755 kg) otherwise
(see `gripper_w_v2_franka_description/README.md`).

*Command lead (frames)* sends each pose that many frames early along the same
path (`replay.pose_command_lead_frames`; default 3 since 2026-10-12, 0 = off, at
most 10; one frame is 50 ms at the Thor config's 20 fps). The YAML still says 0,
so a command-line replay on Thor runs without lead unless it passes the flag. The gateway always passes it explicitly,
so a value left in the YAML on Thor never applies silently. The trace keeps
`desired_*` on the dataset timeline and logs the lead per frame, so the laser
analysis still scores against the trajectory and lists each run's lead.

The gateway re-runs hardware preflight against the exact IP entered in the
panel, launches the noninteractive selected episode from
`replay_cube_pose_in_robot_base.py`, and starts the RealSense monitor. Exported

The gripper follows the recorded BOX opening and trigger, which the gateway copies
from the episode parquet into `fr3_base/state_action.<cube>.csv` (see `frame.json`:
`gripper_width.source` / `rows`). If several BOXes were online while recording, the
names carry a `box<device id>.` prefix. The arm's own BOX (`robot.corenetic_device_id`
in the replay YAML) is left out, and the one remaining BOX is used. If more than one
remains, nothing is guessed. When a trajectory carries no opening, the gateway logs a
warning that the gripper will not move. Before 2026-10-13 such a replay ran with the
gripper never commanded and nothing said so.
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

Only compare runs made with the same robot model and target frame. The SMR offset
is expressed in the frame the desired path drives (`corenetic_socket_ee` since
2026-10-13; `corenetic_gripper_ee` before), so changing that frame moves both: the
SMR z moved by 10.5 mm on 2026-10-13 (-65.8 to -55.3). After the 2026-10-12 URDF change the SMR z moved by 19.2 mm. Mixing
those runs with the 10-10 ones gave a meaningless "repeatability" of 18 mm.

Laser vs FK grows with the orientation range the fit has to cover: one rigid
registration cannot absorb the FR3's nominal kinematic error over a wider range.
On the same run, the full trajectory (27°) gave 0.51 mm and its first half (18°)
gave 0.25 mm.

The beam-loss line reads the line of sight in the gripper frame. A loss at the
edge of a small swing points at the SMR (aperture turned away from the tracker,
or something on the gripper clipping the beam): face the SMR at the tracker at
the trajectory start. A loss well inside the directions already seen points at
something in the scene crossing the beam.

References (1009 ep0 right; W3f in `docs/box_umi_laser_tracker_gt_roadmap.html`):

| Run | Beam | Laser vs desired | Offset (x, y, z) | Laser vs FK | Rest after offset + 150 ms lag |
|---|---|---|---|---|---|
| 2026-10-10, three runs, Desk load 0.7496 kg | first half | 7.7–7.9 | (0.1, −1.6 ~ −2.1, −5.5) | 0.27 | 2.5–3.0 |
| 2026-10-12, Desk load 1.1108 kg, v2 URDF | 920 / 920 | 5.31 (p95 8.9) | (0.0, 0.2, 0.7) | 0.51 (first half 0.25) | 2.6 |
| 2026-10-12, same, repeat | 920 / 920 | 5.40 (p95 8.8) | (0.0, 0.1, 0.7) | 0.51 | 2.8 |
| 2026-10-12, same, **command lead 3**, two runs | 920 / 920 | 2.68–2.80 (p95 4.2–4.8) | (0.0, −0.1, 0.4–0.5) | 0.51–0.52 | 2.6–2.8 |

The 10-10 sag was the Desk load, not the controller. What is left is mostly the
150 ms lag: by speed, 3.1 mm below 10 mm/s and 7.3 mm above 40 mm/s.
The lag comes from the controller sending positions only: panda_py's
`JointPosition` is τ = K(q_d − q) + D(dq_d − dq) with dq_d = 0, which trails by
D/K = 83 ms (the same for the Thor gains and the panda_py defaults, so stiffer
gains do not shorten it). The 20 Hz step targets, each braked to zero velocity by
Ruckig, add the rest. An offline model of the 10-12 run predicts that a 3-frame
lead brings laser vs desired to about 2.6 mm while the rest stays at about 2.4 mm. The two
lead-3 runs measured 2.80 and 2.68 mm with no lag left, and the error no longer
grows with speed (2.3–2.9 mm above 10 mm/s). What is left repeats run to run
(two lead-3 runs differ by 0.62 mm RMS) and is not the Desk payload. The report's
"repeatability over N runs" mixes every run of the episode: compare runs with the
same lead (the Lead column) and the same robot model only.

Correcting each command by the error a previous run of the same trajectory measured
(iterative learning) would take this to about 1 mm RMS, but it is not implemented:
none of the current uses need the arm's path that close to the recording, and it
costs a second run per trajectory. See W3f in `docs/box_umi_laser_tracker_gt_roadmap.html`
for the trade-off and when to build it.
