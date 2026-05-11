# FR3 `single_cube2_20260429_165325` Replay Validation

Date: 2026-05-07

Dataset:

```bash
data/single_cube2_20260429_165325
```

Current progress:

```text
需要找到轨迹不连续的问题。
```

Hardware validation command:

```bash
python tools/fr3/fr3_teleop_trace_replay.py --mode hardware --dataset data/single_cube2_20260429_165325 --output outputs/fr3_traces/ --validate-only
```

Observed report:

```text
episode_count=45
total_frames=9740
fps=30
ik_solver=hirol_lm
trajectory_xyz_min=[0.22406470775604248, -0.24721138179302216, 0.26846185326576233]
trajectory_xyz_max=[0.8161096572875977, 0.22865943610668182, 0.7111644148826599]
max_step_pos_m=0.4423648655394961
max_step_rot_rad=2.380396575561348
max_ee_speed_mps=13.270958622357275
max_ee_rot_speed_radps=71.41170998217777
bad_pose_replacement_enabled=False
contains_bad_pose=True
invalid_pose_count=433
violation_count=4685
validation=FAIL
```

Root cause:

- The dataset `action` stream is a one-frame-shifted copy of the EE observation stream: for finite rows, `action[t] == observation.state[t+1]` for all 9479 checked pairs.
- This means a bad `observation.state` row also appears as a bad `action` row one frame earlier.
- The invalid poses are `NaN` rows, not quaternion normalization noise.
- The dataset also contains finite but physically implausible one-frame outliers. The worst case is episode 1 around frame 36:

```text
episode=1 frame=35 action xyz ~= [0.569, 0.050, 0.441]
episode=1 frame=36 action xyz ~= [0.224, 0.010, 0.711]
episode=1 frame=37 action xyz ~= [0.567, 0.071, 0.438]
```

That single-frame outlier creates a 0.442 m translation jump and a 2.38 rad rotation jump at 30 Hz.

Invalid pose distribution by episode:

```text
ep21: 2
ep22: 17
ep23: 4
ep24: 14
ep25: 1
ep26: 14
ep27: 21
ep31: 7
ep33: 7
ep34: 10
ep35: 12
ep36: 38
ep37: 34
ep38: 22 action / 21 state
ep39: 13
ep42: 1
```

Rerun validation:

Use the existing dataset visualizer to inspect the trajectory:

```bash
DISPLAY=:1 PYTHONPATH=src python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id local/single_cube2_20260429_165325 \
  --root data/single_cube2_20260429_165325 \
  --episode-index 1 \
  --mode local \
  --num-workers 0 \
  --batch-size 32
```

Episode 1 shows the finite one-frame EE pose outlier around frame 36.

The compact `NaN` pose-span check is episode 22:

```bash
DISPLAY=:1 PYTHONPATH=src python -m lerobot.scripts.lerobot_dataset_viz \
  --repo-id local/single_cube2_20260429_165325 \
  --root data/single_cube2_20260429_165325 \
  --episode-index 22 \
  --mode local \
  --num-workers 0 \
  --batch-size 32
```

The Rerun run also reports:

```text
Action and observation.state share identical feature names; logging actions under 'action_target/'.
```

This confirms the dataset viewer sees the action and state vectors as the same EE pose feature contract, matching the numeric finding that finite `action[t]` equals `observation.state[t+1]`.

Conclusion:

This dataset is not safe for direct full-trajectory hardware replay. The hardware replay path must keep `bad_pose_replacement_enabled=False` and fail validation by default. For real replay, first filter or segment the dataset by removing `NaN` rows and finite outliers, then validate a single episode with explicit safety thresholds before enabling `--execute`.
