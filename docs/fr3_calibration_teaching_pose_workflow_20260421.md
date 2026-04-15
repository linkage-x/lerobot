# FR3 Calibration Teaching/Capture Workflow (2026-04-21)

## Goal

This note documents four FR3 scripts under `tools/fr3/calibration` and how to run them:

- `09_teaching_mode_lerobot.py`: all-in-one manual teaching recorder (state + optional images) in LeRobot format.
- `teaching_pose_recorder.py`: lightweight waypoint recorder, saves only robot states to one JSON file.
- `execute_pose_and_capture.py`: executes poses from JSON and records **actual** robot states + camera images into LeRobot format.
- `guided_planar_circle_error_test.py`: guided initial-point selection + one-round planar circle capture for calibration error testing.

Recommended workflow for camera calibration data capture:

1. Use `teaching_pose_recorder.py` to build a fixed-point pose library.
2. Optionally post-process the JSON pose library (augment/perturb/reorder).
3. Use `execute_pose_and_capture.py` to run those poses and capture dataset frames.

Additional error-test workflow:

1. Use `guided_planar_circle_error_test.py` with hand-guiding to set a start point.
2. Press `s` to run one XY circle while recording synchronized robot states and camera frames.
3. Use the generated LeRobot dataset for calibration residual/error analysis.

## Programs

### `tools/fr3/calibration/teaching_pose_recorder.py`

Purpose:

- Press `r` to record current robot EE pose and joints.
- Print each recorded state immediately.
- Save all records into one JSON file.

Default config:

- `tools/fr3/calibration/teaching_pose_recorder.yaml`
- output JSON: `/workspace/outputs/datasets/fr3_teaching_pose_recorder/teaching_pose_records.json`

Key bindings:

- `r`: record one state
- `p`: print current EE pose
- `j`: print current joints
- `c`: print controller state
- `h`: home and re-enter teaching mode
- `q`: save and quit

### `tools/fr3/calibration/execute_pose_and_capture.py`

Purpose:

- Read pose records from JSON.
- Execute all poses one by one.
- Wait for settle.
- Capture camera images + actual robot observation.
- Save episode into LeRobot dataset.

Important runtime behavior:

- `control_mode: joint_space` (default in example YAML): execute by recorded joints, no IK.
- Uses incremental steps by default to reduce reflex abort risk (`joint_space` or `ee_pose`).
- Supports retry on reflex abort (`joint_velocity_violation`) with optional `move_to_start()` recovery.
- `home_before_start: true` by default in example YAMLs.

Default configs:

- RealSense: `tools/fr3/calibration/execute_pose_and_capture_realsense.yaml`
- Hikrobot: `tools/fr3/calibration/execute_pose_and_capture_hikrobot.yaml`

### `tools/fr3/calibration/09_teaching_mode_lerobot.py`

Purpose:

- Legacy all-in-one keyboard recorder.
- Records manual samples directly into LeRobot format (robot state + optional images).

Default config:

- `tools/fr3/calibration/09_teaching_mode_lerobot.yaml`

### `tools/fr3/calibration/guided_planar_circle_error_test.py`

Purpose:

- Put robot into guiding mode (zero stiffness/damping), then manually drag to a desired initial point.
- Press `s` to lock initial point and start one planar circle in XY.
- Capture robot state (`observation.state`, `observation.joints`) and camera frames at each step.
- Save one episode in LeRobot format plus an execution summary JSON.

Default configs:

- RealSense: `tools/fr3/calibration/guided_planar_circle_error_test_realsense.yaml`
- Hikrobot: `tools/fr3/calibration/guided_planar_circle_error_test_hikrobot.yaml`

Keyboard:

- `s`: start circle from current guided point
- `p`: print current EE pose
- `j`: print current joints
- `h`: home and return to guiding mode
- `q`: quit without recording

## Docker Commands

To avoid root-owned output files on host, run with host UID/GID:

### 1) Lightweight pose recording

```bash
docker compose -f docker/docker-compose.yml run --rm \
  --user "$(id -u):$(id -g)" \
  lerobot-user \
  bash -lc 'cd /workspace && PYTHONPATH=/workspace/src /lerobot/.venv/bin/python tools/fr3/calibration/teaching_pose_recorder.py --config_path tools/fr3/calibration/teaching_pose_recorder.yaml'
```

### 2) Execute poses + capture (RealSense)

```bash
docker compose -f docker/docker-compose.yml run --rm \
  --user "$(id -u):$(id -g)" \
  lerobot-user \
  bash -lc 'cd /workspace && PYTHONPATH=/workspace/src /lerobot/.venv/bin/python tools/fr3/calibration/execute_pose_and_capture.py --config_path tools/fr3/calibration/execute_pose_and_capture_realsense.yaml'
```

### 3) Execute poses + capture (Hikrobot)

```bash
docker compose -f docker/docker-compose.yml run --rm \
  --user "$(id -u):$(id -g)" \
  lerobot-user \
  bash -lc 'cd /workspace && PYTHONPATH=/workspace/src /lerobot/.venv/bin/python tools/fr3/calibration/execute_pose_and_capture.py --config_path tools/fr3/calibration/execute_pose_and_capture_hikrobot.yaml'
```

### 4) Legacy all-in-one recorder

```bash
docker compose -f docker/docker-compose.yml run --rm \
  --user "$(id -u):$(id -g)" \
  lerobot-user \
  bash -lc 'cd /workspace && PYTHONPATH=/workspace/src /lerobot/.venv/bin/python tools/fr3/calibration/09_teaching_mode_lerobot.py --config_path tools/fr3/calibration/09_teaching_mode_lerobot.yaml'
```

### 5) Guided planar-circle error test (RealSense)

```bash
docker compose -f docker/docker-compose.yml run --rm \
  --user "$(id -u):$(id -g)" \
  lerobot-user \
  bash -lc 'cd /workspace && PYTHONPATH=/workspace/src /lerobot/.venv/bin/python tools/fr3/calibration/guided_planar_circle_error_test.py --config_path tools/fr3/calibration/guided_planar_circle_error_test_realsense.yaml'
```

### 6) Guided planar-circle error test (Hikrobot)

```bash
docker compose -f docker/docker-compose.yml run --rm \
  --user "$(id -u):$(id -g)" \
  lerobot-user \
  bash -lc 'cd /workspace && PYTHONPATH=/workspace/src /lerobot/.venv/bin/python tools/fr3/calibration/guided_planar_circle_error_test.py --config_path tools/fr3/calibration/guided_planar_circle_error_test_hikrobot.yaml'
```

## Outputs

Typical output paths:

- Pose library JSON: `/workspace/outputs/datasets/fr3_teaching_pose_recorder/teaching_pose_records.json`
- Execute/capture dataset root (RealSense example): `/workspace/outputs/datasets/fr3_execute_pose_capture_realsense`
- Execute/capture report JSON (RealSense example): `/workspace/outputs/datasets/fr3_teaching_pose_recorder/execute_pose_and_capture_report_realsense.json`
- Guided-circle dataset root (RealSense example): `/workspace/outputs/datasets/fr3_guided_planar_circle_error_test_realsense`
- Guided-circle summary JSON (RealSense example): `/workspace/outputs/datasets/fr3_guided_planar_circle_error_test_realsense/summary.json`

## Notes

- For Hikrobot config, replace placeholder serials in `tools/fr3/calibration/execute_pose_and_capture_hikrobot.yaml`.
- For Hikrobot guided-circle config, replace placeholder serials in `tools/fr3/calibration/guided_planar_circle_error_test_hikrobot.yaml`.
- If reflex abort still happens, tune `execution.max_translation_step_m`.
- If reflex abort still happens, tune `execution.max_rotation_step_deg`.
- If reflex abort still happens, tune `execution.command_interval_s`.
- If reflex abort still happens, tune `execution.max_command_steps`.
- For guided-circle speed tuning, tune `motion.duration_s`, `motion.command_interval_s`, and `motion.radius_m`.
