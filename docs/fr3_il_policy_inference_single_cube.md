# FR3 single-cube IL policy inference

This note describes how to run a trained single-cube IL policy on the real FR3
hardware stack with selectable Hikrobot/OpenCV/RealSense/GMSL2 cameras and a
Pika/DAS/Franka Hand/Corenetic gripper. It is the inference companion to
`docs/fr3_il_policy_training_single_cube.md`.

## What Must Match Training

Use the training run's checkpoint and dataset view together. The checkpoint
contains the policy config; the dataset view contains the exact observation and
action feature names.

Required alignment points:

- `checkpoint`: usually `outputs/train/<job_name>/checkpoints/last`
- `dataset_root`: the generated view under `outputs/datasets/<job_name>`
- `camera keys`: camera config entries must match the policy image suffixes,
  for example `observation.images.cam_1` requires camera config key `cam_1`
- `image resolution`: if training used `--image-resize-shape H,W`, the policy
  image features in the checkpoint will use `(3, H, W)`. The runtime resizes
  live camera frames to that shape before inference.
- `observation.state`: default single-cube view is 7D
  `[ee.x, ee.y, ee.z, ee.qx, ee.qy, ee.qz, ee.qw]`
  or absent for image-only ACT runs trained with `--state-keys none`. If
  training used a selector such as
  `--state-keys observation.state.right,observation.state_raw:handheld_gripper.pika_left.width_mm`,
  the generated `observation.state` is 8D:
  `[observation.state.right.ee.x, ..., observation.state.right.ee.qw, observation.state_raw.handheld_gripper.pika_left.width_mm]`.
  The real-robot runtime maps these prefixed names back to the live FR3 EE pose
  and current gripper width.
- `action`: default single-cube view is 8D
  `[ee.x, ee.y, ee.z, ee.qx, ee.qy, ee.qz, ee.qw, gripper]`
- `gripper`: the helper appends
  `observation.state_raw:handheld_gripper.pika_left.width_mm` to the action and
  names that output dimension `gripper`

Check the generated manifest before deployment:

```bash
jq '.' outputs/datasets/<job_name>/meta/il_view_manifest.json
jq '.features."observation.state", .features.action' outputs/datasets/<job_name>/meta/info.json
```

For image-only ACT runs, `observation.state` is absent from the policy view.
The runtime still needs EE state for dataset-world alignment. For single-source
training views, it reads `source_dataset_root` from
`meta/il_view_manifest.json` and estimates the start pose from the original
dataset. Multi-source training views record `source_dataset_roots`; use a
training view that keeps `observation.state` for real-robot deployment, or
extend the runtime alignment logic before deploying an image-only multi-source
view.

The training helper also writes:

```text
outputs/datasets/<job_name>/inference_config.generated.yaml
```

Prefer using this file as the inference entrypoint. It stores the training-side
contract: dataset view, expected checkpoint path, camera keys, image resize
shape, state keys, action layout, and gripper action append. Hardware fields in
the YAML are defaults and can be overridden from the command line.

The launcher also accepts runtime-only options in the same YAML, for example:

```yaml
runtime:
  startup:
    robot_init_state:
      type: joints
      joints_rad: [-0.05, -1.56, -1.72, -2.12, 0.01, 2.12, -0.97]
      gripper: 0.5
  interactive:
    enabled: true
    start_key: s
    stop_key: x
    quit_key: q
  mujoco:
    enabled: true
    model: src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper_ati.xml
    max_chunk_points: 64
```

CLI flags still take priority over values loaded from
`--inference-config`.

## Camera Config

The runtime reads cameras from a YAML file passed with `--camera-config`. The
keys under `robot.cameras` must be the same suffixes used during training. If
you trained with:

```bash
--cameras cam_1,cam_3
```

then the inference camera config must contain `cam_1` and `cam_3`.

Example Hikrobot camera config:

```yaml
robot:
  cameras:
    cam_1:
      type: hikrobot
      serial: "YOUR_CAM_1_SERIAL"
      image_shape: [720, 1280]
      fps: 30
      color_mode: BGR
      transport_layer: usb
      exposure_us: 8000
      gain_db: 0
      timeout_ms: 1000
    cam_3:
      type: hikrobot
      serial: "YOUR_CAM_3_SERIAL"
      image_shape: [720, 1280]
      fps: 30
      color_mode: BGR
      transport_layer: usb
      exposure_us: 8000
      gain_db: 0
      timeout_ms: 1000
```

Save this as, for example:

```text
tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml
```

The runtime converts BGR Hikrobot frames to RGB before passing them to the
policy. If the camera is configured or captured at a larger resolution than the
policy image feature shape, the runtime downsamples the live frame to the policy
shape. For example, a policy trained with `--image-resize-shape 360,640` receives
`360x640` frames even if the Hikrobot stream is configured as `1080p`.

At startup, the runtime validates camera keys against the checkpoint policy
metadata. If the checkpoint expects `observation.images.cam_1`, the camera
config must contain `robot.cameras.cam_1`; otherwise inference fails before the
robot loop starts. During every policy step, `robot.get_observation()` reads the
robot state and cameras, the runtime converts configured BGR streams to RGB,
resizes images to the checkpoint feature shape, and builds
`observation.images.<camera_key>` for the policy.

## Preview Run

Start with preview mode. It loads the policy, opens the selected cameras,
constructs policy observations, predicts actions, applies safety checks, and
prints targets without sending robot commands.

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --inference-config outputs/datasets/<job_name>/inference_config.generated.yaml \
  --camera-config tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml \
  --gripper-backend pika \
  --gripper-port /dev/ttyUSB0 \
  --robot-ip 192.168.1.208 \
  --preview \
  --max-steps 20 \
  --debug-step0-dump-dir outputs/debug/<job_name>_step0
```

Inspect the step-0 dump:

```bash
jq '.' outputs/debug/<job_name>_step0/metadata.json
ls outputs/debug/<job_name>_step0
```

This dump contains the exact `observation.state`, selected camera frames, state
names, action names, and start-alignment diagnostics used by the policy.

## GMSL2 Cameras

GMSL2 camera config uses the same `robot.cameras.<key>` contract. The `<key>`
must still match training, for example a policy trained with
`--cameras gmsl2_front,gmsl2_wrist` expects config keys `gmsl2_front` and
`gmsl2_wrist`.

Example:

```yaml
robot:
  cameras:
    gmsl2_front:
      type: gmsl2
      sensor_id: 0
      device: /dev/video0
      pipeline: v4l2_bayer
      image_shape: [720, 1280]
      fps: 30
      color_mode: bgr
      rotation: no_rotation
      sync_role: auto
      trig_pin: "0x00020007"
      apply_sync_at_connect: true
      timeout_ms: 2000
```

The repo includes a starter file:

```text
tools/fr3/fr3_il_infer_gmsl2_corenetic_camera_config.yaml
```

On the Jetson/Thor host, install the GStreamer/PyGObject dependencies required
by `src/lerobot/cameras/gmsl2`, configure `/dev/video*` and trigger mode first,
then verify the camera stream before real rollout.

## Corenetic Gripper

The Corenetic gripper backend uses `tools/thor/box_sdk/box_client.py` as the
vendor transport/control API and maps the policy gripper output to
`box_sdk.Box.set_clamp_pos(distance_m)`. The policy contract stays the same:
action names still end in `gripper`, and live state still provides
`gripper.pos` normalized to `[0, 1]`.

Use `--gripper-max-width-mm` to match the physical Corenetic gripper aperture.
If the BOX tool TCP differs from the Pika/DAS TCP, also pass a calibrated FR3
tool URDF and target frame:

```bash
FR3_INFER_CAMERA_CONFIG=tools/fr3/fr3_il_infer_gmsl2_corenetic_camera_config.yaml \
FR3_GRIPPER_BACKEND=corenetic \
FR3_GRIPPER_MAX_WIDTH_MM=98 \
FR3_CORENETIC_BIND_IP=192.168.2.45 \
FR3_CORENETIC_REMOTE_IP=192.168.2.60 \
FR3_ROBOT_URDF_PATH=src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper.urdf \
FR3_TARGET_FRAME_NAME=corenetic_gripper_ee \
tools/fr3/run_pick_place_infer_host.sh preview
```

If `FR3_ROBOT_URDF_PATH` and `FR3_TARGET_FRAME_NAME` are omitted, the runtime
falls back to the Pika FR3 URDF/TCP for IK. That is useful for software smoke
tests but should be replaced before evaluating real success rate with a
different physical gripper geometry.

## Real Robot Run

After preview looks sane, run with conservative safety gates:

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --inference-config outputs/datasets/<job_name>/inference_config.generated.yaml \
  --camera-config tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml \
  --gripper-backend pika \
  --gripper-port /dev/ttyUSB0 \
  --robot-ip 192.168.1.208 \
  --max-steps 300 \
  --first-frame-max-pos-delta-mm 20 \
  --first-frame-max-rot-delta-deg 8 \
  --max-step-pos-delta-mm 3 \
  --max-step-rot-delta-deg 2
```

By default the launcher moves the arm to the DAS start joint configuration and
aligns the gripper to the dataset-start mean. Disable those only when you have
manually prepared the start state:

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --inference-config outputs/datasets/<job_name>/inference_config.generated.yaml \
  --camera-config tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml \
  --gripper-backend pika \
  --no-move-to-das-start \
  --no-align-gripper-to-dataset-start \
  --preview \
  --max-steps 20
```

## Startup State

Use `--robot-init-state` when each real rollout should start from a specific
robot state. This is preferred over relying on the legacy DAS start pose,
especially for Pika gripper single-cube policies.

Joint-angle shorthand:

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --inference-config outputs/datasets/<job_name>/inference_config.generated.yaml \
  --camera-config tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml \
  --gripper-backend pika \
  --robot-init-state 'joints=-0.05,-1.56,-1.72,-2.12,0.01,2.12,-0.97'
```

EE-pose shorthand with quaternion `[x,y,z,qx,qy,qz,qw]`:

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --inference-config outputs/datasets/<job_name>/inference_config.generated.yaml \
  --camera-config tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml \
  --gripper-backend pika \
  --robot-init-state 'ee_xyzquat=0.4,0.0,0.3,0.0,0.0,0.0,1.0'
```

YAML file format:

```yaml
robot_init_state:
  type: joints
  joints_rad: [-0.05, -1.56, -1.72, -2.12, 0.01, 2.12, -0.97]
  gripper: 0.5
  timeout_s: 20
  joint_tolerance_rad: 0.01
  gripper_tolerance: 0.02
```

Run with:

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --inference-config outputs/datasets/<job_name>/inference_config.generated.yaml \
  --camera-config tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml \
  --gripper-backend pika \
  --robot-init-state configs/fr3_single_cube_init.yaml
```

If `--robot-init-state` is set, the runtime skips the default
`move_to_das_start` startup motion and moves to the requested state instead.

## Interactive Rollouts

Interactive mode waits for keyboard input before each rollout and lets the
operator stop the current rollout without exiting the process.

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --inference-config outputs/datasets/<job_name>/inference_config.generated.yaml \
  --camera-config tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml \
  --gripper-backend pika \
  --interactive-rollouts \
  --robot-init-state 'joints=-0.05,-1.56,-1.72,-2.12,0.01,2.12,-0.97'
```

Default keys:

- `s`: start the next rollout
- `x`: stop the current rollout and return to the waiting state
- `q`: quit inference

When `robot_init_state` is configured, every new rollout first moves the robot
back to that initial state, then waits for `s`. Key listening uses
`sshkeyboard` when installed and falls back to raw TTY stdin in the Docker
interactive terminal.

## MuJoCo Viewer

Use `--mujoco-viewer` to open a passive MuJoCo visualization during real
inference:

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --inference-config outputs/datasets/<job_name>/inference_config.generated.yaml \
  --camera-config tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml \
  --gripper-backend pika \
  --mujoco-viewer
```

For Pika gripper, `--gripper-backend pika` defaults the viewer model to:

```text
src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper_ati.xml
```

You can also specify it explicitly:

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --inference-config outputs/datasets/<job_name>/inference_config.generated.yaml \
  --camera-config tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml \
  --gripper-backend pika \
  --mujoco-viewer \
  --mujoco-model src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper_ati.xml \
  --mujoco-max-chunk-points 64
```

Viewer semantics:

- the MuJoCo FR3 joint state is synchronized from the real robot
- orange cube: current real EE pose
- green cube: current policy target EE pose
- gradient trajectory: all EE target points in the latest ACT action chunk,
  blue to pink indicating temporal order

These markers are MuJoCo passive viewer overlay geoms, not dynamic mocap bodies,
so they do not affect robot control.

## Full Operator Template

For a Pika single-cube policy, the usual real-run command is:

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --inference-config outputs/datasets/<job_name>/inference_config.generated.yaml \
  --camera-config tools/fr3/fr3_il_infer_hikrobot_camera_config.yaml \
  --gripper-backend pika \
  --gripper-port /dev/ttyUSB0 \
  --robot-ip 192.168.1.208 \
  --interactive-rollouts \
  --robot-init-state 'joints=-0.05,-1.56,-1.72,-2.12,0.01,2.12,-0.97' \
  --mujoco-viewer \
  --first-frame-max-pos-delta-mm 20 \
  --first-frame-max-rot-delta-deg 8 \
  --max-step-pos-delta-mm 3 \
  --max-step-rot-delta-deg 2
```

## Runtime Contract

The current real-hardware runtime is:

- host launcher: `tools/fr3/fr3_act_infer_real.py`
- container runtime: `tools/fr3/fr3_act_infer_real_runtime.py`
- Docker Compose service: `lerobot-infer-fr3-act`

Despite the launcher name, the runtime loads policy type from the checkpoint's
saved config and calls the standard `policy.select_action(...)` path. ACT is the
tested path. Diffusion Policy should use the same observation/action contract,
but treat it as needing preview validation before real execution.

Action decoding accepts these gripper names:

- `gripper`
- `gripper.pos`
- `observation.state_raw.handheld_gripper.pika_left.width_mm`
- `observation.state_raw.handheld_gripper.pika_right.width_mm`

The training helper now writes the appended left-gripper action as `gripper`.

## Safety Checklist

Before removing `--preview`, verify:

- `policy_image_keys` printed by the runtime match your selected cameras.
- `state_names` in the step-0 dump match the training view.
  For the right-arm proprio+1D-gripper run, it should include
  `observation.state.right.ee.*` and end with
  `observation.state_raw.handheld_gripper.pika_left.width_mm`. For image-only
  ACT, `state_names` should be empty.
- `action_names` are 8D and end in `gripper`.
- dumped policy camera arrays have the expected resized shape, for example
  `(360, 640, 3)` for a `--image-resize-shape 360,640` training run.
- `dataset_start_alignment` nearest position and rotation errors are plausible.
- preview predicted EE targets are close enough to current EE pose.
- preview predicted `gripper` values are within expected dataset aperture range.
- if using `--robot-init-state`, the robot reaches that state before the first
  policy step.
- if using `--mujoco-viewer`, the simulated FR3 tracks the real robot joints and
  the orange/green EE cubes appear in plausible locations.

If any of these fail, stop and fix the config instead of lowering safety gates.

## Host Script With A New Checkpoint

`tools/fr3/run_pick_place_infer_host.sh` keeps the old image-only checkpoint as
its default. For a new selector-trained job, override the checkpoint and dataset
view with environment variables:

```bash
FR3_INFER_CHECKPOINT=outputs/train/pick_place_act_pika_right_opencv_realsense_proprio_gripper1d/checkpoints/060000 \
FR3_INFER_DATASET_ROOT=outputs/datasets/pick_place_act_pika_right_opencv_realsense_proprio_gripper1d \
tools/fr3/run_pick_place_infer_host.sh preview
```

Use `real_debug` after preview passes to enable interactive rollouts plus the
MuJoCo current/target/chunk viewer:

```bash
FR3_INFER_CHECKPOINT=outputs/train/pick_place_act_pika_right_opencv_realsense_proprio_gripper1d/checkpoints/060000 \
FR3_INFER_DATASET_ROOT=outputs/datasets/pick_place_act_pika_right_opencv_realsense_proprio_gripper1d \
tools/fr3/run_pick_place_infer_host.sh real_debug
```
