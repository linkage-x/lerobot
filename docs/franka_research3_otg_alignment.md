# Franka Research 3 OTG Alignment Note

## Scope

This note records the current conclusion for the `franka_research3` teleoperation
path in this repository after comparing it against the HIROL reference:

- `/home/hanyu/Codes/HIROLRobotPlatform/teleop/teleoperation.py`
- `/home/hanyu/Codes/HIROLRobotPlatform/teleop/config/left_fr3_with_pika_ati_ik_3d_mouse.yaml`
- `/home/hanyu/Codes/HIROLRobotPlatform/factory/components/motion_configs/left_fr3_with_pika_ati_ik.yaml`
- `/home/hanyu/Codes/HIROLRobotPlatform/smoother/config/smoother_config_fr3_.yaml`

It focuses on the minimal FR3 teleoperation runtime only. It does not introduce
HIROL factory integration.

## HIROL Reference Behavior

The HIROL reference path uses:

- a teleoperation loop configured at `0.005s` in the teleop YAML
- `ruckig` as the joint-space smoother
- smoother internal frequency `800Hz`
- async sender frequency `1000Hz`
- target updates decoupled from high-frequency command sending

Important nuance:

- the `0.005s` teleoperation loop is a front-end teleop update rate
- the `800Hz` smoother and `1000Hz` sender belong to the robot motion backend
- these should not be collapsed into one frequency knob

## Current lerobot Alignment

The `franka_research3` backend in this repository now aligns to the HIROL core
behavior in the following way:

- `ruckig` is used for FR3 joint-space OTG after IK
- smoother limits match the HIROL FR3 ruckig config
- smoother frequency is `800Hz`
- async sender frequency is `1000Hz`
- `send_action()` only updates the latest joint target
- a background OTG loop continuously advances the smoother state
- a separate background sender loop continuously sends the latest joint command

This means the current runtime model is:

`teleop target update -> IK target -> 800Hz OTG smoother -> 1000Hz joint sender`

## Current Config Values

The aligned FR3 runtime defaults are:

- `use_otg: true`
- `otg_control_frequency: 800.0`
- `otg_async_control_frequency: 1000.0`
- `otg_max_velocity: [2.096, 2.096, 2.096, 2.096, 4.208, 3.344, 4.208]`
- `otg_max_acceleration: [8.0, 8.0, 8.0, 8.0, 8.0, 8.0, 8.0]`
- `otg_max_jerk: [4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0, 4000.0]`
- `otg_min_position: [-2.7437, -1.7837, -2.9007, -3.0421, -2.8065, 0.5445, -3.0159]`
- `otg_max_position: [2.7437, 1.7837, 2.9007, -0.1518, 2.8065, 4.5169, 3.0159]`
- `otg_sync_mode: "time"`

These values are intentionally aligned with the HIROL FR3 `ruckig` smoother
configuration rather than ad-hoc local defaults.

## Verification Status

Completed verification:

- Python syntax check for the edited FR3 files passed
- Docker image build for `lerobot-user` passed
- Docker build-time import smoke passed for `placo`, `panda_py`,
  `pika.gripper`, `ruckig`, and `pyspacemouse`
- FR3 robot unit tests passed in Docker: `9 passed`
- regression coverage includes:
  - OTG target path goes through smoother
  - OTG continues after a single `send_action()`
  - sender loop runs faster than smoother when configured that way

Known compatibility note:

- HIROL pins `ruckig==0.14.0`
- the current Python 3.12 Docker environment in this repository builds
  successfully with `ruckig 0.15.x`
- the current Dockerfiles therefore use `ruckig>=0.15,<0.16`

## Hardware Readiness Verdict

The software side is ready for **controlled real-hardware smoke testing** in the
Docker environment.

This is **not yet evidence that full teleoperation is production-ready**. Before
first real movement, the runtime environment still needs the normal hardware
entry checks:

- confirm FR3 is reachable from inside the container
- confirm Pika gripper is reachable from inside the container
- confirm SpaceMouse is visible from inside the container
- confirm emergency stop and human spotter are in place
- start with reduced workspace and conservative teleop rate

## Practical Go / No-Go

Current recommendation:

- **Go** for first-entry real-hardware smoke testing
- **No-Go** for unattended or full-session data collection before container-side
  device connectivity and first-motion checks pass

Suggested first-entry posture:

- run through Docker only
- use a conservative teleop rate first, for example around `200Hz`
- verify gripper open/close separately before combined teleop
- keep motions small and near the current pose
