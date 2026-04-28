# FR3 Quest3 Hardware Setup

## Default Certificate

The Quest3/Vuer HTTPS certificate is copied into this repo:

- `tools/fr3/quest3_certifications/cert.pem`
- `tools/fr3/quest3_certifications/key.pem`

`Quest3TeleopConfig` and `tools/fr3/fr3_quest3_connection_smoke.py` use these files by default. Override them only when testing another certificate:

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_quest3_connection_smoke.py \
  --cert-file /path/to/cert.pem \
  --key-file /path/to/key.pem
```

Quest3 calibration files are stored under the repo by default to avoid permission issues from root-owned HuggingFace cache directories:

```text
outputs/fr3_quest3_calibration/teleoperators/quest3
```

## PC Preparation

1. Install/sync the Python environment:

```bash
uv sync --extra fr3_teleop
```

2. Install ADB on the host if it is not available:

```bash
sudo apt-get update
sudo apt-get install -y android-tools-adb
adb version
```

3. Ensure the host and Quest3 can reach the same port. The default Vuer port is `8012`.

## Quest3 Preparation

1. Enable developer mode for the Quest3 device.
2. Enable USB debugging on the headset.
3. Connect Quest3 to the PC with USB.
4. Run:

```bash
adb devices
```

If the device is shown as `unauthorized`, put on the headset and accept the USB debugging prompt, then rerun `adb devices`.

## USB Reverse Mode

Use this mode first because it avoids Wi-Fi routing issues.

1. Forward the Quest3 device port to the host process:

```bash
adb reverse tcp:8012 tcp:8012
adb reverse --list
```

Expected mapping:

```text
tcp:8012 tcp:8012
```

2. Start the smoke test:

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_quest3_connection_smoke.py
```

3. Open this URL in the Quest3 browser:

```text
https://127.0.0.1:8012?ws=wss://127.0.0.1:8012
```

4. Accept the browser warning for the self-signed certificate if prompted.

## Wi-Fi/LAN Mode

Use this only after USB reverse mode works.

1. Put the PC and Quest3 on the same network.
2. Start the smoke test:

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_quest3_connection_smoke.py
```

3. Use the `Wi-Fi/LAN URL` printed by the script, for example:

```text
https://<pc_ip>:8012?ws=wss://<pc_ip>:8012
```

4. If the page cannot connect, check firewall and routing:

```bash
ip addr
ss -ltnp | grep 8012
```

## Optional Wireless ADB

After USB authorization succeeds:

```bash
adb tcpip 5555
adb shell ip addr show wlan0
adb connect <quest3_ip>:5555
adb devices -l
adb -s <quest3_ip>:5555 reverse tcp:8012 tcp:8012
adb -s <quest3_ip>:5555 reverse --list
```

## Smoke Test Success Criteria

The terminal should continuously print:

- `valid=True` when the selected hand is visible.
- Changing `pinch_value` when pinching.
- Changing `squeeze_value` when squeezing the controller.
- A stable `gripper` value in `[0, 1]`.
- Nonzero `wrist_xyz` when the hand is tracked.

If tracking is lost, `valid=False` is expected and robot motion should remain disabled.

## MuJoCo Teleop

After the smoke test works, run the direct Quest3 Pika gripper scene. This scene removes the FR3 arm body, keeps the table/object/fixture and Pika gripper, and maps the right-hand Quest3 wrist pose into the original MuJoCo workspace:

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_mujoco_teleop.py \
  --teleop-type quest3 \
  --quest3-scene-mode pika_gripper \
  --quest3-gripper-mapping pinch_value \
  --quest3-closed-pinch-value 0.004 \
  --quest3-open-pinch-value 0.111
```

For recording, use the Quest3 Pika config:

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_mujoco_record.py \
  --config_path tools/fr3/fr3_quest3_pika_gripper_record_config.yaml
```

`--quest3-scene-mode pika_gripper` is the default when `teleop.type=quest3`. Use `--quest3-scene-mode fr3_arm` only when you want Quest3 to drive the original full FR3 arm scene.

For current Quest3 Pika-only collision and gripper tuning findings, see
[`fr3_quest3_pika_mujoco_collision_findings_20260428.md`](./fr3_quest3_pika_mujoco_collision_findings_20260428.md).

Default VR-to-scene alignment:

- The first valid Quest3 wrist pose is mapped to the initial Pika TCP pose.
- After that, `sim_xyz = initial_tcp_xyz + (quest3_wrist_xyz - first_wrist_xyz) * (1, 1, 1) + offset`.
- The result is clipped to `x=[0.25,0.82]`, `y=[-0.36,0.36]`, `z=[0.43,0.90]`.
- Change the offset/scale with `--quest3-position-offset OX OY OZ` and `--quest3-position-scale SX SY SZ`.
- Orientation is locked by default because Quest3 wrist frame is not the same as Pika TCP frame. Use `--quest3-follow-orientation` only after calibrating `--quest3-rotation-alignment-xyzw`.
- Use `--quest3-absolute-origin` only when the Quest3 world origin has been explicitly calibrated to the MuJoCo scene.

Pose mapping diagnostics:

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_mujoco_teleop.py \
  --teleop-type quest3 \
  --quest3-debug-pose
```

Move your right hand in one axis at a time and watch `wrist`, `origin`, `mapped_tcp`, and `ee` in the terminal. If an axis direction is wrong, adjust `--quest3-position-scale`, for example `--quest3-position-scale 1 -1 1` to flip the lateral axis.
