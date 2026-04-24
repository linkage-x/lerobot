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

After the smoke test works, run the interactive MuJoCo teleop path with Quest3:

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_mujoco_teleop.py \
  --teleop-type quest3 \
  --quest3-gripper-mapping pinch_value \
  --quest3-closed-pinch-value 0.004 \
  --quest3-open-pinch-value 0.111
```

For recording, use the normal draccus teleop config override:

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_mujoco_record.py \
  --config_path tools/fr3/fr3_sim_record_config.yaml \
  --teleop.type=quest3 \
  --teleop.gripper_mapping=pinch_value \
  --teleop.closed_pinch_value=0.004 \
  --teleop.open_pinch_value=0.111
```

Quest3 motion uses `squeeze` as the default clutch. The gripper command can update even when the clutch is not active.
