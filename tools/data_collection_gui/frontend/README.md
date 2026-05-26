# LeRobot Data Collection GUI

Browser UI for local data collection, recorded dataset review, episode annotation, MuJoCo validation, and guarded FR3 replay.

The default rig is **Thor + 11 x GMSL2 (SG2-AR0234C-G2F) + BOX 采集板** — see
`tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml` and
`tools/thor/box_sdk/README.md`. It replaces the previous Hikrobot 8-camera +
Pika Sense gripper / RealSense default.

> **首次在 Thor / Jetson 上部署？** 先按 [`tools/thor/DEPLOYMENT.md`](../../thor/DEPLOYMENT.md)
> 跑完 apt / pyarrow / box_sdk wheel / 兼容 symlink / nvm + node / npm
> 国内源那几步 —— 列出了所有曾经踩过的坑，新机零踩坑。下面的 Start The
> GUI 假设这些前置依赖已经就位。

## Start The GUI

Run the local Python gateway first:

```bash
cd /home/hanyu/Codes/lerobot
PYTHONPATH=src:. python -m tools.data_collection_gui.gateway \
  --config-path tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml \
  --datasets-root outputs/datasets \
  --port 8765
```

The `--config-path` argument is optional; the gateway defaults to that
GMSL2/BOX YAML when no override is given. To run the legacy Hikrobot +
Pika capture pass `--config-path tools/handheld/handheld_record_example.yaml`.

Before first use of the BOX 采集板 on the Thor host:

```bash
sudo apt update && sudo apt install -y libeigen3-dev liburdfdom-dev
source tools/thor/box_sdk/setup_env.sh
python3 -m pip install --force-reinstall \
  tools/thor/box_sdk/python/box_collection_sdk-0.1.0-py3-none-any.whl
```

`setup_env.sh` exports `LD_LIBRARY_PATH` so the wheel's
`libbox_controller.so` can find the runtime dependencies vendored under
`tools/thor/box_sdk/lib/`, and `BOX_SDK_URDF` so the controller knows where
to find `share/monte_gripper.urdf`.

Then start Vite:

```bash
cd tools/data_collection_gui/frontend
npm install
npm run dev
```

注意使用以下前端页面:
```
➜  Local:   http://localhost:5173/
➜  Network: http://192.168.147.179:5173/
➜  Network: http://192.168.123.99:5173/
➜  Network: http://172.18.0.1:5173/
  ```

Vite proxies `/api/*` to `http://127.0.0.1:8765` by default. If the gateway is on another port:

```bash
GUI_API_TARGET=http://127.0.0.1:8766 npm run dev
```

If the proxy cannot reach the gateway, the frontend falls back to a mock adapter. Safety-critical replay commands fail closed in that mode.

## Main Pages

| Page | Route | Current Status | Main Entry Points |
| --- | --- | --- | --- |
| Live Record | `#/live-record` | Integrated with gateway | Connect, StartEpisode, Save, Discard, Exit, device status, event log |
| Dataset Processing | `#/dataset-processing` | Partial | datasets root selector, dataset scan, Run QC, processing log; Generate EE Trajectory currently shows `待实现` |
| Episode Replay | `#/episode-replay` | Integrated for review and gated replay | recorded dataset selector, episode selector, video/timeline inspector, annotation, Replay Controls |
| Dataset Export | `#/dataset-export` | UI scaffold only | export planning surface for LeRobot v3 / MCAP / Parquet; backend export endpoints are not implemented yet |
| Deferred pages | dashboard, QC report, model evaluation, device manager, task library, annotation audit | UI placeholders | not part of the current production path |

## Episode Replay Flow

1. Select a recorded dataset under `outputs/datasets` or update the datasets root in Dataset Processing.
2. Select an episode with Previous / Next / dropdown. The default episode is `0`.
3. Review the inspector timeline, camera video, EE pose/action values, and diagnostics.
4. Fill Episode Annotation: task prompt, outcome, quality, include-in-training flag, tags, notes, annotator.
5. Run MuJoCo replay from Replay Controls. This is strongly recommended before Preflight and Dry Run, and required before Real Robot.
6. Run Preflight, then Dry Run or Real Robot as appropriate. Real Robot stays disabled unless the selected dataset/episode/fps has a current passed MuJoCo validation.

MuJoCo validation is persisted under:

```text
<dataset>/meta/gui_replay_validations.json
```

Episode annotations are persisted under:

```text
<dataset>/meta/gui_annotations.json
```

Dataset processing/QC metadata is persisted under:

```text
<dataset>/meta/processing.json
```

## Dataset Processing Status

`Run QC` is implemented in the gateway. It checks parquet readability, required columns, frame continuity, timestamp monotonicity, EE continuity when named EE pose columns exist, quaternion norm, gripper range, video presence, and frame-count consistency.

`Generate EE Trajectory` is not implemented in the GUI/backend yet. Clicking it returns `501 Not Implemented` from the gateway and the frontend shows:

```text
待实现：Generate EE Trajectory 功能尚未接入。
```

Datasets that already contain named EE pose columns in `observation.state` or `action` are treated as `pose_ready` with trajectory version `v1`, so they can proceed to QC and replay review without using the missing generator button.

## Command-Line Entry Points

Use these while equivalent GUI integration is partial or unavailable.

### FR3 Preflight And Recording

```bash
cd /home/hanyu/Codes/lerobot
uv run --python .venv/bin/python python tools/fr3/fr3_record_preflight.py \
  --config-path tools/fr3/fr3_record_hikrobot_example.yaml

uv run --python .venv/bin/python python tools/fr3/fr3_record.py \
  --config-path tools/fr3/fr3_record_hikrobot_example.yaml
```

### Dataset Visualization / EE Trajectory Inspection

```bash
cd /home/hanyu/Codes/lerobot
PYTHONPATH=src:. python src/lerobot/scripts/lerobot_dataset_viz.py \
  --repo-id local/fr3_dataset \
  --root outputs/datasets/lerobotv3_0310_100ep \
  --episode-index 0 \
  --mode local
```

### MuJoCo Replay For Recorded FR3 Episodes

```bash
cd /home/hanyu/Codes/lerobot
PYTHONPATH=src:. python tools/fr3/fr3_sim_record_replay.py \
  --dataset outputs/datasets/lerobotv3_0310_100ep \
  --episode 0
```

For headless metric output:

```bash
PYTHONPATH=src:. python tools/fr3/fr3_sim_record_replay.py \
  --dataset outputs/datasets/lerobotv3_0310_100ep \
  --episode 0 \
  --no-viewer
```

### DAS MuJoCo Replay (Deprecated)

```bash
cd /home/hanyu/Codes/lerobot
PYTHONPATH=src:. python tools/fr3/fr3_das_replay.py \
  --dataset outputs/datasets/lerobotv3_0310_100ep \
  --episode 0
```

### Real Robot Replay Launcher

Run with `--dry-run` first to inspect the Docker command:

```bash
cd /home/hanyu/Codes/lerobot
PYTHONPATH=src:. python tools/fr3/fr3_das_replay_real.py \
  --dataset outputs/datasets/lerobotv3_0310_100ep \
  --episode 0 \
  --robot-ip 192.168.1.208 \
  --gripper-port /dev/ttyUSB0 \
  --dry-run
```

Remove `--dry-run` only after hardware preflight and MuJoCo validation are acceptable for the selected episode.

### Experimental Joint-Target CSV Validation

This is not integrated into the GUI. The validator exists, but this checkout does not include the referenced `fr3_generate_branch_consistent_targets.py` generator, so use it only in workspaces where that generator/CSV exists:

```bash
cd /home/hanyu/Codes/lerobot
PYTHONPATH=src:tools/fr3:. python tools/fr3/fr3_sim_replay_validate_joint_targets.py \
  --dataset outputs/datasets/lerobotv3_0310_100ep \
  --episode 0 \
  --joint-targets-csv outputs/analysis/fr3_branch_consistent_targets_ep000.csv
```

## Notes

- The GUI assumes dataset paths are local paths known to the gateway.
- Relative dataset paths are resolved from the repository root.
- Host DISPLAY/X11 is used for MuJoCo viewer paths inside the Docker launcher.
- Real Robot replay is intentionally stricter than Preflight/Dry Run: MuJoCo is recommended for the latter two, but required for Real Robot.
