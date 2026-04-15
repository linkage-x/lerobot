# FR3 Cube 数据采集与回放流程（Host + UV, 2026-04-22）

本文档对应 `third_party/opencv_kalibr/fr3_data_collection_replay` 下的脚本，目标是给出一套可复现的录制和回放流程。

详细参数说明见：

- `third_party/opencv_kalibr/fr3_data_collection_replay/README.md`

## 1) 录制 Lerobot 数据集

使用 `record_robot_cube.py` 在引导模式下录制 Hikrobot 视频和机器人状态（支持只录视觉）。

```bash
cd /home/corenetic/Code/lerobot

fr3_uv() {
  uv run --python .venv/bin/python python "$@"
}

fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/record_robot_cube.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/record_robot_cube.host.yaml \
  --runtime.record_duration_s=10
```

不限时录制（按 `e` 结束当前 episode）：

```bash
fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/record_robot_cube.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/record_robot_cube.host.yaml \
  --runtime.record_duration_s=0
```

录制热键：

- `s`: 开始录制一个 episode
- `e`（默认，可配置）: 提前结束当前 episode 并保存；可再次按 `s` 继续录制下一个 episode
- `q`: 退出

可选仅视觉录制：

```bash
fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/record_robot_cube.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/record_robot_cube.host.yaml \
  --dataset.vision_only=true
```

说明：`vision_only=true` 时脚本不会连接或启动机器人，只连接相机并录制视觉数据。

默认输出：

- `/home/corenetic/Code/lerobot/outputs/datasets/fr3_robot_cube`

## 2) 生成追踪轨迹（供回放）

回放脚本默认直接读取 Lerobot 数据集里的 `observation.state`（EE pose）。  
该字段由 `hikon_cube_tracking_in_robot_base.py` 写回生成。  
如未生成，请先运行：

```bash
cd /home/corenetic/Code/lerobot

.venv/bin/python3 third_party/opencv_kalibr/scripts/hikon_cube_tracking_in_robot_base.py \
  --config third_party/opencv_kalibr/scripts/config_hikon/hikon_cube_tracking_in_robot_base.yaml
```

常用覆盖：

```bash
# 仅快速跑前 300 帧
.venv/bin/python3 third_party/opencv_kalibr/scripts/hikon_cube_tracking_in_robot_base.py \
  --config third_party/opencv_kalibr/scripts/config_hikon/hikon_cube_tracking_in_robot_base.yaml \
  --max-frames 300

# 只做tracking输出，不写回parquet（临时验证）
.venv/bin/python3 third_party/opencv_kalibr/scripts/hikon_cube_tracking_in_robot_base.py \
  --config third_party/opencv_kalibr/scripts/config_hikon/hikon_cube_tracking_in_robot_base.yaml \
  --max-frames 300 \
  --save_to_dataset.write_parquet_inplace=false
```

默认输出目录：

- `/home/corenetic/Code/lerobot/outputs/tracking_analysis/<dataset_name>_tracking_in_robot_base`

说明（当前逻辑）：

- `pika offset` 已迁移到 tracking 阶段（`ee_from_cube.mode: pika_tcp_heuristic`）。
- run_name 默认按 `input.dataset_root` 自动命名（`<dataset_name>_tracking_in_robot_base`）。
- `analysis_lowpass` 会额外输出一版 low-pass 结果用于误差分析：
  - `fused_ee_pose_in_robot_base_records_lowpass.csv`
  - `ee_position_estimated_lowpass_vs_actual_xyz.png`
  - `ee_orientation_estimated_lowpass_vs_actual_rpy.png`
  - `ee_estimated_lowpass_vs_actual_error.png`
- 会在 dataset 内写回 sidecar 与 parquet：
  - `derived/hikon_cube_tracking_in_robot_base/state.npy`
  - `derived/hikon_cube_tracking_in_robot_base/action.npy`
  - `derived/hikon_cube_tracking_in_robot_base/state_action.csv`
  - `observation.state` / `action` parquet inplace 更新（默认开启，需 pyarrow）
- 注意：`save_to_dataset` 仍保持 raw fused（未 low-pass）信号，low-pass 仅用于 tracking analysis 输出与对比。
- 首次 `write_parquet_inplace=true` 写回时会自动保留一列 `observation.state_raw`；后续运行会优先读取该列作为 actual EE ground-truth，避免重复运行覆盖后丢失原始信号。

回放前请确认 `replay*.yaml` 里的 `input.dataset_root` 指向对应 Lerobot 数据集目录（如 `outputs/datasets/fr3_robot_cube`）。
如需 CSV 兼容模式，再设置 `input.source=csv` 和 `input.csv_path`。

## 3) 回放轨迹

`replay_cube_pose_in_robot_base.py` 按 episode 交互回放，不会一次性重播全量数据。

```bash
cd /home/corenetic/Code/lerobot

fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.host.yaml
```

交互热键：

- `s`: 回放当前 episode
- `n`: 回放下一个 episode
- `b`: 回放上一个 episode
- `i`: 输入 episode index（或回放顺序 index）并回放
- `p`: 打印当前位姿
- `h`: 回 home 并移动到当前 episode 起点
- `q`: 退出

每次触发回放时，机器人都会先运动到该 episode 的起点 EE pose，然后再按轨迹回放。
可选 `base_relocalization.mode=moved`：回放前先基于 Hikrobot + 辅助 marker 重定位 robot base，估计 base 偏移后再回放；若重定位失败可按配置取消回放。
另外，`frame_index` 默认是 episode 内局部索引；回放界面现会同时显示 `local` 与 `global` 范围，便于定位跨 episode 的连续帧区间。

常用覆盖：

```bash
# 仅回放前 250 帧
fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.host.yaml \
  --input.max_rows=250

# 指定数据集（lerobot_dataset 模式）
fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.host.yaml \
  --input.source=lerobot_dataset \
  --input.dataset_root=/home/corenetic/Code/lerobot/outputs/datasets/fr3_robot_cube_light_off

# 调低旋转步进，减小旋转抖动
fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.host.yaml \
  --replay.max_rotation_step_deg=0.4 \
  --replay.replay_fps=20

# 机器人底座可能被移动：先做marker重定位再回放
fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.host.yaml \
  --base_relocalization.mode=moved

# 强制等长 episode（默认1000）；提前结束的 episode 自动静止补齐
fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base.host.yaml \
  --episode_length.mode=fixed \
  --episode_length.fixed_length=1000

```

`base_relocalization.mode`：
- `fixed`（默认）：不做重定位，直接按原 base 轨迹回放。
- `moved`：回放前检测辅助 marker 推断当前 base 相对旧 base 的刚体变换。
  - 当相机间估计残差过高或看不到足够 marker 时，判定失败。
  - `abort_replay_on_failure=true` 时直接取消回放并输出失败原因。

## 4) 无机器人 Matplotlib 回放（动画）

使用 `replay_cube_pose_in_robot_base_matplotlib.py` 仅做可视化，不连接机器人。  
动画内容包括：TCP 姿态 `xyz` 轴、实体轨迹、随时间衰减并消失的拖尾。  
现已支持从 Lerobot dataset 读取并按 episode 交互选择回放（`s/n/b/i/l/q`）。

```bash
cd /home/corenetic/Code/lerobot

fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base_matplotlib.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base_matplotlib.host.yaml
```

常用覆盖：

```bash
# 仅可视化前 500 帧并抽帧
fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base_matplotlib.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base_matplotlib.host.yaml \
  --input.max_rows=500 \
  --visualization.frame_step=2

# Matplotlib 也可用固定长度 episode（静止补齐）
fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base_matplotlib.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base_matplotlib.host.yaml \
  --episode_length.mode=fixed \
  --episode_length.fixed_length=1000

# 不弹窗，导出 GIF
fr3_uv third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base_matplotlib.py \
  --config_path third_party/opencv_kalibr/fr3_data_collection_replay/replay_cube_pose_in_robot_base_matplotlib.host.yaml \
  --visualization.show_window=false \
  --visualization.save_gif_path=/home/corenetic/Code/lerobot/outputs/tracking_analysis/fr3_robot_cube_tracking_in_robot_base/replay_no_robot.gif

```

## 5) 关于 offset 的生效点

- `record_robot_cube.py` 不做 cube->EE offset 计算，仅录制原始机器人状态和视频。
- `pika offset` 在 tracking 脚本 `hikon_cube_tracking_in_robot_base.py` 里应用后写入 CSV/sidecar/parquet。
- 回放脚本默认读取 `ee_est_base_*`，不再做二次 pika offset。

## 6) 快速排障

- 录制报 `No Hikrobot devices found`：检查 `camera_transport_layer` 与现场一致、MVS 环境变量、相机占用状态。
- 回放旋转偏差大：检查 tracking `summary.json` 的 `alignment` 字段，再调 `replay.max_rotation_step_deg` 与 `controller.*`。
