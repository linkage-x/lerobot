# Traj-Gen（EE 轨迹生成）on Thor gmsl2 数据

> 2026-06-01 初版：当时合入的是 **hikon 专属** 实现，对 Thor gmsl2 数据不适用（见文末「历史背景」）。
> 2026-06-25 重写：traj-gen 已切到 **gmsl2 AprilTag cube 追踪**，在 Thor 上真机验证可用（生成 + 显示 + PWM 时间戳）。

## 一句话结论

GUI 的 **Generate EE Trajectory / Queue Traj Gen** 按钮（`POST /api/processing/traj-gen`）现在对 Thor gmsl2 数据集**可正常生成 EE pose 并在 Replay 视图显示**。追踪直接吃 `episodes/episode_*/cam_*.mkv` 原始流，**不需要 LeRobot v3 导出**。

## 数据流

```
按钮 / API
  → gateway._queue_traj_gen → _ee_trajectory_command
  → ["bash", run_april_cube_tracking_local.sh, --dataset-root <ds>, --config <thor yaml>]
  → (gateway 跑在 Thor 上，本地直接执行，无 SSH / 无 copy-back)
  → april_cube_tracking_in_robot_base.py 追踪 episodes/*.mkv
  → 写 sidecar: <ds>/derived/april_cube_tracking_in_robot_base/state_action.{left,right,head}.csv
  → gateway timeline 读 sidecar → 前端 Pose3DViewer 渲染 EE 轨迹
```

关键文件与常量（`gateway.py`）：

- `DEFAULT_EE_TRAJECTORY_RUNNER = third_party/opencv_kalibr/run_april_cube_tracking_local.sh`
- `DEFAULT_EE_TRAJECTORY_CONFIG = .../config_thor/april_cube_tracking_in_robot_base_thor.yaml`
- `DEFAULT_TRAJ_SIDECAR_NAME = "april_cube_tracking_in_robot_base"`
- `DEFAULT_TRACKING_RUN_SUFFIX = "_thor_april_tracking_in_robot_base"`（`outputs/tracking_analysis/<ds>...` 分析产物 + 视频 overlay 用）

`run_april_cube_tracking_local.sh` 封装了 python 解释器选择（追踪依赖 cv2/pupil_apriltags 只在 `third_party/opencv_kalibr/.venv`，**不在跑 gateway 的系统 python3**）+ `PYTHONPATH` + 依赖自检 + 数据布局检查。它被两处共用：

1. gateway（Thor 上本地直接调）。
2. `run_april_cube_tracking_on_thor.sh`（开发机用，SSH 到 Thor 跑同一 runner，再 copy-back，附 replay 提示）。

## 前提（Thor 上需就位）

- 标定产物：producer 内参目录 `outputs/calibration/thor_gmsl2_intrinisics_dict_0720`
  和 joint 外参目录 `outputs/calibration/thor_gmsl2_extrinisics_robot_base_0720`
  （thor yaml 的 `calibration.*_run_name` 指向它们；fixed 模式不需要 auxiliary marker）。
  `run/deploy.sh` 会 best-effort 单独同步这两个目录，因为普通 repo sync 会排除整个 `outputs/`；缺失时只告警并继续部署，EE trajectory 功能等标定产物补齐后再可用。需要硬校验时使用 `REQUIRE_EE_CALIBRATION=1 bash run/deploy.sh`。
- cube 物理贴标与 thor yaml 的 `cube_tracker.cubes[*].marker_ids` 一致（left/right/head）。

## 显示路径：v3 vs gmsl2（两条都已支持）

`gateway._read_dataset_timeline` 按数据集形态分流：

| 数据集形态 | 路由 | EE pose 来源 |
| --- | --- | --- |
| 有 `data/chunk-*/*.parquet`（BOX v3 sidecar，如 dadada） | v3 路径 | parquet 行（按 `frame_index` 匹配 sidecar cube poses）|
| 只有 `episodes/`、无 v3 parquet（如 `--no-box` 相机-only 采集） | `_read_gmsl2_timeline` | **直接读 sidecar**（2026-06-25 补；此前该路径不读 sidecar，EE pose 不显示）|

两条路径都把 sidecar 的 `state_*`（即 EE pose）按 **per-episode 局部 `frame_index`** 挂到帧上，填 `cubePoses` / `cubePoseNames`，前端据此渲染。

## 时间戳：PWM 硬同步相机轴

每帧时间戳 = `pts_offset + N/fps`（t0 相对域，见 `tools/thor/ts_sync.md` §5.1/§6），即 PWM 60Hz 帧网格 + 首帧落盘 wall-time 偏移修正：

- v3 路径：直接用 parquet 的 `timestamp` 列（录制器写入时已是该值）。
- gmsl2 路径：`_gmsl2_pts_offset_s()` 从 `meta.json.sync_reference`（`t0_wall_s` + `camera_first_wall_s`）算出同一个 `pts_offset`，再 `+ N/fps`。

EE pose 由多路硬同步相机估算，因此落在相机/PWM 时间轴上；**不使用 BOX MCU 钟**（那是单独的 `box.timestamps` 列，仅作对齐元数据）。

> 注意（非当前 bug）：parquet 帧网格长度 = `round(duration_s×fps)`，可能与真实相机帧数差几帧（box-only 幽灵行）；只有真发生丢帧时均匀网格才会与真实 PWM 边沿漂移。本数据实测 PTS 抖动相消、无真实丢帧，对齐良好。

## 性能

瓶颈是全图 AprilTag 检测（N 相机 × 3 cube × 全分辨率）。已做 **每相机检测一次、3 cube 共享**（`CubeTracker.detect_markers_raw` + `_blocks_from_raw`，数值与逐 cube 检测**完全一致**），dadada（7 相机 / 1251 帧）实测 ~16.5 分钟 → ~7.2 分钟（约 2.3×）。进一步可调 `cube_tracker.apriltag_detector.quad_decimate`（1.0→2.0）或 `--frame-step`。

## 真机验证（2026-06-25，nvidia@192.168.111.122）

- `dadada_20260616_084743`（有 v3 parquet，走 v3 路径）：按钮 → `pose_ready`，3 episode 的 timeline 返回 left/right/head EE pose（ep0 603/603、ep1 394/394、ep2 254/291），ts 吻合 `pts_offset + N/60`。
- `thor_gmsl2_apriltag_raw_April_cam8_0618_thor_*`（无 v3 parquet，走 gmsl2 路径）：runner 生成 sidecar 后 timeline 正确返回 EE pose（修复前该路径返回 0 个 pose）。

## 历史背景（已不适用）

初版合入的是 `hikon_cube_tracking_in_robot_base.py` + `config_hikon/...umi.yaml`，对 Thor gmsl2 数据有 3 处硬性前置依赖会先 `raise`（hikon 专属标定产物缺失、`videos/observation.images.*` 布局不匹配、`data/chunk-*` parquet 布局不匹配），因此当时「在 Thor 数据上跑会 fail」。现已改用 gmsl2 专属的 april thor 实现 + 直接吃 `episodes/`，上述限制不再存在。
