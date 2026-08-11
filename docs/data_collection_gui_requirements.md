# Data Collection GUI Requirements

## 背景

`docs/data_collection_gui.md` 的核心判断是：具身数据采集 GUI 不应只是“视频预览页”，而应是 Live Teleop、多模态观测、数据质检、回放/标注、训练集管理一体化的 Web HMI。结合当前仓库实践，近期最直接的落点是 handheld 多模态采集链路：

- 录制入口：`tools/handheld/handheld_record.py`
- 示例配置：`tools/handheld/handheld_record_example.yaml`
- 数据格式：`LeRobotDataset` v3
- 实时/离线可视化：`rerun-sdk`，handheld recorder 已支持 `display_data`、`display_ip`、`display_port`、`rerun_save_path`
- 真机重播参考：`src/lerobot/scripts/lerobot_replay.py`

初版 GUI 的目标不是替代全部 CLI，而是把现有 CLI 能力产品化为一个可操作、可观察、可复用的工作台。

## 用户与场景

主要用户：

- 数据采集员：按任务录制 episode，保存/丢弃/复录，查看设备状态。
- 算法/数据工程师：回看 episode，检查时间同步、掉帧、夹爪/触觉曲线和异常片段。
- 机器人调试人员：将已录制轨迹下发到真机进行 replay，并观察执行状态和偏差。

核心场景：

1. handheld 多模态录制：基于 `tools/handheld/handheld_record_example.yaml` 启动/停止录制。
2. 采后回放和质检：像 Rerun 一样围绕时间轴同步展示图像、触觉、夹爪宽度、设备时间戳和轨迹。
3. 真机轨迹重播：选择 dataset/episode/robot config，按安全策略将 action 逐帧发送给真实机器人。

## 当前 handheld 数据契约

`handheld_record.py` 会根据已连接设备动态构建特征：

- `observation.images.<camera_name>`：RGB 图像或视频，示例配置包括 8 路 Hikrobot、2 路 OpenCV、2 路 RealSense 占位。
- `observation.state`：当前包含 Pika Sense 夹爪宽度，名字形如 `handheld_gripper.<name>.width_mm`。
- `observation.device_capture_timestamp`：每个设备相对 episode 起点的采集时间。
- `observation.soft_sync`：启用 soft sync 时记录 `target_timestamp_s`、`max_skew_s`、`oldest_device_lag_s`、`global_lag_s`、`timed_out`。
- `observation.tactile.<name>.*`：Paxini Gen2 Omega 触觉左右侧 xyz、magnitude、raw_xyz。
- `task`：来自 `dataset.single_task`。

示例配置默认：

- `repo_id`: `local/handheld_multimodal_v1`
- `root`: `/workspace/outputs/datasets/handheld_multimodal_v1`
- `fps`: 30
- `episode_time_s`: 10
- `num_episodes`: 0，表示无限录制，手动停止
- `video`: true
- `streaming_encoding`: true
- `vcodec`: h264

GUI 必须显示这些字段，并避免隐藏动态 feature 变化。

## 初版功能范围

### 1. Handheld 录制控制台

必须支持：

- 加载并展示 `tools/handheld/handheld_record_example.yaml` 的关键配置。
- 展示 camera、tactile、handheld gripper 三类设备清单、连接状态、FPS、延迟和最近错误。
- 启动录制、停止并保存、停止并丢弃、结束 session。
- 展示当前 episode 的录制进度、帧数、队列状态、保存 episode 总数。
- 支持配置 `display_data`、`rerun_save_path`、`soft_sync.enabled` 的开关或只读状态。
- 显示快捷键语义：`s` 保存早停、`n` 丢弃早停、`Esc` 退出 session。

后端命令建议：

```text
POST /api/handheld/record/start
POST /api/handheld/record/stop-save
POST /api/handheld/record/stop-discard
POST /api/handheld/record/exit
GET  /api/handheld/status
WS   /api/handheld/events
```

### 2. Rerun 风格轨迹回放

必须支持：

- episode 选择、播放/暂停、逐帧、倍速、时间轴 scrubber。
- 多轨道时间线：camera frame、gripper width、tactile magnitude、device timestamp skew、event marker。
- 2D/3D 轨迹视图的初版：先展示 action/state 派生轨迹、关键帧点、当前帧游标。
- 异常提示：timestamp gap、soft sync timeout、camera stale frame、action jump、frame drop。
- 支持打开外部 Rerun：连接已运行 Rerun viewer 或加载 `.rrd` 文件的入口。

前端初版可先用 Canvas/SVG mock 轨迹，后续接 dataset reader 或 Rerun Web Viewer。

### 3. 真机轨迹重播

必须支持：

- 选择 dataset `repo_id/root`、episode、robot config、FPS。
- 预检查：机器人连接、急停/使能、action feature 与 robot action processor contract、episode 长度、workspace/速度限制。
- Replay 控制：arm/enable、dry-run、start、pause、resume、abort。
  - **dry-run 仍未实现（2026-07-31）**：`/api/replay/start` 曾存在，但只改 `state.replay.state`
    并打一条 message，不启动任何进程；两个 replay runtime 也都没有 dry-run 开关。按钮和端点已经
    移除，宁可让这条需求显式未满足，也不要留一个按下去像在执行、实际什么都没做的控件。
- 展示 replay 进度、发送帧率、最近 action、机器人 observation、tracking error。
- 强制安全约束：网页只发高层命令，低层限速、限位、急停、断连保护必须在 Python runtime/robot backend 内实现。

后端命令建议：

```text
POST /api/replay/preflight
POST /api/replay/arm
POST /api/replay/start
POST /api/replay/pause
POST /api/replay/resume
POST /api/replay/abort
GET  /api/replay/status
WS   /api/replay/events
```

## 非功能需求

- Live 与 Replay 使用同一套时间轴模型，减少操作心智负担。
- 录制状态必须以服务端状态为准，前端刷新后能恢复。
- 操作按钮必须有 disabled/loading/error 状态，禁止重复 start/replay。
- 真机 replay 操作必须二次确认，并默认 dry-run。
- 所有 episode 操作必须写审计日志：operator、dataset、episode、command、时间、结果。
- 前端应可在实验室局域网访问；需要硬件权限的能力由本机 Python gateway 承担。

## 前端信息架构

初版单页应用分为四个区域：

- 顶栏：系统状态、gateway 状态、Rerun 状态、当前 dataset。
- 左栏：handheld 录制配置、设备状态、录制控制。
- 中央：轨迹/时间线回放，支持 live/replay 共用。
- 右栏：真机 replay 控制、安全预检、事件日志。

## 后端集成建议

短期保持 CLI 可用，在其外层增加 gateway：

```text
Browser React GUI
  -> FastAPI/WebSocket Gateway
      -> handheld_record runtime wrapper
      -> dataset metadata/episode reader
      -> rerun .rrd export/open helpers
      -> lerobot_replay-compatible real robot replay runtime
```

中期应把 `handheld_record.py` 中的录制循环拆成可被 GUI 驱动的 runtime class，避免通过伪终端模拟 `s/n/Esc`。

## 里程碑

M0：前端骨架

- 静态配置概览
- mock recording/replay 状态机
- Rerun 风格轨迹 Canvas
- API client contract

M1：本机 gateway

- 读取 YAML 配置
- 启动/停止 handheld recording
- WebSocket 推送录制进度、设备状态、错误
- episode 列表和 metadata 读取

M2：真实 dataset 回放

- 读取 LeRobotDataset episode
- 图像/状态/触觉时间线同步
- 异常检测和 marker
- `.rrd` 导出/打开

M3：真机 replay

- replay preflight
- dry-run replay
- 真机 replay start/abort
- tracking error 和安全事件可视化

