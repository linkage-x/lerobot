# Argus Online Sync 多相机同步录制改进总结

日期：2026-07-07

## 结论

当前分支已经把 Thor GMSL2 多相机录制的生产默认路径切到
`argus_online_sync`。核心变化是：

```text
先在 recorder 内部找到各路相机的同一个 SOF 同步帧 cluster，
再把这些同步帧送入硬件 encoder/mux。
```

因此最终保存的每路视频天然同帧数，不再依赖保存后的 ffmpeg
解码、裁剪、重编码来补救同步问题。

最新 8 路、10 轮、每轮 60 秒、60 Hz burn-in 已通过：

```text
dataset_root:
/home/nvidia/lerobot/outputs/datasets/online_sync_burnin_10x60_20260707_035356

episodes_found: 10 / 10
expected_frames: 3600
active_cameras:
cam_03, cam_06, cam_07, cam_08, cam_09, cam_12, cam_13, cam_14
每路 manifest frame count: 3600
每路 sidecar rows: 3600
max SOF delta: 401000 ns = 0.401 ms
ffmpeg materialization: false
driver rc: 0
```

本地报告已保存到：

```text
outputs/datasets/online_sync_burnin_10x60_20260707_035356/
  online_sync_burnin_summary.json
  online_sync_burnin_sync_report.md
```

## 背景问题

旧方案主要有两类：

1. `gstreamer_splitmux`
   - 每路相机独立进程或独立 pipeline 写视频；
   - UI Start/Stop 到达各路进程的时间不同；
   - 不能证明每路视频的第 N 帧来自同一次硬触发；
   - 曾出现多路 episode 帧数差几十帧甚至约 60 帧。

2. `argus_metadata`
   - 能保存每帧 Libargus metadata；
   - 保存后可按 `sof_tsc_ns` 找到同步窗口；
   - 但视频已经先完整写入，需要保存后 materialize；
   - 对 H.265 任意帧窗口切片通常需要重编码，8 路 60 秒保存耗时过长。

因此，根本问题不是 UI Start/Stop 同时性，而是同步判断发生得太晚。

## 当前方案

新增生产 backend：

```yaml
sensors:
  cameras:
    defaults:
      recorder_backend: argus_online_sync
```

当前 `thor_gmsl2_11ch_example.yaml` 已默认使用该 backend。

### 数据路径

当前成功路径是：

```text
Argus BufferOutputStream
  -> 预分配 DmaBuffer pool
  -> IBuffer::getMetadata() 获取 same-buffer metadata
  -> SyncManager 按 SOF TSC 聚类
  -> 只接受 full SOF cluster
  -> NvVideoEncoder 硬件编码
  -> encoded appsrc muxer
  -> cam_XX.mkv
```

每个 logical frame 必须满足：

1. 每个 active camera 都有一帧；
2. 这些帧的 `sof_tsc_ns` 最大差值不超过 `tolerance_ms`；
3. cluster 完整后才进入 encoder；
4. 开始录制后如果中间缺 full cluster，episode 失败，不补帧、不插帧、不复制帧。

这意味着帧数不是选某一路参考相机决定的，而是由所有 active cameras 共同组成的
full cluster 数决定。

## 输出合同

每个 episode 输出：

```text
cam_XX.mkv
cam_XX.argus_frame_metadata.csv
online_sync_manifest.json
meta.json
```

### 视频

视频文件本身应满足：

```text
cam_03.mkv frames == N
cam_06.mkv frames == N
...
```

其中 `N = actual_frames`，固定时长 60 秒、60 Hz 下目标为 `3600`。

### Sidecar

每路 sidecar 一行对应最终视频中的一帧：

```text
camera,logical_frame_index,local_frame_number,
sensor_timestamp_ns,sof_tsc_ns,eof_tsc_ns,internal_frame_count
```

`logical_frame_index` 从 0 连续递增到 `N - 1`。

### Manifest

`online_sync_manifest.json` 是保存 gate 的核心依据，关键字段包括：

```text
ok
fps
target_frames
actual_frames
sync_source
tolerance_ns
active_cameras
frame_count_by_camera
max_abs_delta_ns_by_camera
failure
```

`thor_record.py` 保存前检查 manifest。如果 manifest 缺失、`ok=false`、帧数不一致、
SOF delta 超阈值，episode 会被丢弃。

## UI 使用方式

UI 操作流程不变：

```text
Connect -> Start -> Stop / 自动到时 -> Save / Discard
```

变化只在 recorder 内部：

- UI 仍然启动 `tools/thor/gmsl2/thor_record.py`；
- stdin 协议不变；
- Connect 仍然检测相机、设置 PWM/trigger/camera controls；
- Start 仍然创建 episode 目录；
- Save 阶段从旧的 metadata materialization gate 改为 `online_sync_manifest.json` gate。

## 相比旧方案的改进

### 1. 同步发生在 encoder 前

旧方案是：

```text
先写完整视频 -> 保存后看 metadata -> 再裁剪/重编码
```

当前方案是：

```text
先按 SOF 找 full cluster -> 只把同步帧写入 encoder
```

所以 H.265 的第一帧就是同步 logical frame 0，不需要从非 IDR 帧开始切文件。

### 2. 不再依赖本地 frame id

森云反馈本地 frame id 可能受相机打开顺序影响。当前同步依据是 Libargus
same-buffer metadata 中的 `sof_tsc_ns`，并且只接受跨相机 SOF 接近的 full cluster。

### 3. 不再保存后重编码

最新 burn-in 报告：

```text
ffmpeg_materialization: false
ffprobe_enabled: false
```

保存后的 cleanup 平均约 `0.24s`，已经不是之前数分钟的 materialization。

### 4. 同帧失败时 fail fast

如果录制窗口中间缺任一路相机，当前策略是 fail episode，而不是静默补帧或生成帧数不同的视频。

### 5. 对 transient Argus provider 错误做了 retry

burn-in 中每个 episode 边界第一次创建 Argus provider 都出现一次 transient：

```text
Connection reset by peer / Cannot create camera provider
```

当前 `argus_online_sync_session.py` 会 retry 一次，并清理第一次失败留下的 stale error。
10x60 验证中 retry 全部成功，没有造成 episode 丢弃。

## 验证结果

### 本地检查

已通过：

```bash
python3 -m py_compile \
  tools/thor/gmsl2/argus_online_sync_session.py \
  tools/thor/gmsl2/online_sync_burnin.py \
  tools/thor/gmsl2/thor_record.py \
  tools/thor/gmsl2/gmsl2_record.py

git diff --check
```

### Thor 8 路 60 秒

8 路相机：

```text
cam_03, cam_06, cam_07, cam_08, cam_09, cam_12, cam_13, cam_14
```

单轮 60 秒验证通过：

```text
actual_frames: 3600
每路 frame count: 3600
每路 sidecar rows: 3600
max SOF delta: 微秒到亚毫秒级
```

### Thor 8 路 10x60 burn-in

最新数据：

```text
dataset_root:
/home/nvidia/lerobot/outputs/datasets/online_sync_burnin_10x60_20260707_035356

driver elapsed: 707.93 s
episodes_found: 10
ready_count: 10
saved_count: 10
driver rc: 0
failures: []
```

每个 episode：

```text
actual_frames: 3600
target_frames: 3600
active_cameras: 8 路一致
manifest frame counts: 每路 3600
sidecar rows: 每路 3600
```

每轮最大 SOF delta：

```text
episode_000000: 401000 ns
episode_000001:   9500 ns
episode_000002:   9000 ns
episode_000003:   8500 ns
episode_000004:   8000 ns
episode_000005: 210500 ns
episode_000006:   9500 ns
episode_000007: 378000 ns
episode_000008:   9500 ns
episode_000009:   8000 ns
```

全部低于当前 `tolerance_ms=1.0`。

## 当前耗时拆解

10x60 理论录制时间是 `600s`，实测 driver wall time 是 `707.93s`。

额外约 `107.93s`，主要来自 episode start：

```text
实际录制窗口总和: 600.35s
split/start 总和: 91.36s
stop cleanup 总和: 2.44s
其他开销: 约 13.77s
```

`split_emit_ms` 平均每个 episode 约 `9.14s`。主要原因：

1. 每个 episode 都重新启动 C++ recorder；
2. 重新创建 Argus provider/session/stream；
3. 初始化 8 路 buffer pool、encoder、muxer；
4. 等 `startup_full_clusters=30` 个同步 warmup cluster；
5. 当前每轮第一次 Argus provider 创建会 transient fail，然后 retry 一次。

这不是旧的 ffmpeg 后处理耗时。stop 后 cleanup 只有 `0.21-0.26s`。

## 仍然存在的限制

1. `sof_tsc_ns` 是 Thor/Argus 侧 SOF 时间，不是传感器曝光开始的硬件 timestamp。
   目前它足以证明 SoC 侧接收到的帧在同一个同步 cluster 内，但最强证据仍然是
   供应商提供的 trigger frame id / exposure timestamp / PWM timestamp。

2. 当前实现是每个 episode 新起 recorder。稳定性已通过 10x60，但每轮 start 约
   `7-9.5s`，不是最低延迟方案。

3. Argus provider 在 episode 边界存在 transient reset。当前 retry 能覆盖这次
   burn-in，但仍属于需要继续观察的 Thor/Argus 生命周期风险。

4. C++ recorder 正常完成后使用 `_exit` 避开 NVIDIA Argus/MMAPI 析构链路崩溃。
   文件和 manifest 已经落盘，但这说明底层资源释放路径仍有噪声和技术债。

5. 本轮 10x60 为了避免慢 QC，关闭了 ffprobe。验证依据是 manifest 和 sidecar。
   如需证明视频容器层帧数，可对少量样本单独跑 ffprobe，但不建议放在每轮生产保存路径。

## 后续优化建议

### 短期

- 把 `start_retry_settle_s` 从 `2.0s` A/B 测到 `0.5s` 或 `1.0s`，观察 10x60 是否仍稳定。
- 把 `startup_full_clusters` 从 `30` 测到 `10`，节省约 `0.3s` 级别启动时间。
- 保留 `skip-argus-probe` 的 burn-in 选项，用 online-sync preflight/manifest 作为更贴近生产链路的判断。

### 中期

- 继续跑更多 8 路/10 路/11 路长时间 burn-in。
- 记录每次 transient provider retry 的次数和是否成功，作为 Argus 生命周期健康指标。
- 对少量 episode 抽样跑 ffprobe，确认视频容器帧数和 manifest/sidecar 一致。

### 长期

- 做长驻 recorder daemon：

```text
Connect 时打开 Argus/camera/encoder 资源并保持 warm
Start 时只切换 episode 文件写入
Stop 时在 full cluster 边界关闭当前文件
```

这能把每个 episode 的 start 开销从约 `9s` 降到接近 warmup cluster 时间，但实现复杂度更高。

- 如果森云提供硬件 trigger id / exposure timestamp，应把 SyncManager 的同步 key
  从 `sof_tsc_ns` 升级到硬件同步字段。

## 当前推荐口径

当前版本已经解决了“保存后裁剪才能保证帧数一致”的主要问题：

```text
同步判断前移到 encoder 前；
只编码完整同步 cluster；
最终视频、sidecar、manifest 都以同一个 logical frame count 为合同；
保存阶段不再做 ffmpeg 重编码。
```

本轮 8 路 10x60 结果显示，当前方案可以稳定产出每路 3600 帧、SOF delta 小于
1 ms 的同步 episode。剩余主要问题从“帧不同步/保存后重编码慢”转为“每个
episode 重新启动 Argus recorder 带来的约 9 秒 start 开销，以及 Argus provider
生命周期的 transient reset”。
