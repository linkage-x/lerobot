# 森云 GMSL2 在线同步录制方案

日期：2026-07-06

## 目标

目标不是继续优化保存后的裁剪速度，而是从录制链路上消除后处理裁剪：

```text
先确定同步帧，再把同步帧送入 encoder/mux。
```

这样最终每路 `cam_XX.mkv` 天然具有相同 logical frame count，不需要在
episode 保存后再做 H.265 decode / frame select / re-encode。

## 当前问题

当前 `argus_metadata` recorder 的数据路径是两条并行分支：

```text
video stream    -> nveglstreamsrc -> nvv4l2h265enc/h264enc -> mux -> cam_XX.mkv
metadata stream -> FrameConsumer  -> cam_XX.argus_frame_metadata.csv
```

这个路径可以拿到每帧 `sof_tsc_ns`，也可以在保存后证明哪些帧是同步帧；
但 video branch 在 Start 后已经开始自动写入 encoder/mux。同步判断发生在
视频文件已经写完之后。

因此当前为了让最终视频文件本身帧数一致，需要 materialize：

```text
raw cam_XX.mkv
  -> ffmpeg decode
  -> select synchronized frame window
  -> libx265/libx264 re-encode
  -> aligned cam_XX.mkv
```

实测 8 路 1080p@60Hz、60s episode，即使关闭 ffprobe QC、并行 2 路
materialize，保存等待仍约 563s。慢点不是 QC，而是 H.265 软件重编码。

根因是 H.265 是 GOP/IDR 帧间编码。同步窗口经常从非 IDR 帧开始；如果要生成
从任意同步帧开始且可独立解码的新视频，通常必须重编码。即使以后拿到更强的
硬件 frame id / PWM timestamp，它也只能更准确地告诉我们“哪些帧同步”，不能
自动解决“任意帧无重编码切 H.265 文件”的问题。

## 方案原则

1. 同步发生在 encoder 前，而不是 encoder 后。
2. Start/Stop 不再是 host wall time 边界，而是 logical frame 边界。
3. episode 的帧数由 `target_frames = round(episode_time_s * fps)` 决定。
4. 只写完整同步 cluster；缺任一路的 cluster 不进入视频。
5. 第一帧进入 encoder 的就是 logical frame 0，因此 H.265 GOP 从同步帧开始。
6. 保存后只写 meta/manifest，不做视频重编码。

## 目标数据路径

推荐重构为 online synchronized recorder：

```text
Argus FrameConsumer
  -> acquire frame image + same-frame metadata
  -> per-camera FrameBundle queue
  -> SyncManager by trigger_id / SOF timestamp
  -> full synchronized cluster
  -> per-camera appsrc
  -> nvv4l2h265enc/h264enc
  -> mux
  -> cam_XX.mkv
```

其中每个 `FrameBundle` 至少包含：

```text
camera
local_frame_number
sensor_timestamp_ns
sof_tsc_ns
eof_tsc_ns
internal_frame_count
image buffer handle
```

如果供应商后续提供硬件 trigger frame id 或 PWM timestamp，则 SyncManager 优先
使用硬件 id / timestamp；否则继续使用 Libargus `sof_tsc_ns` 做 cluster。

## Start/Stop 语义

### Connect

Connect 阶段只做资源准备和 preflight：

1. 检测 locked cameras；
2. 配置 PWM、trigger mode、camera controls；
3. 打开 Argus sessions；
4. 验证每路可以拿到 image frame 和 same-frame metadata；
5. 不写 episode 文件。

### UI Start

UI Start 后 recorder 进入 armed 状态，但不立刻写视频。

流程：

```text
1. 所有相机继续采集 warmup frames。
2. SyncManager 聚类 SOF/trigger timestamp。
3. 找到第一个包含所有 active cameras 的 full cluster。
4. 该 cluster 定义为 logical_frame_index = 0。
5. 为每路打开 encoder/mux。
6. 将 cluster 内每路对应 frame push 到对应 appsrc。
```

这样 encoder 的第一帧就是同步帧，不需要后续裁剪前导帧。

### Duration Stop

对于固定时长 episode：

```text
target_frames = round(episode_time_s * fps)
```

recorder 写满 `target_frames` 个 full cluster 后，对所有 appsrc 同时 EOS。

### Operator Stop

如果 UI 中途 Stop：

1. recorder 收到 stop request；
2. 不在当前半帧状态立即停；
3. 在下一个完整 full cluster 后结束；
4. 所有相机写入相同 logical frame count；
5. 所有 appsrc 同时 EOS。

## 输出合同

每个 episode 保存：

```text
cam_XX.mkv
cam_XX.argus_frame_metadata.csv
online_sync_manifest.json
meta.json
```

### 视频合同

在 online sync backend 下，最终视频文件本身应满足：

```text
cam_03.mkv frame count == N
cam_06.mkv frame count == N
...
```

其中 `N` 是 logical synchronized frame count。

### sidecar 合同

sidecar 每行对应最终视频中的一帧：

```text
camera,logical_frame_index,local_frame_number,
sensor_timestamp_ns,sof_tsc_ns,eof_tsc_ns,internal_frame_count
```

`logical_frame_index` 从 0 连续递增到 `N - 1`。由于只有同步 cluster 会进入
encoder，所以每个相机的 sidecar 行数也必须等于 `N`。

### manifest 合同

`online_sync_manifest.json` 记录 episode 级同步信息：

```json
{
  "ok": true,
  "fps": 60,
  "target_frames": 3600,
  "actual_frames": 3600,
  "sync_source": "sof_tsc_ns",
  "tolerance_ns": 1000000,
  "active_cameras": ["cam_03", "cam_06"],
  "max_abs_delta_ns_by_camera": {
    "cam_03": 12000,
    "cam_06": 11000
  },
  "dropped_clusters_before_start": 31,
  "dropped_clusters_after_stop": 0
}
```

如果供应商提供硬件 trigger id，则 `sync_source` 改为 `trigger_frame_id`。

## 关键技术点

### 1. 必须拿到 same-frame image + metadata

彻底方案不能继续依赖 `nveglstreamsrc` 自动拉 video stream，再用另一条 metadata
stream 旁路推断。应用层需要拿到同一帧的图像 buffer 和 metadata。

理想路径：

```text
Argus FrameConsumer acquireFrame()
  -> image buffer
  -> ICaptureMetadata / ISensorTimestampTsc / IInternalFrameCount
```

如果 Argus API 无法在当前 stream 上同时拿到 image buffer 和 metadata，需要先做
Thor 小样验证。不能证明 same-frame 绑定前，不进入生产实现。

### 2. appsrc 必须保留硬件编码路径

目标不是 CPU 编码。同步后的帧应进入 GStreamer `appsrc`，再走 Jetson 硬件编码：

```text
appsrc is-live=true format=time
  -> nvv4l2h265enc / nvv4l2h264enc
  -> h265parse / h264parse
  -> matroskamux / qtmux
  -> filesink
```

每个 pushed `GstBuffer` 设置：

```text
PTS      = logical_frame_index * 1e9 / fps
duration = 1e9 / fps
```

### 3. 不在 encoded packet 后丢帧

不能在 H.265 packet 后面做 gate。packet 后丢帧会再次遇到 GOP/IDR 依赖问题。
gate 必须在 encoder 前。

### 4. 缺帧策略

生产默认策略建议：

```text
missing_frame_policy: fail_episode
```

如果某一路在中间 cluster 缺帧，不补帧、不复制上一帧，直接标记 episode
失败。边界 warmup frame 可以丢，但一旦开始写 logical frame 0，中间不允许缺帧。

## 过渡方案

可以先做较小改动的验证版，但不作为最终生产合同：

```text
nveglstreamsrc -> valve/pad-probe -> encoder
metadata thread -> SyncManager -> 控制 valve 开关
```

这个方案能验证 “online gate + 固定 N 帧 stop” 的收益，但风险是 video buffer
和 metadata row 是否严格一一对应。如果只能靠顺序推断，则仍然有 off-by-one 风险。

建议把它作为实验项，而不是最终方案。

## 配置建议

新增 backend，而不是覆盖当前 backend：

```yaml
sensors:
  cameras:
    defaults:
      recorder_backend: argus_online_sync

    online_sync:
      enabled: true
      sync_source: sof_tsc_ns      # future: trigger_frame_id / pwm_timestamp
      tolerance_ms: 1.0
      startup_full_clusters: 30
      missing_frame_policy: fail_episode
      stop_mode: full_cluster
      target_frame_count_from_episode_time: true
```

保留现有 `argus_metadata` backend 作为 fallback 和对照测试路径。

## 实施阶段

### Phase 1：两路 same-frame 验证

目标：证明应用层可以拿到同一帧的 image buffer 和 metadata。

验收：

```text
2 路相机
600 full clusters
每路 sidecar 600 行
每路输出视频 600 帧
max SOF delta < 1ms
不调用 argus_video_materialize.py
```

### Phase 2：appsrc 硬件编码验证

目标：同步后的 frame 通过 appsrc 进入硬件 encoder，并生成可播放 MKV。

验收：

```text
ffprobe frame count == logical frame count
PTS 连续，步长 1/60s
无 CPU libx265/libx264 编码进程
```

### Phase 3：SyncManager 多路扩展

目标：扩展到当前 Thor 可用的 7/8 路相机。

验收：

```text
60s episode
每路视频 3600 帧
每路 sidecar 3600 行
online_sync_manifest.ok = true
保存等待时间接近 mux close + meta write，不出现数分钟转码等待
```

### Phase 4：UI 集成

目标：保持 `thor_record.py` stdin 协议不变。

UI 仍然看到：

```text
Connected
Episode 0 ready
Recorded N frames
Episode saved
```

但 recorder backend 切到 `argus_online_sync`。

### Phase 5：生产 burn-in

目标：10/11 路、生产长度、多轮 episode。

验收：

```text
每条 episode 所有 active cameras 帧数一致
无 post-save materialize
无中间缺 cluster
max sync delta 稳定在 tolerance 内
连续采集时 UI 不被保存阶段阻塞
```

## 风险与待确认

1. Argus API 是否能在应用层稳定取得 image buffer + same-frame metadata。
2. appsrc 输入 NVMM/NV12 buffer 到 `nvv4l2h265enc` 的零拷贝路径是否可行。
3. 10/11 路 1080p@60Hz 下，应用层同步队列是否会造成 buffer backlog。
4. 如果某一路短时缺帧，生产策略是失败 episode 还是允许整组跳过 cluster。
5. 后续供应商若提供 trigger frame id，需要把 SyncManager 的 primary key 从
   `sof_tsc_ns` 升级为硬件 id。

## 结论

当前后处理方案已经证明可以用 Argus SOF metadata 找到同步帧，但无法避免
H.265 任意帧裁剪的重编码成本。

彻底解法是新增 `argus_online_sync` recorder：

```text
same-frame image + metadata
  -> online SyncManager
  -> synchronized full clusters only
  -> appsrc
  -> hardware encoder
  -> mux
```

这样同步发生在编码前，最终视频天然等帧数，不需要保存后重编码，也不依赖
UI start/stop 同时到达各个相机进程。
