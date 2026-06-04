# 方案 A：常驻 GStreamer Pipeline + splitmuxsink 切片

## 当前架构（基线）

```
StartEpisode → EpisodeSession.start():
  for sid in usable:
    spawn `gst-launch -e ... ! filesink location=episode_NNNNNN/cam_XX.mkv`
    sleep(spawn_stagger_s=1.0)
  sleep(1.0)  # encoder warm-up
  # 共 ~11s for 11 cams

Save → EpisodeSession.stop():
  for s in streams: SIGINT  → gst-launch -e 触发 EOS → muxer finalize
  wait(grace_s=8.0)
```

**核心约束**：filesink 的 `location` 是命令行参数，文件名硬编码到进程。

---

## 目标架构

```
Connect → spawn_persistent_pipelines():
  for sid in usable:
    创建 Gst.Pipeline (Python binding, 不是 gst-launch CLI):
      nvarguscamerasrc → enc → parse → splitmuxsink
                                       ↑
                                       max-size-time=很大 (不自动切)
                                       location=initial_dummy.mkv

  全部 set_state(PLAYING)
  # 仍要 stagger 1s 避免 NvBufSurfaceFromFd 竞争
  # 但只在 Connect 时付一次代价

StartEpisode:
  for pipeline in pipelines:
    splitmuxsink.emit("split-now")
    # format-location 回调返回 episode_NNNNNN/cam_XX.mkv
  # 切片瞬间完成（亚秒级）

Save → 不切，等下一次 StartEpisode 自然触发切片
Discard → 切走当前段，删除前一段文件
Quit/Exit → set_state(NULL) for all pipelines
```

---

## 实施清单

### 1. 新模块：`tools/thor/gmsl2/persistent_session.py`

新建 ~400 行的模块，提供 `PersistentCameraSession` 类，承担当前
`gmsl2_record.EpisodeSession` 的接口角色但语义不同：

```python
class PersistentCameraSession:
    def __init__(self, sids: list[int], cfg: RecorderConfig): ...

    def connect(self) -> list[CameraStream]:
        """Spawn 所有 pipeline 到 PLAYING 状态，返回 stream 列表。"""

    def start_episode(self, episode_dir: Path, episode_idx: int) -> EpisodeHandle:
        """对每个 pipeline emit split-now，文件名通过 format-location 回调注入。
        返回 EpisodeHandle 记录该 episode 的 file paths 和起始 wall-time。"""

    def stop_episode(self, handle: EpisodeHandle) -> list[CameraStream]:
        """切走当前 episode 的录制段。下一段进入临时 location（buffer）。
        返回该 episode 的 CameraStream 列表（保留与旧 EpisodeSession 一致的接口）。"""

    def discard_episode(self, handle: EpisodeHandle) -> None:
        """切走 + 删除对应 mkv 文件。"""

    def disconnect(self) -> None:
        """全部 set_state(NULL)，释放 Argus session。"""
```

GStreamer Python binding 通过 `gi.repository.Gst`（已经在
`thor_lerobot_v3.py:_extract_pts_gstreamer` 用过）。

### 2. Pipeline 构造（关键代码段）

每路用 `Gst.parse_launch` 而非组装单 element：

```python
pipeline_desc = (
    f"nvarguscamerasrc sensor-id={sid} sensor-mode={c.sensor_mode} "
    f"do-timestamp=true "
    f"{exposure_clause}{gain_clause}"
    f"! video/x-raw(memory:NVMM),format=NV12,width={c.width},height={c.height},framerate={c.fps}/1 "
    f"! nvv4l2h{265 if c.codec=='h265' else 264}enc "
    f"  bitrate={c.bitrate_kbps*1000} iframeinterval={c.iframe_interval} "
    f"  preset-level={c.preset_level} control-rate={c.control_rate} insert-sps-pps=1 "
    f"! {c.codec}parse "
    f"! splitmuxsink name=mux_{sid} "
    f"  muxer-factory=matroskamux "
    f"  max-size-time=0 max-size-bytes=0 "  # 完全由 split-now 控制
    f"  async-finalize=true "                # 不阻塞 PLAYING 状态切换文件
    f"  location=/tmp/cam_{sid:02d}_warmup_%05d.mkv"
)
pipeline = Gst.parse_launch(pipeline_desc)
mux = pipeline.get_by_name(f"mux_{sid}")
mux.connect("format-location-full", self._format_location_callback, sid)
```

`format-location-full` 回调由 splitmuxsink 在每次开新文件前调用，
回调里根据当前 episode_dir + sid 返回目标路径。

### 3. format-location 回调

```python
def _format_location_callback(
    self, splitmux, fragment_id, first_sample, sid
):
    state = self._fragment_state.get(sid, FragmentState.WARMUP)
    if state == FragmentState.WARMUP:
        return str(self._warmup_dir / f"cam_{sid:02d}_warmup_{fragment_id:05d}.mkv")
    elif state == FragmentState.EPISODE:
        ep_dir = self._current_episode_dir
        name = f"{self.cfg.name_prefix}_{sid:02d}"
        path = ep_dir / f"{name}.mkv"
        # 记录该 fragment 的 first PTS 和 wall-time
        self._fragment_starts[sid] = (time.time(), first_sample)
        return str(path)
```

### 4. 切片触发

```python
def start_episode(self, episode_dir: Path, episode_idx: int):
    self._current_episode_dir = episode_dir
    episode_dir.mkdir(parents=True, exist_ok=True)
    t0_wall = time.time()
    t0_mono = time.monotonic()
    # 同步触发所有 11 路 split-now
    for sid, pipeline in self._pipelines.items():
        self._fragment_state[sid] = FragmentState.EPISODE
        mux = pipeline.get_by_name(f"mux_{sid}")
        mux.emit("split-now")
    return EpisodeHandle(
        idx=episode_idx, dir=episode_dir,
        t0_wall_s=t0_wall, t0_mono_s=t0_mono,
    )
```

**11 路 split-now 不是原子操作**：触发顺序 ~微秒级，但每个 splitmuxsink
内部要等下一个关键帧才能真正开新文件。在 H.265 IDR 间隔
60（=1s @ 60fps）的配置下，切片可能延迟 0-1s。

**两个缓解方案**：
- a) 把 `iframe_interval` 从 60 改成 30 或更小，最坏延迟降到 0.5s
- b) emit split-now 后强制下一帧 IDR：splitmuxsink 内部本来就要等 IDR，
     可以同时给 encoder 发 `force-IDR` 信号，把延迟压到 ~16ms

### 5. Episode 边界帧精确化

由于切片不是逐帧精确的，必须在 meta.json 中记录每个相机的实际首帧 PTS：

```python
@dataclass
class EpisodeHandle:
    idx: int
    dir: Path
    t0_wall_s: float       # split-now emit 时刻
    t0_mono_s: float
    fragment_first_pts: dict[str, float] = field(default_factory=dict)
    fragment_first_wall: dict[str, float] = field(default_factory=dict)
```

在 stop_episode 后，从 format-location 回调收集的 first_sample
（GstSample）里提取首帧 PTS，写入 meta.json：

```json
"sync_reference": {
  "t0_wall_s": 1716700000.123,
  "split_now_wall_s": 1716700000.125,
  "camera_first_pts": {
    "cam_02": 12.345,
    "cam_07": 12.347
  },
  "camera_first_wall_s": {
    "cam_02": 1716700000.140,
    "cam_07": 1716700000.142
  }
}
```

下游 `thor_lerobot_v3.write_box_lerobot_v3_episode` 不需要变——
它已经接受 `pts_offset_s` 参数。新代码改为使用 `camera_first_pts` 替代。

### 6. Discard 实现

Discard 比当前更复杂——必须先切到下一个文件，再删除前段：

```python
def discard_episode(self, handle: EpisodeHandle):
    # 切换到 EPISODE_DISCARD 状态，让 format-location 返回 /tmp
    for sid in self._pipelines:
        self._fragment_state[sid] = FragmentState.DISCARD_BUFFER
    # 触发切片
    for sid, p in self._pipelines.items():
        p.get_by_name(f"mux_{sid}").emit("split-now")
    # 等待 async-finalize 完成（~100ms）
    time.sleep(0.5)
    # 删除 episode_dir 下的 mkv
    for sid in self._pipelines:
        name = f"{self.cfg.name_prefix}_{sid:02d}"
        (handle.dir / f"{name}.mkv").unlink(missing_ok=True)
```

**坑**：async-finalize 异步写文件，删除前需要确认 fragment 已落盘。
splitmuxsink 有 `format-location-full` 在新文件创建时回调，但旧文件
finalize 没有公开信号。需要等待或用 `Gst.Pad.add_probe` 监听 EOS event。

### 7. 集成到 thor_record.py

`thor_record.py` 的 main loop 改成：

```python
session = PersistentCameraSession(usable, cfg)
streams = session.connect()  # 一次性 spawn
_mark_connected(...)
_emit(f"Episode {ep_idx} ready")

while not stop_event.is_set():
    cmd = _wait_for_command(...)
    if cmd.kind == "quit": break

    handle = session.start_episode(ep_dir, ep_idx)
    box.start_recording(t_start)
    # ...原有的轮询循环...
    streams = session.stop_episode(handle)  # 同样语义

    if decision == "save":
        _write_episode_meta(...)
    elif decision == "discard":
        session.discard_episode(handle)

session.disconnect()
```

`gmsl2_record.EpisodeSession` 保留不变（gmsl2_record.py 的 CLI 入口仍可用）。

### 8. 错误恢复

当前架构里 stream 提前 EOS 会触发 `stop_on_stream_exit`，整个 episode
中止。常驻 pipeline 下情况更糟——一路相机挂掉，后续所有 episode 都缺这路。

需要新增：
- pipeline bus 监听 ERROR/EOS message，自动重启对应 sid 的 pipeline
- 重启需要 1-2s，期间该 episode 该路相机数据缺失，meta.json 标记 `stream_recovered=True`
- 重启失败 N 次后该 sid 标记 dead，不再尝试

### 9. BOX 采集板影响

零影响。BOX 路径完全独立于 GStreamer，`box.start_recording(t_start)` 仍按
episode 维度 start/stop。但 `t0_wall_s` 现在含义变了——是 `split-now` emit
时刻，不是 spawn 时刻。BOX 的 `t_rel_s` 计算需要用 `split_now_wall_s` 而非
旧的 `t0_wall_s`。

### 10. 资源占用

- **CPU**: 11 路硬件编码器持续运行。NVENC 是独立硬件单元，CPU 开销几乎不变
- **功耗**: 编码器持续工作约 +5-8W。Thor 板典型功耗 30W，差异可忽略
- **磁盘**: WARMUP 状态下文件写到 /tmp，需要定期清理（splitmuxsink
  不会自动删旧 fragment）
- **内存**: 11 路 pipeline 常驻 ~200MB（每路 ~15MB GStreamer 元数据 + 编码器
  buffer）

---

## 风险点

| 风险 | 概率 | 缓解 |
|---|---|---|
| 11 路 split-now 后切片延迟 > 100ms 不一致 | 中 | 缩短 iframe_interval；强制 IDR；meta.json 精确记录每路 first PTS |
| async-finalize 时删除文件竞争 | 中 | 等待 EOS event；fallback 写入 trash 目录定时清理 |
| 单路 pipeline 挂掉影响后续 episode | 高 | bus 监听 + 自动重启该路 |
| Connect 时 11 路同时 PLAYING 仍触发 NvBufSurface 竞争 | 高 | 保留 spawn_stagger_s 不变（只在 Connect 一次） |
| GStreamer Python binding 在 Jetson aarch64 上的版本不一致 | 低 | 已经在 thor_lerobot_v3.py 用过 |
| splitmuxsink 内部 bug（async-finalize 在某些版本竞态） | 低 | GStreamer 1.20+ 已修；JetPack 6 系 1.20.3 OK |
| 编码器长时间运行的稳定性（连续 8h+ 录制） | 中 | 需要 burn-in 测试；提供 manual disconnect/reconnect |

---

## 测试矩阵

### 单元测试（mock GStreamer）

- `PersistentCameraSession.connect/disconnect` 状态机
- `format-location` 回调返回正确路径
- Discard 后文件确实被删除

### 集成测试（真硬件）

1. **冷启动 Connect**: 11 路 spawn 时间 + 是否全部进 PLAYING
2. **快速连录**: 录 10 个 5s episode 连续 save，验证：
   - 每个 episode 的 .mkv 都存在且可解码
   - 每个 .mkv 的 PTS 范围匹配 episode 时长 ± 50ms
   - meta.json 的 `camera_first_pts` 反映实际首帧时刻
3. **Discard 后立即 Start**: 验证下一个 episode 的 mkv 不包含上一段数据
4. **单路 EOS 模拟**: kill 某路 encoder PID，验证自动重启
5. **长时间录制**: 连续录 100 个 episode，~30 分钟，验证无内存泄漏
6. **L3b 对齐精度**: 用已有的 PTS 校准代码处理新结构 meta.json，对比旧路径精度

### 回归测试

- 旧 `gmsl2_record.EpisodeSession` CLI 路径不动，独立测试集仍跑
- BOX LeRobot v3 写入路径不变（pts_offset_s 接口已存在）
- gateway.py 的 `_gmsl2_episode_dirs` 等扫描函数无需改

---

## 工作量估算

| 模块 | 工时 | 说明 |
|---|---|---|
| persistent_session.py 编写 | 3-4 天 | GStreamer Python binding + 状态机 |
| thor_record.py 集成 | 1 天 | 替换 EpisodeSession 调用 |
| meta.json 结构演进 | 0.5 天 | sync_reference 字段调整 |
| 错误恢复（pipeline 重启） | 1-2 天 | bus 监听 + 重启逻辑 |
| 单元测试 | 1 天 | mock Gst 状态机 |
| 真硬件集成测试 | 2-3 天 | 11 路稳定性 + 长时间录制 |
| 文档 + DEPLOYMENT.md 更新 | 0.5 天 | |
| **合计** | **9-12 天** | 单人 |

---

## 是否值得做

- **收益**：StartEpisode 等待从 ~11s → <1s。对快速连录场景（数据采集员
  一小时录 50+ 个 episode）累计节省 ~8 分钟/小时
- **代价**：~10 天开发 + 长期维护一套 GStreamer Python pipeline 状态机
- **替代方案 B（warm-up + 降 stagger）**：1 天工作，能把等待压到 ~3s

**决策建议**：除非"快速连录"是硬需求，优先方案 B；如果数据采集任务
本身就是高吞吐场景（比如每天录 500+ episode），方案 A 的投入有回报。

---

## 增量路径

如果决定做方案 A，建议分两个 PR：

1. **PR1**: 新建 `persistent_session.py`，独立可运行的 demo 脚本（不影响 gateway）
2. **PR2**: thor_record.py 切换 + meta.json 演进 + 集成测试

中间留 2-3 天给真硬件 burn-in。

---

## PR1 实施状态（2026-05-28）

✅ **已落盘**：

- `tools/thor/gmsl2/persistent_session.py` — `PersistentCameraSession`
  类 + `_Stream` 状态机 + `build_pipeline_desc` + format-location-full
  回调 + 独立 demo CLI (`python -m tools.thor.gmsl2.persistent_session
  --use-test-source` 在普通机器上可跑 videotestsrc 路径)
- `tests/scripts/test_thor_persistent_session.py` — 11 个 mock GStreamer 的
  单元测试，覆盖 pipeline 描述串组装、format-location 状态机切换、CLOCK_TIME_NONE
  容错、meta.json schema、discard 文件删除、episode index 扫描
- 离线验证：`gst-inspect-1.0 splitmuxsink` 在 GStreamer 1.24 上确认
  `split-now` action signal、`format-location-full` 回调签名、
  `async-finalize` 属性都存在

✅ **Thor 真硬件 burn-in 完成（2026-05-28，sids=0,2,3,4,5,7,9,10,11,14,15）**：

| 指标 | 结果 | 备注 |
|---|---|---|
| 11 路 connect 总耗时 | 11.67s | stagger=1.0 × 11 + 0.5s warmup |
| split-now emit 延迟 | < 0.5ms | 跨 episode 一致 |
| **实际切片落地偏差** (`first_wall_s`) | **19.5ms** | 11 路在 +3ms~+22.5ms 窗口内 |
| first_pts_s 跨相机偏差 | 10.5s | pipeline 内部时钟独立，**用 wall_s 对齐** |
| 3 连续 episode 文件大小一致性 | ~11.9MB/路 | idrinterval=iframeinterval=30 后稳定 |
| stream error / NvBufSurface 失败 | 0 | |
| H.265 mkv 可解码（gst-discoverer） | ✅ | Main Profile, ~20.6 Mbps |

**关键发现**：

1. `nvv4l2h{264,265}enc` 必须同时设 `iframeinterval` 和 `idrinterval`。仅设
   前者时 IDR 间隔走默认 ~256 帧（4s+），splitmuxsink 等 IDR 才能切，导致
   跨相机切片错开 7~10s。同时设两者后压到 IDR 周期内（设计目标 0.5s @
   iframe_interval=30）。代码已修复。
2. `first_pts_s` 不可用于跨相机对齐（pipeline 时钟独立），但精确记录在
   meta.json 中以备单流分析。
3. **`first_wall_s` 是下游 BOX↔camera 对齐的正确锚点**，精度 ~20ms，优于
   旧架构的 ~25ms（L1 sync_reference）。

✅ **PR1 第二轮 burn-in 全部通过（2026-05-28）**：

| 验证项 | 状态 | 关键数据 |
|---|---|---|
| 100 ep 内存泄漏（4 路 × 2s × 100） | ✅ PASS | RSS +23 KB/ep，无 stream error |
| discard 文件清理（4 路，grace 0.5/0.2/0.05s） | ✅ PASS | 9/9 循环全清；50ms 都够 |
| bus 分发延迟（合成 ERROR） | ✅ PASS | 跨线程 0.14ms avg |
| bus 分发延迟（端到端 EOS） | ✅ PASS | 12.8ms 包含 source→sink 全链路 |

详细数据见 `pr1_implementation_log.md`。

⏳ **PR2 前仍待验证**：
- BOX 采集板同时启动时的整体功耗/稳定性

## PR2 待办（thor_record.py 集成）

PR1 通过 burn-in 后，PR2 做：

1. `thor_record.py.main()` 把 `gr.EpisodeSession(...)` 替换成
   `PersistentCameraSession(...)`，Connect 后保留 session，循环里只调用
   `start_episode`/`stop_episode`/`discard_episode`
2. meta.json 的 `sync_reference` 从 `camera_spawn_*_s` 改成新模型的
   `camera_first_*_s`；`gmsl2_record.write_episode_meta` 保留为 legacy
   path（CLI 入口仍可用）
3. **下游对齐用 `first_wall_s` 而非 `first_pts_s`**（burn-in 验证了 pipeline
   PTS 跨相机不可比，wall-clock 才精确到 ~20ms）。
   `thor_lerobot_v3.write_box_lerobot_v3_episode` 的 `pts_offset_s` 参数
   语义改为接收 `first_wall_s - split_now_wall_s`（每路相对 episode start
   的延迟）
4. 错误恢复：bus message 触发的 `StreamError` 不仅记录，还要尝试 NULL→PLAYING
   重启对应 pipeline；连续失败 N 次标记 dead
5. gateway.py 的 `_apply_recorder_output` 不需要改——thor_record 的 stdout
   协议保持不变
6. `gmsl2_episode_dirs` / `_gmsl2_dataset_stats` 等扫描函数不需要改
7. DEPLOYMENT.md 更新：StartEpisode 等待从 ~11s 降到 < 0.1s（split-now emit）
8. PR2 前应在 Thor 上跑一次 100 episode 长 burn-in，确认无内存泄漏
