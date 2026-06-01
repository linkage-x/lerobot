# PR1 实施日志：persistent_session.py

实施日期：2026-05-28
分支：hph/box_dev
硬件：Thor (nvidia@192.168.111.122)
GStreamer：1.24.2

## 背景

方案 A 的第一阶段：实现独立可运行的常驻 pipeline 模块 + demo，
不动 `thor_record.py`，先验证 GStreamer Python binding + splitmuxsink
路径在 Thor 上稳定可用。

## 交付物

- `tools/thor/gmsl2/persistent_session.py` — `PersistentCameraSession` 类 +
  `_Stream` 状态机 + 独立 demo CLI
- `tests/scripts/test_thor_persistent_session.py` — 12 个 mock GStreamer 单测
- `tools/data_collection_gui/docs/option_a_persistent_pipeline_design.md` —
  方案 A 设计文档（含 burn-in 数据）

## 遇到的问题与处理

### 问题 1：SSH 管道导致 stdin 立即 EOF

**现象**：
通过 `printf "\n\nq\n" | ssh thor 'python -m ...'` 驱动 demo 时，
2 路 pipeline 成功连接进 PLAYING，但所有 stdin 命令都没被处理，
程序立即 "disconnected." 退出。

**根因**：
demo 的 `_read_stdin_loop` 读到 EOF（`readline()` 返回空字符串）
立即触发 `stop_event.set()`。SSH 把 `printf` 的输出一次写完后 stdin
关闭，reader 线程读到 EOF 触发 stop 比主循环 dispatch 命令早，
命令队列虽然有内容但循环已经退出。

**处理**：
不去修 reader 线程的 EOF 逻辑（交互场景下 Ctrl-D 确实应该退出），
而是新增一个非交互模式：

```python
ap.add_argument("--auto-episodes", type=int, default=0,
                help="non-interactive mode: record N episodes back-to-back then quit")
ap.add_argument("--inter-episode-gap-s", type=float, default=0.5)
```

主循环判断 `auto_episodes > 0` 时走 for 循环不依赖 stdin，
适合 SSH 驱动的批量 burn-in。

**学到**：
SSH 批量任务不要依赖交互 stdin —— 要么 fake TTY，要么提供
非交互入口。

### 问题 2：跨相机切片偏差 7~10s（远大于设计目标 <50ms）

**现象**：
11 路 burn-in 第一次跑完，meta.json 里 `camera_first_pts_s` 跨相机
差异竟然达到 7~10 秒：

```
episode 0 stopped after 5.01s. fragments:
  cam_15=9.1MB@1.495s
  cam_07=11.5MB@3.197s
  cam_02=10.2MB@8.996s   ← 离 cam_15 7.5s
```

设计目标是 < 50ms（一个 IDR 周期内）。差异 100×。

**根因**：
nvenc 的 `iframeinterval` 只控制 I-frame 周期，**不是** IDR 周期。
splitmuxsink 只能在 IDR 边界切（非 IDR 的 I-frame 在 H.265/H.264 都不
能作为独立解码起点）。nvenc 默认 `idrinterval` 在 JetPack 6 上约
256 帧（4s+ @ 60fps），所以 11 路相机在 split-now 之后要各等到自己的
下一个 IDR，最坏要等满一个 idrinterval。

`gst-inspect-1.0 nvv4l2h265enc` 输出证实：
```
iframeinterval : Encoding Intra Frame occurance frequency
idrinterval    : Encoding IDR Frame occurance frequency  ← 独立属性
```

**处理**：
在 pipeline 描述里同时设两个属性，pin 在同一个值：

```python
encoder = (
    f"{enc_factory} bitrate={...} "
    f"iframeinterval={stream.iframe_interval} "
    f"idrinterval={stream.iframe_interval} "  # ← 新增
    f"preset-level={...} control-rate={...} insert-sps-pps=1"
)
```

单测加 assert 覆盖这个不变式，防止后续被改回去。

**验证**：
修复后再跑 11 路 burn-in：
- `first_pts_s` 跨相机偏差仍然 10.5s（这部分跟 IDR 无关，下面问题 3 解释）
- 但每路文件大小从 10~13MB 散布变成 11.9MB 一致 —— 说明每路实际录到的
  IDR 周期数对齐了
- 关键是 `first_wall_s` 跨相机偏差只有 19.5ms（设计目标内）

**学到**：
NVENC 上 `iframeinterval` ≠ `idrinterval`，搞混会让 splitmuxsink
"看起来在工作但实际切片对不齐"。设计文档里第一版没意识到这一点。

### 问题 3：first_pts_s 不能用于跨相机对齐

**现象**：
即使 idrinterval 修对了，11 路 `camera_first_pts_s` 仍然偏差 10s 量级。

**根因**：
每路 nvarguscamerasrc pipeline 有自己独立的 pipeline clock，
`do-timestamp=true` 把 system clock 转成 buffer PTS 的起点是 PLAYING
时刻。stagger=1.0s × 11 路 → 最后一路 PLAYING 比第一路晚 11s，
PTS 时间轴起点就差 11s。

具体数据（修复后 episode 1）：
```
split_now_wall_s: 1779937344.346777
first_wall - split_now (ms):
  cam_15: +3.0 ms    ← 最早 PLAYING
  cam_02: +8.0 ms
  ...
  cam_00: +18.9 ms   ← 最晚 PLAYING
  cam_07: +22.2 ms
  cam_11: +22.5 ms
first_pts span (s): 10.493     ← pipeline 内部时钟差
first_wall span (ms): 19.5     ← 真实切片落地差
```

**处理**：
不是 bug，是 GStreamer 固有属性。处理方式：
- meta.json 保留 `first_pts_s`（单流分析有用）
- 同时记录 `first_wall_s`（跨流对齐的正确锚点）
- 设计文档 + PR2 待办里明确：**下游对齐用 `first_wall_s`，不要用 `first_pts_s`**
- `thor_lerobot_v3` 现有的 `pts_offset_s` 参数语义在 PR2 中要改成
  "first_wall_s - split_now_wall_s"

**学到**：
分布式时间对齐永远要回到一个公共时钟域。Pipeline PTS 是私有时钟，
host wall-clock 是公共时钟，跨流就只能用后者。

### 问题 4：本机缺 GStreamer Python binding

**现象**：
本地 dev 机器 `python3 -c "import gi"` 失败。

**处理**：
- 单测全部 mock 掉 `gi.repository.Gst`，本机能跑（11/11 通过）
- 真 GStreamer 路径只在 Thor 上验证
- `gst-launch-1.0` 命令行工具本机有，可以离线 `gst-inspect-1.0
  splitmuxsink` 确认 `split-now` / `format-location-full` /
  `async-finalize` API 都存在

**学到**：
跨平台开发：单测必须能在没有目标平台依赖的机器上跑。

## Burn-in 数据

| 配置 | 结果 |
|---|---|
| 11 路 connect 总耗时 | 11.67s（stagger 1.0 × 11 + 0.5s warmup） |
| split-now emit 延迟 | < 0.5ms 跨 episode 一致 |
| **实际切片落地跨相机偏差** (`first_wall_s`) | **19.5ms** |
| 3 连续 episode 文件大小一致性 | 11.9MB / 路（idrinterval 修复后） |
| H.265 mkv 可解码（gst-discoverer） | ✅ Main Profile, ~20.6 Mbps |
| stream error / NvBufSurface 失败 | 0 |

对比设计目标：
- 设计目标 切片精度 < 50ms → 实测 19.5ms ✅
- 设计目标 StartEpisode 等待 < 1s → 实测 < 0.5ms（emit）+ 等 IDR ~17ms ✅
- 设计目标 Connect 一次性付出 → 实测 11.67s（与旧每 episode spawn 持平，
  但只付一次）✅

## 仍待验证 → 已补完（2026-05-28 第二批）

### 验证 1：100+ episode 长 burn-in 内存泄漏

**配置**：4 路 (sids=0,2,3,4) × 1080p60 H.265 × 2s episode × 100 episode，
inter_gap=0.3s。RSS 每 5s 采样一次。

**结果**：
| 指标 | 值 |
|---|---|
| 100 / 100 episode 完成 | ✅ |
| stream error / NvBufSurface 失败 | 0 |
| RSS 起点（ep_done=15） | 463,544 KB |
| RSS 终点（ep_done=99） | 465,896 KB |
| RSS 增长 | **2.3 MB / 100 ep ≈ 23 KB/episode** |
| 总耗时 | ~4 分钟 |
| 数据落盘 | 1.9 GB episodes + 775 MB warmup |

**结论**：**无内存泄漏迹象**。23 KB/episode 增长可解释为 Python list
增长（fragment_history、observations 等），与方案 A 模型一致。
PR2 实施时考虑给 fragment_history 加 trim（保留最近 N 条）。

**遗留问题**：warmup 目录文件累积（775MB / 100 episode）。PR2 应该加
周期清理或者 splitmuxsink 用 `max-files=N` 自动 rotate。

### 验证 2：discard 后 mkv 文件实删 + grace_s 余量

**配置**：4 路真硬件 × 1080p60 H.265 × 3 个 grace 窗口（0.5s / 0.2s /
0.05s）× 3 episode = 9 个 discard 循环。

**结果**：9/9 全部 PASS。
| grace_s | 平均 discard 耗时 | 残留文件 |
|---|---|---|
| 0.5s（默认） | 8.2 ms | 0 |
| 0.2s | 8.2 ms | 0 |
| **0.05s** | **6.3 ms** | **0** |

每个 episode 调用 stop_episode 后会等 `finalize_grace_s` 再 discard_episode
（让 async-finalize 落盘）。**实测 50ms 都够**——splitmuxsink async-finalize
在 4 路 1080p60 H.265 配置下完成时间 < 50ms。

**结论**：默认 0.5s 是过度保守的，可降到 0.1s 缩短 episode 切换间隔
（但 PR1 保持 0.5s 作为安全默认值，PR2 可调）。

### 验证 3：bus message 分发延迟

**配置**：4 路真硬件，monkey-patch `_on_bus_message` 记录每条消息的
跨线程到达时间。两种注入路径：

**Test A：合成 ERROR 直接 post 到 bus**（纯 GLib dispatch 开销）
```
trial 0: 0.15 ms
trial 1: 0.14 ms
trial 2: 0.16 ms
trial 3: 0.14 ms
trial 4: 0.12 ms
→ min=0.12ms max=0.16ms avg=0.14ms (n=5)
```

**Test B：真 EOS via `pipeline.send_event`**（端到端通过 encoder/parser/muxer）
```
EOS bus dispatch = 12.80 ms
```

**结论**：
- GLib MainLoop 跨线程分发 < 200μs（远快于一帧 16.67ms @ 60fps）
- 端到端 EOS 传播 ~13ms，比一帧还短
- PR2 实施错误恢复时可以放心做：检测到 ERROR 后 `_record_error` 几乎
  同步触发，错误处理决策有 ~10ms 时间预算

## 全部三项验证结果摘要

| 验证项 | 状态 | 关键数据 |
|---|---|---|
| 100 ep 内存泄漏 | ✅ PASS | 23 KB/episode 增长，可忽略 |
| discard 文件清理 | ✅ PASS | 9/9 循环；grace=0.05s 都够 |
| bus 分发延迟 | ✅ PASS | dispatch 0.14ms，端到端 EOS 12.8ms |

PR1 全部验证完成，可进入 PR2 (`thor_record.py` 集成)。

## 文件清单

```
tools/thor/gmsl2/persistent_session.py             # 660 行新文件
tests/scripts/test_thor_persistent_session.py      # 12 个单测
tools/data_collection_gui/docs/
  option_a_persistent_pipeline_design.md           # 方案 A 设计 + burn-in 数据
  pr1_implementation_log.md                        # 本文档
```

不影响的文件：`thor_record.py`、`gmsl2_record.py`、`gateway.py`
（PR2 才动）。
