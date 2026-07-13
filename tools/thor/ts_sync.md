# Thor 相机-传感器时间同步技术文档

> 适用于 Thor 数据采集平台（11 × GMSL2 相机 + BOX 采集板）
> 初版 2026-05-27；2026-06-09 按实际代码实现校订（PTS offset 机制、meta 字段、ffprobe 角色）
> 2026-06-15 按真机实测校订（各传感器频率、L3b 校准残差；MCU 时钟 = 1µs/tick，6 路全 engage）
> 2026-06-16 schema 精简（observation.state 31 维 / box.timestamps 6 维）+ 去 meta 冗余（`sync_reference` 删 split_now_wall_s、camera_first_pts_s）
> 2026-07-13 按最近 8 次同步相关改动校订：生产默认相机路径切到 `argus_online_sync`，SOF full-cluster 在 encoder 前对齐；新增 online frame bus / preview bus / replay 多视频同步说明。

## 1. 系统总览

Thor 采集系统包含两套独立的数据源：

| 数据源 | 硬件 | 传输方式 | 帧率 |
|--------|------|----------|------|
| 8-11 路 GMSL2 相机 | SG16A + AR0234C 传感器 | Libargus online-sync → H.265 MKV + metadata sidecar | 60 fps |
| BOX 采集板传感器 | MCU + gripper / IMU / trigger / 6D force / touch×2 | UDP/15000 | 各传感器独立，~50-200 Hz |

**核心挑战：两套数据源的时钟域完全独立，没有硬件级公共时间基准。**

```
┌─────────────────────────────────────────────────────┐
│                    Jetson 主机                       │
│                                                     │
│  ┌─────────┐   PWM 60Hz    ┌──────────────────┐    │
│  │ pwmchip │──────────────▶│ SG16A deserializer│    │
│  └─────────┘   trig_pin    │  11× AR0234C     │    │
│                             │  trig_mode=1     │    │
│                             └──────┬───────────┘    │
│                                    │ GMSL2          │
│                     ┌──────────────▼──────────┐     │
│                     │ Libargus BufferOutput    │     │
│                     │ SOF full-cluster gate    │     │
│                     │ → H.265 MKV + sidecar    │     │
│                     └─────────────────────────┘     │
│                                                     │
│  ┌──────────┐  UDP/15000   ┌──────────────────┐    │
│  │ box_sdk  │◀─────────────│ BOX MCU          │    │
│  │ Python   │              │ 独立晶振时钟      │    │
│  └──────────┘              └──────────────────┘    │
│                                                     │
│  ┌──────────────────────────────────────────┐       │
│  │ 主机 wall-clock: time.time()             │       │
│  │ 唯一的公共时间参考                         │       │
│  └──────────────────────────────────────────┘       │
└─────────────────────────────────────────────────────┘
```

### 1.1 软同步工作原理（TL;DR）

**一句话**：生产默认相机路径现在先在 Libargus recorder 内按同一次 SOF 聚成 full cluster，再把同步帧送入 encoder；BOX 仍以主机时间为桥，通过 MCU→host 线性校准后贴到相机帧网格。

1. **公共 episode 原点**：每条 episode 记录 `t0_wall_s` / `t0_mono_s`。BOX 样本用 `t_rel_s = wall_s - t0_wall_s`；相机生产路径用 recorder 的 `logical_frame_index` 表示已通过 SOF gate 的同步帧序号。
2. **相机生产默认路径：`argus_online_sync`**（详 §3.2 / §5.1）：recorder 从每路 Argus Buffer 取 same-buffer metadata，以 `sof_tsc_ns` 找到所有 active camera 都存在、且 SOF spread 不超过 `online_sync.tolerance_ms` 的 full cluster；只有 full cluster 才进入硬件 encoder/mux。视频第 N 帧、sidecar 第 N 行、online frame bus 的 `logical_frame_index=N` 指向同一个同步 cluster。
3. **相机 legacy 路径**（详 §5.1.2）：`gstreamer_splitmux` / 早期 persistent pipeline 仍使用 `pts_offset = mean(camera_first_wall_s - t0_wall_s)` 重建 `pts_offset + N/fps` 的 t0 相对帧网格。`argus_metadata` 可保存后按 SOF 对齐并 materialize，但不再是生产默认。
4. **BOX**（详 §4 / §5.2）：500Hz 轮询，按各传感器 MCU 时间戳变化去重（原生 199/50Hz 独立记录）；再对每个传感器做 `host = slope·mcu + intercept` 最小二乘回归（实测 slope = 1µs/tick，残差 1–2ms）消除轮询抖动，得 `t_rel_s = 校准时间 − t0_wall_s`。
5. **合并**（详 §6 / §9）：对每个 60Hz 相机 logical frame，在每个传感器序列里二分查找最近样本 → 拼成 `observation.state`；对齐所用的原始 MCU 戳单独存入 `box.timestamps`。

**精度口径**：

- 相机视频间：`argus_online_sync` 以 SOF TSC 在 encoder 前 gate；2026-07-07 8 路 10×60s burn-in 最大 SOF spread 0.401ms，典型单 episode 可到十几微秒量级。
- BOX↔相机：仍受 BOX MCU 校准残差与最近邻量化影响；200Hz 传感器约 ±1–3ms 校准残差并另叠 ±2.5ms 采样量化，touch 50Hz 主导项为 ±10ms 最近邻量化。

**信息载体**：`meta.json` 的 `sync_reference`（episode 原点 + legacy/兼容锚点）+ `cam_XX.argus_frame_metadata.csv`（每个最终视频帧的 SOF/EOF/sensor timestamp）+ `online_sync_manifest.json`（保存 gate 结果）+ 训练 parquet（`timestamp` 帧网格 / `box.timestamps` 对齐戳）+ `box_sensors.jsonl`（原始全速率，可重算校准）。

## 2. 三个时钟域

| 时钟域 | 来源 | 精度 | 特点 |
|--------|------|------|------|
| **PWM 硬件时钟** | Jetson pwmchip, 60Hz 方波 | 亚微秒级 | 仅产生触发边沿，不输出时间戳 |
| **主机 wall-clock** | Linux `time.time()` / `CLOCK_REALTIME` | 微秒级（NTP 校准后） | 所有软件层的公共参考 |
| **BOX MCU 时钟** | 采集板内部晶振 | 未知精度（典型 ±50ppm） | 仅在 MCU 侧单调递增，与主机无校准关系 |

PWM 时钟只负责触发相机快门，不产生可读取的时间戳。当前生产路径用 Argus SOF TSC 验证并保存相机 full cluster；BOX↔相机跨域对齐仍依赖主机 wall-clock 作为桥梁。

## 3. L0：相机间硬同步（PWM slave mode）

### 原理

11 路 AR0234C 传感器通过 SG16A deserializer 板共享同一路 PWM 触发信号。每个相机被设为 `trig_mode=1`（slave mode），帧采集由 PWM 上升沿触发。

```
PWM ──┬──▶ cam_00 (sensor_id=0)  ─┐
      ├──▶ cam_02 (sensor_id=2)  ─┤
      ├──▶ cam_03 (sensor_id=3)  ─┤  同一 PWM 边沿
      ├──▶ ...                   ─┤  触发所有快门
      └──▶ cam_15 (sensor_id=15) ─┘
```

### 配置

```yaml
hardware_sync:
  enabled: true
  fps: 60                  # PWM 频率
  sensor_trig_mode: 1      # 1 = slave (PWM-edge locked)
  trig_pin: 0x00020007     # v4l2-ctl 写入每个相机
```

启动时执行：
1. `sudo sh pwm.sh` — 配置 pwmchip 输出 60Hz 方波
2. `v4l2-ctl -d /dev/videoN -c trig_mode=1,...` — 每个相机切入 slave 模式

### 精度

- **相机间对齐：<1μs** — 所有帧物理锁定到同一 PWM 边沿
- 帧间间隔精确为 1/60 s = 16,666.67 μs

### 注意事项

- **曝光时间约束**：slave mode 下 `exposure_us + readout_time` 必须 < PWM 周期，否则 AR0234 回退到 ~0.8fps。录制器自动 clamp 到 `0.85 × (1e6 / fps)` = 14,166 μs
- **spawn stagger / legacy 路径**：旧 `gstreamer_splitmux` 路径中，11 路 `nvarguscamerasrc` 同时初始化会触发 Argus ISP 的 NVMM buffer 分配竞争（`NvBufSurfaceFromFd Failed`），需错开 1.0s 逐路启动。这不影响 PWM 物理触发，只影响各路开始落盘的起始帧。生产默认 `argus_online_sync` 改为 recorder 内部统一打开 Argus stream，并以 SOF full cluster 作为保存边界。

### 3.2 Encoder 前 SOF full-cluster gate（生产默认）

`argus_online_sync` 在硬件 encoder 之前执行同步判断：

```text
Argus BufferOutputStream
  -> IBuffer::getMetadata() / ISensorTimestampTsc
  -> 按 sof_tsc_ns 找 full cluster
  -> 只有完整且 SOF spread <= tolerance 的 cluster 进入 encoder/mux
  -> cam_XX.mkv + cam_XX.argus_frame_metadata.csv + online_sync_manifest.json
```

每个 accepted cluster 的合同：

- 所有 active camera 都有一帧；
- cluster 内 `max(sof_tsc_ns) - min(sof_tsc_ns) <= online_sync.tolerance_ms`；
- `logical_frame_index` 从 0 连续递增，且每路 sidecar 第 N 行对应视频第 N 帧；
- 录制窗口中间缺任一路 full cluster 时 episode 失败，不补帧、不复制帧、不静默裁剪。

`online_sync_manifest.json` 是保存 gate：`ok=true`、每路 `frame_count_by_camera` 一致、`max_abs_delta_ns_by_camera` 不超过阈值才允许保存。2026-07-07 8 路 10×60s burn-in 结果为每路 3600 帧，最大 SOF spread 0.401ms，无 ffmpeg materialization。

## 4. L3a：高频独立采样 + 逐传感器软同步

### 问题

BOX 采集板的 6 类传感器（gripper / IMU / trigger / 6D force / touch L / touch R）通过同一 UDP 端口上行。SDK 的 `get_sensor_cache()` 返回一个包含所有传感器最新值的缓存快照。

如果以低频（如 20Hz）轮询，所有传感器的有效采样率都被压到 20Hz，无论其硬件原生速率如何。

### 方案

以 500Hz 轮询 SDK，通过比较各传感器的 **MCU 时间戳** 检测新样本：

```python
# poll loop 内的去重逻辑
for sid, ts in sensor_timestamps.items():
    if ts != last_recorded_ts[sid]:   # MCU 时间戳变了 → 新样本
        last_recorded_ts[sid] = ts
        samples[sid].append(SensorSample(
            mcu_timestamp=ts,
            wall_time_s=time.time(),   # 主机收到时刻
            data=decoded_sensor_data,
        ))
```

500Hz 轮询保证对任何 ≤250Hz 的传感器都满足 Nyquist 条件。每个传感器按 MCU 实际推送频率独立记录：

| 传感器 | 实测观测频率 | 每 10s episode 样本数 |
|--------|------------|---------------------|
| IMU | 199 Hz | ~1998 |
| 六维力 | 199 Hz | ~1998 |
| Gripper | 199 Hz | ~1998 |
| Trigger | 199 Hz | ~1998 |
| Touch L | **50 Hz** | ~500 |
| Touch R | 50 Hz | ~500 |

> 频率为 2026-06-15 真机实测（左右触觉垫**均为 50Hz**——早期文档误记 Touch L 为 200Hz）。
> 这影响最近邻对齐误差：touch 在 50Hz 下为 ±10ms（而非 200Hz 的 ±2.5ms）。

### 数据持久化

- **原始数据**：`box_sensors.jsonl`（per episode），每行一个传感器样本
  ```json
  {"sid":"box_imu","mcu_ts":12345,"wall_s":1716700000.123,"t_rel_s":0.456,"data":{...}}
  ```
- **训练格式**：LeRobot v3 parquet，对每个 60Hz 相机帧做逐传感器最近邻插值

### 对齐方式

对齐基于每个 episode 的相机帧网格：

```
argus_online_sync: camera frame N = logical_frame_index N / fps
legacy splitmux:   camera frame N = pts_offset + N / fps   （t0 相对域）
BOX sample time:   calibrated_t_rel_s 或 poll t_rel_s
```

对每个相机帧时间点 t，在每个传感器的时间序列中用**二分查找**找到 t0 相对时间最接近的样本，组成该帧的 state 向量。

### 理论精度

对齐误差 = 传感器采样间隔/2 + 主机侧抖动

- 传感器采样间隔/2：200Hz → ±2.5ms，50Hz → ±10ms
- 主机侧抖动：UDP 传输延迟 + Python poll 调度抖动（典型 ~1-3ms）
- **合计：±3~13ms**

## 5. L3b：相机帧网格 + MCU 时钟校准

L3a 的两个主要误差源：

1. **相机侧**：legacy splitmux 需要从每路独立视频文件推断共同帧网格；生产默认 `argus_online_sync` 已在 encoder 前只保存 full SOF cluster，因此相机侧核心合同变为 `logical_frame_index` 连续且跨相机同义。
2. **BOX 侧**：每个样本的 `wall_time_s` 是主机**收到**时刻，包含 UDP 传输延迟和 poll 调度抖动（每次不同）。

### 5.1 相机侧：online-sync logical frame 与 legacy pts_offset

#### 5.1.1 `argus_online_sync`（生产默认）

`argus_online_sync` 不再把“同步”推迟到保存后处理，而是在 recorder 内部按 Argus same-buffer metadata 做 encoder 前 gate：

```text
raw Argus frame + metadata
  -> sof_tsc_ns full-cluster matcher
  -> logical_frame_index = accepted_cluster_count
  -> encoder/mux + sidecar + optional frame bus
```

因此相机侧训练/回放网格以 logical frame 为准：

```
camera_time[N] = N / fps
```

对应产物合同：

- `cam_XX.mkv`：第 N 帧就是第 N 个 accepted full SOF cluster；
- `cam_XX.argus_frame_metadata.csv`：第 N 行含同一帧的 `logical_frame_index=N`、`sensor_timestamp_ns`、`sof_tsc_ns`、`eof_tsc_ns`；
- `online_sync_manifest.json`：记录 `actual_frames`、每路 frame count、每路最大 SOF delta、`sync_source=sof_tsc_ns`；
- `meta.json.sync_reference.camera_first_wall_s` 在该 backend 下是兼容字段，通常等于 `t0_wall_s`，**不是硬件帧时间戳**；不要用它判断 online-sync 视频内部跨相机同步。

UI 录制路径是 **time-driven**：`thor_record.py` 按操作者 Stop / `episode_time_s` 到时发送 `STOP`，recorder 在下一个 full cluster 边界关闭；因此 `actual_frames` 可能与 `round(duration_s * fps)` 或 parquet 行数差 1-2 帧。合同是“各相机同帧数、同 logical frame 同 SOF cluster”，不是强制固定帧数。`online_sync_burnin.py --enforce-exact-frames` 只用于直接测试固定帧数场景。

#### 5.1.2 legacy `gstreamer_splitmux` / `argus_metadata`

早期 `gstreamer_splitmux` 路径无法在 encoder 前证明“第 N 帧同 SOF”，只能在保存后用首帧 host wall-time 估计共同帧网格。真机 burn-in 发现 `first_sample.pts` 跨流不可比（每路 pipeline clock 独立，`first_pts_s` 可差 10s 量级），因此 legacy 对齐使用 host wall-time：

```
pts_offset = mean_over_cams(first_wall_s[cam] - t0_wall_s)
frame_time[N] = pts_offset + N / fps        （t0 相对域）
```

其中 `first_wall_s` 来自 worker 在 fragment 首帧落盘时记录的 `time.time()`，明细写入 `meta.json.sync_reference.camera_first_wall_s`。这条路径仍用于兼容旧数据和无 online-sync sidecar 的 camera-only 数据。

`argus_metadata` 是过渡方案：先写完整视频，再用 `cam_XX.argus_frame_metadata.csv` 和 `argus_frame_alignment.json` 找同步窗口，必要时 materialize。它比 splitmux 可验证，但保存后 H.265 任意窗口切片/重编码太慢，因此已被 `argus_online_sync` 替代为生产默认。

> **ffprobe 角色**：录制对齐路径不依赖 ffprobe。ffprobe/GStreamer PTS 只用于离线数帧、QC 或 export_v3 转码检查，不能作为跨相机同步锚点。

### 5.2 BOX 侧修正：MCU↔Host 时钟线性回归

录制期间每个传感器积累大量 `(mcu_ts, host_wall_time)` 观测对（10s episode × 200Hz = ~2000 对）。对每个传感器做最小二乘线性回归：

```
host_time = slope × mcu_ts + intercept
```

- `slope` ≈ MCU 时钟周期（ticks → seconds），反映 MCU 晶振频率
- `intercept` = MCU 时钟域到主机时钟域的偏移量

拟合完成后，用每个样本的 MCU 时间戳**反推**更准确的主机时间，消除逐次 poll 的随机抖动：

```python
# 校准前：wall_time_s 包含 ~1-3ms 随机 poll 抖动
# 校准后：calibrated_time = slope * mcu_ts + intercept
#         残差标准差实测约 1-3ms
```

### 5.3 安全阈值

校准不总是有效。`calibrate_sensor_samples()` 在以下情况自动回退到 L3a 的原始 poll 时间
（`t_rel_s` 原值不变）：

- 该传感器样本数 < 10（`len(slist) < 10`，不足以拟合）
- MCU 时间戳全为 0（`not any(mcu_ts)`，传感器未上报有效时间戳）
- 回归残差标准差 > 50ms（`res_std > 0.05`，拟合质量差，可能 MCU 时钟不是线性的）
- 退化拟合（`slope == 0.0`，含样本 < 2 或 x 方差为 0 的情形，`calibrate_mcu_clock` 返回 `(0, 0, inf)`）

注意回退是**逐传感器**的：某个传感器拟合差不影响其他传感器使用校准时间。

### 5.4 合成精度

| 修正项 | 消除/约束的误差 | 修正前 | 修正后 |
|--------|----------------|--------|--------|
| `argus_online_sync` SOF full-cluster gate | 多视频第 N 帧是否同一次相机曝光/读出 | 保存后才检查或无法证明 | encoder 前只保存同一 SOF cluster；burn-in 最大 spread 0.401ms |
| legacy 首帧 wall-time 偏移（pts_offset） | splitmux 管道启动延迟（100-500ms 偏移） | 全局偏移 | 偏移 ~主机墙钟精度（亚毫秒~毫秒级） |
| MCU 时钟校准 | 逐次 poll 随机抖动（1-3ms） | ±1-3ms/样本 | ~1ms/样本（200Hz 主传感器），~2-3ms（50Hz touch） |
| BOX↔相机端到端 | BOX 定时残差 + 最近邻量化 | ±3~13ms | 200Hz 传感器约 ±1-3ms 残差并另叠 ±2.5ms；touch 50Hz 另叠 ±10ms |

> 修正后数值为 2026-06-15 真机实测的回归残差标准差（`recorder_*.log` 的 `MCU clock calibration`）：
> gripper 1.12ms · imu 1.07ms · trigger 1.07ms · 六维力 1.18ms · touch L 2.02ms · touch R 1.97ms。
> 比早期文档估计的 <0.5ms 略保守，但仍远优于 L3a 的 ±3~13ms。`slope` 实测恒为 `1.0e-6`，
> 即 **MCU 时间戳单位 = 微秒**；6 路传感器在真机上全部 engage（无回退）。

**±1~3ms 的理论来源**：该列即 MCU↔Host 线性回归的**残差标准差**。拟合 `host = slope·mcu + intercept`
把延迟里的**固定偏移**收进 `intercept`、**两时钟频率差/漂移**（晶振 ±50ppm）收进 `slope`；**残差 =
剩下既非常量也非线性的部分**，也就是「MCU 给样本打戳 → 主机 `time.time()` 记录」之间延迟的**随机
抖动**。校准能去掉延迟的恒定与线性成分，抖动去不掉，就留成这 1~3ms。抖动按量级：

1. **poll 相位抖动（主因）**：样本落入 SDK 缓存后，要等下一拍 500Hz 轮询才被读到并记 `time.time()` → 0~2ms 近似均匀。
2. **UDP / 内核 / 网卡传输抖动**：打包、网络传输、中断合并、收包缓冲。
3. **Python 调度抖动**：GIL 争用 / GC 暂停 / OS 调度 poll 线程。
4. **设备侧采样→发包延迟波动**。（MCU 时钟 1µs 量化、10s 窗口内高度线性，可忽略。）

touch 残差约为 200Hz 传感器的 2×：样本少 4×（501 vs 1998，拟合更糙）+ UDP 包更大（~744B vs ~120B，传输抖动更大）+ §4 提到的 touch 投递速率 quirk。

> **口径提醒（避免误读这个数）**：±1~3ms 只是 **BOX 侧的「定时」残差**（已知某样本发生在主机时间
> 几时的不确定度）。**每帧端到端**对齐误差还要再叠两项校准**消不掉**的：(a) **最近邻量化 ±采样间隔/2**
> ——199Hz ±2.5ms、**touch 50Hz ±10ms**（见 §4）；(b) legacy splitmux 相机侧 `pts_offset` 的测量抖动
> （online-sync 数据则由 SOF full-cluster gate 约束相机间同帧）。另外延迟的**平均值**被 `intercept` 吸收、不进残差，但它在
> BOX↔相机之间留下一个固定 skew（BOX 内部各传感器对齐时相消）。**所以对 touch，主导误差是 ±10ms
> 的最近邻量化，而非 ±2ms 的校准残差。**

## 6. 完整对齐流程（per episode）

### 6.1 生产默认：`argus_online_sync`

```text
Connect
  ├─ detect locked GMSL2 cameras + Argus preflight
  ├─ apply PWM / trig_mode / exposure clamp
  ├─ start BOX SDK first（避免 Argus/GStreamer 高负载下 SDK .so/UDP 初始化崩溃）
  └─ start persistent argus_online_sync recorder daemon
       └─ idle 阶段持续消费 full SOF cluster，可选发布 preview bus

Start episode
  ├─ box.start_recording(t0_wall_s)
  ├─ recorder START idx frames=0 episode_dir
  └─ recorder 丢弃 startup_full_clusters 后进入 recording window

Recording window
  ├─ 每个 full SOF cluster 进入 encoder/mux
  ├─ 写 cam_XX.argus_frame_metadata.csv 第 N 行
  ├─ 可选发布 /dev/shm/lerobot_online_sync latest cluster 给在线推理
  └─ BOX poll loop 500Hz 去重记录 per-sensor samples

Stop / auto-duration
  ├─ thor_record.py 发送 STOP（time-driven）
  ├─ recorder 在下一个 full cluster 边界关闭 episode
  ├─ 写 online_sync_manifest.json（actual_frames、frame_count_by_camera、max_delta）
  ├─ box.stop_recording() 返回 {sensor_id: [SensorSample, ...]}
  ├─ 写 box_sensors.jsonl（原始数据归档）
  └─ 写 LeRobot v3 parquet
       ├─ 逐传感器 MCU 时钟校准（calibrate_sensor_samples 线性回归 + 安全回退）
       ├─ 帧时间网格 = logical_frame_index / fps
       └─ 对每帧逐传感器二分查找最近邻 → 组成 state 向量
```

保存 gate：`online_sync_manifest.ok` 必须为 true，且所有 active camera 的 `frame_count_by_camera` 一致；`missing_frame_policy=fail_episode` 时 recording window 内缺 full cluster 会丢弃该 episode。

### 6.2 legacy splitmux / argus_metadata

```text
旧 splitmux: 多路独立写 MKV -> 用 first_wall_s 计算 pts_offset -> 写 parquet
argus_metadata: 写完整视频 + sidecar -> 保存后按 SOF 对齐窗口 -> 必要时 materialize
```

这些路径保留用于旧数据兼容和调试，不是当前生产默认。

## 7. 同步级别总览

| 级别 | 精度 | 状态 | 机制 |
|------|------|------|------|
| **L0** 相机硬件触发 | <1µs（物理触发） | ✅ | PWM slave mode，AR0234C 锁定同一触发边沿 |
| **L1** episode 元数据 | — | ✅ | `meta.json.sync_reference` 记录 t0；online-sync 下 `camera_first_wall_s` 为兼容字段，不是硬件帧时间 |
| **L2** encoder 前相机同步 | µs~0.4ms 级（SOF spread） | ✅ | `argus_online_sync` 按 `sof_tsc_ns` 接受 full cluster 后再编码；manifest gate |
| **L2b** online frame bus | 与 L2 同 cluster | ✅ 可选 | recorder-owned `/dev/shm/lerobot_online_sync` 双缓冲 NV12 latest cluster，推理端只读不抢相机 |
| **L2c** replay 多视频同步 | 半帧内重同步 | ✅ | GUI timeline 暴露 per-camera file offset；前端以 master timeline time 持续校正其他 `<video>` |
| **L3a** BOX 高频独立采样 | ±3~13ms | ✅ | 500Hz poll + MCU 时间戳去重 + 逐传感器最近邻 |
| **L3b** BOX 增强对齐 | ±1~3ms（校准残差；端到端另叠最近邻量化） | ✅ | MCU↔Host 时钟线性回归 + 安全回退 |
| **L4** 硬件级全同步 | <1µs | 🔲 | BOX MCU 也由 PWM/硬件 trigger 打戳或触发（需硬件/固件支持） |

### 7.1 代码位置 & 测试映射

| 机制 | 代码 | 测试 / 验证 |
|------|------|-------------|
| PWM / trig_mode / exposure clamp | `gmsl2/gmsl2_record.py` / `gmsl2/thor_record.py` | 真机 v4l2/PWM 检查 |
| legacy splitmux first_wall / pts_offset | `gmsl2/persistent_session.py` / `gmsl2/thor_record.py` `_pts_offset_from_handle` | `tests/scripts/test_thor_record_meta.py`、persistent session tests |
| Argus metadata SOF alignment（保存后） | `gmsl2/argus_frame_sync.py` / `argus_video_materialize.py` | `tests/scripts/test_thor_argus_frame_sync.py` |
| `argus_online_sync` encoder-front recorder | `gmsl2/argus_online_sync_session.py` / `argus_online_sync_video_recorder.cpp` | `tests/scripts/test_thor_argus_metadata_session.py`、`test_thor_online_sync_burnin.py`、Thor burn-in |
| online inference frame bus | `gmsl2/online_sync_frame_client.py` / recorder `--frame-bus-dir` | `tests/scripts/test_thor_online_sync_frame_client.py` |
| preview bus / JPEG bridge | `gmsl2/online_sync_preview_bridge.py` / `PREVIEW_ON/OFF` | `tests/scripts/test_thor_argus_metadata_session.py` preview tests |
| Episode Replay video sync | `tools/data_collection_gui/gateway.py` / `frontend/src/ReplayInspector.tsx` | `tests/scripts/test_data_collection_gui_gateway.py` + browser validation |
| 500Hz poll + MCU 去重 | `box_sdk/box_client.py` `_poll_loop` | `tests/scripts/test_thor_box_client.py` |
| 帧网格 + 最近邻对齐 | `gmsl2/thor_lerobot_v3.py` `_build_episode_rows` / `_nearest_sample_data` | `tests/scripts/test_thor_ts_sync_alignment.py` |
| MCU 时钟校准 + 回退 | `gmsl2/thor_lerobot_v3.py` `calibrate_mcu_clock` / `calibrate_sensor_samples` | `tests/scripts/test_thor_ts_sync_alignment.py` |
| ffprobe/GStreamer PTS（离线数帧/QC） | `gmsl2/thor_lerobot_v3.py` `extract_pts`；`gmsl2/export_v3.py` | `tests/scripts/test_thor_lerobot_v3_pts.py` |

> 这些测试覆盖纯 Python 逻辑、stdin 协议、sidecar/manifest 合同和前端时间轴逻辑。相机硬件 SOF spread、Argus provider 稳定性、BOX↔相机端到端 tap-test 仍需要 Thor 真机验证。

## 8. 注意事项

### 8.1 legacy spawn stagger 与 online-sync persistent recorder

`spawn_stagger_s: 1.0` 主要是 legacy `gstreamer_splitmux` 路径的工程约束：11 路 `nvarguscamerasrc` 同时初始化会触发 Argus ISP / NVMM buffer 竞争（`NvBufSurfaceFromFd Failed`），导致部分相机 EOS 或空 MKV。

生产默认 `argus_online_sync` 使用长驻 recorder daemon 独占 Argus session：Connect 时打开 stream 并持续消费 idle full cluster；Start/Stop 只切 recording window。该路径的 UI 录制是 time-driven，停止时等待下一个 full cluster 收口，因此不要用 `duration_s * fps` 强行解释为唯一合法帧数。

### 8.2 MCU 时钟假设

线性回归假设 MCU 时钟在录制期间**线性且单调**。如果 MCU 时钟有跳变、回绕或非线性漂移，校准会失败（残差 > 50ms 阈值），自动回退到 L3a。

### 8.3 ffprobe 依赖

**录制同步路径不依赖 ffprobe。** `argus_online_sync` 的保存 gate 来自 recorder 内部的 SOF metadata 和 `online_sync_manifest.json`；legacy `pts_offset` 来自 worker 上报的 `first_wall_s`，也与 FFmpeg 无关。

ffprobe 仅在**离线** `export_v3.py` 数帧、QC 或调试时用到（`extract_pts`，反映容器内 PTS）。Jetson 镜像不一定带 ffprobe，所以 `extract_pts` 在 ffprobe 缺失时自动回退到 `_extract_pts_gstreamer`（用 GStreamer `matroskademux` 读 PTS）。二者都不可用才告警。

### 8.4 训练数据 vs 原始数据

LeRobot v3 parquet 中的 `observation.state` 是 **60Hz 下采样** 后的对齐结果。如果下游任务需要更高频率的传感器数据（如力控、IMU 积分），应直接读取 `box_sensors.jsonl` 原始文件，其中保留了每个传感器在原生频率下的完整时间序列。

### 8.5 wall-clock 精度

`argus_online_sync` 的相机间同步依赖 Argus SOF TSC，不依赖 `time.time()` 判断同帧；但 BOX↔相机对齐、legacy splitmux `pts_offset`、以及 episode 元数据仍依赖主机时间。建议：

- 确保 NTP 同步（`timedatectl status` 检查）；
- 避免在录制期间手动修改系统时间；
- `time.monotonic()` 用于持续时间测量不受 NTP 步进影响，但跨进程/跨设备对齐仍需 wall-clock 或硬件时间戳。

### 8.6 EE pose 与 Episode Replay 帧网格

离线 EE 轨迹生成（GUI「Generate EE Trajectory」/ `april_cube_tracking_in_robot_base.py`）由多路硬同步相机流估算 cube/EE pose，按 **per-episode 相机帧序号 N** 写 sidecar（`derived/april_cube_tracking_in_robot_base/state_action.*.csv`）。GUI replay timeline 使用：

- 有 v3 parquet 的数据集：直接用 parquet `timestamp` 列；online-sync 导出的常见口径是 `N/fps`。
- 无 v3 parquet 的 legacy camera-only 数据：`gateway._gmsl2_pts_offset_s()` 从 `meta.json.sync_reference` 估算 `pts_offset + N/fps`。

Episode Replay 不能把多个 `<video>` 元素当成天然同步：浏览器 video clocks 会 drift，且暂停态 seek 容差如果大于半帧会保留可见错位。当前前端用 master video 反推 timeline time，并持续把其他相机 seek 到同一 timeline frame；后端还返回 `cameraVideoOffsetsS` 以兼容 legacy per-camera 文件起点。

### 8.7 Online frame bus 与 preview bus

在线推理不要另开 `nvarguscamerasrc` 或 Argus session。正确路径是打开 recorder-owned frame bus：

```text
/dev/shm/lerobot_online_sync/latest_cluster.json
/dev/shm/lerobot_online_sync/slot{0,1}_cam_XX.nv12
```

该 bus 发布 recorder 接受的 latest full cluster，读端使用 latest-frame 语义，模型慢时跳过旧帧，不反压 recorder。UI 预览使用独立 `preview_frame_bus_dir` 和 `online_sync_preview_bridge.py` 转 JPEG；录制开始前 session 发送 `PREVIEW_OFF`，避免 preview 影响 recording window。

## 9. v3 数据集 schema（2026-06-15 重构）

录制器（`thor_lerobot_v3.py`）写的是 **box/state 最小 v3** parquet（数值特征 + 时间戳元数据）；相机原始文件仍并排存在每个 episode 目录里：`cam_*.mkv`、`cam_*.argus_frame_metadata.csv`、`online_sync_manifest.json`。离线 `export_v3.py` 再把相机转码并合并出带 `observation.images.*` 的训练数据集。

`export_v3` 的多传感器对齐：相机网格为权威。online-sync 数据使用 `logical_frame_index / fps`；legacy splitmux 数据使用 `pts_offset + N/fps`。box 状态按优先级挂到该网格：① 复用录制器已对齐的 session parquet，**按 `frame_index`（而非列表位置）** 配相机帧（box 网格比相机视频长的尾部丢弃、短的 carry-forward）；② 无 parquet 时回退到每集 `box_sensors.jsonl`，用 §5/§6 同一套 MCU 校准 + 逐传感器最近邻在 export 内重做 L3b。每集输出 `timestamp` 重基到 `i/fps` 以匹配重锚定的逐集视频。代码见 `export_v3._align_box_rows_by_frame_index` / `_box_rows_from_raw`。

### 9.1 `observation.state` / `action`（float32，**31** 维）

只放可训练的传感器读数，**不含任何时间戳、不含状态位**。通道分组（`BOX_STATE_NAMES`）：

| 组 | 通道 | 数量 |
|----|------|------|
| gripper | `box_gripper.distance_m` | 1 |
| trigger | `box_trigger.travel_pct` | 1 |
| IMU | `acc_{x,y,z}_g` / `gyr_{x,y,z}_deg_s` / `roll,pitch,yaw_deg` / `quat_{w,x,y,z}` | 13 |
| 六维力 | `box_six_d_force.{fx,fy,fz,mx,my,mz}` | 6 |
| 触觉 L/R | 各 `mean_f{x,y,z}_0p1N` / `max_abs_fz_0p1N` / `active_points`（239 点聚合） | 5+5 |

> 演进：42 维（含重复 `gripper.pos`、7 个 `*.timestamp`、恒定死通道 `box_status.{valid,liwp_index}`）
> → 33 维（去重 gripper + 时间戳移出, 见 §9.2）→ **31 维**（再删掉恒为常量的 `box_status.valid`(恒 1)
> 和 `liwp_index`(HF 路径恒 0)）。原则：state 只留可训练的传感器读数, 不放单调计数/常量, 避免归一化被带歪与时间泄漏。

### 9.2 `box.timestamps` 元数据列（float64，**6** 维，非训练）

每帧对齐用的原始时间戳单独成列（`BOX_TIMESTAMP_NAMES`）：**6 个传感器各一个 `*.timestamp`**
（gripper / trigger / imu / six_d_force / touch_left / touch_right）。曾经一并放进来的
`box_status.liwp_timestamp` / `received_wall_time_s` 因 **HF 录制路径恒为 0** 且与 per-sensor
mcu_ts 冗余，已移除（liwp 是包级时间戳，对齐用 per-sensor 更细更准）。

- **float64**：MCU 计数实测达 2–4e9（µs），超 float32 的 2²⁴ 整数精度；float64 在磁盘 parquet 上无损。
- **caveat**：`LeRobotDataset` 把数值特征统一出成 `torch.float32`，经 loader 读会再被量化
  （2e9 处 ULP≈256µs）。需精确 mcu_ts 时**直接读 parquet**；训练不受影响（它只是对齐元数据）。
- **uint32 回绕（暂不处理，后续优化）**：所有 box 时间戳底层是 SDK 的 uint32 µs，2³²µs ≈ **71.6 分钟**
  回绕一次。当前 episode 远短于此（~10s），**暂不处理**；极少数骑在回绕边界的 episode 会被
  L3b 校准残差阈值(50ms)自动回退到 L3a，不污染数据。**不要求 SDK 改 uint64**（结构体字段偏移
  全变, 属 ABI/协议破坏性改动, 代价高于收益）；将来若做长会话/连续录制, 在客户端 poll 循环里
  unwrap（检测 ts 回退则累加 2³²）即可彻底消除, 无需供应商配合。
- CSV：SDK `.so` 另会向 CWD 写 `box_sensor_data_*.csv`，`BoxClient.stop()` 会清理本会话的
  （见 `box_sdk/TROUBLESHOOTING.md` §7，待 SDK 加关闭开关）。

### 9.3 真机验证状态

| 日期 | 项 | 结果 |
|------|----|------|
| 2026-06-15 | 6 路 BOX 传感器频率 | gripper/imu/trigger/六维力 199Hz，touch L/R 各 50Hz |
| 2026-06-15 | MCU 校准（L3b） | slope=1µs/tick，6 路全 engage，残差 1–2ms |
| 2026-06-15 | 夹爪/扳机运动 | episode 实测 distance 0.0007–0.098m、trigger 0→100% 正确进 `state[0/1]` |
| 2026-06-15 | 触觉接触 | `active_points` 1–52、`max_abs_fz` 饱和 255、239 点原始帧完整 |
| 2026-07-07 | `argus_online_sync` 8 路 10×60s burn-in | 每路 3600 帧，sidecar 3600 行，max SOF delta 0.401ms，ffmpeg materialization=false |
| 2026-07-13 | `sync_test_lht_20260707_090407` episode 0 spot check | 8 路 MP4 均 1330 帧 / 60fps；frame 1235 附近 SOF delta 12–13µs；MP4 第 1235 帧 PTS 均 20.583s |
| 2026-07-13 | Episode Replay | 修复多 `<video>` 独立 clock/seek 容差导致的可见错位；原始视频与 sidecar 本身同步 |
| 待验 | BOX↔相机跨域 tap-test | 需要设计同时可见于相机和 BOX 触觉/力传感器的事件，量化端到端固定 skew 与 jitter |

## 10. TODO / 后续工作

| 优先级 | TODO | 说明 |
|--------|------|------|
| P0 | 在 Thor GUI 实际服务目录重启/发布 replay 修复 | 源码更新后必须重启 gateway/Vite 或重新 build 前端；浏览器需强刷，避免旧 bundle 保留 50ms seek 容差。 |
| P1 | BOX↔相机 tap-test | 用可见敲击/触觉/力事件验证 `logical_frame_index/fps` 与 BOX 校准时间之间的固定 skew 和 jitter。 |
| P1 | 把 `online_sync_manifest.json` 纳入 Dataset Processing/QC 展示 | 对用户直接显示 actual_frames、frame_count_by_camera、max SOF delta、failure reason。 |
| P2 | export_v3 确认使用 online-sync 网格来源, 将legacy来源相关代码移除,确认对整个同步采集链路无影响|
| P2 | frame bus 性能升级（仅在线推理需要） | 纯数据采集落盘无需处理。当前 tmpfs NV12 双缓冲用于实时推理/预览，若 8 路 60Hz 在线推理吞吐吃紧，再升级 CUDA/DMABUF zero-copy IPC 或共享内存 ring
  buffer。 |
| P3 | BOX uint32 µs 时间戳 unwrap | 当前短 episode 不受影响；长会话/连续录制前在客户端 poll loop 检测回绕并累加 2^32。 |
