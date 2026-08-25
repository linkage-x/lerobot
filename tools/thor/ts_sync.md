# Thor 相机-传感器时间同步技术文档

> 适用于 Thor 数据采集平台（11 × GMSL2 相机 + BOX 采集板）
> 初版 2026-05-27；2026-06-09 按实际代码实现校订（PTS offset 机制、meta 字段、ffprobe 角色）
> 2026-06-15 按真机实测校订（各传感器频率、L3b 校准残差；MCU 时钟 = 1µs/tick，6 路全 engage）
> 2026-06-16 schema 精简（observation.state 31 维 / box.timestamps 6 维）+ 去 meta 冗余（`sync_reference` 删 split_now_wall_s、camera_first_pts_s）
> 2026-07-13 按最近 8 次同步相关改动校订：生产默认相机路径切到 `argus_online_sync`，SOF full-cluster 在 encoder 前对齐；新增 online frame bus / preview bus / replay 多视频同步说明。
> 2026-07-29 实施 IMU 姿态去冗余（§9.1.1 / 原 §10 P2）：删 `rpy`、quat 改 xyzw，`observation.state` 31 → 28 维；四元数半球用 213 条真机 IMU 流实测后决定**不强制**。同日核对后关闭原 §10 P0（replay 修复的 Thor 部署）。
> 2026-08-25 新增 §5.5（BOX↔相机残余偏移首次实测 +4.4 ms，gyro↔vision 互相关；同时更正当前固件的传感器实际速率为 520/244/120/60 Hz）；§5.2 的 MCU→host 回归改为去均值形式（原写法在真实量级上有 0.107 ms RMS 的数值误差）

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
| Touch L | **50 Hz**（Paxini）/ **60 Hz**（M2020） | ~500 / ~600 |
| Touch R | 50 Hz（Paxini）/ 60 Hz（M2020） | ~500 / ~600 |

> 频率为 2026-06-15 真机实测（Paxini 左右触觉垫**均为 50Hz**——早期文档误记 Touch L 为 200Hz）；
> 2026-08-17 换 M2020 后实测左右均为 **60Hz**（link stats `device_hz`/`measured_hz` 与 `sensor_status.observed_hz` 一致），见 §9.3。
> 这影响最近邻对齐误差：touch 在 50Hz 下为 ±10ms、60Hz 下为 ±8.3ms（而非 200Hz 的 ±2.5ms）。

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

> **2026-08-25：这个拟合改成了去均值形式。** 原先用教科书的 `n·Sxy − Sx·Sy` 正规方程，
> 在真实量级上（`mcu_ts` ~4e9 µs 对墙钟 ~1.8e9 s）分子要拿两个 ~2.9e25 相减去得到 ~1e14，
> 十一位有效数字没了。用**精确有理数拟合**对 72 个真实 sensor-episode 逐条比对，旧写法的代价是
> **RMS 0.107 ms / p95 0.291 ms / 峰值 0.568 ms**（slope 差 ±1~94 ppm）。
> 形状比幅值重要：`intercept` 无论如何都强制拟合线过质心，所以误差是**绕 episode 中点的转动**
> ——整段均值 <0.3 µs、两端最大、符号逐 episode 随机。**因此它进不了任何偏置量**（§5.5 的 Δt
> 改前改后只动 ≤0.04 ms），只进逐帧 timing，且最差处落在 episode 头尾。
> 去均值后同样 72 条 fit 降到 **0.3 µs**——那是 float64 装 1.8e9 秒墙钟的**表示极限**，不是算法残留。
> 收益很小（折进逐传感器 timing σ 最多让 six_d_force 0.63 → 0.69 ms，≤0.02 mm）；改它的理由是
> 这项误差**逐 episode 零均值、结构性不可见**。回归测试须用真实量级——本仓库原有 18 条 ts_sync
> 测试全跑在 `T0=1000.0` 的玩具量级上，这正是它藏了几个月的原因。

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
| 相机曝光时刻对齐（2026-07-16） | `N/fps` 与硬件 SOF 之间的 per-episode 固定 skew | −11~−53ms（未量化,逐 episode 变） | ~0（BOX NN 改用 `sensor_timestamp_ns − t0_mono`） |
| BOX↔相机端到端 | BOX 定时残差 + 最近邻量化 | ±3~13ms | 200Hz 传感器约 ±1-3ms 残差并另叠 ±2.5ms；touch 50Hz 另叠 ±10ms |

> **相机帧 skew（2026-07-16 真机量化 + 修复）**：`N/fps` 网格默认假设相机 frame 0 == `t0` 且恰好 60.000Hz。
> 但 frame 0 的**选择由软件接收/gate 时序决定,硬件采集发生在更早时刻**：相机持续采集,recorder 收到 START
> 时管线内已有在途/缓存的 cluster,被留作 frame 0 的那帧其 SOF 早于 `t0` 被锁存(故实测 δ 为负),且缓存深度/启动相位
> 逐次不同 → frame 0 相对 `t0` 呈**逐 episode 变化的固定相位偏移**。7 条 `water_pouring_20260715_*` episode 实测该 skew
> **δ = −11 ~ −53ms,逐 episode 变化**（同 session 内也跳）,几乎零漂移（真实帧率 60.000±0.002fps,`N/fps`
> 漂移 ≤0.07ms/10s,可忽略）。它比 MCU 校准残差大一个量级,给 gripper 注入最高 9mm、给 6D 力最高 4.2N 的
> state 误差,峰值落在抓取/接触瞬间。因逐 episode 变,**不能用全局常数 offset 修**;最稳健是直接用逐帧 `sensor_timestamp_ns`（还吸收 ~1ms 帧内抖动）,sidecar 不完整时退化为每 episode 线性拟合(offset+slope)外推,而非全局固定 offset。
> 修复:BOX 最近邻查找目标改用硬件 SOF 采集时刻 `sensor_timestamp_ns/1e9 − t0_mono_s`（`camera_frame_times_rel`）,
> parquet `timestamp` 列仍 `N/fps`。详见 [`experiments/ts_sync_skew_20260716/`](experiments/ts_sync_skew_20260716/README.md)。
> **注意**：这只修了训练侧;闭环真正要求「训练对齐 == 部署对齐」,部署侧（frame bus + 实时 BOX）残余 skew 待测（§10 P1）。

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

### 5.5 BOX↔相机残余偏移：实测 +4.4 ms（2026-08-25）

§5.4 那张表把「BOX↔相机端到端」写成 ±1~3ms 残差 + ±2.5ms 量化，但**没有给出偏移的
均值**——因为回归的 `intercept` 把设备→主机的平均传输延迟整个吸收掉了，它在 BOX 内部
各传感器之间相消，只在 BOX↔相机之间留下一个固定 skew（§5.4 末尾的口径提醒已经点出这
件事，但当时是定性的）。

现在它是定量的。做法是拿**旋转**当共同观测量：marker rig 与 IMU 刚性连接，两者的体坐标
系角速度是同一个矢量在两个系里的表示，于是可以在不知道任何平移标定的前提下解出时间偏移。
工具是 `third_party/opencv_kalibr/metrology/cli/estimate_camera_imu_time_offset.py`。

**符号约定**：`Δt > 0` 表示 **BOX 时间戳偏晚**，即 `ω_gyro(t_cam + Δt) ≈ R_rig_imu · ω_rig(t_cam)`；
修正是 `t_box -= Δt`，等价于「在 `t_cam + Δt` 处查 BOX 样本」。

| session | BOX | 采纳 episode | Δt | 逐 episode σ |
|---|---|---|---|---|
| `..._9ch_v1_20260817_162847` | `box1672693301` | 3 / 3 | **+4.39 ms** | 0.20 ms |
| `..._9ch_v1_20260817_162847` | `box1819152274` | 3 / 3 | **−1.18 ms** | 2.3 ms |
| `..._10ch_v1_20260821_173941` | `box1672693301` | 13 / 13 | **+4.71 ms** | 1.16 ms |

同一个 BOX 在相隔四天的两个 session 上给出 +4.4 / +4.7 ms——**这是可复现的常量**，
不是 §5.4 里那个逐 episode 乱跳的 `N/fps` skew。**左右两个 BOX 的 Δt 不同**（+4.4 vs −1.2 ms），
所以不能共用一个常数：它主要由各自的 UDP 传输延迟决定。

**这个数目前没有被任何代码消费**——录制器不减它，已录数据都带着它。

> **不要把它读成「标完就到亚毫秒了」。** 标定去掉的是**偏置**，去不掉的是逐帧
> **最近邻量化**，而后者按传感器不同、且更大：`间隔/√12` = six_d_force 0.6 ms、
> imu 1.2 ms、gripper/trigger 2.4 ms、**touch 4.8 ms**。也就是说端到端的主导项是
> **查表方式**而不是时钟，touch 要做到亚毫秒只能把最近邻换成插值。

## 6. 完整对齐流程（per episode）

> **同步时机（避免误会）**：注意相机间同步与 box↔相机同步是**两个不同时机**。
> **相机之间（L0/L2）在录制期在线完成**——PWM 硬触发全程锁快门，`argus_online_sync` 在 encoder 前逐帧按
> `sof_tsc_ns` full-cluster gate 放行同一 SOF 簇（§3.2），Stop 只收口写 `online_sync_manifest.json`，**不是 Stop 才同步**。
> 而 **box↔相机的 `observation.state` / `box.timestamps` 同步在录制 Stop 时**由录制器完成
> （`thor_record.py` 在 save 分支算 `camera_frame_times_rel` → `Lr3Writer.append_episode` → `thor_lerobot_v3._build_episode_rows`
> 做 MCU 校准 + 逐传感器最近邻）。**单条 session 的 `data/.../file-000.parquet` 落盘即已同步，不需要点 Export。**
> `export_v3` 是**离线合并/打包**步骤：复用录制器已同步好的 state（按 `frame_index` 配相机帧）、额外做**触觉全阵列**对齐 +
> 视频转码 + 多 session 合并，**不重做** box↔相机的 state 同步（§9）。判断某集是否已同步看 `meta.json.box_camera_alignment`
> （录制时写入）。

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
       ├─ parquet timestamp 网格 = logical_frame_index / fps（loader 共享网格,不变）
       ├─ BOX 最近邻查找目标 = 每帧硬件 SOF 采集时刻 sensor_timestamp_ns/1e9 − t0_mono_s
       │    （camera_frame_times_rel;消除 N/fps 与硬件 SOF 之间的 per-episode 固定 skew,见 §5.4）
       │    sidecar 空洞/短尾按 SOF 线性拟合外推（单一时间基准,不与 N/fps 拼接,外推帧数会 warn）
       ├─ 对每帧逐传感器二分查找最近邻 → 组成 state 向量
       └─ meta.json.box_camera_alignment 记 mode / mean_skew_ms / skew_jitter_ms / frames_with_sof（可审计）
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
| **L3b+** 相机曝光时刻对齐 | 消掉 per-episode −11~−53ms 固定 skew | ✅ | BOX NN 目标改用 `sensor_timestamp_ns − t0_mono`（`camera_frame_times_rel`,2026-07-16） |
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
| state schema（28 维；IMU 姿态只留 quat xyzw） | `gmsl2/thor_lerobot_v3.py` `BOX_STATE_NAMES` / `box_snapshot_to_state` / `_imu_quat_xyzw` | `tests/scripts/test_thor_ts_sync_alignment.py::test_imu_attitude_is_quaternion_only_in_xyzw_order`；`experiments/imu_quat_hemisphere_20260729/` |
| MCU 时钟校准 + 回退 | `gmsl2/thor_lerobot_v3.py` `calibrate_mcu_clock` / `calibrate_sensor_samples` | `tests/scripts/test_thor_ts_sync_alignment.py` |
| 相机曝光时刻对齐（skew 修复） | `gmsl2/thor_lerobot_v3.py` `camera_frame_times_rel` / `_build_episode_rows(frame_times_s=)`；`thor_record.py` 传参 | `tests/scripts/test_thor_ts_sync_alignment.py` §7–§8；`experiments/ts_sync_skew_20260716/` |
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

`export_v3` **不重做** box↔相机的 state 同步（那一步已在录制 Stop 时完成，见 §6 开头「同步时机」）；它只把录制器已同步好的 box state 挂到权威相机网格上。相机网格：online-sync 数据使用 `logical_frame_index / fps`，legacy splitmux 数据使用 `pts_offset + N/fps`。box state 复用录制器已对齐的 session parquet，**按 `frame_index`（而非列表位置）** 配相机帧（box 网格比相机视频长的尾部丢弃、短的 carry-forward）；**无可用 session parquet 时直接跳过该 episode**（不再从 `box_sensors.jsonl` 在 export 内重做 L3b —— 该 raw 重对齐路径已于 2026-07-13 移除，见下方更新记录）。export 另外从 `box_sensors.jsonl` 重采样**触觉全阵列**（taxel 维数按贴片型号，见 §9.3）到帧网格，这是 export 才做的对齐（`_align_touch_rows`）。每集输出 `timestamp` 重基到 `i/fps` 以匹配重锚定的逐集视频。代码见 `export_v3._align_box_rows_by_frame_index`（state）/ `_align_touch_rows`（触觉阵列）。

### 9.1 `observation.state` / `action`（float32，**28** 维）

只放可训练的传感器读数，**不含任何时间戳、不含状态位**。通道分组（`BOX_STATE_NAMES`）：

| 组 | 通道 | 数量 |
|----|------|------|
| gripper | `box_gripper.distance_m` | 1 |
| trigger | `box_trigger.travel_pct` | 1 |
| IMU | `acc_{x,y,z}_g` / `gyr_{x,y,z}_deg_s` / `quat_{x,y,z,w}` | 10（姿态只留四元数，**xyzw 标量在后**，见 §9.1.1） |
| 六维力 | `box_six_d_force.{fx,fy,fz,mx,my,mz}` | 6 |
| 触觉 L/R | 各 `mean_f{x,y,z}_0p1N` / `max_abs_fz_0p1N` / `active_points`（**按贴片实际触点数**聚合，见 §9.3） | 5+5 |

> 演进：42 维（含重复 `gripper.pos`、7 个 `*.timestamp`、恒定死通道 `box_status.{valid,liwp_index}`）
> → 33 维（去重 gripper + 时间戳移出, 见 §9.2）→ 31 维（再删掉恒为常量的 `box_status.valid`(恒 1)
> 和 `liwp_index`(HF 路径恒 0)）→ **28 维**（删 IMU 的 `roll/pitch/yaw_deg`，与 `quat` 冗余，见 §9.1.1）。
> 原则：state 只留可训练的传感器读数, 不放单调计数/常量/同一量的第二种编码, 避免归一化被带歪与时间泄漏。

#### 9.1.1 IMU 姿态去冗余（31 → 28 维，`rpy` 删除 + quat 改 xyzw）✅ 2026-07-29

**结论（已实施）**：删掉 `box_imu.{roll,pitch,yaw}_deg` 三维，只保留四元数，并把顺序从原来的
`w,x,y,z` 改为 **`x,y,z,w`**。`observation.state` / `action` 从 **31 → 28** 维。

**为什么冗余**：`roll/pitch/yaw` 与 `quat` 编码同一个 3-DoF 姿态，且在 `box_client._decode_imu`
里来自 SDK 同一个 `imu` 结构体的两个字段（`imu.roll/pitch/yaw` 与 `imu.quat`），**不是两路独立测量**。
改动前 IMU 那 13 维里只有 9 维是独立信息（acc 3 + gyr 3 + 姿态 3）；改后留 10 维（多出的 1 维是四元数
自带的模长约束，无害）。

**为什么删 rpy 而不是删 quat**（冗余的那份恰好是更差的那份）：

1. **`yaw_deg` 在 ±180° 处 wrap**。归一化统计（mean/std 或分位数）和回归损失落在一个会绕回的通道上
   是**有害的**，不只是浪费维度：同一物理姿态的两个邻近时刻可能相差 360，梯度直接爆。
2. **姿态在 state 里被隐式加权两次**，归一化后两组通道各自贡献一份姿态梯度。
3. 四元数虽有双覆盖（`q` 与 `-q` 同一旋转），但那是**符号**问题，比角度 wrap 好处理；且实测这份 SDK
   流根本不出现符号翻转（见下方半球约定）。

**为什么 xyzw**：与 `scipy.spatial.transform.Rotation`、ROS `geometry_msgs/Quaternion`、MuJoCo 之外的
多数下游一致（MuJoCo 用 wxyz，是少数派）。本仓库 FR3 侧已经在 `xyzquat` / `wxyz_to_xyzw()` 之间反复
转换（见 `docs/fr3_act_infer_real_minimal.md`、`docs/fr3_quest3_teleop_todo.md`），统一到 xyzw 可以
消掉一层「标量位在前还是在后」的心智负担。SDK 侧仍是 wxyz（`imu.quat` → `quat_wxyz`），**重排放在
打包 state 时做，不改 SDK、不改 raw JSONL**。

**数据不丢**：`box_sensors.jsonl` 存的是 `_decode_imu` 的完整 dict，`roll_deg/pitch_deg/yaw_deg`
仍然原样归档，需要时可离线还原或重新导出。删的只是**训练用 state 布局**。

**四元数半球约定：不强制半球，按 SDK 原样透传**（2026-07-29 真机实测后定案）。原计划是「若存在
`q → -q` 翻转就强制 `w >= 0`」，实测结论把它推翻了：

| 口径 | 实测（Thor `outputs/datasets` 全量，213 条 IMU 流 / 323,213 样本） |
|------|------|
| 相邻样本反极翻转 `dot(q[i], q[i+1]) < 0` | **0 次** —— SDK 输出本身就是连续的 |
| 相邻样本最大 L2 步长（原样） | 0.342（真实快速转动，远离 2.0） |
| `w` 变号（姿态自然穿过 `w=0`，即转过 180°）的流 | **31 条**（14.6%） |
| 若强制 `w >= 0`，这 31 条的最大 L2 步长 | **2.001** —— 整体符号跳变被人为注入 |
| 强制半球能改善的流 | **0 条** |

即：**强制半球在这份数据上只有害处**——它把「`w` 过零」变成一个假的整体符号跳变，而它本要防的翻转
根本不存在。因此打包 state 时只做 wxyz→xyzw 重排，**不改符号**。注意这条结论依赖 SDK 侧的连续性，
若将来换固件/换 IMU 解算，应重跑 [`experiments/imu_quat_hemisphere_20260729/`](experiments/imu_quat_hemisphere_20260729/README.md)
复核。另：同一批数据里 `yaw_deg` 的 ±180° wrap 实测确有发生（1 条流 2 次），佐证删 `rpy` 的判断。

**改动点**（全部在录制/导出侧，SDK 不动）：

| 文件 | 改动 |
|------|------|
| `gmsl2/thor_lerobot_v3.py` `BOX_STATE_NAMES` | 删 `box_imu.{roll,pitch,yaw}_deg` 三项；`quat_{w,x,y,z}` 改为 `quat_{x,y,z,w}` |
| `gmsl2/thor_lerobot_v3.py` `box_snapshot_to_state` | 删三行 `_finite_float(imu.get("roll_deg"))` 等；新增 `_imu_quat_xyzw()`，把 SDK 的 `quat_wxyz` 重排为 xyzw（符号透传） |
| `gmsl2/export_v3.py` | 只读 `lr3.BOX_STATE_NAMES`，随定义自动跟随；**新增**：session parquet 宽度与当前 schema 不一致时 `_emit` 警告（旧 31 维数据重导出会丢列名，不再静默） |
| `tests/scripts/test_thor_ts_sync_alignment.py` | 新增 `test_imu_attitude_is_quaternion_only_in_xyzw_order`：rpy 不在 state、quat 顺序为 xyzw、打包确实按 `[1,2,3,0]` 重排且不改符号 |
| `data_collection_gui/frontend/src/SeriesPlot.test.ts` | 新增当前 xyzw 布局分组用例；保留 legacy wxyz+rpy 用例（旧数据集 replay 仍要能画）。分组逻辑本身按名字驱动、`dim` 取自列表位置，无固定下标，源码无需改 |

**迁移影响（破坏性）**：`observation.state` / `action` 宽度和列序都变，**已有数据集的 norm stats 与
checkpoint 不兼容**。已录数据要么保持在旧 schema 下训练，要么重新录制/重跑录制器（raw JSONL 齐全，可重建）。
注意 `export_v3` **不会**把旧的 31 维 session parquet 转成 28 维——它复用录制器已同步的 state，只会
按上表警告并丢掉列名。

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

### 9.3 触觉贴片型号与触点数（2026-08-17）

BOX SDK 对**所有**触觉贴片都用同一个 239 槽定长数组 `TouchSensor.forces` 承载 —— 这是旧 Paxini L5325 的触点数，**不是**当前实际装的贴片的属性。装 M2020（3×3）时，`libbox_controller.so` 解析 `TLV_TYPE_GET_TOUCH_M2020_{LEFT,RIGHT}_DATA` 后调 `fill_touch_from_m2020()`，把 9 个真实触点写进槽 0–8，槽 9–238 恒零。**所以数组长度不能用来判断贴片型号**（没被按压的 Paxini 贴片同样全零）。

判据是链路流 id（`LinkStats.tlv_type`）：

| 链路流 | 贴片 | 触点数 | 布局 |
|--------|------|--------|------|
| `0x0002` / `0x0003` | Paxini L5325 | 239 | 厂商 XYZ 坐标表 |
| `0x0008` / `0x0009` | M2020 | 9 | 3×3，线缆在左，行优先编号 1–9 |

- `box_client.touch_model_from_link_stats()` 据此识别贴片，`_decode_touch()` 把每帧裁到真实触点数并打上 `model` / `points` 标签；识别前**不截断**（按完整 239 槽透传 + 告警），不做零尾启发式猜测。
- 型号一经识别就**锁存**：贴片不可能录制中途换，中途改宽度会在同一 episode 里写出参差不齐的 fx/fy/fz。可用 `box_collection.touch_model` 在配置里钉死。
- `_touch_summary()` 的 `mean_f{x,y,z}` 按**实际触点数**求平均。曾硬编码除以 239：M2020 下同样的物理接触，均值被稀释 239/9 ≈ 26.6 倍（实测 132514 数据集：`mean_fz` 0.435 → 11.556）。
- `export_v3` 每次导出用 `_detect_touch_width()` 从 raw jsonl 定宽（parquet 是定长 list 列）；一次导出里各 episode 宽度不一致 = 采集中途换过贴片，会告警并取最宽。
- 前端 `touchVisualization` 按 `model` 选布局（M2020 画 3×3 带触点编号，Paxini 走 239 点 XYZ 图），Calibration / Device Manager / Episode Replay 三个页面共用。

> **fz 饱和**：fz 是 uint8（0.1N/LSB），上限 25.5N。M2020 实测正中触点（index 5，数组下标 4）在抓取时长期顶到 255 —— 该通道在接触重时是**截断**的，不要当线性力用。

### 9.4 真机验证状态

| 日期 | 项 | 结果 |
|------|----|------|
| 2026-06-15 | 6 路 BOX 传感器频率 | gripper/imu/trigger/六维力 199Hz，touch L/R 各 50Hz |
| 2026-06-15 | MCU 校准（L3b） | slope=1µs/tick，6 路全 engage，残差 1–2ms |
| 2026-06-15 | 夹爪/扳机运动 | episode 实测 distance 0.0007–0.098m、trigger 0→100% 正确进 `state[0/1]` |
| 2026-06-15 | 触觉接触（Paxini L5325） | `active_points` 1–52、`max_abs_fz` 饱和 255、239 点原始帧完整 |
| 2026-08-17 | 触觉贴片换 M2020 | 链路流实测 tlv 8/9 @60Hz（无 2/3）→ SDK 已支持；数据仍是 239 槽因 `fill_touch_from_m2020` 只填槽 0–8（全集扫描 594–597 帧×4 pad，槽 9–238 恒零）。见 §9.3 |
| 2026-07-07 | `argus_online_sync` 8 路 10×60s burn-in | 每路 3600 帧，sidecar 3600 行，max SOF delta 0.401ms，ffmpeg materialization=false |
| 2026-07-13 | `sync_test_lht_20260707_090407` episode 0 spot check | 8 路 MP4 均 1330 帧 / 60fps；frame 1235 附近 SOF delta 12–13µs；MP4 第 1235 帧 PTS 均 20.583s |
| 2026-07-13 | Episode Replay | 修复多 `<video>` 独立 clock/seek 容差导致的可见错位；原始视频与 sidecar 本身同步 |
| 2026-07-29 | replay 修复的 Thor 部署核对 | gateway 源码含 `cameraVideoOffsetsS` 且进程在其后重启；Thor 不 serve 前端 bundle（前端在开发机跑 Vite）→ 原 §10 P0 关闭 |
| 2026-07-16 | BOX↔相机固定 skew 量化（7 ep, `water_pouring_20260715_*`） | `N/fps` vs 硬件 SOF δ=−11~−53ms 逐 episode 变；真实帧率 60.000±0.002fps、`N/fps` 漂移可忽略；注入 gripper≤9mm/力≤4.2N。详见 `experiments/ts_sync_skew_20260716/` |
| 2026-07-16 | 修复:BOX NN 改用 sensor_timestamp（L3b+） | `camera_frame_times_rel`+`_build_episode_rows(frame_times_s=)`；真机数据验证 timestamp 列不变、gripper 修正 max 8.96mm；回归测试 15 passed |
| 2026-07-29 | IMU 四元数半球（213 条 IMU 流 / 323,213 样本） | 反极翻转 **0 次**（SDK 流本身连续）；31 条流自然穿过 `w=0`,强制半球反而注入 L2=2.0 符号跳变 → **决定不强制**。详见 `experiments/imu_quat_hemisphere_20260729/` |
| 2026-08-25 | BOX↔相机残余偏移（gyro↔vision 互相关） | `box1672693301` = **+4.39 / +4.71 ms**（0817/0821 两个 session，逐 episode σ 0.20/1.16 ms），`box1819152274` = **−1.18 ms**；同时解出 `R_rig_imu` ≈ 90° yaw（两 BOX 各 89.5°/88.2°，跨 session 复现 ~1°）。见 §5.5 |
| 2026-08-25 | 传感器实际速率（当前固件） | six_d_force **520 Hz** · imu **244 Hz** · gripper/trigger **120 Hz** · touch **60 Hz**——与 2026-06-15 记的 199/50 Hz 已不同，最近邻量化项须按实测速率算 |
| 2026-08-25 | `sensor_timestamp_ns` 内核打戳抖动 | 逐相机 p50 0.04–0.07 / p95 0.15–0.22 ms（尾部 2.6 ms），0817/0821/0824 三批、每台相机一致；而硬件 `sof_tsc_ns` 跨相机只差 **8 µs**。`camera_frame_times_rel` 用的是前者，450 mm/s 下 p95 折 0.09 mm，属记录项 |
| 2026-08-25 | MCU→host 回归的数值条件性（精确有理数对照，72 sensor-episode） | 旧的原始平方和写法：RMS **0.107** / p95 **0.291** / 峰值 **0.568 ms**，形态为绕 episode 中点的转动（整段均值 <0.3 µs）；改去均值后降到 **0.3 µs**。Δt 不受影响（4.39 → 4.35 ms）。见 §5.2 |
| 待验 | 部署侧 skew（frame bus + 实时 BOX） | 需先上推理部署,再量化在线路径残余 skew 并与训练侧对齐（§10 P1） |

## 10. TODO / 后续工作

2026-07-29 已关闭（原 P0，replay 修复发布）：Thor 上核对确认无待办——`gateway.py` 已含 `cameraVideoOffsetsS`（源码 11:04 同步），gateway 进程 13:10 在其之后重启；Thor 只监听 8765，gateway 只 serve STL/JPEG/PNG，**不 serve 前端 bundle**，也无 Vite/node 进程，前端是在开发机本地跑 Vite（`VITE_GUI_API_BASE` 指向 Thor），源码已含四分之一帧 seek 容差，重启本地 dev server + 浏览器强刷即可。**遗留（非阻塞）**：Thor 上 `tools/data_collection_gui/frontend/dist/` 是 2026-05-27 的陈旧 bundle，当前无人 serve，但将来若配静态服务会踩回 50ms 容差那版——届时先重 build 或删除。

2026-07-29 已完成（原 P2）：IMU 姿态去冗余——删 `box_imu.{roll,pitch,yaw}_deg`、quat 改 xyzw，`observation.state` / `action` 31 → 28 维；四元数半球经真机实测决定不强制。详见 §9.1.1 与 `experiments/imu_quat_hemisphere_20260729/`。**破坏性**：state 宽度与列序都变，旧数据集的 norm stats / checkpoint 不兼容。

2026-07-13 已完成：Dataset Processing/QC 展示 `online_sync_manifest.json` 的 actual_frames、frame_count_by_camera、max SOF delta 和 failure reason；`export_v3` 改为要求 online-sync manifest，以 `actual_frames` 作为相机网格来源，并移除 legacy `pts_offset` / raw `box_sensors.jsonl` 重对齐路径。

| 优先级 | TODO | 说明 |
|--------|------|------|
| P1 | 部署侧 skew 实测 + 训练/部署对齐 | 训练侧固定 skew 已于 2026-07-16 用 `sensor_timestamp` 量化并修复（δ=−11~−53ms,见 `experiments/ts_sync_skew_20260716/`）。**剩余**:闭环真正要求「训练对齐 == 部署对齐」,而部署走 online frame bus + 实时 BOX 的另一条路径,其残余 skew 尚未实测。**理由:目前还没有做推理部署**,故挂 TODO;届时把在线路径也锚到硬件 SOF 采集时刻基准并量化。原 tap-test 已非必需（skew 可纯数据量化）,仅在需要绝对地锚定 BOX↔相机延迟时再做——**该绝对锚定已于 2026-08-25 用 gyro↔vision 互相关完成（§5.5），无需 tap-test**；同一工具可直接用于量化在线路径。 |
| P2 | frame bus 性能升级（仅在线推理需要） | 纯数据采集落盘无需处理。当前 tmpfs NV12 双缓冲用于实时推理/预览；若 8 路 60Hz 在线推理吞吐吃紧，再升级 CUDA/DMABUF zero-copy IPC 或共享内存 ring buffer。 |
| P3 | BOX uint32 µs 时间戳 unwrap | 当前短 episode 不受影响；长会话/连续录制前在客户端 poll loop 检测回绕并累加 2^32。 |
