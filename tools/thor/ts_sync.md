# Thor 相机-传感器时间同步技术文档

> 适用于 Thor 数据采集平台（11 × GMSL2 相机 + BOX 采集板）
> 初版 2026-05-27；2026-06-09 按实际代码实现校订（PTS offset 机制、meta 字段、ffprobe 角色）
> 2026-06-15 按真机实测校订（各传感器频率、L3b 校准残差；MCU 时钟 = 1µs/tick，6 路全 engage）
> 2026-06-16 schema 精简（observation.state 31 维 / box.timestamps 6 维）+ 去 meta 冗余（`sync_reference` 删 split_now_wall_s、camera_first_pts_s）

## 1. 系统总览

Thor 采集系统包含两套独立的数据源：

| 数据源 | 硬件 | 传输方式 | 帧率 |
|--------|------|----------|------|
| 11 路 GMSL2 相机 | SG16A + AR0234C 传感器 | nvarguscamerasrc → H.265 MKV | 60 fps |
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
│                     │ nvarguscamerasrc ×11     │     │
│                     │ do-timestamp=true        │     │
│                     │ → H.265 → MKV per cam   │     │
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

**一句话**：以**主机墙钟**为唯一公共时间轴，把硬件同步的相机与独立晶振时钟的 BOX 都换算到这条轴上，再按 60Hz 帧网格做最近邻对齐。

1. **公共原点**：每条 episode 开录记一个 `t0_wall_s`（split-now 的主机墙钟时刻），相机与 BOX 共用它为零点。
2. **相机**（详 §3 / §5.1）：11 路共用一路 PWM 硬触发 → 帧间 <1µs；管道启动延迟用 `pts_offset = mean(camera_first_wall_s − t0_wall_s)` 修正，**帧 N 时间 = pts_offset + N/fps**（t0 相对域）。
3. **BOX**（详 §4 / §5.2）：500Hz 轮询，按各传感器 MCU 时间戳变化去重（原生 199/50Hz 独立记录）；再对每个传感器做 `host = slope·mcu + intercept` 最小二乘回归（实测 slope = 1µs/tick，残差 1–2ms）消除轮询抖动，得 `t_rel_s = 校准时间 − t0_wall_s`。
4. **合并**（详 §6）：对每个 60Hz 相机帧时间，在每个传感器序列里二分查找最近样本 → 拼成 `observation.state`（§9.1）；对齐所用的原始 MCU 戳单独存入 `box.timestamps`（§9.2）。

**精度**：相机间 <1µs；BOX↔相机 ≈ ±1–3ms（L3b 校准后，详 §7）。

**信息载体**：`meta.json` 的 `sync_reference`（`t0_wall_s` / `camera_first_wall_s` 跨相机锚点）+ 训练 parquet（`timestamp` 帧网格 / `box.timestamps` 对齐戳）+ `box_sensors.jsonl`（原始全速率，可重算校准）。三者经审计：完备、无冗余。

## 2. 三个时钟域

| 时钟域 | 来源 | 精度 | 特点 |
|--------|------|------|------|
| **PWM 硬件时钟** | Jetson pwmchip, 60Hz 方波 | 亚微秒级 | 仅产生触发边沿，不输出时间戳 |
| **主机 wall-clock** | Linux `time.time()` / `CLOCK_REALTIME` | 微秒级（NTP 校准后） | 所有软件层的公共参考 |
| **BOX MCU 时钟** | 采集板内部晶振 | 未知精度（典型 ±50ppm） | 仅在 MCU 侧单调递增，与主机无校准关系 |

PWM 时钟只负责触发相机快门，不产生可读取的时间戳。相机和传感器的时间对齐完全依赖主机 wall-clock 作为桥梁。

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
- **spawn stagger**：11 路 nvarguscamerasrc 同时初始化会触发 Argus ISP 的 NVMM buffer 分配竞争（`NvBufSurfaceFromFd Failed`），需错开 1.0s 逐路启动。这不影响帧对齐（PWM 触发与进程启动时间无关），只影响各路的**起始帧偏移**

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

对齐基于主机 wall-clock：

```
相机帧 N 的时间 = t_start + N / 60         （t_start = 相机全部启动后）
BOX 样本时间   = time.time() at poll       （主机收到 SDK 返回的时刻）
```

对每个相机帧时间点 t，在每个传感器的时间序列中用**二分查找**找到 wall-clock 最接近的样本，组成该帧的 state 向量。

### 理论精度

对齐误差 = 传感器采样间隔/2 + 主机侧抖动

- 传感器采样间隔/2：200Hz → ±2.5ms，50Hz → ±10ms
- 主机侧抖动：UDP 传输延迟 + Python poll 调度抖动（典型 ~1-3ms）
- **合计：±3~13ms**

## 5. L3b：增强对齐（PTS 提取 + MCU 时钟校准）

L3a 的两个主要误差源：

1. **相机侧**：帧时间用 `t_start + N/fps` 推算，但 `t_start` 是 gst-launch 进程启动时刻，不是第一帧 PWM 边沿时刻。管道启动延迟（ISP 初始化、编码器预热）典型 100-500ms，不计入则首帧时间偏移数百毫秒
2. **BOX 侧**：每个样本的 `wall_time_s` 是主机**收到**时刻，包含 UDP 传输延迟和 poll 调度抖动（每次不同）

### 5.1 相机侧修正：首帧 host wall-time 偏移（pts_offset）

> **实现说明（2026-06 校订）**：早期设计用 `ffprobe` 提取参考相机 MKV 的容器内首帧 PTS
> 作为管道启动延迟。Thor 真机 burn-in 发现 `splitmuxsink` 的 `format-location-full`
> 回调里 `first_sample.pts` **跨流不可用**——每路相机有各自的 pipeline clock，即使物理
> 上只差 ~20ms 起始，`first_pts_s` 在 11 路之间能差出 10s 量级，不能作为跨流锚点。
>
> 真正的跨流公共时钟是 **host wall-time**。现在每个 worker 在 `splitmuxsink` 首帧真正
> 落盘时记录 `first_wall_s = time.time()`（`persistent_session_worker.py`），父进程把它聚进
> `FragmentInfo`。Episode 结束后由 `thor_record._pts_offset_from_handle()` 计算：

```
pts_offset = mean_over_cams( first_wall_s[cam] - t0_wall_s )
```

其中 `t0_wall_s` 是 StartEpisode 发出 `split-now` 的时刻（相机/BOX 共用的录制起点）。
`first_wall_s - t0_wall_s` 衡量的就是「split-now 命令 → 该路首帧实际落盘」的延迟，
即管道启动延迟，只是用主机墙钟而非容器 PTS 测量。逐路 delta 的完整明细写入
`meta.json` 的 `sync_reference.camera_first_wall_s`。

由于所有相机共享 PWM 触发，帧间间隔严格 = 1/fps，单个标量 `pts_offset` 即可重建帧时间网格。
对齐在 **t0 相对域** 内进行（BOX 样本时间也是 `t_rel_s = wall_s - t0_wall_s`）：

```
frame_time[N] = pts_offset + N / fps        （相对 t0_wall_s）
                ^^^^^^^^^^
            管道启动延迟修正（frame_origin_s）
```

代码见 `thor_lerobot_v3._build_episode_rows()`：`frame_origin_s = pts_offset_s`。

> **注**：`thor_lerobot_v3.extract_pts()`（ffprobe + GStreamer 双路）仍然存在，但**只在
> 离线 `export_v3.py` 数帧时使用**，不在录制对齐路径里。录制路径的 `pts_offset` 完全来自
> worker 上报的 `first_wall_s`，与 ffprobe 无关。

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
#         残差标准差典型 <0.5ms
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

| 修正项 | 消除的误差 | 修正前 | 修正后 |
|--------|-----------|--------|--------|
| 首帧 wall-time 偏移（pts_offset） | 管道启动延迟（100-500ms 偏移） | 全局偏移 | 偏移 ~主机墙钟精度（亚毫秒~毫秒级） |
| MCU 时钟校准 | 逐次 poll 随机抖动（1-3ms） | ±1-3ms/样本 | ~1ms/样本（200Hz 主传感器），~2-3ms（50Hz touch） |
| 合计 | | ±3~13ms | **±1~3ms** |

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
> ——199Hz ±2.5ms、**touch 50Hz ±10ms**（见 §4）；(b) 相机侧 `pts_offset` 的测量抖动
> （`camera_first_wall_s` 落盘时刻波动）。另外延迟的**平均值**被 `intercept` 吸收、不进残差，但它在
> BOX↔相机之间留下一个固定 skew（BOX 内部各传感器对齐时相消）。**所以对 touch，主导误差是 ±10ms
> 的最近邻量化，而非 ±2ms 的校准残差。**

## 6. 完整对齐流程（per episode）

```
录制开始
  │
  ├─ box.start_recording(t_start)
  │    └─ poll loop 切到 500Hz，开始 per-sensor MCU 时间戳去重
  │
  ├─ 录制进行中...
  │    ├─ 11 路 gst-launch 写 MKV（H.265，帧级 PTS 在容器内）
  │    └─ poll loop 每次检测新 MCU 时间戳 → 存入 per-sensor buffer
  │
  ├─ session.stop()
  │    └─ SIGINT → gst-launch EOS → MKV 正常封口
  │
  ├─ box.stop_recording()
  │    └─ 返回 {sensor_id: [SensorSample, ...]}
  │
  ├─ pts_offset 计算（_pts_offset_from_handle）
  │    └─ mean(first_wall_s[cam] - t0_wall_s)，逐路 first_wall_s 来自 worker 上报
  │       （非 ffprobe；明细写入 meta.json sync_reference.camera_first_wall_s）
  │
  ├─ 写入 box_sensors.jsonl（原始数据归档）
  │
  └─ 写入 LeRobot v3 parquet（_build_episode_rows）
       ├─ 逐传感器 MCU 时钟校准（calibrate_sensor_samples 线性回归 + 安全回退）
       ├─ 帧时间网格 = pts_offset + frame_index / 60   （t0 相对域）
       └─ 对每帧逐传感器二分查找最近邻 → 组成 state 向量
```

## 7. 同步级别总览

| 级别 | 精度 | 状态 | 机制 |
|------|------|------|------|
| **L0** 相机间硬同步 | <1μs | ✅ | PWM slave mode，11 路帧锁定同一边沿 |
| **L1** 软同步元数据 | — | ✅ | meta.json `sync_reference` 记录 t0_wall_s / t0_mono_s / camera_first_wall_s（逐路，跨相机锚点）；per-stream PTS 在 `cameras[].first_pts_s`（不可跨相机比较，不进 sync_reference） |
| **L3a** 高频独立采样 | ±3~13ms | ✅ | 500Hz poll + MCU 时间戳去重 + 逐传感器最近邻 |
| **L3b** 增强对齐 | ±1~3ms（校准残差，实测；端到端另叠最近邻 ±间隔/2，见 §5.4） | ✅ | 首帧 host wall-time 偏移（pts_offset）+ MCU↔Host 时钟线性回归 |
| **L4** 硬件级全同步 | <1μs | 🔲 | BOX MCU 也由 PWM 触发（需硬件改动） |

### 7.1 代码位置 & 测试映射

| 机制 | 代码 | 测试（无需真机） |
|------|------|----------------|
| 500Hz poll + MCU 去重 | `box_sdk/box_client.py` `_poll_loop`（`record_poll_interval_s=0.002`；录制外 `poll_interval_s=0.05`=20Hz） | `tests/scripts/test_thor_box_client.py`（`_FakeBox` 内存 stub） |
| pts_offset（首帧 wall-time） | `gmsl2/thor_record.py` `_pts_offset_from_handle`；worker 侧 `persistent_session_worker.py` `first_wall_s` | （需 worker 事件，部分覆盖于 multiprocess 测试） |
| 帧网格 + 最近邻对齐 | `gmsl2/thor_lerobot_v3.py` `_build_episode_rows` / `_nearest_sample_data` | `tests/scripts/test_thor_ts_sync_alignment.py` |
| MCU 时钟校准 + 回退 | `gmsl2/thor_lerobot_v3.py` `calibrate_mcu_clock` / `calibrate_sensor_samples` | `tests/scripts/test_thor_ts_sync_alignment.py` |
| ffprobe/GStreamer PTS（**离线数帧**） | `gmsl2/thor_lerobot_v3.py` `extract_pts`；`gmsl2/export_v3.py` | `tests/scripts/test_thor_lerobot_v3_pts.py` |

> 这些测试只覆盖纯 Python 对齐逻辑与数据契约。真机才能确认的内容（MCU 时钟是否线性/单调、
> `mcu_timestamp` 真实语义与单位、各传感器真实频率与 poll 抖动、BOX↔相机端到端实测精度）
> 仍需 BOX 到位后验证，见 §8.2 / §8.5。

## 8. 注意事项

### 8.1 spawn stagger 与首 episode

11 路相机以 1.0s 间隔错开启动（`spawn_stagger_s: 1.0`），共需 ~11s。这期间 PWM 已经在发送触发信号，但各相机在不同时刻开始响应。`t_start` 设在全部相机启动完成之后，BOX 录制也从此刻开始，确保帧时间和传感器时间使用同一基准。

若 stagger 改小（如 0.5s），会触发 Argus ISP 的 `NvBufSurfaceFromFd Failed` 竞争错误，导致部分相机在几秒后 EOS 退出、MKV 仅含 336 字节空头。经验值：`1.0s` 可保证 11/11 路全部成功。

### 8.2 MCU 时钟假设

线性回归假设 MCU 时钟在录制期间**线性且单调**。如果 MCU 时钟有跳变、回绕或非线性漂移，校准会失败（残差 > 50ms 阈值），自动回退到 L3a。

### 8.3 ffprobe 依赖

**录制对齐路径不依赖 ffprobe。** `pts_offset` 来自 worker 上报的 `first_wall_s`（见
§5.1），与 FFmpeg 无关。若某路相机未上报有效 `first_wall_s`，`_pts_offset_from_handle`
返回 `None`，writer 用 `frame_origin_s = 0`（即退回 L3a 精度，仅丢失管道启动延迟修正）。

ffprobe 仅在**离线** `export_v3.py` 数帧时用到（`extract_pts`，反映容器内 PTS）：
Jetson 镜像不一定带 ffprobe，所以 `extract_pts` 在 ffprobe 缺失时自动回退到
`_extract_pts_gstreamer`（用 GStreamer `matroskademux` 读 PTS）。二者都不可用才告警。

### 8.4 训练数据 vs 原始数据

LeRobot v3 parquet 中的 `observation.state` 是 **60Hz 下采样** 后的对齐结果。如果下游任务需要更高频率的传感器数据（如力控、IMU 积分），应直接读取 `box_sensors.jsonl` 原始文件，其中保留了每个传感器在原生频率下的完整时间序列。

### 8.5 wall-clock 精度

整个软同步依赖 `time.time()` 的绝对精度。如果 Jetson 未配置 NTP 或系统时间有跳变，对齐质量会下降。建议：
- 确保 NTP 同步（`timedatectl status` 检查）
- 避免在录制期间手动修改系统时间
- `time.monotonic()` 用于持续时间测量不受 NTP 步进影响，但跨进程对齐仍需 wall-clock

### 8.6 EE pose 轨迹生成复用同一帧网格（2026-06-25）

离线 EE 轨迹生成（GUI「Generate EE Trajectory」/ `april_cube_tracking_in_robot_base.py`）由多路 PWM 硬同步相机流估算 cube/EE pose，按 **per-episode 相机帧序号 N** 写 sidecar（`derived/april_cube_tracking_in_robot_base/state_action.*.csv`）。GUI replay timeline 给每帧的时间戳走的就是本文 §5.1 的帧网格 `pts_offset + N/fps`：

- 有 v3 parquet 的数据集：直接用 parquet `timestamp` 列。
- 无 v3 parquet（相机-only / `--no-box`）的数据集：`gateway._gmsl2_pts_offset_s()` 从 `meta.json.sync_reference`（`t0_wall_s` + `camera_first_wall_s`）现算同一个 `pts_offset`。

即 EE pose 落在相机/PWM 时间轴上，与 BOX MCU 钟（`box.timestamps`）无关。详见 `tools/data_collection_gui/docs/traj_gen_thor_gmsl2_compatibility.md`。

## 9. v3 数据集 schema（2026-06-15 重构）

录制器（`thor_lerobot_v3.py`）写的是 **box-only 的最小 v3** parquet（数值特征 + 时间戳元数据），
相机以 `cam_*.mkv` 原始文件并排存在每个 episode 目录里；离线 `export_v3.py` 再把相机转码并
合并出带 `observation.images.*` 的训练数据集。两侧共享同一 `t0_wall_s`（见 §5.1）。

`export_v3` 的多传感器对齐：相机网格为权威（episode 内各相机 PWM 锁定帧数取最小，转码到恒定
`i/fps` 网格）。box 状态按优先级挂到该网格：① 复用录制器已对齐的 session parquet，**按
`frame_index`（而非列表位置）** 配相机帧（box 网格比相机长的 `round(duration*fps)` 尾部丢弃、
短的 carry-forward）；② 无 parquet 时回退到每集 `box_sensors.jsonl`，用 §5/§6 同一套（MCU 校准
+ 逐传感器最近邻于 `pts_offset + N/fps`）在 export 内重做 L3b。每集输出 `timestamp` 重基到
`i/fps` 以匹配重锚定的逐集视频。代码见 `export_v3._align_box_rows_by_frame_index` /
`_box_rows_from_raw`。

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

### 9.3 真机验证状态（2026-06-15）

| 项 | 结果 |
|----|------|
| 6 路传感器频率 | gripper/imu/trigger/六维力 199Hz，touch L/R 各 50Hz |
| MCU 校准（L3b） | slope=1µs/tick，6 路全 engage，残差 1–2ms |
| 夹爪/扳机运动 | episode 实测 distance 0.0007–0.098m、trigger 0→100% 正确进 `state[0/1]` |
| 触觉接触 | `active_points` 1–52、`max_abs_fz` 饱和 255、239 点原始帧完整 |
| 多模态采集 | 9 路相机（argus_failed=[]）+ 夹爪 + 双触觉同步采集；相机 first_wall_s 展布 ~10ms，pts_offset≈11ms |
| LeRobotDataset 加载 | ✅ 可加载（box.timestamps 经 loader 降为 float32，见 §9.2） |
| 待验 | 跨域 tap-test（§7.1）、相机视频经 `export_v3` 合并后的端到端 |
