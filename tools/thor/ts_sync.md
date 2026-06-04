# Thor 相机-传感器时间同步技术文档

> 适用于 Thor 数据采集平台（11 × GMSL2 相机 + BOX 采集板），2026-05-27

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

| 传感器 | 典型观测频率 | 每 10s episode 样本数 |
|--------|------------|---------------------|
| IMU | ~200 Hz | ~2000 |
| 六维力 | ~200 Hz | ~2000 |
| Gripper | ~200 Hz | ~2000 |
| Trigger | ~200 Hz | ~2000 |
| Touch L | ~200 Hz | ~2000 |
| Touch R | ~50 Hz | ~500 |

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

### 5.1 相机侧修正：MKV PTS 提取

Episode 结束后从参考相机（最早启动的那路）的 MKV 中用 `ffprobe` 提取帧级 PTS：

```bash
ffprobe -v quiet -select_streams v:0 \
  -show_entries packet=pts_time -of csv=p=0 cam_00.mkv
```

PTS 由 `nvarguscamerasrc do-timestamp=true` 写入，反映帧到达 GStreamer pipeline clock 的实际时刻。首帧 PTS（`pts[0]`）即管道启动延迟。

由于所有相机共享 PWM 触发，帧间间隔严格 = 1/fps，只需一路 PTS 即可重建帧时间网格：

```
actual_frame_time[N] = t_start + pts[0] + N / fps
                                 ^^^^^^
                            管道启动延迟修正
```

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

校准不总是有效。以下情况自动回退到 L3a 的原始 poll 时间：

- 该传感器样本数 < 10（不足以拟合）
- 回归残差标准差 > 50ms（拟合质量差，可能 MCU 时钟不是线性的）
- MCU 时间戳全为 0（传感器未上报有效时间戳）

### 5.4 合成精度

| 修正项 | 消除的误差 | 修正前 | 修正后 |
|--------|-----------|--------|--------|
| PTS 提取 | 管道启动延迟（100-500ms 偏移） | 全局偏移 | 偏移 <1ms |
| MCU 时钟校准 | 逐次 poll 随机抖动（1-3ms） | ±1-3ms/样本 | <0.5ms/样本（回归残差） |
| 合计 | | ±3~13ms | **±0.5~1ms** |

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
  ├─ PTS 提取
  │    └─ ffprobe 参考相机 MKV → pts[0]（管道启动延迟）
  │
  ├─ 写入 box_sensors.jsonl（原始数据归档）
  │
  └─ 写入 LeRobot v3 parquet
       ├─ 逐传感器 MCU 时钟校准（线性回归）
       ├─ 帧时间网格 = pts[0] + frame_index / 60
       └─ 对每帧逐传感器二分查找最近邻 → 组成 state 向量
```

## 7. 同步级别总览

| 级别 | 精度 | 状态 | 机制 |
|------|------|------|------|
| **L0** 相机间硬同步 | <1μs | ✅ | PWM slave mode，11 路帧锁定同一边沿 |
| **L1** 软同步元数据 | — | ✅ | meta.json 记录 t0_wall_s / spawn_offset / sync_reference |
| **L3a** 高频独立采样 | ±3~13ms | ✅ | 500Hz poll + MCU 时间戳去重 + 逐传感器最近邻 |
| **L3b** 增强对齐 | ±0.5~1ms | ✅ | MKV PTS 提取 + MCU↔Host 时钟线性回归 |
| **L4** 硬件级全同步 | <1μs | 🔲 | BOX MCU 也由 PWM 触发（需硬件改动） |

## 8. 注意事项

### 8.1 spawn stagger 与首 episode

11 路相机以 1.0s 间隔错开启动（`spawn_stagger_s: 1.0`），共需 ~11s。这期间 PWM 已经在发送触发信号，但各相机在不同时刻开始响应。`t_start` 设在全部相机启动完成之后，BOX 录制也从此刻开始，确保帧时间和传感器时间使用同一基准。

若 stagger 改小（如 0.5s），会触发 Argus ISP 的 `NvBufSurfaceFromFd Failed` 竞争错误，导致部分相机在几秒后 EOS 退出、MKV 仅含 336 字节空头。经验值：`1.0s` 可保证 11/11 路全部成功。

### 8.2 MCU 时钟假设

线性回归假设 MCU 时钟在录制期间**线性且单调**。如果 MCU 时钟有跳变、回绕或非线性漂移，校准会失败（残差 > 50ms 阈值），自动回退到 L3a。

### 8.3 ffprobe 依赖

PTS 提取依赖 Jetson 上安装 `ffprobe`（FFmpeg 套件）。若不可用，回退到 `pts_offset = 0`（即 L3a 精度）。录制器会在日志中输出警告。

### 8.4 训练数据 vs 原始数据

LeRobot v3 parquet 中的 `observation.state` 是 **60Hz 下采样** 后的对齐结果。如果下游任务需要更高频率的传感器数据（如力控、IMU 积分），应直接读取 `box_sensors.jsonl` 原始文件，其中保留了每个传感器在原生频率下的完整时间序列。

### 8.5 wall-clock 精度

整个软同步依赖 `time.time()` 的绝对精度。如果 Jetson 未配置 NTP 或系统时间有跳变，对齐质量会下降。建议：
- 确保 NTP 同步（`timedatectl status` 检查）
- 避免在录制期间手动修改系统时间
- `time.monotonic()` 用于持续时间测量不受 NTP 步进影响，但跨进程对齐仍需 wall-clock
