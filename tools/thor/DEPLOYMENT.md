# Thor 数据采集机部署清单

把 `tools/data_collection_gui` + `tools/thor/{gmsl2,box_sdk}` 这套（11 路
GMSL2 + BOX 采集板）部署到一台新的 Thor / Jetson 主机时，按顺序跑完下面
这份清单。**所有曾经踩过的坑都在这里**，目标是新机零踩坑。

## 0. 前提

- JetPack 6+（Ubuntu 24.04 noble，`uname -m` 显示 `aarch64`）。
- `nvidia` 账号有 NOPASSWD `sudo`（GMSL2 摄像头需要 `sudo v4l2-ctl` /
  `sudo sh pwm.sh` —— 检查是否为60Hz）。
- 仓库 clone / rsync 到 `~/lerobot`。
- 国内网络：跳板 / GFW 干扰严重，**默认不能直连 `raw.githubusercontent.com`、
  `nodejs.org`、`pypi.org`**。下面所有源都改国内镜像。

## 1. 系统层 apt 依赖

```bash
sudo apt update
sudo apt install -y \
  libeigen3-dev liburdfdom-dev libtinyxml2-dev \
  libboost-filesystem1.74.0 libboost-system1.74.0 \
  curl rsync git \
  v4l-utils python3-pip
```

| 包 | 谁要用 | 不装会怎样 |
| --- | --- | --- |
| `libeigen3-dev` | `libgripper_kinematics.so` 链接时 | Box 控制器 dlopen 失败 |
| `liburdfdom-dev` | `libpinocchio_parsers.so` 运行时 | `Box()` 构造时报缺 `liburdfdom_model.so.*` |
| `libtinyxml2-dev` | 同上 | `Box()` 构造时报缺 `libtinyxml2.so.*` |
| `libboost-filesystem1.74.0` / `libboost-system1.74.0` | BOX SDK wheel 的旧 Boost soname | `Box()` 构造时报缺 `libboost_filesystem.so.1.74.0` 或 `libboost_system.so.1.74.0` |
| `curl` | nvm 安装脚本、本清单后续命令 | `curl: command not found` |
| `v4l-utils` | GMSL2 录制时 v4l2-ctl push 控件 | hardware_sync 启用时报错（默认禁用时不影响） |
| `python3-pip` | 安装 pyarrow / box_sdk wheel | `pip: command not found` |

## 2. BOX SDK 兼容 symlink

SDK 的 `.so` 是按旧 soname 编的（`libtinyxml2.so.9` / `liburdfdom_model.so.3.0`），
JetPack 6 系统装的是 `.so.10` / `.so.4.0`。一个 helper 脚本自动建符号链接，
并检查 Boost 1.74 运行时包是否已安装：

```bash
bash ~/lerobot/tools/thor/box_sdk/install_compat_links.sh
```

幂等。脚本会自动判断 aarch64 / x86_64，并选 `/usr/lib/<multiarch>/` 下版本
最高的同名库。不装好 §1 的 `liburdfdom-dev` / `libtinyxml2-dev` 这步会报 "no
system candidate" 警告；不装 Boost 1.74 兼容包时会报 missing runtime。

## 3. Python 运行时依赖

```bash
# 1) GUI gateway + 数据集 replay/QC 用
python3 -m pip install --user --break-system-packages \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  pyarrow

# 2) BOX 采集板 SDK Python 绑定
python3 -m pip install --user --break-system-packages \
  -i https://pypi.tuna.tsinghua.edu.cn/simple \
  --force-reinstall \
  ~/lerobot/tools/thor/box_sdk/python/box_collection_sdk-0.1.0-py3-none-any.whl
```

| 包 | 谁要用 | 不装会怎样 |
| --- | --- | --- |
| `pyarrow` | gateway 读 LeRobot v3 parquet（Replay / QC / Dataset Processing 页面） | **Episode Replay 页面白屏（已加防御渲染，会显示 "pyarrow unavailable" 提示但功能不可用）**；QC `501` |
| `box_collection_sdk` (wheel) | `BoxClient.start()` / `set_mode` / `get_sensor_cache` | gateway 起得来，但 thor_record 的 BoxClient.start() 直接 no-op，前端 6 个 box 设备一直 `error` |

`--break-system-packages` 是 PEP 668 在 Ubuntu noble 上的强制要求；
`--user` 装到 `~/.local/lib/python3.12/site-packages/`，不动系统目录。

要拉 LeRobot 本体的依赖（torch / numpy 等用于训练 / mujoco），按
`pyproject.toml` 单独建 venv，跟数据采集 GUI 解耦。GUI 路径**只**需要
`pyyaml + pyarrow + box_collection_sdk`。

## 4. Node / npm

国内网络下 `nodejs.org` 经常拉不动。用 nvm + gitee 镜像源 + npmmirror：

```bash
# 4.1 nvm 从 gitee 镜像 clone（raw.githubusercontent.com 国内访问不通）
git clone --depth 1 -b v0.40.1 https://gitee.com/mirrors/nvm.git ~/.nvm

# 4.2 把 nvm 写进 .bashrc
cat >> ~/.bashrc <<'EOF'

export NVM_DIR="$HOME/.nvm"
[ -s "$NVM_DIR/nvm.sh" ] && \. "$NVM_DIR/nvm.sh"
[ -s "$NVM_DIR/bash_completion" ] && \. "$NVM_DIR/bash_completion"
EOF

# 4.3 装 node LTS（走 npmmirror）
source ~/.nvm/nvm.sh
export NVM_NODEJS_ORG_MIRROR=https://npmmirror.com/mirrors/node/
nvm install --lts
nvm use --lts

# 4.4 npm 切到国内 registry
npm config set registry https://registry.npmmirror.com
```

可达性自检：

```bash
node -v && npm -v && \
  curl -fsSL --max-time 8 -o /dev/null -w "ok %{http_code} %{time_total}s\n" \
    https://npmmirror.com/mirrors/node/
```

## 5. 安装前端依赖

```bash
cd ~/lerobot/tools/data_collection_gui/frontend
npm install
```

正常在 5 秒内装完 77 个包（国内 registry 速度 1MB/s+）。要是慢到 1 分钟
以上 —— 检查是不是又用回了官方 registry。

## 6. BOX 网口地址持久化

BOX 固件固定向 `192.168.2.45:15000` 推送传感器 UDP。Thor 重启后手动添加的
`192.168.2.45/24` 会丢失，因此首次部署或重刷网络后必须安装 repo 内的
systemd 服务：

```bash
cd ~/lerobot
bash tools/thor/box_sdk/install_box_net_service.sh
```

该服务会在开机后执行 `tools/thor/box_sdk/ensure_box_net.sh`，幂等地完成：

- `enP2p1s0` up
- 添加 `192.168.2.45/24`
- 默认不修改内核过滤策略
- 打印到 `192.168.2.60` 的实际源地址路由
- 主动从 `192.168.2.45` probe 一次 `192.168.2.60`，刷新 BOX 重上电后的 ARP/邻居状态

临时检查或手动恢复也可以直接运行：

```bash
cd ~/lerobot
bash tools/thor/box_sdk/ensure_box_net.sh
```

只有在确认需要关闭反向路径过滤时，才显式传入：

```bash
BOX_NET_DISABLE_RPFILTER=1 bash tools/thor/box_sdk/ensure_box_net.sh
```

## 7. 启动脚本

仓库里已有 `~/lerobot/run/run_gateway.sh` / `run_vite.sh` / `restart_gateway.sh`
三个本地脚本（**当前不在 git 跟踪范围内**，按本机环境写）。模板：

```bash
# ~/lerobot/run/run_gateway.sh
#!/usr/bin/env bash
set -e
cd ~/lerobot
bash tools/thor/box_sdk/ensure_box_net.sh >/dev/null
. tools/thor/box_sdk/setup_env.sh
exec env PYTHONPATH=src:. PYTHONUNBUFFERED=1 \
  python3 -m tools.data_collection_gui.gateway \
  --config-path tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml \
  --datasets-root outputs/datasets \
  --port 8765 \
  --repo-root /home/nvidia/lerobot
```

```bash
# ~/lerobot/run/run_vite.sh
#!/usr/bin/env bash
set -e
cd ~/lerobot/tools/data_collection_gui/frontend
export NVM_DIR="$HOME/.nvm"
. "$NVM_DIR/nvm.sh"
nvm use --lts >/dev/null
exec npm run dev -- --host 0.0.0.0 --port 5173
```

```bash
# ~/lerobot/run/restart_gateway.sh
#!/usr/bin/env bash
pkill -f "tools.data_collection_gui.gateway" 2>/dev/null || true
sleep 1
setsid bash ~/lerobot/run/run_gateway.sh </dev/null >~/lerobot/run/logs/gateway.log 2>&1 &
disown
sleep 3
pgrep -af "tools.data_collection_gui.gateway" | head -3
tail -5 ~/lerobot/run/logs/gateway.log
```

**ssh 后台启动注意**：直接 `nohup ... &` 在 ssh 退出时 exit 255。要用
`setsid bash <script> </dev/null >log 2>&1 &; disown` 才能干净 detach。

## 8. 一次性自检

```bash
# 7.1 box_sdk 能起来
# 注：BOX_SDK_URDF 现在由 BoxClient.start() 自动设置，无需 source setup_env.sh
# 即可调通；但 LD_LIBRARY_PATH 仍需要包含 box_sdk/lib（setup_env.sh 顺手设了，
# 也可手动 export）。
. ~/lerobot/tools/thor/box_sdk/setup_env.sh
python3 -c "
from tools.thor.box_sdk import box_client
c = box_client.BoxClient(box_client.from_yaml_dict({'enabled': True}))
print('start:', c.start())  # 应该 True
c.stop()
"

# 7.2 gateway snapshot 反映 GMSL2/BOX 默认
cd ~/lerobot && PYTHONPATH=src:. python3 -c "
from pathlib import Path
from tools.data_collection_gui import gateway
state = gateway.make_state(Path.cwd(), gateway.DEFAULT_CONFIG_PATH)
snap = gateway._snapshot(state)
print('repoId:', snap['configSummary']['repoId'])    # local/thor_gmsl2_11ch_v1
print('cameras:', sum(1 for d in snap['devices'] if d['kind']=='camera'))   # 16
print('box:', [d['id'] for d in snap['devices'] if d['kind']=='box_collection'])
"

# 7.3 测试集 14/14 过
PYTHONPATH=src:. python3 -m pytest --noconftest -q \
  tests/scripts/test_thor_box_client.py \
  tests/scripts/test_data_collection_gui_gateway.py::test_default_config_is_thor_gmsl2_box \
  tests/scripts/test_data_collection_gui_gateway.py::test_box_collection_devices_use_remote_endpoint_in_detail \
  tests/scripts/test_data_collection_gui_gateway.py::test_box_collection_disabled_hides_devices \
  tests/scripts/test_data_collection_gui_gateway.py::test_recorder_output_marks_box_collection_devices \
  tests/scripts/test_data_collection_gui_gateway.py::test_recorder_script_picks_thor_when_configured \
  tests/scripts/test_data_collection_gui_gateway.py::test_recorder_script_defaults_to_handheld_for_legacy_configs

# 7.4 启动 + 浏览器
bash ~/lerobot/run/restart_gateway.sh
bash ~/lerobot/run/run_vite.sh &
# 开发机访问 http://<jetson-ip>:5173/
```

## 9. 本地开发 → Thor 部署

### 一键部署

开发机改完代码后一条命令完成 同步 → 重启 gateway → 启动前端：

```bash
bash run/deploy.sh              # 全流程：sync + restart gateway + start vite
bash run/deploy.sh --sync-only  # 只同步代码，不重启/不启前端
bash run/deploy.sh --no-frontend # 同步 + 重启 gateway，不启前端
```

脚本执行：
1. `rsync` 增量同步到 `nvidia@192.168.111.122:~/lerobot`（~1s）
2. SSH 到 Thor：kill 旧 gateway → `ensure_box_net.sh` → `setup_env.sh` → 后台启动 gateway
3. 本地 `npm run dev` 启动 vite，代理 → `192.168.111.122:8765`
4. 浏览器打开 `http://localhost:5173/`

### 单独操作

```bash
bash run/sync_to_thor.sh            # 只同步（增量 ~1s）
bash run/sync_to_thor.sh --dry-run  # 预览同步内容
```

`sync_to_thor.sh` 排除 `.git/`、`node_modules/`、`outputs/`、Thor 本地的
`run/run_gateway.sh` / `run/restart_gateway.sh` / `run/logs/` 等。

## 10. 已知问题与已修复的坑

### 已修复（2026-05-26）

* **11 路同时启动 NvBufSurfaceFromFd Failed**：11 路 nvarguscamerasrc
  同时初始化时 Argus ISP 内部 NVMM buffer 分配竞争导致大部分管道在
  几秒内自行 EOS 退出，产出空 MKV（336 字节头）。**修复（2026-05-26）**：
  YAML `spawn_stagger_s: 0.0` → `1.0`，错开 1 秒启动后 11/11 路全部正常
  录制；Connect 多花 ~10 秒。**进一步修复（2026-05-28，PR2）**：切到
  `PersistentCameraSession`，11 路 nvarguscamerasrc 在 Connect 时一次
  spawn 完毕，后续 episode 切片靠 splitmuxsink `split-now`，每个
  StartEpisode 不再付 ~11s 启动开销。
* **Save 按钮不停止录制**：gateway 发 `"s\n"` 给 recorder stdin，但
  thor_record.py 只认 `"save"` / `"y"` / `"yes"`。`"s"` 被忽略。
  **修复**：改为发 `"save\n"`。
* **probe_argus 超时后 gst-launch 僵尸进程**：`subprocess.run(timeout=8)`
  超时只抛异常不杀子进程，gst-launch 变成孤儿占住 Argus 资源，后续
  所有同 sensor-id 的 probe 也会超时。**修复**：改用 `Popen` +
  `proc.kill()` + `proc.wait()`。
* **BOX SDK 首个 UDP 包后 SIGABRT**：BOX SDK C++ 端初始化 gripper
  kinematics 时加载 URDF，未设 `$BOX_SDK_URDF` 时 fallback 到相对路径
  `thirdpart/monte_gripper.urdf`，文件不存在 → `std::invalid_argument`
  未捕获 → `std::terminate()` → SIGABRT。从命令行直接跑 thor_record
  / pytest 而忘 `source setup_env.sh` 时必触发。**修复（2026-05-28）**：
  `BoxClient.start()` 进入时 `os.environ.setdefault("BOX_SDK_URDF",
  Path(__file__).resolve().parent / cfg.urdf_relpath)`，跟 cwd 无关，
  不再依赖外部 source。`LD_LIBRARY_PATH` 仍由调用方在 `dlopen` 前设好
  （gateway 启动包装脚本 / `setup_env.sh` 一直在做）。
* **operator stdin "Enter Enter" 第二个被吞**：`_drain_until` 在
  duration_reached 之后 0.2s 内会把队列里非 save/discard/quit 命令一并
  pop 当 noise，导致 GUI 连发 `\n\n` 想 "save + start next" 时第二个
  `\n` 丢掉。**修复（2026-05-28）**：drain 看到队首不是接受的 kind
  时立即 return None，不消费队列。

### 仍存在

* **BOX 采集板传感器流上行**：供应商确认夹爪端固定只向
  `192.168.2.45` 上行。Thor 已在 `enP2p1s0` 配置 `192.168.2.45/24`，
  `box_collection.bind_ip` 已改为 `192.168.2.45`，并由 `thor-box-net.service` 开机恢复该地址；按该地址复测后，
  原始抓包可见 `192.168.2.60 -> 192.168.2.45 UDP/15000`，
  `get_sensor_cache` 已由 rc=4 变为 rc=0 / valid=1。LeRobot wrapper
  已确认 6 个 BOX sensor 全部 seen/fresh，gripper distance 与各 sensor
  timestamp 可正常记录。
* **MAX96726 sid 锁定数 ≠ YAML 槽位**：YAML 默认 detect_all + `sensor_ids: []`
  的 16 个相机槽是"期望"；实际锁到几个看插了几路相机线。多出来的槽位
  Connect 后会变红 `error`，正常现象。
* **StartEpisode 后没有数据落盘**：已定位到 GUI recorder 旧逻辑在任一路
  GStreamer stream 提前退出时走 discard，并删除整个 episode 目录。现在
  `thor_record.py` 会保留该 episode、写 `meta.json`，并在前端输出类似
  `Stream exited early: cam_03(rc=0, log=cam_03.gst.log)` 的根因。2026-05-27
  实测失败相机为 `cam_03` / `cam_04`，日志为 `NvBufSurfaceFromFd Failed` 后 EOS。
* **未跟踪的本地脚本**：`~/lerobot/run/` 下三个脚本不在 git 里 —— 是因为
  端口 / IP / venv 路径会因部署点而不同，复制本文 §7 的模板自己填。

## 11. 相机-传感器时间同步架构

### 当前状态（2026-05-26）

11 路 GMSL2 相机通过 60Hz PWM 信号硬同步（`sensor_trig_mode=1`），所有相机
帧对齐到同一 PWM 上升沿（亚微秒精度）。BOX 采集板传感器（gripper / IMU /
trigger / 6D force / touch）通过 UDP/15000 发送，MCU 内部时钟独立于主机。

**两套数据源的时钟域完全独立，没有硬件级公共时间基准。**

### 已实现：sync_reference 元数据

每个 episode 的 `meta.json` 包含 `sync_reference` 块。PR2（持久 pipeline）后
字段从 `camera_spawn_*` 改成 `camera_first_*`，含义不同（见下）：

```json
{
  "sync_reference": {
    "t0_wall_s": 1716700000.123,
    "t0_mono_s": 12345.678,
    "split_now_wall_s": 1716700000.125,
    "camera_first_wall_s": { "cam_02": 1716700000.430, "cam_07": 1716700000.850 },
    "camera_first_pts_s":  { "cam_02": 1.234, "cam_07": 2.456 }
  }
}
```

- `t0_wall_s` / `t0_mono_s`：`time.time()` / `time.monotonic()` at
  StartEpisode 触发时刻
- `split_now_wall_s`：`splitmuxsink.emit("split-now")` 真正发出的 wall time
- `camera_first_wall_s[cam]`：该相机首个 EPISODE fragment 在 host wall-clock
  上真正开新文件的时刻（FragmentInfo capture），**跨相机可比**，与
  `split_now_wall_s` 之差即每路切换延迟（典型 100–500 ms，取决于
  `iframe_interval` 和最近一帧 PWM 时刻）
- `camera_first_pts_s[cam]`：该 fragment 首个 buffer 的管道 PTS（per-stream
  时钟，**跨相机不可比**，PR1 burn-in 实测偏差 10s 量级）

旧字段 `camera_spawn_wall_s` / `camera_spawn_offset_s` 在持久 pipeline 模式
下已无意义（11 路只在 Connect 时 spawn 一次，每个 episode 不再 spawn）。

### 后处理对齐公式

```
相机帧 N 的时间 = camera_first_wall_s[cam_id] + N / fps
BOX snapshot 时间 = t0_wall_s + t_relative_s
```

跨相机对齐用 `camera_first_wall_s` 做锚点，PR1 burn-in 实测跨相机
精度 19.5 ms（被 `iframe_interval` 主导，调小可改善）。

### 已实现：高频独立采样 + 逐传感器软同步（L3）

录制时 `BoxClient` 以 500Hz (`record_poll_interval_s=0.002`) 轮询 SDK 的
`get_sensor_cache()`，通过比较各传感器的 MCU 时间戳去重，每个传感器按 MCU
原生推送频率独立记录。每个样本标注：

- **MCU 时间戳**：传感器硬件时钟，仅用于去重（检测新样本），不参与对齐
- **主机 `time.time()`**：轮询到数据的时刻，用于与相机帧对齐

对齐基于主机 wall-clock：相机帧时间 `t0 + frame_index / 60` 和 BOX 样本时间
`time.time()` 都引用主机时钟。但注意两侧均有抖动来源：

- 相机侧：spawn offset 是 gst-launch 进程启动时刻，非首帧 PWM 边沿时刻
- BOX 侧：UDP 传输延迟 + 轮询时机（`time.time()` 是收到时刻，非 MCU 采集时刻）

录制结束时：
- 原始数据写入 `box_sensors.jsonl`（每行一个传感器样本，按时间排序）
- LeRobot v3 parquet 中对每个 60Hz 相机帧，逐传感器做最近邻插值（下采样）

**对齐精度取决于传感器推送频率和主机侧抖动**：
- IMU (典型 200Hz) → ±2.5ms + 轮询抖动
- 六维力 (典型 100Hz) → ±5ms + 轮询抖动
- Gripper (典型 50Hz) → ±10ms + 轮询抖动
- 触觉 (典型 100Hz) → ±5ms + 轮询抖动

### 已实现：增强对齐（PTS 提取 + MCU 时钟校准）

L3 基础对齐依赖两个近似：相机帧时间用 `spawn_offset + N/fps` 推算（忽略
管道启动延迟），BOX 样本时间用轮询时刻 `time.time()` 代替（包含 UDP 延迟
和 poll 抖动）。增强对齐从两侧消除这些抖动源：

**相机侧：MKV PTS 提取**

Episode 结束后用 `ffprobe` 从参考相机的 MKV 中提取逐帧 PTS。PTS 由
`nvarguscamerasrc do-timestamp=true` 写入，反映实际帧采集时刻相对管道时钟
的偏移。由于 11 路相机共享 PWM 触发，帧间间隔严格为 `1/fps`，只需一路的
PTS 即可重建整个帧时间网格：

```
actual_frame_time[N] = camera_spawn_wall_s + pts[0] + N / fps
```

其中 `pts[0]` 是首帧实际到达时刻，包含了管道启动延迟（典型 100-500ms）。

**BOX 侧：MCU↔Host 时钟线性回归**

录制期间每个传感器有大量 `(mcu_timestamp, host_wall_time)` 观测对。对每个
传感器做最小二乘线性回归：

```
host_time_estimated = slope × mcu_ts + intercept
```

- `slope` ≈ MCU 时钟周期（ticks → seconds）
- `intercept` = 两个时钟域的偏移量

拟合后用 `mcu_timestamp` 反推更准确的主机时间，消除逐次 poll 的 UDP 延迟
和抖动。残差标准差即为校准精度（典型 <0.5ms）。

**对齐流程**

1. `ffprobe` 提取参考相机 PTS → 得到实际帧时间网格
2. 逐传感器线性回归 → 得到每个样本的校准后主机时间
3. 对每个相机帧，逐传感器在校准后时间线上做最近邻 → 组成 state 向量

精度从 ±2.5~10ms 提升到 ±0.5~1ms，代价是 episode 结束后 ~1s 后处理。

### 待实现：导出阶段对齐

数据集导出（Dataset Export 页面 / CLI）时应将原始 MKV + meta.json 转换为
LeRobot v3 parquet 训练格式，在此阶段执行对齐：

1. 从 MKV 提取 per-frame PTS（`gst-discoverer` 或 `ffprobe`）
2. 用 `sync_reference` 将各相机帧映射到公共 wall-clock 时间线
3. 将 BOX snapshot 按 `t_relative_s` 插值到每个相机帧时间点
4. 写入 `observation.state`（BOX 传感器值）和 `observation.images.*`（视频引用）

**前提**：BOX 传感器上行修复后才有可对齐的数据（当前 rc=4）。
`sync_reference` 元数据已就位，导出代码待 BOX 上行通后实现。

### 同步级别路线图

| 级别 | 精度 | 状态 | 说明 |
| --- | --- | --- | --- |
| L0 硬同步（相机间） | <1μs | ✅ 已工作 | PWM slave mode，11 路帧对齐 |
| L1 软同步元数据 | ±25ms | ✅ 已实现 | sync_reference in meta.json |
| L2 导出时对齐 | ±25ms | 🔲 待实现 | BOX 上行通后实现，写入 parquet |
| L3a 录制时高频对齐 | ±2.5ms~±10ms | ✅ 已实现 | 500Hz poll + per-sensor MCU 时间戳去重 + 逐传感器最近邻插值 |
| L3b 增强对齐 | ±0.5~1ms | ✅ 已实现 | MKV PTS 提取 + MCU↔Host 时钟线性回归 |
| L4 硬件级全同步 | <1μs | 🔲 需硬件改 | BOX MCU 也由 PWM 触发 |


备注:
thor的isp文件路径/var/nvidia/nvcam/settings/camera_overrides.isp(从驱动sdk拷过去)