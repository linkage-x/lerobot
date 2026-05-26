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

## 6. 启动脚本

仓库里已有 `~/lerobot/run/run_gateway.sh` / `run_vite.sh` / `restart_gateway.sh`
三个本地脚本（**当前不在 git 跟踪范围内**，按本机环境写）。模板：

```bash
# ~/lerobot/run/run_gateway.sh
#!/usr/bin/env bash
set -e
cd ~/lerobot
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

## 7. 一次性自检

```bash
# 7.1 box_sdk 能起来
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

## 8. 本地开发 → Thor 同步

开发机改完代码后用 rsync 同步到 Thor：

```bash
bash run/sync_to_thor.sh            # 增量同步 (~1s)
bash run/sync_to_thor.sh --dry-run  # 预览
```

脚本自动排除 `.git/`、`node_modules/`、`__pycache__/`、`outputs/` 等。
同步后在 Thor 上重启 gateway：

```bash
ssh nvidia@192.168.111.122
pkill -f "tools.data_collection_gui.gateway"; sleep 1
cd ~/lerobot && PYTHONPATH=src:. PYTHONUNBUFFERED=1 \
  setsid python3 -m tools.data_collection_gui.gateway \
  --config-path tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml \
  --datasets-root outputs/datasets --port 8765 --host 0.0.0.0 \
  </dev/null >/tmp/gateway.log 2>&1 &
```

本地前端 vite 代理默认指向 `http://192.168.111.122:8765`，改完代码
`npm run dev` 即可连到 Thor 上的 gateway。

## 9. 已知问题与已修复的坑

### 已修复（2026-05-26）

* **11 路同时启动 NvBufSurfaceFromFd Failed**：11 路 nvarguscamerasrc
  同时初始化时 Argus ISP 内部 NVMM buffer 分配竞争导致大部分管道在
  几秒内自行 EOS 退出，产出空 MKV（336 字节头）。**修复**：YAML
  `spawn_stagger_s: 0.0` → `1.0`，错开 1 秒启动后 11/11 路全部正常
  录制。Connect 多花 ~10 秒是预期行为。
* **Save 按钮不停止录制**：gateway 发 `"s\n"` 给 recorder stdin，但
  thor_record.py 只认 `"save"` / `"y"` / `"yes"`。`"s"` 被忽略。
  **修复**：改为发 `"save\n"`。
* **probe_argus 超时后 gst-launch 僵尸进程**：`subprocess.run(timeout=8)`
  超时只抛异常不杀子进程，gst-launch 变成孤儿占住 Argus 资源，后续
  所有同 sensor-id 的 probe 也会超时。**修复**：改用 `Popen` +
  `proc.kill()` + `proc.wait()`。

### 仍存在

* **BOX 采集板传感器流上行**：`Box()` 起得来、`set_mode` ACK、夹爪可动，
  但 `get_sensor_cache` 一直返回 rc=4 / no cached sensor data。当前 gateway
  会把 BOX SDK 会话启动成功视为 6 个 BOX 设备连通，并把 rc / 错误文本 /
  poll 时间 / 每个传感器最后 timestamp 写入 episode `meta.json` 的
  `box_collection.snapshots[*].status`。供应商已确认下行通路 OK，方向是
  ARM 网关 / 接收路径。诊断步骤见
  `tools/thor/box_sdk/TROUBLESHOOTING.md`（tcpdump 范围扩、`rp_filter`、
  RX offload、IGMP、`set_packet_observer` 探测等 8 步）。
* **MAX96726 sid 锁定数 ≠ YAML 槽位**：YAML 默认 detect_all + `sensor_ids: []`
  的 16 个相机槽是"期望"；实际锁到几个看插了几路相机线。多出来的槽位
  Connect 后会变红 `error`，正常现象。
* **未跟踪的本地脚本**：`~/lerobot/run/` 下三个脚本不在 git 里 —— 是因为
  端口 / IP / venv 路径会因部署点而不同，复制本文 §6 的模板自己填。

## 10. 相机-传感器时间同步架构

### 当前状态（2026-05-26）

11 路 GMSL2 相机通过 60Hz PWM 信号硬同步（`sensor_trig_mode=1`），所有相机
帧对齐到同一 PWM 上升沿（亚微秒精度）。BOX 采集板传感器（gripper / IMU /
trigger / 6D force / touch）通过 UDP/15000 发送，MCU 内部时钟独立于主机。

**两套数据源的时钟域完全独立，没有硬件级公共时间基准。**

### 已实现：sync_reference 元数据

每个 episode 的 `meta.json` 包含 `sync_reference` 块：

```json
{
  "sync_reference": {
    "t0_wall_s": 1716700000.123,
    "t0_mono_s": 12345.678,
    "camera_spawn_wall_s": { "cam_02": 1716700000.123, "cam_07": 1716700001.124 },
    "camera_spawn_offset_s": { "cam_02": 0.000, "cam_07": 1.001 }
  }
}
```

- `t0_wall_s`：`time.time()` at episode start，公共纪元锚点
- `camera_spawn_offset_s`：各相机 gst-launch 启动相对 t0 的偏移（因 stagger 不同）
- BOX snapshot 携带 `t_relative_s = time.time() - t0`，直接对齐到同一时间轴

### 后处理对齐公式

```
相机帧 N 的时间 = t0_wall_s + camera_spawn_offset_s[cam_id] + N / fps
BOX snapshot 时间 = t0_wall_s + t_relative_s
```

最近邻插值即可。20Hz BOX poll 下精度 ±25ms。

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
| L3 录制时高频对齐 | ±8ms | 🔲 可选 | BOX poll 提到 60Hz，帧级对齐表 |
| L4 硬件级全同步 | <1μs | 🔲 需硬件改 | BOX MCU 也由 PWM 触发 |
