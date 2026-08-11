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
1. `rsync` 增量同步到 `nvidia@192.168.111.122:~/lerobot`（~1s）；EE-trajectory 标定目录 best-effort 同步，缺失只告警，需硬校验时使用 `REQUIRE_EE_CALIBRATION=1 bash run/deploy.sh`
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

### 已修复（2026-05-29）

* **PR2 持久 pipeline 在 11 路 + hardware_sync 下挂死（"全绿假阳性 + 空 episode 目录"）**：
  - **表现**：Connect 后 11 路 camera 状态全绿；StartEpisode 进度条照走、
    `episodes/episode_NNNNNN/` 目录已创建；但目录下没有 `cam_*.mkv`。
    日志里能看到 `bus EOS` 被静默 + `set_state(PLAYING)` 卡死整个 Python 线程
    （后续 `cam_NN PLAYING (+x.xs)` log 直接断流）。错误冒泡是
    `NvArgusCameraSrc: UNAVAILABLE/TIMEOUT/CANCELLED` 满天飞 + 末尾
    `(Argus) Error 0x00000005 ... ClientSocketManager.cpp send()`。
  - **根因（PR2，`6cfa9236`）**：旧 `EpisodeSession` 每个 episode 启动 11
    个独立 `gst-launch` 子进程，nvargus-daemon 看到的是 11 个独立 RPC client
    （11 条独立 socket）；PR2 切到 `PersistentCameraSession` 后变成**1 个
    Python 进程持 11 个 `Argus::CaptureSession` 共用一条 socket**，daemon
    对"单 client 同时持 N sessions"的恢复路径不稳定，且 pipeline 长期 PLAYING
    意味着错误状态长期不重置。PR2 的 burn-in 只在 2 cams + `--skip-hardware-sync`
    下验证（commit message 有写），没覆盖 11 cams + PWM slave 的生产负载。
  - **修复（2026-05-29）**：每路相机一个子进程，恢复 daemon-client 隔离性
    （详见下"PR3 多进程隔离"段）。
* **bus EOS 被静默吞**：`persistent_session._Stream._on_bus_message` 仅对
  EOS 打 INFO，不当作 stream error，导致 start_episode() 在上游已死的
  pipeline 上 happily emit split-now → `format-location-full` 永远不回调 →
  episode dir 创建但永远没有 mkv。**修复**：EOS 走 `_record_error`，
  并在 connect() 末尾 `poll_errors()` 一次，任何累积错误立刻 raise
  `RuntimeError` 给 thor_record，前端看到一行清楚的 `ERROR: persistent
  pipeline connect failed: …` 而非"卡 30s 后没消息"。
* **`set_state(PLAYING)` 不响应任何 timeout**：nvarguscamerasrc 在 Argus
  死锁状态下，`pipeline.set_state(Gst.State.PLAYING)` 返回 `ASYNC` 后内部
  线程被 Argus library 锁住，整个 Python 线程跟着挂。**修复**：start
  路径在 `set_state(PLAYING)` 后立刻 `get_state(timeout=6s)` 真等 PLAYING
  到位；超时即 raise，错误带 sid 与 "Argus daemon likely needs
  recover_argus.sh" 提示。多进程化后这条 timeout 被继承到 worker 内部，
  即便 worker 整个挂死，父进程的 `wait_ready` 仍能限时退出。
* **probe_argus 在 connect 路径里再压一遍 daemon**：旧模型下 probe 完进程
  退出，子进程后续再上来时 Argus 已清理；PR2 后变成 probe → 立刻并发 11 路
  PLAYING，daemon 同一会话上压力翻倍，导致 recover 末尾 11/11 都过的
  sid 在 thor_record 自己的 probe 里又会 timeout。**修复**：gateway
  spawn thor_record（任何 `thor_record` 的脚本路径）时自动追加
  `--skip-argus-probe`，相机健康检查全部交给 `recover_argus.sh` 那一轮。

#### PR3 多进程隔离（本次重点）

| 维度 | PR2（5/28~5/29 早） | PR3（5/29 本次） |
| --- | --- | --- |
| 进程数 | 1 个 Python | 1 个父 Python + N 个 worker |
| nvargus-daemon 看到的 RPC client 数 | 1 | N |
| set_state(PLAYING) 死锁影响范围 | 全部相机 | 仅该 worker |
| connect 失败诊断 | "假绿、空 episode" 不可见 | 单路 raise 带 sid + 原因 |
| burn-in 覆盖 | 2 cams + skip-hw-sync | 11 cams + PWM slave 待外场验证 |

实现：
- 新增 `tools/thor/gmsl2/persistent_session_worker.py`：子进程入口，
  持一条 `nvarguscamerasrc → ... → splitmuxsink` pipeline + 自己的 GLib
  MainLoop + bus watch + `format-location-full` 回调。
- 父子两条 `multiprocessing.Queue` 通信：
  - `cmd_q`（父→子）：`start_episode(episode_dir)` / `stop_episode` / `disconnect`
  - `evt_q`（子→父）：`playing` / `fragment(dict)` / `error(msg,debug)` /
    `eos` / `episode_done(dict|None)` / `disconnected`
- `PersistentCameraSession` 公共 API（`connect/disconnect/start_episode/
  stop_episode/discard_episode/poll_errors/cleanup_warmup_files/
  restart_stream`）和 `EpisodeHandle/FragmentInfo` 都不变，`thor_record.py`
  一行不动。
- `connect()` 改成**严格串行 spawn**：每 spawn 一个 worker，调
  `wait_ready(timeout)` 阻塞到 `playing` 事件或任何 terminal error，再 sleep
  `spawn_stagger_s` 让 daemon settle，然后 spawn 下一个。N 路 PLAYING 不再
  并发，daemon 一次只面对一个新 client。
- `restart_stream(sid)` 现在是"terminate 老 worker → spawn 新 worker → 等
  PLAYING"，对应 daemon 视角的"踢掉一个 client + 接受一个新 client"，预期
  在错误恢复路径上比旧的 NULL→PLAYING 更可靠。
- `_apply_event_to_proxy` 是 module-level 自由函数，整个事件分发协议
  在不依赖 `gi.repository` / 真子进程的情况下可单测。

测试：
- `tests/scripts/test_thor_persistent_session.py` 17/17（旧契约不破）
- `tests/scripts/test_thor_persistent_session_multiprocess.py` 23/23（新）
  覆盖事件分发全 6 种事件、`wait_ready` 三个分支、connect raise 与
  cleanup、poll_errors 聚合、stop_episode 收 fragments、隔离性
  （一路 error 不阻塞另一路 ready）
- 总 40/40 通过；测试纯 Python，dev host 无 gi 也能跑

外场验证步骤：
1. `~/lerobot/tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1`
2. 重启 gateway 让代码生效
3. 前端 Connect → 期待 11 路 PLAYING 后 emit `Connected 11 pipelines …`；
   若有路失败，前端会拿到 `ERROR: persistent pipeline connect failed:
   [cam_NN] (具体原因)`，可按 sid recover 或单测重启该 worker

#### PR3 后续：connect 速度优化 + partial-failure 容忍

PR3 首版串行 spawn × wait_ready 让 Connect 总时间从 PR2 的 ~11s 拉长到
外场实测 ~55s（11 路 × ~5s/路）。原因是 Phase 1 严格串行：每路 worker
spawn → 等 PLAYING → stagger → 下一路 spawn。

调整为**滚动 spawn**：

* **Phase 1 — rolling spawn**：按 `spawn_stagger_s` 节奏依次 `proxy.spawn()`，
  但**不**等 PLAYING。每个 worker 一被父 spawn 出来就自己跑 Argus open；
  daemon 看到的仍是按 stagger 节奏到达的独立 client，过载保护不变。
* **Phase 2 — serial wait_ready**：父进程随后串行调 `wait_ready`，
  早 spawn 的 worker 这时往往已经 PLAYING，`ready_evt` 已经 set，
  `wait_ready` 立即返回。只有真正的 straggler 在这阶段阻塞。
* 总时长降到 ≈ N × stagger + max(单路 Argus open)。在 stagger=1s
  的设置下，11 路从 ~55s 压到外场预估 ~12–18s。

同时把 connect 改成**容忍 partial failure**：

* 单路 worker fail 不再立即 `raise`；失败的 sid 立即 disconnect 且从
  `_streams` 移除，剩余 sid 继续录制。
* 错误留在 `session.poll_errors()` 里。`thor_record.py` 在 connect 之后
  立刻 poll 一次，把 `WARNING: N stream(s) failed: cam_XX(...)` emit 给
  gateway，并再 emit 一行 `Cameras (active): cam_02, cam_04, ...` 让
  GUI 的相机列表跟实际录制路一致。
* 只有当**所有**路都失败时 `connect()` 才 raise（用 `failed on all N
  stream(s)` 措辞），thor_record 才走 ERROR + exit(1) 路径。
* 公共 API 新增 `pcs.active_sids` 只读 property。

测试矩阵：

* `tests/scripts/test_thor_persistent_session_multiprocess.py`
  - `test_connect_partial_failure_keeps_successful_proxies`（一路 error）
  - `test_connect_partial_failure_handles_timeout`（一路 timeout）
  - `test_connect_raises_when_all_proxies_fail`（全 fail 才 raise）
  - `test_active_sids_starts_empty`
  - `test_rolling_spawn_calls_spawn_for_all_streams_before_first_wait_ready`
    （验证 Phase 1 全部 spawn 完成才进 Phase 2）
* 合计 43/43 通过（17 旧 + 26 新）。

#### PR4：connect 末尾自动 restart 失败 sid

外场观察 11 路 lock + connect 仍只有 4/11 路落盘 mkv 的情况下，调查指向
两类失败：(1) 部分 sid 在第一次 worker spawn 时撞上 daemon 短暂 race，
recover_argus.sh 的"daemon restart + per-sid retry"模式总能救回；
(2) 极少数 sid 是 sensor lock 真挂或 PWM 触发错位，重启也救不回。

PR4 在 `connect()` 末尾加 **Phase 3 retry 一轮**：

* `poll_errors()` 拿到第一轮失败的 sid 列表。
* 对每个失败 sid 调一次 `restart_stream(sid)`：terminate 老 worker
  （daemon 端那条 socket 关闭、对应的 CaptureSession 被回收）→ spawn
  新 worker → `wait_ready`。这是 PR3 已经实现的 API，仅在这里被调用。
* 再 `poll_errors()`：survived 的 sid 错误清掉；仍 fail 的 sid 才计入
  partial-failure 分支，最终从 `_streams` 移除并出现在 `WARNING:` 行。
* **每路最多 1 次 retry**：永久故障的相机不会让 connect 陷入循环。

测试：

* `test_connect_retry_rescues_flaky_sid`：第一次 fail 第二次 OK 的 sid
  最终在 `active_sids` 里，`poll_errors()` 不返回它的错误。
* `test_connect_retry_drops_sid_that_keeps_failing`：双 fail 的 sid 被
  正确 drop，错误留在 poll_errors。
* `test_connect_retry_does_not_retry_more_than_once_per_sid`：断言总
  spawn 调用次数 = 2（原始 + 1 retry），不能更多。
* 合计 46/46（17 旧 + 29 新）。

#### PR4.1：retry 并行化（连续外场实测后）

外场 11 路 lock + 7 路第一轮 fail 的实况显示，PR4 的**串行**
restart_stream 把 Phase 3 撑到 ~25-30s，整次 Connect ~44s，不及预期。

时间归因（实测 08:58:46 那次）：
```
Phase 1 滚动 spawn (11 路)  ~10s   ✅
Phase 2 wait_ready          ~7s    ✅
Phase 3 串行 retry (7 路)   ~26s   🚧 瓶颈
合计                        ~44s
```

并行 retry 是直接修复：每个 worker 是独立子进程、独立 RPC socket，
N 路并发 terminate+respawn 在 nvargus-daemon 视角就是 N 个独立 client
断线+重连，daemon 的内部 serialization 是 per-socket 的，不会跨 socket
塞死。这正是 PR3 多进程隔离的设计目标。

实现：用 `concurrent.futures.ThreadPoolExecutor(max_workers=N)` 并行
对每个 fail sid 调 `restart_stream`：

```python
with ThreadPoolExecutor(max_workers=len(live_retry_sids),
                        thread_name_prefix="pcs-retry") as ex:
    list(ex.map(self.restart_stream, live_retry_sids))
```

线程安全性：`restart_stream` 内部对 `self._streams[sid]` 的写是
不同 sid → 写不同 dict key（CPython GIL atomic）；`mp.get_context` /
`mp.Process` / `mp.Queue` 都是 thread-safe；不需要额外加锁。

测试：

* `test_connect_retries_run_in_parallel_not_serially`：5 路同时 retry，
  每路 retry 后延迟 200ms 才推 `playing`；串行 lower bound = 5 × 200ms
  = 1.0s，断言实测 elapsed < 0.6 × lower_bound，强制证明 retry 是并行。
* `test_parallel_retry_drops_only_truly_dead_sids`：4 路同时 retry，
  even sid 成功 / odd sid 永久 fail，验证并发条件下不会"成功 sid 被
  错误判 fail"或"fail sid 错误留在 active"。
* 合计 48/48（17 旧 + 31 新）。

预期：
* Phase 3 从 26s 压到 max(单路 retry) ≈ 5s
* 总 Connect 时间 ~17-22s，接近 ~15s 目标
* 成功率与 PR4 相同（7/11 → 接近 9-11/11，hard fail 路仍 hard fail）

#### PR5：Connect 时自动 recover_argus.sh

外场使用模式：每次开机/上次崩溃后，操作员要先 ssh 到 thor 跑
`tools/thor/gmsl2/recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1`
让 nvargus-daemon 进入干净状态，再回到前端点 Connect。PR3 + PR4 +
PR4.1 把 daemon 恢复"健康初始状态后"的稳定性做到位了，但**没有解决
"daemon 当前就不健康"** 这件事 — 操作员仍要手工干预。

PR5 把这一步集成进 Connect 路径：

- thor_record.py 在 `PersistentCameraSession.connect()` 之后判断：
  * 抛出 `RuntimeError`（全 fail）→ 触发
  * `len(pcs.active_sids) < threshold_fraction × verified_count` → 触发
    （默认 60%，即 11 路里 < 7 路成功就 recover）
- 触发时：把当前 `pcs` 整个 `disconnect()` 释放掉所有 Argus session
  → `subprocess.run("bash recover_argus.sh --sdk <path>")` → 重建一个新
  pcs 重新 connect 一次。
- `max_attempts` 默认 1：永久故障的硬件不会让 connect 陷入循环。
- 失败路径：recover 自身 fail 或 retry 后仍 fail，emit 清晰的 `ERROR:`
  到前端并 `return 1`，操作员仍能手工 ssh 上去诊断。

YAML 新增可选块（`thor_gmsl2_11ch_example.yaml`）：

```yaml
auto_recover:
  enabled: true
  sdk_dir: ~/Desktop/SG16A_AGTH_G3Y_A1   # 未设则用 hardware_sync.sdk_dir
  threshold_fraction: 0.6
  max_attempts: 1
  timeout_s: 300
```

命令行 `--no-auto-recover` 关掉。

前端可见的新 emit 行：

* `Auto-recover: only K/N cameras up (threshold 60%); running recover_argus.sh (sdk=...)`
* `Auto-recover OK; retrying connect` 或 `Auto-recover failed: <tail>`
* （成功路径继续原有 `Connected K pipelines in X.Xs`）

测试：

* `tests/scripts/test_thor_record_auto_recover.py` 共 18 个测试覆盖：
  - `_auto_recover_from_yaml`：None / 空 dict / 完整覆写 / 非 dict 防御
  - `_resolve_recover_sdk_dir`：explicit 优先 / hardware_sync fallback /
    `~` 展开 / 相对路径解析
  - `_should_trigger_recovery`：< / = / > 阈值，零 expected 防御，
    零 active 触发
  - `_run_recover_argus`：脚本缺失、rc=0、rc≠0、timeout、未预期异常、
    tail 截断 400 字符
* 测试不真起 subprocess（runner 通过依赖注入 mock）
* 合计 66/66 通过（17 旧 + 31 multiprocess 新 + 18 auto-recover 新）

外场预期：
- 操作员只需点 Connect；recover_argus.sh 不再需要手工执行
- 健康 daemon 下零开销（不触发 recover）
- 不健康 daemon 下 connect 总时间 ≈ 第一次 connect (~22s) + recover.sh
  (~30-40s) + 第二次 connect (~15-22s) ≈ 80s 上下，但**操作员不用动手**
- 永久硬件故障（线松、PWM 不通）仍走原有 ERROR + exit 路径，便于诊断

#### PR6：前端日志丢失修复（recorder log ring buffer）

外场观察：connect 阶段后端 log 应该有 30+ 行
（spawned×11 + PLAYING/failed×11 + retry decision + restart_stream×N），
但前端 Live Record 面板里只看到 4-5 行，让操作员误以为程序卡死。

根因：gateway `_apply_recorder_output(state, line)` 把每行**覆盖式**
写入 `state.recording.lastOutput`。前端每秒拉一次 snapshot，只拿到
当前那一行 — 中间 N-1 行被悄无声息丢掉。前端旧版"useState 累积
lastOutput"逻辑无法补救：它只能累积自己看见的，看不见的累积不到。

PR6 改成**服务端 ring buffer**：

* `RecordingStatus.recentOutput: list[str]` 字段，cap 300 行（足够覆盖
  最差 connect 周期：11 路 × 6 events/路 + recover round + retry round）。
* `_apply_recorder_output` 每收到一行就 append 到 ring buffer，超过
  cap 时原地裁剪（`del recentOutput[:len-cap]`，O(1) 摊销）。
* `_connect_recorder` 在新建 recorder process 时清空 `recentOutput` 和
  `lastOutput`，避免上次 crash 的尾巴混进新 session。
* 前端 `RecordingStatus.recentOutput?: string[]`；`LiveRecordPage` 直接
  `const logLines = snapshot.recording.recentOutput ?? []` 渲染。
  旧的"useState/useRef/useEffect 累积 lastOutput"逻辑整段删掉。
* `RecorderLogStream`（auto-scroll-to-bottom + 用户上滚时不打断）保留。

效果：以后从 thor stderr 写出的每一行都会出现在前端 log 框里，操作员
能直接看到 Phase 1/2/3 / Auto-recover 的全部决策行，"看不到等于挂了"
的误判消失。

代码量：gateway.py +14 行 / +字段；types.ts +5 行；App.tsx 净 -27 行
（删掉了脆弱的客户端累积逻辑）。

### 已修复（2026-06-08）

* **`_warmup` 撑爆磁盘（gateway 长时间挂在 Connected 不录制）**
  - **表现**：周末过后 `outputs/datasets/<dataset>/_warmup` 涨到 120G，
    整盘被占满。`du -ah outputs/datasets` 显示体积几乎全在某个 dataset 的
    `_warmup` 子目录里。
  - **根因**：持久管线在 Connect 后一直 PLAYING（为秒开 episode + 实时预览），
    `splitmuxsink` 配的是 `max-size-time=0 max-size-bytes=0`（让单条 EPISODE
    保持一个 .mkv），因此**它从不自动切片**。idle 期间每路相机只往一个
    `cam_NN_warmup_00000.mkv` 无限追加；唯一的回收 `cleanup_warmup_files()`
    只在每条 episode 结束时跑，于是"挂着 armed 但不录"就让 `_warmup` 无界增长。
  - **修复**：worker 新增 `roll_warmup` 命令（仅 WARMUP 态 `force-IDR`+`split-now`，
    对 EPISODE 态是 no-op，绝不会切断录制中的分片）；`thor_record` 的 idle tick
    每 `dataset.warmup_roll_s`（默认 30s）`roll_warmup()` 后 `cleanup_warmup_files
    (keep_last_n=dataset.warmup_keep_last_n)`，把 `_warmup` 钳制在「N 路 ×
    keep_last_n × roll_s 码流」。`fragment_history` 同步改成 `deque(maxlen=256)`
    堵掉 idle 滚动放大的小泄漏。EPISODE 录制路径字节不变。
  - **运维提示**：**长时间不录制时按 Disconnect**，从源头停掉持久管线（省电、
    省 Argus/NVENC 占用，也不再写 `_warmup`）。清理盘上已有的旧 `_warmup`：
    先停 gateway/recorder，再
    `find outputs/datasets -maxdepth 2 -type d -name _warmup -exec rm -rf {} +`
    （`_warmup` 全是可丢弃的热身码流，删了不影响已录 episode）。

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
* **未跟踪的本地脚本**：`~/lerobot/run/` 下三个脚本不在 git 里 —— 是因为
  端口 / IP / venv 路径会因部署点而不同，复制本文 §7 的模板自己填。
* **`NvBufSurfaceFromFd Failed` (NVMM dmabuf race)**：多路 `nvarguscamerasrc`
  在并发 Argus open 阶段偶发，是 driver 层 dmabuf pool 状态竞争。当前
  PR4/PR4.1 的自动 retry 能救回大部分，PR5 的 auto-recover 兜底剩下的；
  彻底消除需 PR7（worker 两阶段 spawn：先 READY 再由父进程串行触发 PLAYING）
  或更新到修了这条 race 的 nvargus driver。

### Device Manager 实时预览管线（per-device 快照轮询 + 错峰 spawn）

Device Manager 页面把 11 路相机摆成 4×3 网格做"摆位监看"（调物体位置时
不用进录制就能看每路画面，hover 单格弹出该相机的 fps / latency / config
详情）。这套预览管线和录制管线**完全独立**，但踩的是同一类 NVMM/Argus 坑，
设计时把录制路径的两条教训直接搬了过来。

**为什么不是"每路一条 MJPEG 长连接"（最直觉的做法）**

最初版给每个格子挂一个 `<img src=.../camera.mjpeg>`，11 路画面只亮起
4–6 路，根因有两条、缺一不可：

1. **后端单进程槽互杀**：旧 `_serve_camera_preview_mjpeg` 用单个
   `state.camera_preview_process`，每来一个请求第一件事就 kill 上一个。
   网格同时发 11 个请求（gateway 是 `ThreadingHTTPServer`，每请求一线程），
   它们互相 kill，最后只剩最后启动的一路。
2. **浏览器 per-origin 连接上限**：MJPEG 是**长连接**，Chrome/Firefox 对
   同一 origin 只允许 ~6 条并发连接。11 条流 → 只有 6 条能建立，剩下 5 条
   在浏览器里**无限排队**，永远拿不到数据。这条是硬上限，跟 NVMM 无关。

**最终方案：服务端持久管线 + 短快照轮询**

- 后端每路相机一条**持久** gst pipeline（`nvarguscamerasrc ! nvvidconv !
  videorate ! jpegenc ! fdsink`，输出裸 JPEG 流），由一个 reader 线程按
  SOI(FFD8)/EOI(FFD9) 切帧，只在内存里留**最新一帧**
  （`state.camera_preview_frames[device_id]`）。
- 前端不再用长连接：每个格子按 ~5fps 轮询 `GET /api/device-preview/
  camera.jpg?key=cam_NN&t=<ts>`，返回内存里那张最新 JPEG。每个请求**极短**
  （命中缓存即返回），6 条连接轮着用 11 路绰绰有余，连接上限问题消失。
  轮询循环靠 `<img>` 的 onLoad/onError 自节流，慢相机不会堆叠请求。
- **错峰 spawn（NVMM race）**：`camera_preview_spawn_lock` + `_camera_preview_
  stagger_s()` 串行化各路 Argus open，两次 spawn 间隔
  `cameras.preview_spawn_stagger_s`（未设则回退 `min(spawn_stagger_s, 0.5)`）。
  这就是录制路径躲 `NvBufSurfaceFromFd Failed` 的同一招。冷启首帧前端会
  连续 503，前端按退避重试（~15s 内不报错），等 Argus open + AWB settle
  出第一帧后转入稳态轮询。
- **空闲自动回收（TTL）**：reader 线程发现某路 `_PREVIEW_IDLE_TTL_S`（5s）
  内没有任何快照轮询，就自行 terminate 该 pipeline、释放 Argus session。
  离开 Device Manager 页面 → 轮询停 → 5s 后所有预览管线自动收掉，不泄漏
  Argus。reader 用 `select` 唤醒，即便管线卡住也能按 TTL 退出。

**Connect 卡死修复（预览占用相机）**

外场现象：在 Device Manager 预览过之后点 Connect，录制一直卡住——预览
pipeline 还持着那几路 sensor 的 Argus session，recorder 的
`nvarguscamerasrc` open 撞上去就 hang（"相机还在被占用"）。

修复：`/api/handheld/record/connect` 在 spawn recorder **之前**先

1. 置 `state.camera_preview_suspended = True`（快照路由看到该标志立刻 409，
   防止并发轮询在空档里又把预览 respawn 回来）；
2. `_stop_all_camera_previews()` 杀掉所有预览 pipeline（在 state lock **之外**
   做，因为 `terminate()/wait()` 会阻塞）；
3. sleep 一个 stagger 间隔让 nvargus-daemon settle，再进 `_connect_recorder`。

`camera_preview_suspended` 在 recorder 退出（`_snapshot` 检测到 exit）、
`_stop_recorder`、以及 connect 失败的 except 分支里复位为 False，录制一结束
预览即可恢复。

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

录制时 `BoxClient` 以 2kHz wait (`record_poll_interval_s=0.0005`) 轮询 SDK 的
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

### 对齐时机：在录制时完成，不在导出阶段（本小节记录设计变更）

> ⚠️ 早期本节把「相机↔BOX 对齐」列为**导出阶段待实现**的 PTS 级后处理（依赖 BOX 上行修复 `rc=4`）。
> **该设计已被取代，勿再据此理解。** BOX 上行早已打通，对齐**在录制期完成、不在导出阶段**，而且分成
> 两个不同时机（别混为一谈）：
>
> - **相机之间（L0/L2）**：全程由 PWM 硬触发锁同一快门边沿（L0）；`argus_online_sync` 在**录制过程中、
>   encoder 之前**按 `sof_tsc_ns` full-cluster gate **逐帧在线**放行同一 SOF 簇（L2）。**不是 Stop 才做**——
>   Stop 只是收口并写 `online_sync_manifest.json`。
> - **BOX 传感器↔相机（L3b+）**：录制中只**缓存**原始样本；**Stop 写 parquet 那一步**才由录制器
>   （`thor_record.py` → `thor_lerobot_v3._build_episode_rows`）做 MCU 校准 + 以硬件 SOF 时刻
>   `sensor_timestamp_ns/1e9 − t0_mono` 逐传感器最近邻。**单条 session 的 parquet 落盘即已同步。**
>
> PTS 只用于离线数帧/QC，不作跨相机同步锚点。完整流程见 [`ts_sync.md`](ts_sync.md) §3/§5/§6。
> 下节的导出**不重做**这两套对齐，只复用录制期已同步的结果。

### 按任务合并导出（复用录制时对齐，不重做 box↔相机同步）

Dataset Export 页面的「Consolidate a Task」：选一个任务 → 把它所有
session（`<name>_<时间戳>/`，按时间戳排序）合并成**一个** LeRobot v3
数据集，写到独立 exports 根（默认 `outputs/exports/<name>/`，gateway
`--exports-root` 可改）。原始 session 不动、可重跑。

- 实现：`tools/thor/gmsl2/export_v3.py`（CLI + `export_task_to_v3()`），
  gateway 端点 `/api/tasks/export?id=<taskId>` spawn 它并把进度 stream 到
  `datasetExport` 状态。
- **只依赖 `pyarrow` + `gstreamer`**，跑在采集机最小 Python 环境里
  （**不**走 `LeRobotDataset` / torch / datasets / av —— 那些 Thor 上没有，
  见 §3）。v3 元数据用 pyarrow 手写（沿用 `Lr3Writer` 的方式）。
- **视频转 H.264 + CFR**：源 MKV 是 HEVC（`nvv4l2h265enc`）。每个 episode
  每路相机用 gst `nvv4l2decoder ! videorate ! framerate=fps ! nvv4l2h264enc`
  转成 **H.264** 的 per-episode mp4（`videos/<key>/chunk-000/file-<ep>.mp4`），
  dev 机无 nvv4l2 时回退 `ffmpeg -vf fps -c:v libx264`。
  - **为什么必须 CFR + H.264**：v3 loader 用**单个** timestamp 查询一个
    episode 的**所有**相机，源 MKV 的容器 PTS 是 `do-timestamp` 记的到达
    时刻、抖动且各路不一；`videorate` 强制重排到 `i/fps` 网格（PWM 本就是
    60fps 硬同步，这是物理正确的），使所有相机帧落在同一时钟、查询
    tolerance(1e-4) 内命中。H.264 则兼容 torchvision 兜底解码器。
- **逐相机帧数可能不同**（不均匀丢帧），而单 timestamp 要服务所有相机，
  所以 `n_frames = min(所有相机帧数, box 行数)`，多出的尾帧截掉
  （日志里 emit `camera frame counts vary ...; truncating to N`）。
  相机帧数从源 MKV 的 `extract_pts`（gst 读容器 PTS，不解码，Thor 可用）得到。
- box parquet（录制时 `Lr3Writer` 写的）按 `(episode_index, frame_index)`
  对齐到相机帧，截到较短一方写入 `observation.state` / `action`。
- **对齐来源（不是 export 现算的）**：相机之间由录制期 `online_sync_manifest.json` 的
  SOF full-cluster 保证（L2）；export 要求该 manifest 存在且校验各路帧数一致，
  否则跳过该 episode。box↔相机 state 由录制器已做 L3b+ 同步，export 只按
  `(episode_index, frame_index)` 复用、**不重算**，截到较短一方写入
  `observation.state` / `action`。export 自身只负责把视频 CFR 重排到 `i/fps`
  网格（见上）与合并，不做 PTS 级 wall-clock 对齐（PTS 仅用于数帧）。
  `meta/export_sources.json` 记录每个全局 episode 来自哪个 session 便于溯源。
- 输出根与原始 `outputs/datasets/` 分开，避免被 Task 进度计数当成 session
  重复统计。
- 已在 Thor 实测：6 episodes / 2 sessions / 10 cam @ 1920×1080 @ 60fps，
  产物在 full-deps 机上经 `LeRobotDataset` 加载 + 逐帧解码（首/中/末帧 ×
  全 10 路）通过。

#### 设计取舍记录（为什么是 pyarrow+gst per-episode，而不是官方 API / 单文件拼接）

> 背景前提：**采集机是最小 Python 环境**（`pyarrow + pyyaml + box_sdk`，
> 外加系统 `gstreamer`；**无 torch / torchvision / datasets / av / ffmpeg**），
> 这是 §3 故意的"采集机最小化、训练依赖单独 venv 解耦"的结果。下面所有取舍
> 都以"导出必须能在这个最小环境里原地跑"为约束。已实测确认 Thor 缺
> torch/torchvision/datasets/av/huggingface_hub，仅有 pandas/PIL/numpy/pyarrow。

**为什么不直接在 Thor 用官方 `LeRobotDataset` API（最省事、布局最标准）**
- `from lerobot.datasets.lerobot_dataset import LeRobotDataset` 顶层硬 import
  `torch`/`torch.utils`/`datasets`/`pandas`/`PIL`/`huggingface_hub`，且其
  `video_utils` 顶层 import `av`/`torch`/`torchvision`——少一个这个类都 import 不了。
- 真正的卡点是 **torch/torchvision 在 Jetson**：要 NVIDIA 专供、与 JetPack/CUDA
  版本死锁的 aarch64 wheel（GB 级、易装错、升级 JetPack 要重配），违背采集机最小化。
- 即便装上，**它也不会用 Jetson 硬件编解码**：`encode_video_frames` 走 `av`
  软编（v3 默认 `libsvtav1`/AV1，ARM 上个位数 fps），`add_frame` 还逐帧落 PNG
  再读回。实测对比：方案二硬件通路 **92s**；方案一软编 + 逐帧 PNG 预计**慢约一个
  数量级**。讽刺点：编码本身只用 `av` 不用 torch，torch 纯粹因为类/`add_frame`
  跟张量耦合——为写视频得拖进整个训练栈。
- 结论：官方 API 适合**训练机**（GPU/强 x86，软编也快、布局标准），不适合采集机。

**为什么不拼接成单文件（官方 v3 那种一相机一文件 + `from/to_timestamp` 切片）**
- 在最小 gst 环境里**帧级精确拼接做不干净**，实测三种都在 episode 衔接处引入
  ±1 帧 PTS 缺口（步长 2/60）：无损拼编码后的 mp4（2 个缺口）、解码层拼+末端
  videorate（多 1 帧 + 1 缺口）。而 LeRobot 解码按 `timestamp` 用 **1e-4s** 容差
  找帧，边界帧一超容差就 `FrameTimestampError`。
- per-episode 单段 CFR 文件实测 `PTS == i/60` 偏差 **0.0**、全帧解码通过。官方那种
  拼接是 `LeRobotDataset` 编码器在全依赖机上逐帧精确写时间戳做到的，gst `concat`
  元件给不了这个精度。
- 代价仅是每相机文件数 = episode 数（有上界），且仍是合法可加载的 v3 布局。

**最终方案（方案二）的关键技术选择**
- **pyarrow 手写 v3 meta**（复用 `thor_lerobot_v3` 的 pyarrow-only helpers），
  不依赖 torch/datasets。
- **gst `nvv4l2` 硬件转码**：源 HEVC → **H.264**（torchvision 兜底解码器偏 h264）。
- **CFR `videorate` 把 PTS 重排到 `i/fps`**：v3 loader 用单个 timestamp 查一个
  episode 的**所有**相机，源容器 PTS（`do-timestamp` 记的到达时刻）抖动且各路不一，
  必须落到同一网格；PWM 本就是 60fps 硬同步，这是物理正确的（实测偏差 0）。
- **逐相机帧数取 min**（不均匀丢帧），单 timestamp 服务所有相机。
- **相机并行转码**（默认 `--jobs 8`）：Thor 实测 336s → **92s（~3.6×）**；没到 8×
  是因 videorate 的 NVMM↔CPU 拷贝卡在 CPU/带宽，不是线性扩展。

**何时重新考虑**：若"官方单文件布局"成为硬需求，**优先走训练机导出**（把 raw
session rsync 过去用官方 API），而**不是**在采集机装 torch——后者又慢又重又违背
最小化原则。


### 2026-06-03 实机排障记录：Connect/Preview/Argus 资源状态

背景：Thor 11 路 GMSL2 在前端点击 Connect 后，早期日志表现为多路
`NvBufSurfaceFromFd Failed`、`dmabuf_fd -1`、`failed to reach PLAYING`，
随后 Device Manager 11 个 preview 窗口显示 `running` 但无画面；点击
StartEpisode 时 recorder 没有进入录制，前端看起来像卡死。

本轮确认的三个独立问题：

1. **preview 解码分支会耗尽 VIC/NVDEC 资源**
   - 旧 recorder-owned preview 从编码后的 H26x tee 出来再 `nvv4l2decoder`
     回 JPEG。11 路 preview 等于额外开 11 路 `prevdec_*`。
   - 内核日志证据：`tegra-vic ... prevdec_*:src: all memory contexts are busy`。
   - 修复：preview tee 改到 encoder 前的 raw NVMM 分支，走
     `nvvidconv -> I420 -> videorate -> jpegenc -> appsink`，不再为 preview
     创建 11 个硬解码器。

2. **Device Manager idle preview 会偷偷启动独立 `nvarguscamerasrc`**
   - gateway idle 状态下的 `/api/device-preview/camera.jpg` 原本会为每个相机
     启动独立 `gst-launch-1.0 nvarguscamerasrc ... jpegenc`，这会和 recorder
     抢同一批 Argus sensor，并可能留下 stuck/leaked session。
   - 证据：idle preview 请求后 `ps` 可见 `gst-launch-1.0 ... nvarguscamerasrc
     sensor-id=...`，`fuser/lsof` 显示 nvargus-daemon 持有 `/dev/video*`。
   - 修复：GMSL2 配置下 Device Manager 永远只读 recorder-owned
     `/dev/shm/lerobot_preview/cam_XX.jpg`。recorder 未连接或还没有 JPEG 时返回
     `503 {"error":"no recorder preview frame yet"}`，**不再启动外部 Argus client**。
   - 验证：idle 请求 `cam_00/cam_10/cam_11/cam_14/cam_15` 均返回 503；之后
     `ps` 无 `gst-launch-1.0.*nvarguscamerasrc`，`fuser /dev/video0/10/11/14/15`
     无占用。

3. **recorder 内部 auto-recover 会误杀 gateway/自身**
   - `recover_argus.sh` 设计给人工恢复用，默认会：
     `pkill -TERM -f 'python.*tools\.thor\.gmsl2\.thor_record'` 和
     `pkill -TERM -f 'python.*tools\.data_collection_gui\.gateway'`。
   - `thor_record.py` 内部 auto-recover 调用该脚本时没有带 `--skip-kill`，所以
     partial/fail 后会把 recorder 自己和 GUI gateway 一起杀掉。前端表现就是
     StartEpisode/Connect 后 snapshot connection refused 或 UI 卡住。
   - 修复：`thor_record._run_recover_argus()` 调用
     `bash recover_argus.sh --sdk <sdk> --skip-kill`。人工 SSH 执行 recover 仍保留
     默认 kill stale process 行为。
   - 回归测试：`tests/scripts/test_thor_record_auto_recover.py` 覆盖 recover 命令
     必须包含 `--skip-kill`。

本轮保留/新增的稳定性策略：

- Connect 采用保守模型：`spawn one -> wait PLAYING -> stable window -> next`，
  retry 顺序执行，不再 parallel retry。这样避免 11 路 Argus/NVMM/NVENC 启动窗口重叠。
- `PLAYING` 后仍要有 `connect_stable_s` 稳定窗口。实机日志证明某些 sid 会先
  `PLAYING`，随后立刻 EOS/TIMEOUT；stable window 可以把这种假成功抓出来并进入 retry。
- recorder-owned preview 必须在全部 active recording streams 通过 connect/stable 后再启用。
- gateway/recorder stdout 固定落盘：
  `outputs/logs/data_collection_gui/gateway_*.log` 和
  `outputs/logs/data_collection_gui/recorder_*.log`，以后排查不要依赖 UI 截图。
- Connect 新增全局 wall-clock deadline：YAML 字段
  `sensors.cameras.connect_timeout_s` 默认 120s，覆盖首轮 connect、sequential
  retry 以及内部 auto-recover 的剩余时间预算。超过预算时 recorder 会主动 teardown
  当前 persistent session，并输出明确协议行：
  `ERROR: persistent pipeline connect failed: connect exceeded global deadline ...`。
- gateway 退出摘要不再直接使用最后一行 stdout。非零退出时优先从
  `recentOutput` 里选择 `ERROR:` 行，其次选择 `NvBufSurfaceFromFd Failed`、
  `dmabuf_fd -1`、`Failed to create CaptureSession`、`TIMEOUT`、`bus EOS` 等
  Argus/GStreamer 关键故障行，避免 UI 只显示
  `CONSUMER: Waiting until producer is connected...` 这类无诊断价值的尾行。

2026-06-03 06:00 UTC 后的关键实机事实：

- 手工 clean recover 后，lock check 仍为 11 路：
  `0,2,3,4,5,7,9,10,11,14,15`。
- 但 `recover_argus.sh` 的**单路 probe**第一轮仍出现底层失败：
  `sid=10/11/14/15` `PROBE_FAIL rc=124`，包含
  `NvBufSurfaceFromFd Failed` / `Argus Error Status UNAVAILABLE`。
- 同一个 recover 脚本重启 nvargus-daemon 后 retry 这四路，全部 `PROBE_OK`。
- 这说明剩余问题已经不是前端 preview 或 recorder 并发独有问题，而是
  Argus/driver/sensor stream-on 状态机会进入可恢复坏状态。
- 后续 Connect 期间内核继续出现硬件层错误，例如：
  - `ar0234c 20-0023: i2c write failed, 0x3060 = 00`
  - `ar0234c 20-0023: i2c write failed, 0x3012 = 02`
  - `ar0234c 20-0023: Error turning on streaming`
  - `ar0234c 17-0022: Error turning on streaming`
  - Argus 日志里对应 `Sensor GUID 16/17 is in error state`、`waitForIdle() timed out`、
    `Forced destruction will now proceed, which may leave the libargus server in a bad state`。

当前根因分层结论：

- 已修复的软件层问题：
  - preview 不再消耗硬解码资源；
  - Device Manager idle preview 不再抢 Argus；
  - recorder 内部 auto-recover 不再杀 gateway/自身；
  - Connect 不再用 parallel retry 制造第二波 Argus 启动风暴。
- 仍存在的底层问题：clean recover 后单路 probe 也会出现 stream-on/I2C/Argus
  timeout，说明至少部分链路在驱动/serializer-deserializer/sensor/CSI/RCE 层不稳定。
  Python/GStreamer 编排只能做到隔离、降载、检测、recover，不能从根上修复这类
  `ar0234c ... i2c write failed`。

后续排查建议：

1. 在完全无 gateway/recorder/preview 进程时反复运行：
   `recover_argus.sh --sdk ~/Desktop/SG16A_AGTH_G3Y_A1`，统计第一轮 probe 失败 sid
   和 retry 是否总能救回。若同一 bus/address 反复失败，优先查线束/供电/serializer/deserializer。
2. 保留最近三类日志给驱动/供应商：
   - `outputs/logs/data_collection_gui/recorder_*.log`
   - `journalctl -u nvargus-daemon --since <connect-start>`
   - `journalctl -k --since <connect-start>` 中 `ar0234|max96726|nvcsi|vi|timeout|i2c|streaming` 行
3. 若 UI 需要更好的失败体验，下一步工程项是给 Connect 加全局 wall-clock deadline：
   超过预算则主动终止 recorder、提示需要 recover，而不是让 operator 等单路 Argus
   timeout 串行耗尽。

### 同步级别路线图

| 级别 | 精度 | 状态 | 说明 |
| --- | --- | --- | --- |
| L0 硬同步（相机间） | <1μs | ✅ 已工作 | PWM slave mode，11 路帧对齐 |
| L1 软同步元数据 | ±25ms | ✅ 已实现 | sync_reference in meta.json |
| L2 导出时对齐 | ±25ms | 🔲 待实现 | PTS 级对齐待 BOX 上行通后实现（best-effort 合并导出已可用，见上） |
| L3a 录制时高频对齐 | ±2.5ms~±10ms | ✅ 已实现 | 500Hz poll + per-sensor MCU 时间戳去重 + 逐传感器最近邻插值 |
| L3b 增强对齐 | ±0.5~1ms | ✅ 已实现 | MKV PTS 提取 + MCU↔Host 时钟线性回归 |
| L4 硬件级全同步 | <1μs | 🔲 需硬件改 | BOX MCU 也由 PWM 触发 |


备注:
thor的isp文件路径/var/nvidia/nvcam/settings/camera_overrides.isp(从驱动sdk拷过去)