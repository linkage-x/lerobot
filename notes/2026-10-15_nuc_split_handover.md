# FR3 机器人端 / 推理端拆分：交接文档

日期：2026-10-15（写于 16:55 左右）
相关脚本和证据：`notes/2026-10-15_nuc_split/`

## 1. 背景

- **tele**（旧工作站，192.168.100.155）硬盘坏了，数据无法恢复。
- **alpha**（hph@192.168.1.155，i9-14900K，主板 MSI MS-7E07）是临时借来顶替的，但 CPU 散热有问题：
  - 实测空闲 1.4 W 时就有 74 °C；
  - 8 核满载 0.2 秒冲到 100 °C；
  - Python 会随机抛出不可能出现的异常。
- **alpha 不能用于真机。** 系统日志显示，问题从 2026-09-08 起就存在，比我们第一次登录（10-15 10:59）早了一个多月。证据见 `notes/2026-10-15_nuc_split/alpha_pre_handover_kernel_errors.txt`（111 行，内容包括内核 Oops、GPF、intel_powerclamp 和随机段错误）。
  - 我们对 alpha 只做过软件改动：
    - 装了内核 7.0.0-38 和 6.8.0-147；
    - NVIDIA 用户态库升到 595.99.02；
    - libfranka 装进 /usr/local，建了 .venv-fr3；
    - 改了 NM 的 `ros` 配置；
    - 做过一次强制断电重启。
  - 强制重启之后 Wi-Fi 网卡不再被识别。要如实告诉负责人。
  - 16:3x 时 alpha 已经 ping 不通，应该是被关机或收走了。
- **决定的方案**：
  - 机器人端（FR3、两台 RealSense、夹爪、录制）放到 **NUC** 上；
  - 策略推理放到**开发机的 RTX 4090 D** 上；
  - 两边走有线局域网通信。
- **硬性要求**：tele 修好后，原来的单机链路（`bash run/deploy.sh workstation`）必须照常可用。所以远程推理只能做成**可选开关**，不加这个参数时代码路径和现在完全一样。

## 2. 三台机器的现状

| 机器 | 地址 | 角色 | 状态 |
|---|---|---|---|
| NUC（delta-NUC11BTMi9） | `delta@192.168.1.156`（备用：Wi-Fi 192.168.93.93） | 机器人端 | 环境已就绪，相机已验证 |
| 开发机（hanyu） | 192.168.1.179（网卡 enp3s0，1 Gb） | 推理端 + 训练 | venv 已就绪，ckpt 还没在 GPU 上加载验证 |
| alpha | hph@192.168.1.155 | 不要再用于真机 | 已离线 |
| FR3 控制柜 | 192.168.1.206 | — | 从 NUC 能 ping 通，平均 0.34 ms |

### 2.1 NUC 体检结果（全部实测）

- **硬件**：NUC 11 Extreme，i9-11900KB（8 核 16 线程），30 GB 内存。系统 Ubuntu 24.04，内核 **6.8.1-1056-realtime**。
- **实时性**：用户 delta 在 `realtime` 组里，rtprio 上限 99。1 kHz SCHED_FIFO 测 10 秒：
  - 空闲时最坏 41 µs，16 线程满载时最坏 23 µs；
  - 超过 1 ms 的次数为 0。
- **散热**：空闲中位 48 °C；16 线程满载冲到 93–97 °C，停下后 1 秒内回到 46 °C。散热通路正常。PL1/PL2 是出厂的 65/109 W。
- **网络**：
  - 到 FR3 ping 平均 0.34 ms、最大 0.65 ms，0 丢包；
  - 到开发机 TCP 吞吐 0.94 Gbit/s，瓶颈是开发机的千兆网卡。
- **相机**（librealsense 读到的序列号）：
  - 侧面 D435I `243122071795`，腕部 D405 `315122271876`，就是 tele 原来那两台。
  - 两台同时开 640×480@60，测 3 轮，每轮 10 秒：都在 59.7–59.8 fps，没有进程卡在 D 状态。
  - 有一条警告：D405 时间戳“未来 0.1 ms”，代码已自动回退处理，不影响使用。
  - **坑**：`/sys/bus/usb/devices/*/serial` 里的 USB 描述符序列号（251843061332 / 242623071642）**不是** librealsense 的设备序列号，不要拿它去对配置。
- **磁盘**：`/home` 已用 97%，只剩约 7 GB。`~/data` 里有 84 GB 别人的数据，**用户要求不要动**。我们的东西全部放在 `/data`（剩余约 440 GB）。
- **USB 拓扑**：两台 RealSense 都挂在 PCH 那个 xHCI 控制器（Bus 004，20 Gbps）上，D405 中间隔了两级 hub。目前 60 fps 正常。雷电 4 那个控制器（Bus 001/002）空着，带宽不够时可以把一台相机挪过去。
- **串口**：CH340 对应 `/dev/ttyUSB0`，Prolific 对应 `/dev/ttyACM0`。~~还没确认哪个是我们的夹爪。~~ 18:3x 已确认是 ttyUSB0，现在叫 `/dev/pika_gripper`（见第 4 节第 6 项）。

## 3. 已完成的操作（含回滚方式）

### NUC（delta@192.168.1.156）

1. **免密 sudo**：用户自己开的。
2. **有线网络**：NM 配置 `ros` 改为按 **MAC 1C:69:7A:D9:89:EF** 绑定（不再绑接口名）。
   - 地址：静态 **192.168.1.156/24**，网关和 DNS 都是 192.168.1.1，DAD 3 秒，autoconnect-priority 100。
   - 选这个地址的依据：扫过 192.168.1.0/24，.154 和 .156–.169 都没人用；.155 是 alpha 的保留地址。
   - 风险：DHCP 地址池覆盖了大半个网段，以后有可能被别的机器分配到同一个地址。最好在路由器（192.168.1.1）上做静态保留。
   - 回滚：`sudo nmcli con modify ros ipv4.method auto ipv4.addresses "" ipv4.gateway "" ipv4.dns ""`
3. **apt 安装的包**：
   - `setup_workstation_teleop_env.sh` 列出的那些（build-essential、cmake、eigen、poco、urdfdom 等）；
   - git、pkg-config、libssl-dev、boost、libfmt-dev；
   - **python3.12-dev**（没有它，`evdev` 编译不过）、python3.12-venv。
4. **uv**：0.12.5，装在 `~/.local/bin/uv`，是从开发机拷过去的。
5. **libfranka 0.15.0**：针对 venv 里 cmeel 的 pinocchio 编译，装进 `/usr/local`。脚本是 `notes/2026-10-15_nuc_split/build_nuc_env.sh`，和 alpha 上用的 `build_ws_env.sh` 只差路径。
6. **代码**：在 `/data/lerobot`，用 `REMOTE_DIR=/data/lerobot bash run/sync_to_target.sh delta@192.168.1.156` 同步。
7. **venv**：`/data/lerobot/.venv-fr3`。
   - 导入检查全部通过：panda_py、pika、pinocchio 3.4.0、placo、ruckig 0.15.3、peft 0.21.2。
   - uv 缓存放在 `/data/.cache/uv`，**之后跑 uv 时要带上 `UV_CACHE_DIR=/data/.cache/uv`**，否则会写爆 /home。
   - 构建日志：`/data/lerobot/run/logs/build_nuc_env.log`。
8. **ckpt**：完整拷贝到 `/data/lerobot/outputs/train/fr3_spacemouse-insert__delta_ee_from_prev_cmd__L4_full48_plus_dagger108_plus_funnel105_030000__pi05__20260930_053656/`，共 1.9 GB。这是 LoRA ckpt，不含 base 权重。
9. **meta stub**：在 `/data/lerobot/outputs/exports/training_views/fr3_spacemouse-insert__delta_ee_from_prev_cmd__L4_full48_plus_dagger108_plus_funnel105_030000/`。
   - 是用 `build_view_meta_stub.py` 重建的。
   - 用运行时自己的函数校验过：起始位姿和 10-08 日志最大偏差 2.5e-7，裁剪框、提示词、fps 30 都一致。
   - **这是 1 个 episode 的占位数据，只能用于 rollout，绝对不能拿来训练。**

### 开发机（hanyu@192.168.1.179）

1. **ufw**：新增一条规则，`allow from 192.168.1.156 to any port 8790 proto tcp`，注释是 `lerobot policy server <- NUC delta`。
   - 实测从 NUC 到 8790 的吞吐是 0.94 Gbit/s。
   - 回滚：`sudo ufw status numbered`，找到这条规则再 `sudo ufw delete <编号>`。
2. **venv**：`/home/hanyu/Codes/lerobot_workstation/.venv-fr3`。
   - 装了 extras：fr3-workstation-teleop、fr3-train、kinematics。**没装 fr3-host**，因为开发机不直接连机器人，也不需要 libfranka。
   - 版本：torch 2.10.0+cu128（CUDA 可用）、peft 0.21.2、transformers 5.19.0。
   - 日志：`run/logs/devbox_venv_sync.log`。
3. **基础模型**：开发机的 HF 缓存里已经有了，并且**和 Hugging Face 官方 API 核对过**：
   - `lerobot/pi05_base`：model.safetensors 的 sha256 是 `0eb11ca9…59b0f`，和官方 main 一致，大小 14.47 GB。
   - `google/paligemma-3b-pt-224`：revision `35e4f46`，和官方 main 一致；3 个分片和 tokenizer 的 sha256 都对得上。
   - snapshot 目录里有一个 0 字节的 `chat_template.jinja`，官方仓库里没有这个文件。pi05 用不到它，可以不管。
4. **CPU 体检**：开发机也是 i9-14900K，但主板不同（七彩虹 Z790D5 FROZEN），散热正常：
   - 空闲中位 44 °C；
   - 8 核满载最高 88 °C，2 秒内回到 45 °C；
   - PL1 380 W，没被压。

## 4. 未完成 / 卡住的事项（按顺序做）

> **17:30 进展（接手会话）**：第 1–4 项已完成。
> - 第 1 项：软链已建，`/home/tele/Models/pi05_base/model.safetensors` → HF 缓存 blob `0eb11ca9…59b0f`（root 所有）。
> - 第 2 项：开发机 meta stub 已建（`STUB_OK`）。
> - 第 4 项：`resolve_dataset_root` 已改为：train_config 里的绝对路径不存在、且没有显式 `--dataset-root` 时，退回到 `_REPO_ROOT/outputs/...`；路径存在或显式指定时行为不变。新增 4 个测试，`tests/scripts/test_fr3_act_infer_real.py` 126 个全过。
> - 第 3 项：`verify_stub.py` 在开发机上不带 `--dataset-root` 跑通（`VERIFY_OK`，起始位姿偏差 2.5e-7，`load_policy_stack` 68 s）。`ckpt_load_probe_devbox.py`：加载 61 s，显存 9.5 GB，chunk (1,50,5)，延迟首次 0.434 s、稳态 0.167–0.177 s（不含 RTC 引导，纯 `predict_action_chunk`）。
> - 开发机 venv 补装了 `pytest 8.4.2`、`pytest-timeout 2.4.0`（按 pyproject 的 test extra 版本范围）。
> - 交接时同事会话里有 4 个后台等待循环（远程 `pgrep -f` 匹配到自身 `bash -c`，永远不会结束），已确认消失；它们等的 setup/build 早已完成。
>
> **第 5 项拆分代码：核心已完成并验证（18:xx）**，deploy/gateway 部分未做。
> - 新文件：`tools/fr3/policy_remote.py`（协议、`RemotePolicyClient`、`serve_connection`）、`tools/fr3/fr3_policy_server.py`（开发机上跑的服务）、`tests/scripts/test_fr3_policy_remote.py`（12 个测试）。
> - `fr3_act_infer_real_runtime.py` 的改动：
>   - 新参数 `--policy-server HOST[:PORT]`，默认不开；
>   - `predict_action_chunk_for_rollout` 开头：`policy` 是 `RemotePolicyClient` 时转发，否则代码路径不变；
>   - 从 `load_policy_stack` 里抽出 `prepare_rollout_policy_config`（行为不变），新增 `load_remote_policy_stack`；
>   - 远程模式下非 RTC 直接 SystemExit。
> - 和原设计的差异：**NUC 上不加载 pre/post processor**，改用 `ServerSideProcessor` 占位（`reset()` 空操作，被调用就报错）。原因：pi05 的 preprocessor 会从 HF Hub 拉 PaliGemma tokenizer，NUC 连不上 Hub，会卡在 SYN-SENT。RTC 路径在本地只对它们调用 `reset()`。
> - 序列化用 `torch.save` + `torch.load(weights_only=True)`，不用 pickle。握手时比对指纹（ckpt 从 `outputs/` 起的路径、chunk_size、n_action_steps、use_amp、RTC 参数），不一致就拒绝连接。
> - 验证：
>   - 单元测试：`test_fr3_policy_remote.py` 12 个、`test_fr3_act_infer_real.py` 126 个，以及其他 10 个相关文件，逐个文件跑，全过。
>   - 真模型一致性（`notes/2026-10-15_nuc_split/remote_consistency_probe.py`）：同一进程内回环，固定种子，本进程和远程的 chunk **逐位相同**（bootstrap 和带 RTC 前缀两种情况）。
>   - 跨机器（`remote_client_probe.py`，NUC → 开发机:8790，不碰硬件）：每个请求约 0.97 MB，稳态总延迟中位 0.227 s、最大 0.241 s，其中网络加编解码约 11 ms；首个请求 0.463 s。
> - 服务端启动命令：`PYTHONPATH=src:. .venv-fr3/bin/python -u tools/fr3/fr3_policy_server.py --checkpoint <ckpt>/checkpoints/030000 [--port 8790] [与 NUC 端相同的 --rtc-* 等参数]`
> - 机器人端只需在原命令上加 `--policy-server 192.168.1.179`。NUC 上建议加 `HF_HUB_OFFLINE=1`。
> - 代码已经用 `sync_to_target.sh` 同步到 NUC 的 `/data/lerobot`。
>
> **18:xx 部署方案已实现并验证（用户要求：policy-server 前端可选）。** 上面“服务端启动命令”一条已作废。
> - alpha 的 10 个文件改动已按用户要求撤销（`git checkout`），`workstation` 目标恢复成 tele 的 192.168.100.155。
> - 设计依据（从物理约束推）：
>   - 接线在哪，哪台机器就跑硬件相关的一切：NUC 跑 gateway、录制、遥操作、rollout 进程和 1 kHz 控制环。
>   - GPU 在哪，前向推理就在哪：开发机。同一张卡还要训练，所以模型不能一直占着显存。
>   - 权重在训练出来的机器上，NUC 只需要 config 和 view 元数据。
>   - 操作者每次选的是“哪个 checkpoint”和“在哪算”，不应该再手动去开一个参数要和 NUC 对齐的服务端。
> - 实现：
>   - policy server 改成**常驻、按需加载**（协议 v2）：hello 里带加载规格（ckpt 和 view 的 `outputs/` 相对路径 + RTC 等覆盖参数），服务端没加载就加载（pi05 约 68 s），一样就复用；指纹仍然双向核对。同一时间只允许一个 rollout 持有，第二个直接拒绝。空闲 900 s 自动卸载，给训练腾显存。`status`/`unload` 不需要握手，gateway 用来看状态，不会抢占策略。
>   - `run/policy_server.sh start|restart|stop|status`：在 GPU 机上启停，按解释器 argv 找进程，不会误匹配。
>   - `bash run/deploy.sh nuc`：同步代码到 NUC → 重启开发机上的 policy server（保证两边协议一致）→ 重启 NUC 上的 gateway，带 `--policy-server 192.168.1.179:8790` → 本机起前端。policy server 跑的是开发机自己的 checkout，**不会**从别处 rsync 过去（防止 `--delete` 冲掉开发树）。
>   - Rollout 页面 Mode 卡片里，Start 上方有 “Policy compute” 选项：本机 / 每个配置的 server，实时显示它加载了什么、谁在用、显存。没有 GPU 的机器默认选 server；会提示“Start 会先加载这个 ckpt（约 1 分钟）”，有 “Free GPU” 按钮。选择会被 gateway 按白名单校验，并记入 last-params。
>   - launcher 新增 `FR3_POLICY_SERVER`，留空时行为不变；gateway 每次都显式设置或清除它，不从 shell 继承。
> - 验证：`deploy.sh nuc --no-frontend` 跑通；`/api/policy-servers` 从 NUC 能看到开发机状态；通过 gateway 起 `env` 模式（不碰硬件），日志里有 `policy_server=192.168.1.179:8790`。NUC 探针：冷启动按需加载 67.9 s，第二次复用，稳态 0.224–0.227 s。测试逐个文件跑：policy_remote 18、policy_servers 7、rollout 255、gateway 241、act_infer_real 126，前端 304 + tsc，全过。
> - 当前状态：开发机 policy server 在跑（`run/policy_server.sh status`），NUC gateway 在跑（http://192.168.1.156:8765）。

1. **【需要用户操作】开发机上的 pi05_base 路径。**
   - ckpt 的 `adapter_config.json` 里，base 写死为 `/home/tele/Models/pi05_base`。加载时 `make_policy` 只从这个目录读 `model.safetensors`，config 由调用方传入。
   - 开发机上 `/home/tele/Models` 属于 root 而且是空的。我想用 sudo 在里面建软链，被 Claude Code 的权限分类器拦下了，因为是往别的用户目录写。
   - 需要用户自己执行：
     ```bash
     S=$(readlink -f ~/.cache/huggingface/hub/models--lerobot--pi05_base/snapshots/7de663972b7817d2c4cf2d84c821153dfea772e9/model.safetensors)
     sudo mkdir -p /home/tele/Models/pi05_base && sudo ln -sfn "$S" /home/tele/Models/pi05_base/model.safetensors
     ```
   - 另一种做法：在代码里加一个 base 路径的重映射。缺点是要改 `src/lerobot`，会给上游同步添麻烦。
2. **开发机的 meta stub**：还没建。命令是 `.venv-fr3/bin/python notes/2026-10-15_nuc_split/build_view_meta_stub.py "$PWD"`，要用 `PYTHONPATH=src:.`。
3. **在开发机 GPU 上验证 ckpt 加载和推理延迟**：`PYTHONPATH=src:. .venv-fr3/bin/python notes/2026-10-15_nuc_split/ckpt_load_probe_devbox.py <ckpt>/checkpoints/030000/pretrained_model [base_dir]`。
   - 第二个参数可以直接指向 HF snapshot 目录，这样不用等第 1 步也能先验证。
   - 参考值：tele 在 9-30 测到 rtc_lat 420–660 ms。
4. **路径可移植性**：`train_config.json` 里的 `dataset.root` 是 `/home/hph/Code/lerobot/outputs/...` 这样的绝对路径。`resolve_dataset_root` → `_resolve_repo_path` 遇到绝对路径会原样返回，所以在 NUC 和开发机上都会找不到。
   - 计划：绝对路径不存在时，退回到 `_REPO_ROOT / "outputs/..."`，也就是取 `outputs/` 之后的部分；路径存在时行为不变。
   - 现在临时可以用运行时的 `--dataset-root` 参数指定。
5. **拆分代码**：还没开始写。设计要点：
   - **唯一的切入点**是 `tools/fr3/fr3_act_infer_real_runtime.py` 里的 `predict_action_chunk_for_rollout`（约 4528 行）。调用点有两个：异步规划线程 `AsyncActionChunkPlanner`（约 4682 行）和同步启动路径（约 5793 行）。
   - 新增 `--policy-server host:port` 参数：
     - **不传**：照旧在本进程加载和推理，单机链路完全不动；
     - **传了**：NUC 不加载模型权重，只加载 config 和 pre/post processor（CPU 上）。processor 有用处，比如 3499/3521 行用 `deepcopy(postprocessor)` 做预览。
     - `policy.reset()` 改成 RPC。
     - 服务端在开发机上调用**同一个** `predict_action_chunk_for_rollout`，返回 `original_actions`（RTC 前缀要用的原始动作）和 `processed_actions`。
   - RTC 的延迟统计 `latency_s` 是在调用方测的，网络耗时会自动算进去，不用改。
   - 先只支持 RTC 路径，也就是 pi05。非 RTC 的 `predict_action`（ACT 时间集成）在远程模式下直接报错拒绝。
   - 服务端口 **8790**（防火墙已经放行）。消息里只传裁剪缩放后的图像，原始视频流不过网线；录制直接写 NUC 本地的 `/data`。
   - `run/deploy.sh` 和 `run/sync_to_target.sh` 新增 `nuc` 目标。**`workstation` 目标一个字都不改。**
   - 所有改动只在 `tools/fr3` 和 `run/` 里，不碰 `src/lerobot`。
   - **验证方法**：
     - 在开发机上用假机器人跑现有测试，**要一个文件一个文件地跑**；
     - 加一个一致性测试：固定随机种子，本进程推理和回环服务推理得到的 chunk 必须一致；
     - 最后才由用户做真机验证。
6. **夹爪串口**（18:xx 拆解）：
   - 基本确定是 **ttyUSB0（CH340，1a86:7522）**：tele 上的路径是 `usb-0:9.1.4:1.0-port0`，alpha 上是 `10.1.4`，NUC 上 CH340 是 `7.1.4:1.0-port0`。三台机器都挂在“某个口 → hub 1 号口 → 4 号口”下面，同一个 hub 和接法；`-port0` 后缀也只有 ttyUSB 才有。Prolific 那个是 cdc_acm 直连 13 号口，是别的设备。
   - 最终确认需要动一下夹爪（`tools/fr3/fr3_pika_gripper_verify.py` 会下发开合），这算让机器人动，**由用户来跑**：`PYTHONPATH=src:. .venv-fr3/bin/python tools/fr3/fr3_pika_gripper_verify.py --port /dev/ttyUSB0`。
   - 真正的问题：by-path 编码的是“插在哪台机器的哪个口”，而 record config、launcher 和 3 个 runtime 的默认值都写死了 tele 的 by-path。所以在 NUC 上 gateway 的设备面板、录制和 rollout 都找不到夹爪（`env` 模式日志里已经能看到）。
   - 方案（待用户选）：
     - **A（推荐）**：用 udev 规则按设备身份起名 `/dev/pika_gripper`（匹配 1a86:7522），所有配置都改成这个名字。和主机、和插哪个口都无关。代价是 tele 修好后也要装这条规则（tele 换盘后反正要重装环境），而且会改到 `workstation` 用的配置文件。NUC 上目前只有一个 CH340，不会歧义。
     - **B**：用 `/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0`。不用装规则，但这个 CH340 没有序列号，以后多插一个 CH340 就会歧义；同样要改配置。
     - **C**：配置不动，`nuc` 目标在部署时生成一份 NUC 专用的覆盖配置，只替换 `gripper_port`。不影响 tele，但多了一份“机器描述”，还要让 launcher 和 runtime 也读它，改动面最大。
   - **已完成（18:3x，用户选 A，并授权我跑夹爪验证）**：
     - 验证：`/dev/ttyUSB0` 开合 1.0/0.5/0.0/1.0，误差 ≤0.6 mm，PASS。上电后第一次只读是 103.7 mm（超 `max_width_mm=90`，verify 报 FAIL），那只是没收到过指令时停在机械最大位；开合一次后张开读数 89.6 mm，和训练数据一致（ckpt 归一化统计里 gripper 中位数 0.996）。
     - 规则按 **USB 口**（udev `ID_PATH`，和 by-path 同一个键）匹配，没有只按 vid/pid：Pika Sense 等别的 CH340 插上会抢同一个名字。所以规则是“每台主机、每个口”一条，换主机或换口要重跑。
     - 新增 `tools/fr3/install_pika_gripper_udev.sh`（`--scan` 自动找口，或 `--port`），写 `/etc/udev/rules.d/99-pika-gripper.rules`。NUC 上已装：`/dev/pika_gripper -> ttyUSB0`。`setup_workstation_teleop_env.sh` 末尾也会调用它（找不到夹爪只警告），**tele 重装环境时会自动装上**。
     - record config、launcher、3 个 runtime 默认值都改成 `/dev/pika_gripper`；gateway 测试断言跟着改；verify 的 `--scan` 提示改为推荐装规则。逐文件测试通过（gateway 241、act_infer_real 126、auto_collect 37、grasp_envelope 21、terminal_trials 71 等）。
   - **顺带修了 libfranka 的库路径**：`/usr/local/lib/libfranka.so.0.15.0` 依赖 venv 里 cmeel 的 pinocchio/urdfdom/boost，但没有 RUNPATH，构建时靠 `LD_LIBRARY_PATH`。结果 `from panda_py import Panda` 在没先 import pinocchio 时报 `libpinocchio_parsers.so.3.4.0: cannot open shared object file`，gateway 的 FR3 探针和真机代码路径（`franka_research3` → `panda_py`）都会挂。已 `patchelf --set-rpath <venv>/…/cmeel.prefix/lib`，原文件备份为 `libfranka.so.0.15.0.orig`；`build_nuc_env.sh` 补了 `-DCMAKE_INSTALL_RPATH`。现在探针报 `Connection to FCI refused`，即 Desk 里没开 FCI，开机器人时由用户打开。
7. **CPU 调频 / 实时性**（拆解）：
   - 现状：NUC 已经是 **PREEMPT_RT 内核**（6.8.1-1056-realtime），启动参数里有 `isolcpus=4-7,12-15 nohz_full=… rcu_nocbs=…`；delta 在 `realtime` 组，rtprio 上限 99；调频是 powersave + EPP=performance，turbo 开着。
   - 发现：**没有任何进程绑定到隔离核 4–7/12–15**（代码里没有 taskset/sched_setaffinity），所以这些隔离核现在是闲着的，libfranka 的控制线程跑在普通核上，和相机、策略客户端、gateway 抢 CPU。
   - 建议的顺序（都不需要动机器人）：
     1. 装 `rt-tests`，用 `cyclictest -m -p 90 -i 1000 -D 5m` 分别在空载、以及 gateway 加相机都开着的情况下测，看 max latency；
     2. 同样的负载下切到 performance 调频（`cpupower frequency-set -g performance`）再测一遍；
     3. 测试控制线程绑到隔离核的效果（launcher 加 `taskset -c 4-7`，或者只绑 libfranka 线程）。
   - 判据：1 kHz 环要求每周期远小于 1 ms，cyclictest max 能稳定在 100 µs 以下就足够。只有数据不够好才改，改了就写成 systemd unit 固化。最终验收是用户跑一次 rollout 不出 `communication_constraints_violation`。
   - **已测（18:39–18:57，`rt_bench.sh`，`-m -p 90 -i 1000`，16 核各一线程；加压 = stress-ng cpu8+vm2+io2 限制在 0-3,8-11；gateway 在跑）**。单位 µs，iso = 隔离核 4-7,12-15，hk = 普通核：

     | 场景 | 时长 | iso p99.99 / max | hk p99.99 / max | >250 µs |
     |---|---|---|---|---|
     | powersave 空载 | 60 s | 50 / 109 | 21 / 92 | 0 |
     | powersave 加压（**现状**） | 60 s | 98 / 123 | 51 / 90 | 0 |
     | powersave 加压（**现状**） | 10 min | 92 / 115 | 47 / 95 | 0 |
     | performance 空载 | 60 s | 54 / 121 | 29 / 41 | 0 |
     | performance 加压 | 60 s | 91 / 108 | 26 / 97 | 0 |
     | powersave 加压 + 锁 C-state | 60 s | 25 / 44 | 52 / 88 | 0 |
     | performance 加压 + 锁 C-state | 60 s | 24 / 33 | 69 / 98 | 0 |

     “锁 C-state” 是 cyclictest 默认行为（把 `/dev/cpu_dma_latency` 持有为 0）；其余各行加了 `--laptop`，不锁，和 libfranka 实际遇到的情况一致（libfranka 不持有 PM QoS）。这台机器用 ACPI idle，C2/C3 退出延迟是 253/1048 µs。
   - 结论：
     - **现状已经够用**：10 分钟、约 960 万个周期里，最坏 115 µs，没有超过 250 µs 的，离 1 ms 预算很远。**不改。**
     - **performance 调频没用**：EPP 本来就是 performance，hk 核 max 都在 90–98。调频保持 powersave。
     - **只把控制线程绑到隔离核反而略差**：隔离核闲着会掉进深 C-state，iso max 108–123，hk 是 90–97。所以“只加 taskset”不是改进。
     - **绑隔离核 + 锁 C-state** 是唯一明显的改进（iso max 33–44）。如果 rollout 出现 `communication_constraints_violation`，就按这个改。做法：运行期间由 launcher 持有 `/dev/cpu_dma_latency`（写 0，打开期间有效；也可以只对隔离核设 `/sys/devices/system/cpu/cpu4/power/pm_qos_resume_latency_us`）；控制线程按 TID 单独 `taskset -p -c 5`。**不要**整个进程 `taskset -c 4-7`：隔离核之间不做负载均衡，所有线程会挤到同一个核上。注意机器人网卡 enp88s0 的 4 个队列中断目前实际落在 4–7 上（irqbalance 没开），绑核时要么避开它们所在的核，要么把这些中断挪走。
8. **网络 / 路由器**（拆解）：
   - 现状：局域网是一个大二层。主路由 192.168.100.1 用 DHCP 发 **192.168.0.0/16**（开发机拿到 .147.179，NUC 的 Wi-Fi 拿到 .93.93）。192.168.1.0/24 是机器人网段：机器人 .206、开发机手配 .179、NUC 手配 .156，另外还有一台路由器 192.168.1.1（MAC d4:ad:20:fb:d5:af，和 192.168.238.164 是同一厂商、相邻的 MAC，像是一台 WAN 口吃主网 DHCP 的二级路由）。
   - 风险：如果主路由的 DHCP 池没有排除 192.168.1.0/24，就可能把 .156/.179 发给别的设备，造成 IP 冲突，表现为 NUC 和开发机之间间歇性断连，rollout 里就是推理超时。
   - 需要有管理权限的人在 192.168.100.1 上做：确认地址池范围；最好把 192.168.1.0/24 整段从池里排除，或者至少给 NUC（1C:69:7A:D9:89:EF）、开发机、机器人做静态保留。
   - 不需要管理员也能做的：在 NUC 的 NM 连接上开冲突检测，`nmcli con modify ros ipv4.dad-timeout 3000`，有冲突时 NM 会拒绝启用并在日志里报出来。
   - **已完成（18:4x）**：`ros` 的 `ipv4.dad-timeout` 是 3000（下次激活生效，没有 `con up`，ssh 就走这条链路）。
   - **检测时发现 ARP flux**：`arping -D` 收到 `8c:b8:7e:f7:e9:32` 回应 .156，那是 **NUC 自己的 Wi-Fi 网卡**（有线和 Wi-Fi 在路由器那边是同一个二层；Linux 默认 `arp_ignore=0`，任何网卡都替本机所有地址回 ARP）。从开发机 arping 能看到每个请求两条回复，Wi-Fi 那条晚 100–150 ms，接受后到回复的协议栈会把 .156 缓存成 Wi-Fi MAC，FCI 的 UDP 就会绕 Wi-Fi。没有第三方设备占用 .156。
   - 修复：新增 `/etc/sysctl.d/60-lerobot-arp-flux.conf`（`net.ipv4.conf.all.arp_ignore=1`、`arp_announce=2`），已生效。现在开发机只收到有线 MAC 的回复，NUC 本机 `arping -D -I enp88s0 192.168.1.156` 0 响应。回滚就删掉这个文件再 `sysctl --system`。
   - 以后在 NUC 本机做冲突检查，用 `sudo arping -D -I enp88s0 -c 3 192.168.1.156`，有回应（退出码 1）就是真冲突；或者从开发机 `sudo arping -I enp3s0 192.168.1.156`，看有没有 1c:69:7a:d9:89:ef 以外的 MAC。
   - 顺带：NUC 的有线默认路由走 192.168.1.1（metric 100），比 Wi-Fi 优先，NUC 连不上 HF 可能和这个有关。现在远程推理不需要 HF，可以不管。

## 5. 规矩和坑（接手前必读）

- **会让机器人动的操作一律不要做**，包括 rollout、遥操作和任何下发运动指令的程序。这些都由用户本人来启动。不会让机器人动的可以随便跑。
- 加载开发机之前，先 `pgrep` 看一下有没有真机在跑。
- 不要在工作站上一次性跑完 `tests/scripts`。09-23 这样做把 GNOME 会话和 gateway 一起拖死了。要一个文件一个文件地跑。
- `import lerobot` 必须带 `PYTHONPATH=$PWD/src`（或 `src:.`），否则会导入另一份 clone。
- 开发机 conda 里的环境（`base`、`lerobot`）**都是错的**，要用 `.venv-fr3`。
- NVIDIA 相关包要 hold 住，不要让它在驱动还在运行时被自动升级。09-24 就出过这种事故。
- 不要把 checkpoint 上传到 W&B。
- 用 `pgrep -f` 等进程结束时，不要匹配到等待脚本自己的命令行。今天就踩了这个坑，等了一个永远不会结束的循环。
- **仓库里有未提交的改动**（alpha 的 10 个文件已撤销）：
  - 拆分：`tools/fr3/policy_remote.py`、`fr3_policy_server.py`、`fr3_act_infer_real_runtime.py`、`run_pick_place_infer_workstation.sh`；
  - 部署：`run/deploy.sh`、`run/sync_to_target.sh` 新增 `nuc` 目标，`workstation` 未改；新增 `run/policy_server.sh`；
  - gateway 和前端：`tools/data_collection_gui/{policy_servers.py,gateway.py,rollout.py}`，前端 `api.ts`、`types.ts`、`styles.css`、`pages/RolloutPage.tsx`、`pages/policyServers{,.test}.ts`；
  - 测试：`tests/scripts/test_fr3_policy_remote.py`、`test_data_collection_gui_policy_servers.py`、`test_fr3_act_infer_real.py`；
  - 夹爪改名：新增 `tools/fr3/install_pika_gripper_udev.sh`；`fr3_record_config.yaml`、`run_pick_place_infer_workstation.sh`、`fr3_{auto_collect,grasp_envelope,terminal_trials}_runtime.py` 改成 `/dev/pika_gripper`；`fr3_pika_gripper_verify.py` 提示、`setup_workstation_teleop_env.sh`、`test_data_collection_gui_gateway.py`；
  - `notes/2026-10-15_nuc_split*`。
  - 用户说提交才提交。
- NUC 磁盘：16:56 有人把 `~/data` 移到了 `/data/delta/data`，原位置留了软链，所以 `/home` 现在只用了 6%。**仍然不要动它。**

## 6. 脚本索引（`notes/2026-10-15_nuc_split/`）

| 文件 | 用途 |
|---|---|
| `build_nuc_env.sh` | NUC 环境构建：apt → 不含 fr3-host 的 uv sync → libfranka 0.15.0 → 完整的 setup 脚本 |
| `build_ws_env.sh` | alpha 上用过的同一流程，路径是 /home/hph |
| `build_view_meta_stub.py` | 按 10-08 日志的数值重建 funnel105 的 meta stub，参数是仓库根目录 |
| `verify_stub.py` | 用运行时自己的函数校验 stub。最后一段在 cuda 上加载策略，NUC 上只跑前 25 行 |
| `ckpt_load_probe_devbox.py` | 加载 LoRA ckpt 并测推理延迟，第二个参数可以覆盖 base 目录 |
| `cam_dual.py` | 两台 RealSense 同时开 60 fps 的帧率测试 |
| `burst_local.py` | 多核满载 4 秒的 CPU 温度测试，核数在 `range(8)` 那里改 |
| `alpha_pre_handover_kernel_errors.txt` | alpha 在我们接手前的内核错误证据 |
| `rt_bench.sh` | cyclictest 矩阵：调频 × 负载（stress-ng 限制在非隔离核）× C-state 是否锁住；不动机器人。NUC 上结果在 `/data/lerobot/outputs/rt_bench/` |
