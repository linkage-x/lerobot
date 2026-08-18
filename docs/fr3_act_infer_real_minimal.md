# FR3 ACT 真机推理最小说明

本文是当前仓库内 FR3 ACT 真机推理的权威入口说明。目标是回答 4 个问题：

- 现在应该运行哪个入口
- 默认启动会做什么
- policy 输入/输出按什么语义解释
- 如何配置相机、机器人初始状态、交互 rollout 和 MuJoCo 可视化

## 当前入口

- 宿主机 launcher: `tools/fr3/fr3_act_infer_real.py`
- 容器内 runtime: `tools/fr3/fr3_act_infer_real_runtime.py`
- compose service: `lerobot-infer-fr3-act`

默认 checkpoint:

- `outputs/train/2026-03-19/10-48-39_act/checkpoints/060000`

默认相机配置:

- `tools/fr3/fr3_act_infer_camera_config.yaml`

新增 MuJoCo 模型:

- DAS: `src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.xml`
- Pika: `src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper_ati.xml`

## 默认启动行为

直接运行：

```bash
python3 tools/fr3/fr3_act_infer_real.py
```

当前默认会按这个顺序执行：

1. 用 `panda_py` 将机械臂移动到 DAS 起始关节角。
2. 连接 `FrankaResearch3`。
3. 将 gripper 对齐到 dataset start mean。
4. 启动 ACT inference loop。

启动时**不会**移动到 DAS 起始关节角：那组关节角属于 DAS 台架。`T_B_Ws` 是用第一帧观测
对齐 dataset start pose 解出来的，所以起始位姿决定整条轨迹落在工作空间的哪里；home 到别的
台架的位姿会让每一个 target 都带上一个固定偏移。请用 `--robot-init-state`（或 launcher 自
己的 homing 步骤）home 到录制这批 episode 时用的位姿。

如需恢复旧行为（仅在 DAS 台架上有意义）：

```bash
python3 tools/fr3/fr3_act_infer_real.py --move-to-das-start
```

如需关闭其它默认启动动作：

```bash
python3 tools/fr3/fr3_act_infer_real.py --no-align-gripper-to-dataset-start
```

## inference config

推荐从训练生成的 inference YAML 启动，这样 checkpoint、dataset root、硬件和 runtime 参数都能集中记录：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --inference-config outputs/datasets/<job_name>/inference_config.generated.yaml
```

当前 launcher 支持从 YAML 中读取这些字段：

```yaml
runtime:
  checkpoint: outputs/train/<run>/checkpoints/<step>
  camera_config: tools/fr3/fr3_act_infer_camera_config.yaml
  dataset_root: outputs/datasets/<dataset>
  policy_fps: 10
  max_steps: null
  preview: false
  hardware:
    robot_ip: 192.168.1.208
    gripper_port: /dev/ttyUSB0
    gripper_backend: das  # das or pika
  startup:
    move_to_das_start: false  # default; true only on the DAS rig
    align_gripper_to_dataset_start: true
    dataset_start_gripper_tolerance: 0.05
    robot_init_state:
      type: joints
      joints_rad: [-0.05, -1.56, -1.72, -2.12, 0.01, 2.12, -0.97]
      gripper: 0.5
  interactive:
    enabled: true
    start_key: s
    stop_key: x
    quit_key: q
  mujoco:
    enabled: true
    model: src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.xml
    max_chunk_points: 64
  safety:
    first_frame_max_pos_delta_mm: 30
    first_frame_max_rot_delta_deg: 10
    max_step_pos_delta_mm: 5
    max_step_rot_delta_deg: 3
```

CLI 参数会覆盖 YAML 中的同名配置。

## 当前输入/输出合同

### 状态输入

runtime 当前按下列语义构造 `observation.state`：

- 真机 pose 从 `E = das_gripper_ee` 转到 `I = gripper_base_link`
- pose 再映射到 dataset/replay 使用的世界合同
- gripper 观测先从硬件归一化 `[0,1]` 反变换回 dataset aperture 单位
- 最终按 dataset metadata 中的 `state_names` 组装 policy 输入

这一步是当前版本最重要的修复点。此前 preview 和 real-run 不一致，根因不是主坐标链，而是 live gripper observation 与 dataset gripper 单位不一致。

如果 checkpoint 来自 `mask2ee` 训练：

- runtime 不需要额外 CLI 开关
- 推理会跟随 checkpoint 中保存的 policy config 自动继续 mask 掉 `x y z qx qy qz qw`
- `gripper` 仍然保留

详细合同见：

- `docs/fr3_mask2ee_training_inference_contract_20260326.md`

### 图像输入

当前图像语义已确认正确：

- `left -> observation.images.left`
- `right -> observation.images.right`

默认相机键直接与 policy 图像键同名，不再需要额外 `camera-key-map`。

### camera-config 会做什么

runtime 会读取 `--camera-config` 指向的 YAML 中的 `robot.cameras` 字段，并据此构造 `FrankaResearch3Config.cameras`。支持的 camera type:

- `opencv`
- `intelrealsense`
- `hikrobot`

推理开始前会用 checkpoint 的 policy metadata 检查图像键：

- checkpoint 需要 `observation.images.left`，则 camera config 必须有 `left`
- checkpoint 需要 `observation.images.right`，则 camera config 必须有 `right`
- 缺任何一个都会直接报错，不会静默错接相机

每个 policy step 中，runtime 会：

1. 调用 `robot.get_observation()` 从机器人和相机读当前观测。
2. 从观测里取出 policy 需要的相机 key。
3. 如果该相机配置是 `ColorMode.BGR`，转换成 RGB。
4. 如果相机实际分辨率和 policy feature shape 不一致，resize 到 checkpoint 需要的 `H,W`。
5. 组装成 `observation.images.<camera_key>` 输入 policy。

典型 OpenCV 配置：

```yaml
robot:
  cameras:
    left:
      type: opencv
      device_id: /dev/video22
      image_shape: [480, 640]
      fps: 30
      color_mode: BGR
      fourcc: MJPG
      backend: V4L2
    right:
      type: opencv
      device_id: /dev/video24
      image_shape: [480, 640]
      fps: 30
      color_mode: BGR
      fourcc: MJPG
      backend: V4L2
```

### tactile 输入

当前默认 checkpoint 仍是 tactile ACT。runtime 支持两种路径：

- 真机 DAS tactile 输入
- 仅限 preview 的 `--tactile-fallback=baseline_idle`

已确认 infer 容器内可以拿到真实 SDK tactile callback，但 `448-byte` payload 与 dataset `left_raw/right_raw` 的最终 wire-format 映射还没有彻底闭环。

### action 解码

当前 action 默认按 8 维解释：

- `x y z qx qy qz qw gripper`

runtime 会：

1. 将 quaternion action 转成 rotvec
2. 将 gripper 从 dataset aperture 单位映射回硬件所需的 `0..1`
3. 在发送前应用首帧 gate 和每步小步限幅
4. 调用 `robot.send_action(...)`

joint-space OTG / 高频控制仍由 `FrankaResearch3` 内部负责。

## robot_init_state

`--robot-init-state` 用于指定每次推理前机器人应该回到的初始状态。它支持关节角，也支持末端位姿。

关节角 shorthand:

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --robot-init-state 'joints=-0.05,-1.56,-1.72,-2.12,0.01,2.12,-0.97'
```

末端位姿 shorthand，四元数格式为 `[x,y,z,qx,qy,qz,qw]`：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --robot-init-state 'ee_xyzquat=0.4,0.0,0.3,0.0,0.0,0.0,1.0'
```

末端位姿也可以用 rotvec：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --robot-init-state 'ee_xyzrotvec=0.4,0.0,0.3,3.14,0.0,0.0'
```

也可以使用 YAML 文件：

```yaml
robot_init_state:
  type: joints
  joints_rad: [-0.05, -1.56, -1.72, -2.12, 0.01, 2.12, -0.97]
  gripper: 0.5
  timeout_s: 20
  joint_tolerance_rad: 0.01
  gripper_tolerance: 0.02
```

调用：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --robot-init-state configs/fr3_init.yaml
```

如果 `robot_init_state` 是 EE pose，YAML 可写成：

```yaml
robot_init_state:
  type: ee_xyzquat
  xyzquat: [0.4, 0.0, 0.3, 0.0, 0.0, 0.0, 1.0]
  gripper: 0.5
  timeout_s: 20
  ee_pos_tolerance_m: 0.005
  ee_rot_tolerance_deg: 2.0
```

执行语义：

- 非交互模式：启动后先移动到 `robot_init_state`，再进入一次 policy rollout。
- 交互模式：每次 rollout 前都会先移动到 `robot_init_state`，然后等待 start key。
- 如果没有配置 `robot_init_state`，交互模式下停止当前 rollout 后只会保持当前机器人状态等待下一次 start。

## 交互 rollout

开启交互模式：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --interactive-rollouts \
  --robot-init-state 'joints=-0.05,-1.56,-1.72,-2.12,0.01,2.12,-0.97'
```

默认按键：

- `s`: 从等待状态进入当前 rollout
- `x`: 停止当前 rollout，回到等待状态；如果设置了 `robot_init_state`，下一轮会先回到初始状态
- `q`: 退出整个 inference

可改按键：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --interactive-rollouts \
  --rollout-start-key s \
  --rollout-stop-key x \
  --rollout-quit-key q
```

键盘监听优先使用 `sshkeyboard`。如果容器里没有安装 `sshkeyboard`，runtime 会退回到 TTY stdin raw-key 监听；Docker compose service 已设置 `stdin_open: true` 和 `tty: true`。

## MuJoCo viewer

开启 MuJoCo 可视化：

```bash
python3 tools/fr3/fr3_act_infer_real.py --mujoco-viewer
```

viewer 行为：

- MuJoCo 中的 FR3 机器人会同步真机 `fr3_joint1..7`。
- 橙色 cube 表示真机当前 EE pose。
- 绿色 cube 表示当前 policy 输出的 EE target。
- 每当 ACT 产生新的 action chunk，viewer 会画出 chunk 内所有 EE target 点构成的轨迹。
- 轨迹颜色从蓝到粉渐变，表示 chunk 内 action 的先后顺序。

按夹爪自动选择默认 XML：

- `--gripper-backend das` 默认使用 `fr3_das_ati.xml`
- `--gripper-backend pika` 默认使用 `fr3_pika_gripper_ati.xml`

Pika 真机推理并开启 viewer：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --gripper-backend pika \
  --mujoco-viewer
```

显式指定 XML：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --mujoco-viewer \
  --mujoco-model src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper_ati.xml \
  --mujoco-max-chunk-points 64
```

当前两个 cube 和轨迹是 MuJoCo passive viewer 的 `user_scn` overlay geom，不是参与动力学的 mocap body。这样不会影响机器人控制链路，只用于在线监控。

## 推荐命令

查看最终 Docker 命令：

```bash
python3 tools/fr3/fr3_act_infer_real.py --dry-run
```

先做安全预览：

```bash
python3 tools/fr3/fr3_act_infer_real.py --preview --max-steps 5
```

如果只想排除 tactile 缺失的影响：

```bash
python3 tools/fr3/fr3_act_infer_real.py --preview --max-steps 5 --tactile-fallback baseline_idle
```

正式真机执行：

```bash
python3 tools/fr3/fr3_act_infer_real.py
```

带初始状态、交互控制和 MuJoCo viewer 的 DAS 推荐模板：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --interactive-rollouts \
  --robot-init-state 'joints=-0.05,-1.56,-1.72,-2.12,0.01,2.12,-0.97' \
  --mujoco-viewer
```

Pika gripper 模板：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --gripper-backend pika \
  --interactive-rollouts \
  --robot-init-state 'joints=-0.05,-1.56,-1.72,-2.12,0.01,2.12,-0.97' \
  --mujoco-viewer
```

更保守的安全门：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --preview \
  --max-steps 5 \
  --first-frame-max-pos-delta-mm 20 \
  --first-frame-max-rot-delta-deg 8 \
  --max-step-pos-delta-mm 3 \
  --max-step-rot-delta-deg 2
```

## 当前已确认事项

已经闭环：

- `left/right` 图像语义正确
- preview 与 real-run 的 gripper 观测语义已统一
- `move_to_das_start` 已深度集成到实际 inference runtime
- launcher 默认行为已与真机启动要求一致
- `robot_init_state` 可指定初始关节角或初始 EE pose
- 交互 rollout 已支持 start/stop/quit
- MuJoCo viewer 已支持真机关节同步、当前/目标 EE cube、action chunk 渐变轨迹
- 已提供 FR3 + ATI + Pika gripper 的 MuJoCo XML

仍未闭环：

- DAS tactile `448-byte` payload 到 dataset `left_raw/right_raw` 的最终硬件语义映射
- 真机长 rollout 下的失败恢复、operator confirm gate、在线记录
- 本地宿主机环境缺少 `mujoco` Python 包，MuJoCo viewer 需要在含 MuJoCo 依赖和 X11 显示权限的运行环境中实际验证

## 支撑文档

推荐阅读顺序：

1. 本文：运行入口与当前合同
2. `docs/fr3_real_infer_docs_index_20260324.md`：文档职责索引
3. `docs/fr3_mask2ee_training_inference_contract_20260326.md`：mask2ee 训练/推理合同
4. `docs/fr3_act_infer_runtime_fix_20260324.md`：本轮修复记录
5. `docs/fr3_infer_image_semantics_validation_20260323.md`：图像链路结论
6. `docs/tactile/fr3_das_tactile_packet_investigation_20260323.md`：tactile open issue
7. `docs/fr3_replay_tracking_findings_20260319.md`：replay 侧追踪问题
