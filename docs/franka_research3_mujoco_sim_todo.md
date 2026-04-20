# FR3 MuJoCo 仿真待办

## 范围

本文档记录了在最近一次架构评审中讨论的 FR3 遥操作仿真路线的当前执行计划。

当前的直接目标不是构建完整的数字孪生，而是建立一个本地 MuJoCo 仿真闸门，
以便在下一次真实硬件 smoke test 之前，提前发现坐标系、IK、OTG 和遥操作目标相关问题。

## 当前决策

1. 将 MuJoCo 作为 FR3 遥操作启动阶段的下一套仿真后端。
2. 保持仿真运行时为独立的 Docker 服务/Profile，但运行在支持 GPU 的 Docker 环境中，并复用现有的 LeRobot 依赖栈。
3. 复用与硬件启动相同的 FR3 资产和运动学约定：`fr3_pika_gripper_ati.urdf`、`pika_gripper_ee`、FR3 关节名称，以及当前 IK/OTG 配置路径。
4. 将 EnvHub 保留为纯仿真层后续的打包目标，而不是完整 FR3 遥操作工作流的第一步实现。

## 优先级顺序

### P0：基础设施

- [x] 记录架构与实现计划。
- [x] 添加专用的 `lerobot-fr3-sim` Docker compose 服务/Profile。
- [x] 将 MuJoCo 设为 FR3 仿真的显式依赖，而不是依赖传递安装。
- [x] 创建本地可用且兼容 EnvHub 的 FR3 MuJoCo 环境入口。

### P1：最小本地仿真闸门

- [x] 使用导入的 FR3/Pika 资产加载本地 FR3 MuJoCo 模型。
- [x] 暴露一个兼容 Gym、具备确定性 reset/step 的环境。
- [x] 返回关节状态观测，以及与相同坐标系一致的末端执行器位姿观测。
- [x] 添加本地 smoke 入口，用于验证模型加载和单步 rollout。
- [x] 在专用仿真容器内创建 MuJoCo EGL context，以验证 GPU 运行时路径。

### P2：遥操作链路验证

- [x] 添加与 HIROL MuJoCo 工作流一致的 target/TCP 可视化标记。
- [x] 将当前遥操作 target 语义桥接到本地仿真链路。
- [x] 在不接触硬件的情况下验证 `enabled` 状态切换、工作空间裁剪和首帧保持行为。
- [x] 添加 FK/IK 一致性和 target 坐标系对齐的回归检查。

### P3：后续拆分

- [ ] 将纯环境层拆分为独立、可直接用于 EnvHub 的仓库结构。
- [ ] 将本地硬件发现、SpaceMouse 编排和真实机器人 smoke 脚本保留在 EnvHub 核心包之外。
- [ ] 仅在本地 MuJoCo 闸门稳定且可复现后再发布。

## 容器建议

建议使用独立的仿真容器/服务，而不是把仿真折叠进默认的 FR3 硬件运行时服务中。

原因：

- 仿真闸门应当在没有 FR3、串口、USB 或特权硬件访问的情况下也能运行
- 这一 FR3 仿真路线中的 MuJoCo 应默认假设运行时具备 GPU，并使用面向 GPU 的容器 Profile
- 代码路径仍应复用相同的 Python 环境和 FR3 资产
- 最终得到的服务后续还可以用于本地 CI 和未来的 EnvHub 打包

实现规则：

- 独立服务/Profile
- 共享 LeRobot Docker 基础镜像，但 MuJoCo 执行使用 GPU 导向配置
- 共享仓库挂载卷
- 不对真实硬件挂载做硬依赖

## 近期下一步

1. 添加与 HIROL MuJoCo 工作流一致的 target/TCP 可视化标记。
2. 将当前遥操作 target 语义桥接到本地仿真链路。
3. 添加 FK/IK 一致性、`enabled` 状态切换和首帧保持行为的回归检查。

## 最新验证

已于 2026 年 3 月 10 日在专用的 `lerobot-fr3-sim` GPU 容器中完成验证：

- `nvidia-smi -L` 检测到 `NVIDIA GeForce RTX 4090 D`
- `MUJOCO_GL=egl` 成功创建 `mujoco.GLContext`
- `tools/fr3/fr3_mujoco_env_smoke.py` 成功加载 FR3/Pika 模型，并完成 reset 加三次零动作 step
- 同一个 smoke 入口还完成了一次相对 target 的遥操作探测，并返回了对齐的 target/TCP marker 位姿
- `tests/envs/test_fr3_mujoco.py` 已覆盖 FK/IK 往返、target/TCP marker 暴露、工作空间裁剪、首帧保持，以及 target 坐标系对齐

当前非阻塞告警：

- 导入的 URDF 在中立位姿下报告了符合预期的自碰撞对

## 当前本地遥操作入口

`tools/fr3/fr3_mujoco_teleop.py` 用于运行基于 SpaceMouse 的 FR3 MuJoCo 遥操作。
- `src/lerobot/envs/fr3_mujoco_teleop.py` 负责 target/TCP marker 更新辅助逻辑和无头遥操作循环
- `src/lerobot/envs/fr3_mujoco.py` 现在会通过与硬件后端相同的 Ruckig OTG 路径推进遥操作 target，而不是直接跳到 IK 关节目标
- `docker compose -f docker/docker-compose.yml --profile sim --profile teleop run --rm lerobot-fr3-sim-teleop` 只是进入本地交互式容器入口，默认不会自动启动遥操作脚本
- 实际运行带 viewer 的仿真遥操作命令如下：

```bash
xhost +si:localuser:root
docker compose -f docker/docker-compose.yml --profile sim --profile teleop run --rm \
  -e DISPLAY=$DISPLAY \
  -e PYTHONPATH=/workspace/src \
  lerobot-fr3-sim-teleop \
  python tools/fr3/fr3_mujoco_teleop.py \
  --enable-cameras --camera-width 640 --camera-height 480
```

- 其中 `PYTHONPATH=/workspace/src` 是当前工作区源码导入所必需的；否则容器内已安装的旧版 `lerobot` 包可能找不到 `lerobot.envs.fr3_mujoco_teleop`
- 当前本地 X11 viewer 的使用仍要求宿主机通过 `xhost +si:localuser:root` 允许容器中的 root 访问显示，并且当前 shell 需要已有可用的 `DISPLAY`

### 2026-04-20 回归排查结论（已过时，见 2026-04-20 第二版更新）

已确认"动一下 SpaceMouse 后末端继续下掉 / enabled 窗口 OTG 误差放大"是**当前工作区控制语义改动引入的回归**，不是 MuJoCo XML 资产本身的问题。

已在用户实测中确认有效的回退项：

- 将 `src/lerobot/envs/fr3_mujoco.py::_set_joint_state()` 恢复为旧的 `qpos/qvel + mj_forward` 语义，而不是 `data.ctrl + mj_step`
- 将 `enabled=True` 的 target 生成路径恢复到基线实现
- 将 `FR3MujocoEnvConfig.teleop_control_frequency` 恢复到 `200.0`
- 将 `tools/fr3/fr3_mujoco_teleop.py` 的默认 `tool-mode` 恢复到 `binary`

本轮结论：

- 高嫌疑根因在 `src/lerobot/envs/fr3_mujoco.py` 的控制链路重构，而不是 `fr3_pika_ati.xml`
- `src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_ati.xml` 中新增的 `gripper_left_pad_collision` / `gripper_right_pad_collision` 已回退
- 回退理由：用户已确认 `/home/hanyu/Codes/HIROLRobotPlatform/teleop/teleoperation.py` 与 `/home/hanyu/Codes/HIROLRobotPlatform/teleop/config/left_fr3_with_pika_ati_ik_3d_mouse.yaml` 使用同款 XML 且可正常仿真，因此不应在 LeRobot 侧额外分叉出一套 pad collision 几何

### 2026-04-20 第二版更新（已知问题状态）

以下两个问题已被验证为当前 MuJoCo 仿真环境的已知限制，详见 `docs/franka_research3_mujoco_sim_teleop_params_alignment.md`：

**问题 1：SpaceMouse 松开后臂继续运动（姿态漂移）**

- 根因：`_set_gripper_command(simulate=True)` 期间的 gripper 物理积分残留通过共享 `env.data` 影响臂姿态
- 临时规避：在夹爪完全张开状态断连，或使用 `initial_gripper=0.04`
- 长期方向：freeze 时完整保存 `env.data` 状态并逐帧回填检查

**问题 2：夹爪可夹住红色方块但无法抬起**

- 根因：MuJoCo freejoint 方块摩擦力/密度不足 + gripper simulate 期间臂姿态不稳定
- 临时修复：在 `fr3_pika_ati_scene.xml` 中将 `density` 改为 `2000`，`friction` 改为 `3.0 0.1 0.02`
- 已知限制：这是 MuJoCo 抓取仿真的固有精度问题，真实硬件无此问题

## 下一步标定

下一步推荐工作不是继续做零散的比例系数调参，而是建立一个可重复执行的 replay-and-compare 闭环。

当前工具链已经就位：

- `tools/fr3/fr3_teleop_trace_replay.py` 可在 Docker 中以 `sim` 或 `hardware` 模式启动回放
- `tools/fr3/fr3_teleop_trace_replay_runtime.py` 负责运行固定输入 profile，并在目标运行时内记录 TCP/关节轨迹
- `tools/fr3/fr3_teleop_trace_compare.py` 用于比较两组轨迹包，并输出 `scale_x/y/z` 的残差修正倍数建议
- `src/lerobot/calibration/fr3_teleop.py` 负责共享的固定 profile、轨迹 schema 和比较逻辑

剩余的操作步骤，是分别跑出一份仿真轨迹和一份真实硬件轨迹：

1. 定义一个固定的遥操作输入 profile，使末端执行器位移随时间按确定性方式变化。
2. 将同一份 profile 回放到 MuJoCo 遥操作链路中，并记录 TCP 位姿轨迹。
3. 将同一份 profile 回放到真实 FR3 遥操作链路中，并记录 TCP 位姿轨迹。
4. 在相同时间窗口内比较两组轨迹，并给出 `scale_x`、`scale_y` 和 `scale_z` 的修正建议。
5. 优先调整仿真与控制链路的一致性；只有在这之后，才用比例系数微调去补偿操作者手感上的剩余差异。

原因：

- 当前不一致主要是动态行为差异，而不只是静态增益问题
- MuJoCo 遥操作现在已经使用 OTG，因此轨迹比较能够暴露剩余的硬件控制器行为差异，而不是把问题掩盖掉
- 固定回放 profile 可以作为后续遥操作改动的可复用验收闸门
