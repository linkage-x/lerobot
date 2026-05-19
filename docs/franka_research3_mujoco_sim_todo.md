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

### P2.1：已知问题

- [ ] **Keyboard teleop 未生效**（2026-04-20）：键盘遥操作（`KeyboardEndEffectorTeleop`）按键后仿真内 EE 无响应。
    可能原因：`KeyboardEndEffectorTeleop.get_action()` 返回的 `delta_x/delta_y/delta_z` 到 SpaceMouse 语义 action 的映射链路存在断点，
    或 `fr3_mujoco_teleop.py` 中 `run_sim_teleop_loop` 对 keyboard 输出的 action 转换逻辑有遗漏。
    进一步 debug 方向：在 `step_teleop_action` 处加断点，确认 action dict 的 `enabled`/`target_x/y/z` 是否随键盘输入变化。

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

## Dataset Viewer Replay

To replay the `single_cube2` 7D EE-pose dataset episode in MuJoCo with the
passive viewer on display `:1`:

```bash
export DISPLAY=:1 && python tools/fr3/fr3_teleop_trace_replay.py --mode sim --dataset data/single_cube2_20260429_165325 --episode 0 --fps 30 --output outputs/fr3_traces/single_cube2_ep0_trace.json
```

The sim replay path defaults to the HIROL-style Pinocchio Levenberg-Marquardt
IK solver used by `left_fr3_with_pika_ati_ik_3d_mouse.yaml`. Use
`--ik-solver hirol_gaussian_newton` for HIROL-style Gaussian-Newton IK, or
`--ik-solver placo` to run the previous Placo-based IK path.

## Next Calibration Step

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

## 2026-04-20 第三版对齐任务

在修完 `enabled=True` 重锚定和 continuous physics 后，仍观察到：

- `SpaceMouse` 仅做很小的 `x/y` 输入时，`target_pose.z` 基本不变
- 但实际 `tcp_pose.z` 仍持续下探

这说明剩余问题不在输入侧，而在 arm 控制链本身。下一步按以下顺序执行：

1. 让后台 OTG thread 真正使用 `otg_async_control_frequency`
   - 目标：修复当前 “OTG 已在后台执行，但实际步频由 `continuous_physics_frequency` 决定” 的语义不一致
   - 验收：当 `use_otg=True` 且 `continuous_physics=True` 时，后台控制 tick 频率与 `otg_async_control_frequency` 一致

2. 将 `step_teleop_action()` 的 IK 来源从 env-local `_MujocoArmKinematics` 切到 `PlacoKinematicsDriver`
   - 目标：去掉当前 MuJoCo env 内部的简化 DLS IK 主路径，改用与硬件侧更一致的 kinematics 求解器
   - 验收：`FR3MujocoEnv` 的主 FK/IK 路径使用 `PlacoKinematicsDriver`，相关 FK/IK 回归保持通过

3. 完成上述两项后，再重新验证 “小幅 x/y 输入时末端是否仍明显下探”
   - 若问题仍在，再继续检查更深一层的 controller / joint-space execution path

### 2026-04-20 第三版执行进展

- 已完成：
  - `continuous_physics_dt` 在 `use_otg=True` 时已改为真正使用 `otg_async_control_frequency`
  - `FR3MujocoEnv` 主 FK/IK 路径已从 env-local `_MujocoArmKinematics` 切换到 `PlacoKinematicsDriver`
- 当前结果：
  - 相关聚焦回归通过
  - 但“小幅 x/y 输入、target_z=0 时实际 tcp_z 仍持续下探”的症状依旧存在
  - 同时在 `PlacoKinematicsDriver + continuous physics` 的最小复现中观察到原生内存错误
    （`malloc_consolidate(): invalid chunk size`）
- 当前判断：
  - 仅修正 async OTG 频率语义和切换 kinematics 来源还不足以消除末端下探
  - 剩余问题更可能落在更深一层的 controller / joint-space execution path
  - `PlacoKinematicsDriver` 在当前 MuJoCo continuous thread 使用方式下还需要额外稳定性验证

### 2026-04-20 `PlacoKinematicsDriver` 稳定性隔离结论

- 已执行的隔离验证：
  1. 单线程、非 `continuous_physics` 模式下，对 `PlacoKinematicsDriver` 连续执行 300 次 FK/IK 压测
- 更细一层的脚本化隔离显示：
    - `import placo`、`import backends`、`import PlacoKinematicsDriver` 符号导入都可正常退出
    - `placo.RobotWrapper`、`placo.KinematicsSolver`、`RobotKinematics` 也可在最小脚本中正常退出
    - 但一旦实际实例化 `PlacoKinematicsDriver`，无论是否调用 FK/IK，子进程退出时都会稳定触发 glibc 堆错误
    - `FR3MujocoEnvConfig + PlacoKinematicsDriver` 组合场景同样复现
    - 在同一复现脚本里新增"本地普通包装类"和"本地 `@dataclass` 包装类"两种等价实现：
      - 两者都只是二次持有同一个 `RobotKinematics`，并复刻 `PlacoKinematicsDriver` 的 rad/deg 转换逻辑
      - **2026-04-20 更新**：在 `continuous_physics=True` 场景下（即 MuJoCo 后台物理线程持续运行的场景），
        本地包装类和 `PlacoKinematicsDriver` **都会**触发 SIGSEGV (exit 139)，
        因为问题不在 `PlacoKinematicsDriver` 这层包装，而在 `RobotKinematics` 底层调用的 placo C++ binding 本身。
        因此"本地包装不崩"的结论仅在单线程 standalone 场景成立。

- 实测结果（2026-04-20）：
  - `continuous_physics=True` + 60秒压测（~200万次 FK/IK 调用）：
    - `PlacoKinematicsDriver` → exit 139 (SIGSEGV)
    - `LocalKinematicsDriver`（等价的本地包装）→ exit 139 (SIGSEGV)
    - `RobotKinematics`（直连）→ exit 139 (SIGSEGV)
  - 单线程 standalone FK/IK 场景：三者均正常退出，无崩溃
  - **结论**：根因在 placo C++ binding 与 continuous_physics MuJoCo 环境的交互，与上层包装方式无关

- 结论：
  - `PlacoKinematicsDriver` 在 `continuous_physics=True` 场景下与 placo C++ binding 一起触发 SIGSEGV，与包装方式无关
  - 问题收敛在 placo C++ binding 与 MuJoCo continuous_physics 线程的交互，而非 Python 层代码路径
  - `_MujocoArmKinematics` 在相同场景下稳定，是当前唯一可靠的 MuJoCo FK/IK 路径
  - 后续若继续沿 PlacoKinematics 方向推进，应先在 placo 库侧修复该线程安全问题

### 2026-04-20 主路径回退执行结果

- 已执行：
  - `FR3MujocoEnv._build_kinematics()` 已回退为使用 env-local `_MujocoArmKinematics`
  - `PlacoKinematicsDriver` 不再位于 `FR3MujocoEnv` 主 FK/IK 执行路径中
- 保留项：
  - `otg_async_control_frequency` 对后台 continuous physics tick 的语义修正仍保留
  - `PlacoKinematicsDriver` 相关隔离结论保留为后续单独排查 `placo` binding 生命周期问题的依据
