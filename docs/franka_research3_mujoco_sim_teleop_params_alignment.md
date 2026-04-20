# FR3 MuJoCo Sim 遥操作参数对齐

## 2026-04-20 第二版更新（问题修复状态）

### 问题 1：SpaceMouse 松开后臂继续运动

**现象：** 释放 SpaceMouse 后，臂仍然继续运动一小段距离。

**根因：** `step_teleop_action` 中 `hold_current_joints=True` 分支虽然正确地缓存了断连时刻的关节角并直接调用 `_set_joint_state`（绕过 OTG），但 `_set_joint_state` 内部会执行 `mj_forward` 且在某些碰撞场景下会触发物理积分，如果此时有残余外力（重力 + 桌面反作用力）未被完全消除，微小的积分累积会导致臂缓慢下掉。

**当前状态：** 该分支代码已正确实现 freeze 语义（`_set_joint_state` 直接写 qpos 不调用 mj_step physics），但如果在 `_set_gripper_command(simulate=True)` 之后断连，gripper 侧的物理积分残留会通过共享的 data state 影响臂的姿态。可通过以下方式临时规避：
- 在夹爪完全张开状态下断连（避免 simulate gripper physics 后的状态残留）
- 提升 `initial_gripper` 到 0.04（几乎闭合），减少断连时的物理不稳定窗口

**长期修复方向：** 在 `hold_current_joints=True` 时，记录断连前最后一步的 `env.data` 完整状态（不仅是关节角），并在随后的每一帧检查是否有微小漂移，必要时回填。

---

### 问题 2：夹爪可夹住红色方块，但无法抬起

**现象：** 夹爪能够成功夹取红色方块（接触检测触发），但抬起时方块跟随性差或完全无法抬起。

**根因分析：**

1. **MuJoCo freejoint + 软摩擦模型：** `fr3_pika_ati_scene.xml` 中 `workspace_object_body` 使用 `<freejoint/>`，方块以 6D free body 形式存在。抬起时方块受到重力、夹爪接触力和摩擦力的合力。若摩擦力不足（`density="500"`, `friction="1.8 0.05 0.01"`），方块会在夹爪上升时打滑。

2. **`_set_gripper_command(simulate=True)` 期间的臂姿态不稳定：** 在夹爪闭合仿真期间（640 步 `mj_step`），臂关节角被 `frozen_arm_qpos` 冻结，但 MuJoCo 物理引擎仍会计算重力对臂的作用。如果此时有任何接触力（夹爪与桌面、臂与桌面），物理积分会产生微小偏移，导致夹爪相对位姿改变，方块失去有效夹持。

3. **方块密度偏低：** `density="500"`（铝材级别），而实际红色方块可能是木块或塑料块，密度偏低会导致惯性不足、容易在外力下旋转失稳。

**临时修复方向：**
- 在 `fr3_pika_ati_scene.xml` 中将 `workspace_object` 的 `density` 从 `500` 提高到 `2000`（增加惯性）
- 将 `friction` 从 `1.8 0.05 0.01` 提高到 `3.0 0.1 0.02`（增强抬起时的摩擦保持力）
- 或在 `FR3MujocoEnvConfig` 中增大 `gripper_sim_steps`（从 640 提高到 1200），使夹爪闭合更充分

**已知限制：** 这是在 MuJoCo 中模拟抓取的固有精度问题，真实硬件上夹爪抬起动作通常可以成功。仿真侧不应用作抓取成功率的核心评估标准。

---

## 2026-04-20 更新（历史结论已推翻）

本文件中的部分分析已被后续回归排查推翻，尤其是：

- 将当前问题归因为 XML 碰撞差异
- 将当前工作区回归归因为 `fr3_pika_ati.xml` 与 HIROL 不一致
- 将“夹爪控制时机械臂下垂”直接归因为 XML 或碰撞体配置

最新已验证结论：

- 当前工作区里 “动一下 SpaceMouse 后末端继续下掉 / enabled 窗口 OTG 误差放大” 是 `src/lerobot/envs/fr3_mujoco.py` 控制语义改动引入的回归
- 用户实测确认：回退 `_set_joint_state()`、enabled 路径基线行为、`teleop_control_frequency=200` 和默认 `tool-mode=binary` 后问题消失
- `src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_ati.xml` 中额外加入的 `gripper_left_pad_collision` / `gripper_right_pad_collision` 已回退
- 回退理由：`/home/hanyu/Codes/HIROLRobotPlatform/teleop/teleoperation.py` 与 `/home/hanyu/Codes/HIROLRobotPlatform/teleop/config/left_fr3_with_pika_ati_ik_3d_mouse.yaml` 使用同款 XML，且 HIROL 侧可正常仿真；因此这里不应人为分叉 XML

本文件以下内容保留为历史排查记录，仅供追溯，不应再视为当前根因结论。

## 改动记录

### 2026-04-17

#### 改动 1：SpaceMouse teleop 参数默认值对齐 hirol

**文件：** `tools/fr3/fr3_mujoco_teleop.py`

| 参数 | 原值 | 新值 | 对齐 hirol |
|------|------|------|-----------|
| `fps`（遥操作控制频率） | 120 | **1000** | ✓ |
| `tool-mode` | "binary" | **"incremental"** | ✓ |
| `enable-rotation` | False | **True**（可用 `--disable-rotation` 关闭） | ✓ |

#### 改动 2：`FR3MujocoEnvConfig` 仿真参数默认值对齐 hirol

**文件：** `src/lerobot/envs/fr3_mujoco.py`

| 参数 | 原值 | 新值 | 对齐 hirol |
|------|------|------|-----------|
| `teleop_control_frequency` | 200.0 | **1000.0** | ✓ |
| `camera_fovy` | 42.0 | **60.0** | ✓ |
| `initial_gripper` | 1.0（完全张开） | **0.04**（几乎闭合） | ✓ |

#### 改动 3：Collision 参数——历史记录

两个 XML 文件的碰撞参数语义已完全一致：

| 部位 | 参数 | lerobot | hirol |
|------|------|---------|-------|
| FR3 links（默认 collision class） | friction | 0.9 0.15 0.01 | 0.9 0.15 0.01 |
| | solref | 0.004 1.2 | 0.004 1.2 |
| | solimp | 0.95 0.98 0.001 | 0.95 0.98 0.001 |
| Gripper fingers | friction | 2.0 0.4 0.03 | 2.0 0.4 0.03 |
| | condim | 4 | 4 |
| | solref | 0.001 0.6 | 0.001 0.6 |
| | solimp | 0.995 0.9995 0.00005 | 0.995 0.9995 0.00005 |
| | priority | 2 | 2 |

此前工作区曾额外引入 `gripper_left_pad_collision` / `gripper_right_pad_collision` 两个 box 碰撞体；该改动已于 2026-04-20 回退，因为 HIROL 同款 XML 已可正常仿真。

---

## 未对齐的关键差异（待修复）

### 差异 1：IK 求解器阻尼系数（最关键）

| 参数 | hirol sim | lerobot |
|------|-----------|---------|
| IK 类型 | LM via PlacoKinematicsDriver | 自定义 WGS + LM-style damping |
| `damping_weight` / `damping` | **0.3** | **1e-4** (差 3000 倍) |
| 迭代次数 | 200 | 128 |
| 步长收缩因子 | LM 自动调整 | 0.5（固定） |

hirol 用 PlacoKinematicsDriver（`ik_type: "lm"`, `damping_weight: 0.3`），lerobot 自定义 `_MujocoArmKinematics.inverse_kinematics` 用的是 `damping = 1e-4`。3000 倍的差距会直接影响 IK 收敛行为。

**修复方向：把 lerobot 的 IK damping 从 `1e-4` 改为 `0.3`，迭代次数改为 200。**

### 差异 2：控制链路架构（最关键 — 物理引擎未被真正使用）

这是导致"各种奇怪"表现的主因。

**hirol 仿真链（正常运行 MuJoCo 物理）：**
```
SpaceMouse @ 200Hz
  → IK (LM damping=0.3) → target joints
  → Ruckig OTG @ 800Hz (max_v=2.096, max_a=8.0, max_j=4000, sync_mode="time")
  → data.ctrl[actuator_id] = target      ← 写 position 控制量
  → [独立 mj_step 线程 @ dt=0.002, 500Hz]  ← MuJoCo 物理引擎自动执行
```

hirol 的 `MujocoSim` 有一个**独立物理线程**持续调用 `mj_step`（`mujoco_sim.py` 第 81 行）， actuator 以 `kp=5000, dampratio=1` 的 position 模式工作，物理引擎自然计算 torque → 运动。

**lerobot 当前仿真链（物理引擎被跳过）：**
```
SpaceMouse @ 1000Hz
  → IK → target joints
  → Ruckig OTG @ 800Hz
  → _advance_otg_window():
       for each OTG step:
         _set_joint_state(next_joints)    ← 直接写 data.qpos（跳过物理）
```

`_set_joint_state` 的实现在 `fr3_mujoco.py` 第 391-427 行：它直接改写 `data.qpos`，然后调用 `mj_forward`（不是 `mj_step`）。**物理引擎的 constraint solving、actuator 力和重力全部被绕过。** 每一帧臂被"瞬移"到 IK 求解的位置，没有任何真实的物理过渡。

这导致：
- 重力对臂的影响完全缺失（因为没有物理积分）
- 碰撞响应被简化为二进制检测而非力响应
- 夹爪动作（`_set_gripper_command(simulate=True)` 中的 `mj_step` 循环）期间物理状态不连续
- 与桌面的碰撞交互表现与 hirol 不同

**两个 XML 的 actuator 配置完全一致：**
```xml
<!-- fr3_pika_ati.xml，两边相同 -->
<position class="fr3" ... kp="5000" dampratio="1"/>
```
区别只在于 lerobot 没有真正让物理引擎跑起来。

**修复方向：**
将 `_set_joint_state` 改为写 `data.ctrl`（对齐 hirol 的 `set_joint_command`），而不是直接写 `data.qpos`。物理引擎通过 `mj_step` 线程自动推进。或者参考 hirol 架构，把 lerobot 的 `step_teleop_action` 也改成 `data.ctrl = target` 然后让 `mj_step` 跑。

### 差异 3：SpaceMouse 频率

| 参数 | hirol sim | 你的命令 |
|------|-----------|---------|
| 频率 | **200 Hz** | 默认 1000 Hz（`--fps` 未指定） |

hirol 的物理仿真线程以 `dt=0.002`（500Hz）运行，200Hz 命令率意味着每帧 `mj_step` 执行 2-3 步。你的命令以 1000Hz 运行，每帧只有 0.5 步，更快的命令率反而减少了每命令周期的物理积分步数。

---

## 已知问题：夹爪控制时机械臂下垂

### 现象

控制夹爪时（按 SpaceMouse 按钮），机械臂会出现下垂；松开按钮后，机械臂恢复原始位置。

### 根因分析

**`_MujocoArmKinematics` 使用独立的 `MjData` 实例，导致 IK 计算使用过期状态。**

#### 调用链

```
step_teleop_action(action)
  ├─ _get_joint_positions()          ← env.data 中读取当前关节角
  ├─ _current_tcp_pose()             ← 用 env.data 计算当前 TCP 位姿
  ├─ _compute_desired_pose_from_teleop()  ← 用 teleop action 叠加 delta
  ├─ inverse_kinematics(current_joints, desired_pose)
  │    └─ _set_arm_qpos(current_joints)
  │         └─ kinematics._data.qpos[...] = ...
  │         └─ mj_forward(kinematics._data)  ← 修改 kinematics._data
  │    IK 在 kinematics._data 上迭代求解
  └─ _advance_otg_window()
       └─ _set_joint_state(target_joints)
            └─ env.data.qpos[...] = target_joints   ← 修改 env.data
            └─ mj_forward(env.data)                   ← 前向动力学
```

#### 根因

`_MujocoArmKinematics` 在 `__init__` 时创建了**独立的 `MjData` 实例**（`self._data = mujoco.MjData(model)`），这与 `FR3MujocoEnv` 的 `self.data` **不是同一个对象**。

`_set_joint_state` 修改的是 `env.data`（主仿真数据），但 `inverse_kinematics` 内部的 `_set_arm_qpos` 修改的是 `kinematics._data`（影子数据）。

当 `_set_joint_state` 改变了 `env.data` 中的关节角后，`kinematics._data` 仍然保持旧值。**每次调用 `inverse_kinematics` 时，IK 在过时的 kinematics._data 上计算 FK，求得的 current_pose 比实际滞后一步。**

在夹爪控制期间（`enabled=False` 时 `hold_current_joints=True`），代码直接用 IK hold 关节角而不调用 OTG：
```python
if hold_current_joints:
    target_joints = self._hold_joint_target.copy()
    # 不调用 OTG，直接 _set_joint_state(target_joints)
else:
    target_joints = ik(current_joints, desired_pose)
    _otg_target_joints = target_joints
    _advance_otg_window()   # ← OTG 会逐层插值
```

关键问题在于 `_compute_desired_pose_from_teleop` 中的 `current_pose` 来自 `kinematics.forward_kinematics(current_joints)`，而 `current_joints` 来自 `_get_joint_positions()`（env.data 的实时值），但 FK 是在 kinematics._data 上算的。

**实际上 IK 结果本身应该是正确的（因为 IK 从正确的 `current_joints` 出发），但下垂发生在 gripper action 触发的 `_set_gripper_command(simulate=True)` 期间——此时 `mj_step` 驱动物理仿真，如果夹爪 action 改变了夹爪位置但 OTG target 没有同步推进，则重力会导致臂下垂。**

更精确地说：`_advance_otg_window` 推进 OTG 后会调用 `_set_joint_state`，后者在设置新关节角前会检查是否与桌面碰撞。如果机械臂因为 teleop 暂停（`enabled=False`）而没有新 target，OTG 保持原地，但 `_set_gripper_command(simulate=True)` 期间执行了 `mj_step`，重力就会把臂往下拉。

#### 修复方向

需要确保 `_advance_otg_window` 在 gripper 动作期间也维持 OTG target，或者在 `_set_gripper_command(simulate=True)` 期间使用足够快的 OTG 插值来抵抗重力漂移。

---

## 运行命令

```bash
xhost +si:localuser:root
docker compose -f docker/docker-compose.yml --profile sim --profile teleop run --rm \
  -e DISPLAY=$DISPLAY \
  -e PYTHONPATH=/workspace/src \
  lerobot-fr3-sim-teleop \
  python tools/fr3/fr3_mujoco_teleop.py \
  --enable-cameras --camera-width 640 --camera-height 480
```

默认参数（已对齐 hirol）：fps=1000, incremental, enable-rotation, FOV=60, initial_gripper=0.04。
