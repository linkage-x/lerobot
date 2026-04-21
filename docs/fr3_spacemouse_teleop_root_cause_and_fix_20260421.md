# FR3 SpaceMouse Teleop 根因与修复方案

日期：2026-04-21

## 背景

在 `tools/fr3/fr3_mujoco_teleop.py` 中使用 SpaceMouse Compact 控制 FR3 MuJoCo 仿真时，出现如下现象：

- 只要轻微拨动 SpaceMouse，末端就会稳定地往 `-x`、`-z` 方向偏移
- 即使启动参数为 `--disable-rotation`，末端仍然会出现“低头”式俯仰漂移

## 结论

这不是单点故障，而是两类问题叠加：

1. 输入映射错误
- `src/lerobot/teleoperators/spacemouse/teleop_spacemouse.py` 当前将 SpaceMouse 原始平移量硬编码重映射为 `[-raw_y, raw_x, raw_z]`
- 该映射没有显式声明控制坐标系，也没有证明它与 FR3 基坐标系一致
- 结果是：用户手部动作的主分量会被稳定地解释成错误的机器人平移命令，导致系统性 `-x/-z` 偏移

2. “禁用旋转”没有真正锁姿态
- `enable_rotation=False` 现在只会把用户输入的旋转增量清零
- 但 IK 仍然允许求解器为了满足位置误差而牺牲姿态
- `src/lerobot/model/kinematics.py` 中 `orientation_weight=0.01`，远弱于 `position_weight=1.0`
- 结果是：求解器会用“低头”来换取位置收敛

3. 漂移被控制环路放大
- `src/lerobot/envs/fr3_mujoco.py` 当前每一步都用 `current_pose` 重建目标姿态
- 一旦某一步 IK 让末端轻微低头，下一步又会把这个低头姿态当成新的基线
- 结果是：姿态误差被逐步积分，形成持续漂移

4. 仿真执行模型与 IK/FK 模型不一致
- 在 FR3 MuJoCo 环境中，实际物理执行由 MuJoCo 模型负责
- 但此前 `_build_kinematics()` 默认返回的是 `LocalKinematicsDriver`，也就是 `URDF + placo` 运动学链
- 新增 `pika_task_tcp` 后，这两套模型在 TCP 定义上并不再严格一致
- 结果是：`target_pose` 在 `placo FK` 看起来正确，但同一组关节角写进 MuJoCo 后，真实 `tcp_pose` 会朝另一个方向漂移

这条根因是本次进一步定位后确认的关键补充：

- `SpaceMouse -> target_pose` 这一层已经可以按 `+x/+y/+z` 正常推进
- 真正反向的是 `target_pose -> IK joints -> MuJoCo tcp_pose`
- 当切换到 MuJoCo 自身的雅可比 IK `_MujocoArmKinematics` 后，`+x/+y/+z` 三个单轴目标都可以在 MuJoCo 模型内准确收敛

## 非根因

以下现象目前判断为噪声，不是本次问题的根因：

- `Home directory not accessible: Permission denied`
- `pkg_resources is deprecated as an API`
- `release_decay TRUNCATE`

它们可能影响运行环境或日志可读性，但不足以解释确定性的 `-x/-z + 低头` 行为。

## 修复目标

修复后系统应满足以下不变量：

1. SpaceMouse 输入到机器人命令的坐标变换必须显式可见、可测试、可配置
2. 当 `enable_rotation=False` 时，用户语义应为“姿态锁定”，而不是“用户不再提供旋转输入”
3. 控制器目标状态与执行器当前状态必须分离
- 控制器维护目标姿态
- IK/物理引擎只负责逼近目标，不能反过来偷偷改写目标

## 修复顺序

### P0：先止血，锁姿态

修改：

- `src/lerobot/model/kinematics.py`
- `src/lerobot/envs/fr3_mujoco.py`

动作：

- 为 IK 增加“锁姿态”模式或等效强约束参数
- 当 `enable_rotation=False` 时，显著提高姿态约束强度
- 禁止用 `current_pose.R` 作为下一步禁转模式下的目标姿态基线

预期收益：

- 先消掉“低头漂移”
- 即使平移方向还未完全修正，也能先恢复基本可控性

### P1：纠正控制器目标状态所有权

修改：

- `src/lerobot/envs/fr3_mujoco.py`

动作：

- 引入 latched `reference_pose`
- teleop 启动或重新使能时锁存参考姿态
- 启用 teleop 后，目标姿态从控制器持有的参考目标推进，而不是从执行器当前姿态重建

预期收益：

- 消除“求解器误差 -> 下一个周期目标基线”的正反馈链路

### P2：显式化 SpaceMouse 轴映射

修改：

- `src/lerobot/teleoperators/spacemouse/configuration_spacemouse.py`
- `src/lerobot/teleoperators/spacemouse/teleop_spacemouse.py`
- `tools/fr3/fr3_mujoco_teleop.py`

动作：

- 用显式 `3x3` 轴映射矩阵替代硬编码 `[-raw_y, raw_x, raw_z]`
- 将 bias 去除与坐标变换分开
- 为 FR3 仿真提供明确默认映射

预期收益：

- 系统性 `-x/-z` 偏移变成可验证、可调试、可测试的问题

### P3：补充调试和验证

修改：

- `tests/teleoperators/test_spacemouse.py`
- `tests/envs/test_fr3_mujoco.py`
- 必要时增加新的测试文件

动作：

- 增加单轴输入到机器人命令的映射测试
- 增加“禁转时姿态不漂移”的回归测试
- 增加“重复小步命令不积累姿态漂移”的回归测试
- 增加“IK 解写回 MuJoCo 后，真实 TCP 必须落在目标 frame” 的执行层回归测试

## 验收标准

1. 单轴推动 SpaceMouse 时，机器人只沿预期轴向运动
2. `enable_rotation=False` 时，重复平移操作不再出现“低头”
3. 对角线平移只产生预期的合成位移，不出现额外俯仰
4. 单元测试和 FR3 MuJoCo 相关回归测试全部通过

## 实施策略

本次实施按以下顺序推进：

1. 先修 `kinematics.py` 与 `fr3_mujoco.py`
2. 再修 `teleop_spacemouse.py` 的映射模型
3. 最后补测试和最小调试能力

这样做的原因是：

- 先止血，立即去掉最危险的姿态漂移
- 再修坐标映射，避免把“方向错”和“姿态漂”混在一起调参
- 最后用测试把这两个问题分层锁死，防止回归

## 2026-04-21 补充结论

对 `FR3MujocoEnv` 做单轴注入后，已经确认：

1. `target_pose` 本身会按期望方向变化
2. 旧的 `LocalKinematicsDriver` 会给出一组在 `placo FK` 中正确、但在 MuJoCo 中错误的关节解
3. MuJoCo 自带 `_MujocoArmKinematics` 则能在真实仿真模型里把 TCP 准确送到 `+x/+y/+z` 目标

因此，`spacemouse 移动时末端向 -x/-z 方向移动且低头` 的核心根因已收敛为：

- 输入层存在映射问题
- 控制层曾存在姿态锁不严的问题
- 但当前仿真中最直接导致“target 正确、真实 TCP 反向”的主根因，是 **MuJoCo 仿真执行与 URDF/placo IK/FK 混用**

对应修复原则也因此明确为：

- FR3 MuJoCo 环境中，主 IK/FK 路径必须默认使用 MuJoCo 自身模型
- `URDF/placo` 只能用于离线对照或独立分析，不能再作为仿真主控制链的一部分

## 2026-04-21 再补充：OTG 与剩余下沉现象

在将 FR3 MuJoCo 环境的主 IK/FK 路径切到 MuJoCo 自身模型后，问题进一步收敛为两层：

1. 仿真 teleop 上的 OTG 会放大小步吞没问题
- 对于 `SpaceMouse` 的小增量输入，OTG 会先把关节目标切成更细的小步
- 在当前 MuJoCo FR3 arm 执行层里，这些小步在首个控制窗口内很容易被重力瞬态淹没
- 实测结果是：仿真 teleop 默认关闭 OTG 后，`+x` 已经能出现正向响应；显式开启 OTG 时，首个小步几乎完全被吞掉

因此，仿真 teleop 工具层先做了一个工程性止血：

- `tools/fr3/fr3_mujoco_teleop.py` 默认 `use_otg=False`
- `tools/fr3/fr3_mujoco_keyboard_teleop.py` 默认 `use_otg=False`
- 如需恢复旧行为，可显式添加 `--enable-otg`

这一步不是最终修复，但能明显减少“轻推一下几乎不动，或被错误瞬态主导”的问题。

2. 剩余的 `-z` 下沉和“低头感”来自 MuJoCo arm actuator 的执行特性
- 当前 FR3 arm 在 MuJoCo 中使用的是 `<position>` actuator，而不是 torque actuator
- 这类 actuator 的控制语义本质上是 `ctrl = q_target`
- 当 `ctrl` 刚好等于当前关节角时，初始恢复力矩接近 0
- 在没有显式重力补偿前馈的情况下，机械臂会先在重力作用下产生位置误差，然后 position servo 才开始拉回

直接证据：

- 即使完全不动 `SpaceMouse`，只保持当前 joint target，不断推进物理仿真，TCP 也会自己向 `-x/-z` 缓慢漂移
- 这说明“轻推一下就下沉”不再是输入层问题，而是执行层的静力不平衡与瞬态响应问题

因此，当前剩余问题的主根因已经从“映射错误”收敛为：

- 仿真 teleop 的 OTG 不适合当前这套小步 Cartesian 操作
- MuJoCo FR3 arm position actuator 在现有参数下对重力瞬态抑制不够

接下来的收敛方向应当集中在执行层，而不是继续修改 SpaceMouse 轴映射：

1. 仿真 teleop 默认禁用 OTG，保留显式 `--enable-otg` 作为对照开关
2. 调整 MuJoCo FR3 arm actuator 参数，提高对重力瞬态的抑制能力
3. 继续观测在 translation-only 模式下，低头现象中有多少来自真实姿态漂移，有多少只是 `TCP z` 下沉造成的视觉感受

## 2026-04-21 再补充：执行层收敛结果

在保留 “仿真 teleop 默认禁用 OTG” 的前提下，进一步给仿真 teleop 工具增加了 MuJoCo FR3 arm position actuator `kp` 覆盖能力，并将默认值提高到 `20000`：

- `tools/fr3/fr3_mujoco_teleop.py`
- `tools/fr3/fr3_mujoco_keyboard_teleop.py`
- `src/lerobot/envs/fr3_mujoco.py` 新增 `arm_actuator_kp`

这样做的目的不是改变控制语义，而是减小 position servo 在重力作用下的首拍下沉量。

在默认 teleop 配置下：

- `use_otg=False`
- `arm_actuator_kp=20000`
- `continuous_physics=True`

对连续 5 帧小步目标做实测，得到：

1. 纯 `+x` 命令
- 目标累计位移约 `[+0.009984, -0.000001, -0.000037] m`
- 真实 TCP 累计位移约 `[+0.009505, -0.000008, -0.000869] m`

2. 纯 `+z` 命令
- 目标累计位移约 `[-0.000016, -0.000001, +0.009963] m`
- 真实 TCP 累计位移约 `[-0.000472, -0.000009, +0.009145] m`

这说明：

- 主运动方向已经和目标方向一致
- 剩余误差主要表现为亚毫米到毫米级的 `z` 下沉与少量 `x` 串扰
- 不再是先前那种“目标在正向，但真实 TCP 整段朝 `-x/-z` 漂”的失控状态

对 translation-only 模式下的剩余“低头感”做姿态误差量化后，得到：

- 纯 `+x` 连续 5 帧后的姿态误差约 `0.117 deg`
- 纯 `+z` 连续 5 帧后的姿态误差约 `0.117 deg`

因此，当前剩余问题已经可以更准确地表述为：

- 主要问题已从“反向运动 + 明显低头”收敛为“较小的执行层跟踪误差”
- 视觉上的“低头感”更多来自 `TCP z` 位置误差和执行层串扰，而不再是大的姿态锁失效
