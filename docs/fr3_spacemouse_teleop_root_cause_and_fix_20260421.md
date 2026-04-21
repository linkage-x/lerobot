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
