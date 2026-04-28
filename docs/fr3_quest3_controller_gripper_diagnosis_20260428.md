# FR3 Quest3 Controller 夹爪无法抓起方块 — 诊断讨论 2026-04-28

## 问题

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_mujoco_teleop.py \
  --teleop-type quest3 --quest3-use-controller
```

Quest3 控制器模式下，夹爪无法将红色方块夹起来。

## 关键管线逻辑

```
Quest3 控制器 → Vuer WebSocket → Quest3Teleop.get_action()
    → _controller_gripper() [binary: 0.0|1.0]
    → Quest3PikaMujocoEnv.step_teleop_action()
    → _set_gripper_command() → MuJoCo position actuator kp=50 on tendon
    → 手指 mesh collision ←→ 红色方块 box collision
```

### controller 模式 gripper 是二值的

`src/lerobot/teleoperators/quest3/teleop_quest3.py:370`：

```python
def _controller_gripper(self) -> float:
    if self._controller_states.get("right_trigger", 0.0) > 0.01:
        return 1.0  # open
    else:
        return 0.0  # close
```

右扳机 > 0.01 → 全开 (1.0)，否则 → 全关 (0.0)。**无比例控制。**

对比：hand tracking 模式使用 `_gripper_from_pinch_value()` 将 pinch_value [0.111→0.004] 平滑映射到 [0.0, 1.0]，是连续比例控制。

### close settle 默认不生效

`src/lerobot/envs/quest3_pika_mujoco.py:522`：

```python
if self.cfg.continuous_physics and self._last_gripper < previous_gripper:
    settle_steps = max(int(self.cfg.quest3_gripper_close_settle_steps), 0)
    if settle_steps > 0:
        self._step_physics(settle_steps)
```

`continuous_physics` 默认 `False`，因此 **close settle 的 120 额外步从不执行**。

### 场景物理参数

- Gripper actuator: position type, `ctrlrange="-0.11 0"`, `kp="50"`, `dampratio="1"`
- Finger collision: STL mesh, `friction="2.0 0.4 0.03"`, `condim="4"`
- Red block: box geom, `density="500"`, `friction="1.8 0.05 0.01"`, `condim="4"`
- 平行夹爪：tendon + equality constraint 保证 `left_qpos = -right_qpos`

## 根因分析

### Amelia 的代码追踪

**根因：binary 关闭 + continuous_physics=False → 无 grip settle。**

单步物理不足以建立手指-方块的接触——position actuator 贯穿全行程，方块在被静态摩擦稳住之前就被弹飞或滑走。

次要因素：
- 方块 `friction="1.8 0.05 0.01"` — 扭转/滚动摩擦太低，提起后方块易旋转脱落
- `kp=50` + 二进制关闭 — 高刚度瞬时关闭产生冲击力打飞方块
- 无比例扳机映射 — 用户无法"轻轻握"

### Dr. Quinn 的 TRIZ 系统分析

五环失败链路：

| 环节 | 问题 | 影响 |
|------|------|------|
| 第一环：二值开关 | trigger 数字化为开/关 | 失去精细控制 |
| 第二环：软弱无力 | kp=50，初始力矩仅 ~1.5N | 夹不紧 |
| 第三环：无稳合时间 | close settle 被 continuous_physics 条件跳过 | 接触未稳固即被切断 |
| 第四环：无保持状态 | trigger 松→全关/按→全开，无中间态 | 纯开环 |
| 第五环：接近角度不可控 | 无 FK/IK，人手直驱 mocap | 可能斜撞，mesh 几何包不住 |

判断这是**系统级时序-力-控制三重不匹配**，单一修复不足，需组合方案。

### Winston 的时序分析

```
T0: 松扳机 → cmd=1.0 → actuator kp=50 收肌腱
T1: 手指触方块 → 碰撞力产生
T2: 物理在 settle steps 内达稳态
T3: 用户抬手 → 方块是否跟随取决于 T2
```

瓶颈在 T1→T2：settle 步未执行，actuator 未压紧到稳态就迎来抬手动作。

### Murat 的风险评估与诊断建议

根因假说（按置信度）：

1. **#1 (最高)** — 手指关到目标位置，但接触力不足以夹起。验证：dump `mjData.actuator_force` 和 tendon 长度 vs 目标长度。
2. **#2** — 方块被手指推飞。验证：看接触点数量和法向量方向。
3. **#3** — close settle 缺失是关键变量但非根因。验证：先 `continuous_physics=True` 跑一次。

**二分诊断第一步：** 在方块正上方关合，不动方块说明 actuator 到位 OK，问题在接触阶段。

## 实测结果 (2026-04-28)

### continuous_physics 默认已是 True

`fr3_mujoco_runtime.py:80` 中 argparse 默认 `continuous_physics=True`，close settle (120步) 实际在运行。**settle 缺失不是根因。**

### 关键发现：手指 STL mesh 穿桌

| 状态 | 手指体 Z (世界) | 指尖 Z (估计) | 方块 Z | 桌面 Z |
|------|:---:|:---:|:---:|:---:|
| XML 默认 (mocap_z=0.083) | 0.341 | ~0.23 | 0.44 | 0.40 |
| Quest3 初次 reset (tcp_z=0.55) | 0.659 | ~0.55 | 0.44 | 0.40 |
| 手指与方块等高 (z=0.44) | 0.44 | ~0.33 | 0.44 | 0.40 |

**手指 STL mesh 从体中心向下延伸约 11cm**。要碰到 z=0.44 的方块，手指体必须在 z=0.44，指尖就到 z≈0.33 — 穿透桌面 7cm。

### 实测抓取行为

- 手指在方块高度关闭时，qpos 被方块限制在 ±0.036（未完全闭合），接触力约 0.5N/指
- 抬起夹爪后，方块下沉 dz≈-0.02m（被压入桌面而非提起）
- 参数修改 (kp 50→200, friction 2→5, density 500→50) 对结果**完全无影响** — 这是几何/接触问题，不是摩擦/力的参数问题
- 即使 full FR3 scene + IK 也无法在本次测试中夹起方块（TCP 未能到达目标位置）

### 根因

**手指 STL mesh 碰撞几何过长，在方块高度处操作必然穿透桌面。** 接触力学不稳定，方块被桌面手手指联合挤压而非被摩擦力夹起。这不是控制参数问题，是场景物理几何设计问题。

## 修复路线

### P0 — 确认 full FR3 SpaceMouse 路径是否正常

先跑 SpaceMouse 默认路径确认 full FR3 arm 场景本身可以夹起方块：
```bash
docker compose ... python tools/fr3/fr3_mujoco_teleop.py
```
如果 full FR3 可以、Quest3 Pika 不行 → 确认是 quest3_pika_gripper_scene 的几何问题。

### P1 — 修复手指碰撞几何

两个方向（可并行尝试）：

**方向 A: 调整手指/方块/桌面的相对高度**
- 将 `workspace_object_body` 的初始 z 提高 (如 0.44→0.50)，使方块位于手指 mesh 的可及范围内
- 或降低桌面高度
- 风险：改变了抓取任务的空间布局，影响数据一致性

**方向 B: 用 box 碰撞体替代/补充 STL mesh**
- 为手指添加简单 box collision geom，精确放置在方块高度
- 保持 STL mesh 碰撞关闭或作为辅助
- 这类似 full FR3 的做法 (`fr3_pika_ati.xml` 手指 collision 用的也是 mesh，但 arm IK 可以把夹爪放到精确位置)

### P2 — 比例扳机控制（辅助改进）

改 `teleop_quest3.py:370`：
```python
def _controller_gripper(self) -> float:
    trigger = self._controller_states.get("right_trigger", 0.0)
    return 1.0 - trigger  # trigger越深越关
```
实测表明比例控制不能替代几何修复，但仍是值得做的控制器改进。

### 已排除的假说

以下假说已被实测排除：
- **close settle 缺失**：`continuous_physics` 默认为 True
- **kp 不足**：kp 50→200 无效果
- **摩擦不足**：finger friction 2→5 无效果
- **方块太重**：density 500→50 无效果
- **二值扳机**：比例关闭 (gradual close) 测试结果与二值相同 (qpos=-0.036, dz≈-0.02)

## 参与讨论

- 💻 **Amelia** — 代码追踪与修复优先级
- 🔬 **Dr. Quinn** — TRIZ 系统根因分析
- 🏗️ **Winston** — 架构评估与方案取舍
- 🧪 **Murat** — 风险评估与诊断验证策略
