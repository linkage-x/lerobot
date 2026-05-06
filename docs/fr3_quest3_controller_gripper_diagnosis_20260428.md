# FR3 Quest3 Controller 夹爪无法抓起方块 — 诊断讨论 2026-04-28

## 问题

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_mujoco_teleop.py \
  --teleop-type quest3 --quest3-use-controller
```

Quest3 控制器模式下，夹爪无法将红色方块夹起来。

## 事实更新 (2026-05-06)

以下事实优先于本文前面较早阶段的推断：

- **夹爪闭合会被物体宽度限制。** 这说明手指与方块已经建立接触，问题不能再表述为“完全碰不到方块”。
- **失败发生在抬升阶段。** 夹爪闭合后再抬起，方块不会随夹爪一起上升。
- **控制语义不是当前主矛盾。** controller 模式下 trigger 的开/关语义即使有改进空间，也不足以单独解释“已接触、但提不起来”这个现象。
- 因此，本文后文中把问题直接归结为“控制语义反了”或“手指几何必然无解”的表述，都应该视为**待继续验证的假说**，而不是已经坐实的结论。

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
- 上述现象说明：**夹爪闭合确实受到物体宽度影响，接触不是缺失的**
- 抬起夹爪后，**方块不会随夹爪一起上升**；一次观测中方块下沉 dz≈-0.02m（被压入桌面而非提起）
- 参数修改 (kp 50→200, friction 2→5, density 500→50) 对结果**完全无影响** — 这是几何/接触问题，不是摩擦/力的参数问题
- 较早一轮 full FR3 + IK 的单次测试里曾出现 “TCP 未能到达目标位置、未完成抓起”，但这不能再作为当前结论；后续已确认默认 full FR3 teleop 路径存在可成功抓起的实际运行。

### 当前假说（待继续验证）

目前更稳妥的说法是：

- **问题已经进入“持握失败 / 抬升失败”阶段，而不是“接触建立失败”阶段。**
- 手指 STL mesh / 桌面 / 方块三者的几何关系仍然高度可疑，但基于当前现象，最多只能说它可能导致接触法向、有效摩擦锥或抬升时的受力方向不对。
- 还不能仅凭“手指较长”这一点就下结论说场景几何**必然无解**。
- 后续定位重点应该放在：抬升瞬间的接触点分布、法向方向、actuator force / tendon length、以及 TCP 轨迹是否在抬升前已经把方块压入桌面。

### 抬升阶段直接证据 (2026-05-06)

本轮使用脚本 [fr3_quest3_pika_lift_diagnostics.py](/home/hanyu/Codes/lerobot/tools/fr3/fr3_quest3_pika_lift_diagnostics.py) 对 close-then-lift 轨迹做了结构化采样，直接记录：

- `actuator_force`
- `tendon_length`
- `workspace_object` 相关接触点、接触法向、接触法向力
- 方块中心高度、方块底面相对桌面顶面的高度差

对比了两条相邻轨迹：

- **成功对照**：`x=0.47, close_z=0.30, lift_zs=[0.32,0.34,0.36,0.38]`
- **失败轨迹**：`x=0.47, close_z=0.34, lift_zs=[0.36,0.38,0.40,0.42]`

关键事实：

- **成功轨迹**中，close 结束时仍是 `2` 个 finger-object 接触；进入 lift 后，这 `2` 个 finger-object 接触一直保持存在。
- 成功轨迹里，方块中心高度从 `0.445107` 持续上升到 `0.476147`，`object_bottom_minus_table_top` 从 `0.004728` 增长到 `0.035926`，说明方块稳定离桌。
- 成功轨迹里，`actuator_force` 从 `4.935` 逐步降到 `2.446`，`tendon_length` 从 `-0.098736` 逐步收缩到 `-0.048944`，表现为“夹着物体一路上抬，同时夹爪继续收口”。

- **失败轨迹**中，close 结束时同样存在 `2` 个 finger-object 接触，且 `actuator_force=4.324`、`tendon_length=-0.086518`，说明 actuator 并没有在 close 阶段失效。
- 但失败轨迹一进入第一次 upward lift（`tcp_z: 0.34 -> 0.36`），接触集合就从 `2` 个 finger-object **切换为 `4` 个 table-object 接触**；采样点中已经看不到 finger-object 接触。
- 失败轨迹里，方块中心高度从 close 末尾的 `0.456534` 掉到 `0.439596`，`object_bottom_minus_table_top` 变为 `-0.000404`，最低到 `-0.000990`。这说明**方块在抬升初段并没有被夹着上去，而是回落并轻微压入桌面**。
- 失败轨迹的 table 接触法向全部为 `[0, 0, 1]`，接触点分布在方块底面四角附近，说明失败后主导接触已经变成“方块 resting on table”，不是“方块 held by gripper”。
- 失败轨迹里，early lift 阶段 `actuator_force` 仍维持在 `4.17` 左右，`tendon_length` 仍在 `-0.083` 左右；直到更高的 `tcp_z=0.40/0.42`，左右 finger qpos 才接近 `0`，`actuator_force` 与 `tendon_length` 也塌到接近 `0`。这说明**先发生的是丢失持握 / 方块落桌，随后才是夹爪完全闭合到空载状态**。

这一轮证据支持如下判断：

- 问题不是“夹爪根本闭不上”。
- 问题也不是“close 阶段 actuator 没出力”。
- 更像是：**高 close pose (`close_z=0.34`) 下虽然能形成瞬时 finger-object 接触，但该接触在 upward lift 开始后无法维持，方块很快转为桌面支撑状态。**
- 因此，后续重点应放在“为什么这组高位接触不能形成可保持的摩擦夹持”，而不是继续优先怀疑 gripper open/close 语义。

### close_z 临界区间细扫 (2026-05-06)

继续沿 `x=0.47` 固定、`close_z` 从 `0.30` 扫到 `0.34`，并比较：

- close 末尾的 `finger-object` 接触点高度
- close 末尾的接触法向
- 第一次 upward lift (`tcp_z += 0.02`) 后，接触是否仍保持为 `finger-object`

粗扫结论：

- `close_z = 0.300 ~ 0.335` 时，close 末尾接触表现非常平滑：
  - 平均接触高度从 `0.416295` 单调上升到 `0.435927`
  - 平均接触法向基本稳定在 `[0, 0, -0.53x]`
  - 第一次 upward lift 后仍保持 `2` 个 `finger-object` 接触
  - 方块中心高度每次都稳定上升约 `+0.0111m`

- 到 `close_z = 0.340` 时，表现不是继续平滑变化，而是**直接跳相**：
  - close 末尾接触点高度从上一档的 `0.436327 / 0.435527` 突然变成 `0.399798 / 0.399798`
  - 接触法向 z 分量从稳定的 `-0.531 / -0.531` 跳到 `-0.407 / -0.420`
  - 第一次 upward lift 后不再保留 `finger-object`，而是直接变成 `table-object`
  - 方块高度变化从约 `+0.011m` 突然翻成 `-0.0169m`

进一步把阈值细化到 `0.0002m` 步长后，翻转点落在：

- `close_z = 0.3396`：仍然成功，仍保持 `finger-object`
- `close_z = 0.3398`：已经失败，第一次 upward lift 后只剩 `table-object`

也就是说，当前场景里存在一个非常窄的临界带，约在：

```text
close_z ∈ (0.3396, 0.3398)
```

这一轮细扫说明：

- 问题不是随着 `close_z` 升高而“逐步变差”。
- 更像是接触几何在高位触发了一个**离散的接触模式切换**：
  - 低于阈值时，接触落在可维持夹持的 finger 工作面上；
  - 高于阈值时，接触突然迁移到另一组几何特征上，导致 upward lift 一开始就丢失 finger-object 持握。

这也解释了为什么 `0.3396` 和 `0.3398` 只差 `0.2mm`，结果却从“稳定抬起”直接变成“落桌”。

### 0.3396 vs 0.3398 接触几何对照 (2026-05-06)

进一步只对比两档：

- `close_z = 0.3396`：临界带下方，仍可抬起
- `close_z = 0.3398`：临界带上方，第一次 upward lift 后立即丢失持握

对比方式：

- 将 close 末尾接触点变换到 **finger body 局部坐标**
- 同时变换到 **方块局部坐标**
- 由方块局部坐标判断接触落在顶边 / 侧边 / 下缘哪个区域

#### 1. `0.3396`：左右 finger 仍是对称夹持

close 末尾两侧接触都落在方块**下缘附近**，但左右是对称的：

- 左指接触点（方块局部）约为 `(+0.0020, +0.0250, -0.0400)`
- 右指接触点（方块局部）约为 `(-0.0238, -0.0250, -0.0400)`

可归类为：

- 左指：`edge(-z, +y)`
- 右指：`edge(-z, -y)`

也就是两侧都还在夹方块的左右侧下缘，仍构成一组可维持的 opposing contacts。

对应 finger 局部坐标：

- 左指：`(x,y,z) ≈ (0.0198, -0.0278, 0.0067)`
- 右指：`(x,y,z) ≈ (0.0221, 0.0292, -0.0188)`

两者都位于各自 mesh 的外缘/前缘附近，但还保持左右镜像关系。第一次 upward lift 后，这两个 `finger-object` 接触继续存在。

#### 2. `0.3398`：左指先跳到另一类接触

close 末尾时，右指仍大致在原来的右侧下缘：

- 右指接触点（方块局部）约为 `(-0.0052, -0.0250, -0.0400)`
- 归类仍是 `edge(-z, -y)`

但**左指已经不在左/右侧下缘这组对称接触里了**，而是跳到了：

- 左指接触点（方块局部）约为 `(-0.0250, -0.0033, -0.0400)`
- 归类为 `edge(-z, -x)`

这意味着左指接触从“方块 `+y` 侧下缘”切到了“方块 `-x` 侧下缘”。

也就是说，不是两指同时平滑上移，而是：

- **右指还在原本的侧面夹持区域附近**
- **左指先跳到另一条下缘 / 侧棱上**

这正是对称夹持被破坏的开始。

对应 finger 局部坐标也发生了明显变化：

- 左指：从 `0.3396` 的 `(0.0198, -0.0278, 0.0067)` 跳到 `0.3398` 的 `(0.0305, -0.0333, 0.0127)`
- 右指：从 `(0.0221, 0.0292, -0.0188)` 变到 `(0.0329, 0.0348, 0.0128)`

其中左指最关键：它不再对应方块的 `+y` 侧面接触，而是迁移到了 `-x` 侧边/下缘附近。

#### 3. 法向也印证了“左指先跳相”

`0.3396` close 末尾：

- 左指 world normal 约 `( +0.032, +0.847, -0.531 )`
- 右指 world normal 约 `( -0.032, -0.847, -0.531 )`

这是非常典型的左右镜像。

`0.3398` close 末尾：

- 左指 world normal 约 `( -0.632, +0.660, -0.407 )`
- 右指 world normal 约 `( -0.659, -0.624, -0.420 )`

左指的 x 分量从接近 `0` 突然变成大幅负值，说明它的受力法向已经明显转到另一块斜面/侧棱上，不再是原先那组近似镜像的工作面。

#### 4. 结论

`0.3396 -> 0.3398` 的失稳，不是“两边一起慢慢变差”，而更像是：

- **左指先从方块侧面下缘接触跳到了另一条下缘/斜面接触**
- 右指还留在原来的侧面附近
- 于是原本的 opposing contacts 被破坏
- 第一次 upward lift 后，`finger-object` 无法维持，接触立即退化成 `table-object`

所以当前最可信的微观机制是：

- **不是某一侧单纯“力不够”**
- 而是临界高位下，左指先发生了接触区域切换，导致夹持拓扑从“左右对称夹持”变成“错位接触”，抬升时立刻失稳**

### 左指跳相区域的简化碰撞近似实验 (2026-05-06)

基于上面的局部坐标，又做了一轮“只改左指碰撞”的实验，目标是验证：

- 能否把左指跳相区域替换成更稳定的简单碰撞近似
- 并且同时满足：
  - `close_z=0.3398` 从失败变成功
  - `close_z=0.3396` 保持原本成功

实验方式分两类：

1. **整块替换左指 mesh collision**
   - 单个斜放 box
   - 两个 box 组合
   - 小 capsule

2. **保留左指 mesh collision，仅额外补一个局部 stabilizing patch**
   - 小 box patch
   - 小 capsule patch

结果：

- **整块替换**没有找到可行候选。
  多组 box / capsule 近似都会把 `0.3396` 和 `0.3398` 一起打坏，常见结果是：
  - 方块被异常弹飞后落桌
  - 或夹爪几乎空载闭合，最终 `object_center_z` 掉到 `0.04 ~ 0.06`

- **局部补 patch** 也没有找到可行候选。
  最好的几组也只能把失败模式改成另一种失败模式，例如：
  - `0.3398` 仍然无法 lift
  - `0.3396` 也被一起拉回 `table-object` 支撑状态

当前可以下的结论是：

- **“用几个手工 box/capsule 直接替掉左指跳相区域”目前看不可行。**
- 问题不是简单的“这里缺一小块接触面”，而更像是左指 mesh 的整段工作几何与右指、方块下缘之间共同决定了稳定夹持拓扑。
- 也就是说，想靠一个局部 primitive 临时补丁把 `0.3398` 拉回成功，至少在当前这轮实验里没有证据支持。

这轮实验的工程含义：

- 如果后续真要改碰撞近似，方向更应该是：
  - 基于现有 mesh 工作面做**成体系的简化碰撞建模**，而不是手补一两个 box
- 或直接对任务位姿加约束，把 `close_z` 限制在临界值以下

而不是继续期望“局部打一块补丁”就能稳定修复。

### STL 三角面层面的根因补充 (2026-05-06)

进一步把左指在 `0.3396` 和 `0.3398` 的接触点投到 `pika_gripper_left_link.STL` 三角面上后，发现这不是“接触点在同一工作面上轻微漂移”，而是**跨到了另一组法向明显不同的三角面**。

关键观察：

- `0.3396` 时，左指接触点最近的三角面集中在 STL 的一组局部面片上，这些三角面的法向大致是：
  - `(-0.73, -0.68, -0.08)`
  - `(-0.71, -0.66, -0.23)`
  - `(-0.64, -0.59, -0.50)`

  这说明当时左指接触仍落在一组**朝下/朝侧方**的工作面上，和前面看到的 `finger-object` 稳定夹持是一致的。

- `0.3398` 时，左指接触点最近的三角面换成了 STL 上另一组局部面片，这些三角面的法向大致是：
  - `(-0.69, -0.63, +0.36)`
  - `(-0.66, -0.61, +0.43)`
  - `(-0.64, -0.59, +0.50)`
  - `(-0.61, -0.56, +0.56)`

  注意这里的 **z 分量从负变正**，说明接触不再落在原来的下压/侧压工作面上，而是跳到了一个**上扬的斜面/棱面族**。

这一步把前面的“接触模式切换”进一步具体化为：

- `0.3396`：左指接触还在一组可用于对称夹持的下缘/侧缘工作面上
- `0.3398`：左指接触跨过一条 STL 局部锐边/面片分界，跳到上扬斜面

因此，当前最可信的几何根因已经可以收敛为：

- **左指 mesh 在临界高位附近存在一组面片法向突变很大的局部几何特征**
- 方块一旦在该高度触发这组面片，左指接触法向就不再和右指形成稳定的对向夹持
- 于是第一次 upward lift 时，接触拓扑立刻失稳并退化成 `table-object`

换句话说，根因不是抽象的“mesh 太复杂”，而是：

- **左指 STL 某个局部棱边/斜面分界在 `close_z ≈ 0.3397` 附近被离散触发**

### close-phase z clamp 回归与回退 (2026-05-06)

曾尝试过一个运行时规避方案：

- **只在“准备闭合 / 正在闭合”阶段限制 TCP z，要求 `close_z < 0.3397`**

这个方案在纸面上看起来合理，但实测引入了新的坏行为，已回退。回归现象包括：

- 实测仍然夹不起来
- 松开瞬间夹爪会“消失几帧”
- MuJoCo 报出 `QACC` 不稳定警告

对比脚本证据表明，这个 clamp 方案本身就是新回归来源：

- clamp 开启时，第一次 close step 的 `target_z` 被压到 `0.3397`
- 但实际 `tcp_z` 会跳到异常值，左右 finger 也会过早收成接近 `0 / 0`
- 随后 release 阶段出现数值不稳定

因此，这条路已经被否定：

- **不要在 env 里通过硬改 mocap target z 来规避这个临界带**

目前更稳妥的结论是：

- 如果后续还要做“高度规避”，也应该放在更上层的 teleop / 策略逻辑里，以软约束或用户提示方式实现
- 不应再在 MuJoCo env 的底层 target pose 上做这种硬钳制

### full FR3 默认 teleop 路径可抓起的约束意义 (2026-05-06)

用户补充了一条重要现象：

```bash
docker compose -f docker/docker-compose.yml --profile sim --profile teleop run --rm \
  -e DISPLAY=$DISPLAY \
  -e PYTHONPATH=/workspace/src \
  lerobot-fr3-sim-teleop \
  python tools/fr3/fr3_mujoco_teleop.py \
  --enable-cameras --camera-width 640 --camera-height 480
```

这条命令在默认配置下**可以把方块抓起来**。

这条现象的约束力很强，因为按当前代码路径，它默认并不是：

- `Quest3 + quest3_pika_gripper_scene`

而是：

- **`SpaceMouse + FR3MujocoEnv(full FR3 arm)`**

对应分流逻辑见：

- `tools/fr3/fr3_mujoco_runtime.py:372` — 只有 `teleop_type=quest3` 且 `quest3_scene_mode=pika_gripper` 才会进 `Quest3PikaMujocoEnv`
- `tools/fr3/fr3_mujoco_runtime.py:376` — 否则走 `FR3MujocoEnv`

更关键的是，对比 XML 后可以确认：

- 桌面几何相同
- `workspace_object` 方块几何相同
- 左右 finger collision 仍然是同一套 `pika_gripper_left_link.STL` / `pika_gripper_right_link.STL`
- gripper actuator 仍然是同一个 tendon position actuator

因此，这条成功现象直接否定了一个过强结论：

- **不能再说“这套 Pika finger collision mesh 全局必然抓不起来”。**

当前更准确的说法应该是：

- **同一套 finger collision mesh 在 full FR3 路径下可以成功抓取，但在 Quest3 Pika direct-gripper 路径下会被带进一个很窄的离散接触切换带。**

这会把根因进一步收窄为“局部几何 + 控制链 / 接近轨迹”的耦合问题，而不是“左指 STL 本体在任何路径下都绝对错误”。

更具体地说，当前最可信的解释是：

- full FR3 路径中的 arm IK / joint chain / TCP 接近轨迹，使系统自然避开了 `close_z ≈ 0.3397` 的坏带，或以不同姿态进入接触
- Quest3 Pika direct scene 中，gripper base 被直接 mocap 驱动，更容易把左指推进那组上扬三角面
- 因而真正失败的不是“是否存在可抓姿态”，而是“Quest3 Pika 直驱路径是否会把系统送入局部接触 bifurcation”

这条事实的工程含义：

- 问题已经不应再被表述为“gripper mesh 本身完全不可用”
- 更应表述为：**Quest3 Pika direct-gripper 路径对同一套 gripper collision geometry 更脆弱，更容易触发左指局部接触面簇切换**
- 下一步最有区分度的验证，就是跑 `Quest3 + full FR3 arm + controller`，看 Quest3 controller 输入本身是否也会触发同类失败；如果该路径能抓起，则问题将更集中到 `Quest3PikaMujocoEnv / quest3_pika_gripper_scene` 这条 direct-gripper 路径

## 修复路线

### 截至 2026-05-06 的建议优先级

基于当前证据，后续动作不应再围绕“是不是 controller 语义反了”或“再调一调 kp / friction”展开，而应围绕**左指局部几何导致的离散接触切换**来做。

#### P0 — 保留证据链，避免回到已否定路线

当前已经落地的经验应视为基线：

- 问题发生在 **lift onset**，不是 close 阶段完全无接触
- 临界带极窄，约在 `close_z ∈ (0.3396, 0.3398)`
- 触发点是 **左指先跳相**，不是两边连续退化
- `close-phase z clamp` 已被证明会引入新回归，不能作为 env 层修复

#### P1 — 直接面向 mesh 局部几何做修复

更合理的主线是：

- 针对 `pika_gripper_left_link.STL` 在该局部锐边/斜面分界附近做几何重建
- 目标不是“补一块 primitive”，而是让左/右 finger 在临界高度附近仍保持同类 working face
- 修复判据应直接看：
  - `0.3396` 继续成功
  - `0.3398` 从失败翻回成功
  - lift 后仍保持 `finger-object` 接触

可行实现方向：

- 改 STL / collision mesh 本体，抹平这组局部法向突变过大的面片
- 或为左右 finger 一起重建一套成对的 simplified collision，而不是只补左指单边 patch

#### P2 — 如果短期只求稳定跑通，用上层策略绕开临界带

如果目标是先稳定采集或先让 Quest3 流程跑通，可以在更上层做规避，但不要在 env 里硬改目标位姿：

- 在 teleop / policy 层做闭合高度软门控或提示
- 避免用户在 `close_z ≈ 0.3397` 附近触发闭合
- 一旦已经形成稳定持握，后续 lift 不应再受限

这条路的定位是 **workaround**，不是根修。

#### P3 — 比例扳机控制仍可作为辅助手段

把 controller 模式从二值开合改成比例开合仍然值得做，因为它会改善操作手感和接近阶段的可控性；但按当前证据，它不是“接触建立后仍抬不起物体”的主根因。

### 已排除的假说

以下假说已被实测排除或显著降级：

- **close settle 缺失**：`continuous_physics` 默认为 True
- **kp 不足**：`kp 50 -> 200` 无效果
- **摩擦不足**：finger friction `2 -> 5` 无效果
- **方块太重**：density `500 -> 50` 无效果
- **二值扳机是主根因**：比例关闭测试与二值关闭结果同类失败
- **局部 primitive patch 能直接修好左指跳相区**：多组 box / capsule 实验未找到可行候选
- **env 层 close-phase z clamp 可安全规避问题**：已引入夹爪异常消失与数值不稳定，故已回退

## 参与讨论

- 💻 **Amelia** — 代码追踪与修复优先级
- 🔬 **Dr. Quinn** — TRIZ 系统根因分析
- 🏗️ **Winston** — 架构评估与方案取舍
- 🧪 **Murat** — 风险评估与诊断验证策略
