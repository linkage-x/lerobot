# FR3 + DAS Gripper + SLAM 开发指导文档（当前冻结版）

## 1. 目的

本文档用于冻结当前阶段已经明确的坐标系、外参、数据语义与部署链路，避免后续在以下环节出现 frame contract 不一致的问题：

- 数据采集
- 数据集导出
- policy 训练
- replay
- 在线控制
- IK 接口

当前文档的目标不是解决所有问题，而是让后续开发能够在统一、可验证、可追责的假设下继续推进。

## 2. 当前已经明确的结论

### 2.1 坐标系定义

采用如下 frame：

| 符号 | 含义 |
| --- | --- |
| `B` | FR3 base frame |
| `E` | end-effector / TCP frame |
| `I` | DAS gripper IMU body frame |
| `W_s` | 每个 episode 的 SLAM world frame |

其中 `W_s` 定义为该 episode 首帧 IMU body frame：

$$
W_s \equiv I_0
$$

这意味着：

- `W_s` 是 episode-local
- 不同 episode 之间的 `W_s` 不可直接比较
- 所有 SLAM pose 只能在本 episode 内自洽使用

### 2.2 变换记号

统一采用：

$$
\mathbf{p}_A = T(A, B)\,\mathbf{p}_B
$$

即 `T(A,B)` 把一个在 `B` 中表达的点变到 `A`。由此：

- `T(W_s, I_t)`：SLAM 输出
- `T(I, E)`：IMU 到 TCP 的固定外参
- `T(B, E_t)`：FR3 base 下的 TCP pose

## 3. E frame 的正式定义

### 3.1 E 的原点

当前冻结为：

> E 原点位于两指尖中心 TCP 附近

结合 MuJoCo 多视角可视化，这个位置在几何上是合理的。

### 3.2 E 的轴语义

当前正式冻结如下：

| 轴 | 语义 |
| --- | --- |
| `-z_E` | 接近方向（approach） |
| `-y_E` | 夹爪闭合方向（closing） |
| `x_E` | 法向（normal） |

E 为右手系。即在代码或可视化中：

- `+z_E` 是接近方向的反向
- `+y_E` 是闭合方向的反向
- `+x_E` 是法向

此定义后续必须在数据集字段说明、replay、policy label、runtime adapter、IK 接口中统一使用。

## 4. I 和 E 的来源与当前状态

### 4.1 I 的来源

`I` 当前定义为 `link_imu`，来自 DAS_Gripper_V4 中 IMU 对应的 fixed joint。

因此已知 $T(\text{base}, I)$，这里的 `base` 指夹爪机械 base frame。

### 4.2 E 的来源

`E` 当前定义为 `das_gripper_ee`，来自补充 URDF 中：

```
parent = gripper_base_link
child  = das_gripper_ee
```

因此已知 $T(\text{base}, E)$。

### 4.3 T(I,E) 的构造方式

当前采用：

$$
T(I, E) = T(\text{base}, I)^{-1}\,T(\text{base}, E)
$$

这是目前候选固定外参的来源。

## 5. T(I,E) 的当前结论

### 5.1 是否已经能确定

可以确定一个候选固定外参 `T(I,E)`。它不是拍脑袋得到的，而是由两份 URDF 链条组合得到。

### 5.2 是否已经可信

- 在位置层面，已经通过 MuJoCo 可视化 sanity check
- 在几何语义层面，已具备较强可信度
- 在工程层面，可以进入正式开发使用，但应标注为"候选冻结版"

### 5.3 仍未完全闭环的部分

还需要继续验证的不是"位置有没有大错"，而是：

**A. 真实 IMU body frame 一致性**

需要确认真实设备中 SLAM 所使用的 IMU frame 是否与 URDF 中 `link_imu` 完全一致。否则可能出现：

- 固定 90° / 180° 旋转偏差
- replay 位置看着对，姿态总不对

**B. 运行链路一致性**

需要在代码中验证：

$$
T(B, E_t) = T(B, W_s)\,T(W_s, I_t)\,T(I, E)
$$

以及 replay / FK consistency 是否成立。

## 6. MuJoCo 可视化验证结论

### 6.1 已确认的现象

在修正 parent frame 后，多视角可视化显示：

- `I` 位于夹爪主体内部上方，符合 IMU 安装位置直觉
- `E` 位于两指尖中心附近，符合任务 TCP 位置直觉
- `I -> E` 连线从机身内部连向抓取工作区，几何上合理

### 6.2 因此可以得出的结论

当前 `T(I,E)` 至少满足：

- 原点位置合理
- 相对偏移合理
- 不存在显著的父子层级错误
- 不存在明显的左右翻转/前后颠倒/飞离模型问题

### 6.3 对开发的意义

当前外参已经足够支持继续写数据管线与 runtime adapter。后续主要是"验证和固化"，不是"从零猜测"。

## 7. MuJoCo 模型结构结论

### 7.1 das_base_frame 的角色

当前明确：

- `gripper_base`：装配容器 frame
- `das_base_frame`：MuJoCo 中对应 URDF `gripper_base_link` 的 frame

后续所有相对夹爪 base 定义的对象，都应挂在 `das_base_frame` 下，包括：

- `imu_vis`、`ee_vis`
- `link1` ~ `link6`

### 7.2 当前开发规则

以后凡是新增与 DAS base 相关的内容（新的 site、tcp_guess、传感器可视化、collision proxy、任务 marker），都应优先挂在 `das_base_frame`，而不是 `gripper_base`、`fr3_link7` 或 `world/root`。

## 8. 训练与部署的 contract

### 8.1 当前训练语义

当前 canonical action label 建议保持为：

$$
a_t = T(W_s, E_t^\star)
$$

即在 `W_s` 下表达的绝对 TCP pose，与当前录制/SLAM 体系更一致。

### 8.2 当前 observation 语义

建议 observation 中如果包含位姿，统一用 `E` 语义，而不是混用 `I`：

$$
o_t = \{\text{images}_t,\ \text{tactile}_t,\ T(W_s, E_t),\ \ldots\}
$$

如果底层原始 pose 仍是 `T(W_s, I_t)`，则应显式导出：

$$
T(W_s, E_t) = T(W_s, I_t)\,T(I, E)
$$

不要在不同模块中各自重复推导。

### 8.3 为什么不能忽略 T(I,E)

如果忽略 `T(I,E)`，直接把学到的 action 作用在 ee，本质上是在假设 $I \equiv E$，这不成立。结果会是：

- 位置产生固定偏差
- 姿态产生固定偏差
- 精细接触任务明显变差
- 训练语义和部署语义失配

因此，**部署时必须显式使用 `T(I,E)`**。

## 9. 在线控制链路（推荐实现）

### 9.1 episode 初始化

每个 episode 开始时应计算 $T(B, W_s)$。因为 $W_s \equiv I_0$，又有：

$$
T(B, E_0) = T(B, I_0)\,T(I, E)
$$

所以：

$$
T(B, W_s) = T(B, I_0) = T(B, E_0)\,T(E, I)
$$

其中 $T(E, I) = T(I, E)^{-1}$。

### 9.2 runtime pose 变换

对任意时刻 `t`：

$$
T(B, E_t) = T(B, W_s)\,T(W_s, I_t)\,T(I, E)
$$

如果 policy 直接输出的是 E 语义下的 action $\hat{T}(W_s, E_t^\star)$，则在线控制时应转换为：

$$
\hat{T}(B, E_t^\star) = T(B, W_s)\,\hat{T}(W_s, E_t^\star)
$$

然后送给 FR3 IK。

### 9.3 强制规则

| 规则 | 内容 |
| --- | --- |
| 规则 1 | 所有送入 IK 的 pose 必须是 `B` 下的 `E` pose |
| 规则 2 | 所有从 SLAM 来的 pose 默认是 `I` 语义，不能直接当 `E` 用 |
| 规则 3 | `T(I,E)` 只应在少数明确位置使用，推荐只允许在 dataset export 和 runtime control adapter 两处实现 |

## 10. 数据集字段建议（当前版）

### 10.1 episode-level 必备字段

| 字段 | 说明 |
| --- | --- |
| `episode_id` | |
| `start_timestamp` | |
| `q_0` | episode 起始关节角 |
| `T(B,E_0)` | 由 FR3 FK 给出 |
| `T(I,E)` | 固定外参 |
| `T(B,W_s)` | episode 初始化计算得到 |
| `robot_model_version` | |
| `extrinsic_calib_version` | |
| `slam_version` | |

如果没有 `T(B,W_s)`，则该 episode 不可直接部署。

### 10.2 frame-level 必备字段

| 字段 | 说明 |
| --- | --- |
| `frame_index` | |
| `timestamp` | |
| `image_left` | |
| `image_right` | |
| `image_third` | |
| `tactile` | |
| `joint_state` | |
| `T(W_s,I_t)` | 原始 SLAM 输出 |
| `T(W_s,E_t)` | 派生 observation pose |
| `action_pose_ws` | |

### 10.3 强烈推荐的派生字段

| 字段 | 理由 |
| --- | --- |
| `T(B,E_t)` | 避免不同模块各自重算，方便排查 frame bug |
| `action_pose_b` | 更容易做 replay / FK / deployment 比较 |

## 11. 必做验证项

### 11.1 First-frame identity check

每个 episode 起始帧检查：

$$
T(W_s, I_0) \approx I
$$

### 11.2 Chain consistency check

抽样验证：

$$
T(B, E_t) = T(B, W_s)\,T(W_s, I_t)\,T(I, E)
$$

### 11.3 Replay check

给定 logged `action_pose_ws`，重建：

$$
T(B, E_t^\star) = T(B, W_s)\,T(W_s, E_t^\star)
$$

检查 replay 时 FR3 target 是否与预期一致。

### 11.4 FK agreement check

如果某帧 joint state 可用，比较：

- robot FK 给出的 $T(B, E_t)$
- SLAM + extrinsic 链给出的 $T(B, E_t)$

如果差异持续过大，应优先怀疑：

- `T(I,E)` 方向错
- IMU frame 真实定义不一致
- FK 所用 TCP 定义和 `E` 不一致

## 12. 当前开发建议：接下来该做什么

### 优先级 1：冻结实现 contract

在代码库中明确只保留这几个统一入口：

- `transform_utils.py`
- `dataset_schema.md`
- `runtime_adapter.py`

不要让各个脚本临时各写一版 `I -> E`。

### 优先级 2：把 E 的轴语义写进代码注释和文档

必须在代码里明确写出：

```
-z_E: approach（接近方向）
-y_E: closing（闭合方向）
 x_E: normal（法向）
```

否则后面换人开发很容易再乱。

### 优先级 3：做一版"静态 replay 检查脚本"

输入：`T(W_s,I_t)`、`T(I,E)`、`T(B,W_s)`

输出：`T(B,E_t)`，然后和 FK 结果对比。

这会是目前性价比最高的验证工具。

### 优先级 4：确认真实 IMU body axes

这是当前最容易被忽略、但最可能埋雷的点。需要确认真实 SLAM 输出所使用的 IMU frame 是否与 URDF `link_imu` 保持：

- 原点一致
- 轴方向一致
- 无固定偏转

## 13. 当前可冻结的技术决策

| 决策 | 内容 |
| --- | --- |
| 决策 A | `I = link_imu` |
| 决策 B | `E = das_gripper_ee` |
| 决策 C | E 轴语义：`-z` 接近，`-y` 闭合，`x` 法向 |
| 决策 D | 使用 URDF 链构造候选固定外参：$T(I,E) = T(\text{base},I)^{-1} T(\text{base},E)$ |
| 决策 E | SLAM pose 默认是 `I` 语义，部署前必须转到 `E` |
| 决策 F | 送入 IK 的 pose 必须是 `T(B,E)`，不能混用 `T(B,I)` |

## 14. 当前不要做的事

| 禁止 | 原因 |
| --- | --- |
| 不要在不同脚本里各写一份 `T(I,E)` 推导逻辑 | 导致 contract 散落，难以维护 |
| 不要把 `T(W_s,I_t)` 直接当成 `T(W_s,E_t)` 用 | `I ≠ E`，会引入固定偏差 |
| 不要在没定义清楚 E 轴语义前就开始大规模训练 | 训练后修改代价极高 |
| 不要把 MuJoCo 可视化结果当成最终真值 | 仍要保留 replay/FK 检查 |

## 15. 一页版结论

- 当前已能构造候选 `T(I,E)`，其位置几何上已通过 MuJoCo sanity check。
- E 的正式语义已冻结：`-z` 接近，`-y` 闭合，`x` 法向。
- `das_base_frame` 在 MuJoCo 中应作为 URDF `gripper_base_link` 的对应帧。
- SLAM 输出是 `I` 语义，不能直接拿来当 `E` 控制。
- 部署前必须通过 `T(I,E)` 把 pose/action 转到 `E` 语义。
- 所有 IK 输入必须是 `B` 下的 `E` pose。
- 下一步重点不是再猜位置，而是做 runtime replay/FK consistency 和真实 IMU frame 一致性验证。


