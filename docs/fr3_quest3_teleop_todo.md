# FR3 Quest3 Pika Teleop TODO

## 背景结论

目标是把当前 FR3 MuJoCo/record 路径里的 SpaceMouse 遥操替换或扩展为 Quest3 遥操：在 VR 中使用手部关键点驱动 Pika gripper 操作物体，并支持两种使用方式。

- 穿戴设备模式：操作者佩戴 Quest3，手部关键点/手柄姿态相对头显或初始校准位姿驱动 FR3/Pika。
- 挂脖模式：Quest3 不作为沉浸显示设备，而作为胸前/颈部视觉追踪设备；需要在 MuJoCo viewer 中显示 Pika gripper 的目标 pose，便于操作者看屏幕完成操作。

当前 LeRobot FR3 MuJoCo 遥操入口主要假设 teleop 输出 SpaceMouse 风格 action：

```python
{
    "enabled": bool,
    "target_x": float,
    "target_y": float,
    "target_z": float,
    "target_wx": float,
    "target_wy": float,
    "target_wz": float,
    "gripper": float,  # 0 closed, 1 open
}
```

关键代码位置：

- FR3 MuJoCo record 当前直接实例化 `SpaceMouseTeleop`：`tools/fr3/fr3_mujoco_record.py`
- FR3 MuJoCo runtime 参数和 SpaceMouse config 构造：`tools/fr3/fr3_mujoco_runtime.py`
- MuJoCo teleop action 消费逻辑：`src/lerobot/envs/fr3_mujoco.py`
- MuJoCo viewer marker 逻辑：`src/lerobot/envs/fr3_mujoco_teleop.py`
- SpaceMouse teleop action contract 可参考：`src/lerobot/teleoperators/spacemouse/teleop_spacemouse.py`

HIROL Quest3 代码可作为输入侧参考，但不建议直接整体搬入 LeRobot：

- Quest3 代码：`/home/hanyu/Codes/HIROLRobotPlatform/teleop/XR/quest3/meta_quest3.py`
- 运行说明：`/home/hanyu/Codes/HIROLRobotPlatform/teleop/XR/quest3/README_QUEST.md`
- 配置样例：`/home/hanyu/Codes/HIROLRobotPlatform/teleop/XR/quest3/config/*.yaml`

原因：HIROL 代码依赖 `teleop.base.*`、`hardware.base.utils`、`glog` 等 HIROL 内部模块；LeRobot 环境没有 `vuer` 依赖；而且该实现里有需要修正的细节，例如 hand tracking 分支里 `tool_right = left_hand_pose_7d` 疑似应为 `right_hand_pose_7d`，`extract_hand_states()` 的 `getattr(self, f"{prefix}_pinch_state_shared")` 与实际成员 `_left_pinch_state_shared` 命名不一致。

## 推荐落地路线

先做一个 LeRobot-native 的 Quest3 teleoperator，输出现有 SpaceMouse 风格 action，以最小改动接入现有 FR3 MuJoCo/record。等稳定后，再引入更高级的绝对 pose 控制和挂脖 viewer ghost gripper。

## Phase 0: 依赖和联通性

- [x] 在 LeRobot 环境中增加 Quest3 运行依赖：新增 optional dependency `lerobot[quest3]`，Quest3 实现使用标准 `logging`，不依赖 `glog`。
- [x] 增加 `lerobot[fr3_teleop]` 组合 extra，用 uv 统一安装 FR3 MuJoCo、Quest3、SpaceMouse HID、Placo IK 和 Ruckig OTG 依赖。
- [ ] 在 Docker 镜像或启动脚本中加入 Quest3 端口说明：默认使用 `8012` 或独立配置端口，避免和现有 `18765` camera stream 冲突。
- [x] 写 `tools/fr3/fr3_quest3_connection_smoke.py`：只启动 Vuer server，打印 right hand wrist pose、pinch/squeeze 值和 gripper 映射值。
- [x] 固化 Quest3 设备准备步骤：ADB reverse、证书路径、无线连接、浏览器 URL，见 `docs/fr3_quest3_hardware_setup.md`。
- [ ] 明确网络模式：USB reverse 为默认调试路径；Wi-Fi 模式作为备选，并记录 latency/jitter。

验收：

- Quest3 浏览器页面能连接 LeRobot 进程。
- `right_hand_positions`、`right_hand_orientations`、`pinchValue/squeezeValue` 持续更新。
- 断开 Quest3 后 teleop 不崩溃，输出 safe zero action。

## Phase 1: LeRobot Quest3 Teleoperator

- [x] 新增 `src/lerobot/teleoperators/quest3/configuration_quest3.py`，注册 `TeleoperatorConfig.register_subclass("quest3")`。
- [x] 新增 `src/lerobot/teleoperators/quest3/teleop_quest3.py`，实现 `Teleoperator` 接口：`connect()`、`disconnect()`、`get_action()`、`action_features`、`sync_gripper_baseline()`。
- [x] 不依赖 HIROL `TeleoperationDeviceBase`，只复用其 Vuer handler 思路：`HAND_MOVE`、`CONTROLLER_MOVE`。
- [x] 使用线程安全对象缓存最新 Quest3 数据；保留 `safe_mat_update()` 防止奇异矩阵污染状态。
- [ ] 配置项至少包括：`mode`、`hand`、`control_frame`、`use_hand_tracking`、`cert_file`、`key_file`、`host`、`port`、`frequency`、`deadband`、`translation_scale`、`rotation_scale`、`gripper_mapping`、`lost_tracking_timeout_s`。当前缺口：`control_frame` 尚未显式建模。
- [x] 在 `src/lerobot/teleoperators/utils.py` 或 device-class fallback 路径中确保 `quest3` 能被 `make_teleoperator_from_config()` 创建。

验收：

- `teleop.type=quest3` 可通过 draccus 配置创建。
- `get_action()` 返回字段与 SpaceMouse 完全一致。
- 未校准、丢 tracking、Quest3 未连接时，`enabled=False` 且 gripper 保持最近安全值。

## Phase 2: 手部关键点到 Pika Gripper Command

- [ ] 定义手部关键点 contract：保留 Quest3/Vuer 的 25 landmarks；记录腕点、拇指指尖、食指指尖、掌心估计点的 index。
- [ ] 实现 gripper 开合映射的第一版：优先使用 Quest3 `pinchValue`，备选使用 thumb-tip 到 index-tip 距离。
- [ ] 标定 open/close 距离：启动时采集 `open_distance_m` 和 `closed_distance_m`，归一化到 Pika `gripper in [0, 1]`。
- [ ] 加入滤波：EMA、最大变化率、deadband、tracking lost hold，参考 SpaceMouse 已有 `gripper_cmd_ema_alpha` / `gripper_cmd_max_rate`。
- [ ] 支持二值模式和连续模式：二值模式适合早期稳定抓取；连续模式适合精细操作。
- [ ] 增加 `tools/fr3/fr3_quest3_gripper_smoke.py`：不动机械臂，只打印/可选控制 Pika gripper，验证 pinch 到 gripper command 的单调性和延迟。

验收：

- 张手时 Pika 打开，捏合时 Pika 闭合。
- 手部 tracking 短暂丢失不导致突然闭合或突然打开。
- 连续控制下 command 抖动低于配置阈值。

## Phase 3: 手部关键点到 EE Pose

- [ ] 定义手 pose：用 wrist pose 作为基础；用手掌法向和食指方向构造 gripper orientation；必要时只先控制平移，锁定 orientation。
- [ ] 实现校准/使能手势：例如 pinch hold 或控制器 A 键作为 clutch；未 clutch 时 `enabled=False`。
- [ ] 实现相对控制第一版：记录 clutch 起始的 hand pose 和当前 robot TCP pose，输出相对 `target_x/y/z/wx/wy/wz`，接入现有 `FR3MujocoEnv.step_teleop_action()`。
- [ ] 实现 workspace clip：沿用 FR3 MuJoCo `workspace_min/max`，并在 teleop 侧加速度/速度限制，避免 IK 目标跳变。
- [ ] 实现坐标系配置：从 HIROL 的 `T_ROBOT_OPENXR` / `T_OPENXR_ROBOT` 迁移为可配置矩阵，并为 FR3 base frame 写默认值。
- [ ] 明确 `pika_gripper_ee` vs `pika_task_tcp` 的控制帧选择，和当前 FK/IK 使用的 `target_frame_name` 对齐。

验收：

- 在 MuJoCo 中手移动方向和 gripper 目标移动方向一致。
- 停止 clutch 后机器人保持当前关节，不继续漂移。
- 姿态控制可以单独开关；平移-only 模式可稳定完成 pick/place 的接近动作。

## Phase 4: 穿戴设备模式

- [ ] 配置 `mode="wearable"`：使用 head pose 做相对参考，类似 HIROL `absolute` 模式中减去 head position 的做法。
- [ ] 在 VR 中显示机器人相机画面：复用 HIROL 的 `ImageBackground`/binocular 机制，输入图像可来自 MuJoCo camera grid 或真实相机。
- [ ] 为穿戴模式提供 recenter 操作：重设 head/world 原点、hand pose 原点和 robot TCP 对齐。
- [ ] 增加可视化调试 overlay：当前 tracking 状态、pinch/gripper 值、clutch 状态、latency。

验收：

- 戴上 Quest3 后可以看到机器人视角画面。
- recenter 后手部运动和 gripper 目标方向一致。
- tracking 丢失或页面断连时机器人停止运动。

## Phase 5: 挂脖模式

- [ ] 配置 `mode="neck"`：把 Quest3 当作固定/准固定视觉追踪设备，不使用 head-relative 交互假设。
- [ ] 增加 neck mount 标定：求 `T_robot_quest`，支持手动输入、棋盘/AprilTag、或通过已知 gripper pose 对齐。
- [ ] 手 pose 从 Quest3 camera/world frame 变换到 robot base frame：`T_robot_hand = T_robot_quest @ T_quest_hand`。
- [ ] 增加挂脖模式 dead zone 和低通滤波，因为胸前视角下手部关键点更容易受遮挡。
- [ ] MuJoCo viewer 显示 Pika gripper 目标 pose：扩展 `marker_geoms_from_info()` 或新增 `quest3_target_pose` marker，显示目标 gripper 的 xyz 轴、指尖 opening、tracking 状态颜色。
- [ ] 如需要更直观，后续再做 ghost Pika mesh；第一版用 axis + 两个 fingertip spheres 即可落地。

验收：

- 不佩戴 Quest3，仅挂脖/固定时，屏幕 viewer 能看到手驱动的 Pika 目标 pose。
- 目标 pose marker 与实际 MuJoCo TCP marker 可同时显示，颜色区分 target/tcp/quest3。
- 手移出视野时 marker 变灰，机器人停止更新目标。

## Phase 6: 接入现有 FR3 MuJoCo Teleop/Record

- [x] 在 `tools/fr3/fr3_mujoco_runtime.py` 增加 `--teleop-type {spacemouse,quest3}` 和 `--quest3-*` 参数。
- [x] 抽象 runtime teleop config 构造：SpaceMouse 仍走原逻辑；Quest3 创建 `Quest3TeleopConfig`。
- [x] 修改 `tools/fr3/fr3_mujoco_teleop.py` 和 `tools/fr3/fr3_mujoco_record.py`，不要硬编码 `SpaceMouseTeleop`。
- [ ] 录制路径中保存 Quest3 诊断元数据：tracking valid、pinch value、hand pose latency。不要替代 canonical `timestamp`。
- [ ] 增加 `tools/fr3/fr3_mujoco_quest3_teleop.py` 或命令示例，先只支持 sim。
- [ ] 真实 FR3 接入前，要求通过 MuJoCo smoke 和 replay validation。

验收：

- 同一套 MuJoCo 环境可用 `--teleop-device spacemouse` 或 `--teleop-device quest3` 切换。
- Quest3 录制的数据集仍符合当前 ee2ee dataset contract。
- Quest3 断连不会中断保存流程，最多停止当前 episode。

## Phase 7: 安全和测试

- [ ] 单元测试：keypoint 到 gripper command 的归一化、deadband、滤波、tracking lost 行为。
- [ ] 单元测试：OpenXR 到 FR3 base 坐标变换方向；用固定矩阵验证 xyz 方向。
- [ ] MuJoCo smoke：Quest3 mock 数据驱动 1000 steps，检查没有 NaN、IK 不爆、workspace clip 生效。
- [ ] 延迟测试：记录 Vuer event timestamp、teleop loop timestamp、env step timestamp。
- [ ] 安全策略：断连、无手、奇异矩阵、pinch 突变、超 workspace、viewer 关闭，都输出 `enabled=False`。
- [ ] 真实 Pika gripper 前置 smoke：先不动 FR3 机械臂，只控制 gripper。

## 预估工作量

- MVP：Quest3 hand tracking 驱动 MuJoCo Pika gripper 开合，不控制 arm pose：1-2 天。
- 可操作 MuJoCo：相对 hand pose 控制 FR3 EE + Pika gripper + viewer marker：3-5 天。
- 穿戴模式画面链路和 recenter 稳定：2-4 天。
- 挂脖模式可用：取决于 `T_robot_quest` 标定方式，约 3-6 天。
- 真实 FR3 安全接入：建议另算 2-4 天，必须先完成 sim 验收。

## 风险点

- Quest3/Vuer hand tracking 的坐标系和 handedness 需要实测，不能只相信矩阵常量。
- 手部关键点 occlusion 会直接影响 gripper command，必须有 tracking validity 和 hold/stop 策略。
- 当前 FR3 MuJoCo action contract 是相对 delta，不是绝对 pose；直接把 Quest3 absolute pose 塞进去会导致方向和尺度错误。
- `vuer` 和 HTTPS/证书/ADB reverse 是工程联通性风险，应该先做连接 smoke。
- 挂脖模式没有稳定标定就不可控，必须把 `T_robot_quest` 标定列为一等任务。
