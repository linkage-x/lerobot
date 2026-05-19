# FR3 Quest3 Pika MuJoCo Collision Findings 2026-04-28

本文记录 Quest3 Pika-only MuJoCo scene 调 gripper collision / actuator 的当前经验，供后续排查抓取、碰撞体可视化和控制语义时参考。

## 结论

Quest3 Pika-only scene 应尽量复用 full FR3 Pika 的原始 mesh collision 和 actuator 参数，不要通过外部加厚 box、底部 lip、托举面等额外碰撞体来提高抓取成功率。

当前推荐基线：

- gripper finger collision 使用 `pika_gripper_left_link` / `pika_gripper_right_link` mesh。
- finger collision 参数对齐 full FR3：`friction="2.0 0.4 0.03"`、`condim="4"`、`solref="0.001 0.6"`、`solimp="0.995 0.9995 0.00005"`、`priority="2"`。
- red block 参数对齐 full FR3 scene：`density="500"`、`friction="1.8 0.05 0.01"`、`condim="4"`。
- gripper actuator XML 参数对齐 full FR3：`ctrlrange="-0.11 0"`、`kp="50"`、`dampratio="1"`。
- Quest3 Pika-only 默认初始 TCP 高度要给 gripper 足够空间，目前使用 `(0.48, 0.0, 0.55)`。
- Quest3 Pika-only 控制默认锁住 roll / pitch，只保留 yaw。

## 不要再做

不要在 finger 外面加 `gripper_left_pad_collision` / `gripper_right_pad_collision` 或底部 lip 来托起红块。这样虽然可能短期提高视觉上的抓取成功率，但会让碰撞体明显肥厚，和 gripper mesh 外形不一致，也会掩盖真实问题。

不要用 kinematic grasp assist 去改红块 freejoint，让红块跟随 gripper 位姿。此前出现的“隔空抬起”就是这类逻辑造成的：红块不是被接触和摩擦夹起来，而是被代码直接更新 freejoint 状态。

## SpaceMouse Full FR3 路径是怎么做的

下面这个命令没有传 `--teleop-type`，所以 argparse 默认走 `spacemouse`：

```bash
docker compose -f docker/docker-compose.yml --profile sim --profile teleop run --rm \
  -e DISPLAY=$DISPLAY \
  -e PYTHONPATH=/workspace/src \
  lerobot-fr3-sim-teleop \
  python tools/fr3/fr3_mujoco_teleop.py \
  --enable-cameras --camera-width 640 --camera-height 480
```

该路径使用的是 full FR3 scene：

- `build_runtime_teleop_config()` 返回 `SpaceMouseTeleopConfig`。
- `build_runtime_env()` 返回 `FR3MujocoEnv`。
- `should_use_quest3_pika_env()` 只在 `teleop.type == "quest3"` 且 `quest3_scene_mode == "pika_gripper"` 时才会切到 `Quest3PikaMujocoEnv`。
- full FR3 XML 是 `fr3_pika_ati_scene.xml` include `fr3_pika_ati.xml`。
- gripper 通过 full arm IK 和 MuJoCo 真实 contact / friction 抓取红块，没有额外外部托举碰撞体。

因此，Quest3 Pika-only 的正确对齐方向是把 Pika-only scene 的物理参数向 full FR3 原始 mesh 靠近，而不是引入新的 box 支撑结构。

## Quest3 Pika-only 与 Full FR3 的关键差异

Quest3 Pika-only scene 没有 FR3 arm body，`gripper_base` 是 mocap body，Quest3 wrist 直接驱动 Pika gripper。这个结构方便 Quest3 直接操作，但也意味着：

- mocap gripper 可以瞬间进入桌面、红块或 fixture 附近，接触约束比 full arm IK 更容易卡住。
- gripper base 相关非 finger collision 不适合贸然打开，否则直接 mocap 驱动时容易把桌面或物体顶飞。
- finger mesh collision 可以恢复，但初始高度和 workspace clipping 必须留出足够空间。

当前保留 gripper base 相关 collision `contype="0"` / `conaffinity="0"`，只让 finger mesh collision 参与夹取。这是 Pika-only direct mocap scene 相对 full FR3 的有意差异。

## Gripper 命令语义

高层 gripper command 继续使用：

- `1.0` 表示打开。
- `0.0` 表示闭合。

Quest3 controller 当前语义：

- 右 trigger 按下：输出 `0.0`，逐渐闭合。
- 右 trigger 松开：输出 `1.0`，逐渐打开。
- 左 trigger 也输出打开命令 `1.0`，可作为显式打开输入。

在 Quest3 Pika-only env 中，实际 actuator ctrl 方向由 `_gripper_ctrl_from_command()` 处理。不要仅凭 XML `ctrlrange="-0.11 0"` 推断高层 command 方向，必须用 qpos 实测确认。

实测基线：

- reset 默认打开，左右 finger qpos 约 `-0.048 / 0.048`。
- close command 后，左右 finger qpos 收到接近 `0 / 0`。
- open command 后，左右 finger qpos 回到约 `-0.047 / 0.047`。

## 初始高度问题

恢复 finger mesh collision 后，`initial_tcp_position=(0.48, 0.0, 0.37)` 太低。表现是：

- reset 或低位操作时 gripper mesh 已经接近桌面、红块或 fixture。
- contact constraint 会限制 finger joint，导致松开 trigger 后 gripper 打不开。
- 这会被误判为 actuator 参数、trigger mapping 或 collision mesh 太厚的问题。

把默认初始 TCP 高度提高到 `z=0.55` 后，同样的 mesh collision 和 actuator 参数可以正常开合。

排查这类问题时，先做无接触高度测试：

```bash
uv run --extra fr3_teleop python - <<'PY'
from lerobot.envs.quest3_pika_mujoco import Quest3PikaMujocoEnv, Quest3PikaMujocoEnvConfig

for z in (0.37, 0.55, 0.75):
    env = Quest3PikaMujocoEnv(
        Quest3PikaMujocoEnvConfig(
            continuous_physics=False,
            enable_cameras=False,
            initial_tcp_position=(0.48, 0.0, z),
        )
    )
    try:
        env.reset(include_camera_obs_in_observation=False, include_camera_obs_in_info=False)
        for _ in range(800):
            env.step_teleop_action({"gripper": 0.0, "tracking_valid": True}, 1 / 240,
                                   include_camera_obs_in_observation=False,
                                   include_camera_obs_in_info=False)
        closed = env._get_gripper_joint_positions()
        for _ in range(800):
            env.step_teleop_action({"gripper": 1.0, "tracking_valid": True}, 1 / 240,
                                   include_camera_obs_in_observation=False,
                                   include_camera_obs_in_info=False)
        opened = env._get_gripper_joint_positions()
        print("z", z, "closed", closed, "opened", opened)
    finally:
        env.close()
PY
```

如果高位能正常开合、低位打不开，优先查 contact / 初始位姿，而不是加大碰撞体。

## 碰撞体可视化建议

定位 collision 时建议先在 MuJoCo viewer 中打开 collision geom 可视化，确认以下几点：

- finger mesh collision 是否和黑色 finger visual 基本重合。
- finger collision 是否相对 gripper hand 有异常夹角。
- gripper 初始高度是否让 finger mesh 穿入桌面、红块或 fixture。
- red block 是否被 kinematic 逻辑或非 finger collision “带走”。

如果 viewer 中看到绿色或半透明 debug box 比 finger mesh 肥很多，优先检查是否又引入了外部 pad/lip，而不是继续调 friction。

## 验证命令

基础编译：

```bash
uv run --extra fr3_teleop python -m py_compile \
  src/lerobot/envs/quest3_pika_mujoco.py \
  src/lerobot/teleoperators/quest3/teleop_quest3.py \
  tools/fr3/fr3_mujoco_runtime.py
```

确认没有托举和外部 pad/lip 残留：

```bash
rg -n "grasp_assist|pad_collision|lip_collision|gripper_pad_debug|_gripper_.*pad" \
  src/lerobot/envs/quest3_pika_mujoco.py \
  src/lerobot/robots/franka_research3/assets/franka_fr3/quest3_pika_gripper_scene.xml
```

确认 roll / pitch 锁住、yaw 生效：

```bash
uv run --extra fr3_teleop python - <<'PY'
from lerobot.envs.quest3_pika_mujoco import Quest3PikaMujocoEnv, Quest3PikaMujocoEnvConfig
import numpy as np

env = Quest3PikaMujocoEnv(Quest3PikaMujocoEnvConfig(continuous_physics=False, enable_cameras=False))
try:
    env.reset(include_camera_obs_in_observation=False, include_camera_obs_in_info=False)
    baseline = env.data.mocap_quat[env._mocap_id].copy()
    env._mocap_baseline_pos = env.data.mocap_pos[env._mocap_id].copy()
    env._mocap_baseline_quat_wxyz = baseline.copy()
    env._apply_incremental_mocap(np.zeros(3), np.array([0.5, -0.4, 0.0]))
    no_yaw = env.data.mocap_quat[env._mocap_id].copy()
    env._apply_incremental_mocap(np.zeros(3), np.array([0.5, -0.4, 0.3]))
    with_yaw = env.data.mocap_quat[env._mocap_id].copy()
    print("roll_pitch_delta_changes_quat", bool(np.linalg.norm(no_yaw - baseline) > 1e-8))
    print("yaw_delta_changes_quat", bool(np.linalg.norm(with_yaw - baseline) > 1e-8))
finally:
    env.close()
PY
```

期望输出：

```text
roll_pitch_delta_changes_quat False
yaw_delta_changes_quat True
```

