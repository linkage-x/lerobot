# FR3 Nintendo Controller Teleop Best Practices

日期：2026-05-09

本文记录当前 Nintendo Joy-Con / Pro Controller 接入 FR3 MuJoCo teleop 的推荐实现。重点结论是：Nintendo 手柄只用 IMU 做相对旋转，平移使用摇杆。不要默认用加速度双积分做末端平移。

## 1. 推荐控制模型

当前最佳实践是 hybrid 控制：

- `ZL`：clutch。按下后建立相对姿态零点，松开后停止旋转控制并重置 IMU clutch 状态。
- IMU gyro：只输出 `target_wx`、`target_wy`、`target_wz`，用于末端相对旋转。
- 左摇杆前后：`target_x` 平移。
- 左摇杆左右：`target_y` 平移，当前默认左右方向已反向，匹配 FR3 场景操作直觉。
- 右摇杆前后：`target_z` 平移。
- `R`：按住时夹爪逐渐闭合，松开后夹爪逐渐打开。
- `experimental_imu_translation`：默认关闭，仅保留实验入口。

默认轴向约定：

```text
left_stick.y  ->  target_x
-left_stick.x ->  target_y
right_stick.y ->  target_z
gyro.x        ->  target_wx
-gyro.y       ->  target_wy
-gyro.z       ->  target_wz
```

默认比例：

```text
translation_scale = 0.001
vertical_scale    = 0.001
rotation_scale    = 1.0
stick_deadband    = 0.12
```

## 2. 为什么不默认使用 IMU 平移

Quest3 controller 可以用 clutch 后的相对位移，是因为它有 6DoF 空间追踪。Nintendo Joy-Con / Pro Controller 只有 IMU，没有外部位置观测。单靠 accelerometer 双积分得到位置会不可避免地产生漂移，尤其是：

- 加速度里包含重力分量。
- 手柄姿态轻微变化会改变重力投影。
- gyro/accel bias 和采样噪声会被积分放大。
- FR3 teleop action 会把非零 `target_*` 持续应用到 reference pose。

因此默认实现和 `joycon-robotics` 的稳定策略一致：位置由摇杆/按键步进控制，IMU 只负责姿态。

## 3. ZL clutch 和旋转漂移处理

FR3 MuJoCo 中 `target_wx/wy/wz` 是每帧的相对旋转增量。环境会计算：

```text
desired_rotation = reference_rotation @ Rotation.from_rotvec(target_w)
```

并在 enabled 时把 `reference_pose` 更新为本帧 `desired_pose`。这意味着 Nintendo teleop 侧只要持续输出很小的非零 `target_w*`，末端姿态就会逐帧漂移。

当前实现使用以下策略抑制漂移：

- `_latest_reading()` 区分 fresh report 和 stale report。
- stale report 不重复输出上一帧旋转增量。
- ZL 首帧只建立 gyro baseline，不输出旋转。
- 静止判定同时作用于旋转和平移分支。
- 静止时清零 `rel_gyro_dps`，并把 gyro baseline 重新锚到当前静止读数。
- `imu_stationary_gyro_dps` 默认 `3.0`，`imu_stationary_accel_norm_tolerance_g` 默认 `0.08`。

这解决了按住 `ZL` 后手柄未动但末端慢慢旋转的问题。根因通常是 ZL 按下瞬间第一帧 gyro baseline 被手指抖动污染，后续每帧 `gyro - baseline` 形成小的持续旋转命令。

## 4. FR3 MuJoCo 使用方式

基础 teleop：

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_mujoco_teleop.py \
  --teleop-type nintendo \
  --nintendo-controller pro
```

录制：

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_mujoco_record.py \
  --teleop.type=nintendo \
  --teleop.controller=pro
```

如果只想用 Joy-Con 右手柄或左手柄：

```bash
uv run --extra fr3_teleop python tools/fr3/fr3_mujoco_teleop.py \
  --teleop-type nintendo \
  --nintendo-controller right
```

## 5. 调参建议

优先调整这些参数：

- `--nintendo-translation-scale`：左摇杆 X/Y 平移速度，默认 `0.001`。
- `--nintendo-vertical-scale`：右摇杆 Z 平移速度，默认 `0.001`。
- `--nintendo-rotation-scale`：IMU 旋转速度，默认 `1.0`。
- `--nintendo-stick-deadband`：摇杆死区，默认 `0.12`。
- `--nintendo-imu-gyro-deadband-dps`：gyro 相对 baseline 死区，默认 `1.5`。
- `--nintendo-imu-stationary-gyro-dps`：静止 gyro 判定阈值，默认 `3.0`。
- `--nintendo-stale-timeout-s`：允许沿用上一帧按钮/摇杆状态的时间，默认 `0.25`。

如果静止时还有很小旋转漂移，优先增大：

```bash
--nintendo-imu-gyro-deadband-dps 2.5
--nintendo-imu-stationary-gyro-dps 4.0
```

如果旋转响应太迟钝，再适度降低这两个阈值。

## 6. 实验 IMU 平移

保留实验开关：

```bash
--nintendo-experimental-imu-translation
```

打开后会把 accelerometer 积分得到的平移增量叠加到摇杆平移上。这个模式只适合排查和实验，不建议用于数据采集或真实机器人控制。没有外部 tracking 的情况下，它无法达到 Quest3 controller 的平移稳定性。

## 7. 环境和连接注意事项

Nintendo teleop 依赖 `hidapi`。Ubuntu 24.04 上如果系统已加载 Nintendo kernel driver，需要使用 hidraw backend 的 hidapi。确认方式：

```bash
python - <<'PY'
import hid
print(hid.enumerate(0x057e, 0x2009))
PY
```

如果 `import hid` 失败，在当前 repo 环境中安装：

```bash
uv pip install --python .venv/bin/python "hidapi>=0.14.0,<0.15.0"
```

如果 `open_path` 失败且系统使用 `/dev/hidraw*`，优先重装 hidraw backend：

```bash
HIDAPI_SYSTEM_HIDAPI=1 HIDAPI_WITH_HIDRAW=1 \
uv pip install --python .venv/bin/python --reinstall --no-cache --no-binary hidapi \
  "hidapi>=0.14.0,<0.15.0"
```

配对流程：

- Joy-Con：长按 SYNC 到 LED 滚动，在 Bluetooth 设置里配对，然后按 L/R 绑定键。
- Pro Controller：长按 SYNC 到 LED 滚动，在 Bluetooth 设置里配对或 USB 连接。
- 如果 `joycond` 提示 pairing，按对应 trigger 完成绑定。

## 8. 回归测试

当前关键回归测试在 `tests/teleoperators/test_nintendo.py`：

- factory 和 FR3 runtime argparse 能创建 Nintendo teleop config。
- 左摇杆/右摇杆平移映射正确。
- `ZL` clutch 建立相对旋转零点。
- stale IMU report 不重复输出旋转增量。
- ZL 首帧 noisy gyro baseline 不会导致静止旋转漂移。
- experimental IMU translation 默认关闭且可显式开启。
- `R` 按住闭合夹爪，松开打开夹爪。

推荐提交前运行：

```bash
env PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 uv run pytest tests/teleoperators/test_nintendo.py
python -m py_compile \
  src/lerobot/teleoperators/nintendo/configuration_nintendo.py \
  src/lerobot/teleoperators/nintendo/teleop_nintendo.py \
  tools/fr3/fr3_mujoco_runtime.py \
  tests/teleoperators/test_nintendo.py
```
