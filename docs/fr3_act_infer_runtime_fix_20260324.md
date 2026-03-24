# FR3 ACT Runtime Fixes 2026-03-24

## 本轮进度

- 修复了 ACT 真机推理启动阶段的 gripper 观测语义不一致问题。
- 将 `move_to_das_start` 深度集成到 `fr3_act_infer_real_runtime.py`，不再依赖跨模块 `tools.*` 导入。
- 将真机默认启动行为改为：
  1. 先移动到 DAS 起始关节角。
  2. 再将 gripper 对齐到 dataset start mean。
  3. 再进入 ACT inference。
- 保留两个显式关闭开关：
  - `--no-move-to-das-start`
  - `--no-align-gripper-to-dataset-start`

## 现象与根因

### 现象

- `--preview --max-steps=1` 时，step0 的 EE delta 很小，但 log 中 `preview_policy_obs_gripper` 显示 live gripper 被大幅虚拟修正。
- 真机执行加起始对齐后，step0 首帧仍可能出现大幅 raw target 跳变，并被 `hold_first_frame` 拦截。

### 根因

- live `robot.get_observation()['gripper.pos']` 是硬件归一化 `[0,1]`。
- dataset / ACT policy 输入里的第 8 维按 aperture 单位处理。
- 推理 runtime 之前没有把 live gripper observation 转回 dataset 单位，导致 preview 路径和 real-run 路径处在不同语义空间。

## 修复摘要

### 1. gripper 观测语义统一

在 `tools/fr3/fr3_act_infer_real_runtime.py` 中新增：
- `denormalize_live_gripper_observation()`
- `convert_gripper_observation_to_dataset_units()`

用途：
- 启动阶段的 dataset-start 对齐诊断使用 dataset 单位比较。
- 送入 policy 前，live gripper observation 先转换到 dataset 单位。
- preview offset 与 real-run physical alignment 最终对齐到同一语义空间。

### 2. move_to_das_start 深度集成

- 将 DAS 起始关节角常量直接内联到 `fr3_act_infer_real_runtime.py`。
- 在 runtime 启动 `FrankaResearch3.connect()` 前直接调用 `panda_py.Panda(...).move_to_joint_position(...)`。
- 避免了 `PYTHONPATH=/lerobot/src` 下 `from tools...` 导入失败的问题。

### 3. 默认启动行为调整

`python3 tools/fr3/fr3_act_infer_real.py` 现在默认等价于：
- move to DAS start: enabled
- align gripper to dataset start: enabled

如需关闭：
- `python3 tools/fr3/fr3_act_infer_real.py --no-move-to-das-start`
- `python3 tools/fr3/fr3_act_infer_real.py --no-align-gripper-to-dataset-start`

## 经验记录

- 真机 preview 和 real-run 不一致时，优先检查输入合同，而不是先怀疑坐标链。
- 对于 runtime 真正执行的路径，避免依赖仓库根相对导入；容器里 `PYTHONPATH` 往往只暴露 `src/`。
- 启动安全动作如果是高频必需步骤，应默认开启并提供 opt-out，而不是反过来。
- 真机 wrapper 的测试应避免依赖 `tmp_path workspace` 下不存在的默认 checkpoint；必要时直接传 `/lerobot/...` 路径。

## 验证

在 Docker infer 容器内执行：

```bash
docker compose --profile infer -f docker/docker-compose.yml run --rm lerobot-infer-fr3-act \
  bash -lc 'cd /lerobot && PYTHONPATH=/lerobot/src /lerobot/.venv/bin/pytest -q tests/scripts/test_fr3_act_infer_real.py'
```

结果：

- `26 passed in 1.41s`
