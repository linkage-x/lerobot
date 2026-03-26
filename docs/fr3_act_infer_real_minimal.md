# FR3 ACT 真机推理最小说明

本文是当前仓库内 FR3 ACT 真机推理的权威入口说明。目标是回答 4 个问题：

- 现在应该运行哪个入口
- 默认启动会做什么
- policy 输入/输出按什么语义解释
- 当前还有哪些问题没有闭环

## 当前入口

- 宿主机 launcher: `tools/fr3/fr3_act_infer_real.py`
- 容器内 runtime: `tools/fr3/fr3_act_infer_real_runtime.py`
- compose service: `lerobot-infer-fr3-act`

默认 checkpoint:

- `outputs/train/2026-03-19/10-48-39_act/checkpoints/060000`

默认相机配置:

- `tools/fr3/fr3_act_infer_camera_config.yaml`

## 默认启动行为

直接运行：

```bash
python3 tools/fr3/fr3_act_infer_real.py
```

当前默认会按这个顺序执行：

1. 用 `panda_py` 将机械臂移动到 DAS 起始关节角。
2. 连接 `FrankaResearch3`。
3. 将 gripper 对齐到 dataset start mean。
4. 启动 ACT inference loop。

如需关闭默认启动动作：

```bash
python3 tools/fr3/fr3_act_infer_real.py --no-move-to-das-start
python3 tools/fr3/fr3_act_infer_real.py --no-align-gripper-to-dataset-start
```

## 当前输入/输出合同

### 状态输入

runtime 当前按下列语义构造 `observation.state`：

- 真机 pose 从 `E = das_gripper_ee` 转到 `I = gripper_base_link`
- pose 再映射到 dataset/replay 使用的世界合同
- gripper 观测先从硬件归一化 `[0,1]` 反变换回 dataset aperture 单位
- 最终按 dataset metadata 中的 `state_names` 组装 policy 输入

这一步是当前版本最重要的修复点。此前 preview 和 real-run 不一致，根因不是主坐标链，而是 live gripper observation 与 dataset gripper 单位不一致。

如果 checkpoint 来自 `mask2ee` 训练：

- runtime 不需要额外 CLI 开关
- 推理会跟随 checkpoint 中保存的 policy config 自动继续 mask 掉 `x y z qx qy qz qw`
- `gripper` 仍然保留

详细合同见：

- `docs/fr3_mask2ee_training_inference_contract_20260326.md`

### 图像输入

当前图像语义已确认正确：

- `left -> observation.images.left`
- `right -> observation.images.right`

默认相机键直接与 policy 图像键同名，不再需要额外 `camera-key-map`。

### tactile 输入

当前默认 checkpoint 仍是 tactile ACT。runtime 支持两种路径：

- 真机 DAS tactile 输入
- 仅限 preview 的 `--tactile-fallback=baseline_idle`

已确认 infer 容器内可以拿到真实 SDK tactile callback，但 `448-byte` payload 与 dataset `left_raw/right_raw` 的最终 wire-format 映射还没有彻底闭环。

### action 解码

当前 action 默认按 8 维解释：

- `x y z qx qy qz qw gripper`

runtime 会：

1. 将 quaternion action 转成 rotvec
2. 将 gripper 从 dataset aperture 单位映射回硬件所需的 `0..1`
3. 在发送前应用首帧 gate 和每步小步限幅
4. 调用 `robot.send_action(...)`

joint-space OTG / 高频控制仍由 `FrankaResearch3` 内部负责。

## 推荐命令

查看最终 Docker 命令：

```bash
python3 tools/fr3/fr3_act_infer_real.py --dry-run
```

先做安全预览：

```bash
python3 tools/fr3/fr3_act_infer_real.py --preview --max-steps 5
```

如果只想排除 tactile 缺失的影响：

```bash
python3 tools/fr3/fr3_act_infer_real.py --preview --max-steps 5 --tactile-fallback baseline_idle
```

正式真机执行：

```bash
python3 tools/fr3/fr3_act_infer_real.py
```

更保守的安全门：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --preview \
  --max-steps 5 \
  --first-frame-max-pos-delta-mm 20 \
  --first-frame-max-rot-delta-deg 8 \
  --max-step-pos-delta-mm 3 \
  --max-step-rot-delta-deg 2
```

## 当前已确认事项

已经闭环：

- `left/right` 图像语义正确
- preview 与 real-run 的 gripper 观测语义已统一
- `move_to_das_start` 已深度集成到实际 inference runtime
- launcher 默认行为已与真机启动要求一致

仍未闭环：

- DAS tactile `448-byte` payload 到 dataset `left_raw/right_raw` 的最终硬件语义映射
- 真机长 rollout 下的失败恢复、operator confirm gate、在线记录

## 支撑文档

推荐阅读顺序：

1. 本文：运行入口与当前合同
2. `docs/fr3_real_infer_docs_index_20260324.md`：文档职责索引
3. `docs/fr3_mask2ee_training_inference_contract_20260326.md`：mask2ee 训练/推理合同
4. `docs/fr3_act_infer_runtime_fix_20260324.md`：本轮修复记录
5. `docs/fr3_infer_image_semantics_validation_20260323.md`：图像链路结论
6. `docs/tactile/fr3_das_tactile_packet_investigation_20260323.md`：tactile open issue
7. `docs/fr3_replay_tracking_findings_20260319.md`：replay 侧追踪问题
