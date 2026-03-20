# FR3 ACT 真机推理最小实现说明

本文记录当前仓库内已经落地的 FR3 ACT 真机推理最小实现，包括：

- 启动方式
- 默认参数
- 运行链路
- 文件职责
- 当前约束与已知边界

## 1. 当前落地内容

本次实现新增了 3 个核心入口：

- compose service:
  - `lerobot-infer-fr3-act`
- 宿主机 launcher:
  - `tools/fr3/fr3_act_infer_real.py`
- 容器内 runtime:
  - `tools/fr3/fr3_act_infer_real_runtime.py`

对应改动文件：

- `docker/docker-compose.yml`
- `tools/fr3/fr3_act_infer_real.py`
- `tools/fr3/fr3_act_infer_real_runtime.py`
- `tests/scripts/test_fr3_act_infer_real.py`

## 2. 默认设计决策

当前最小实现冻结如下：

- 专用 compose service:
  - `lerobot-infer-fr3-act`
- 默认执行器:
  - `fr3 + das`
- 默认 checkpoint:
  - `outputs/train/2026-03-19/10-48-39_act/checkpoints/060000`
- 默认相机配置:
  - `tools/fr3/fr3_record_config.yaml`
- action 平滑空间:
  - `joint space`
- 低频 policy 更新频率默认值:
  - 来自训练数据集 metadata 的 `fps`

注意这里采用的是双频模型：

- 低频:
  - policy inference loop，默认按 dataset `fps`
- 高频:
  - FR3 驱动内部 OTG/Ruckig joint-space 平滑与发送线程

也就是说，runtime 本身不会再手写第二套 Ruckig 循环，而是复用 `FrankaResearch3` 已有的 OTG 通路。

## 3. 当前执行链路

当前真机执行链路如下：

1. 读取 checkpoint 下的 `pretrained_model/train_config.json`
2. 解析训练时 dataset root / repo_id / policy 配置
3. 加载 dataset metadata，用于：
   - 读取 `fps`
   - 读取 `observation.state` / `action` 特征名
   - 读取 normalization stats
4. 加载 `tools/fr3/fr3_record_config.yaml` 中的相机配置
5. 创建 `FrankaResearch3` 真机对象，默认：
   - `gripper_backend=das`
   - `target_frame_name=das_gripper_ee`
   - `urdf=src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_das_ati.urdf`
6. 采集当前 observation
7. 将 observation 适配成 policy 输入
8. 执行 ACT 推理
9. 将 policy 输出的 EE 动作解码为机器人命令
10. 调用 `robot.send_action(...)`
11. `FrankaResearch3` 内部执行：
   - EE absolute action -> IK -> joint target
   - joint target -> OTG/Ruckig -> 高频 joint command 下发

## 4. 观测与相机适配

### 4.1 状态输入

当前 runtime 直接从 FR3 当前观测中构造 `observation.state`。

默认按 dataset metadata 中的 state names 取值。若 metadata 缺失，则回退到：

- `x`
- `y`
- `z`
- `qx`
- `qy`
- `qz`
- `qw`
- `gripper`

内部映射为：

- `x -> ee.x`
- `y -> ee.y`
- `z -> ee.z`
- `qx -> ee.qx`
- `qy -> ee.qy`
- `qz -> ee.qz`
- `qw -> ee.qw`
- `gripper -> gripper.pos`

这里会先通过 `KeepAbsoluteEEObservation` 把真机 rotvec 观测转换成 quaternion 形式，再拼成 policy state。

### 4.2 相机输入

`fr3_record_config.yaml` 当前相机键是：

- `ee`
- `side`
- `front`

而当前默认 checkpoint 期待的图像键是：

- `observation.images.left`
- `observation.images.right`

所以 runtime 里加了一层 camera key map。当前默认值是：

- `ee:left`
- `side:right`

也就是：

- `ee -> observation.images.left`
- `side -> observation.images.right`

`front` 当前不会默认送进 policy。

如需覆盖，可传：

```bash
--camera-key-map ee:left,front:right
```

### 4.3 tactile 输入

默认 checkpoint `060000` 是 tactile ACT。

当前最小实现没有接入真实 tactile 设备，因此对所有缺失的 `STATE` 输入特征，runtime 会自动补零。

这意味着下面这些键若 checkpoint 需要但机器人观测未提供，会补零：

- `observation.tactile.left_clean`
- `observation.tactile.right_clean`
- 以及其他缺失的 state features

## 5. action 解码

当前默认 dataset/action 语义按 8 维处理：

- `x`
- `y`
- `z`
- `qx`
- `qy`
- `qz`
- `qw`
- `gripper`

runtime 会：

1. 将 quaternion 转成 rotvec
2. 组装为：

```python
{
    "ee.x": ...,
    "ee.y": ...,
    "ee.z": ...,
    "ee.wx": ...,
    "ee.wy": ...,
    "ee.wz": ...,
    "gripper.pos": ...,
}
```

3. 调用 `robot.send_action(...)`

夹爪值当前按数据集里的 aperture 语义处理，再映射成机器人需要的 `0..1` normalized 值：

- DAS backend:
  - 使用 `das_min_distance_m` / `das_max_distance_m`
- Pika backend:
  - 使用 `gripper_max_width_mm`

## 6. 启动命令

### 6.1 推荐方式：通过宿主机 launcher

先查看最终 docker 命令：

```bash
python3 tools/fr3/fr3_act_infer_real.py --dry-run
```

直接启动：

```bash
python3 tools/fr3/fr3_act_infer_real.py
```

指定 checkpoint / dataset root / robot ip：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --checkpoint outputs/train/2026-03-19/10-48-39_act/checkpoints/060000 \
  --dataset-root outputs/datasets/lerobotv3_0310_100ep \
  --robot-ip 192.168.1.208
```

限制步数做短时验证：

```bash
python3 tools/fr3/fr3_act_infer_real.py --max-steps 100
```

### 6.2 直接使用 docker compose

当前 launcher 等价于：

```bash
docker compose --profile infer -f docker/docker-compose.yml run --rm lerobot-infer-fr3-act \
  bash -lc 'cd /lerobot && PYTHONPATH=/lerobot/src /lerobot/.venv/bin/python tools/fr3/fr3_act_infer_real_runtime.py \
  --checkpoint=/lerobot/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000 \
  --camera-config=/lerobot/tools/fr3/fr3_record_config.yaml \
  --gripper-backend=das \
  --camera-key-map=ee:left,side:right'
```

### 6.3 按你要求的 sudo 风格

如果宿主机需要你显式传 `HOME` 和 `sudo`，可使用：

```bash
sudo env HOME=/home/hph docker compose --profile infer -f docker/docker-compose.yml run --rm lerobot-infer-fr3-act \
  bash -lc 'cd /lerobot && PYTHONPATH=/lerobot/src /lerobot/.venv/bin/python tools/fr3/fr3_act_infer_real_runtime.py \
  --checkpoint=/lerobot/outputs/train/2026-03-19/10-48-39_act/checkpoints/060000 \
  --camera-config=/lerobot/tools/fr3/fr3_record_config.yaml \
  --gripper-backend=das \
  --camera-key-map=ee:left,side:right'
```

## 7. 常用可覆盖参数

宿主机 launcher 支持：

- `--checkpoint`
- `--camera-config`
- `--dataset-root`
- `--policy-fps`
- `--max-steps`
- `--robot-ip`
- `--gripper-port`
- `--gripper-backend`
- `--camera-key-map`
- `--dry-run`

例如：

```bash
python3 tools/fr3/fr3_act_infer_real.py \
  --policy-fps 15 \
  --camera-key-map ee:left,front:right \
  --max-steps 200
```

## 8. compose service 说明

`lerobot-infer-fr3-act` 当前基于硬件可见的 `x-lerobot-common` 运行时配置，具备：

- `privileged: true`
- `network_mode: host`
- `/dev` 挂载
- `/dev/bus/usb` 挂载
- 显卡支持

因此它面向真机 / 真相机运行，而不是轻量训练容器。

## 9. 当前已知边界

当前实现是最小可运行版本，边界如下：

1. 未接入真实 tactile 设备
   - tactile 特征当前补零

2. camera map 目前依赖手工配置
   - 默认 `ee:left,side:right`
   - 不会自动推断最优相机分配

3. 当前 action 下发走 `robot.send_action(...)`
   - joint-space OTG 是在 `FrankaResearch3` 内部完成
   - runtime 自身不直接调用 `send_joint_positions(...)`

4. 当前未增加真机安全保护层
   - 例如：
     - 首帧对齐确认
     - workspace 二次审计
     - 速度/姿态额外限幅
     - operator confirm gate

5. 当前未落地 rollout 记录、在线日志与失败恢复

## 10. 当前验证情况

已完成：

- 新增文件语法编译通过
- launcher 命令构造检查通过
- 最小单测文件已补齐

未完成：

- 宿主机本地 `pytest` 执行
- 容器内真实依赖环境下的端到端推理验证
- 真机联调验证

原因是当前宿主机 Python 环境缺少项目依赖与 `pytest`，因此本次验证停留在代码结构与语法层。

## 11. 后续建议

如果继续推进，下一步建议按这个顺序做：

1. 容器内跑通一次 `--max-steps 5`
2. 加入启动前安全确认和首帧 hold
3. 明确当前 checkpoint 的 tactile 缺失是否可接受
4. 视需要把 `front` 相机也纳入 camera mapping 策略
5. 增加真机 smoke test / short rollout 文档
