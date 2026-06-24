# LeRobot Offline Eval

`lerobot-eval-offline` 用于对 `lerobot_train.py` 或 `tools/fr3/fr3_train_il_policy.py` 训练出的策略 checkpoint 做离线推理。它不连接真实机器人，也不创建仿真环境，只从 LeRobot dataset 的指定 episode 逐步读取 observation，预测 action chunk，并把每个 chunk 画成 matplotlib 3D 轨迹。

## 基本用法

如果当前环境没有 matplotlib，先安装可视化依赖：

```bash
pip install 'lerobot[matplotlib-dep]'
```

```bash
lerobot-eval-offline \
  --dataset-root dataset_test/updated/single_cube2_20260429_165325 \
  --train-config outputs/datasets/single_cube2_updated_act_cam1_cam2_pika_right_imgonly \
  --checkpoint outputs/train/single_cube2_updated_act_cam1_cam2_pika_right_imgonly/checkpoints \
  --episode-index 0
```

如果当前 editable install 的 console script 还没刷新，也可以直接运行 `python src/lerobot/scripts/lerobot_eval_offline.py ...`。

如果 `--step-start` 和 `--step-end` 不提供，脚本会对该 episode 的所有 step 都做推理并画图。

## 指定 step 范围

`--step-end` 是闭区间，也就是会包含这个 step。

```bash
lerobot-eval-offline \
  --dataset-root dataset_test/updated/single_cube2_20260429_165325 \
  --train-config outputs/datasets/single_cube2_updated_act_cam1_cam2_pika_right_imgonly/train_config.generated.json \
  --checkpoint outputs/train/single_cube2_updated_act_cam1_cam2_pika_right_imgonly/checkpoints/057500 \
  --episode-index 3 \
  --step-start 20 \
  --step-end 80
```

## 参数说明

- `--dataset-root`: 要离线 eval 的 LeRobot dataset root，例如 `dataset_test/updated/single_cube2_20260429_165325`。
- `--train-config`: 训练 config 文件或包含 config 的目录。传目录时会自动找 `train_config.generated.json` 或 `train_config.json`。
- `--checkpoint`: 模型 checkpoint。可以传 `pretrained_model`、单个 step 目录、`checkpoints/last`，也可以直接传 `checkpoints`，脚本会优先用 `last`，否则用最大的数字 step。
- `--episode-index`: 要 eval 的 episode index。
- `--step-start`, `--step-end`: episode 内的 step 起止 id，`step-end` 为闭区间。
- `--trajectory-dims`: 画 3D 轨迹使用的 action 三个维度，默认 `auto`。也可以写 `0,1,2` 或 `ee.x,ee.y,ee.z`。
- `--device`: 覆盖 checkpoint config 里的设备，例如 `cuda`、`cuda:0`、`cpu` 或 `auto`。
- `--use-amp`: 使用 autocast 推理。
- `--output-dir`: 输出目录，默认 `outputs/eval_offline`。
- `--show`: 保存图片后打开 matplotlib 窗口。

## 输出

每次运行会在 `outputs/eval_offline/<dataset>_epXXXX_stepsXXXX-XXXX/` 下保存：

- `action_chunks_3d.png`: 每个 observation step 预测出的 action chunk 3D 轨迹图。
- `action_chunks_3d_interactive.html`: 可交互 3D 轨迹文件，浏览器打开后可以拖拽旋转、滚轮缩放、Shift+拖拽平移。
- `action_chunks.npz`: 数值结果，包含 `chunks`、`step_ids`、`trajectory_dims`。
- `metadata.json`: 本次运行使用的数据集、checkpoint、step 范围、action 维度名称等信息。

图中每条模型预测 action chunk 使用不同颜色。chunk 起点用圆点标记，终点用 `X` 标记，旁边的文字是 episode 内 step id。数据集里的 GT action 会作为黑色虚线连续轨迹叠加显示，覆盖当前推理窗口从 `step-start` 到最后一个预测 chunk 尾部 step 的范围。

## 和 `tools/fr3/fr3_train_il_policy.py` 的关系

`tools/fr3/fr3_train_il_policy.py` 会先生成一个解耦的 dataset view，再调用标准 LeRobot training。离线 eval 脚本只依赖这个 view 里保存的通用信息：

- `meta/info.json` 里的 observation/action feature schema；
- `meta/stats.json` 和 checkpoint 中保存的 processor；
- `train_config.generated.json` 里的 resize、video backend、tolerance 等 dataset 读取配置；
- checkpoint 下的 `pretrained_model/config.json`、`model.safetensors`、`policy_preprocessor.json`、`policy_postprocessor.json`。

因此它不会 import FR3 真实机器人 runtime，也不会读取相机或机械臂配置。
