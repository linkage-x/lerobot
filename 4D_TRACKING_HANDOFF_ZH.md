# `ntl/4d_track` 分支：4D 物体跟踪交接指南

本文给接手本分支的同事使用。这里的“4D 跟踪”指在连续视频帧中恢复物体的三维位置和轨迹，即 **3D 空间 + 时间**。当前主流程针对 Thor 采集的 `cam_13` / `cam_03` 双目视频：SAM3 提供逐帧物体掩码，Fast-FoundationStereo 提供米制深度，Track4World 恢复随时间变化的物体表面点轨迹，并在两个相机间融合相同标签的物体均值轨迹。

## 1. 分支和入口

- 主仓库分支：`ntl/4d_track`。
- 一键入口：[run_thor_sam3_recovery.py](third_party/Track4World/scripts/run_thor_sam3_recovery.py)。日常使用只需要指定 `--dataset-root`。
- 深度后端实验入口：[run_thor_depth_experiment.py](tools/track4world/run_thor_depth_experiment.py)。`--depth-backend ffs` 用原双目模型求深度，`--depth-backend lingbot-depth` 默认以 FFS 深度为输入，对 `cam_03` 和 `cam_13` 分别细化深度；两个选项的物体轨迹都分别输出，不做双相机轨迹融合。安装、命令及实验限制见 [LINGBOT_DEPTH_EXPERIMENT.md](tools/track4world/LINGBOT_DEPTH_EXPERIMENT.md)。
- 更完整的英文操作说明：[THOR_SAM3_RECOVERY_TUTORIAL.md](third_party/Track4World/THOR_SAM3_RECOVERY_TUTORIAL.md)；掩码校正和结果格式见 [SAM3_INTERACTIVE_RECOVERY.md](third_party/Track4World/SAM3_INTERACTIVE_RECOVERY.md)。
- 代码依赖位于 `third_party/Track4World`、`third_party/sam3` 和 `third_party/Fast-FoundationStereo` Git 子模块。切换分支后执行 `git submodule update --init --recursive`，并确认 Track4World 子模块包含 `scripts/run_thor_sam3_recovery.py`。接手新机器时，先确保主仓库提交和对应子模块提交都已推送到远端。

当前简化入口会按以下顺序执行：

```text
Thor 同步双目 MKV → 8 fps 视频 → 双目校正和深度 → SAM3 交互选物体
                  → 全视频掩码 → 每相机 2D/3D 点轨迹 → 双相机物体均值融合
```

## 2. 运行前准备

从仓库根目录执行命令。需要 NVIDIA GPU、可显示 OpenCV 窗口的桌面会话、`ffmpeg`，以及两个 Python 环境：

| 用途 | 当前机器上的路径或环境 |
| --- | --- |
| SAM3 交互与视频掩码 | `conda activate sam3-py312` |
| Fast-FoundationStereo、Track4World、绘图 | `/home/dante/miniconda3/envs/sam3/bin/python` |
| SAM3 权重 | `third_party/Track4World/checkpoints/sam3.pt` |
| 双目模型权重 | `third_party/Fast-FoundationStereo/weights/23-36-37/model_best_bp2_serialize.pth` |

**Git 不包含本地数据、标定、模型权重、运行结果和 Conda 环境。** `outputs/`、Track4World 的 `checkpoints/` 与 `results/`、Fast-FoundationStereo 的 `weights/` 均被忽略。新机器需要单独复制以下内容，保持目录结构，或者修改入口脚本中的固定路径：

- 要处理的 `outputs/datasets/...` 数据集；每个 episode 应有 `meta.json`、`cam_13.mkv` 和 `cam_03.mkv`。
- `outputs/calibration/manual_run_20261002T014903Z/camera_calibration/summary.json`。
- 上表中的两个模型权重，以及两个运行环境。环境重建可参考 [SAM3_INTERACTIVE_RECOVERY.md](third_party/Track4World/SAM3_INTERACTIVE_RECOVERY.md#runtime-setup)。

本入口固定 `cam_13` 为左目（序列号 `H120K-I05130029`）、`cam_03` 为右目（`H120K-I05130020`），并使用上述 2026-10-02 外参。两个相机的**出厂内参**从所选 episode 的 `meta.json` 自动提取。外参文件引用的 2026-09-29 重新标定内参目前不在此工作区，不能把出厂内参与那套内参视为等价。入口会检查序列号、双目同步和校正质量；如果更换相机或机械安装，应重新标定并调整入口脚本，不能沿用旧外参。

## 3. 先跑一个 episode

下面是 2026-10-05 数据集的完整命令。**默认仅处理 episode 1**，对应 `episodes/episode_000001/`；编号从 0 开始。

```bash
cd /home/dante/Tianle/box_project/lerobot
conda activate sam3-py312
python third_party/Track4World/scripts/run_thor_sam3_recovery.py \
  --dataset-root '/home/dante/Tianle/box_project/lerobot/outputs/datasets/1003——1005/thor_gmsl2_10ch_v1_20261005_164229'
```

第一次运行会生成 8 fps 双目输入、校正视频和深度，再打开两个相机的 SAM3 选物体窗口。已完成的几何结果在输入视频和标定没有变化时会复用。需要用其他 Track4World 环境时，在命令末尾添加 `--track-python /path/to/python`。

建议先做不启动模型的输入检查：

```bash
python third_party/Track4World/scripts/run_thor_sam3_recovery.py \
  --dataset-root '/home/dante/Tianle/box_project/lerobot/outputs/datasets/1003——1005/thor_gmsl2_10ch_v1_20261005_164229' \
  --check-only
```

只准备双目视频与几何、不打开 SAM3 时，把 `--check-only` 换成 `--prepare-only`。准备完成后，仍需运行上面的正常命令以生成掩码和轨迹。

## 4. 选择 episode 和交互标注

在同一终端设置路径变量，后续命令可以更短：

```bash
THOR_DATASET_ROOT='/home/dante/Tianle/box_project/lerobot/outputs/datasets/1003——1005/thor_gmsl2_10ch_v1_20261005_164229'
```

处理指定 episode（例如 0、2、5）：

```bash
python third_party/Track4World/scripts/run_thor_sam3_recovery.py \
  --dataset-root "$THOR_DATASET_ROOT" --episodes 0 2 5
```

处理此数据集全部 33 个 episode（0–32）：

```bash
python third_party/Track4World/scripts/run_thor_sam3_recovery.py \
  --dataset-root "$THOR_DATASET_ROOT" --all-episodes
```

`--episodes` 与 `--all-episodes` 不能同时使用。全量运行会依次处理所有 episode，建议先完成 episode 1 并检查质量。

SAM3 窗口中，左键添加物体正样本点，右键添加背景负样本点，Enter 保存当前掩码，然后在终端输入物体名称。`u` 撤销上一个已保存物体，`r` 重命名，`q` 完成当前相机。**同一物体在两个相机中使用完全相同且唯一的名称**，否则不会形成双目融合轨迹。

如果运行中断，重跑相同命令即可复用已保存的选择；`--reuse-selections` 直接复用，`--select-again` 重新选择，两者互斥。传播后应检查两路 `segmentation.mp4` 和 `mask_diagnostics.json`，尤其是遮挡和物体出入画的位置。

## 5. 看结果并判断是否可用

默认 episode 1 的结果目录为：

```text
third_party/Track4World/results/thor_gmsl2_10ch_v1_20261005_164229/
└── episode_000001/stereo_cam13_cam03_robot_base/
    ├── stereo_calibration_report.json
    ├── cam_13_geometry/ 与 cam_03_geometry/
    └── sam3_selected/
        ├── cam_13/ 与 cam_03/
        │   ├── selection/selection.json
        │   ├── masks/segmentation.mp4
        │   ├── tracks_measured/lerobot_object_tracks.npz
        │   └── tracks_kalman/lerobot_object_tracks.npz
        ├── united_observation/
        └── united_observation_kalman/
```

先打开结果根目录的 `sam3_selected_index.html`，再看 `sam3_selected/united_observation/scene_index.html` 的双目 2D/3D 场景视频。`tracks_measured` 是以观测为主的轨迹；`tracks_kalman` 包含受限 Kalman 预测，分析真实观测时优先使用前者。`united_observation/manifest.json` 记录双目融合和相机间不一致情况，`united_object_mean_trajectories.npz` 保存融合后的物体均值轨迹。

交付或用于下游分析前，至少核对：

1. `stereo_calibration_report.json` 中的校正残差和视差方向。构建器会拒绝 95% 竖向残差大于 2 像素或左右目顺序错误的结果。
2. 两路掩码视频中物体是否持续跟随目标；`mask_diagnostics.json` 的面积突变是复查线索，并不自动代表错误。
3. 每相机 `tracks_measured/quality_report.json`、轨迹视频及 `lerobot_object_tracks.npz` 的缺测比例；遮挡期间的空缺比错误的伪轨迹更可信。
4. 双目 `united_observation/manifest.json` 中两相机是否经常出现位置分歧。只有同名物体才会融合；相机间物体**均值**会融合，但点 ID 仍属于各自相机，并不表示两路观测到同一个表面点。

需要读取逐帧 3D 轨迹时，每路 `lerobot_object_tracks.npz` 的 `xyz_m` 是米制 XYZ，`uv_px` 是图像坐标；主要维度为 `[帧, 物体, 点槽位, 坐标]`。`point_ids` 才是稳定点标识，数组槽位不是点 ID。`frame_index` 对应原始数据集行，`video_frame_index` 对应处理后 8 fps 视频帧；不要直接把两者混用。仅取实测 3D 样本的建议掩码是：

```python
import numpy as np

tracks = np.load("lerobot_object_tracks.npz")
measured_valid = (
    tracks["valid_3d"]
    & tracks["observed"]
    & tracks["dataset_frame_valid"][:, None, None]
)
xyz_m = tracks["xyz_m"]
```

这些文件是 LeRobot 可关联的 **sidecar**，不会把原始 LeRobot 数据集改写成含 4D 轨迹的数据集。接入训练或评估时需显式处理 60 fps 原视频与 8 fps 轨迹的时间映射，以及无有效数据集行的样本。

## 6. 掩码漂移时修正

先从 `selection/selection.json` 确认物体 ID（从 1 开始），再从掩码视频确认要修正的 **8 fps 视频帧号**（从 0 开始）。简化入口负责准备与完整运行；指定后续帧掩码修正时使用底层入口。以下示例中的 `237` 和 `2` 需要换成实际帧号、物体 ID：

```bash
python third_party/Track4World/scripts/run_interactive_sam3_recovery.py \
  --source-format thor \
  --dataset-root "$THOR_DATASET_ROOT" \
  --run-root third_party/Track4World/results/thor_gmsl2_10ch_v1_20261005_164229 \
  --episodes 1 --cameras cam_13 --reuse-selections --segmentation-only \
  --correction 1:cam_13:237:2
```

修正后重新检查该相机的掩码视频，再用第 3 节的正常命令加 `--reuse-selections` 重新生成双目轨迹与场景视频。具体的遮挡处理方式见 [SAM3_INTERACTIVE_RECOVERY.md](third_party/Track4World/SAM3_INTERACTIVE_RECOVERY.md#review-and-correct-sam3-masks-after-occlusion)。

## 7. 已知边界

- 2026-10-05 数据集在本机做过全部 33 个 episode 的**输入与标定预检查**；没有在本交接工作中完成 33 个 episode 的 GPU 全流程运行。首次使用先跑单 episode 并查看结果。
- 当前外参与数据集中的出厂内参组合是可用输入方案，但米制精度仍需通过场景几何、标定残差和双目一致性评估。若拿到外参求解时使用的 2026-09-29 标定内参，应重新生成几何并比较质量。
- 自动掩码、点关联和跨相机标签匹配都是估计；遮挡、同类重复物体及视角差异可能导致身份漂移。预测轨迹与实测轨迹必须分开使用。
- 本分支的简化入口目前面向 **Thor 60 fps、cam_13/cam_03、固定外参**。其他相机组合、分辨率或采集配置需要检查脚本假设并重新验证，不能直接照搬结果。
