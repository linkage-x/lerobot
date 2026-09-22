# Thor 外参标定与 P0 机器人回放操作指南

本文适用于当前提交，说明如何通过 UI 完成固定相机
外参标定、生成机器人基座坐标系下的轨迹，并选择数据集、episode 和左右臂执行
回放。

回放直接使用固定相机到 `fr3_base` 的外参，不启动相机，也不使用辅助 AprilTag
重新定位机器人。

## 1. 当前生产标定

生产追踪配置位于：

```text
third_party/opencv_kalibr/hikon_cube_tracking_offline/config_thor/april_cube_tracking_in_robot_base_thor.yaml
```

当前配置指向同一个标定快照：

```yaml
intrinsics_run_name: p0_single_tag_camera_calibration_bundle_20260922/intrinsics
fixed_camera_run_name: p0_single_tag_camera_calibration_bundle_20260922/extrinsics
```

Thor 上的快照目录是：

```text
/home/nvidia/lerobot/outputs/calibration/p0_single_tag_camera_calibration_bundle_20260922/
```

可用以下命令检查生产指针和结果清单：

```bash
cd /home/nvidia/lerobot

rg -n "intrinsics_run_name|fixed_camera_run_name" \
  third_party/opencv_kalibr/hikon_cube_tracking_offline/config_thor/april_cube_tracking_in_robot_base_thor.yaml

python3 -m json.tool \
  outputs/calibration/p0_single_tag_camera_calibration_bundle_20260922/manifest.json

python3 -m json.tool \
  outputs/calibration/p0_single_tag_camera_calibration_bundle_20260922/extrinsics/summary.json
```

该 bundle 是“最新内参 + 已通过的 P0 单标签外参”的不可变快照，不是使用全部
最新内参重新完成的一次联合解算。`cam_03` 和 `cam_09` 的最新内参与原外参解算时
使用的内参不同，所以 `manifest.json` 中
`fully_consistent_with_extrinsics_solve` 为 `false`。要求严格一致时，应使用最新
内参重新解算外参，再将新结果提升为生产标定。

## 2. 启动 UI

在开发机执行：

```bash
cd /home/corenetic/Code/lerobot
bash run/deploy.sh thor
```

脚本会同步当前代码到 Thor、重启网关并启动本地前端。打开脚本输出的 localhost
地址，然后进入“多相机标定”页面。

如果 Thor 重启后 GMSL2/Argus 未恢复，先在 Thor 上执行：

```bash
cd /home/nvidia/lerobot
bash tools/thor/gmsl2/recover_argus.sh \
  --sdk /home/nvidia/Desktop/SG16A_AGTH_G3Y_A1
```

## 3. 通过 UI 采集和解算外参

UI 引导使用 ChArUco A 板：`charuco_400`、12 × 9 方格、方格边长 30 mm。
采集时不要同时展示 B 板，因为两块板的 ID 范围重叠。

1. 在“多相机标定”页面连接相机，点击“开始引导标定”。
2. 如果沿用生产内参，可以跳过各相机的单独内参录制步骤。
3. 不要跳过“外参 · 协同”。缺少该段时无法解算相机间位姿。
4. 在公共工作区缓慢移动标定板。每段应让多台相机同时看到标定板，并最终形成
   连通的相机观测图。
5. 默认每段录制 30 秒，到时自动保存。
6. 开始解算前设置选项：
   - 只重算外参时，关闭“同时重算内参”。
   - 视频未变化且角点缓存正常时，关闭“强制重新检测角点”。
   - 需要生产结果时，关闭“只解算，不导出（实验）”。
7. 检查解算状态、相机连通性、重投影残差，以及相对当前生产标定的相机基线和
   姿态变化。不要只依据单个残差值判断结果。
8. 确认后点击“提升为生产标定”。仅完成解算不会更新生产指针。

提升后再次执行第 1 节的 `rg` 和 `json.tool` 命令，确认配置确实指向新结果。

## 4. 自动单 AprilTag 标定流程

需要重新采集机械臂携带的单 AprilTag，并执行完整内外参流程时，在开发机执行：

```bash
cd /home/corenetic/Code/lerobot

CAPTURE_BACKEND=thor_gmsl2 \
CALIBRATION_MARKER=apriltag \
CAPTURE_MAX_RECORDS=30 \
bash third_party/opencv_kalibr/run_automatic_calibration_on_thor.sh
```

此命令会移动真实机械臂，并依次执行姿态采集、GMSL2 视频录制、内参解算、固定
相机到机器人基座的外参解算和误差报告。运行前必须清空工作区、确认急停可用并由
操作员全程看护。

默认标定物为 `tag36h11`、ID 6、边长 160 mm。主要输出包括：

```text
outputs/datasets/fr3_execute_pose_thor_gmsl2_apriltag_all_merged/
outputs/calibration/thor_gmsl2_fixed_camera_in_base_from_moving_tag36h11_id6_160mm/
outputs/calibration/automatic_calibration_thor_gmsl2_tag36h11_id6_160mm_error_report.txt
```

Thor GMSL2 流程不需要辅助 marker。解算完成后仍要审核并提升结果，或明确更新生产
追踪配置；不要仅凭目录时间戳假定结果已投入生产。

## 5. 从数据集生成回放轨迹

1. 部署包含目标生产标定指针的最新代码。
2. 在 UI 的“数据集处理”页面选择数据集和 episode。
3. 点击“生成 EE 轨迹”。
4. 确认处理结果使用固定机器人基座外参，并生成目标侧的 sidecar：

```text
<dataset>/derived/april_cube_tracking_in_robot_base/state_action.left.csv
<dataset>/derived/april_cube_tracking_in_robot_base/state_action.right.csv
```

可在 Thor 上检查默认数据集：

```bash
DATASET=/home/nvidia/lerobot/outputs/datasets/thor_gmsl2_9ch_v1_20260921_163918

ls -l "${DATASET}/derived/april_cube_tracking_in_robot_base"/state_action.*.csv
python3 -m json.tool "${DATASET}/meta/processing.json"
```

只有包含有限值和足够样本的目标侧才能回放。当前默认数据集只有左侧轨迹有效；
右侧数据全部为非有限值时不能使用 `--side right`。

## 6. 回放机器人和夹爪

默认数据集为：

```text
/home/nvidia/lerobot/outputs/datasets/thor_gmsl2_9ch_v1_20260921_163918
```

在 Thor 上执行 episode 2 的左臂轨迹：

```bash
sudo bash /home/nvidia/box_api/replay_p0_native_arm_only_20260908/run_native_arm.sh \
  --episode-index 2 \
  --side left \
  --gripper-width-mm 88 \
  --replay-speed-factor 0.01 \
  --start-speed-factor 0.03 \
  --gripper-command-rate-hz 15 \
  --execute
```

运行前程序会要求输入 `YES`。参数说明：

- `--episode-index`：要回放的 episode 编号。
- `--side left|right`：选择左臂或右臂 sidecar。
- `--dataset-root PATH`：覆盖默认数据集目录。
- `--replay-speed-factor`：轨迹阶段速度因子，范围 `(0, 1]`。
- `--start-speed-factor`：移动到第一个末端姿态的速度因子，范围 `(0, 1]`。
- `--gripper-command-rate-hz`：夹爪命令更新频率。
- `--gripper-width-mm`：旧接口仍要求该参数；dataset 模式实际使用数据集轨迹。
- `--gripper-mode off`：完全禁用夹爪运动。

指定其他数据集和 episode：

```bash
sudo bash /home/nvidia/box_api/replay_p0_native_arm_only_20260908/run_native_arm.sh \
  --dataset-root /home/nvidia/lerobot/outputs/datasets/MY_DATASET \
  --episode-index 0 \
  --side left \
  --gripper-width-mm 88 \
  --replay-speed-factor 0.01 \
  --start-speed-factor 0.03 \
  --gripper-command-rate-hz 15 \
  --execute
```

回放顺序是：

1. 读取数据集轨迹并完成 IK；
2. 机器人移动到数据集的第一个末端姿态；
3. 夹爪读取并保持当前真实开口，不以“闭合”状态初始化；
4. 启动机械臂回放控制器；
5. 从同一个轨迹起点同步发送末端轨迹和数据集夹爪命令。

默认 IK 优先尝试以下实机起始关节姿态：

```text
current_start_pose_20260828_2048
[-0.2982022355, -0.2054683757, 0.2008775164, -2.7071624978,
 -0.0935047555, 2.9366955832, 0.8043834377]
```

如果该分支违反 legacy workspace 约束，求解器会尝试其他合法种子。日志中的最终
种子才是该 episode 实际使用的 IK 分支。

### 安全说明

数据集适配器当前使用 `--unchecked-execution`，会跳过项目侧的导数、状态、TCP、
网络和碰撞门控。底层机器人安全机制不能替代人工检查。执行时必须：

- 清空机械臂和夹爪工作范围；
- 确认急停和刹车按钮可用；
- 使用较低速度因子首次验证每个 episode；
- 由操作员全程观察，出现异常立即停止。

回放不会打开任何相机，也不会检测辅助 marker。

## 7. 绘制轨迹与夹爪同步曲线

在 Thor 上生成所有 episode 的时间—位置—夹爪诊断图和 CSV：

```bash
cd /home/nvidia/lerobot

/home/nvidia/Code/infer/.venv-fr3/bin/python3 \
  tools/thor/plot_p0_replay_sync.py \
  --dataset-root /home/nvidia/lerobot/outputs/datasets/thor_gmsl2_9ch_v1_20260921_163918 \
  --side all
```

默认输出目录：

```text
<dataset>/derived/p0_replay_sync_diagnostics/
```

每个有效 episode 会生成：

- `episode_NNN_<side>_time_xyz_gripper.png`：时间与 `x/y/z`、夹爪宽度曲线；
- `episode_NNN_<side>_time_xyz_gripper.csv`：相同数据的数值版本；
- 汇总 JSON：记录数据源、有效范围和生成结果。

数据集当前保存的是实测夹爪状态，不是独立的原始期望动作。因此曲线可检查回放
输入的时间对齐和开合时刻，但不能恢复未记录的原始夹爪控制指令。

## 8. 日志与常见检查

每次硬件回放的日志位于：

```text
/home/nvidia/box_api/replay_p0_native_arm_only_20260908/logs/<run-id>/
```

重点文件包括：

- `dataset_input.json`：数据集、episode、侧别、sidecar 哈希和回放参数；
- `gripper_initialization.json`：夹爪连接时读取的真实宽度和初始化结果；
- native planner/controller 日志：实际 IK 种子、规划与执行状态。

如果夹爪仍然提前动作，先比较第 7 节生成的 CSV 与日志时间戳，并确认日志中夹爪
的第一个数据集命令发生在机械臂回放控制器启动之后。
