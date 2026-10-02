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

### 4.0 Capture session、续采和合并规则

相机或 FR3 base 移动后必须开始新的布局世代；不同布局的数据绝对不能联合求解。
同一布局内，每次 manual/automatic capture 是一个独立 session，可以显式联合。

目录名
`fr3_execute_pose_thor_gmsl2_apriltag_p0_tag6_replay_20260921_merged`
容易误解：`20260921` 表示机器人姿态来源；其相机视频和实测机器人 pose 实际在
2026-10-01 17:02 重新采集，因此它是 10 月 1 日当前布局的 round 1，不是旧布局图像。

需要带多相机画面、颜色提示以及每相机 `valid/target`、`remaining` 的手动续采时，
在 Thor 桌面终端运行：

```bash
cd /home/nvidia/lerobot

bash tools/thor/run_p0_two_marker_calibration_local.sh guided \
  --dataset-root outputs/datasets/fr3_execute_pose_thor_gmsl2_apriltag_p0_tag6_replay_20260921_merged \
  --quality-summary outputs/calibration/p0_tag6_replay_20261001_extrinsics/summary.json \
  --target-per-camera 30 \
  --exclude-camera-ids 1,4,10 \
  --detection-workers 1 \
  --frame-bus-every-n 30 \
  --detection-scale 0.25 \
  --execute \
  --confirmation P0_TWO_MARKER_TEACHING
```

程序先把 round 1 的图像、实测 `T_base_ee` 和关节值导入一个新的
`manual_run_*` session。`--quality-summary` 用上一轮求解的严格筛选数量初始化 UI；
如果省略，则使用 AprilTag 实时检测数量。随后 `Enter` 追加一次同步
图像和机器人状态；`q` 仅在每台相机达到 target 后求解；`f` 强制尝试；`Esc`
保留已提交 capture 但不求解。求解使用已有 fisheye 内参、鲁棒离群点过滤和当前
FR3 base；质量门通过后保存候选并自动生成 XY/XZ/YZ/XYZ 四视图。默认不会修改
active；审核后显式增加 `--activate` 才更新 active calibration。

审核候选 summary 和四视图后，可完全离线激活同一 session：

```bash
bash tools/thor/run_p0_two_marker_calibration_local.sh solve \
  --solve-run latest \
  --activate
```

若进程中断，继续同一个 manual session，不要重新导入 round 1：

```bash
bash tools/thor/run_p0_two_marker_calibration_local.sh guided \
  --resume-run latest \
  --target-per-camera 30 \
  --exclude-camera-ids 1,4,10 \
  --detection-workers 1 \
  --frame-bus-every-n 30 \
  --detection-scale 0.25 \
  --execute \
  --confirmation P0_TWO_MARKER_TEACHING
```

如果第二轮也是自动回放采集，可在纯离线模式中显式决定是否合并。只用第二轮：

```bash
bash tools/thor/run_p0_two_marker_calibration_local.sh calibrate \
  --dataset-root outputs/datasets/ROUND2_MERGED
```

合并同一布局的 round 1 + round 2：

```bash
bash tools/thor/run_p0_two_marker_calibration_local.sh calibrate \
  --dataset-root outputs/datasets/fr3_execute_pose_thor_gmsl2_apriltag_p0_tag6_replay_20260921_merged \
  --dataset-root outputs/datasets/ROUND2_MERGED
```

重复 `--dataset-root` 就是显式选择合并；程序不会自动搜索或混入其他历史数据。
导入来源会写入 `imported_datasets.json`，重复传入同一路径会跳过，原始数据集保持
只读不变。

### 4.1 在 Thor 本机记录示教位姿并自动采集

当前分支已将旧 standalone 入口拆分为两个维护中的阶段。两步都在 Thor 桌面终端
执行；`teaching` 不打开相机，只记录机器人位姿，`capture` 再自动执行这些位姿并由
Thor GMSL2 recorder 采集视频。给每轮标定使用一个新的、相同的 `--key`：

```bash
cd /home/nvidia/lerobot

# 1. 手动拖动 FR3；按 r 记录当前位姿，按 q 保存并退出。
bash tools/thor/run_p0_two_marker_calibration_local.sh teaching \
  --key p0_tag6_20260930 \
  --execute \
  --confirmation P0_TWO_MARKER_TEACHING

# 2. 自动执行刚才的位姿，同时让 Thor 多相机录制。
bash tools/thor/run_p0_two_marker_calibration_local.sh capture \
  --key p0_tag6_20260930 \
  --max-records all \
  --execute \
  --confirmation P0_TWO_MARKER_AUTOMATIC_CAPTURE
```

第二轮如果复用第一轮 teaching pose，但需要写入新的 capture session，使用新的
`--key`，并把原 teaching JSON 作为 `--input-json`，避免覆盖第一轮：

```bash
bash tools/thor/run_p0_two_marker_calibration_local.sh capture \
  --key p0_tag6_current_layout_round2 \
  --input-json outputs/datasets/p0_tag6_current_layout_round1/teaching_pose_records.json \
  --max-records all \
  --exclude-camera-ids 1,4,10 \
  --execute \
  --confirmation P0_TWO_MARKER_AUTOMATIC_CAPTURE
```

第一步保存到
`outputs/datasets/p0_tag6_20260930/teaching_pose_records.json`。第二步默认保存合并数据集到
`outputs/datasets/fr3_execute_pose_thor_gmsl2_apriltag_p0_tag6_20260930_merged/`。
可先在任一命令末尾添加 `--dry-run`，只检查当前路径和最终命令，不连接硬件。
第二步会真实移动机器人；执行前必须清空整个工作区、确认急停可用并全程看护。

旧 standalone 标定保存在 `outputs/calibration/.../manual_run_*/captures.json`，而不是
`outputs/datasets/<key>/teaching_pose_records.json`。其每条记录已有
`joint_values_rad`，可直接作为自动执行输入。例如重放 2026-09-21 的 87 个采集姿态
并重新录制相机：

```bash
bash tools/thor/run_p0_two_marker_calibration_local.sh capture \
  --key p0_tag6_replay_20260921 \
  --input-json outputs/calibration/p0_single_tag_camera_calibration/manual_run_20260921T071714Z/captures.json \
  --max-records all \
  --execute \
  --confirmation P0_TWO_MARKER_AUTOMATIC_CAPTURE
```

这里的 `--key` 是新输出的名称，`--input-json` 才是实际读取的旧示教轨迹。只有在
机器人、标定板安装和工作区仍允许这些关节位姿安全执行时才能重放。

`capture` 每次启动时读取 MAX96726 当前 locked IDs，默认排除 `cam_01`、`cam_04`
和 `cam_10`，再把其余 locked cameras 写进临时 recorder config。它还会清除
`DISPLAY/WAYLAND_DISPLAY`，强制 Argus 使用 headless EGL；否则 Thor 桌面或 X11
环境可能导致所有相机依次报 `DRI3` / `NvBufSurfaceMapEglImage failed`。如需临时修改
排除集合，可使用 `--exclude-camera-ids 1,4,10`。

### 4.2 一键执行采集和完整解算

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
