# Traj-Gen（EE 轨迹生成）对 Thor gmsl2 数据的兼容性结论

> 记录时间：2026-06-01 · 关联 merge commit `caf4315e`（把 box0529 的 traj-gen 实现合入 `hph/thor_release_v0`）

## 一句话结论

合入的 traj-gen 是 **hikon 传感器/标定专属** 的实现，**对 Thor gmsl2 录制的数据（如 `outputs/datasets/thor_gmsl2_11ch_v1_20260601_071432`）目前不适用**。在 Thor 数据上跑不会得到"空轨迹"，而是会在标定/布局检查处 **直接 `raise`，子进程 `exit_code != 0`**，gateway 把任务标成 `failed`。

"空轨迹符合预期"只在 **gateway 回放/读取侧** 成立；**生成侧（hikon 脚本本体）会报错**。

## 两侧分别的行为

### ✅ Gateway 回放/读取侧 —— 对无 cube 数据健壮

`tools/data_collection_gui/gateway.py` 里消费 traj-gen 产物的链路，对 Thor 数据（无 cube 信息）会优雅降级为空轨迹，不崩：

- `_read_sidecar_cube_poses`（gateway.py:2278）：无 `derived/hikon_cube_tracking_in_robot_base/` 目录 → 返回 `{}`
- `_read_video_cube_overlays`（gateway.py:2521）：无 `outputs/tracking_analysis/<name>_tracking_in_robot_base/summary.json` → 直接 `return {}`
- `_cube_pose_from_parquet_row`（gateway.py:2203）：parquet 无 `observation.state.<cube>` 列 → `None`
- 最终 `cubePoseNames`（gateway.py:2948）过滤成"只保留实际出现在帧里的名字" → `[]`，每帧 `cubePoses={}` / `videoOverlays={}`

### ❌ 生成侧 —— hikon 脚本对 Thor 数据有 3 处硬性前置依赖会先 `raise`

脚本：`third_party/opencv_kalibr/hikon_cube_tracking_offline/hikon_cube_tracking_in_robot_base.py`
配置：`.../config_hikon/hikon_cube_tracking_in_robot_base_umi.yaml`
（gateway 默认值见 `gateway.py:34-37` 的 `DEFAULT_EE_TRAJECTORY_SCRIPT` / `DEFAULT_EE_TRAJECTORY_CONFIG`）

在做任何 cube 检测之前，以下检查会按顺序触发：

1. **hikon 专属标定产物缺失**（脚本 `main()` 行 1644-1649）— 要求这三个 summary 存在，否则 `FileNotFoundError`：
   ```
   outputs/calibration/hikon_intrinsics_latest/summary.json
   outputs/calibration/hikon_fixed_camera_in_base_from_moving_charuco/summary.json
   outputs/calibration/hikon_auxiliary_aruco_markers_in_base_single_hk07/summary.json
   ```
   这些是 hikon 相机的内参 / 固定相机在 base 下位姿 / 辅助 aruco 标记基座变换。Thor 设备上没有这套 hikon 标定。

2. **视频目录布局不匹配**（`list_dataset_video_streams` 行 1059-1072）— 脚本要求
   `dataset_root/videos/observation.images.*/chunk-*/file-*.mp4`；
   而 Thor gmsl2 数据是 `episodes/episode_*/*.mkv` 布局（参见 `gateway.py:584` 的 `_has_gmsl2_episodes`：gmsl2 按 `episodes/.../meta.json` + per-episode mkv 组织）。→ `FileNotFoundError` / `No observation.images.* videos found`。

3. **parquet 布局不匹配**（脚本行 722-724 / 1227-1229，以及 `load_ee_ground_truth_from_lerobot_dataset` 行 1826）— 脚本要求
   `dataset_root/data/chunk-*/file-*.parquet`；gmsl2 数据同样走 `episodes/` 布局，没有这棵 `data/` 树。

## 给完善 traj-gen 的同事的方向

这不是改一行能解决的，核心是 hikon 脚本对 **gmsl2 布局 + hikon 标定** 有强假设。两个可选方向：

1. **真正支持 Thor 数据生成轨迹**：
   - 准备 gmsl2 相机的内参/外参标定 summary，把 config 的 `calibration.{root_dir, *_run_name}` 指过去；
   - 解决布局差异：在脚本里增加 `episodes/*.mkv` + episodes 布局的读取分支，或在 traj-gen 前做一步 gmsl2 → lerobot-v3（`videos/observation.images.*` + `data/chunk-*`）的格式适配。

2. **只让 GUI "生成 EE 轨迹" 在 Thor 数据上不崩、给空结果**：
   - 在脚本入口对"无适用相机/标定"做 **软失败**（打印 warning + 写空 sidecar + `exit 0`），而不是 `raise`；
   - 这样 gateway 会把任务标成 `complete`（空轨迹），与回放侧的健壮行为对齐。

## 相关测试

合并时把 HEAD 过时的 `test_traj_gen_is_explicitly_not_implemented`（断言抛 `NotImplementedError`）删掉，改用 box0529 的 `test_traj_gen_starts_hikon_tracking_with_selected_dataset_root`（`tests/scripts/test_data_collection_gui_gateway.py`）。该测试 mock 了 `subprocess.Popen`，只验证命令拼装，**不覆盖脚本对 Thor 数据的实际行为**，所以上面的报错风险不会被这个单测发现。
