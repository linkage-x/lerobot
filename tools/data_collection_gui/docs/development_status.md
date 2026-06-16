# 开发现状（2026-05-28 末）

下次接手按本文档入口往下读，所有路径都是 repo 内相对路径。

---

## 总览

| 任务 | 状态 |
|---|---|
| GUI P0~P1 需求实施 | ✅ 已合并 (commit `d69ff7ab update ui`) |
| PR1: persistent_session 模块 + burn-in | ✅ 已提交 (commit `e1a6290e`) |
| PR2: thor_record 集成 PersistentCameraSession | ✅ **代码就绪 + BOX 共启验证通过，未提交** |
| BoxClient 自动设 `BOX_SDK_URDF` | ✅ 新增（消除"忘 source setup_env.sh 就崩"事故） |
| 长 burn-in (100+ ep with BOX) | ⏳ 待跑 |
| DEPLOYMENT.md 更新 | ⏳ 待 PR2 落地 |

---

## PR1 完成清单（已提交）

**Commit**: `e1a6290e Add persistent GStreamer pipeline session (PR1 of Option A)`

**文件**：
- `tools/thor/gmsl2/persistent_session.py` — `PersistentCameraSession` + `_Stream` 状态机 + 独立 demo CLI
- `tests/scripts/test_thor_persistent_session.py` — 12 个 mock GStreamer 单测
- `tools/data_collection_gui/docs/option_a_persistent_pipeline_design.md` — 方案 A 设计 + burn-in 数据
- `tools/data_collection_gui/docs/pr1_implementation_log.md` — PR1 实施日志（4 个问题的处理）

**关键 burn-in 数据**：
- 100 episode RSS 增长 23KB/ep（无泄漏）
- discard 文件清理 9/9 PASS（grace=0.05s 都够）
- bus dispatch 0.14ms / 端到端 EOS 12.8ms
- 11 路 connect 11.67s，跨相机切片精度 19.5ms (`first_wall_s`)

---

## PR2 当前状态（未提交，本地 3 个修改文件）

**本地未提交改动**：
```
 M tests/scripts/test_thor_persistent_session.py    # +2 个 cleanup_warmup_files 测试
 M tools/thor/gmsl2/persistent_session.py           # +cleanup_warmup_files / +restart_stream
 M tools/thor/gmsl2/thor_record.py                  # 主循环切到 PersistentCameraSession + BOX 先启
 M tools/thor/box_sdk/box_client.py                 # BoxClient.start() 自动设 $BOX_SDK_URDF
```

### PR2 代码改动具体内容

#### `tools/thor/gmsl2/persistent_session.py`
新增两个 PR2 用到的方法：
- `cleanup_warmup_files(keep_last_n=3)` — 删除 warmup 目录里过期的 fragment（每路 sid 保留最近 N 个）
- `restart_stream(sid)` — NULL→PLAYING 重启单路 pipeline（PR2 暂未在 thor_record 中调用，留 PR3）

#### `tools/thor/gmsl2/thor_record.py`
1. import `persistent_session as ps`
2. **`_extract_pts_offset()`** 改成 **`_pts_offset_from_handle(handle)`**：从 EpisodeHandle 的
   `first_wall_s - t0_wall_s` 算 per-camera 平均偏移，**不再用 ffprobe**
3. **`_write_episode_meta()`** 签名从 `(EpisodeResult, ...)` 改成 `(EpisodeHandle, ..., wallclock_start_utc, wallclock_end_utc)`。
   meta.json 的 `sync_reference` 字段从旧的 `camera_spawn_*_s` 改成新的：
   - `split_now_wall_s` — start_episode() 触发时刻
   - `camera_first_wall_s` — 每路实际开新 fragment 的 wall time（跨相机对齐锚点）
   - `camera_first_pts_s` — 每路 buffer PTS（per-stream，不可跨相机比）
4. 新增 `_stream_configs(usable, cfg)` helper：RecorderConfig → StreamConfig 列表
5. **`main()` 主循环替换**：
   - **BOX 先启**：`box = bc.BoxClient(box_cfg); box.start()` 在 `pcs.connect()` 之前
     （rate-report 完成后再 spawn 11 路 nvarguscamerasrc，避免 BOX SDK 在
     GStreamer 已经跑的状态下做内部初始化）
   - `dataset_root.mkdir()` + BOX 段之后 `pcs = ps.PersistentCameraSession(...); pcs.connect()`
   - 循环里 `gr.EpisodeSession(...).start(ep_dir)` → `pcs.start_episode(ep_dir, ep_idx)`
   - 循环里 `session.stop(streams)` → `pcs.stop_episode(handle)`
   - 流早退检测 `dead = [s for s in streams if s.proc.poll() is not None]` → `pcs.poll_errors()`
   - Discard 路径 `shutil.rmtree(ep_dir)` → `pcs.discard_episode(handle)` + `ep_dir.rmdir()`
   - 每个 episode 后 `pcs.cleanup_warmup_files(keep_last_n=3)`
   - finally 块：`pcs.disconnect()` 在 `box.stop()` 之前（对称：后开的先关）

#### `tools/thor/box_sdk/box_client.py`
- `BoxClient.start()` 进入时先 `os.environ.setdefault("BOX_SDK_URDF", ...)`，
  路径由 `Path(__file__).resolve().parent / cfg.urdf_relpath` 构造（box_client.py
  本身就在 `tools/thor/box_sdk/` 下，跟 `share/monte_gripper.urdf` 同级）。
  这消除了"忘 source `tools/thor/box_sdk/setup_env.sh` → 首个 UDP 包来时
  C++ urdf_parser throw 没捕获 → SIGABRT"的事故。
  注：`LD_LIBRARY_PATH` 必须仍由调用方在 `dlopen` 前设好，BoxClient 改不了。

### PR2 验证结果

| 测试 | 配置 | 结果 |
|---|---|---|
| 单测全跑 | 47 测试 (45 旧 + 2 新) | ✅ ALL PASS |
| Thor import sanity | thor_record import | ✅ OK |
| **E2E 测试（无 BOX）** | 2 路 + `--no-box --skip-hardware-sync` + 2 个 episode (1 save + 1 discard) | ✅ 完美 |
| **E2E 测试（含 BOX，未 source setup_env.sh）** | 2 路 + `--skip-hardware-sync` | ✅ box.start + pcs.connect + Episode 0 ready + 30s RdSync 心跳无崩 |
| **长 burn-in (100 ep with BOX)** | 2 路 + `episode_time_s=3` + `num_episodes=100` | ✅ 100/100 saved, 0 abort, 471s (4.7s/ep), GStreamer 干净退出 |

### E2E 无 BOX 测试细节（成功）

- Connect 2 路 = 2.6s
- StartEpisode emit < 0.5ms
- Episode 0 (auto-save on duration_reached=3s)：cam_00.mkv 5.8MB + cam_02.mkv 7.0MB
- Episode 1 (discard)：split → unlink 2 个 mkv → `ep_dir.rmdir()` 成功
- `_warmup/` 目录每路保留 3 个 fragment（cleanup_warmup_files 工作）
- meta.json 含完整 `sync_reference`：`split_now_wall_s` + `camera_first_wall_s` (cam_00 +305ms, cam_02 +725ms from split_now)
- stdout 协议保持（"Episode N ready" / "Recorded K frames" / "Episode saved." / "Episode discarded"）

**注意**：测试用的 yaml 默认 `iframe_interval=60` → 首 episode 切片延迟 305-725ms。生产建议改为 30（PR1 burn-in 用的就是 30）。

### 100-ep burn-in RSS 增长（已定位 + 两阶段修复）

PR1 mock 单测时 RSS 是 ~23 KB/ep（无泄漏）。真实硬件 100 ep burn-in 上 PR2 commit
时观测到 **~2.8 MB/ep**（100×）。最初列了 4 个猜测（nvargus / glib bus / box
SDK / BoxClient `_record_samples`），**全部错了**。

#### 根因定位（系列 ablation）

| 测试 | RSS 涨幅 |
|---|---|
| `--no-box` 50 ep（GStreamer-only） | ~0 |
| Box-only idle 90s（SDK 跑 UDP + 写 CSV，无 recording） | ~0 |
| Box-only + `start_recording`/`stop_recording` 循环 50 ep | 140 KB/ep |
| Box-only + `lr3.write_box_lerobot_v3_episode` 50 ep（**无 GStreamer**） | **7.7 MB/ep** |
| 全栈 thor_record + GStreamer + BOX + lr3 100 ep | **2.9 MB/ep**（被 GStreamer baseline 稀释） |

**根因**：`tools/thor/gmsl2/thor_lerobot_v3.py:write_box_lerobot_v3_episode`
是个 O(N²) 增量写法。每个 episode 都：

1. `pq.read_table(data_path).to_pylist()` —— N×per_ep_rows 个 Python dict
2. `existing_rows = [row for row in existing_rows if ...]` —— 又一份 copy
3. `sorted([*existing_rows, *rows], ...)` —— 又一份
4. `state_values = [list(row["observation.state"]) for row in all_rows]` —— 又一份
5. `action_values = [list(row["action"]) for row in all_rows]` —— 又一份
6. `_stats([[float(row[col])] for row in all_rows])` × 5 列 —— 5 份
7. 每个 episode 又 filter `all_rows` 重建 episode_rows

每 ep peak Python heap ≈ N × per_ep_rows × ~10 副本。glibc malloc arena
跟着峰值需求扩张后**不归还 OS**，所以 RSS 永久跟着最大需求增长。

#### 第一阶段修复（已合）

用 Arrow Table-native 路径替换 to_pylist + sorted + dict comprehension：
- `pa.concat_tables` + `Table.sort_by` 在 Arrow columnar buffer 上做，不再
  把整个 table 解 to 成 Python dict
- 新 helper `_table_column_stats(table, col, width=)` 用 numpy + `np.quantile`
  直接对 Arrow buffer 做向量化 stats（zero_copy_only=False，必要时单次拷贝
  到 float64 numpy view）
- Episode rollup 用 `table.group_by("episode_index").aggregate([(index, min),
  (index, max), (index, count)])`，O(N) Arrow agg 替代两层 Python for + filter

#### Fix 后实测

`lr3.write_box_lerobot_v3_episode` 单独 ablation：
- 修复前 50 ep：29 MB → 417 MB（**+388 MB / 7.7 MB/ep**）
- 修复后 50 ep：30 MB → 350 MB（+320 MB / 6.4 MB/ep）
- 修复后 100 ep：29 MB → 475 MB（+446 MB / 4.5 MB/ep；ep 25-50 / 60-75 plateau）

全栈 100 ep（thor_record + GStreamer + BOX）：
- 修复前：462 MB (ep 2) → 901 MB (ep 100) = **+439 MB / 4.5 MB/ep**
- 修复后：488 MB (ep 4) → 764 MB (ep 99) = **+276 MB / 2.9 MB/ep（改善 36%）**
- 模式：plateau-跳级（典型 glibc arena 扩张），不再是平滑线性

#### 第二阶段修复（已合）

剩余 ~2.9 MB/ep 的根因是第一阶段仍然每 ep 执行 `pq.read_table(data_path)`、
`pa.concat_tables(...)`、`Table.sort_by(...)`、再 `pq.write_table(...)`：虽然已经
避免 Python dict 级别的 O(N²) 副本，但每次保存仍有 O(N) Arrow working memory。
glibc arena 会跟着最大 episode 数时的峰值扩张，所以 RSS 仍呈 plateau-jump 增长。

落地方案选 **长生命周期 `pyarrow.parquet.ParquetWriter`**，而不是每 ep 独立
parquet 文件：
- 保持 LeRobot v3 现有单 `data/chunk-000/file-000.parquet` layout，不需要改
  downstream loader。
- 新增 `tools/thor/gmsl2/thor_lerobot_v3.py:Lr3Writer`，生命周期为
  `open_box_lerobot_v3_writer(...)` / `append_episode(...)` / `finalize()`。
- `append_episode(...)` 只把当前 episode rows 写成一个 row group，热路径不再读
  existing parquet、不 concat、不 sort，peak heap 只跟当前 episode 行数相关。
- `thor_record.py` 在 `pcs.connect()` 后打开 writer，在保存分支 append；discard
  分支不调用 writer，因此不会写 partial episode。
- `finally` 中先 `lr3_writer.finalize()`，再 `pcs.disconnect()` / `box.stop()`：
  `finalize()` 关闭 parquet footer，并基于最终 parquet 写 `meta/stats.json`、
  `meta/episodes/chunk-000/file-000.parquet`、`meta/info.json`。
- 保留旧 `write_box_lerobot_v3_episode(...)` 兼容入口；真实 GUI recorder 走新的
  stateful writer 路径。

回归测试：`tests/scripts/test_thor_lerobot_v3_pts.py` 覆盖连续 append 两个 episode
时 `pq.read_table` 在热路径中不被调用，只允许 `finalize()` 读最终 parquet 一次来
计算 stats。

### 操作员 stdin "Enter Enter" 修复

`_drain_until` 之前在 `duration_reached` 的 0.2s 操作员窗口内会把队列里**所有**
非 `save/discard/quit` 命令（典型是 `start` = `\n`）pop 掉当 noise。这意味着 GUI
连发 "Enter Enter" 想 "save 当前 + start 下一个" 时，第二个 Enter 会被吞掉。
修复：drain 看到队首不是接受的 kind 时立刻 return None（让默认 auto-save 触发），
**不 consume** 队首命令。

副作用：burn-in 预填 110 个 `\n` 之前每 ep 被 drain 吞掉 4 个，22 ep 后队列空了；
修复后正确地一 ep 消耗一个 `\n`，100 ep 用 100 个 `\n` 跑完。

### BOX 启动崩溃根因诊断与修复（已完成）

之前看到 `[TLV_LOG_UPLOAD] ... udp recv tlv[0] ... Aborted (core dumped)` 误判为
"BOX SDK + GStreamer 并发崩"。**实际根因完全无关**：

```
Error:   File thirdpart/monte_gripper.urdf does not exist
         at line 55 in ./urdf_parser/src/model.cpp
terminate called after throwing an instance of 'std::invalid_argument'
  what():  The file thirdpart/monte_gripper.urdf does not contain a valid URDF model.
```

原始测试 wrapper 只 `tail -30`，错过了崩溃前的这条 C++ 错误。真正流程：

1. `box.start()` 成功，BOX SDK UDP listener 启动
2. 首个 UDP 包到（`type=0x0100`，握手）
3. BOX SDK C++ 端初始化 gripper kinematics，加载 URDF
4. 找不到 `BOX_SDK_URDF` env，fallback 到相对路径 `thirdpart/monte_gripper.urdf`
5. 该路径不存在 → `std::invalid_argument` 抛出，没人 catch → `std::terminate()` → SIGABRT

**修复**：`BoxClient.start()` 进入时自动 `os.environ.setdefault("BOX_SDK_URDF",
absolute_path)`，路径基于 `__file__` 推导，无需依赖 cwd 也无需 `source
setup_env.sh`。`LD_LIBRARY_PATH` 仍需调用方在 `dlopen` 前设好（gateway 启动包装
脚本 / `setup_env.sh` 一直在做这件事，符合现状）。

**验证**：仅设 `LD_LIBRARY_PATH=tools/thor/box_sdk/lib:$LD_LIBRARY_PATH`，
**不** source `setup_env.sh`，跑 2 路 + BOX 含 `--skip-hardware-sync`：
- `Box devices: box_gripper, box_imu, box_trigger, box_six_d_force, box_touch_left, box_touch_right`
- `Box rates: box_gripper=198, box_imu=198, ...`
- `Connected 2 pipelines in 2.6s`
- `Episode 0 ready`
- 持续 30s `RdSync: S=200, F=0, AvgRdT=3ms, ...` 心跳，无 abort marker

---

## 下一步

### 1. 拿到当前状态
```bash
cd ~/Codes/lerobot
git log --oneline -3
# e1a6290e Add persistent GStreamer pipeline session (PR1 of Option A)
# d69ff7ab update ui
# 9e7a45d8 Support deployment on jetson thor.

git status
# 应该看到 PR2 的 4 个修改文件 (persistent_session.py / thor_record.py /
# box_client.py / test_thor_persistent_session.py)
```

### 2. 复现成功的 E2E 测试

`tools/data_collection_gui/scripts/run_thor_pr2_test.sh`（见下文 §5）会做：
- prepare /tmp/pr2.yaml（2 路 + episode_time_s=3）
- 设 LD_LIBRARY_PATH 指向 box_sdk/lib
- mkfifo /tmp/stin + 后台 writer 撑开
- setsid 启动 thor_record，30s 后 tail /tmp/log 验证心跳

注意 FIFO 必须有写端 holder（`tail -f /dev/null > /tmp/stin &`），否则 python 在
`< /tmp/stin` 上 block 在 FIFO open() 永远不启动。

### 3. 长 burn-in (100+ ep with BOX)

参考 PR1 burn-in 脚本（已删，需重写）：
- 启 thor_record，stdin 喂 `\n y\n` × 100（auto-save by duration_reached=3s 后 'y' 确认）
- 监控 RSS（每 5 个 ep 取一次 `ps -o rss= -p <pid>`）
- 验收：100 ep 全部 saved，RSS 线性增长 < 50KB/ep，无 Aborted

### 4. PR2 commit

burn-in 通过后：
```bash
git add tools/thor/gmsl2/persistent_session.py \
    tools/thor/gmsl2/thor_record.py \
    tools/thor/box_sdk/box_client.py \
    tests/scripts/test_thor_persistent_session.py

git commit -m "Switch thor_record to PersistentCameraSession (PR2)..."
```

同时更新 `tools/thor/DEPLOYMENT.md`：
- §10 已知问题里的 "11 路同时启动 NvBufSurfaceFromFd Failed" 标为"已过时——
  持久 pipeline 模式只在 Connect 时 spawn 一次"
- §11 同步架构里 sync_reference 模型从 `camera_spawn_*_s` 改到 `camera_first_*_s`
- 加一节 "StartEpisode 等待从 ~11s 降到 < 0.1s"
- §7.1 box_sdk 验证段补充说明：`BOX_SDK_URDF` 现在由 `BoxClient.start()` 自动
  设置，`setup_env.sh` 中的 `BOX_SDK_URDF` 行只在 `box_client` 之外的命令行调用
  （如 `python3 demo.py`）时仍然需要。

### 5. 留存的复现脚本

测试 wrapper 见 `/tmp/run_thor_test.sh`（本地）或同步后的 Thor 端 `/tmp/run_thor_test.sh`。
关键点：
```bash
set -u
cd ~/lerobot
export LD_LIBRARY_PATH=/home/nvidia/lerobot/tools/thor/box_sdk/lib:${LD_LIBRARY_PATH:-}
rm -f /tmp/stin /tmp/log && mkfifo /tmp/stin
tail -f /dev/null > /tmp/stin &   # FIFO 写端 holder
setsid bash -c "cd ~/lerobot && \
    export LD_LIBRARY_PATH=...lib:\${LD_LIBRARY_PATH:-} && \
    PYTHONPATH=src:. exec python3 -m tools.thor.gmsl2.thor_record \
        --config-path /tmp/pr2.yaml --skip-hardware-sync \
        < /tmp/stin > /tmp/log 2>&1" </dev/null >/dev/null 2>&1 &
sleep 30 && tail -80 /tmp/log
```

---

## 关键技术决策（参考性记录）

1. **跨相机对齐用 `first_wall_s` 而非 `first_pts_s`**：PR1 burn-in 实测，pipeline PTS 跨相机不可比（偏差 10s 量级），wall-clock 才可比（偏差 19.5ms）。
2. **`iframeinterval` 必须搭配 `idrinterval`**：nvenc 上两者分开，仅设 iframeinterval 会让 splitmuxsink 等默认 ~256 帧才能切。
3. **discard 复用 stop_episode 的切片**：stop_episode 已经 emit split-now 切走 EPISODE 文件，discard 只需 unlink + rmdir。
4. **warmup 目录定期清理 keep_last_n=3**：splitmuxsink async-finalize 不会自动 rotate，长 session 必须主动清。
5. **错误恢复用 poll_errors，pipeline 重启留 PR3**：PR1 burn-in 验证 bus dispatch 0.14ms，PR2 只做检测+中止 episode，pipeline 自动重启（已加 `restart_stream` 方法但未调用）留下个 PR。
6. **保留 `gmsl2_record.EpisodeSession` 不动**：`gmsl2_record.py` 的 CLI 入口仍然能跑（独立 gst-launch 子进程），PR2 只切了 `thor_record.py` 这条 gateway 入口。
7. **V4L2 gain 与 Argus gainrange 必须拆开**：AR0234 已验证的 `gain: 320` 是驱动单位，只能用于 `v4l2-ctl`；`nvarguscamerasrc gainrange` 使用 0..4 float scale。2026-05-28 的 `Error 0x00000005` / `Bad gain range: [320.00, 4.00]` 指向该混用问题，因此 recorder 新增 `argus_gain`，默认 0.0，不再把 `gain` 传给 Argus。

---

## 历史文档索引

- `tools/data_collection_gui/docs/gui_requirements_analysis.md` — 客户需求分析（P0~P1 已完成）
- `tools/data_collection_gui/docs/option_a_persistent_pipeline_design.md` — 方案 A 完整设计 + burn-in 数据
- `tools/data_collection_gui/docs/pr1_implementation_log.md` — PR1 实施日志（4 个问题处理 + 3 项验证）
- `tools/data_collection_gui/docs/development_status.md` — 本文档
- `tools/data_collection_gui/docs/traj_gen_thor_gmsl2_compatibility.md` — traj-gen（EE 轨迹生成）已切到 gmsl2 AprilTag 追踪，Thor 真机可用：按钮跑 `run_april_cube_tracking_local.sh` 生成 sidecar，v3 与 gmsl2 两条 timeline 路径都显示 EE pose，时间戳走 PWM 网格
- `tools/thor/DEPLOYMENT.md` — Thor 部署清单（PR2 完成后需更新）
