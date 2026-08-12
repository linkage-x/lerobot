# Dataset Processing 页面：算法侧接口需求

本文是 GUI 端 Dataset Processing 页面对算法侧后端服务的需求草案，供数据/标注/算法同事认领、实现并对齐接口。前端目前以 mock 数据驱动，gateway 已预留 `/api/processing/*` 路由。

## 1. 背景与定位

Dataset Processing 处于「Live Record → Episode Replay → Dataset Export」链路中间，负责把刚录到的 raw episode 加工成可被 Replay/Training 直接消费的数据：

- 接收 Live Record 写盘的原始 dataset（`LeRobotDataset` v3 目录）。
- 触发并跟踪 EE 轨迹生成、姿态对齐、QC 校验等异步任务。
- 维护每个 dataset 的 trajectory 版本（v1, v2, …），暴露给 Episode Replay 与 Dataset Export 选用。
- 给出可读的失败原因和重跑入口；只有 QC 通过的 dataset 才会出现在 Dataset Export 列表。

页面目前已实现的前端形态参见 `tools/data_collection_gui/frontend/src/App.tsx` 的 `DatasetProcessingPage`。

## 2. 角色与典型流程

- 数据采集员：录完一个 episode，跳到本页查看是否需要处理；点 Generate EE Trajectory 等。
- 算法工程师：批量发起 traj-gen / QC，查看失败原因；管理 trajectory 版本。
- 调度脚本（无人值守）：可被 cron 或 watcher 调用，对新出现的 dataset 自动跑流水线。

主流程：

1. Live Record 完成保存 → dataset 出现在 Processing 列表，状态 `pose_missing`。
2. 触发 `Generate EE Trajectory` → 后端建 job，状态进入 `queued` → `running`。
3. 完成后状态变为 `pose_ready`，写入 `meta/ee_trajectory_<version>.parquet`（或类似产物）。
4. 触发 `Run QC` → 状态进入 `running`，QC 通过后变为 `qc_pass`，失败为 `qc_failed` 并附带 reason。
5. 失败可重跑 `Re-run`；通过后才能在 Dataset Export 中看到这条数据。

## 3. 数据契约（前端已使用）

前端 `types.ts::ProcessingItem`：

```ts
type ProcessingStatus =
  | "pose_missing"   // raw 已存盘，未生成 EE 轨迹
  | "queued"         // 任务排队中
  | "running"        // 后台任务进行中
  | "pose_ready"     // EE 轨迹生成完毕，等待 QC
  | "qc_warn"        // QC 跑完、只有 warn 没有 fail：可导出，但导出前须显式确认
  | "qc_pass"        // QC 通过，可被 Dataset Export 消费
  | "qc_failed"      // QC 失败，需修复或重跑
  | "error";         // 任务执行失败

type ProcessingItem = {
  path: string;                    // dataset root 的绝对路径
  name: string;                    // dataset 目录名
  status: ProcessingStatus;
  trajectoryVersion: string | null;// 例如 "v2"
  qcSummary: string;               // 单行 QC 摘要
  message: string;                 // 当前状态的人话描述
  updatedAt: string;               // ISO/本地化时间戳
  totalEpisodes: number;
  totalFrames: number;
  validFramesPct: number | null;   // QC 后填，0~100
  logTail: string[];               // 最近 N 行日志，仅展示
};
```

`recordedDatasets` 与 `processing` 通过 `path` 一一对应。前端会按 `path` 做 join。

## 4. 后端 API（前端已串好的 mock 路由）

`tools/data_collection_gui/gateway.py` 已预留以下入口，目前仅返回 mock 状态：

| Method | Path                                                  | 行为                              |
|--------|-------------------------------------------------------|----------------------------------|
| GET    | `/api/snapshot`                                       | 整体状态，包含 `processing[]`     |
| POST   | `/api/processing/traj-gen?path=<dataset>`             | 触发 EE 轨迹生成                  |
| POST   | `/api/processing/qc?path=<dataset>`                   | 触发 QC                          |
| POST   | `/api/processing/rerun?path=<dataset>`                | 重新执行最近一次任务              |

需要扩展（建议）：

| Method | Path                                                  | 行为                              |
|--------|-------------------------------------------------------|----------------------------------|
| GET    | `/api/processing/job?path=<dataset>`                  | 获取该 dataset 的当前/最近 job 详情 |
| GET    | `/api/processing/job/log?path=<dataset>&job_id=...`   | 流式或一次性返回完整日志           |
| GET    | `/api/processing/versions?path=<dataset>`             | 列出现有 trajectory 版本           |
| POST   | `/api/processing/promote?path=<dataset>&version=...`  | 将某个版本标记为 active（被 Replay/Export 默认使用）|

请求/响应 schema 沿用现有 `_json_response` 风格；保留 `Access-Control-Allow-Origin: *` 以便前端 dev server 直连。

## 5. 算法子任务清单

### 5.1 EE Trajectory Generation（pose_missing → pose_ready）

输入：

- dataset root（`outputs/datasets/<run>`）
- raw 数据，至少包含：
  - `data/chunk-*/file-*.parquet`，列含 `observation.state`, `action`, `observation.images.*` timestamps
  - 对 handheld 流程：`observation.handheld_gripper.*`, `observation.tactile.*`
  - 对 teleop / sim 流程：`observation.state` 已含 joint pose
- 配置/校准：camera intrinsics、TCP-to-camera 变换、机器人 URDF 或 base frame 约定

期望输出：

- 在 dataset 内写入 trajectory 产物，建议路径：
  - `derived/ee_trajectory_<version>/data/chunk-*/file-*.parquet`
  - 与原 `data/` 并列，frame 数与原 episode 对齐
  - schema 至少包含：`frame_index`, `episode_index`, `timestamp`, `ee.x/y/z/qx/qy/qz/qw`, `ee.frame`（如 `base_link`），可选 `gripper.pos`
- 写 `derived/ee_trajectory_<version>/manifest.json`：
  - `version`, `created_at`, `source_columns`, `algorithm`, `params`, `valid_frames_pct`
- 更新 `meta/processing.json`（见 §6）

算法侧需要明确：

- 当 `observation.state` 已含完整 EE pose 时（如 fr3 mujoco dataset）：直接复制并做单位/坐标系归一即可，记录"identity"算法。
- 当数据仅有 joint angles：用 FK + 已知 URDF 推 EE。
- 当数据来自 handheld（无 joint）：用相机/IMU/SLAM 估计 EE 6D pose；这里需要算法同事给出参考实现（如多目立体 + AprilTag 或 visual-inertial）。

### 5.2 Quality Check（pose_ready → qc_pass / qc_failed）

要覆盖的检查项至少包括：

1. Schema 完整性：`info.json` 声明的所有 features 都有非空列；时间戳列单调递增。
2. 时间对齐：每帧 `observation.device_capture_timestamp` 的最大 skew ≤ 阈值（默认 50 ms，可配置）。
3. 视频完整性：每个 `observation.images.<cam>` 的 mp4 帧数 == parquet 帧数；时长与 fps 匹配。
4. EE 轨迹连续性：相邻帧 `||Δposition||` ≤ 阈值（默认 5 cm），相邻 quat 的角差 ≤ 阈值（默认 30°）。
5. Gripper 行为合法：`gripper.pos` 范围 [0, 1]，无连续 NaN。
6. 触觉/Soft sync 字段（如存在）：值域合理、无掉帧丢失。

每条检查产出 `{ name, severity, status: "pass"|"warn"|"fail", value, threshold, message }`。整条 dataset 的 status 取最严重等级。

**warn 必须是自己的状态（`qc_warn`），不能并进 `pose_ready`。** 曾经是并进去的：只要有一条 warn，
dataset 就从 Dataset Export 列表里消失，而 Processing 页显示的却是「等待 QC」——导出被挡住，理由却
没人看得见，操作者学到的结论是「跑 QC 会弄坏导出」。现在 `qc_warn` 可导出，但 gateway 会拒绝没有
`acknowledge_warnings=1` 的请求，前端据此弹确认框把每条 warn 原文列出来；确认后放行，并把「越过了哪
几条 warn」写进事件日志。判据同 replay 的 MuJoCo 门：跑过并失败的校验可以在把错误摆到眼前的确认框里
被越过，没跑过的不行。

输出：

- `derived/qc/<version>.json`：完整 check 列表 + 概要。
- 更新 `meta/processing.json` 中该版本的 `qc` 字段。

### 5.3 Job 调度与日志

- Job 异步执行，但 `/api/processing/traj-gen` 应在入队后立即返回 `running` 或 `queued` 状态（不阻塞 HTTP）。
- 后台 worker：建议进程/线程都行，避免每个 GET 重复扫盘——可用一个常驻 supervisor。
- 日志：建议落在 `derived/jobs/<job_id>.log`，gateway 暴露 tail 接口。
- 失败语义：
  - 输入数据缺失：`pose_missing` → 保持原状态 + `message` 告知；不要进入 `error`。
  - 算法异常：`error`，并把 traceback 摘要写入 `message` 与日志末尾。
  - QC 失败：`qc_failed`，前端会让用户读 `qcSummary` 决定怎么修。

### 5.4 版本与 promote

- 每次 traj-gen 自增 `version`，旧版本保留不要覆盖（便于回滚比较）。
- `active` 版本写在 `meta/processing.json::active_version`；Replay/Export 默认取这个值。
- promote API 仅改 `active_version`，不删旧文件。

## 6. dataset 内的状态文件建议

放在 dataset 根的 `meta/processing.json`，由后端写入、前端只读：

```json
{
  "active_version": "v3",
  "versions": {
    "v1": { "created_at": "2026-05-07T17:12:10Z", "algorithm": "identity", "qc": { "status": "pass", "valid_frames_pct": 99.4 } },
    "v3": { "created_at": "2026-05-08T09:11:02Z", "algorithm": "fk_franka", "qc": { "status": "fail", "reason": "skew>50ms" } }
  },
  "current_job": {
    "id": "20260508-0911-fk_franka",
    "kind": "traj-gen",
    "status": "running",
    "started_at": "2026-05-08T09:11:02Z",
    "progress_pct": 42,
    "log_path": "derived/jobs/20260508-0911-fk_franka.log"
  }
}
```

gateway 在生成 snapshot 时读这个文件，组装出 `ProcessingItem`。

## 7. 测试与 fixture

- 推荐用 `outputs/datasets/fr3_quest3_pika_gripper_20260507_171208` 作为 happy-path fixture：已有 `observation.state` 中完整 EE pose，traj-gen 跑 identity 算法即可生成 v1。
- 故意构造一个 `qc_failed` fixture：取上述 dataset，删除某个 mp4 文件，跑 QC 应该报 video frame count mismatch。
- handheld fixture：从 `tools/handheld/handheld_record_example.yaml` 录一个短 episode；这时 traj-gen 应跑 SLAM/估计算法（待算法侧选型）。

## 8. 非目标 / 后续

本期不要求：

- 不要在本页做真机回放——那是 Episode Replay 的事。
- 不要做训练数据 split / shuffle——那是 Dataset Export 的事。
- 不要在本页内联视频/3D 预览——用户已经能在 Episode Replay 看。

后续可扩展：

- 批量操作（多选 dataset 一起跑 traj-gen / QC）。
- 任务优先级与并发上限。
- 与 Annotation 流水线打通：QC 通过的 dataset 自动进入待标注队列。

## 9. 协作约定

- 任何路径都用 dataset root 的相对路径或绝对路径，避免依赖 cwd。
- 写盘要原子（先写 `.tmp` 再 rename），防止 gateway 在中间状态读取。
- 时间戳一律用 ISO-8601 UTC；前端按本地时区渲染。
- 状态机由后端做唯一权威，前端仅渲染；前端不会推断 `pose_missing → queued`。

## 10. Owners / 后续 review

- 后端 API：data-collection-gui 维护者
- traj-gen 算法：fr3 / handheld 算法组
- QC 检查项：数据质量负责人
- 文档维护：本文件改动请同步通知 GUI 与算法 owner
