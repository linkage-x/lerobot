# 数据采集 GUI 需求分析（2026-05-28）

## 现有架构概况

单机单用户架构：Python HTTP gateway（`gateway.py`）跑在 Jetson Thor 上，
React SPA 前端。无用户系统、无数据库、无多客户端通信协议。

| MVP 页面 | 状态 |
|---|---|
| Live Record | 完整，支持 GMSL2/Handheld 录制 |
| Dataset Processing | 完整，QC + 轨迹生成队列 |
| Episode Replay | 完整，多相机视频 + 时间线 + 触觉热力图 + 3D 姿态 + 标注 |
| Dataset Export | 完整，LeRobot v3 / MCAP / Parquet |

| Deferred 页面 | 状态 |
|---|---|
| Dashboard / QC Report / Model Evaluation / Device Manager / Task Library / Annotation & Audit | 均为空壳占位 |

---

## 需求逐条分析

### 1. 管理员端

#### 1.1 定义和下发任务给多个客户端
- **现状**：无多客户端架构，单机本地 gateway
- **结论**：需做，但需拆分阶段。Phase 1 做本地任务定义，Phase 2 做跨机下发

#### 1.2 数据统计
- **现状**：Dataset Processing 页有基础统计（dataset 数、episode 数、frame 数、QC 状态）
- **结论**：先做数据集维度汇总（Dashboard 页），按任务/用户聚合等后续

#### 1.3 查看数据采集以及标注（可视化查看、抽查、打标签、退回）
- **现状**：Episode Replay 页支持多相机视频回放 + annotation（outcome/quality/tags/notes）
- **结论**：大部分已有。需新增退回工作流：annotation 加 `reviewStatus` + `reviewComment`

#### 1.4 采集用户管理
- **现状**：无用户概念
- **结论**：建议用轻量方案：启动时 `--operator` 或 UI 选择操作员名字，不做认证

### 2. 采集端

#### 2.1 查看当前数据以及数据回放（相机、轨迹、触觉、力信号）
- **现状**：✅ 已有。ReplayInspector 覆盖多相机视频 + 时序图 + 触觉 + 3D 姿态
- **结论**：不需额外开发

#### 2.2 采集任务列表
- **现状**：无任务概念
- **结论**：本地 `tasks.json` + Task Library 页面实现

#### 2.3 数据标注

##### 2.3.1 Task Description
- ✅ 已有 `EpisodeAnnotation.taskPrompt`

##### 2.3.2 子任务切片
- **现状**：时间轴存在但无区间选择
- **结论**：核心新功能。时间轴区间选择 + 每段描述

##### 2.3.3 失败数据处理
- **现状**：有 outcome=failure 但无快捷操作
- **结论**：加"标记碰撞"/"放弃"快捷按钮，用标注过滤（不移动文件）

##### 2.3.4 分割标注
- **结论**：暂不做，等 2.3.2 验证

#### 2.4 数据导出
- ✅ 已有

### 3. 硬件连接端

#### 3.1 实时显示设备 + 历史列表 + 连接状态
- **现状**：DeviceList 有实时状态。无持久化历史
- **结论**：实现 Device Manager 页面。"历史列表"简化为设备详情面板

#### 3.2 设备配置信息
- **现状**：相机参数大部分已显示。BOX 设备详情不足
- **结论**：在 Device Manager 页面加详情面板

#### 3.3 离线设备移除
- **结论**：不做。GMSL2 是硬接线，改 YAML 即可。UI 上加"隐藏 error 设备"toggle

---

## 优先级与实现计划

| 优先级 | 需求 | 工作量 | 状态 |
|---|---|---|---|
| P0 | 2.3.2 子任务切片标注 | 中 | ✅ 已完成 |
| P0 | 2.3.3 失败数据快捷标记 | 小 | ✅ 已完成 |
| P1 | 2.2.1 本地任务管理 | 中 | ✅ 已完成 |
| P1 | 1.3 退回工作流 | 小 | ✅ 已完成 |
| P1 | 3.1+3.2 设备详情面板 | 小 | ✅ 已完成 |
| P2 | 1.2 数据统计 Dashboard | 中 | — |
| P2 | 1.1 多客户端任务下发 | 大 | — |
| P3 | 1.4 用户管理 | 中 | — |
| Skip | 3.3 离线设备移除 | — | 不做 |
| Skip | 2.3.4 分割标注 | — | 等 2.3.2 验证 |
