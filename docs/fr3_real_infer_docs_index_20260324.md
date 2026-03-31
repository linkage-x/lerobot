# FR3 Real Inference Docs Index (2026-03-24)

Use this index to avoid duplicating FR3 real-robot inference knowledge across multiple notes.

## Source Of Truth Order

1. `docs/fr3_act_infer_real_minimal.md`
   - FR3 真机推理主入口文档，记录启动默认值、输入输出合同、推荐命令和当前开放问题
2. `docs/fr3_mask2ee_training_inference_contract_20260326.md`
   - FR3 `mask2ee` 训练与推理合同，包含 checkpoint 语义和 train/infer 一致性要求
3. `docs/fr3_chunk_action_quantile_normalization_design_20260327.md`
   - FR3 chunk-based policy 的按 offset 稳健动作归一化设计与 ACT 首版落地记录
4. `docs/fr3_relative_trajectory_mvp_20260331.md`
   - FR3 从 SLAM 世界坐标 `W_s` 绝对 `ee2ee` 合同迁移到相对轨迹 policy contract 的最小可行版设计
5. `docs/fr3_relative_trajectory_mvp_implementation_checklist_20260331.md`
   - FR3 相对轨迹 MVP 的实施清单，按文件列出第一批代码改动顺序
6. `docs/fr3_act_infer_runtime_fix_20260324.md`
   - 2026-03-24 真机推理 runtime 修复的简要变更记录
7. `docs/fr3_infer_image_semantics_validation_20260323.md`
   - `left/right` 图像语义问题的最终结论
8. `docs/fr3_infer_frame_alignment_findings_20260323.md`
   - 最初 step0 对不齐问题的最终结论
9. `docs/fr3_pickplace_policy_dataset_validation_plan_20260324.md`
   - 用于定位 pick-place 剩余失败根因的调查计划和假设树
10. `docs/fr3_pickplace_policy_dataset_validation_runbook_20260325.md`
   - dataset-fed validation 工作流的操作手册，包含具体命令和产物约定
11. `docs/tactile/fr3_das_tactile_packet_investigation_20260323.md`
   - 仍未关闭的 tactile 线协议调查记录
12. `docs/fr3_replay_tracking_findings_20260319.md`
   - replay tracking 证据和分支不稳定性分析

## Documentation Rules

- put durable runtime behavior and operator guidance in `fr3_act_infer_real_minimal.md`
- put one-off implementation deltas in dated `*_fix_*.md` or `*_findings_*.md` notes
- when an investigation closes, keep only the final conclusion and remove step-by-step dead ends
- keep specialized evidence-heavy analysis separate from the main runtime doc
- pair investigation plans with an operator-facing runbook once the next action is stable

## Current Open Threads

- pick-place still does not complete on hardware; next step is to execute the dataset-fed validation runbook and localize the dominant blocker
- tactile `448-byte` payload to dataset `left_raw/right_raw` mapping
- long-rollout runtime robustness and operator safety gates
- relative-trajectory FR3 MVP: remove policy dependence on SLAM world frame `W_s` while keeping wrist-camera-first execution
- `mask2ee` is currently ACT-only; a future TODO is to generalize it into a shared state-masking mechanism for more policies with end-to-end tests
- 评估已落地的 ACT `p02/p98` 按 offset 动作归一化是否能改善 `frame 24/32/40` 的长尾误差，并决定是否推广到更多 policy
