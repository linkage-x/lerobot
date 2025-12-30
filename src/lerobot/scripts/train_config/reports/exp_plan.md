# W&B 实验迭代自动化计划（ACT/OT 系列）

目标
- 用 W&B 曲线的早期信号自动评估现有/新实验，按规则调整超参，少量步数快启快停，循环迭代，收敛到更好的 `eval/offline_eval/avg_l1` 与更稳定的 OT 指标。

前置与依赖
- 代码入口：`python -m lerobot.scripts.lerobot_train --config_path=<cfg.json>`（支持 CLI 覆盖 `--steps`/`--wandb.*`）。
- 数据与现有配置：`src/lerobot/scripts/train_config/*.json`；报告样例：`src/lerobot/scripts/train_config/reports/act_ot.md`。
- W&B：项目形如 `kjust-pinduoduo/lerobot`；本地已存在指标键参考：`wandb_keys.json`。

关键指标与判据（早停/早期评估用）
- 训练收敛：`train/l1_loss` 2k 步内较首值下降 ≥40%；平台期判据：连续 500 步下降 <2%。
- OT 质量：`train/ot_ot_pi_sum` 逐步上升到 0.05–0.3 为宜；过低(<1e-3) 表示 reg/tau 偏小或 label 权重偏大；过高(>0.6) 可能拉偏 BC。
- 对角性：`train/ot_ot_pi_diag` 趋近 `pi_sum` 越好（时间一致性强）。
- 代价下降：`train/ot_ot_cost/observation.state` 1k–3k 步内相对首值下降 ≥10% 为合理。
- 稳定性：`train/grad_norm` 不应长时间 >100；学习率恒定 1e-5 时抖动大需加大 reg 或降 `lambda_ot`。

决策规则（基于上面指标）
- π 过低（<1e-3 且不升）：提高平滑度与松弛度：`ot.loss_config.reg *= 1.5~2.0`；`tau_src,tau_tgt` ∈ {0.5, 1.0, 2.0} 或使用更大窗口；保留 `heuristic=true`。
- π 偏高（>0.3）但 `state cost` 不降：减小 label 权重 `weight_label` ∈ {0.001, 0.0005, 0}；必要时小降 `lambda_ot`。
- L1 平台期（2k 步降幅 <2%）：尝试增大 batch 或微调 lr（±2x），并保留当前 OT 设置不变以对照。
- 曲线抖动大：增大 `reg` 或 `window_size`（只 jitter 源序列），并确保 `save_freq` 足够大以减少覆盖式保存。

自动化流程（迭代 Loop）
1) 选取与拉取
   - 读取输入：
     - 现有 run 列表（可从 W&B project 以 tag/group 过滤）或本地 `compare_runs*.json`。
     - 目标配置模板集合：`src/lerobot/scripts/train_config/act_fr3_ot_*.json`。
   - 用 W&B Public API 拉取每个 run 的 `history`（键见 `wandb_keys.json`）。
2) 分析
   - 计算首、末、最优、Δ、pct（已有 JSON 结构示例：`compare_runs_*.json`）。
   - 按“关键指标与判据”给每个 run 打标签（如 π 过低/过高、cost 不降、L1 平台等）。
   - 生成 Markdown 小结，追加至 `src/lerobot/scripts/train_config/reports/act_ot.md`。
3) 生成新实验
   - 基于“决策规则”从当前最佳/问题 run 派生 1–3 个变体：
     - 仅改 OT：`reg/tau/window/weight_label/lambda_ot`；
     - 仅改 BC：`optimizer_lr/batch_size`。
   - 输出新的配置文件至 `src/lerobot/scripts/train_config/`（命名含关键信号与短 id）。
4) 小步试跑（快速信号）
   - 每个新实验仅跑 `--steps=2000`，`--eval_freq=200`，`--log_freq=50`；
   - 以 `--wandb.group=policy:<type>-seed:<seed>-dataset:<repo>` 归组；
   - 进程完成即自然终止（由 `steps` 控制），无需手动杀进程。
5) 汇总与迭代
   - 通过 W&B API 读取新 run 的前 2k 步曲线，复用“分析+判据”，更新报告与下一轮待跑队列。
   - 触发下一轮（最多并发 N=2–4，按 GPU 资源限制）。

执行接口（拟实现）
- 分析脚本：`utils/auto_loop/analyze_wandb.py`
  - 输入：`--entity <e> --project <p> [--group <g>|--run_ids <...>]`。
  - 输出：`reports/data/<date>_<topic>.json|md`；返回建议动作列表。
- 生成与编排：`utils/auto_loop/generate_and_run.py`
  - 依据建议动作复制/修改模板 cfg，启动训练子进程：
    - 示例：`python -m lerobot.scripts.lerobot_train --config_path src/lerobot/scripts/train_config/act_fr3_ot.json --steps 2000 --eval_freq 200 --log_freq 50 --wandb.project <p> --wandb.entity <e>`。
  - 支持 `--resume true --config_path <checkpoint>/train_config.json --steps <new_total>` 形式做累积步训练。
- 主循环：`utils/auto_loop/autorun.py`
  - 周期：分析 → 生成 → 小步试跑 → 复盘 → 决策；
  - 并发：队列执行，GPU 资源检查（如 `nvidia-smi`）。

容错与安全
- 网络/鉴权：需要 `wandb login` 或设置 `WANDB_API_KEY`；失败时退化到本地 `compare_runs*.json` 分析。
- 训练异常：捕获非 0 退出，保留子进程日志与 `outputs/train/<date>/...`；
- 提前停止：如监测到 `grad_norm` 爆炸或 loss=nan，立刻终止该子进程并记录为“无效实验”。
- 重复实验去重：依据 `(cfg hash, seed)` 去重；

需要你确认的信息
- W&B `entity/project`、首批要分析的 run id 或筛选条件（tag/group/query）。
可直接跑的 6 个迭代配置（都已写入仓库）
  按优先顺序建议先跑 1→2→3，4 和 5 作为对照，6 是保守基线强化版。

  1. 平衡方案（适中窗口+适中 sharpness + 适中正则）

  - 文件：src/lerobot/scripts/train_config/act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015.json
  - 关键：window=8, sharpness=0.75, tau=1.0, reg=0.15, lambda_ot=0.14, action_lbl=0.015, topk_src_episodes=15
  - 适用：pair_info 较准但仍有轻微错位；希望稳步提升离线指标

  2. 严格对齐（无窗口+更强动作标签+更高 OT 权重）

  - 文件：src/lerobot/scripts/train_config/act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003.json
  - 关键：window=0, sharpness=2.0, tau=0.5, reg=0.15, lambda_ot=0.18, action_lbl=0.03, batch_ratio=0.75, topk=10
  - 适用：pair_info 很准；目标是把图像对齐与动作标签都压实

  3. 无窗口鲁棒（极弱对齐假设，适合错位/抖动）

  - 文件：src/lerobot/scripts/train_config/act_fr3_ot_99_20_next_nowindow_tau2_reg015_lambda015_action001.json
  - 关键：no_window=true, tau=2.0, reg=0.15, lambda_ot=0.15, action_lbl=0.01
  - 适用：时序错位明显/对齐不稳，先靠“广义一对多”放宽

  4. 收紧窗口+中等 sharpness（更“挑剔”的邻域）

  - 文件：src/lerobot/scripts/train_config/act_fr3_ot_99_20_next_w6_sharp1_lambda016_action001.json
  - 关键：window=6, sharpness=1.0, reg=0.20, lambda_ot=0.16, action_lbl=0.01
  - 适用：pair_info 较准，想进一步强化局部对齐

  5. 轻 OT（偏向 BC 稳定性）

  - 文件：src/lerobot/scripts/train_config/act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005.json
  - 关键：window=12, tau=0.25, reg=0.10, lambda_ot=0.08, action_lbl=0.005
  - 适用：当你观察到 ot_loss 太容易降为 0 或训练对 OT 特别敏感时，用作稳健对照

  6. Top‑K 源筛选 + 适中窗口（强调高质对）

  - 文件：src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20.json
  - 关键：topk=20, window=8, sharpness=0.5, tau=1.0, reg=0.20, lambda_ot=0.12, action_lbl=0.02
  - 适用：pair_info 质量参差，希望把学习集中在更优配对上

- 每轮小步数与最大并发（默认 2k 步、并发 需要你检查目前空闲的gpus，从而确定。 例如，空闲2,那么并发2）。
- 训练入口仅限 ACT+OT 系列。
- 资源与时间窗：单次 loop 预算时长、允许夜间运行。

落地里程碑
- M1: 接入 W&B 分析与报告自动生成（不跑新实验）。
- M2: 支持基于规则生成 1–3 个变体并小步试跑，回填报告。
- M3: 循环编排 + 断点恢复 + 并发控制，跑通 2–3 轮迭代。

附：当前本地统计样例（来自 `compare_runs*.json`）
- 多组 run 在 1–2k 步内：`train/l1_loss` 下降 45–70%，`pi_sum` 由 0.1 降至 1e-3~1e-2；`state cost` 降 10–20%。
- 个别新 run：`pi_sum` 偏高且 `ot_loss` 回升，建议减小 `weight_label` 与 `lambda_ot` 以稳 BC→OT 迁移。
