# ACT-OT 三实验对比与后续计划

对比的 W&B 实验：
- rk711khx: https://wandb.ai/kjust-pinduoduo/lerobot/runs/rk711khx
- gpki6xab: https://wandb.ai/kjust-pinduoduo/lerobot/runs/gpki6xab
- 3tef3yy2: https://wandb.ai/kjust-pinduoduo/lerobot/runs/3tef3yy2

注：以下数值来自本地汇总文件（compare_runs_3.json / compare_runs_new.json），键与量纲与线上一致，仅截取关键指标用于决策。

---
## 关键指标对比（摘要）

1) rk711khx（记作 r1）
- eval/offline_eval/avg_loss: first=0.6585, last=0.3429, best=0.3429@1250（稳定下降）
- train/ot_loss: first=0.1960 → last=0.0512，best≈2.0e-4@250（早期快速变小）
- train/ot_pi_sum: 0.1668 → 0.0931，best≈1.5e-5@250（早期高度稀疏，后期回升）
- train/ot_cost/observation.state: 1327.7 → 1093.1，best=761.1@1000（成本显著降低）

2) gpki6xab（记作 r2）
- eval/offline_eval/avg_loss: first=0.6582, last=0.3460, best=0.3451@1100（与 r1 接近）
- train/ot_loss: first≈2.38e-10 → last=0.0307，best≈2.38e-10@50（起始近 0，后期上升）
- train/ot_pi_sum: first≈1.38e-11 → last=0.0720（起始近 0，后期上升）
- train/ot_cost/observation.state: 1371.8 → 1090.1，best=815.8@1200

3) 3tef3yy2（记作 r3）
- eval/offline_eval/avg_loss: first=0.6584, last=0.4015（最弱）
- train/ot_loss: first≈2.38e-10 → last=0.00157（极小）
- train/ot_pi_sum: first≈1.39e-11 → last≈3.58e-4（极稀疏）
- train/ot_cost/observation.state: 1371.7 → 1195.7，best=888.5@550（成本降幅最小）

解读要点
- r1: 评估最稳；OT 早期极稀疏（pi_sum→~1e-5），随后回到 0.09 左右，说明 reg/tau 较小会在初期“塌缩”为强对角/弱耦合，后续才恢复。
- r2: 起步即接近“零耦合”（ot/π≈0），随后 π_sum 上升并保持低于 r1，评估与 r1 接近。推测窗口/对角先验更强或 reg 更大；但早期极小 π_sum 仍是风险。
- r3: π_sum 始终极小（~3.6e-4），ot_loss 极低但 eval 最差，符合“过度平滑/质量流失”现象：不平衡 OT 吸收了过多质量，OT 分支几乎不工作，难以提供有用对齐信号。

与 OT_LOSS_DESIGN.md 对齐
- 建议对 reg、tau_src/tau_tgt 网格化调参；监控 `ot_pi_sum/diag` 与 `ot_cost/<term>` 的量级与趋势，避免“π 过小/退化”。
- 当 π_sum 极小且 eval 变差时：提高 tau（放宽边缘）、适度提高 reg（更平滑）、降低 label 权重或分阶段增大 λ_ot。

---
## 下一步配置（3 个）

目标：抑制“早期零耦合/过稀疏”与“π_sum 过小”两类现象，同时验证窗口与权重对评估稳定性的影响。

1) 更平滑、轻放宽（面向 r1 的稳态对照）
- 文件: `src/lerobot/scripts/train_config/act_fr3_ot_next_reg02_tau1.json`
- 变更: reg=0.20, tau_src=tau_tgt=1.0, λ_ot=0.10（窗口启用, heuristic=true, action_lbl.weight_label=0.02）
- 预期: 初期 π 不再塌缩到 ~0，维持可解释耦合，同时保持与 r1 相近的训练/评估趋势。

2) 适中放宽 + 略增 λ_ot + 窗口加权（面向 r2 的对角先验对照）
- 文件: `src/lerobot/scripts/train_config/act_fr3_ot_next_reg015_tau2_lambda015_weighted_window.json`
- 变更: reg=0.15, tau=2.0, λ_ot=0.15, sharpness=1.0（启用窗口权重），其余同基线
- 预期: π_sum 在 0.07~0.15 区间更平稳；对角偏好更强但不至塌缩，评估与 r1 接近或略优。

3) 强放宽 + 降低 action 标签权重 + 关窗（面向 r3 的“π_sum 极小”修复）
- 文件: `src/lerobot/scripts/train_config/act_fr3_ot_next_reg02_tau5_lambda01_no_window_action005.json`
- 变更: reg=0.20, tau=5.0, λ_ot=0.10, no_window=true, action_lbl.weight_label=0.005
- 预期: 允许跨更远时间步的匹配并显著提高 π_sum；减小 label 竞争，避免把 BC 推偏；若 π_sum ≥ 0.2 且 eval 稳定，可在下一轮把 λ_ot 提至 0.15–0.20。

---
## 验证要点与判定标准
- 监控：`train/ot_ot_pi_sum`、`train/ot_ot_pi_diag`、`train/ot_ot_cost/observation.state` 与 `eval/offline_eval/avg_loss`。
- 判定成功：
  - r1/r2 路线：避免初期 π_sum → ~0 的塌缩；eval 持平或更低；
  - r3 路线：π_sum 明显上升（≥1e-2 → 1e-1 量级），同时 eval 不反弹。

---
## 运行与记录
- 训练示例：
  - `python -m lerobot.scripts.train --config_path src/lerobot/scripts/train_config/act_fr3_ot_next_reg02_tau1.json`
  - `python -m lerobot.scripts.train --config_path src/lerobot/scripts/train_config/act_fr3_ot_next_reg015_tau2_lambda015_weighted_window.json`
  - `python -m lerobot.scripts.train --config_path src/lerobot/scripts/train_config/act_fr3_ot_next_reg02_tau5_lambda01_no_window_action005.json`
- 建议：保持相同 seed/评估数据；开启 W&B；关键超参随 run 一并入库以便复盘。



---

## ACT-OT Report Update (2025-12-29 09:02)
- Note: Tuning batch phgb325y vlly2hzl kmaf5wox

### act (phgb325y)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/phgb325y
- Config:
  - window_size=10, reg=0.2, tau=(1,1), heuristic=True
  - weight_embed=0.3333, weight_label=0, lambda_ot=0.1
- Metrics:
  - train/loss: first=16.15, last=0.6064, best=0.6056@3550
  - train/l1_loss: first=0.6076, last=0.1594, best=0.1107@2100
  - eval/avg_l1: first=0.8106, last=0.2683, best=0.2524@3350
  - ot_loss: first=0.5037, last=0.6893, best=0.3513@1200
  - ot_pi_sum: first=0.554, last=0.4268, best=0.4268@3600
  - ot_pi_diag: first=0.554, last=0.4268, best=0.4268@3600
  - ot_cost(state): first=nan, last=nan, best=nan@None

### act (vlly2hzl)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/vlly2hzl
- Config:
  - window_size=10, reg=0.15, tau=(2,2), heuristic=True
  - weight_embed=0.3333, weight_label=0, lambda_ot=0.15
- Metrics:
  - train/loss: first=16.25, last=0.6659, best=0.6659@3500
  - train/l1_loss: first=0.605, last=0.158, best=0.1101@2100
  - eval/avg_l1: first=0.8122, last=0.2566, best=0.2566@3500
  - ot_loss: first=0.6313, last=0.556, best=0.2167@400
  - ot_pi_sum: first=0.8266, last=0.8478, best=0.5564@350
  - ot_pi_diag: first=0.8266, last=0.8478, best=0.5564@350
  - ot_cost(state): first=nan, last=nan, best=nan@None

### act (kmaf5wox)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/kmaf5wox
- Config:
  - window_size=10, reg=0.2, tau=(5,5), heuristic=True
  - weight_embed=0.3333, weight_label=0, lambda_ot=0.1
- Metrics:
  - train/loss: first=16.25, last=0.6374, best=0.6374@3450
  - train/l1_loss: first=0.5985, last=0.1518, best=0.1072@2100
  - eval/avg_l1: first=0.8088, last=0.2833, best=0.2697@3150
  - ot_loss: first=0.6705, last=0.4704, best=0.1858@750
  - ot_pi_sum: first=0.9317, last=0.9525, best=0.9143@1700
  - ot_pi_diag: first=0.9317, last=0.9525, best=0.9143@1700
  - ot_cost(state): first=nan, last=nan, best=nan@None



---

## ACT-OT Report Update (2025-12-29 09:23)
- Note: Tuning batch hpdecad1 hbp2xock vyg9tuy1

### act (hpdecad1)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/hpdecad1
- Config:
  - window_size=10, reg=0.2, tau=(1,1), heuristic=True
  - weight_embed=0.3333, weight_label=0, lambda_ot=0.12
- Metrics:
  - train/loss: first=16.16, last=0.9327, best=0.9327@2550
  - train/l1_loss: first=0.6093, last=0.2082, best=0.1104@2100
  - eval/avg_l1: first=0.8117, last=0.2901, best=0.2813@2450
  - ot_loss: first=0.6259, last=0.415, best=0.2841@400
  - ot_pi_sum: first=0.5926, last=0.7464, best=0.2268@350
  - ot_pi_diag: first=0.5926, last=0.7464, best=0.2268@350
  - ot_cost(state): first=nan, last=nan, best=nan@None

---

## Root-Cause Notes (fgc0bcxr, gv4yyjzk, 843np9lc 共性问题)

现象（用户反馈）
- `ot_pi_sum / ot_pi_diag` 没有“趋于 1”；`ot_loss` 与 `action_lbl` 不单调下降。

原因分析（要点）
- 度量未归一：在“不平衡 OT”下，`π` 的总质量不是 1；我们记录的是原值（质量和与对角和），不是“对角占比”。因此“趋于 1”不成立。应改用 `diag_ratio = ot_pi_diag / (ot_pi_sum + 1e-9)` 作为“对角占比”指标，目标是趋近 1；而 `ot_pi_sum` 本身的绝对值受 `tau/reg/成本量纲` 影响较大。
- 成本矩阵 M 动态变化：图像项使用 learnable embedding，训练中表征分布变化导致每步的 M 改变；再叠加“随机窗口/样本对”，`ot_loss` 与各 term（含 `action_lbl`）会出现非单调波动。这是期望内行为，并不代表退化。
- 不平衡松弛与窗口策略：较大的 `tau`、窗口权重（sharpness）与对角先验会改变质量分布与对角偏好，但不会让 `π_sum→1`。当 `tau` 较大时，更允许“质量缺失/新增”，`π_sum` 通常背离 1。
- 量纲竞争：多 term 线性组合时，若各项的尺度未对齐（如图像 embed 与动作 label），`ot_loss` 的主导项会随训练阶段切换，引起曲线起伏。

证据/佐证
- 最近数批次里 `ot_pi_diag ≈ ot_pi_sum`（见多条 run 的同值），说明对角质量占比已接近 1（即 diag_ratio≈1），但绝对值<1 是不平衡设置与成本尺度的结果。
- `ot_cost(state)` 为 NaN 是因为当前未启用该 term（非异常）；建议直接记录 `ot_cost/action_lbl` 与各图像项的 `ot_cost/<term>` 以区分分量趋势。

改进与验证建议
- 指标层面：
  - 新增日志 `train/ot_diag_ratio = ot_pi_diag / (ot_pi_sum + 1e-9)`；并对每个 term 记录 `ot_cost/<term>`，避免只看总 `ot_loss`。
  - 增加一个“固定评估对齐集”（固定窗口与样本对）用于周期性 OT 评估，以降低曲线噪声、观察真实收敛趋势。
- 训练/超参：
  - 若希望 `π_sum` 更接近 1：减小 `tau_src/tau_tgt`（如 1.0→0.5），增大/保持中等 `reg`（0.15–0.2）以稳定求解。
  - 若对角占比不足（diag_ratio<0.8）：减小窗口（如 10→6）或开启/增大 `sharpness`，并确保 `base_index_{src,tgt}` 对齐。
  - 量纲对齐：对各 term 的 `weight_*` 做温和网格（图像 embed 总和≈动作 label 的量级），必要时做 per-term 标准化（以初始化分布的分位数/方差缩放）。

### act (hbp2xock)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/hbp2xock
- Config:
  - window_size=10, reg=0.2, tau=(1,1), heuristic=True
  - weight_embed=0.3333, weight_label=0, lambda_ot=0.18
- Metrics:
  - train/loss: first=16.2, last=0.9659, best=0.9659@2500
  - train/l1_loss: first=0.6132, last=0.2294, best=0.1074@2100
  - eval/avg_l1: first=0.8218, last=0.3099, best=0.2846@2450
  - ot_loss: first=0.3495, last=0.3438, best=0.1118@400
  - ot_pi_sum: first=0.819, last=0.8226, best=0.5727@350
  - ot_pi_diag: first=0.819, last=0.8226, best=0.5727@350
  - ot_cost(state): first=nan, last=nan, best=nan@None

### act (vyg9tuy1)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/vyg9tuy1
- Config:
  - window_size=10, reg=0.2, tau=(2,2), heuristic=True
  - weight_embed=0.3333, weight_label=0, lambda_ot=0.15
- Metrics:
  - train/loss: first=16.23, last=0.9981, best=0.9981@2400
  - train/l1_loss: first=0.6021, last=0.1926, best=0.111@2100
  - eval/avg_l1: first=0.8054, last=0.2824, best=0.2824@2400
  - ot_loss: first=0.3921, last=0.2847, best=0.1173@400
  - ot_pi_sum: first=0.9, last=0.9289, best=0.7395@350
  - ot_pi_diag: first=0.9, last=0.9289, best=0.7395@350
  - ot_cost(state): first=nan, last=nan, best=nan@None



---

## ACT-OT Report Update (2025-12-29 09:51)
- Note: Tuning batch fgc0bcxr gv4yyjzk 843np9lc

### act (fgc0bcxr)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/fgc0bcxr
- Config:
  - window_size=10, reg=0.2, tau=(1,1), heuristic=True
  - weight_embed=0.3333, weight_label=0, lambda_ot=0.12
- Metrics:
  - train/loss: first=16.17, last=0.606, best=0.606@3650
  - train/l1_loss: first=0.6099, last=0.19, best=0.1093@2100
  - eval/avg_l1: first=0.8108, last=0.2961, best=0.2582@3350
  - ot_loss: first=0.4902, last=0.6518, best=0.3496@300
  - ot_pi_sum: first=0.5649, last=0.5973, best=0.4271@3600
  - ot_pi_diag: first=0.5649, last=0.5973, best=0.4271@3600
  - ot_cost(state): first=nan, last=nan, best=nan@None

### act (gv4yyjzk)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/gv4yyjzk
- Config:
  - window_size=10, reg=0.2, tau=(1,1), heuristic=True
  - weight_embed=0.3333, weight_label=0, lambda_ot=0.16
- Metrics:
  - train/loss: first=16.19, last=0.6108, best=0.6108@3550
  - train/l1_loss: first=0.6121, last=0.1856, best=0.1093@2100
  - eval/avg_l1: first=0.8197, last=0.2561, best=0.2468@3500
  - ot_loss: first=0.4341, last=0.3111, best=0.2154@1800
  - ot_pi_sum: first=0.7007, last=0.8421, best=0.6464@1750
  - ot_pi_diag: first=0.7007, last=0.8421, best=0.6464@1750
  - ot_cost(state): first=nan, last=nan, best=nan@None

### act (843np9lc)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/843np9lc
- Config:
  - window_size=10, reg=0.2, tau=(2,2), heuristic=True
  - weight_embed=0.3333, weight_label=0, lambda_ot=0.15
- Metrics:
  - train/loss: first=16.28, last=0.7436, best=0.7436@3500
  - train/l1_loss: first=0.6009, last=0.1605, best=0.1095@2100
  - eval/avg_l1: first=0.8039, last=0.2569, best=0.2569@3500
  - ot_loss: first=0.8776, last=1.238, best=0.7203@300
  - ot_pi_sum: first=0.5452, last=0.3901, best=0.358@2650
  - ot_pi_diag: first=0.5452, last=0.3901, best=0.358@2650
  - ot_cost(state): first=nan, last=nan, best=nan@None


### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/fromwandb_iztuqxzc.json` | out: `outputs/train/loop_1767063739/00_fromwandb_iztuqxzc_6c6w`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/fromwandb_xx5st9ij.json` | out: `outputs/train/loop_1767063739/01_fromwandb_xx5st9ij_6c6w`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/fromwandb_ithx0mon.json` | out: `outputs/train/loop_1767063739/02_fromwandb_ithx0mon_6c6w`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline.json` | out: `outputs/train/loop_1767063739/03_act_fr3_ot_99_20_baseline_6c6w`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline.json` | out: `outputs/train/loop_1767065585/00_act_fr3_ot_99_20_baseline_tfsz`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline.json` | out: `outputs/train/loop_1767065775/00_act_fr3_ot_99_20_baseline_7lxw`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_cqm6fm.json` | out: `outputs/train/loop_1767065775/01_act_fr3_ot_99_20_baseline_cqm6fm_7lxw`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_f41by4.json` | out: `outputs/train/loop_1767065775/02_act_fr3_ot_99_20_baseline_f41by4_7lxw`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_16lch8.json` | out: `outputs/train/loop_1767065775/03_act_fr3_ot_99_20_baseline_16lch8_7lxw`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline.json` | out: `outputs/train/loop_1767065917/00_act_fr3_ot_99_20_baseline_jfjg`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_cqm6fm.json` | out: `outputs/train/loop_1767065917/01_act_fr3_ot_99_20_baseline_cqm6fm_jfjg`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_f41by4.json` | out: `outputs/train/loop_1767065917/02_act_fr3_ot_99_20_baseline_f41by4_jfjg`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_16lch8.json` | out: `outputs/train/loop_1767065917/03_act_fr3_ot_99_20_baseline_16lch8_jfjg`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline.json` | out: `outputs/train/loop_1767066105/00_act_fr3_ot_99_20_baseline_w3o9`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/mhw95uug
  - eval_l1: 0.2268 | train_l1: 0.2082 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.1764
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_cqm6fm.json` | out: `outputs/train/loop_1767066105/01_act_fr3_ot_99_20_baseline_cqm6fm_w3o9`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/82gv3pc4
  - eval_l1: 0.2195 | train_l1: 0.2097 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.0024
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_f41by4.json` | out: `outputs/train/loop_1767066105/02_act_fr3_ot_99_20_baseline_f41by4_w3o9`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/b27xe91z
  - eval_l1: 0.2201 | train_l1: 0.2060 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.0185
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_16lch8.json` | out: `outputs/train/loop_1767066105/03_act_fr3_ot_99_20_baseline_16lch8_w3o9`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/sylcwvnb
  - eval_l1: 0.2201 | train_l1: 0.2076 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 48.8491

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_0j57ea.json` | out: `outputs/train/loop_1767070484/00_act_fr3_ot_99_20_baseline_0j57ea_rlua`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_cqm6fm_idj1ci.json` | out: `outputs/train/loop_1767070484/01_act_fr3_ot_99_20_baseline_cqm6fm_idj1ci_rlua`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_f41by4_v9z4nx.json` | out: `outputs/train/loop_1767070484/02_act_fr3_ot_99_20_baseline_f41by4_v9z4nx_rlua`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_16lch8_qllxw3.json` | out: `outputs/train/loop_1767070484/03_act_fr3_ot_99_20_baseline_16lch8_qllxw3_rlua`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_0j57ea.json` | out: `outputs/train/loop_1767070517/00_act_fr3_ot_99_20_baseline_0j57ea_8i14`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_cqm6fm_idj1ci.json` | out: `outputs/train/loop_1767070517/01_act_fr3_ot_99_20_baseline_cqm6fm_idj1ci_8i14`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_f41by4_v9z4nx.json` | out: `outputs/train/loop_1767070517/02_act_fr3_ot_99_20_baseline_f41by4_v9z4nx_8i14`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_16lch8_qllxw3.json` | out: `outputs/train/loop_1767070517/03_act_fr3_ot_99_20_baseline_16lch8_qllxw3_8i14`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_1imqb7.json` | out: `outputs/train/loop_1767071162/00_act_fr3_ot_99_20_baseline_1imqb7_vqru`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/7u4jqe53
  - eval_l1: 0.2239 | train_l1: 0.2068 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.1608
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_0j57ea_fix.json` | out: `outputs/train/loop_1767071162/01_act_fr3_ot_99_20_baseline_0j57ea_fix_vqru`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/e95ey90b
  - eval_l1: 0.2213 | train_l1: 0.2071 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 48.9187
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_cqm6fm_idj1ci_fix.json` | out: `outputs/train/loop_1767071162/02_act_fr3_ot_99_20_baseline_cqm6fm_idj1ci_fix_vqru`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/36hmhp0y
  - eval_l1: 0.2176 | train_l1: 0.2054 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.1771
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_f41by4_v9z4nx_fix.json` | out: `outputs/train/loop_1767071162/03_act_fr3_ot_99_20_baseline_f41by4_v9z4nx_fix_vqru`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/66vlqac1
  - eval_l1: 0.2199 | train_l1: 0.2067 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 48.8905
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_16lch8_qllxw3_fix.json` | out: `outputs/train/loop_1767071162/04_act_fr3_ot_99_20_baseline_16lch8_qllxw3_fix_vqru`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/5a6xm58w
  - eval_l1: 0.2192 | train_l1: 0.2065 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.0392

### Auto Loop Round 2

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_1imqb7_hnu25.json` | out: `outputs/train/loop_1767071162/r1_act_fr3_ot_99_20_baseline_1imqb7_hnu25_ojnh`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_0j57ea_fix_h3xgd.json` | out: `outputs/train/loop_1767071162/r1_act_fr3_ot_99_20_baseline_0j57ea_fix_h3xgd_n4ss`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_cqm6fm_idj1ci_fix_yhhv6.json` | out: `outputs/train/loop_1767071162/r1_act_fr3_ot_99_20_baseline_cqm6fm_idj1ci_fix_yhhv6_pw8x`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_f41by4_v9z4nx_fix_425jc.json` | out: `outputs/train/loop_1767071162/r1_act_fr3_ot_99_20_baseline_f41by4_v9z4nx_fix_425jc_85xu`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_16lch8_qllxw3_fix_pi4p2.json` | out: `outputs/train/loop_1767071162/r1_act_fr3_ot_99_20_baseline_16lch8_qllxw3_fix_pi4p2_bpza`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Stable Round (post-fix)

- cfg: `outputs/train/loop_1767071162/00_act_fr3_ot_99_20_baseline_1imqb7_vqru/checkpoints/002000/pretrained_model/train_config.json` | out: `outputs/train/loop_1767071162/00_act_fr3_ot_99_20_baseline_1imqb7_vqru`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/36hmhp0y
  - eval_l1: 0.2239 | train_l1: 0.2068 | ot_pi_sum: 0.0000 | ot_pi_diag: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.1608
- cfg: `outputs/train/loop_1767071162/01_act_fr3_ot_99_20_baseline_0j57ea_fix_vqru/checkpoints/002000/pretrained_model/train_config.json` | out: `outputs/train/loop_1767071162/01_act_fr3_ot_99_20_baseline_0j57ea_fix_vqru`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/36hmhp0y
  - eval_l1: 0.2213 | train_l1: 0.2071 | ot_pi_sum: 0.0000 | ot_pi_diag: 0.0000 | ot_loss: 0.0000 | grad_norm: 48.9187
- cfg: `outputs/train/loop_1767071162/02_act_fr3_ot_99_20_baseline_cqm6fm_idj1ci_fix_vqru/checkpoints/002000/pretrained_model/train_config.json` | out: `outputs/train/loop_1767071162/02_act_fr3_ot_99_20_baseline_cqm6fm_idj1ci_fix_vqru`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/36hmhp0y
  - eval_l1: 0.2176 | train_l1: 0.2054 | ot_pi_sum: 0.0000 | ot_pi_diag: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.1771
- cfg: `outputs/train/loop_1767071162/03_act_fr3_ot_99_20_baseline_f41by4_v9z4nx_fix_vqru/checkpoints/002000/pretrained_model/train_config.json` | out: `outputs/train/loop_1767071162/03_act_fr3_ot_99_20_baseline_f41by4_v9z4nx_fix_vqru`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/36hmhp0y
  - eval_l1: 0.2199 | train_l1: 0.2067 | ot_pi_sum: 0.0000 | ot_pi_diag: 0.0000 | ot_loss: 0.0000 | grad_norm: 48.8905
- cfg: `outputs/train/loop_1767071162/04_act_fr3_ot_99_20_baseline_16lch8_qllxw3_fix_vqru/checkpoints/002000/pretrained_model/train_config.json` | out: `outputs/train/loop_1767071162/04_act_fr3_ot_99_20_baseline_16lch8_qllxw3_fix_vqru`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/36hmhp0y
  - eval_l1: 0.2192 | train_l1: 0.2065 | ot_pi_sum: 0.0000 | ot_pi_diag: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.0392

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_fuznb2.json` | out: `outputs/train/loop_1767076416/00_act_fr3_ot_99_20_baseline_fuznb2_891p`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_d65wiz.json` | out: `outputs/train/loop_1767076416/01_act_fr3_ot_99_20_baseline_d65wiz_891p`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_67qpwg.json` | out: `outputs/train/loop_1767076416/02_act_fr3_ot_99_20_baseline_67qpwg_891p`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_djxzba.json` | out: `outputs/train/loop_1767076416/03_act_fr3_ot_99_20_baseline_djxzba_891p`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_2uzuuv.json` | out: `outputs/train/loop_1767076936/00_act_fr3_ot_99_20_baseline_2uzuuv_48yd`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_3hori4.json` | out: `outputs/train/loop_1767076936/01_act_fr3_ot_99_20_baseline_3hori4_48yd`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_8geccf.json` | out: `outputs/train/loop_1767076936/02_act_fr3_ot_99_20_baseline_8geccf_48yd`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_h7osts.json` | out: `outputs/train/loop_1767076936/03_act_fr3_ot_99_20_baseline_h7osts_48yd`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_2uzuuv.json` | out: `outputs/train/loop_1767077070/00_act_fr3_ot_99_20_baseline_2uzuuv_57ap`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_3hori4.json` | out: `outputs/train/loop_1767077070/01_act_fr3_ot_99_20_baseline_3hori4_57ap`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_8geccf.json` | out: `outputs/train/loop_1767077070/02_act_fr3_ot_99_20_baseline_8geccf_57ap`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_h7osts.json` | out: `outputs/train/loop_1767077070/03_act_fr3_ot_99_20_baseline_h7osts_57ap`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_2uzuuv.json` | out: `outputs/train/loop_1767077495/00_act_fr3_ot_99_20_baseline_2uzuuv_wp2g`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/5j0d6owm
  - eval_l1: 0.2169 | train_l1: 0.2079 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.0545
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_3hori4.json` | out: `outputs/train/loop_1767077495/01_act_fr3_ot_99_20_baseline_3hori4_wp2g`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/nrg29hlq
  - eval_l1: 0.2195 | train_l1: 0.2078 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.0779
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_8geccf.json` | out: `outputs/train/loop_1767077495/02_act_fr3_ot_99_20_baseline_8geccf_wp2g`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/gwwdw7vx
  - eval_l1: 0.2135 | train_l1: 0.2090 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.1843
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_h7osts.json` | out: `outputs/train/loop_1767077495/03_act_fr3_ot_99_20_baseline_h7osts_wp2g`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/6gym78gx
  - eval_l1: 0.2257 | train_l1: 0.2077 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.0004

### Auto Loop Round 2

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_2uzuuv_fgnu7.json` | out: `outputs/train/loop_1767077495/r1_act_fr3_ot_99_20_baseline_2uzuuv_fgnu7_mhur`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_3hori4_mu09e.json` | out: `outputs/train/loop_1767077495/r1_act_fr3_ot_99_20_baseline_3hori4_mu09e_28pj`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_8geccf_dkxiw.json` | out: `outputs/train/loop_1767077495/r1_act_fr3_ot_99_20_baseline_8geccf_dkxiw_iech`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_baseline_h7osts_nuter.json` | out: `outputs/train/loop_1767077495/r1_act_fr3_ot_99_20_baseline_h7osts_nuter_xohr`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -




### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015.json` | out: `outputs/train/loop_1767082351/00_act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015_me3a`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/6ump4y6d
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003.json` | out: `outputs/train/loop_1767082351/01_act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003_me3a`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/sjmlql8x
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20.json` | out: `outputs/train/loop_1767082351/02_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_me3a`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/cxc7ozpk
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005.json` | out: `outputs/train/loop_1767082351/03_act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005_me3a`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/ioqo5a4l
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 2

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015_pqs89.json` | out: `outputs/train/loop_1767082351/r1_act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015_pqs89_of1m`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003_dz4ux.json` | out: `outputs/train/loop_1767082351/r1_act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003_dz4ux_4w3q`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_029s2.json` | out: `outputs/train/loop_1767082351/r1_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_029s2_isg7`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005_4v9bt.json` | out: `outputs/train/loop_1767082351/r1_act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005_4v9bt_f98z`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003.json` | out: `outputs/train/loop_1767084579/00_act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003_9hqk`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/a807vuc4
  - eval_l1: 0.2118 | train_l1: 0.1706 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 46.7389
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20.json` | out: `outputs/train/loop_1767084579/01_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_9hqk`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/4ddvio77
  - eval_l1: 0.1917 | train_l1: 0.2751 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 47.8953
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015_pqs89.json` | out: `outputs/train/loop_1767084579/02_act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015_pqs89_9hqk`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/takgnbuh
  - eval_l1: 0.1938 | train_l1: 0.2619 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 47.8069
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005_4v9bt.json` | out: `outputs/train/loop_1767084579/03_act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005_4v9bt_9hqk`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/vx86o6ya
  - eval_l1: 0.2233 | train_l1: 0.2576 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 47.7216

### Auto Loop Round 2

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003_vages.json` | out: `outputs/train/loop_1767084579/r1_act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003_vages_nox8`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_xx3zy.json` | out: `outputs/train/loop_1767084579/r1_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_xx3zy_z2fr`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015_pqs89_ib7zp.json` | out: `outputs/train/loop_1767084579/r1_act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015_pqs89_ib7zp_s0cf`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005_4v9bt_6zfng.json` | out: `outputs/train/loop_1767084579/r1_act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005_4v9bt_6zfng_zbtv`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA.json` | out: `outputs/train/loop_1767087812/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_7e71`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/bcoclyah
  - eval_l1: 0.1827 | train_l1: 0.1500 | ot_pi_sum: 0.1273 | ot_loss: 0.2148 | grad_norm: 48.8630
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005_4v9bt_calB.json` | out: `outputs/train/loop_1767087812/01_act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005_4v9bt_calB_7e71`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/wgl64e6g
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003_calC.json` | out: `outputs/train/loop_1767087812/02_act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003_calC_7e71`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/25uhjkan
  - eval_l1: 0.2178 | train_l1: 0.1629 | ot_pi_sum: 0.1241 | ot_loss: 0.0125 | grad_norm: 48.5131
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015_pqs89_calD.json` | out: `outputs/train/loop_1767087812/03_act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015_pqs89_calD_7e71`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/zdquhflx
  - eval_l1: 0.1840 | train_l1: 0.1520 | ot_pi_sum: 0.2243 | ot_loss: 0.0049 | grad_norm: 48.7280

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA.json` | out: `outputs/train/loop_1767089892/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_n8dz`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/vip558th
  - eval_l1: 0.1913 | train_l1: 0.1516 | ot_pi_sum: 0.1193 | ot_loss: 0.2390 | grad_norm: 48.9440
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005_4v9bt_calB.json` | out: `outputs/train/loop_1767089892/01_act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005_4v9bt_calB_n8dz`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/7uhp4ces
  - eval_l1: 0.2173 | train_l1: 0.1927 | ot_pi_sum: 0.0006 | ot_loss: 0.0076 | grad_norm: 49.1913
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003_calC.json` | out: `outputs/train/loop_1767089892/02_act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003_calC_n8dz`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/wqbm1ks5
  - eval_l1: 0.2279 | train_l1: 0.1612 | ot_pi_sum: 0.0804 | ot_loss: 0.4158 | grad_norm: 48.6837
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015_pqs89_calD.json` | out: `outputs/train/loop_1767089892/03_act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015_pqs89_calD_n8dz`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/6tnrcv0m
  - eval_l1: 0.1950 | train_l1: 0.1478 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 49.0529

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2a.json` | out: `outputs/train/loop_1767091222/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2a_0a1s`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/pszrye5a
  - eval_l1: 0.1920 | train_l1: 0.1466 | ot_pi_sum: 0.1786 | ot_loss: 0.4879 | grad_norm: 48.9706
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2b.json` | out: `outputs/train/loop_1767091222/01_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2b_0a1s`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/swhra59x
  - eval_l1: 0.1926 | train_l1: 0.1551 | ot_pi_sum: 0.1944 | ot_loss: 0.2957 | grad_norm: 48.8626
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2c.json` | out: `outputs/train/loop_1767091222/02_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2c_0a1s`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/jwqprvka
  - eval_l1: 0.1929 | train_l1: 0.1493 | ot_pi_sum: 0.1154 | ot_loss: 0.2288 | grad_norm: 48.9778
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d.json` | out: `outputs/train/loop_1767091222/03_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_0a1s`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/4autifec
  - eval_l1: 0.1902 | train_l1: 0.1491 | ot_pi_sum: 0.1190 | ot_loss: 0.2391 | grad_norm: 48.9278

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2a.json` | out: `outputs/train/loop_1767143634/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2a_dpex`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/gisgwy42
  - eval_l1: 0.1939 | train_l1: 0.1456 | ot_pi_sum: 0.1786 | ot_loss: 0.4879 | grad_norm: 48.9582
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2b.json` | out: `outputs/train/loop_1767143634/01_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2b_dpex`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/kn8tu6zb
  - eval_l1: 0.1921 | train_l1: 0.1529 | ot_pi_sum: 0.1944 | ot_loss: 0.2958 | grad_norm: 49.0005
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2c.json` | out: `outputs/train/loop_1767143634/02_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2c_dpex`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/djlnu4ex
  - eval_l1: 0.1906 | train_l1: 0.1492 | ot_pi_sum: 0.1153 | ot_loss: 0.2288 | grad_norm: 48.9404
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d.json` | out: `outputs/train/loop_1767143634/03_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_dpex`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/5n1p82ig
  - eval_l1: 0.1913 | train_l1: 0.1486 | ot_pi_sum: 0.1192 | ot_loss: 0.2390 | grad_norm: 49.1412

### Auto Loop Round 2

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2a_de701.json` | out: `outputs/train/loop_1767143634/r1_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2a_de701_r1as`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2b_k1ikm.json` | out: `outputs/train/loop_1767143634/r1_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2b_k1ikm_d4g0`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2c_8vhql.json` | out: `outputs/train/loop_1767143634/r1_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2c_8vhql_yoqi`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_0qz20.json` | out: `outputs/train/loop_1767143634/r1_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_0qz20_ou2t`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_v1_reg055.json` | out: `outputs/train/loop_1767146115/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_v1_reg055_wdd3`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/62t0ogj7
  - eval_l1: 0.1909 | train_l1: 0.1494 | ot_pi_sum: 0.1210 | ot_loss: 0.2443 | grad_norm: 48.8667
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_v2_tau15.json` | out: `outputs/train/loop_1767146115/01_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_v2_tau15_wdd3`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/v1ps13jw
  - eval_l1: 0.1933 | train_l1: 0.1496 | ot_pi_sum: 0.1524 | ot_loss: 0.3522 | grad_norm: 49.0288
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_v3_wl0007.json` | out: `outputs/train/loop_1767146115/02_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_v3_wl0007_wdd3`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/yh4ruune
  - eval_l1: 0.1947 | train_l1: 0.1492 | ot_pi_sum: 0.1541 | ot_loss: 0.2530 | grad_norm: 49.0015
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2c_v4_reg044.json` | out: `outputs/train/loop_1767146115/03_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2c_v4_reg044_wdd3`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/cnohyhu3
  - eval_l1: 0.1911 | train_l1: 0.1495 | ot_pi_sum: 0.1169 | ot_loss: 0.2328 | grad_norm: 48.9396

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_long_s42.json` | out: `outputs/train/loop_1767147680/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_long_s42_brop`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/e8gk2wg9
  - eval_l1: 0.1810 | train_l1: 0.1337 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 15.7986
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_long_s87.json` | out: `outputs/train/loop_1767147680/01_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_long_s87_brop`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/iv64vs09
  - eval_l1: 0.1895 | train_l1: 0.0907 | ot_pi_sum: 0.1478 | ot_loss: 0.1942 | grad_norm: 17.8097
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_long_s123.json` | out: `outputs/train/loop_1767147680/02_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_long_s123_brop`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/dex6o8jz
  - eval_l1: 0.1493 | train_l1: 0.0861 | ot_pi_sum: 0.0194 | ot_loss: 0.1544 | grad_norm: 16.0658
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_v1_reg055.json` | out: `outputs/train/loop_1767147680/03_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_v1_reg055_brop`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/62t0ogj7
  - eval_l1: 0.1748 | train_l1: 0.1362 | ot_pi_sum: 0.0000 | ot_loss: 0.0000 | grad_norm: 15.8245

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p1_bs075_win6.json` | out: `outputs/train/loop_1767151485/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p1_bs075_win6_rbqh`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/bcwqp50y
  - eval_l1: 0.1922 | train_l1: 0.1519 | ot_pi_sum: 0.0404 | ot_loss: 0.1441 | grad_norm: 49.1080
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_lambda012.json` | out: `outputs/train/loop_1767151485/01_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_lambda012_rbqh`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/7xj1ky65
  - eval_l1: 0.1920 | train_l1: 0.1492 | ot_pi_sum: 0.1193 | ot_loss: 0.2390 | grad_norm: 48.9155
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p3_reg044.json` | out: `outputs/train/loop_1767151485/02_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p3_reg044_rbqh`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/3c75nuo9
  - eval_l1: 0.1896 | train_l1: 0.1473 | ot_pi_sum: 0.1167 | ot_loss: 0.2329 | grad_norm: 48.8808
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p4_window0.json` | out: `outputs/train/loop_1767151485/03_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p4_window0_rbqh`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/tyfdgab4
  - eval_l1: 0.1940 | train_l1: 0.1462 | ot_pi_sum: 0.2479 | ot_loss: 0.0064 | grad_norm: 48.9799

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_test.json` | out: `outputs/train/loop_1767167871/00_act_fr3_ot_99_20_test_kdg6`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_test.json` | out: `outputs/train/loop_1767169919/00_act_fr3_ot_99_20_test_ru1g`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_lambda012_long200k.json` | out: `outputs/train/loop_1767165048/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_lambda012_long200k_75mv`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/ziriuya7
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config` | out: `outputs/train/loop_1767171451/00_train_config_io5t`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_lambda012_long200k.json` | out: `outputs/train/loop_1767171493/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_lambda012_long200k_5mb8`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/ziriuya7
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_lambda012_long200k.json` | out: `outputs/train/loop_1767171557/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_lambda012_long200k_79fo`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_bs075_win6.json` | out: `outputs/train/loop_1767171571/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_bs075_win6_644y`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_test.json` | out: `outputs/train/loop_1767171605/00_act_fr3_ot_99_20_test_9rde`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_lambda012_long200k.json` | out: `outputs/train/loop_1767172318/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_lambda012_long200k_uuhe`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_bs075_win6.json` | out: `outputs/train/loop_1767172331/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_bs075_win6_so7o`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015.json` | out: `outputs/train/loop_1767172361/00_act_fr3_ot_99_20_balanced_w8_sharp075_tau1_reg015_lambda014_action015_lepu`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/6ump4y6d
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003.json` | out: `outputs/train/loop_1767172361/01_act_fr3_ot_99_20_align_w0_tau05_reg015_lambda018_action003_lepu`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/a807vuc4
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_next_nowindow_tau2_reg015_lambda015_action001.json` | out: `outputs/train/loop_1767172361/02_act_fr3_ot_99_20_next_nowindow_tau2_reg015_lambda015_action001_lepu`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_next_w6_sharp1_lambda016_action001.json` | out: `outputs/train/loop_1767172361/03_act_fr3_ot_99_20_next_w6_sharp1_lambda016_action001_lepu`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005.json` | out: `outputs/train/loop_1767172361/04_act_fr3_ot_99_20_light_w12_tau025_lambda008_action0005_lepu`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/ioqo5a4l
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20.json` | out: `outputs/train/loop_1767172361/05_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_lepu`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/4ddvio77
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_best200k.json` | out: `outputs/train/loop_1767157286/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_best200k_hxea`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/bdvbsdkg
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_test.json` | out: `outputs/train/loop_1767172854/00_act_fr3_ot_99_20_test_k930`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_bs075_win6.json` | out: `outputs/train/loop_1767172854/01_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_bs075_win6_k930`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_reg055.json` | out: `outputs/train/loop_1767172854/02_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_reg055_k930`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/027sv85w
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_wl0007.json` | out: `outputs/train/loop_1767172854/03_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_wl0007_k930`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -

### Auto Loop Round 1

- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_bs075_win6.json` | out: `outputs/train/loop_1767173687/00_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_bs075_win6_d7wk`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_reg055.json` | out: `outputs/train/loop_1767173687/01_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_reg055_d7wk`
  - W&B: https://wandb.ai/kjust-pinduoduo/lerobot/runs/027sv85w
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_topk30.json` | out: `outputs/train/loop_1767173687/02_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_topk30_d7wk`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -
- cfg: `src/lerobot/scripts/train_config/act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_wl0007.json` | out: `outputs/train/loop_1767173687/03_act_fr3_ot_99_20_w8_tau1_reg02_lambda012_action002_sharp05_topk20_calA_calA2d_p2_wl0007_d7wk`
  - eval_l1: - | train_l1: - | ot_pi_sum: - | ot_loss: - | grad_norm: -