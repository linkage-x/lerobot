DP+OT Experiment Evaluation (run: p80qgmx9)

Overview
- Policy: Diffusion (n_obs_steps=2, n_action_steps=8, horizon=16)
- Dataset: left_fr3_ip_dst_q2cq (bs=16, steps=20k planned)
- Optim: policy presets, lr=1e-5 (Adam), cosine sched (warmup=500) [lr observed constant]
- OT setup (ot-sim2real aligned):
  - features:
    - images embed: use_learned_embed=true, weight_embed=1.0
    - action label: dim_slice=[0,8], weight_label=0.05
  - reg=0.02, tau_src=0.01, tau_tgt=0.01, heuristic=true
  - lambda_ot=0.1, batch_ratio=0.5, window_size=20

Curves (summary)
- train/loss: 15.12 → 0.52 (−96.5%), best@2600
- train/l1_loss: 0.52 → 0.11 (−79.4%)
- train/grad_norm: 249.7 → 27.4 (−89.0%)
- eval/offline_eval/avg_loss: 0.66 → 0.32 (−51.2%), best 0.31
- train/ot_loss (scaled contribution): ~0.004 at last; ot_pi_sum/diag ~0.067
- Throughput: update_s ~0.185 → 0.107 s; dataloading_s ~0.225 → 0.167 s
- LR log: constant at 1e-5 (likely scheduler not varying the base lr in logs)

Quick Assessment
- Convergence: Strong early improvement (≤3k steps) with diminishing returns; offline L1 ~0.32 indicates room to improve.
- Stability: Grad norm decreased smoothly; no spikes. OT loss stays small and stable with unbalanced Sinkhorn.
- OT signal: pi_sum/diag are low-moderate; could strengthen OT coupling slightly if not hurting BC.
- LR scheduling: Logged lr is flat at 1e-5; cosine may not be reflected in logs, or scheduler is ineffective at current settings.

Bottlenecks / Risks
- Underpowered LR: 1e-5 may be conservative; diffusion policy typically benefits from 1e-4 to 3e-4 with cosine warmup.
- Eval frequency: eval_freq=50 is heavy; slows training with minimal insight early on.
- OT weight balance: action label weight=0.05 may be strong for early stages; can suppress BC if increased further.

Iteration Plan

Iter 1 — Optim + Logging Hygiene (low risk)
- Training schedule
  - policy.optimizer_lr: 1e-4 (from 1e-5)
  - policy.scheduler_name: cosine, scheduler_warmup_steps: 1000
  - steps: 50_000 (extend training horizon)
  - eval_freq: 1000 (from 50), log_freq: 100, save_freq: 10_000
- OT config (keep mild)
  - lambda_ot: 0.1
  - action weight_label: 0.02 (from 0.05) to reduce label dominance
  - reg: 0.02, tau: 0.01, window_size: 20 (unchanged)
- Expected: faster learning, better stability; monitor offline_eval/avg_loss and ot_pi_*.

Iter 2 — Strengthen OT if coupling is weak (pi_diag < 0.10 after 10k steps)
- Increase lambda_ot → 0.15–0.2
- Sharpen plan: reg → 0.01 (more peaked), if pi_sum drops too much (<0.05) then increase tau to 0.02 to retain mass
- Keep action weight_label at 0.02; images embed stays at 1.0
- Expected: better cross-domain alignment; watch for BC loss regressions.

Iter 3 — Longer Training + Regularization
- Steps: 100_000
- Optimizer alternative (if LR still flat): use_policy_training_preset=false
  - optimizer: AdamW lr=3e-4, weight_decay=1e-4, betas=(0.9,0.999)
  - scheduler: diffuser cosine, warmup=1000
- Optional: Slightly larger batch (if GPU allows) to stabilize OT estimates
- Expected: further reduction in offline L1; ensure OT terms don’t overfit.

Verification & Monitors
- Core: train/loss, train/l1_loss, eval/offline_eval/avg_loss
- OT: train/ot_loss, ot_pi_sum, ot_pi_diag, per-term costs
- LR: ensure train/lr varies post-warmup; if it stays constant, switch to explicit AdamW + diffuser scheduler
- Throughput: update_s, dataloading_s — verify eval_freq reduction yields higher training throughput

Rollback Triggers
- If offline_eval/avg_loss worsens >10% after OT strengthening, revert lambda_ot to 0.1 and reg to 0.02
- If LR change causes instability (grad_norm spikes or loss oscillation), drop lr to 5e-5 and retry

Next Actions (proposed)
- Apply Iter 1 changes to config and launch a 50k-steps run
- Post 10k steps, compare:
  - offline_eval/avg_loss trend
  - ot_pi_diag (target ≥0.1–0.2 without harming BC)
- Decide on Iter 2 adjustments based on above

Appendix — Current Key Config (for traceability)
- batch_size=16, steps=20k, use_policy_training_preset=true
- policy.optimizer_lr=1e-5, scheduler=cosine (warmup=500)
- OT: features = [images embed (1.0), action label (slice 0:8, 0.05)], reg=0.02, tau=0.01, heuristic=true, lambda_ot=0.1


## W&B Comparison (2025-12-24 08:24:12)

- Run A [hinbjgk8] — diffusion | https://wandb.ai/kjust-pinduoduo/lerobot/runs/hinbjgk8
  - train/loss last: None | eval/avg_loss last: None (best: None) | train/l1 last: None
  - grad_norm last: None | ot_pi_sum: None | ot_pi_diag: None
- Run B [44ahhcag] — diffusion | https://wandb.ai/kjust-pinduoduo/lerobot/runs/44ahhcag
  - train/loss last: None | eval/avg_loss last: None (best: None) | train/l1 last: None
  - grad_norm last: None | ot_pi_sum: None | ot_pi_diag: None
- Run C [pysen9m8] — diffusion | https://wandb.ai/kjust-pinduoduo/lerobot/runs/pysen9m8
  - train/loss last: None | eval/avg_loss last: None (best: None) | train/l1 last: None
  - grad_norm last: None | ot_pi_sum: None | ot_pi_diag: None

**Next Config Proposals (Iter4)**
- iter4a (mild OT + higher LR): use AdamW lr=3e-4, wd=1e-4, cosine warmup=1000; lambda_ot=0.1; action weight_label=0.02; reg=0.02; tau=0.01; steps=50k
- iter4b (stronger OT): same as iter4a but lambda_ot=0.2; reg=0.01; tau=0.02; keep action weight_label=0.02
- iter4c (longer run): iter4a with steps=100k; if GPU allows, batch_size +25% to stabilize OT estimates
- Monitors: eval/offline_eval/avg_loss, train/l1; ot_pi_diag target 0.1–0.2; rollback if eval loss worsens >10%

### Added: Concrete Iter4 Configs

- iter4a JSON: `src/lerobot/scripts/train_config/diffusion_fr3_ot_iter4a.json`
  - Optim: AdamW lr=3e-4 wd=1e-4, cosine (warmup=1000); steps=50k; bs=16; use_policy_training_preset=false
  - OT: lambda_ot=0.1; features=[image embed 1.0, action label 0.02]; reg=0.02; tau=0.01; heuristic=true

- iter4b JSON: `src/lerobot/scripts/train_config/diffusion_fr3_ot_iter4b.json`
  - As iter4a but stronger OT: lambda_ot=0.2; reg=0.01; tau=0.02; action label=0.02

- iter4c JSON: `src/lerobot/scripts/train_config/diffusion_fr3_ot_iter4c.json`
  - As iter4a but longer: steps=100k; batch_size=20 (adjust down if OOM)

Run Examples
- `CUDA_VISIBLE_DEVICES=2 accelerate launch --num_processes 1 src/lerobot/scripts/lerobot_train.py --config_path=src/lerobot/scripts/train_config/diffusion_fr3_ot_iter4a.json`
- Replace config path with iter4b/iter4c accordingly.

What To Watch
- Primary: `eval/offline_eval/avg_loss` downtrend (10k-steps checkpoint)
- OT: `train/ot_ot_pi_diag` in 0.1–0.2; if <0.08 consider moving from iter4a→iter4b; if >0.25 ensure BC not regressing
- LR dynamics: `train/lr` after warmup; if flat or unstable, consider halving lr or increasing warmup
## Run pczv9hci Analysis

- URL: https://wandb.ai/kjust-pinduoduo/lerobot/runs/pczv9hci
- Snapshot (steps≈1600):
  - train/loss≈0.037, train/grad_norm≈0.488, train/lr≈3e-4 (cosine)
  - eval/offline_eval/avg_loss≈0.0673 over 20 batches（极好，早期就收敛较快）
  - OT: ot_pi_diag≈0.652, ot_pi_sum≈0.652（强对角对齐），ot_loss≈0.0088
  - OT per-term cost: action_lbl≈1.86e-2，images≈1e-5 量级（img_ee / img_side / img_third_person）

Observations
- 早期指标非常好；学习率和优化器设置工作正常。
- ot_pi_diag 与 pi_sum 接近，说明传输质量强、几乎全在对角，对齐“太强”可能掩盖 BC 信号。
- image embed 成本量级远小于 action label，当前 OT 主要受动作标签主导；可适当降低 label 权重或提升图像嵌入权重以平衡。

Stage Conclusion
- 继续当前优化设置（AdamW 3e-4 + cosine）是合理的。
- 优先测试：弱化标签支配、降低对角偏置，确保 OT 不盖过 BC，同时验证更长训练的收益。

Next Experiments (Top-3 Priority)
1) iter5a（降低标签权重，保持对角先验）
   - AdamW lr=3e-4 wd=1e-4 cosine warmup=1000，steps=50k，bs=16
   - OT: lambda_ot=0.1, weight_embed(image)=1.0, weight_label(action)=0.005, reg=0.02, tau=0.01, heuristic=true
   - 目标：减弱标签主导，保留温和 OT，观察 eval/offline_eval/avg_loss、ot_pi_diag 是否仍然健康（~0.1–0.3）

2) iter5b（去掉对角先验 + 降低 tau）
   - 同 iter5a，但 heuristic=false，tau_src=tau_tgt=0.005
   - 目标：去除对角偏置，验证跨域对齐是否仍有效，避免过拟合严格对齐

3) iter5c（增强图像嵌入权重 + 长训练）
   - AdamW lr=3e-4，cosine warmup=1000，steps=100k，bs=16（资源允许可 20）
   - OT: lambda_ot=0.1, weight_embed(image)=2.0, weight_label(action)=0.01, reg=0.02, tau=0.01, heuristic=true
   - 目标：提升图像特征在 OT 中的权重，结合更长训练观察泛化

Monitors & Rollback
- 监控 eval/offline_eval/avg_loss、train/l1；OT 指标关注 ot_pi_diag（目标 0.1–0.2 区间较稳健）。
- 若 eval/offline_eval/avg_loss 恶化 >10%，或 ot_pi_diag >0.4 持续，回滚到弱 OT（减小 lambda_ot 或增大 reg）。
