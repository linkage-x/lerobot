# Act/OT Comparison Report

Generated: 2025-12-25T02:34:43.202013Z

## Runs
- zecwqro4 · act · https://wandb.ai/kjust-pinduoduo/lerobot/runs/zecwqro4
- 4gb7zc4r · act · https://wandb.ai/kjust-pinduoduo/lerobot/runs/4gb7zc4r
- 80h5t9k8 · act · https://wandb.ai/kjust-pinduoduo/lerobot/runs/80h5t9k8

## Metrics Summary (key curves)

| run_id | ot_cost/action_lbl last_mean | last_std | slope_last | jitter | ot_loss last_mean | eval_l1 last_mean |
|---|---:|---:|---:|---:|---:|---:|
| zecwqro4 | 0.01996 | 0.005859 | -4.435e-06 | 0.4299 | 0.01659 | 0.4441 |
| 4gb7zc4r | 0.01892 | 0.005671 | 9.063e-05 | 0.4556 | 0.01363 | 0.3944 |
| 80h5t9k8 | 0.01996 | 0.005859 | -4.435e-06 | 0.4299 | 0.01563 | 0.4457 |

### Interpretation
- last_mean/last_std: 用尾段窗口（<=100，或 10%）的均值/方差衡量稳定性与收敛水平；
- slope_last: 尾段趋势斜率，负值向下收敛；
- jitter: 一阶差分（按序列中位数缩放）的标准差，越大表示抖动越强；

## Next-Step Experiments

详见以下三个配置建议（A/B/C），关注：A 稳定性、B OT 参数、C action_lbl 头部质量。

### A) stable_ot
```yaml
experiment: stable_ot
train:
  optimizer: adamw
  lr: 2.0e-4
  weight_decay: 0.05
  scheduler: cosine
  warmup_ratio: 0.10
  batch_size: 256
  grad_accum: 4
  grad_clip_norm: 1.0
  ema: true
  ema_decay: 0.999
  seed: 42
data:
  seq_len: 128
  window_stride: 2
loss:
  weights:
    ot: 0.5
    action_lbl: 1.0
  ot:
    cost: l2
    sinkhorn_epsilon: 0.10
    sinkhorn_iters: 200
    weight_warmup_ratio: 0.30
logging:
  smooth_window: 200
eval:
  interval_steps: 2000
```

### B) ot_tune
```yaml
experiment: ot_tune
train:
  optimizer: adamw
  lr: 3.0e-4
  weight_decay: 0.02
  scheduler: cosine
  warmup_ratio: 0.05
  batch_size: 128
  grad_accum: 2
  grad_clip_norm: 1.0
data:
  seq_len: 160
  window_stride: 2
loss:
  weights:
    ot: 0.7
    action_lbl: 1.0
  ot:
    cost: l2_squared
    sinkhorn_epsilon: 0.07
    sinkhorn_iters: 160
    compute_every_n_steps: 2
    detach_cost_grad: true
eval:
  interval_steps: 2000
```

### C) action_lbl_focus
```yaml
experiment: action_lbl_focus
train:
  optimizer: adamw
  lr: 2.5e-4
  weight_decay: 0.05
  scheduler: cosine
  warmup_ratio: 0.10
  batch_size: 192
  grad_accum: 2
  grad_clip_norm: 1.0
loss:
  weights:
    ot: 0.2
    action_lbl: 2.0
  action_lbl:
    label_smoothing: 0.05
    focal_gamma: 1.5
    class_balance: true
  ot:
    cost: l2
    sinkhorn_epsilon: 0.05
    sinkhorn_iters: 120
eval:
  interval_steps: 1500
```

## Jitter Diagnosis & Fixes
- 小批/短序列导致方差大；增大 batch 或 grad_accum，并适度加长 seq_len。
- Sinkhorn 超参偏“激进”（epsilon 过小、迭代不足）会放大噪声；建议 epsilon 0.07–0.12、iters 160–200。
- 每步都算 OT 会把匹配随机性直接注入梯度；可 compute_every_n_steps=2–4，或复用匹配结果。
- 优化器侧：降低 lr、增加 warmup、启用 grad_clip、EMA(0.999–0.9995)。
- 指标侧：在 W&B 增大平滑窗口；评估用 episode 聚合，训练看尾段均值/方差与斜率。

## Regression Check: Latest 3 vs Baselines

### Latest 3
- z851dpqu · act · https://wandb.ai/kjust-pinduoduo/lerobot/runs/z851dpqu
  - ot_cost/action_lbl last_mean=- last_std=- slope=- jitter=-
  - ot_loss last_mean=-; eval_l1 last_mean=-; train_loss last_mean=-
- 64fubjbc · act · https://wandb.ai/kjust-pinduoduo/lerobot/runs/64fubjbc
  - ot_cost/action_lbl last_mean=- last_std=- slope=- jitter=-
  - ot_loss last_mean=-; eval_l1 last_mean=-; train_loss last_mean=-
- qrsts3lb · act · https://wandb.ai/kjust-pinduoduo/lerobot/runs/qrsts3lb
  - ot_cost/action_lbl last_mean=- last_std=- slope=- jitter=-
  - ot_loss last_mean=-; eval_l1 last_mean=-; train_loss last_mean=-

### Baselines
- 4gb7zc4r · act · https://wandb.ai/kjust-pinduoduo/lerobot/runs/4gb7zc4r
  - ot_cost/action_lbl last_mean=- last_std=- slope=- jitter=-
  - ot_loss last_mean=-; eval_l1 last_mean=-; train_loss last_mean=-
- 80h5t9k8 · act · https://wandb.ai/kjust-pinduoduo/lerobot/runs/80h5t9k8
  - ot_cost/action_lbl last_mean=- last_std=- slope=- jitter=-
  - ot_loss last_mean=-; eval_l1 last_mean=-; train_loss last_mean=-


## Postmortem: Regression Root Causes

- z851dpqu · eval_l1=0.6379, ot_action_lbl=0.02185, jitter=0.3865
  - optimizer: lr=0.0002, wd=0.05, clip=1, sched=cosine_decay_with_warmup
  - ot: lambda=0.1, reg=0.1, tau=(0.5,0.5), win=10, action_lbl_w=0.02
- 64fubjbc · eval_l1=0.9249, ot_action_lbl=0.01508, jitter=0.3784
  - optimizer: lr=0.0003, wd=0.02, clip=1, sched=cosine_decay_with_warmup
  - ot: lambda=0.2, reg=0.07, tau=(0.5,0.5), win=10, action_lbl_w=0.02
- qrsts3lb · eval_l1=0.918, ot_action_lbl=0.0377, jitter=0.3784
  - optimizer: lr=0.00025, wd=0.05, clip=1, sched=cosine_decay_with_warmup
  - ot: lambda=0.05, reg=0.05, tau=(0.5,0.5), win=10, action_lbl_w=0.05

- baseline 4gb7zc4r · eval_l1=0.3944, ot_action_lbl=0.01892, jitter=0.4556
  - optimizer: lr=1e-05, wd=0.0001, clip=10, sched=None
  - ot: lambda=0.1, reg=0.1, tau=(0.5,0.5), win=10, action_lbl_w=0.02
- baseline 80h5t9k8 · eval_l1=0.4457, ot_action_lbl=0.01996, jitter=0.4299
  - optimizer: lr=1e-05, wd=0.0001, clip=10, sched=None
  - ot: lambda=0.1, reg=0.2, tau=(2,2), win=20, action_lbl_w=0.02

### Key Differences Observed
- New runs increased lr by 20–30x (1e-5 → 2e-4/3e-4) and wd by ~500x (1e-4 → 0.05/0.02); clip norm from 10 → 1; added cosine scheduler.
- OT hyperparams shifted modestly (lambda_ot={0.05,0.1,0.2}, reg={0.05,0.07,0.1}); not enough alone to explain the large eval L1 regression.
- Jitter did not explode; in fact slightly lower, but eval L1 worse → underfitting/optimization mismatch likely due to optimizer hyperparams, not OT noise.

### Recommendations
1) Revert optimizer to baseline ACT preset: lr=1e-5, wd=1e-4, grad_clip_norm=10, no scheduler; keep batch_size and steps.
2) Keep OT settings near baseline 4gb7zc4r (lambda_ot=0.1, reg=0.1, tau=(0.5,0.5), win=10); only sweep one knob at a time:
   - Sweep A: reg ∈ {0.08, 0.1, 0.12} (holding lambda_ot=0.1) – expect stability vs. sharpness trade-off.
   - Sweep B: lambda_ot ∈ {0.08, 0.1, 0.12} (holding reg=0.1) – balance BC/OT contributions.
   - Sweep C: action_lbl.weight_label ∈ {0.02, 0.04} – moderate increase only; keep weight_embed=0.
3) If you want window_size=20 (like 80h5t9k8), change only that in an isolated run; don't mix with lr/scheduler changes.
4) Maintain use_policy_training_preset=true unless there's a strong reason; if you override, do not move lr beyond 3e-5 without evidence.



---

## ACT-OT Report Update (2025-12-29 01:55)
- Note: Compare to baseline

### act (8dfuhytp)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/8dfuhytp
- Config:
  - window_size=10, reg=0.12, tau=(0.5,0.5), heuristic=True
  - weight_embed=0.3333, weight_label=0, lambda_ot=0.1
- Metrics:
  - train/loss: first=10.25, last=0.09822, best=0.09775@14600
  - train/l1_loss: first=0.5189, last=0.08222, best=0.04157@12400
  - eval/avg_l1: first=0.4598, last=0.413, best=0.3813@4000
  - ot_loss: first=0.07093, last=0.02124, best=0.001865@7800
  - ot_pi_sum: first=0.01748, last=0.9808, best=0.01748@100
  - ot_pi_diag: first=0.01748, last=0.9808, best=0.01748@100
  - ot_cost(state): first=nan, last=nan, best=nan@None

### act (avj1u2ok)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/avj1u2ok
- Config:
  - window_size=10, reg=0.1, tau=(0.5,0.5), heuristic=True
  - weight_embed=0.3333, weight_label=0, lambda_ot=0.08
- Metrics:
  - train/loss: first=10.25, last=0.07377, best=0.07377@20000
  - train/l1_loss: first=0.5185, last=0.05834, best=0.03905@12400
  - eval/avg_l1: first=0.453, last=0.3982, best=0.3849@5000
  - ot_loss: first=0.03514, last=0.01061, best=8.743e-24@300
  - ot_pi_sum: first=0.00735, last=0.9903, best=1.638e-24@300
  - ot_pi_diag: first=0.00735, last=0.9903, best=1.638e-24@300
  - ot_cost(state): first=nan, last=nan, best=nan@None

#### Baseline Comparison: 8dfuhytp & avj1u2ok
- eval/offline_eval/avg_l1 (last):
  - 8dfuhytp: 0.4130 vs baseline 4gb7zc4r 0.3944 (+4.7%), vs baseline 80h5t9k8 0.4457 (-7.3%)
  - avj1u2ok: 0.3982 vs baseline 4gb7zc4r 0.3944 (+1.0%), vs baseline 80h5t9k8 0.4457 (-10.7%)
- best (min over run):
  - 8dfuhytp best 0.3813 (better than 4gb7zc4r by -3.3%) at step ~4000
  - avj1u2ok best 0.3849 (better than 4gb7zc4r by -2.4%) at step ~5000
- Takeaways:
  - Both runs roughly match baseline; avj1u2ok is within ~1% of 4gb7zc4r at end, and both beat 80h5t9k8.
  - Best-of-run dips below baseline suggest mild late-phase drift; consider early stopping or gentler decay.
  - Config shows weight_label=0 with weight_embed=0.3333; try adding small label weight (e.g., 0.02) to align with baseline.
  - Keep lambda_ot in 0.08–0.1 and reg around 0.1; avoid changing multiple knobs simultaneously.
