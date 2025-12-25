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