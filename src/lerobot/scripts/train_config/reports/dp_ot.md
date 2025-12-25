# DP-OT Training Report

This compares three runs and analyzes two metrics: `train/ot_cost/action_lbl` (jitter) and `eval/offline_eval/avg_loss` (upward trends). It ends with three follow-up experiment configs.

Runs
- ufiskzhp: https://wandb.ai/kjust-pinduoduo/lerobot/runs/ufiskzhp
- 7xv5wewo: https://wandb.ai/kjust-pinduoduo/lerobot/runs/7xv5wewo
- uazctjvr: https://wandb.ai/kjust-pinduoduo/lerobot/runs/uazctjvr

## Data Summary
### Run ufiskzhp
- train/ot_cost/action_lbl: n=142, last=0.008028, mean=0.01106, std=0.002039, cv=0.184, med_abs_rel_change=0
- eval/offline_eval/avg_loss: n=142, last=0.6897, min=0.0343, delta(last-min)=0.6554, slope_recent=-2.16e-06

### Run 7xv5wewo
- train/ot_cost/action_lbl: n=100, last=0.00645, mean=0.005505, std=0.001051, cv=0.191, med_abs_rel_change=0
- eval/offline_eval/avg_loss: n=100, last=0.09293, min=0.02716, delta(last-min)=0.06577, slope_recent=2.645e-06

### Run uazctjvr
- train/ot_cost/action_lbl: n=100, last=0.00645, mean=0.005505, std=0.001051, cv=0.191, med_abs_rel_change=0
- eval/offline_eval/avg_loss: n=100, last=0.09434, min=0.02676, delta(last-min)=0.06757, slope_recent=2.29e-06

## Interpretation
- train/ot_cost/action_lbl: relative jitter (cv) is ~0.18–0.19 across runs; ufiskzhp mean level is ~2x the other two, indicating a stronger or harder OT alignment stage.
- eval/offline_eval/avg_loss: all three runs show an early low then subsequent rise; ufiskzhp rises dramatically (last 0.69 vs min 0.034), while the other two rise modestly (last ~0.093 from ~0.027).
- ufiskzhp recent slope is slightly negative (minor recovery after the spike), the other two are slightly positive (continued slow drift up).

Likely causes
- OT sharpness: low epsilon or few Sinkhorn iters can cause spiky transport plans and noisy gradients, amplifying jitter and hurting generalization.
- Loss weight competition: a high `ot_action_lbl` weight can pull the policy away from the BC prior, causing eval loss to increase after initial gains.
- Optimizer dynamics: LR too high, weak warmup, or missing gradient clipping/EMA can produce instability near plateaus.
- Eval mismatch: evaluating non-EMA weights or using a different batch/domain causes rising eval loss despite training progress.

Recommendations (data hygiene)
- Always evaluate EMA weights; keep eval batch/seed/dataset fixed across runs.
- Log learning rate, OT epsilon/iters, and loss weights alongside curves for post-hoc correlation.

## Next Experiments (3 configs)

1) Stabilize OT and optimizer (reduce sharpness, lower LR)
```yaml
name: dp_ot_stable_v1
optimizer:
  lr: 5e-5
  weight_decay: 0.02
scheduler:
  type: cosine
  warmup_ratio: 0.05
train:
  batch_size: 128           # or keep batch and double grad_accum
  grad_accum_steps: 2
  gradient_clip_norm: 1.0
  ema_decay: 0.999
  eval_use_ema: true
loss:
  weights:
    ot_action_lbl: 0.5      # down-weight OT to reduce competition
ot:
  epsilon: 0.5              # higher entropy -> smoother couplings
  sinkhorn_iters: 60        # more stable convergence
  stop_thresh: 1e-3
```

2) Improve generalization (regularize, early stop), keep OT params
```yaml
name: dp_ot_reg_v1
optimizer:
  lr: 1e-4
  weight_decay: 0.05
scheduler:
  type: cosine
  warmup_steps: 2000
model:
  head_dropout: 0.2
  label_smoothing: 0.05
train:
  gradient_clip_norm: 1.0
  ema_decay: 0.999
  eval_use_ema: true
early_stopping:
  monitor: eval/offline_eval/avg_loss
  mode: min
  patience_eval_intervals: 5
loss:
  weights:
    bc: 1.0
    ot_action_lbl: 0.3
data:
  augment:
    action_noise_std: 0.01
    state_noise_std: 0.01
```

3) Stage-wise schedules (start smooth, then strengthen OT)
```yaml
name: dp_ot_sched_v1
optimizer:
  lr: 1e-4
scheduler:
  type: cosine
  warmup_ratio: 0.1
train:
  grad_accum_steps: 2
  gradient_clip_norm: 1.0
  ema_decay: 0.9995
  eval_use_ema: true
loss:
  weights:
    ot_action_lbl_schedule:
      type: linear
      start_step: 0
      end_ratio_of_steps: 0.1   # ramp in first 10% steps
      start_value: 0.0
      end_value: 1.0
ot:
  epsilon_schedule:              # high -> low entropy
    type: linear
    start_value: 0.5
    end_value: 0.1
    end_ratio_of_steps: 0.3
  sinkhorn_iters: 50
```

Execution notes
- Keep seed/splits constant; evaluate with EMA; snapshot configs to W&B for traceability.
- Success criteria: lower cv of `train/ot_cost/action_lbl` and non-increasing `eval/offline_eval/avg_loss` after the initial minimum.
