## Experiments 2025-12-25 07:15:51

### Run 4ojzwqrs (diffusion)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/4ojzwqrs
- train/ot_cost/action_lbl: n=36, last=0.00362119, mean=0.00347044, std=0.00067, cv=0.192
- eval/offline_eval/avg_loss: n=36, last=0.037424, min=0.0268465@9000, delta=0.0106, slope_recent=5.17e-07
- train/ot_loss: last=0.002505, mean=0.0026239, min=0.00175181
- train/loss: last=0.0126353, mean=0.0288855, min=0.0126353
- train/lr: last=8.0919e-05, mean=8.33666e-05, min=1.901e-05
- train/ot_pi_diag: last=0.962953, mean=0.961218, min=0.922905

### Run cvc8migk (diffusion)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/cvc8migk
- train/ot_cost/action_lbl: n=36, last=0.00362119, mean=0.00347044, std=0.00067, cv=0.192
- eval/offline_eval/avg_loss: n=36, last=0.0335109, min=0.0287931@17000, delta=0.00472, slope_recent=1.07e-07
- train/ot_loss: last=0.00237956, mean=0.00237626, min=0.0016903
- train/loss: last=0.0129133, mean=0.0228727, min=0.0129133
- train/lr: last=0.000219802, mean=0.000270699, min=0.000219802
- train/ot_pi_diag: last=0.936793, mean=0.936981, min=0.870206

### Run pj1f7xs4 (diffusion)
- Link: https://wandb.ai/kjust-pinduoduo/lerobot/runs/pj1f7xs4
- train/ot_cost/action_lbl: n=36, last=0.0120706, mean=0.0115681, std=0.0022, cv=0.192
- eval/offline_eval/avg_loss: n=36, last=0.0357696, min=0.0262587@9000, delta=0.00951, slope_recent=4.82e-07
- train/ot_loss: last=0.00603046, mean=0.00614208, min=0.00464341
- train/loss: last=0.0140571, mean=0.0255956, min=0.0140571
- train/lr: last=0.000187862, mean=0.000186825, min=7.604e-05
- train/ot_pi_diag: last=0.815479, mean=0.814661, min=0.643709

#### Interpretation
- 4ojzwqrs: lowest OT magnitude (mean~0.00346), highest ot_pi_diag, and best eval (stable, lowest).
- cvc8migk: OT similar to 4ojzwqrs but higher LR; eval plateau slightly higher; consider lowering LR or longer warmup.
- pj1f7xs4: strongest OT (mean~0.0115), lowest ot_pi_diag; eval shows mild rebound then stabilizes; reduce action_lbl or raise reg.
