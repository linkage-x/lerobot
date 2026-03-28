# FR3 Mask2EE Training And Inference Contract (2026-03-26)

## Scope

This note records the contract for the FR3 `mask2ee` ACT+DAS training mode.

Goal:

- train a policy that does not rely on live EE pose in `observation.state`
- force the policy to predict action primarily from image observations and tactile inputs
- guarantee that inference uses the same masked proprio contract as training

This note is intentionally about the `mask2ee` contract only. For the general FR3 real-inference entrypoint, use `docs/fr3_act_infer_real_minimal.md`.

## Current Implementation Scope

Current status:

- implemented for `ACT` only
- not yet implemented as a policy-agnostic mechanism across the broader policy set

Important clarification:

- the current repository does not provide a universal `mask2ee` switch that automatically applies to every policy consuming `observation.state`
- only ACT has the config field, index resolution, and runtime masking path today

This note therefore describes the current ACT contract plus the future direction for generalization.

## Canonical Training Config

Current config:

- `src/lerobot/configs/franka_research3_mask2ee_act_das.yaml`

Key flag:

- `policy.mask_ee_pose_in_state: true`

Current intent:

- keep tactile enabled
- keep gripper in `observation.state`
- zero only the EE pose dimensions `x y z qx qy qz qw`

## What Gets Masked

For the current FR3 ee2ee dataset layout, the canonical `observation.state` order is:

- `x`
- `y`
- `z`
- `qx`
- `qy`
- `qz`
- `qw`
- `gripper`

`mask2ee` zeros:

- `x y z qx qy qz qw`

`mask2ee` keeps:

- `gripper`

This is deliberate. The mode is `mask2ee`, not `mask_all_proprio`.

## How The Mask Is Resolved

The ACT config resolves the indices to mask from dataset metadata, not from hard-coded FR3 assumptions.

Resolution path:

1. training loads dataset metadata
2. `make_policy(...)` injects `observation.state.names` into the ACT config
3. ACT resolves which indices correspond to `x y z qx qy qz qw`
4. ACT stores those indices in the policy instance

Current code points:

- state-name injection: `src/lerobot/policies/factory.py`
- index resolution: `src/lerobot/policies/act/configuration_act.py`
- runtime masking: `src/lerobot/policies/act/modeling_act.py`

This means the behavior is tied to the dataset contract saved with the checkpoint, not to ad-hoc CLI flags at inference time.

## Future TODO: Generalize State Masking Beyond ACT

This section is a documented TODO only. It is not implemented yet.

### Goal

Extract `mask2ee` into a reusable state-masking mechanism that can be shared across most policies that consume `observation.state`.

Target outcome:

- one common way to describe which state dimensions must be masked
- consistent train/infer behavior across policies
- dataset-metadata-driven index resolution, instead of duplicating FR3-specific assumptions in each policy

### Why This Is Worth Doing

Today only ACT supports `mask2ee`.

That creates three problems:

- feature parity is policy-specific
- future policy experiments can silently drift away from the intended proprio contract
- the same EE-pose-masking logic would need to be re-implemented and re-tested per policy

### Likely Candidate Policies

These are the obvious next policies to evaluate because they directly consume `OBS_STATE` in their model paths:

- Diffusion
- VQBeT
- TDMPC
- PI0 / PI05 / PI0Fast
- WallX
- XVLA where proprio is enabled
- other future policies that directly encode `observation.state`

This does not mean every policy should necessarily expose the feature by default. It means the masking primitive should be reusable.

### Recommended Design Direction

Prefer a shared mechanism with these properties:

1. configuration-level intent is policy-independent
2. state-name resolution from dataset metadata is shared
3. actual masking happens immediately before the model consumes `observation.state`
4. train and inference paths must reuse the exact same masking logic
5. the mechanism should fail loudly if required state names cannot be resolved

Recommended scope split:

- shared utility for resolving named state dimensions to indices
- shared helper for producing a masked copy of `batch[OBS_STATE]`
- thin policy-specific integration that opts into the common helper

Avoid:

- hard-coding FR3 index positions inside multiple policy implementations
- implementing train-only masking and forgetting inference
- introducing a generic config field without end-to-end tests for each adopted policy

### Required Test Coverage When This TODO Is Implemented

The future implementation should not be considered complete without all of the following:

- config parsing test for every policy config that exposes the feature
- unit test for grouped and flat `observation.state.names` metadata resolution
- unit test that unresolved EE names raise a loud error
- training-path test proving `forward(...)` masks the intended state dimensions
- inference-path test proving action selection uses the same masking behavior
- factory-level test proving dataset metadata injection works end-to-end
- regression test proving the original input batch is not mutated in place

### Acceptance Criteria For The Future Work

The future work should be considered done only when:

- at least one non-ACT policy supports the shared mechanism
- train and inference both honor the same checkpoint-level contract
- the documentation here can remove the phrase `implemented for ACT only`
- the test suite covers both shared utilities and policy integrations

## Why Inference Also Masks EE Pose

This is the part that must stay true over time.

Checkpoint save/load path:

1. training saves the policy config and train config under the checkpoint's `pretrained_model/`
2. FR3 real inference loads `train_config.json` back from that checkpoint
3. the runtime reconstructs the policy from the checkpoint config
4. the same ACT policy code path runs at inference
5. `predict_action_chunk(...)` applies the same state masking used by `forward(...)`

Consequence:

- a checkpoint trained with `mask_ee_pose_in_state=true` will also mask EE pose at inference
- a checkpoint trained without that flag will not mask EE pose at inference

There is no separate `--mask2ee` inference flag because the behavior must come from the checkpoint contract.

## Operational Rules

Use these rules to avoid silent train/infer mismatch.

- train with `src/lerobot/configs/franka_research3_mask2ee_act_das.yaml`
- infer from the checkpoint produced by that run
- do not assume an older ee2ee checkpoint will magically become `mask2ee`
- do not resume a formal `mask2ee` run from a checkpoint that was trained without masking
- if dataset `observation.state.names` changes, verify that the names still resolve `x y z qx qy qz qw`

## Recommended Commands

Training template:

```bash
RUN_NAME=fr3_mask2ee_act_das_$(date +%Y%m%d_%H%M%S)

sudo env HOME=/home/hph docker compose --profile train -f docker/docker-compose.yml run --rm \
  -e WANDB_API_KEY=$WANDB_API_KEY \
  lerobot-train-fr3-act-das \
  lerobot-train \
  --config_path=src/lerobot/configs/franka_research3_mask2ee_act_das.yaml \
  --dataset.root=outputs/datasets/lerobotv3_0310_100ep_aligned_ts \
  --output_dir=outputs/train/${RUN_NAME} \
  --job_name=${RUN_NAME} \
  --policy.device=cuda \
  --policy.push_to_hub=false \
  --batch_size=8 \
  --num_workers=12 \
  --steps=100000 \
  --log_freq=200 \
  --eval_freq=0 \
  --save_checkpoint=true \
  --save_freq=20000 \
  --tolerance_s=1e-3 \
  --wandb.enable=true \
  --wandb.project=fr3-mask2ee-act-das
```

Preview inference template:

```bash
CKPT=outputs/train/${RUN_NAME}/checkpoints/020000

sudo env HOME=/home/hph python tools/fr3/fr3_act_infer_real.py \
  --checkpoint=${CKPT} \
  --camera-config=tools/fr3/fr3_act_infer_camera_config.yaml \
  --dataset-root=outputs/datasets/lerobotv3_0310_100ep_aligned_ts \
  --preview \
  --max-steps=5 \
  --first-frame-max-pos-delta-mm=20 \
  --first-frame-max-rot-delta-deg=8 \
  --max-step-pos-delta-mm=3 \
  --max-step-rot-delta-deg=2
```

Real-robot inference template:

```bash
CKPT=outputs/train/${RUN_NAME}/checkpoints/020000

sudo env HOME=/home/hph python tools/fr3/fr3_act_infer_real.py \
  --checkpoint=${CKPT} \
  --camera-config=tools/fr3/fr3_act_infer_camera_config.yaml \
  --dataset-root=outputs/datasets/lerobotv3_0310_100ep_aligned_ts \
  --max-steps=30 \
  --first-frame-max-pos-delta-mm=20 \
  --first-frame-max-rot-delta-deg=8 \
  --max-step-pos-delta-mm=3 \
  --max-step-rot-delta-deg=2
```

## Recommended Intermediate Checkpoint Evaluation Cadence

This section records the current recommendation for the ongoing FR3 `mask2ee` training runs.

Current assumption:

- training uses `save_freq=20000`
- training uses `eval_freq=0`
- this means there is no automatic in-training evaluation signal beyond the training logs

### Practical Gate Order

Use this order for intermediate checkpoints:

1. wait for `020000`
2. run offline dataset-fed evaluation first
3. only if offline behavior is directionally sane, run `preview`
4. only if preview is stable, run a short real-robot rollout
5. only after a short real rollout looks controlled, consider a longer rollout

### Recommended First Checkpoint

For the current run shape, the first meaningful checkpoint to inspect is:

- `020000`

Reason:

- earlier in-run states are not saved anyway with `save_freq=20000`
- the first saved checkpoint is the earliest low-cost chance to detect broken observation/action contracts
- for `mask2ee`, early optimization is expected to be noisier than a version that directly exposes live EE pose

### Recommended Sequence For `020000`

1. offline check:

```bash
sudo env HOME=/home/hph python tools/fr3/fr3_check_policy_dataset_frame.py \
  --checkpoint=outputs/train/${RUN_NAME}/checkpoints/020000 \
  --dataset-root=outputs/datasets/lerobotv3_0310_100ep_aligned_ts \
  --episodes=0,13 \
  --frame-indices=0,1,2,4,8,16,24,32,40
```

2. preview only if the offline results are at least directionally plausible:

```bash
sudo env HOME=/home/hph python tools/fr3/fr3_act_infer_real.py \
  --checkpoint=outputs/train/${RUN_NAME}/checkpoints/020000 \
  --camera-config=tools/fr3/fr3_act_infer_camera_config.yaml \
  --dataset-root=outputs/datasets/lerobotv3_0310_100ep_aligned_ts \
  --preview \
  --max-steps=5 \
  --first-frame-max-pos-delta-mm=20 \
  --first-frame-max-rot-delta-deg=8 \
  --max-step-pos-delta-mm=3 \
  --max-step-rot-delta-deg=2
```

3. short real rollout only if preview is stable:

```bash
sudo env HOME=/home/hph python tools/fr3/fr3_act_infer_real.py \
  --checkpoint=outputs/train/${RUN_NAME}/checkpoints/020000 \
  --camera-config=tools/fr3/fr3_act_infer_camera_config.yaml \
  --dataset-root=outputs/datasets/lerobotv3_0310_100ep_aligned_ts \
  --max-steps=10 \
  --first-frame-max-pos-delta-mm=20 \
  --first-frame-max-rot-delta-deg=8 \
  --max-step-pos-delta-mm=3 \
  --max-step-rot-delta-deg=2
```

4. if the short real rollout is controlled, then consider increasing to:

- `--max-steps=30`

### Stop Conditions

If `020000` fails at a gate, stop and wait for the next checkpoint instead of forcing a real rollout.

Current stop rules:

- offline bad: do not run preview or real rollout yet
- preview unstable: do not run real rollout yet
- short real rollout unstable: do not extend rollout length yet

In practice, the next checkpoint to compare against is usually:

- `040000`

### Why This Gate Order Matters

This gate order is intentional:

- offline checks are cheap and isolate model quality from robot execution risk
- preview checks runtime contract and safety gates without sending actions
- short real rollouts reduce the chance of conflating immature policy behavior with hardware-side execution issues

Given `eval_freq=0`, the offline check is the main quality gate for deciding whether an intermediate checkpoint is mature enough to justify preview or hardware time.

## Failure Modes To Recheck

If `mask2ee` appears ineffective, check these first:

- the checkpoint was actually trained from the `mask2ee` config
- the runtime is pointed at the new checkpoint, not an older ee2ee checkpoint
- `observation.state.names` still expose `x y z qx qy qz qw gripper`
- the policy input still includes the tactile/image keys you expect

## Summary

The durable rule is:

- `mask2ee` is a checkpoint-level contract, not an inference-time toggle

If the checkpoint was trained with `mask_ee_pose_in_state=true`, both training and inference will zero EE pose before the ACT model sees `observation.state`.

As of now, that statement is guaranteed for ACT only. Generalizing the same contract to more policies is a planned follow-up, not a completed feature.
