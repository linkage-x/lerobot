# FR3 Relative-Trajectory MVP Implementation Checklist (2026-03-31)

## Scope

This note turns the MVP design in
`docs/fr3_relative_trajectory_mvp_20260331.md`
into a concrete first-batch implementation checklist.

This is still an MVP checklist.

It only covers the first cut:

- keep raw dataset storage in absolute `W_s`
- keep `ACT` with `n_obs_steps = 1`
- remove policy dependence on `W_s` by making the action chunk relative to the
  current live observation frame
- reuse `mask2ee` to block absolute EE pose leakage in `observation.state`

It does not cover the future history-based UMI-style extension.

## Current Decisions

These decisions are now fixed for the next baseline round.

- Keep raw dataset storage in absolute SLAM world frame `W_s`.
- Keep the policy contract private to training and inference adapters.
- Use `relative_ee_action=true` so the action chunk target is expressed relative
  to the current observation anchor.
- Use `mask_ee_pose_in_state=true` so absolute EE pose in
  `observation.state` is not exposed to the policy.
- Keep `n_obs_steps=1` for the MVP. Do not add history-relative proprio yet.
- Keep the current early-validation horizon at `chunk_size=50` and
  `n_action_steps=50`.
- Treat `franka_research3_rel2ee_act_das.yaml` as an exploratory config, not a
  clean baseline, because it changes multiple variables at once
  (`no_tactile + rel2ee + qoff + chunk50`).
- Use the batch inspection tool to separate:
  - `relative_action_before_qoff`
  - `processed_action_after_qoff`
- Current inspection result:
  - `relative_action_before_qoff` at late offsets remains geometrically
    reasonable
  - `processed_action_after_qoff` at late offsets is often amplified strongly
    by qoff
- Therefore the next baseline run should keep `rel2ee` but disable
  `action_chunk_quantile_normalization` first, before adding qoff back.

## First-Batch Change Order

Implement in this order.

Do not start from runtime.

The cheapest path is:

1. define the checkpoint contract
2. add training-side target conversion
3. add runtime-side inverse decode
4. add FR3-specific validation hooks

## File-Level Checklist

### 1. Add A New Config Variant

File:

- `src/lerobot/configs/franka_research3_rel2ee_act_das.yaml`

Why first:

- the new config is the contract declaration
- all later code should target this config, not mutate the current `ee2ee`
  configs

First-batch edits:

- copy from `franka_research3_ee2ee_act_das.yaml`
- set `policy.mask_ee_pose_in_state: true`
- keep `chunk_size` and `n_action_steps` conservative for early validation
- do not introduce `n_obs_steps > 1`
- keep tactile and wrist-camera inputs unchanged

Acceptance check:

- config parses
- checkpoint produced from this config is distinguishable from old `ee2ee`

### 2. Add Config Parsing Test

File:

- `tests/configs/test_fr3_train_config.py`

Why now:

- config-level contract should fail loudly before any training run starts

First-batch edits:

- add one parse test for `franka_research3_rel2ee_act_das.yaml`
- assert:
  - `policy.type == "act"`
  - `policy.mask_ee_pose_in_state is True`
  - tactile settings match the intended contract
  - no unsupported `n_obs_steps` behavior is introduced

Acceptance check:

- test passes locally in the existing config suite

### 3. Add A Relative Action Chunk Processor Step

Primary file:

- `src/lerobot/processor/relative_ee_action_processor.py`

Secondary file:

- `src/lerobot/processor/__init__.py`

Why before ACT integration:

- the transform should exist as an isolated reusable primitive first

Processor responsibility:

- input:
  - current observation EE pose in absolute frame
  - future action chunk in absolute frame
- output:
  - future action chunk expressed relative to current observation pose

Recommended first-batch contract:

- observation anchor: current frame `t`
- action target: `T_E(t)^-1 * T_E(t+k)`
- keep gripper as absolute normalized target
- use relative translation + relative rotvec + gripper

First-batch edits:

- implement one processor step for training/preprocessing
- implement the inverse processor step for inference/postprocessing or runtime
  decode, depending on final seam choice
- register the step(s)
- add minimal config serialization support

Acceptance check:

- processor round-trip is testable outside FR3 runtime

### 4. Add Unit Tests For Relative Action Conversion

File:

- `tests/processor/test_relative_ee_action_processor.py`

Why here:

- this is the highest-signal way to de-risk the representation before touching
  training and inference wiring

Required first-batch tests:

- absolute current pose + absolute future target -> expected relative pose
- inverse transform reconstructs the original absolute target
- quaternion sign continuity does not flip valid rotations unpredictably
- gripper remains unchanged
- identity current pose gives expected no-op geometry

Acceptance check:

- tests prove geometric correctness independent of FR3 runtime

### 5. Wire Relative Target Conversion Into ACT Training

Primary file:

- `src/lerobot/policies/act/processor_act.py`

Possible supporting file:

- `src/lerobot/policies/factory.py`

Why this is the first integration point:

- the checkpoint contract must be created during training before inference can
  consume it

First-batch edits:

- detect the new FR3 relative contract from config
- inject the new relative action processor into the ACT preprocessing path
- keep existing normalizer and tactile handling intact
- do not break old `ee2ee` or `mask2ee` checkpoints

Important constraint:

- the relative conversion must happen before the model loss is computed on the
  action chunk

Acceptance check:

- a training batch entering ACT sees relative action chunk targets, not `W_s`
  absolute targets

### 6. Keep Observation-Side Behavior Minimal

Primary file:

- `src/lerobot/policies/act/modeling_act.py`

Supporting file:

- `src/lerobot/policies/act/configuration_act.py`

Why small scope matters:

- MVP should not add new ACT architecture behavior

First-batch edits:

- reuse existing `mask_ee_pose_in_state` path
- do not add history support
- do not change `n_obs_steps` restrictions

Acceptance check:

- the new checkpoint masks EE pose at both train and inference time because the
  config is checkpoint-owned

### 7. Add Runtime Decode From Relative Action To Absolute Robot Command

Primary file:

- `tools/fr3/fr3_act_infer_real_runtime.py`

Possible supporting file:

- `src/lerobot/robots/franka_research3/processor_franka_research3.py`

Why this comes after training wiring:

- runtime should decode a contract that training can already produce

First-batch edits:

- add a decode branch for the new checkpoint contract
- use current live EE pose as the composition anchor
- convert predicted relative action to an absolute base-frame EE command
- keep final robot command format unchanged

Important design rule:

- for the new relative contract, runtime should not require `T_B_Ws` as a
  semantic dependency

Compatibility rule:

- keep the existing `W_s`-based decode path for old checkpoints

Acceptance check:

- preview mode prints sane absolute EE targets when fed a relative-contract
  checkpoint

### 8. Add Runtime Contract Tests

Primary file:

- `tests/scripts/test_fr3_act_infer_real.py`

Possible new file:

- `tests/scripts/test_fr3_rel2ee_infer_runtime.py`

Required first-batch tests:

- checkpoint contract routing chooses the relative decode path when expected
- relative action composed with live anchor pose produces correct absolute
  command
- old `ee2ee` checkpoints still route through the old path
- `mask2ee` behavior is still honored for the new config

Acceptance check:

- relative and absolute checkpoint contracts can coexist safely

## Current Status

As of 2026-03-31, the first-batch contract seam is in place.

Completed:

- `rel2ee` config variant exists and parses
- ACT training preprocessor conditionally converts absolute EE chunks to
  relative EE chunks
- FR3 real runtime conditionally converts relative policy outputs back to
  absolute dataset-frame commands before the existing decode path
- unit tests cover processor geometry, ACT wiring, config parsing, and FR3
  runtime decode routing

Not started:

- a real `rel2ee` training run
- offline batch/distribution sanity checks on real dataset samples
- preview-mode validation with a real checkpoint
- `ee2ee` vs `rel2ee` task-level A/B

## Next Todo List

Implement in this order after the first-batch code changes above.

### P0. Run The First Real `rel2ee` Training

Files / entrypoints:

- `src/lerobot/configs/franka_research3_rel2ee_act_das.yaml`
- `src/lerobot/scripts/lerobot_train.py`

What to do:

- launch a real training run using the `rel2ee` config
- confirm the saved checkpoint carries `relative_ee_action=true`
- archive the exact train config and processor JSON together with the first
  checkpoint

Acceptance check:

- first checkpoint is produced
- saved preprocessor config includes the relative EE action step

### P0. Inspect One Real Training Batch Before Trusting The Run

Files / entrypoints:

- `src/lerobot/scripts/lerobot_train.py`
- policy preprocessor config saved next to the checkpoint

What to check:

- `observation.state` is still masked as intended
- `action` entering the ACT loss is relative, not `W_s`-absolute
- relative quaternion output is continuous and gripper is unchanged

Acceptance check:

- one real batch has been inspected and recorded in notes or logs before
  interpreting training curves

### P1. Offline Sanity Check On Dataset Samples

What to do:

- sample episodes from `outputs/datasets/lerobotv3_0310_100ep_aligned_ts`
- compare stored absolute action chunks with derived relative action chunks
- verify round-trip reconstruction back to absolute poses
- inspect per-dimension ranges for obvious spikes or sign-flip artifacts

Acceptance check:

- no geometric reconstruction errors
- no unexpected distribution pathology in converted relative targets

### P2. Preview-Mode Runtime Validation

Files / entrypoints:

- `tools/fr3/fr3_act_infer_real_runtime.py`

What to do:

- run `--preview` with a real `rel2ee` checkpoint
- inspect step-0 dump bundle if needed
- check first-frame reject and clamp rates

Acceptance check:

- preview prints sane absolute commands after relative decode
- no obvious first-step jump caused by wrong anchor composition

### P3. Task-Level A/B

What to compare:

- old `ee2ee`
- new `rel2ee`

Metrics to record:

- success rate
- first-frame hold/clamp frequency
- step index where divergence begins
- whether back-half drift is reduced

Acceptance check:

- enough evidence to decide whether `rel2ee` is better than `ee2ee` for the
  target task family

### P4. Engineering Follow-Ups That Help But Do Not Block MVP

Suggested items:

- make train/runtime logs print the active checkpoint contract explicitly
- add a small repeatable batch-inspection workflow for `rel2ee`
- evaluate whether generic eval / async inference paths need a future
  observation-aware postprocess contract

### P5. Stage-Two Work, Explicitly Deferred

Do not mix these into MVP validation:

- `n_obs_steps > 1`
- history-relative proprio
- explicit velocity cue design
- generic postprocess redesign that requires observation-aware inverse decode

### 9. Add Training/Inference Contract Note

Primary file:

- `docs/fr3_relative_trajectory_mvp_20260331.md`

New or updated supporting docs:

- `docs/fr3_mask2ee_training_inference_contract_20260326.md`
- `docs/fr3_act_infer_real_minimal.md`

Why documentation now:

- once the first checkpoint exists, operator-facing docs must say how to train
  and infer with it

First-batch edits:

- record the exact config path
- record that raw dataset storage is still absolute `W_s`
- record that policy action semantics are current-frame-relative
- record that runtime no longer semantically depends on `T_B_Ws` for this
  checkpoint family

Acceptance check:

- an operator can tell which runtime path a checkpoint uses without reading code

## Implementation Notes By File

### `src/lerobot/configs/franka_research3_rel2ee_act_das.yaml`

Purpose:

- declare the new checkpoint family

Do:

- reuse current FR3 ACT+DAS defaults
- turn on `mask2ee`

Do not:

- add multi-step observation history
- mix this config with old `ee2ee` naming

### `src/lerobot/processor/relative_ee_action_processor.py`

Purpose:

- isolate the geometry transform

Do:

- keep the implementation purely geometric
- make it independently testable

Do not:

- embed FR3 runtime device access
- make it depend on `T_B_Ws`

### `src/lerobot/policies/act/processor_act.py`

Purpose:

- own ACT-specific processor composition

Do:

- add the relative step conditionally by config

Do not:

- rewrite unrelated tactile or normalization code in the same patch

### `tools/fr3/fr3_act_infer_real_runtime.py`

Purpose:

- own checkpoint-specific live decode

Do:

- branch by checkpoint contract
- compose relative output with current live pose

Do not:

- delete the old absolute `W_s` path in the first batch

## Recommended Patch Breakdown

Keep the first implementation in small patches.

Recommended sequence:

1. config + config tests
2. relative processor + unit tests
3. ACT training wiring
4. runtime decode wiring
5. runtime tests
6. docs updates

This reduces rollback cost and makes the geometric transform testable before it
is hidden inside FR3 runtime code.

## First-Batch Success Criteria

The first batch is complete only when:

- a new relative-contract config exists and parses
- a geometric relative action processor exists with unit coverage
- ACT training can produce a checkpoint with relative action semantics
- FR3 preview inference can decode that checkpoint without `W_s` semantics
- old `ee2ee` checkpoints still run unchanged

## Explicitly Deferred To Batch 2

Do not include these in the first batch:

- `n_obs_steps > 1`
- history deque support for ACT observations
- relative proprio history in `observation.state`
- explicit velocity features
- redesign of the ACT observation encoder
- dataset canonical schema migration

Those belong to the full UMI-style follow-up once the MVP proves the action-side
relative contract is valuable.
