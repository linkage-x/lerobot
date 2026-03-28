# FR3 Pick-Place Policy Dataset Validation Runbook (2026-03-25)

## Scope

This runbook turns the plan in `docs/fr3_pickplace_policy_dataset_validation_plan_20260324.md` into an operator-facing execution sequence.

Goal:

- determine whether the current FR3 pick-place failure is dominated by `checkpoint quality`, `live observation contract`, or `execution fidelity`
- minimize real-robot trial count until the failure is localized

This runbook is intentionally biased toward existing scripts already present in `tools/fr3/`.

## Decision Gates

Use these gates to control whether the next phase is allowed to run.

### Gate 0: Decode Consistency

`infer decode` and `replay decode` must agree to near-zero error on sampled dataset frames.

If this gate fails:

- stop
- debug frame transforms, pose decode, and replay/infer contract mismatch
- do not interpret later policy-vs-dataset errors as model-quality evidence yet

### Gate 1: Dataset-Fed Policy Sanity

When fed exact dataset observations, the checkpoint must still preserve task-relevant action structure.

If this gate fails:

- stop real-robot pick-place attempts
- prioritize `checkpoint / dataset / preprocessing / action decode`

### Gate 2: Live Contract Check

Only run the live-vs-dataset comparison after Gate 1 is at least directionally healthy.

If this gate fails:

- prioritize `live observation contract`
- do not spend additional real-robot attempts on preview or actuation until the contract issue is narrowed

## Required Inputs

Before starting, confirm the following are known:

- checkpoint path, for example `outputs/train/2026-03-19/10-48-39_act/checkpoints/060000`
- dataset root, if the checkpoint metadata does not already resolve it
- `3-5` known-success pick-place episode IDs
- one writable run directory under `outputs/fr3_policy_validation/`

## Run Directory Layout

Create one dated run directory per investigation pass.

Suggested layout:

```text
outputs/fr3_policy_validation/2026-03-25_run01/
  raw/
  report/
  report/phase_labels.csv
  report/per_frame_action_error.csv
  report/topk_worst_frames.csv
  report/decision.md
```

## Phase Label File

Create a minimal manual phase label file for the sampled successful episodes.

Path:

- `outputs/fr3_policy_validation/<run_id>/report/phase_labels.csv`

Format:

```csv
episode,frame_start,frame_end,phase
13,0,12,approach
13,13,20,pre_grasp_alignment
13,21,24,close_gripper
13,25,34,lift
13,35,52,transport
13,53,64,place
13,65,70,open_gripper
```

Notes:

- phase labeling is intentionally manual for the first pass
- do not block on automation here
- the first useful question is where the earliest divergence begins, not whether the labels are perfectly elegant

## Phase 0: Decode Consistency Check

Purpose:

- verify that runtime infer pose decode and replay pose decode are consistent before interpreting model behavior

Primary script:

- `tools/fr3/fr3_compare_infer_replay_decode.py`

Run once per sampled episode.

Command template:

```bash
python tools/fr3/fr3_compare_infer_replay_decode.py \
  --dataset outputs/datasets/lerobotv3_0310_100ep \
  --episode <episode_id> \
  --source action \
  --max-frames 8
```

Save output:

- `outputs/fr3_policy_validation/<run_id>/raw/phase0_decode_ep<episode_id>.log`

Expected output fields:

- `[CHECK] frame=... pos_err_mm=... rot_err_deg=...`
- `[RESULT] max_pos_err_mm=... max_rot_err_deg=...`

Pass condition:

- errors are near zero across sampled frames

Fail action:

- debug replay/infer decode mismatch first
- do not continue to Phase 1

## Phase 1A: Sparse Dataset-Fed Policy Sanity Check

Purpose:

- quickly determine whether the checkpoint still produces task-shaped actions when fed exact dataset observations
- identify earliest divergence before paying the cost of dense frame sweeps

Primary script:

- `tools/fr3/fr3_check_policy_dataset_frame.py`

Start with sparse frame indices per episode.

Suggested initial frame list:

- `0,1,2,4,8,16,24,32,40`

Add extra frames if needed around expected `close_gripper` and `lift` boundaries.

Command template:

```bash
python tools/fr3/fr3_check_policy_dataset_frame.py \
  --checkpoint outputs/train/2026-03-19/10-48-39_act/checkpoints/060000 \
  --episodes <ep1,ep2,ep3> \
  --frame-indices 0,1,2,4,8,16,24,32,40
```

Save output:

- `outputs/fr3_policy_validation/<run_id>/raw/phase1_sparse.log`

Expected output fields:

- `[CHECK] episode=... frame=... pos_err_mm=... rot_err_deg=... grip_err_mm=...`
- `[SUMMARY]`
- `[WORST]`

Immediate questions to answer from Phase 1A:

- is `first-frame` already wrong
- does `gripper` diverge earlier than pose
- do phase boundaries still look preserved
- does the policy appear to hover or emit only safe micro-motions instead of committing to grasp/lift

## Phase 1B: Dense Dataset-Fed Sweep

Purpose:

- refine the localization once the sparse pass shows the checkpoint is at least directionally plausible

Use the same script with a denser frame list concentrated around phase boundaries and failure-heavy ranges.

Command template:

```bash
python tools/fr3/fr3_check_policy_dataset_frame.py \
  --checkpoint outputs/train/2026-03-19/10-48-39_act/checkpoints/060000 \
  --episodes <ep1,ep2,ep3> \
  --frame-indices <dense_frame_list>
```

Save output:

- `outputs/fr3_policy_validation/<run_id>/raw/phase1_dense.log`

Focus:

- `pre_grasp_alignment -> close_gripper`
- `close_gripper -> lift`
- `transport -> place`

## Report Artifacts

The current Phase 1 script prints structured lines. The preferred path is to build all report artifacts in one step with:

- `tools/fr3/fr3_build_policy_validation_report.py`

Command template:

```bash
python tools/fr3/fr3_build_policy_validation_report.py \
  --log outputs/fr3_policy_validation/<run_id>/raw/phase1_sparse.log \
  --phase-labels outputs/fr3_policy_validation/<run_id>/report/phase_labels.csv \
  --run-id <run_id> \
  --checkpoint outputs/train/2026-03-19/10-48-39_act/checkpoints/060000 \
  --dataset outputs/datasets/lerobotv3_0310_100ep \
  --per-frame-csv outputs/fr3_policy_validation/<run_id>/report/per_frame_action_error.csv \
  --topk-out outputs/fr3_policy_validation/<run_id>/report/topk_worst_frames.csv \
  --output-md outputs/fr3_policy_validation/<run_id>/report/decision.md
```

If you need the intermediate steps separately, use:

- `tools/fr3/fr3_parse_policy_dataset_frame_log.py`
- `tools/fr3/fr3_generate_policy_validation_decision.py`

Command template:

```bash
python tools/fr3/fr3_parse_policy_dataset_frame_log.py \
  --log outputs/fr3_policy_validation/<run_id>/raw/phase1_sparse.log \
  --phase-labels outputs/fr3_policy_validation/<run_id>/report/phase_labels.csv \
  --run-id <run_id> \
  --checkpoint outputs/train/2026-03-19/10-48-39_act/checkpoints/060000 \
  --output-csv outputs/fr3_policy_validation/<run_id>/report/per_frame_action_error.csv \
  --topk-out outputs/fr3_policy_validation/<run_id>/report/topk_worst_frames.csv
```

For each run, produce these report files:

### `per_frame_action_error.csv`

```csv
run_id,checkpoint,episode,frame,phase,is_first_frame,pos_err_mm,rot_err_deg,grip_err_mm
```

### `topk_worst_frames.csv`

```csv
run_id,episode,frame,phase,pos_err_mm,rot_err_deg,grip_err_mm,dominant_metric
```

### `decision.md`

Generate the initial skeleton with:

```bash
python tools/fr3/fr3_generate_policy_validation_decision.py \
  --per-frame-csv outputs/fr3_policy_validation/<run_id>/report/per_frame_action_error.csv \
  --phase-labels outputs/fr3_policy_validation/<run_id>/report/phase_labels.csv \
  --output-md outputs/fr3_policy_validation/<run_id>/report/decision.md \
  --dataset outputs/datasets/lerobotv3_0310_100ep
```

The generated file intentionally leaves true decision fields human-editable while pre-filling evidence summaries and candidate signals.

The resulting header shape is:

```md
# FR3 Policy Validation Decision

- run_id:
- checkpoint:
- dataset:
- sampled_episodes:
- decode_gate:
- dataset_fed_result:
- earliest_divergence:
- worst_phase:
- dominant_metric:
- dominant_blocker:
- next_action:
```

Interpretation rules:

- `dominant_metric=gripper` usually points to timing or task-transition failure
- `dominant_metric=pos` or `rot` from frame 0 can indicate contract, decode, or preprocessing issues
- a healthy mean with broken phase transitions is still a fail for pick-place

## Phase 2: Phase-Specific Analysis

Purpose:

- avoid being misled by global averages

Use `phase_labels.csv` to summarize the Phase 1 data by task phase.

Minimum outputs:

- earliest divergence phase
- worst phase by median or p95 error
- whether `close_gripper` and `lift` transitions are preserved

Required conclusion sentence:

- exactly one sentence naming the current dominant blocker hypothesis

Examples:

- `Dataset-fed output already diverges at pre_grasp_alignment, dominated by gripper timing.`
- `Dataset-fed pose remains plausible, but close->lift transition is absent.`
- `Dataset-fed output is directionally healthy; shift focus to live observation contract.`

## Phase 3: Live-vs-Dataset Contract Check

Only run this phase after Gate 1 is directionally healthy.

Purpose:

- determine whether the live step0 bundle looks like the dataset contract the checkpoint expects

Primary script:

- `tools/fr3/fr3_compare_live_capture_to_dataset.py`

Command template:

```bash
python tools/fr3/fr3_compare_live_capture_to_dataset.py \
  --capture-dir outputs/fr3_live_step0/latest \
  --checkpoint outputs/train/2026-03-19/10-48-39_act/checkpoints/060000
```

Save output:

- `outputs/fr3_policy_validation/<run_id>/raw/phase3_live_contract.log`

Expected output fields:

- `[START_MATCH]`
- `[FRAME_MATCH]`
- `[HYPOTHESIS] live_ee_frame=E|I ...`
- `[START_IMAGE]`
- `[FRAME_IMAGE]`

Questions to answer:

- does live step0 look closer to a dataset start or to a mid-episode frame
- which frame hypothesis looks less wrong: `E` or `I`
- do image metrics look directionally consistent with the nearest dataset reference

## Returning To Preview Or Real Actuation

Return to preview only when all of the following hold:

- Gate 0 passed
- Phase 1 did not show early catastrophic divergence
- Phase 3 does not show an obvious live contract mismatch

If those conditions are not met:

- do not spend additional real-robot trials
- keep the investigation offline until the dominant blocker is narrower

## Recommended Execution Order

Run in this exact order:

1. select `3-5` known-success pick-place episodes
2. create `phase_labels.csv`
3. run Phase 0 for each sampled episode
4. run Phase 1A sparse dataset-fed checks
5. generate `per_frame_action_error.csv`, `topk_worst_frames.csv`, and `decision.md`
6. only if Phase 1A is directionally healthy, run Phase 1B dense checks
7. only if Gate 1 is directionally healthy, run Phase 3 live contract checks
8. decide whether preview is justified

## Minimum Definition Of Done

This runbook execution is considered complete only when these files exist for the run:

- `raw/phase0_decode_ep*.log`
- `raw/phase1_sparse.log`
- `report/per_frame_action_error.csv`
- `report/decision.md`

The run is considered successful when `decision.md` answers at least one of:

- does the checkpoint still reproduce pick-place actions on dataset observations
- if not, which phase breaks first
- if yes, which live modality or frame hypothesis is most suspicious
- is the dominant blocker currently `checkpoint quality`, `input contract`, or `execution fidelity`
