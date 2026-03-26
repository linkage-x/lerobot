# FR3 Real Inference Docs Index (2026-03-24)

Use this index to avoid duplicating FR3 real-robot inference knowledge across multiple notes.

## Source Of Truth Order

1. `docs/fr3_act_infer_real_minimal.md`
   - authoritative runtime entrypoint, startup defaults, input/output contract, recommended commands, open issues
2. `docs/fr3_mask2ee_training_inference_contract_20260326.md`
   - durable contract for FR3 `mask2ee` training, checkpoint semantics, and inference consistency
3. `docs/fr3_act_infer_runtime_fix_20260324.md`
   - compact change log for the 2026-03-24 runtime fix only
4. `docs/fr3_infer_image_semantics_validation_20260323.md`
   - closed conclusion for `left/right` image semantics
5. `docs/fr3_infer_frame_alignment_findings_20260323.md`
   - closed conclusion for the original step0 mismatch investigation
6. `docs/fr3_pickplace_policy_dataset_validation_plan_20260324.md`
   - investigation plan and hypothesis tree for localizing the remaining pick-place failure
7. `docs/fr3_pickplace_policy_dataset_validation_runbook_20260325.md`
   - operator runbook for executing the dataset-fed validation workflow with concrete commands and artifacts
8. `docs/tactile/fr3_das_tactile_packet_investigation_20260323.md`
   - still-open tactile wire-format investigation
9. `docs/fr3_replay_tracking_findings_20260319.md`
   - replay tracking evidence and branch-instability analysis

## Documentation Rules

- put durable runtime behavior and operator guidance in `fr3_act_infer_real_minimal.md`
- put one-off implementation deltas in dated `*_fix_*.md` or `*_findings_*.md` notes
- when an investigation closes, keep only the final conclusion and remove step-by-step dead ends
- keep specialized evidence-heavy analysis separate from the main runtime doc
- pair investigation plans with an operator-facing runbook once the next action is stable

## Current Open Threads

- pick-place still does not complete on hardware; next step is to execute the dataset-fed validation runbook and localize the dominant blocker
- tactile `448-byte` payload to dataset `left_raw/right_raw` mapping
- long-rollout runtime robustness and operator safety gates
- `mask2ee` is currently ACT-only; a future TODO is to generalize it into a shared state-masking mechanism for more policies with end-to-end tests
