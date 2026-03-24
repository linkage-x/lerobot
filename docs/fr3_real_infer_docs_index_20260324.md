# FR3 Real Inference Docs Index (2026-03-24)

Use this index to avoid duplicating FR3 real-robot inference knowledge across multiple notes.

## Source Of Truth Order

1. `docs/fr3_act_infer_real_minimal.md`
   - authoritative runtime entrypoint, startup defaults, input/output contract, recommended commands, open issues
2. `docs/fr3_act_infer_runtime_fix_20260324.md`
   - compact change log for the 2026-03-24 runtime fix only
3. `docs/fr3_infer_image_semantics_validation_20260323.md`
   - closed conclusion for `left/right` image semantics
4. `docs/fr3_infer_frame_alignment_findings_20260323.md`
   - closed conclusion for the original step0 mismatch investigation
5. `docs/tactile/fr3_das_tactile_packet_investigation_20260323.md`
   - still-open tactile wire-format investigation
6. `docs/fr3_replay_tracking_findings_20260319.md`
   - replay tracking evidence and branch-instability analysis

## Documentation Rules

- put durable runtime behavior and operator guidance in `fr3_act_infer_real_minimal.md`
- put one-off implementation deltas in dated `*_fix_*.md` or `*_findings_*.md` notes
- when an investigation closes, keep only the final conclusion and remove step-by-step dead ends
- keep specialized evidence-heavy analysis separate from the main runtime doc

## Current Open Threads

- tactile `448-byte` payload to dataset `left_raw/right_raw` mapping
- long-rollout runtime robustness and operator safety gates
