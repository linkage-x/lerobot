# FR3 Inference Frame Alignment Findings (2026-03-23, updated 2026-03-24)

## Scope

This note keeps only the durable conclusion from the FR3 real-robot inference frame-alignment investigation.

## Final Conclusion

The large step0 inference mismatch was not primarily caused by the main pose transform chain.

The useful conclusions are:

- `left/right` image semantics are correct.
- The replay-style `I/E` extrinsic handling was necessary and should stay.
- The remaining step0 mismatch was traced to gripper observation semantics, not the geometry chain itself.

More specifically:

- live `robot.get_observation()['gripper.pos']` is hardware-normalized in `[0,1]`
- dataset / ACT policy input expects gripper values in aperture-style dataset units
- preview previously hid this by applying a virtual correction only inside policy input
- real-run still used a live observation in the wrong semantic space, so the first policy target could jump and hit `hold_first_frame`

## What This Means

For the step0 issue investigated on 2026-03-23:

- the geometry track is no longer the primary suspect
- the gripper-unit contract is the important closed finding
- the authoritative runtime behavior is now documented in:
  - `docs/fr3_act_infer_real_minimal.md`
  - `docs/fr3_act_infer_runtime_fix_20260324.md`

## What Was Still Valuable From The Investigation

The earlier investigation was still useful because it established three durable facts:

- introducing the replay-style `I/E` extrinsic removed the obvious large frame bias
- dataset-start alignment diagnostics were informative once they measured real spread instead of a trivial self-check
- the next high-value debugging target should be policy input contract mismatches before revisiting transform algebra

## Status

Closed for the original step0 mismatch question.

Still open, but outside this note:

- tactile wire-format closure
- long-rollout runtime robustness

## 2026-04-03 Live Baseline Update

A newer live inference baseline proved materially more stable than the default
startup sequence:

```bash
docker compose --profile infer -f docker/docker-compose.yml run --rm \
  lerobot-infer-fr3-act bash -lc '
cd /lerobot &&
PYTHONPATH=/lerobot/src:/lerobot/tools/fr3 \
/lerobot/.venv/bin/python tools/fr3/fr3_act_infer_real_runtime.py \
  --checkpoint=/lerobot/outputs/train/fr3_rel2ee_act_das_noqoff_na10_20260401_153958/checkpoints/100000 \
  --replan-every=1 \
  --first-frame-max-pos-delta-mm=20 \
  --first-frame-max-rot-delta-deg=8 \
  --no-align-gripper-to-dataset-start
'
```

Durable takeaway:

- for this checkpoint, startup gripper auto-alignment was a larger source of
  instability than the geometry chain
- disabling startup gripper alignment produced a much more usable live
  inference baseline
- with `replan_every=1`, the later rollout `TRACK` logs stayed mostly within a
  few millimeters and well below `1 deg`, so the execution layer was no longer
  the dominant problem in that run

This does not prove train/infer start-state equivalence. It is a practical
baseline for online diagnosis while the startup gripper contract remains
imperfect.
