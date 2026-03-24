# FR3 ACT Runtime Fixes 2026-03-24

This note is the change log for the 2026-03-24 FR3 ACT runtime update.

## What Changed

- unified live gripper observation semantics with dataset aperture semantics before policy input
- integrated `move_to_das_start` directly into the actual inference runtime
- changed launcher/runtime defaults so real execution now performs:
  1. move to DAS start
  2. align gripper to dataset start
  3. start inference

## Why It Mattered

The main preview vs real-run mismatch came from a gripper-unit mismatch, not from the primary pose transform chain.

## Current Source Of Truth

Use these docs instead of treating this file as a full design spec:

- `docs/fr3_act_infer_real_minimal.md`
- `docs/fr3_infer_image_semantics_validation_20260323.md`
- `docs/tactile/fr3_das_tactile_packet_investigation_20260323.md`

## Verification

```bash
docker compose --profile infer -f docker/docker-compose.yml run --rm lerobot-infer-fr3-act \
  bash -lc 'cd /lerobot && PYTHONPATH=/lerobot/src /lerobot/.venv/bin/pytest -q tests/scripts/test_fr3_act_infer_real.py'
```

Result:

- `26 passed in 1.41s`
