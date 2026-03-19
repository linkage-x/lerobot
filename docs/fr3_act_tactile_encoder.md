# FR3 ACT Tactile Encoder Decision Log

## Context

- Dataset: `outputs/datasets/lerobotv3_0310_100ep`
- Policy target: ACT training on FR3 EE-to-EE pick-place data
- Tactile features exported in dataset:
  - `observation.tactile.left_clean`
  - `observation.tactile.right_clean`
  - `observation.tactile.left_raw`
  - `observation.tactile.right_raw`
  - `observation.tactile.valid_mask`

## Dataset Facts

- Single-side tactile shape: `[50, 10]`
- Layout: `row-major`
- Valid cells: `448`
- Invalid cells: `52`
- Clean tactile rule per side:

```text
clean = (raw - baseline) * valid_mask
```

- `valid_mask` is fixed sensor geometry, not frame-varying missingness
- Invalid cells in `*_clean` are already forced to `0.0`

## Decision

First implementation uses only:

- `observation.tactile.left_clean`
- `observation.tactile.right_clean`

It does not explicitly consume `observation.tactile.valid_mask`.

## Rationale

- `*_clean` already encodes the canonical valid region by zeroing invalid cells
- `valid_mask` is static, so the model can learn fixed invalid regions from data layout
- Using only clean tactile reduces implementation complexity and avoids widening the input contract too early
- This is sufficient for a first ACT integration focused on getting tactile into training reliably

## Rejected First-Pass Designs

### Flatten tactile into `observation.environment_state`

Rejected because:

- destroys 2D spatial structure
- ignores local contact topology
- makes ACT treat tactile as generic low-dimensional state

### Use `clean + mask` as explicit channels

Deferred, not rejected permanently.

Reason for deferral:

- technically valid
- but extra complexity is not justified for the first pass because mask is fixed and already applied in `clean`

## Chosen Encoder Shape

Use a shared lightweight 2D tactile encoder:

- input per side: `1 x 50 x 10`
- source tensors: `left_clean`, `right_clean`
- shared CNN weights across left and right tactile maps
- add a learned side embedding after tactile feature extraction
- convert tactile feature maps into tactile tokens
- append tactile tokens to the ACT encoder input sequence

## Why This Matches ACT

ACT already fuses heterogeneous tokens:

- latent token
- proprio token
- optional environment-state token
- image tokens

Tactile tokens fit this structure naturally and preserve more useful information than a flattened vector.

## Current Implementation Scope

1. Extend `ACTConfig` with tactile feature keys and tactile encoder hyperparameters
2. Add a shared tactile CNN encoder in `src/lerobot/policies/act/modeling_act.py`
3. Append tactile tokens before image tokens in the ACT encoder sequence
4. Update `src/lerobot/configs/franka_research3_ee2ee_act_das.yaml` to enable tactile by default
5. Add config parsing and forward smoke tests

## Training Entry Decision

For the tactile dataset, use the DAS config/service:

- config: `src/lerobot/configs/franka_research3_ee2ee_act_das.yaml`
- compose service: `lerobot-train-fr3-act-das`

Preferred explicit command form:

```bash
sudo env HOME=/home/hph docker compose --profile train -f docker/docker-compose.yml run --rm lerobot-internal \
  lerobot-train \
  --config_path=src/lerobot/configs/franka_research3_ee2ee_act_das.yaml \
  --dataset.root=outputs/datasets/lerobotv3_0310_100ep
```

## Follow-Up Work

- compare `clean-only` vs `clean+mask`
- compare `clean-only` vs `clean+raw`
- run short smoke training, then full ACT training
