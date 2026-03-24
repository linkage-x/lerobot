# FR3 DAS Tactile Wire Investigation (2026-03-23)

## Scope

This note records the current real-hardware findings for FR3 inference tactile input on the DAS serial link.

## Confirmed Facts

- DAS controller and tactile callbacks arrive on the same serial node: `/dev/ttyUSB0`
- The installed SDK inside the infer container is used for validation:
  - `/opt/dependencies/gen_con_sdk_python_release/scripts/databus.py`
  - `/opt/dependencies/gen_con_sdk_python_release/start_gripper.py`
- Real hardware callbacks were observed from the SDK in the infer container
- The tactile callback `record_data` length observed on hardware is currently `448 bytes`

## Current Mismatch

The dataset tactile representation is:

- per-side shape: `50 x 10`
- per-side valid cells: `448`
- shared valid mask for left/right: `docs/tactile/tactile_valid_mask_50x10.json`

However, the real SDK callback currently exposes a `448-byte` payload total. That is not yet enough to prove how left/right tactile images are encoded on the wire.

Open question:

- does the `448-byte` payload represent one side only
- or a compressed/shared representation that still needs extra decoding

## Working Hypothesis

The current working hypothesis is:

- one tactile callback payload corresponds to one side
- the `448` values are written into the `50x10` image by scattering into valid-mask cells in row-major order

This hypothesis is only partially validated. It matches the single-side valid-cell count and yields a plausible single tactile silhouette when visualized, but it is not yet a full proof of left/right wire semantics.

## Validation Script

The current capture script is:

- `tools/fr3/fr3_capture_tactile_frame.py`

It performs:

1. grab one SDK tactile frame from `/dev/ttyUSB0`
2. assume `448` valid values for one side
3. scatter them into `50x10` using the shared valid mask in row-major order
4. save:
   - raw `50x10` PNG
   - upscaled preview PNG
   - baseline `50x10` PNG
   - baseline preview PNG
   - mask-only PNG
   - baseline-diff heatmap PNG

Example command:

```bash
sudo env HOME=/home/hph docker compose --profile infer -f docker/docker-compose.yml run --rm lerobot-infer-fr3-act \
  bash -lc 'cd /lerobot && PYTHONPATH=/lerobot/src /lerobot/.venv/bin/python tools/fr3/fr3_capture_tactile_frame.py --timeout-s 8 --encoder-freq 5 --tactile-freq 5 --baseline-side left --side-name ttyUSB0_row_major_left_with_baseline'
```

## Latest Result

Latest captured output directory:

- `outputs/tactile_capture/20260323_075458/`

Latest observed stats:

- `payload_len=448`
- `baseline_side=left`
- `baseline_abs_diff mean=0.62`
- `baseline_abs_diff max=28.0`

These numbers are consistent with a near-baseline resting frame under the current row-major single-side hypothesis.

## Current Status

What is already supported:

- real SDK tactile callbacks are reachable from the infer container
- one-frame capture and visualization are working
- baseline and mask visual comparisons are working

What is not yet closed:

- definitive left/right wire-format semantics
- a hardware-validated mapping from the SDK wire payload to dataset `left_raw/right_raw`
- end-to-end proof that the runtime tactile decoder matches the device protocol

## Next Recommended Check

Run the same capture twice and compare:

- `--baseline-side left`
- `--baseline-side right`

If one side consistently produces a smaller baseline-diff and spatially coherent contact regions during manual pressing, that is strong evidence for the corresponding wire-to-image mapping.
