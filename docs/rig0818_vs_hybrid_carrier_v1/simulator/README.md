# rig0818 × Hybrid Carrier V1 paired error simulator

This answers the roadmap's 0907-on-gripper Go/No-Go with a virtual A/B: the same trajectory, the same cameras and the same TCP, with only the target changed. There are two layers:

| Layer | Inputs | Pipeline | Role |
|---|---|---|---|
| **Real-input image loop (L2 + L3)**<br>`l3_runner.py` → `l3_report.json` | 0804 production fisheye calibration, the 132514 bimanual trajectory, gripper URDF plus declared housing/hand | ray-traced fisheye images → production ArUco / `HybridCarrierTracker` → production corner BA / `carrier_frame_pose` → production offline SE(3) smoother; the estimator is perturbed per session | **Decides** (top of the page) |
| Synthetic sandbox (M1 / MC-lite, L2 pilot)<br>`simulator-core.js`, `l2_runner.py` | declared seven-camera rig, synthetic trajectory | FIM / Monte Carlo in the browser; synthetic-camera detector pilot | Sensitivity only |

## Why this simulator, and why the page renders in WebGL

The question is how well cameras measure a target's geometry; robot dynamics and contact are not involved. A physics engine (MuJoCo, Isaac) would add nothing here, and its pinhole rasterizer cannot reproduce the production 1920×1080 equidistant-fisheye images. Instead, Python renders exactly what the detector sees (subpixel fisheye rays, exact triangle depth, metric textures, exposure integrated along the recorded motion) and runs the repository's production detection and solve code on those images. The browser then replays the same scene in WebGL, using the same OpenCV fisheye model per vertex, so any production camera's view can be inspected frame by frame next to the errors.

## Run

From the repository root:

```bash
# 1. Freeze inputs (only when a source file changes; writes inputs/ + hashes)
python3 docs/rig0818_vs_hybrid_carrier_v1/simulator/freeze_inputs.py
# 2. Full experiment (resumable; about 75 min on 28 workers)
python3 docs/rig0818_vs_hybrid_carrier_v1/simulator/l3_runner.py
# 3. Page
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8000/docs/rig0818_vs_hybrid_carrier_v1/simulator/`.

Each stage caches its output under `outputs/rig_target_ab/<run-name>/`: `select` (mount choice on dev episode 0), `render`, `native`, `solve`, `smooth`, `convergence` and `report`. To rebuild only the statistics, pass `--stages report`. Changing the scene or the rendering configuration requires a new `--run-name`, and the runner refuses to mix renders from two scenes. Changing the estimator settings clears the cached solve/smooth results automatically. `--max-per-block N` gives a small debug run.

## What is real and what is assumed

- **Read from the repository, hashed:** camera K/D/extrinsics (`thor_gmsl2_selfcal_0804_fisheye_*`; the same pairing the 132514 trajectory was tracked with), the 132514 corner-BA poses and the marker→TCP bundle, the URDF finger mechanism, and both CAD descriptors. The production 0907 layout file is checked point by point, and so is the browser's R0 geometry.
- **Measured R0 geometry:** `inputs/r0_layout_measured.json` is a byte-for-byte copy of Thor `~/lerobot/outputs/metrology/newrig_20260818_115119/layout_cadprior_no09_10.json`. That layout is the 0818b capture, solved in the CAD frame with a 1 mm CAD prior and cam_09 excluded; its pivot |p_rig| is 0.90 mm, the best of the four CAD-frame layouts there. It is used for both the rendered truth and the solver. `freeze_inputs.py --r0-layout` refuses a data-defined frame, a wrong id set or mirrored winding. Without the file, the runner falls back to the CAD proxy with measured edge lengths and G0 stays open. Two of the real stickers (7 and 14) sit a half and a quarter turn from the proxy's first corner, which the proxy's self-consistent simulation could not see.
- **Assumed and labelled as such:** the BOX housing and hand/forearm envelope, the rig0818 bracket rods, and where each target would sit on the gripper. Mounts use one socket stud for both targets, and yaw × boom tilt is chosen per arm on dev data. The socket-at-TCP control keeps each target's dev-chosen yaw and tilt, so the dome stays up, and moves only the socket onto the TCP point. Until 2026-09-11 it took the URDF TCP axes, which laid both targets on their side.
- **Truth vs estimator:** images are always rendered from truth. An L3 session perturbs only what the solver is told: camera pose and intrinsics, marker layout, body scale, the marker→TCP bundle, and label time. All three arms get the same draw. Sigmas and their evidence are in `noise_sources.json`.

## Reading the result

- Every denominator counts all requested frames. Δp95 = p95(R0) − p95(H1) is computed on frames where both arms returned a pose. Its 95% interval comes from a bootstrap that resamples sessions and 2 s time blocks.
- **Primary scenario:** gripper mount, evidence-level perturbation, smoothed output (the dataset label).
  - `CONDITIONAL_GO` requires the interval's lower bound ≥ 0.8 mm plus the H1 absolute criteria.
  - `NO_GO` means the upper bound is < 0.8 mm.
  - The formal decision equals the numeric one only when every blocking G0 check is closed.
  - The page recomputes the decision from the numbers and ignores any decision string in the file.
- **Rigid-body breakdown** (`groups.<group>.mechanism`): the same per-frame poses re-read at the anchor centroid, the rig origin (socket), a common arm and the TCP, plus rotation. The TCP error is the centroid error plus rotation × arm. A number reported at the socket, such as every number from the 09-09 bench, says nothing about the rotation term. `bench_crosscheck` recomputes the bench's own statistics on the simulated poses: noise floor, facet refinement step, cameras decoding per frame.
- The noiseless ArUco corner bias is reported, not gated. It is a property of the detector on these images; R0's 4.4 mm quiet zone against a black bracket is the main source.

## Tests

```bash
cd docs/rig0818_vs_hybrid_carrier_v1/simulator && npm test          # JS kernel, L2/L3 readers and decision rule
python3 -m unittest discover -s docs/rig0818_vs_hybrid_carrier_v1/simulator/tests -p 'test_*.py'
# With the static server on :8000 and a local Playwright (or playwright-core) install:
PLAYWRIGHT_MODULE=/path/to/playwright-core/index.mjs node docs/rig0818_vs_hybrid_carrier_v1/simulator/tests/browser-smoke.mjs
```

The Python tests cover:
- parity with the production layout and the browser geometry;
- the gripper TCP chain against the task bundle;
- mount clearance and tilt;
- constant-twist extrapolation;
- the bbox-culled rasterizer against brute-force ray casting;
- ray/pixel reprojection;
- occluders hiding markers from the real decoder;
- estimator-only, reproducible perturbations.

The browser smoke test covers the verdict, cards, forest and CDF charts, the WebGL orbit and fisheye views, the counterexample jump, the sandbox, and mobile overflow.

The synthetic L2 pilot still runs with `python3 docs/rig0818_vs_hybrid_carrier_v1/simulator/l2_runner.py --poses 48`.
