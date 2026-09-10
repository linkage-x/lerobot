# rig0818 × Hybrid Carrier V1 browser simulator

This implements the roadmap's M0 → M1/L1 → MC-lite first round, plus an executable L2 image pilot. The static web app has no runtime package dependencies: fisheye projection, visibility, FIM, TCP propagation, paired Monte Carlo, block bootstrap and charts run in a module worker. Canvas renders the same geometry. The separate Python/OpenCV runner renders actual images and runs the repository's native detection and multi-camera pose solver.

Run it from the repository root so the page can read the two frozen CAD descriptors:

```bash
python3 -m http.server 8000 --bind 127.0.0.1
```

Then open `http://127.0.0.1:8000/docs/rig0818_vs_hybrid_carrier_v1/simulator/`.

The default R0 geometry is **proxy** because the measured-layout JSON named in the 0818 report is absent. Load a `marker_layout/measured_v1` file to replace the proxy; corners must be clockwise about the outward normal. CAD plate loops are explicitly reversed for the proxy. Imported pad geometry follows the layout frame rather than retaining the CAD origin. The declared TCP offset must be expressed in that same frame. Camera and task contracts remain synthetic. G2 is always `Inconclusive` in the pilot: a status string or favorable numeric result cannot promote it to production evidence.

The purpose-built optical simulator is appropriate here because the experiment concerns camera measurements and target geometry, rather than robot dynamics. L2 uses subpixel fisheye rays, exact triangle intersections and metric-space marker textures; it does not warp a flat sticker homography directly onto a fisheye image. Native OpenCV marker decoding is also an independent check of the renderer's corner order. The camera model follows [OpenCV's fisheye specification](https://docs.opencv.org/4.x/db/d58/group__calib3d__fisheye.html).

Run an L2 pilot from the repository root (requires the existing NumPy, SciPy, OpenCV-contrib and Node installations):

```bash
python3 docs/rig0818_vs_hybrid_carrier_v1/simulator/l2_runner.py --poses 48
```

Refresh the page or click **刷新本地结果**. The top result selector switches between MC-lite and L2. R0/H0 use the same corner BA; H0/H1 use identical colour images and corner observations, while H1 adds measurements from the native Hybrid detector. H1 uses the H0 joint solution as its initialization and falls back to it if the joint edge fit fails. A separate row reports native per-camera cold starts followed by joint BA. All initializations come from decoded image observations, never truth.

`l2_report.json` includes the complete scene, per-frame requests including failures, true/estimated poses, decoded IDs, native outcomes, covariance convention, source hashes and PNG previews. Reports can be imported locally without upload. `--layout /path/to/layout.json` supplies measured R0. `--speed .28 --angular-speed 55 --exposure-samples 5` integrates motion over an exposure. `--supersample 1|2|3|4` supports rendering convergence checks. The CLI is deliberately offline; it does not contact Thor or change production services.

The CDF always uses **all requested frames**. Cards show each arm's accepted-frame p95; the delta/CI uses only common-success frames and episode-block resampling. Missing frames stay in coverage and yield denominators. Equal installation distributions and paired draws are used across arms. Angular timing error is included, zero occlusion removes synthetic occluders, and one-anchor recovery is not silently enabled for H1.

Remaining roadmap work: measured R0 and production camera/task snapshots; renderer convergence thresholds frozen over the full image/pose domain; production strategy gate/fallback parity; continuous tracking/reacquisition; calibrated noise/failure distributions; full L3 truth/estimator perturbation and raw/smoothed replay; held-out G2 validation. The bundled ideal-colour image pilot cannot certify these.

Run the deterministic numerical tests with:

```bash
cd docs/rig0818_vs_hybrid_carrier_v1/simulator
npm test
```

Every completed run can export a machine-readable `decision.json` and per-frame CSV from the page.

The audit export buttons refer to MC-lite. The complete L2 report is also available as [l2_report.json](l2_report.json).

```bash
python3 -m unittest discover -s docs/rig0818_vs_hybrid_carrier_v1/simulator/tests -p 'test_*.py' -v
# With the static server running and a local Playwright install:
PLAYWRIGHT_MODULE=/path/to/playwright/index.mjs node docs/rig0818_vs_hybrid_carrier_v1/simulator/tests/browser-smoke.mjs
```

Browser smoke checks the worker, rerun, arm selection, scrubber, seed reset, CSV download, L2 loading and mobile overflow. Screenshots are written to `/tmp/rig-ab-desktop.png` and `/tmp/rig-ab-mobile.png`.
