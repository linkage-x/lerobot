# Recovered implementation and first image pilot — 2026-09-10

The browser simulator and independent OpenCV image loop run locally. Open
`http://127.0.0.1:8000/docs/rig0818_vs_hybrid_carrier_v1/simulator/`
after starting the static server as described in [README.md](README.md).

The saved [L2 report](l2_report.json) contains 48 requested poses, four synthetic
episodes, seven fisheye cameras, 2× supersampling, 0.22 obstruction severity,
stationary exposures, 0.4 px optical blur and 1 intensity-level Gaussian noise.
R0 is a CAD/size proxy. H0 and H1 share the same rendered colour images.

| Arm | TCP translation p95 | Rotation p95 | Accepted | ≤3 mm and ≤0.5° |
|---|---:|---:|---:|---:|
| R0 | 0.772 mm | 0.516° | 48/48 | 91.7% |
| H0 | 1.047 mm | 1.012° | 48/48 | 62.5% |
| H1 | 0.806 mm | 0.714° | 48/48 | 81.3% |

Paired Δp95(R0−H1) = **−0.035 mm**, with episode-block bootstrap 95% interval
**[−0.232, +0.357] mm**. H1 improves the H0 anchor-only branch here, but does not
establish an advantage over R0. Four synthetic episodes and ideal diffuse
rendering are insufficient for a production claim. The application therefore
reports **INCONCLUSIVE**, not Go. This is image-layer error, not L3 total error.

Validation: 12 JavaScript tests, four independent Python image-oracle checks,
and desktop/mobile browser smoke passed. Checks include ArUco corner winding,
quadrants 0/2/3, fisheye off-axis decoding, occlusion, TCP covariance coordinate
invariance, paired denominators, angular timing perturbation, worker execution,
reruns, CSV download and L2 report loading.

Recovery found and corrected: reversed R0 CAD/ArUco winding; a candidate-specific
installation/noise advantage; ignored angular timing; successful-only CDF;
single-anchor H1 gate mismatch; missing input validation; and card/footer overlap.
The invalid pre-fix image report was moved to
`/tmp/rig_ab_l2_invalid_winding_report.json` and is not presented by the page.

Remaining milestones are recorded in the [roadmap](../roadmap.html): frozen
measured/production inputs, complete L2 production-strategy parity and continuous
tracking, calibrated L3 perturbations/smoothing, and held-out G2 evaluation.
