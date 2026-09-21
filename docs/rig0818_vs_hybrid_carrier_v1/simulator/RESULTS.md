# Results — 2026-09-11

> The `*_report.json` files linked below are generated, and are not tracked in
> git (see `README.md` → Run for how to rebuild them). Every number quoted here
> was read off them at the time of the run.

## Real-input image loop (the decision)

**Formal decision: NO-GO.** Every blocking G0 check is now closed. The measured rig0818 layout was fetched from Thor and frozen into `inputs/r0_layout_measured.json`: the 0818b capture's `layout_cadprior_no09_10.json`, in the CAD frame, with pivot |p_rig| 0.90 mm. It drives both the rendered truth and the solver. On the held-out episodes the 0907 carrier on the gripper is about 3 mm *worse* than rig0818 at p95; the gate asks for it to be at least 0.8 mm better.

Run: `l3_runner.py --run-name measured_r0`. The carrier's renders and native tracking do not depend on R0, so they were reused from the CAD-proxy run through hard links. The control was re-rendered after its orientation fix in `measured_r0_socketfix`. That run hard-links the unchanged gripper-group renders, native tracks, solves and smooths from `measured_r0`, and every gripper-group statistic in the rebuilt report is identical to the previous one. Two native crashes hit this workstation during the run, both inside numpy's C core: a worker segfault and a main-process general-protection fault. The pool resubmitted the worker's job; the report stage was re-run on the same inputs and completed. The report stage is deterministic: two runs under different `PYTHONHASHSEED` values produce identical reports. See [l3_report.json](l3_report.json) and the [page](index.html).

Gripper mount, held-out episodes 1–2 of 132514, 1221 frames (every 2nd frame at 60 fps), TCP translation p95 in mm:

| Scenario | R0 | H0 | H1 | Δp95 R0 − H1 [95% CI] |
|---|---:|---:|---:|---:|
| Nominal (image noise only) · per-frame BA | 1.74 | 4.24 | 4.20 | −2.46 [−3.04, −1.62] |
| Nominal · offline smoothed | 2.07 | 4.52 | 4.45 | −2.39 [−3.11, −1.50] |
| Evidence-level perturbation · per-frame BA | 4.72 | 8.47 | 7.99 | −3.28 [−4.16, −2.48] |
| **Evidence-level · offline smoothed (primary)** | **4.80** | **8.37** | **7.97** | **−3.17 [−4.12, −2.40]** |
| 2× stress · per-frame BA | 9.06 | 15.49 | 13.13 | −4.07 [−6.36, −1.99] |
| 2× stress · offline smoothed | 9.12 | 15.16 | 12.85 | −3.73 [−6.24, −1.72] |
| Control: socket moved onto the TCP (no arm), same yaw and tilt, no own gripper · evidence · smoothed (611 frames) | 2.83 | 2.97 | 2.96 | −0.13 [−0.63, +0.33] |

Pose coverage is 100% for all three arms, on the gripper mount and on the control.

**Correction (2026-09-11):** the first version of the control took the URDF TCP axes (TCP z is −y of the box), which laid both targets on their side (dome-up dot product p50 0.21). There the carrier returned a pose on only 61.7% of frames, and the control read −2.06 [−3.21, −0.58] mm. That was a modelling error, and it contradicted the bench's finding that coverage is not the carrier's weak point. The control now keeps each target's dev-chosen yaw and tilt and moves only the socket.

- **Every gripper-mount interval is below zero**, not just below the 0.8 mm gate. That holds for all three perturbation levels and both outputs.
- **The corrected control ties.** With the socket on the TCP and no arm, the difference is −0.13 [−0.63, +0.33] mm smoothed and +0.11 [−0.45, +0.60] mm per-frame. Even with no arm, the upper bound stays below the gate.
- **Explained afterwards ([follow-up](#the-production-smoothers-residual)):** at zero arm and nominal noise, the production smoother adds more error to the carrier than to R0. Per-frame p95 goes from 0.51 to 1.43 mm for the carrier, against 0.47 to 0.65 mm for R0, and the effect is already visible at slow speed. The translation floor is not the cause. The smoother charges a rotation about the base origin only against the rotation σ, so rotation σ times the TCP's 0.78 m distance from that origin becomes the effective position σ: 0.34 mm for R0, 0.73 mm for the carrier. The decision does not depend on it.
- **Where the gap comes from** (primary scenario):
  - R0 → H0 is −3.57 [−4.48, −2.75] mm: three 47 mm anchors instead of five 61 mm markers. The loss is in rotation, not translation; see [why the 09-09 bench looked better](#why-the-09-09-bench-looked-better).
  - H0 → H1 is +0.40 [+0.01, +0.78] mm: painted-edge refinement on the same images. It is accepted on 98.9% of frames and never trips the 12 mm guard. It helps, but by about a ninth of what the anchors lose.
  - Rotation separates the two targets. Rotation p95 is 0.49° (R0) vs 0.73° (H1) nominal, and 1.19° vs 1.64° at the evidence level.
- **No stratum reverses it.** H1 is significantly worse in every speed band and in every occlusion band with data; at ≥0.45 m/s the gap is −8.57 [−10.20, −3.47] mm. The ≥20% occlusion band has only 2 frames, so it has no interval.
- **Mounts** were chosen on dev episode 0 (198 frames) from the same eight yaw × tilt candidates for both targets:
  - With measured geometry, R0 moves to yaw 180°, level (dev p95 2.64 mm). The proxy's choice, 270° with the boom tilted up 25°, is 3.72 mm on this geometry.
  - The carrier stays at yaw 270°, level. Its best dev p95 at any mount (2.93 mm, yaw 90°, 78% coverage) is still above R0's 2.64 mm.
- **Renderer:** edge shift between 4× and 8× supersampling is p95 0.037 px (threshold 0.05 px), with 2 decode mismatches out of 212.
- **Detector:** on noiseless renders, ArUco corners sit this far from the analytic projection:
  - R0: p50 0.35 px, p95 2.24 px;
  - carrier: p50 0.28 px, p95 1.11 px.

  R0's 4.4 mm quiet zone against the black bracket is the main source.
- **Sticker orientation:** the real stickers 7 and 14 sit a half and a quarter turn from the CAD proxy's first corner, although their centres agree within 1.5 mm. A proxy layout used on real data would silently misread those two markers.
- **Tracking** (nominal, smoothed): relative-pose error p95 at 33 ms / 100 ms / 400 ms / 1 s is 1.8 / 2.3 / 2.5 / 2.7 mm for R0 and 3.1 / 4.5 / 5.2 / 5.8 mm for H1. BA σ 95% ellipsoids cover only 37–50% of frames, so the reported σ is optimistic.
- **G0 passed:**
  - measured R0 layout, production-layout parity, paste quadrants, R0 browser parity;
  - native tracker parity (0 of 1221 and 0 of 611 frames differ);
  - renderer convergence, all-frame denominators, held-out split;
  - real cameras, real task trajectory, real finger mechanism.
- **Disclosed, not blocking:** the housing and hand envelope, the mounts themselves, and the 0902 extrinsics. The 4 ms exposure was measured afterwards on the 132514 footage at about 8.7 ms. At 9 ms the NO-GO holds and the gap widens ([follow-up](#exposure-measured-at-about-87-ms)). The formal report is still the 4 ms run.

### Why the 09-09 bench looked better

The [2026-09-09 bench](../../hybrid_carrier_v1_roadmap_20260904.html) and this run do not disagree once both are read at the same point on the body. A rigid target's error at any point is its error at the feature centroid plus its rotation error times the arm from the centroid to that point.

- **The bench** reports every number at the rig origin (the socket), with the carrier held by hand in open space.
- **The gate** sits at the gripper TCP, 290–370 mm away.

Below, the run's own per-frame poses are re-read at four points on each target (TCP translation p95 in mm, per-frame BA; the last row is rotation p95 in degrees). The same table is on the [page](index.html), built from `groups.gripper.mechanism` in the report.

| Read at | Nominal R0 | Nominal H1 | Evidence R0 | Evidence H1 |
|---|---:|---:|---:|---:|
| Anchor centroid | 0.72 | **0.56** | 3.09 | 3.50 |
| Rig origin (socket, where the bench reports) | 1.06 | 1.06 | 3.34 | 3.78 |
| Common arm: 293 mm from each centroid toward its TCP | 1.74 | 3.29 | 4.60 | 6.45 |
| TCP label (the decision) | 1.74 | 4.20 | 4.72 | 7.99 |
| Rotation (°) | 0.49 | 0.73 | 1.13 | 1.59 |

- **At the socket the carrier matches rig0818, and at its own centroid it is better** (nominal). That is what the bench saw.
- **It loses in rotation, and the ratio is set by geometry.**
  - The anchor corners spread over an rms radius of 63 mm; R0's spread over 96 mm.
  - Rotation precision from a point constellation scales with the inverse of that radius: 96/63 = 1.52.
  - The measured nominal rotation ratio is 0.73/0.49 = 1.49.
- **The arm turns rotation into millimetres.**
  - At a common 293 mm arm, the evidence-level gap is already 1.85 mm.
  - The carrier's dev-chosen mount (yaw 270°, the only yaw with full dev coverage) points its boom away from the TCP: 368 mm from its centroid, against R0's 293 mm. That adds the rest.
- **The corrected control makes the same point with rendered images.** With the socket on the TCP there is no arm, and the two targets tie. Per-frame p95 at nominal is 0.47 mm (R0) against 0.51 mm (carrier); smoothed at the evidence level the difference is −0.13 [−0.63, +0.33] mm.
- **The painted edges do not rescue rotation in the joint solve.**
  - Only views that decode two anchors produce edges (`min_anchors = 2`).
  - On the task that is a median of 2 cameras per frame; 51% of the carrier's views decode a single anchor.
  - Facet refinement moves the corner-BA pose by p50 0.09 mm here. The bench's 40-frame production run moved it by less than 0.05 mm.
  - The threefold advantage from `compare_fiducial_targets` was single-camera, face-on pinhole PnP, where the edges carry most of the information.

The bench's own statistics, computed on the simulated poses:

- **Noise floor** (1 s windows at ≤ 0.10 m/s, residual about a local cubic, per-axis rms):
  - carrier: 0.18 mm / 0.11° here, 0.23 mm / 0.16° on the bench;
  - R0: 0.19 mm / 0.08° here.

  The renderer is slightly cleaner than real images, not harsher on the carrier. The bench fits the estimate, because it has no truth. Here the error signal is fitted instead, because the recorded truth trajectory itself leaves 1.4 mm about a cubic over a second.
- **Visibility.** On the task, a median of 4 cameras decode either target per frame. The carrier decodes on cam_12 and cam_14 in 7% and 9% of frames. On real task footage, the production cube contributes on 8–9% and 7–8%. The bench's seven cameras per frame come from holding the target up in open space.
- **What the bench did not measure.**
  - It compared the carrier with the production AprilTag cube, not rig0818.
  - Its statistic was single-camera against joint consistency, not accuracy at a TCP.

  Neither is the gate.

Absolute values here are not comparable to the main roadmap's A′ 3.8–4.0 mm. Both targets sit on an assumed stud about 290–370 mm from the TCP, and the evidence level adds calibration, bundle and timing draws. The decision rests on the paired Δ.

### Superseded CAD-proxy run (kept as a sensitivity check)

[l3_report_cad_proxy.json](l3_report_cad_proxy.json) is the same pipeline with R0 built from CAD plate centres plus measured edge lengths. Its primary Δ was −2.58 [−3.34, −1.92] mm. Its formal decision was INCONCLUSIVE only because the measured layout was still missing. That report predates the deterministic block labels, so its intervals can move by a few hundredths of a millimetre if regenerated. Swapping in the measured layout moved R0's chosen mount and widened the gap by 0.6 mm; the direction did not change. Its control still has the old sideways orientation.

## Follow-ups (2026-09-11)

The scripts for both are in [followups/](followups/); their caches live under `outputs/rig_target_ab/followups_20260911/`.

### Exposure: measured at about 8.7 ms

- **What production records.** The cameras run Argus auto exposure: `exposure_us: 0` in `tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml` since the file was created, and the gateway on Thor starts with that file. The per-frame Argus metadata has no exposure column. Its EOF − SOF is 14.68 ms on all nine cameras and every frame, which is the readout, and the nine SOFs agree within ±10 µs. The "about 14 ms" once quoted here was the recorder's clamp at 0.85 of the frame period, not a measurement.
- **Measured from motion blur.** A global-shutter exposure E smears the image by L = v·E along the motion, so only edges whose gradient is parallel to the motion get wider.
  - On 13 fast frames of 132514, the cube's image speed is 430–590 px/s, from the recorded trajectory projected through the 0804 cameras.
  - Fitting the widening against the perpendicular edges gives a median E of 8.7 ms (IQR 7.5–9.3 ms).
  - Static frames of the same cameras give L of 0 px (max 0.74 px).
  - One cam_06 frame read 17 ms, longer than the frame period. The cube was rotating there, which its centre velocity misses.
- **Rerun at 9 ms** with five exposure samples, the mounts frozen from this run, and the control thinned to every 16th frame: [l3_report_exposure9ms.json](l3_report_exposure9ms.json).

| Gripper mount, TCP translation p95 in mm | 4 ms (formal) | 9 ms |
|---|---:|---:|
| Nominal · per-frame BA, R0 / H1 | 1.74 / 4.20 | 2.02 / 5.14 |
| Evidence · smoothed, R0 / H1 | 4.80 / 7.97 | 4.95 / 8.44 |
| **Primary Δp95 R0 − H1 [95% CI]** | **−3.17 [−4.12, −2.40]** | **−3.48 [−4.46, −2.57]** |

- **Longer exposure costs the carrier more than R0:** +0.94 against +0.28 mm at nominal. Every level and output stays below zero, and coverage stays 100%.
- **Label time.** The simulator stamps a label at mid-exposure. A production label anchored at the trigger edge or at SOF sits at least E/2 ≈ 4.4 ms from mid-exposure unless compensated. At the task's p95 speed of 0.48 m/s that is about 2 mm. Both arms share it here; the absolute budget and cross-modal alignment do not.

### The production smoother's residual

- **Mechanism.** `metrology/trajectory_smoothing.py` perturbs each observed pose on the left, `T = Exp(δ)·T_obs`.
  - A rotation in δ turns the pose about the base origin and moves the TCP by φ × t. That move is charged only against the rotation σ.
  - The TCP position is therefore held by roughly √(σ_t² + (σ_r·|t|)²), not by σ_t.
  - With |t| ≈ 0.78 m, σ_r·|t| is 0.34 mm for R0 and 0.73 mm for the carrier, against a 0.1 mm translation floor.
  - The carrier's σ comes from the anchor corner BA alone. H0 and H1 report the same σ, so the painted edges do not tighten it.
- **Re-smoothing the cached per-frame solves.** The copy reproduces the cached output exactly. Segments converge within 15 of the 50 allowed iterations, and a linear loss is worse, so neither convergence nor the robust loss is the cause.

| Socket control, nominal | R0 p95 | H1 p95 | H1 second-difference jitter p95 |
|---|---:|---:|---:|
| Per-frame BA | 0.47 | 0.51 | 0.80 |
| Production smoother | 0.65 | 1.42 | 3.84 |
| Body-frame residual, `T = T_obs·Exp(δ)` | 0.46 | 0.51 | 0.83 |

- **Noise-free truth** through the production smoother moves the TCP by 0.44 mm p95 at this run's frame spacing, and by 0.70 mm at 60 Hz (1.3–1.4 mm at medium and high speed). With the body-frame residual the figures are 0.04 and 0.25 mm.
- **The decision does not depend on it.** Pooled over the 24 evidence sessions, Δp95 R0 − H1 is −3.28 mm per-frame, −3.17 mm with the production smoother, and −3.19 mm with the body-frame residual. At that level session-fixed errors dominate, and no smoother removes them.
- **Production labels** go through the same smoother and settings. What a fix would change there is recorded in the main roadmap's offline smoothing section.

## Synthetic L2 pilot (sandbox, not part of the decision)

48 requested poses, four synthetic episodes, seven declared fisheye cameras, 2× supersampling, 0.22 obstruction severity, stationary exposures, 0.4 px blur, 1 DN noise. R0 is the CAD / measured-size proxy; H0 and H1 share the same rendered colour images. On 2026-09-10 the [report](l2_report.json) was regenerated with the committed code and seed and reproduces this table exactly.

| Arm | TCP translation p95 | Rotation p95 | Accepted | ≤3 mm and ≤0.5° |
|---|---:|---:|---:|---:|
| R0 | 0.772 mm | 0.516° | 48/48 | 91.7% |
| H0 | 1.047 mm | 1.012° | 48/48 | 62.5% |
| H1 | 0.806 mm | 0.714° | 48/48 | 81.3% |

Paired Δp95(R0−H1) = −0.035 mm, episode-block bootstrap 95% interval [−0.232, +0.357] mm, with a 120 mm synthetic lever arm. The direction agrees with the real-input run. The gap is smaller because the lever arm is a third as long.
