# FR3 Replay Tracking Findings 2026-03-19

## Scope

This note records the current findings from real-hardware FR3 DAS replay
tracking experiments on `episode 0` of:

- `outputs/datasets/lerobotv3_0310_100ep`

It focuses on whether relaxing `Ruckig` limits improves tracking, and whether
the main replay error is dominated by OTG smoothing, execution lag, or
`state/action` source mismatch.

Related artifacts:

- [outputs/analysis/fr3_peak_analysis_ep000.md](/home/hph/Code/lerobot-replay/outputs/analysis/fr3_peak_analysis_ep000.md)
- [outputs/analysis/fr3_peak_analysis_ep000.csv](/home/hph/Code/lerobot-replay/outputs/analysis/fr3_peak_analysis_ep000.csv)
- [outputs/analysis/fr3_ik_branch_flips_ep000.md](/home/hph/Code/lerobot-replay/outputs/analysis/fr3_ik_branch_flips_ep000.md)
- [outputs/analysis/fr3_ik_branch_flips_ep000.csv](/home/hph/Code/lerobot-replay/outputs/analysis/fr3_ik_branch_flips_ep000.csv)
- [outputs/analysis/fr3_branch_consistent_targets_ep000.md](/home/hph/Code/lerobot-replay/outputs/analysis/fr3_branch_consistent_targets_ep000.md)
- [outputs/analysis/fr3_branch_consistent_targets_ep000.csv](/home/hph/Code/lerobot-replay/outputs/analysis/fr3_branch_consistent_targets_ep000.csv)
- [outputs/analysis/fr3_sim_replay_joint_validation_ep000.md](/home/hph/Code/lerobot-replay/outputs/analysis/fr3_sim_replay_joint_validation_ep000.md)
- [outputs/analysis/fr3_sim_replay_joint_validation_ep000_summary.csv](/home/hph/Code/lerobot-replay/outputs/analysis/fr3_sim_replay_joint_validation_ep000_summary.csv)
- [outputs/analysis/fr3_sim_replay_joint_validation_ep000_details.csv](/home/hph/Code/lerobot-replay/outputs/analysis/fr3_sim_replay_joint_validation_ep000_details.csv)

## Runtime Changes Used

The replay runtime now supports:

- separate `hw vs action` and `hw vs state` error reporting
- `state vs action` source-gap reporting
- joint lag diagnostics:
  - `q_meas vs q_cmd`
  - `q_meas vs q_target`
  - `q_cmd vs q_target`
- OTG sweep CLI flags:
  - `--otg-max-velocity`
  - `--otg-max-acceleration`
  - `--otg-max-jerk`
  - `--otg-velocity-scale`
  - `--otg-acceleration-scale`
  - `--otg-jerk-scale`
- OTG bypass flag:
  - `--disable-otg`
- branch-consistent joint replay flags:
  - `--joint-targets-csv`
  - `--joint-target-column-prefix`

## Experiment Matrix

Three real-hardware replay runs were executed:

1. Default OTG
2. Relaxed acceleration and jerk
3. OTG disabled

Commands:

```bash
python3 tools/fr3/fr3_das_replay_real.py --episode 0

python3 tools/fr3/fr3_das_replay_real.py \
  --episode 0 \
  --otg-velocity-scale 1.0 \
  --otg-acceleration-scale 1.25 \
  --otg-jerk-scale 1.5

python3 tools/fr3/fr3_das_replay_real.py --episode 0 --disable-otg
```

## Results

| Mode | `hw vs action` pos mean / p95 | `hw vs state` pos mean / p95 | `q_meas vs q_cmd` L2 mean / p95 | `q_meas vs q_target` L2 mean / p95 | Outcome |
| --- | --- | --- | --- | --- | --- |
| Default OTG | `12.59 / 27.97 mm` | `10.20 / 22.88 mm` | `2.23 / 4.57 deg` | `3.75 / 9.13 deg` | baseline |
| `acc=1.25, jerk=1.5` | `12.34 / 27.64 mm` | `10.08 / 21.89 mm` | `2.23 / 4.55 deg` | `3.63 / 8.71 deg` | no material improvement |
| `--disable-otg` | `26.73 / 115.17 mm` | `25.59 / 114.88 mm` | n/a | n/a | replay became unstable; reflex abort |

The `state/action` source mismatch stayed present in all OTG-enabled runs:

- `state vs action` position gap: `mean=4.65 mm`, `p95=15.60 mm`
- `state vs action` rotation gap: `mean=0.57 deg`, `p95=1.32 deg`

## Key Observations

### 1. Relaxing OTG limits did not solve tracking

Changing acceleration and jerk from:

- `otg_max_acceleration: 8 -> 10`
- `otg_max_jerk: 4000 -> 6000`

did not produce a meaningful change in replay quality. The tracking metrics
stayed within noise relative to the default OTG configuration.

Conclusion:

- OTG limit conservatism is not the primary replay bottleneck for this episode.

### 2. Disabling OTG made replay much worse

With `--disable-otg`, the run degraded badly in the second half and ended with:

- `libfranka: Move command aborted: motion aborted by reflex! ["joint_velocity_violation"]`

The tail of the episode reached:

- `hw vs action` position error around `116 mm`
- `hw vs state` position error around `117 mm`
- `hw vs action/state` rotation error around `18 deg`

Conclusion:

- OTG is currently a required protection and stabilization layer, not an
  optional source of avoidable lag.

### 3. Peak replay errors correlate with aggressive `action` segments

The dominant problematic windows observed online were:

- `frame 160-163`
- `frame 240`
- `frame 264-265`

These frames show:

- `action` EE steps notably larger than `state` EE steps
- large `state/action` source gaps
- large `q_meas vs q_target` values while `q_meas vs q_cmd` stays smaller

Representative examples from real-hardware diagnostics:

- around `frame 162`:
  - `state_action_gap = 15.98 mm / 0.98 deg`
  - default OTG `q_meas vs q_target = 9.56 deg`
  - default OTG `q_meas vs q_cmd = 4.89 deg`
- around `frame 163`:
  - `state_action_gap = 16.57 mm / 1.33 deg`
  - default OTG `q_meas vs q_target = 11.41 deg`
  - default OTG `q_meas vs q_cmd = 4.74 deg`

Conclusion:

- the problem is more consistent with aggressive target changes plus source
  mismatch than with the OTG command stream itself lagging excessively.

## Offline Peak-Frame Analysis

An offline analysis script was added:

- [tools/fr3/fr3_analyze_peak_segments.py](/home/hph/Code/lerobot-replay/tools/fr3/fr3_analyze_peak_segments.py)

It computes, for selected peak frames:

- `state/action` EE pose gap in base frame
- `state` IK joint targets
- `action` IK joint targets
- raw joint deltas
- wrapped shortest-path joint deltas

Frames analyzed:

- `160, 162, 163, 240, 264, 265`

Summary from [outputs/analysis/fr3_peak_analysis_ep000.md](/home/hph/Code/lerobot-replay/outputs/analysis/fr3_peak_analysis_ep000.md):

| frame | `ee_gap(mm)` | `ee_gap(deg)` | `joint_gap_wrap_max(deg)` | `joint_gap_raw_max(deg)` |
| --- | --- | --- | --- | --- |
| 160 | `17.44` | `1.13` | `147.62` | `332.40` |
| 162 | `15.98` | `0.98` | `159.00` | `332.40` |
| 163 | `16.57` | `1.33` | `169.80` | `321.60` |
| 240 | `19.09` | `1.22` | `175.02` | `273.07` |
| 264 | `10.09` | `0.92` | `170.22` | `321.60` |
| 265 | `7.83` | `0.94` | `176.53` | `321.60` |

Interpretation:

- even relatively small `state/action` EE differences, on the order of
  `8-19 mm` and about `1 deg`, can map to dramatically different joint-space IK
  solutions
- the wrapped joint deltas still reach `148-177 deg`
- the raw joint deltas expose branch flips and periodicity effects up to
  `273-332 deg`

This strongly suggests that the current replay issue is not simply “track the
same joint target faster.” In these peak regions, `state` and `action` are
often asking IK for different branches.

## Full-Episode IK Branch-Flip Detection

A full-episode detector was added:

- [tools/fr3/fr3_detect_ik_branch_flips.py](/home/hph/Code/lerobot-replay/tools/fr3/fr3_detect_ik_branch_flips.py)

It flags three event types:

- `cross_stream_divergence`
- `state_stream_jump`
- `action_stream_jump`

Default thresholds:

- cross-stream:
  - `ee_gap_mm <= 25`
  - `ee_gap_deg <= 2`
  - `joint_wrap_max >= 120` or `joint_wrap_l2 >= 180`
- single-stream jump:
  - `ee_step_mm <= 25`
  - `ee_step_deg <= 2`
  - `joint_wrap_max >= 120` or `joint_wrap_l2 >= 180`

Detected counts for `episode 0`:

| type | count |
| --- | --- |
| `cross_stream_divergence` | `275` |
| `state_stream_jump` | `101` |
| `action_stream_jump` | `106` |

This result is important even though the threshold is intentionally permissive:

- branch-instability is not localized to only `160-165` or `264-265`
- the episode shows widespread IK-branch sensitivity
- the previously identified peak windows are part of a broader pattern, not isolated anomalies

Representative windows:

- `156-166`
  - `cross_stream_divergence` appears on nearly every frame in the window
  - `state_stream_jump` appears on `156`, `157`, `162`
  - `action_stream_jump` appears on `159`, `160`
- `236-244`
  - `cross_stream_divergence` appears on every frame in the window
- `260-266`
  - `cross_stream_divergence` appears on `262-266`
  - `state_stream_jump` appears on `262`, `265`
  - `action_stream_jump` appears on `264`

Interpretation:

- replay instability is consistent with a globally branch-sensitive IK mapping,
  not just a few outlier frames
- any replay strategy that blindly treats `action` as a smooth EE target stream
  is likely to keep producing large joint-target discontinuities

## Current Working Hypothesis

The dominant replay issue for this episode is:

1. the offline IK wrapper was only calling `placo`'s `solve(True)` once per
   target pose
2. for poses far from the current seed, one solver step often left very large
   residual error while still returning a joint vector
3. the replay tooling then treated that partially converged joint vector as a
   valid exact IK solution
4. on top of that, `state/action` source mismatch still crosses IK branch
   boundaries in several peak segments

This is more plausible than the alternative hypothesis that the default OTG
limits are simply too strict.

Concrete convergence check:

- for `action` frame `0`, one IK step left about `1024 mm / 113 deg` residual
- the same target converged to `0 mm / 0 deg` after `20` solver iterations
- for `action` frame `160`, one IK step left about `145 mm / 1 deg` residual
- the same target converged to `0 mm / 0 deg` after `5` solver iterations

As a result, the primary offline root cause was not OTG and not FK itself; it
was premature return from the IK solver loop.

## Offline Simulation Gate

A same-chain offline replay validation harness was added:

- [tools/fr3/fr3_sim_replay_validate_joint_targets.py](/home/hph/Code/lerobot-replay/tools/fr3/fr3_sim_replay_validate_joint_targets.py)

It validates joint-target streams by replaying them through the same FR3 DAS
URDF and FK stack used by the offline tooling, then reports:

- FK vs `action` pose error
- FK vs `state` pose error
- per-frame joint step continuity
- bad-frame counts and peak frames

Validation was run on three streams:

- `naive_joint`
- `bc_joint`
- `state_ref_joint`

using the measured real-hardware start pose:

- `start_pose_b_xyzquat = [0.180826, -0.540151, 0.291981, 0.707237, -0.014223, 0.706431, 0.023869]`

Summary from [outputs/analysis/fr3_sim_replay_joint_validation_ep000.md](/home/hph/Code/lerobot-replay/outputs/analysis/fr3_sim_replay_joint_validation_ep000.md):

| stream | `pos_vs_action` mean / p95 | `pos_vs_state` mean / p95 | `joint_step_max` mean / p95 / max | `bad_action` | `bad_state` | `bad_joint_step` |
| --- | --- | --- | --- | --- | --- | --- |
| `naive_joint` | `14.78 / 22.36 mm` | `17.33 / 26.30 mm` | `2.02 / 3.45 / 160.57 deg` | `238` | `240` | `2` |
| `bc_joint` | `2.83 / 13.38 mm` | `3.37 / 14.57 mm` | `1.55 / 3.30 / 87.26 deg` | `28` | `30` | `2` |
| `state_ref_joint` | `10.75 / 21.39 mm` | `6.48 / 5.72 mm` | `3.98 / 3.85 / 174.77 deg` | `34` | `32` | `7` |

Interpretation:

- after fixing IK convergence, all three streams became offline-FK-consistent
  enough to be meaningfully compared
- `bc_joint` is now clearly the best offline stream on this episode in both
  `action` and `state` pose tracking
- `naive_joint` is no longer catastrophically wrong, but still trails `bc_joint`
  by a wide margin in mean and p95 pose error
- `state_ref_joint` tracks `state` better than `naive_joint`, but still shows
  larger branch-jump outliers than `bc_joint`
- current remaining issue is no longer “offline IK is broken”; it is choosing
  and validating the best branch-consistent target stream before hardware replay

Practical implication:

- the robot should still not be used as the first debugging surface
- any new joint-target generation approach must first pass this offline gate
  before another real-hardware replay is attempted
- `bc_joint` is now the only candidate stream that looks strong enough to
  justify a controlled next hardware trial

## Practical Conclusion

Current recommendation:

- keep OTG enabled
- do not spend more time on blind OTG sweep tuning for this episode until the
  `state/action` to IK-branch issue is addressed
- current best-known real-hardware replay path is the validated default:
  `python3 tools/fr3/fr3_das_replay_real.py --episode 0`
- experimental hardware paths are now intentionally gated:
  - `--joint-targets-csv` requires `--allow-experimental-joint-replay`
  - `--disable-otg` requires `--allow-unsafe-otg-bypass`

Higher-value next steps:

1. detect and label likely IK branch flips offline across the whole episode
2. evaluate whether `action` replay should be constrained toward the recorded
   `state` IK branch
3. compare branch-consistent IK seeding strategies instead of further relaxing
   OTG limits

## Branch-Consistent Replay Direction

Current proposed direction:

- keep replay in joint-space OTG
- stop treating `action[t]` as an unconstrained EE target stream
- generate a branch-consistent joint target sequence offline first

Proposed formulation:

1. build a reference branch `q_state_ref[t]` by running sequential IK on
   `state[t]`
2. for each `action[t]`, solve multiple IK candidates from different seeds:
   - previous branch-consistent command
   - current `q_state_ref[t]`
   - previous `q_state_ref[t-1]`
3. score candidates using:
   - EE pose error to `action[t]`
   - joint distance to previous command
   - joint distance to `q_state_ref[t]`
4. reject candidates that exceed large branch-change thresholds
5. if all IK candidates are rejected, fall back to the reference branch or a
   nearby branch-consistent alternative

The purpose is not “track `action` faster.” The purpose is:

- stay on a stable joint branch
- avoid turning small `state/action` EE gaps into huge joint-target jumps
- preserve OTG as a smoothing and safety layer after branch-consistent target
  selection

Implementation starts with an offline generator before touching the real-hardware
runtime.

## Branch-Consistent Implementation Status

The implementation now exists in two pieces:

1. offline target generation:
   - [tools/fr3/fr3_generate_branch_consistent_targets.py](/home/hph/Code/lerobot-replay/tools/fr3/fr3_generate_branch_consistent_targets.py)
2. runtime replay hook:
   - [tools/fr3/fr3_das_replay_real.py](/home/hph/Code/lerobot-replay/tools/fr3/fr3_das_replay_real.py)
   - [tools/fr3/fr3_das_replay_real_runtime.py](/home/hph/Code/lerobot-replay/tools/fr3/fr3_das_replay_real_runtime.py)
   - [franka_research3.py](/home/hph/Code/lerobot-replay/src/lerobot/robots/franka_research3/franka_research3.py)

The new replay path keeps OTG enabled but replaces online EE-to-IK target
generation with offline `bc_joint` targets from CSV.

Example workflow:

```bash
python3 tools/fr3/fr3_generate_branch_consistent_targets.py --episode 0

python3 tools/fr3/fr3_das_replay_real.py \
  --episode 0 \
  --joint-targets-csv outputs/analysis/fr3_branch_consistent_targets_ep000.csv
```

## Current Generator Behavior

Current summary from
[outputs/analysis/fr3_branch_consistent_targets_ep000.md](/home/hph/Code/lerobot-replay/outputs/analysis/fr3_branch_consistent_targets_ep000.md):

- naive divergent frames: `277`
- branch-consistent divergent frames: `209`
- naive ref L2 mean / p95: `213.39 / 287.29 deg`
- branch-consistent ref L2 mean / p95: `165.87 / 287.26 deg`
- chosen pose-score gap to best mean / p95: `7.92 / 67.83`

Interpretation:

- the current selector reduces joint-branch divergence materially relative to
  naive `action` IK
- it does so while usually staying close to the best available candidate in
  pose-score space
- the dominant chosen modes are still IK-derived (`ik_prev_bc`, `naive_direct`,
  `ik_state_ref`, `ik_prev_state_ref`), with `state_ref_direct` fallback used
  less often than before

## Important Caveat

In the current offline Placo IK/FK analysis loop, the absolute FK residual to
the target pose remains unexpectedly large even for candidates returned by the
same IK backend. Because of that, the selector currently uses pose score as a
relative ranking signal among candidates, not as a trustworthy absolute
feasibility metric.

This means the present implementation should be treated as:

- a branch-stability intervention that is ready for replay experiments
- not yet a proof that the offline kinematics stack is metrically consistent in
  absolute EE pose

The next validation step is therefore a real-hardware A/B replay:

1. default online `action -> IK -> OTG`
2. offline `bc_joint -> OTG`

## Real-Hardware Validation Of `bc_joint -> OTG`

One end-to-end real-hardware replay was executed with:

```bash
python3 tools/fr3/fr3_das_replay_real.py \
  --episode 0 \
  --joint-targets-csv outputs/analysis/fr3_branch_consistent_targets_ep000.csv
```

Result:

- `hw vs action` position mean / p95: `308.65 / 454.29 mm`
- `hw vs state` position mean / p95: `308.67 / 454.29 mm`
- `q_meas vs q_cmd` joint L2 mean / p95: `179.91 / 293.56 deg`
- `q_meas vs q_target` joint L2 mean / p95: `330.68 / 405.87 deg`
- `q_cmd vs q_target` joint L2 mean / p95: `288.67 / 405.21 deg`

This is dramatically worse than the default replay baseline and confirms:

- the new runtime hook is wired correctly enough to execute the offline joint
  sequence end to end
- but the current offline `bc_joint` sequence is not valid for hardware replay
- the dominant problem is still upstream in target generation, not in OTG
  integration

Practical conclusion:

- keep the `bc_joint` replay hook
- do not treat the current offline generator output as hardware-safe
- next work should focus on why the offline kinematics stack produces target
  sequences that are branch-consistent on paper but catastrophically wrong on
  the real robot

## Potential Bug: Startup Blend Needed For The First Few Frames

Current real-hardware replay for the legacy dataset
`outputs/datasets/lerobotv3_0310_100ep_aligned_ts` still relies on a startup
blend window for the first `12` frames.

Observed behavior:

- at replay start, the live hardware EE pose and the raw `frame 0` target do
  not coincide
- the typical gap is about `26 mm` in translation plus about `15 deg` in
  rotation
- the runtime therefore interpolates from the live EE pose to the dataset
  target over the first `12` frames
- after the blend window, commanded poses align with the replay reference pose
  stream; remaining error is dominated by hardware tracking, skipped frames, or
  later control degradation

Why this is marked as a bug candidate:

- if replay is expected to reproduce the dataset pose frame-by-frame from
  `frame 0`, requiring a synthetic startup interpolation indicates that the
  initialization contract is not fully self-consistent
- the current blend hides that mismatch online instead of resolving it at the
  contract level

Current assessment:

- this is a replay-start contract issue, not a training-data issue
- it does not currently block training
- it also does not block the current grasp-capable replay path once the blend
  window has passed

Current decision:

- keep the blend path for now because it stabilizes startup and avoids an
  immediate `frame 0` pose jump on hardware
- treat the first-few-frames blend requirement as a documented potential bug
  point
- do not prioritize further investigation at this stage
