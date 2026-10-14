# Canonical world frame — GMSL2 8-camera rig

`world_reference.json` is the frozen definition of the world this rig measures
in, and it is **tracked on purpose**.

Every re-solve of the extrinsics picks its own gauge camera, so without a frozen
reference each calibration silently redefines the world: the absolute poses in
last week's episodes keep their numbers and stop meaning what they said. The
reference is what makes `world_frame_id` a contract — two recordings are
comparable in absolute terms exactly when they carry the same one.

**This file cannot be regenerated.** Re-running `freeze` mints a *new*
`world_frame_id` for the same physical frame, which orphans the ID stamped into
every episode and derived trajectory recorded so far. Restoring it from git is
the only recovery. That is also why it does not live under `outputs/`: that tree
is 7 GB of regenerable artefacts and gets deleted to reclaim space.

`world_graph.json` holds the world nodes and any recovered cross-world
registration edges. An edge can cost a laser-tracker session to obtain and is
equally unrecoverable, so it is tracked too.

The two status files are gitignored: they are the latest verdict, rewritten by
every check and re-derivable from the reference plus a bundle report.

## Keeping the workstation and the rig on the same world

Copy this directory to Thor; do **not** run `freeze` separately there. Both
machines must name the same physical frame with the same ID, and a second freeze
produces a second ID for the same frame — the precise failure the whole
mechanism exists to prevent.

## Current contents

Frozen 2026-08-19 from `thor_gmsl2_selfcal_0804_fisheye_extrinsics`, the run
production was already using, so adopting it changed no exported pose (verified
to 4.4e-16 m). The axes are inherited from the 0720 robot-base alignment and its
8.3 mm RMS / 2.2° error — which is now a frozen constant offset of the axes
rather than an error re-inherited on every re-solve. Absolute alignment to the
FR3 bases is Phase 9's `T_WB` and is measured separately.

See `metrology/README.md` (§ Phase 2.4) and the roadmap's Phase 2.4 for the
method.

Since then two re-mounts minted islands with no known transform to 08-19:
`world_20260923_143048` (09-23 re-mount, `calib_20260923_cam13refit`) and
`world_20260928_063531` (09-28 re-install, `calib_20260928_143107`, the current
reference). The 09-23 island was exported but never committed here, so episodes
recorded 09-23 18:55 → 09-28 were stamped 08-19; they were corrected on
2026-09-28 with `restamp_world.py` (below).

### The FR3 base (2026-10-09)

`fr3_base` is a node with one edge, `world_20260928_063531 -> fr3_base`. It is
**not** a world anything is recorded in. The reference keeps naming the camera
island, and the edge says how to leave it. The edge was solved by
`register_fr3_base.py solve` from the 10-02 P0 single-tag run
(`outputs/calibration/p0_single_tag_camera_calibration/manual_run_20261002T014903Z`,
137 captures, FR3 `T_base_tcp` + tag6 on the tool), with the cameras **held at
their 0928 poses**:

- Reprojection error is 1.98 px RMS. The fitted tag scale is 0.991, so the
  printed "160 mm" tag is about 158.6 mm.
- Out of sample, a fold's edge predicts the other fold's tag, and the fixed
  cameras measure where it actually is. The tag-centre error is
  p50 1.9–2.0 / p95 4.1–4.2 mm, and that is roughly what a re-expressed pose
  is off by in the workspace. In-sample it is p50 1.5 / p95 2.9 mm.

The P0 run's own solution frees the cameras, which makes the FR3 base a new
world. Pointing production at it without also changing this reference would
stamp FR3-base poses with the island's id (linkage-x/lerobot#51).

To use it, run `export_v3.py --target-world fr3_base`. This re-expresses
`observation.ee_pose.*.base`, `action.ee_pose.*.base` and
`observation.cube_pose.*.base` (camera-frame columns are untouched), and
records the source world block and the edge in `info.json`'s `world_frame`.
Without an edge connecting the two worlds the export is refused, never treated
as an identity.

The GUI's exports pass `--target-world auto`: `fr3_base` when an edge reaches
it, otherwise the recorded world, and the "Export complete" line names which.
It is not offered as a choice, because the consumer (the FR3) fixes it.
`info.json` also gets `tcp_frame` (the URDF link the ee poses carry, from each
session's tracking-run summary), since deployment has to drive that same link.

On the arm, `tools/fr3/fr3_act_infer_real_runtime.py --dataset-alignment`
reads the frame from the policy's own training view (not from a source
re-exported since then), and picks how `T(B,W_s)` is obtained:

- `start_pose` (unstamped or camera-world data): all 6 DoF come from one start
  pose, so a start-orientation error turns the whole trajectory.
- `start_position` (the default for `fr3_base` data): the rotation is the
  measured one, and the demos are only shifted to the arm's start position.
  As of 10-09 most box demos are 1.1–1.3 m from the base, beyond the FR3's
  reach, so they have to be moved.
- `absolute`: identity. Use it only for demos recorded inside the arm's reach.

Both `fr3_base` modes refuse to run unless the IK target is `tcp_frame.link`.
For the box labels that is `corenetic_gripper_ee`, which is what the tracker
hops to from the bundle's `link_lt_gripper_tcp` (an R_x(pi) turn). Run with
`--robot-urdf-path src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper.urdf
--target-frame-name corenetic_gripper_ee`. The labels' origin is the insert-v2
socket centre, not that link's origin, and the offset between the two is still
unrecorded (see the 0929 marker->TCP bundle). `start_position` keeps the error
from that offset small.

The edge is valid only while the rig cameras stay at their 0928 poses and the
FR3 is not re-mounted. A new island or a moved arm needs a new P0 capture and
`register_fr3_base.py solve` → `apply`. `apply` refuses a second edge between
worlds that are already connected.

## Correcting a stamp that is wrong

The stamp records what this file said at record time. If the rig moved and the
new world was not committed first, the stamp is wrong and cannot be recomputed
from the episode. `python -m tools.thor.gmsl2.restamp_world <dataset roots…>
--expect-from <id now> --to <true id> --reason "<evidence>"` records the
decision: dry run by default (`--apply` to write), the target must be a node of
`world_graph.json`, only episodes carrying `--expect-from` are touched, and the
original block is kept under `world_frame.restamp.original`. The real fix is to
commit a new island here *before* recording in it.

### Re-solving and replacing the edge (calibration page, 2026-10-13)

The calibration page's 「FR3 基座注册边」 panel handles the edge from start to end
except for the capture:

- It lists the P0 runs under `outputs/calibration/p0_single_tag_camera_calibration/`.
- It solves any run with `register_fr3_base.py solve`, against the reference's
  own intrinsics. Candidate reports go to `outputs/calibration/fr3_base_registration/`.
- It shows each candidate's out-of-sample verdict. Next to it is how far the
  candidate would move the live edge, measured at that run's own tool positions.
- Swapping in a candidate is a separate, confirmed step: `apply --replace`.

The capture stays at the robot, with `run_p0_two_marker_calibration_local.sh guided`,
because the arm is hand-guided in zero-stiffness mode and the tool opens its own
recorder.

`apply` checks two things before it changes anything:

- It refuses a report solved against a different `world_reference.json`, by sha.
- With `--replace`, it moves the old direct edge into the new edge's `supersedes`.
  It does not go to a top-level key, because `metrology.world_frame.WorldGraph`
  rewrites this file with nodes and edges only.

The swap happens only on the machine the GUI runs on. Copy `world_graph.json` back
here and commit it. `run/sync_to_target.sh` keeps a target graph that has a newer
stamp than this tree's, and says so, rather than silently putting the old edge back.

A passing verdict alone does not show that a candidate is better. Each solve
checks itself only against its own run's captures. On 2026-10-13 all three runs
captured at this mount were re-solved:

- 10-02: reproduces the live edge exactly.
- 10-10 a: held-out p95 5.6 mm.
- 10-10 b: held-out p95 5.2 mm.

The two 10-10 runs each move the live edge by a mean of 2.6 mm at their own tool
positions (0.4°), in different directions. At the 10-13 replay trajectory that is
about 4 mm. Neither removes the 4.6 mm constant the laser measured there. The
method's run-to-run repeatability is therefore about 4 mm. An edge that is to beat
it needs an independent check, such as a laser replay, before it goes live.

### The laser edge (`register_fr3_base.py laser`, 2026-10-13)

The same panel composes the edge through the laser tracker instead of a tag:

    T_world_base = T_world_tracker (station) @ T_tracker_base (FR3 FK vs the laser)

- `T_world_tracker` is a tracker station: the camera-tracked UMI tool held still
  at parked poses, with the SMR on it (`fit_tracker_mount station`, the
  tracker-mount panel, `outputs/laser_tracker/station_*.json`).
- `T_tracker_base` is fitted by `metrology.fr3_replay_laser_error` on a real
  replay with the SMR on the FR3 (the `T_tracker_base` field of its report).

The two halves are one edge only if the tracker did not move between them, and
neither half can show that on its own. Two replays, one on each side of the
station, that agree are the evidence, so the panel lists replay times against the
station's. The tool rejects replays that disagree by more than 1 mm over the
workspace, and warns when there is only one replay or the replays do not bracket
the station.

Other gates:

- The station must certify, in the reference world, with an rms of at most 1.5 mm.
  Its own `certifies` judges only the geometry. On 2026-10-13 a station certified
  at 3.43 mm, because the camera world was 0.65 % short of the laser across it.
- Each replay needs at least 200 frames with the beam on.
- Laser vs FK must have p95 ≤ 1 mm.
- The orientation span must be at least 8°. Below that, the SMR offset trades
  against the transform.

The report has the same schema as a single-tag one, so `apply` and the
candidate table take it as they are. The report's `workspace_points_base_m` are
the replays' TCP positions, which the panel uses for "how far it moves the live
edge".

### Checking edges at a ball (`fr3_base_ball_check.py`, 2026-10-13)

This check referees between edges without the laser and without any TCP
definition. A ball fixed to the table is pivoted about twice:

- **The FR3**, in teaching mode (`tools/fr3/fr3_tcp_pivot.py collect` + `fit`,
  run from the calibration page's "FR3 球窝 pivot" panel). The fit gives the
  socket centre in `corenetic_gripper_ee` and the ball `c` in fr3_base. It also
  gives the socket centre in `corenetic_socket_ee`: that is ~0 if the replay
  target's offset is the socket, and not ~0 if that offset, fitted end to end
  through the live edge, absorbed part of the edge's error.
  Several runs, each from its own heading, fit one socket together
  (`fr3_tcp_pivot.py fit-joint`, the panel's "合并拟合所选"); a single run's
  yaw of ±15° leaves the component along the tool axis undetermined. On
  2026-10-14, nine runs on one ball gave (1.45, −0.78, +5.24) mm in
  `corenetic_socket_ee` (σ ≤ 0.35 mm, leave-one-run-out ≤ 0.36 mm), i.e. the
  socket at z −5.06 in `corenetic_gripper_ee`, not −10.3. A caliper reads
  flange face → socket centre 166.9 mm (the pivot says 166.93, −10.3 would be
  161.7), so the −10.3 had absorbed about 5 mm of the edge's error along the
  tool axis. `corenetic_socket_ee` is now (1.45, −0.78, −5.06) mm and the
  socket frame of later fits is ~0. The end-to-end replay error carries that
  5 mm again until the edge is fixed. Output:
  `outputs/fr3_tcp_pivot/joint/*_joint.json`.
- **The handheld UMI**, about the same ball, recorded as an ordinary episode.
  Production's labels, fitted as a pivot, give the ball `b` in the camera world.

Each edge's error at the ball is `T_world_base c − b`. Each side fits its own
lever arm, so a wrong `corenetic_socket_ee` or a wrong marker→TCP constant moves
that side's `s`, not the ball. The handheld `s` is a free reading of the
marker→TCP constant's error.

One ball position shows only the edge's translation there. Positions about
0.3 m apart also show its rotation (1° is about 5 mm at 0.3 m). With three or
more non-collinear positions, the pairs imply an edge of their own
(`ball_edges`): rigid, with the camera world's scale reported but not
absorbed. With four or more, each position is also predicted from the others
(leave one out); its verdict is `ok` when every prediction is within 6 mm
(as the single-tag gate). The handheld side fits one `s` for all positions of
a world by default (`--handheld-s shared`); the per-position fit is still
reported.

The panel's 存为候选边 (`fr3_base_ball_check.py edge --check CHECK --out
REPORT`) writes a passing ball edge as a `register_fr3_base.py` candidate in
`outputs/calibration/fr3_base_registration/ball_<check>.json`. The FR3-base
edge panel lists it with its leave-one-out numbers and how far it moves the
live edge at the balls, and swaps it in like any other candidate. The report
carries the `world_reference.json` hash the check ran against, so `apply`
refuses it once the reference changes. A later check scores it like the
other candidates, marked as fitted on balls (in-sample small).

The panel scores the live edge and every candidate report in
`outputs/calibration/fr3_base_registration`. It writes
`outputs/calibration/fr3_base_ball_check/check_*.json` and changes nothing else.

A handheld position is refused when:

- it has no world stamp;
- it has fewer than 60 solved frames;
- under 60 % of its frames lie within 6 mm of the fitted ball (the ball moved);
- its kept frames' residual p95 is over 6 mm.

The fit keeps one frame per orientation (the first one). A ball moved and
pivoted again through the same orientations is therefore invisible to the
residual, which is why the all-frames on-ball fraction is checked as well.

The handheld fit drops frames more than twice the kept frames' median
residual off the ball and refits until the kept set stops changing. At three
times the median, a 3 mm label scatter kept everything: on 2026-10-14 the
insert slid about 6 mm off the ball for 3–4 s of a 20 s pivot, and those
frames moved the ball 8 mm (σ 5 mm). At twice the median they go, and σ drops
below 1 mm. The FR3 side may be a `fit-joint` report with one shared centre
(several runs on the same ball), which halves the centre's σ.

#### Result, 2026-10-14: three positions

The ball was at (648, −182, 209), (641, 229, 211) and (424, −456, 214) mm in
fr3_base, 0.35–0.72 m apart, all at nearly one height
(`check_20261014T073919Z.json`).

- The FR3's free `s` agreed with `corenetic_socket_ee` to < 1 mm at all three
  positions, in three different arm postures. Since the socket frame is now
  the measured socket, the FR3 centre can also be taken with `s` held at 0:
  the mean seated socket position. At position 1 that came within 0.3 mm of
  the nine-run joint fit. With `s` held, the FR3 needs no large yaw.
- The live edge (10-09 single tag) was off by 12.0, 13.3 and 10.7 mm. In world
  axes the errors were (5.4, 10.3, 3.1), (6.3, 11.2, 3.5) and (−0.7, 10.7, 0.1);
  world y is about fr3_base −z. So most of the error is a vertical translation
  of about 10.7 mm, plus about 0.5° about a near-vertical axis.
- Ball-to-ball distances need no edge. The camera world read them 0.2–0.8 %
  shorter than the FR3's FK (713.4 vs 719.1 mm over the longest pair). A
  similarity fit gives a scale of −0.67 % with residuals of 1.1–2.3 mm. That
  agrees with the 10-13 station's −0.65 % against the laser. A rigid edge
  cannot absorb the scale: the three-ball rigid edge (`ball_edges`) still
  leaves 1.7–3.3 mm.
- Still open, with only three points, one height and no held-out position:
  replace the edge with the ball edge (rigid), use a similarity edge, or fix
  the camera world's scale at its source. Next, record two more positions, one
  at another height and one held out. The check does not yet turn `ball_edges`
  into a candidate report.

#### Result, 2026-10-14 (later): five positions

Two more positions were added (`check_20261014T084524Z.json`): position 4 at
(630, −211, 427) mm, about 215 mm higher than the first three, and position 5
at (476, 274, 435) mm, held out. The five centres span 631 × 240 × 191 mm
(singular values), so tilts about horizontal axes are now visible.

- Position 4: two FR3 runs. With `s` held at 0 their centres agree to 1.2 mm;
  their free fits differ by 5.6 mm because one run turned only 59° (weak
  direction 0.10). They were joined in one `fit-joint`
  (`20261014_162528_joint.json`). The handheld segment saw 3–5 cameras per
  frame and its centre held within 1 mm across the four quarters.
- Position 5: the first handheld segment solved only 66 % of its frames,
  mostly with two cameras, turning 68°. It was recorded again: 81 % solved,
  mostly 3–4 cameras, 95°, 97 % of frames on the ball. The two segments'
  centres are about 4 mm apart; the second one is used.
- Fitted per position, the handheld's TCP-to-socket `s` wandered by about
  10 mm (its z from −3.8 to +6.2 mm). The insert never changed, so that is fit
  error: the centre and `s` trade off along directions the orientations did
  not cover. Holding one `s` for all positions ((2.5, −2.9, 2.0) mm) and
  refitting only the centres was the largest single improvement. On the FR3
  side `s` is held at 0.
- Live edge: 12.0, 13.3, 10.7, 12.3 and 7.6 mm. At position 5 the vertical
  error is about 5 mm against about 10.7 mm elsewhere, so a translation alone
  does not explain it.

| Edge from the balls | Fit rms | Leave-one-out rms | Position 5 held out |
|---|---|---|---|
| Live (10-09 single tag) | — | — | 5.7 mm |
| Translation only | 4.9 mm | 6.1 mm | 8.9 mm |
| Rigid | 2.1 mm | 3.1 mm | 3.4 mm |
| Similarity (scale −0.44 %) | 1.6 mm | 3.0 mm | — |

"Held out" fits positions 1–4 (handheld `s` from those four only) and
predicts position 5. Dropping any one position turns the rigid edge by at
most 0.22°. The five-position rigid edge differs from the live one by 1.37°
and moves poses in the workspace by 5–14 mm, mostly vertically.

This agrees with earlier evidence. The 10-13 laser end-to-end saw only
4.6 mm because the old socket offset (−10.3 mm along the tool axis) had
absorbed about 5 mm of this edge's error. The scale (−0.44 to −0.60 %,
depending on how `s` is taken) matches the 10-13 station's −0.65 %. The rigid
edge's remaining 2–3 mm is about what that scale predicts at 0.3–0.6 m from
the centroid.

Recommendation: adopt the rigid ball edge, without scale. The base-to-world
transform is physically rigid; the scale belongs to the camera world and is
fixed there. Adopting it means swapping it in through the calibration page,
re-exporting fr3_base datasets (earlier exports keep the old edge and must
not be mixed with new ones), and refitting the edge from the same ball data
whenever the camera world changes.

Through the tool (`check_20261014T090257Z.json`, rejection as the per-position
fit: 3 mm floor, twice each position's median): shared `s` (2.4, −2.5, 2.1) mm,
residual rms 2.3 mm, leave one out 1.1 / 5.0 / 2.6 / 5.3 / 2.1 mm (rms 3.6),
scale −0.34 %, verdict `ok`. Saved as candidate
`ball_check_20261014T090257Z.json`, not swapped in yet: at the balls it moves
poses by 11.5 mm on average (14.0 max) and turns 1.14°. The analysis above
held the FR3 socket at `s` = 0 instead of the fits' own `s`, which gave 3.1 mm
leave one out; the tool takes the fit reports' centres as they are.

Swapped in on 2026-10-14 (`world_graph.json`, commit `042e5a5e`; the 10-09
single-tag edge is kept under `supersedes`). Through the new edge the five
handheld centres land 0.7 / 2.6 / 1.0 / 4.1 / 1.6 mm from the FR3's (old edge
13.5 / 14.4 / 12.8 / 10.6 / 6.8 mm). Those are the fitting points, so they
only show the swap went in as solved.

Laser acceptance, same day: `thor_gmsl2_10ch_v1_20261013_110115` episode 0
exported through the new edge and replayed on the FR3 with the laser
(`real_replay/20261014T092154Z_episode_000000_right`; 1060 frames, laser vs
FK p95 0.71 mm). `register_fr3_base.py laser` with the 10-13 station composes
an edge that sees neither the balls nor the handheld labels. Over the replay's
354 workspace points (94 × 186 × 107 mm):

| Laser edge vs | Mean | Max | Mean offset (base, mm) | Rotation |
|---|---|---|---|---|
| Ball edge (new) | 5.9 mm | 6.4 mm | (−4.9, −0.3, −3.4) | 0.54° |
| 10-09 single tag (old) | 9.7 mm | 10.3 mm | (0.3, −4.2, +8.7) | 1.20° |

- The old edge's offset against the laser is mostly vertical, as the ball
  check found: two independent paths point at the same error.
- The 5.9 mm left is what the laser path itself carries: the station has rms
  3.4 mm, translation σ 4.9 mm and a camera-vs-laser scale of −0.65 %
  (`register_fr3_base.py` rejects it as an edge for rms > 1.5 mm). Combined
  with the ball edge's 3.6 mm leave one out, about 6 mm is expected.
- The tracker did not move: the edges from today's replay and the two 10-13
  replays of the same episode differ by 1.1 mm and 0.10°.
- Replay tracking (command vs actual) was 1.9 mm mean, 3.6 mm p95, as with the
  old edge (2.1 / 3.8 mm); it is lag, not the edge.

So the laser confirms the swap at the 5–6 mm level and cannot test it at
3 mm. That needs the camera world's scale fixed at its source (the station's
−0.65 % and the balls' −0.34 % both say the camera world is small), then the
ball check rerun and both the edge and the station refitted.
