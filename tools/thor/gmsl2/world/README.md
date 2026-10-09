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
For the box labels that is `link_lt_gripper_tcp` (corenetic URDF), and its
origin is the insert-v2 socket centre, not that link's origin. The offset
between the two is still unrecorded (see the 0929 marker->TCP bundle).

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
