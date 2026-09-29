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

## Correcting a stamp that is wrong

The stamp records what this file said at record time. If the rig moved and the
new world was not committed first, the stamp is wrong and cannot be recomputed
from the episode. `python -m tools.thor.gmsl2.restamp_world <dataset roots…>
--expect-from <id now> --to <true id> --reason "<evidence>"` records the
decision: dry run by default (`--apply` to write), the target must be a node of
`world_graph.json`, only episodes carrying `--expect-from` are touched, and the
original block is kept under `world_frame.restamp.original`. The real fix is to
commit a new island here *before* recording in it.
