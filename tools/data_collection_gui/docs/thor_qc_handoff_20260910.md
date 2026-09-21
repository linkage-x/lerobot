# Thor QC diagnosis and recovered context — 2026-09-10

## Recovered work

- Main checkout: `6b742239` (tracking-target GUI and detection evidence).
- Tracking submodule: `bbc47c8` (explicit `RigTrackingStrategy`, default cube
  strategy, and per-instance `HybridCarrierRigStrategy`).
- The requested independent detector repository already exists with its own Git
  history, assets, PDF, examples, method and publication assessment documents.
  Its working tree includes later detector/render/validation edits; this session
  did not modify them. Publication readiness remains an open research question,
  not an established result.

## Dataset evidence (read-only SSH inspection)

Dataset: `nvidia@192.168.111.122:/home/nvidia/lerobot/outputs/datasets/thor_gmsl2_10ch_v1_20260909_141151`.

| Episode | Online-sync frames per camera | Parquet rows | Valid BOX snapshots | Raw BOX file |
|---|---:|---:|---:|---|
| 0 | 1205 | 0 | 0/40 | absent |
| 1 | 1203 | 1201 | 40/40 | present |
| 2 | 1206 | 1202 | 40/40 | present |

Episode 0 reports `no cached sensor data`. Camera synchronization passed, but
there were no recorded BOX observations from which to write sensor rows. Camera
synchronization and sensor availability are separate checks. The missing BOX
observations cannot be recovered from these files; do not fill them with zeros
or reuse another episode's samples.

The existing trajectory summary dated `2026-09-09T12:12:31Z` lists only episodes
1 and 2 for all seven tracking cameras, with frame limits `[1201, 1202]`.
The carrier sidecar contains 2403 rows with those same episode IDs. This confirms
that the recorded inputs of that run excluded episode 0; it is not an independent
validation of pose accuracy or all timestamp alignment.

## Local correction

The local tracker already pairs raw videos to parquet episodes by ID and omits
video-only episodes. The GUI QC still described the old positional pairing and
incorrectly claimed that all EE generation was impossible.

- Video-only episode: `warn`, explicitly disclose that it is skipped and missing
  sensor observations are not reconstructed.
- Parquet episode without video: retain `fail`, including when extra video-only
  episodes also exist.
- No videos, parquet, or dataset metadata were changed by this fix.

Validation: 12 QC tests passed with `PYTEST_DISABLE_PLUGIN_AUTOLOAD=1` in the local
`.venv`. Coverage includes initial/trailing video-only episodes, matched episodes,
and missing videos with/without an additional video-only episode. The default
pytest invocation encountered an unrelated ROS plugin dependency (`lark`).
A direct local integration check also passed using the existing scientific Python
installation: 2403 rows across episodes 1/2, episode 0 exclusion, correct global
row boundary at index 1201, and rejection when a required video is missing.
The standalone tracker pytest file could not collect in `.venv` because SciPy
was absent; the direct check exercised the production functions without installing
or modifying dependencies.

## Thor deployment boundary

The user stated that a colleague is using Thor. This session performed only
read-only SSH inspection: no remote writes, service restarts, deployments, or
trajectory jobs. The colleague's workflow must remain uninterrupted.

At inspection time, Thor's on-disk tracker lacked
`align_video_streams_to_parquet_episodes`; its gateway file also differed from the
local checkout and did not contain the pairing check that produced the saved QC
log. A gateway process restart was observed but was not initiated by this session.
The current deployment cannot be inferred from that older saved processing log.

When Thor is available for coordinated maintenance, first compare its then-current
files and backend with the local checkout. Deploy the matching tracker and QC
behavior together, preserving unrelated remote changes. Do not replace the whole
remote gateway with this local file blindly. Re-run QC afterward; episode 0 should
remain visibly incomplete (`warn`), while the two parquet episodes can be tracked
by ID. There is no need to delete or renumber the raw episode directories.
