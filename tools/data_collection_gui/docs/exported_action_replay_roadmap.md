# Exported Dataset Action Replay Roadmap

Updated: 2026-07-14

This roadmap tracks the path from Thor exported datasets to GUI visualization,
MuJoCo validation, and eventually guarded real-robot replay when `action` is
or will be derived from `observation.state`.

## Target Policy

For exported datasets, deriving commands from observations is acceptable only as
an explicit derived-action mode:

```text
observation.state[i + 1]
  -> derived action target
  -> IK
  -> OTG
  -> MuJoCo validation
  -> optional Real Robot unlock
```

Real-robot replay should be unlocked only when all of these are true:

- the dataset declares the action semantics explicitly;
- the command stream used by the replay runtime is the same stream validated by MuJoCo;
- the selected dataset / episode / fps / thresholds have a current passed MuJoCo validation;
- trajectory contract checks pass on the command stream itself, not only on display poses;
- preflight hardware checks pass.

## Current Implementation

| Area | Status | Current behavior / evidence |
| --- | --- | --- |
| Raw recording writer action semantics | Done | `tools/thor/gmsl2/thor_lerobot_v3.py` writes BOX `observation.state` only; `action` is left for export. |
| Exported main `action` derivation | Done | `tools/thor/gmsl2/export_v3.py` derives `action[i] = observation.state[i + 1]`, holding the final state. |
| Exported EE pose sidecar columns | Done | `export_v3.py` can include `observation.ee_pose.{cube}.base` and `action.ee_pose.{cube}.base` from `derived/april_cube_tracking_in_robot_base/state_action.*.csv`. |
| Export dataset visibility in GUI Replay | Done | Gateway scans `outputs/exports`, returns `datasetKind=exported`, and Replay lists exported datasets. |
| Export page open-in-replay entry | Done | Dataset Export shows `Open Replay` when `datasetExport.outputPath` is available. |
| MuJoCo validation infrastructure | Done | Gateway starts MuJoCo replay, requires structured `mujoco_replay_result`, stores metrics, persists validation, and checks current dataset/episode/fps/thresholds. |
| Basic trajectory contract | Done | Gateway checks EE pose presence, max EE step, gripper range/step, and optional Z bounds in `_trajectory_contract_for_episode`. |
| Real hardware preflight | Done | `Preflight` runs host-side hardware checks before `safety=ready`. |
| FR3 real replay default path | Done | `fr3_das_replay_real_runtime.py` reads LeRobot main `action` and sends it through the validated `action[t] + OTG` path; joint-target CSV replay remains experimental. |
| Exported real-robot replay gate | Conservative Done | Current gateway and frontend still block all `datasetKind=exported` real-robot replay, with a tooltip explaining that exported action is derived. |

## Important Gaps

### 1. Main `action` is not yet a canonical EE pose command for Thor exports

Current Thor export writes the main LeRobot `action` column with the same width
and names as BOX `observation.state`:

```text
observation.state: BOX state vector
main action: next BOX state vector
```

The FR3 MuJoCo and real replay runtimes expect the main `action` column to begin
with an EE pose:

```text
action[:7] = [x, y, z, qx, qy, qz, qw]
action[7]  = optional gripper command
```

Thor export may also contain `action.ee_pose.*` columns, but the existing FR3
replay runtimes do not consume those columns as the canonical command stream.
Before exported Thor datasets can drive real replay, we must choose and implement
one canonical command source:

- Option A: write main `action` as the selected EE pose command stream for replay;
- Option B: teach replay runtimes and GUI validation to select an explicit
  `action.ee_pose.<target>.base` column;
- Option C: create a replay-ready derived dataset variant whose main `action`
  is EE pose while BOX state remains in `observation.state` / auxiliary columns.

### 2. Action semantics metadata is missing

There is no durable metadata that says whether `action` is:

- recorded real command;
- derived from `observation.state[i + 1]`;
- EE pose command;
- joint command;
- placeholder / BOX next-state vector.

Needed metadata example:

```json
{
  "action_semantics": {
    "kind": "derived_next_observation_state",
    "source": "observation.state[i+1]",
    "command_space": "ee_pose_7d_gripper",
    "canonical_column": "action",
    "requires_mujoco_validation": true,
    "real_robot_unlock_policy": "mujoco_pass_and_preflight"
  }
}
```

Candidate locations: `meta/info.json` for loader-visible schema metadata, and
`meta/export_sources.json` for provenance. Prefer `meta/info.json` for gating.

### 3. Current trajectory contract validates display EE pose, not necessarily command action

`gateway._ee_pose_from_row` prefers `observation.state` EE axes and only falls
back to `action`. Therefore `_trajectory_contract_for_episode` can pass based on
Replay timeline display poses even if the actual command stream consumed by
MuJoCo/real replay is different.

Required follow-up:

- add an action-stream-specific contract that reads the canonical command column;
- check finite values, shape, names, quaternion norm, gripper range, max position
  step, max rotation step, optional workspace/Z bounds;
- ensure the contract uses the same column that the replay runtime will execute.

### 4. Exported real-robot gate is still too conservative for the target policy

Current behavior: all exported datasets are blocked from `Real Robot` replay.

Target behavior:

```text
exported + no compatible action semantics:
  Real Robot disabled

exported + derived action semantics + no current MuJoCo pass:
  Real Robot disabled, tooltip asks user to run MuJoCo validation

exported + derived action semantics + current MuJoCo pass + preflight pass:
  Real Robot enabled with an explicit derived-action confirmation
```

This requires changing both frontend gating and backend `_require_mujoco_validation`.
The backend must remain authoritative.

## TODO Roadmap

### P0: Make action semantics explicit

- Add `action_semantics` metadata during `export_v3.py` finalization.
- Include whether main `action` is replay-ready or only a training/next-state label.
- Add tests that exported datasets declare the expected semantics.

### P0: Pick the canonical command stream for Thor exported replay

- Decide whether replay consumes main `action` or `action.ee_pose.<target>.base`.
- If using `action.ee_pose.*`, add a GUI/config selector for target stream
  (`left`, `right`, `head`, or task-specific default).
- If using main `action`, export a replay-ready action vector with EE pose names
  and move BOX next-state labels to an auxiliary column.

### P0: Validate command action directly

- Implement `_action_contract_for_episode(...)` against the canonical command stream.
- Include shape/names, finite values, quaternion norm, max position step, max
  rotation step, gripper range/step, workspace/Z bounds.
- Make MuJoCo validation fail if the action contract fails.
- Persist the action contract result inside `mujocoValidation.trajectoryContract`
  or a new `actionContract` field.

### P1: Allow exported real replay after validation

- Replace the hard exported-dataset block in `_require_mujoco_validation` with
  a semantics-aware gate.
- Frontend `Real Robot` button tooltip should say why it is locked and how to
  unlock it, instead of saying exported datasets are always forbidden.
- Real replay confirmation should include: dataset path, episode, fps, action
  semantics, canonical command column, MuJoCo max error, robot IP, gripper port,
  and OTG settings.

### P1: Keep MuJoCo and real replay consuming the same command stream

- Ensure `_mujoco_replay_command` and `_real_replay_command` pass the same
  action-source configuration.
- Add tests proving MuJoCo validation cannot unlock Real Robot for a different
  action column, episode, fps, or threshold set.

### P2: Improve audit artifacts

- Store action semantics and validation contract results in a human-readable
  validation report under `meta/` or `outputs/analysis/`.
- Surface the action semantics badge in Replay Inspector.
- Add a one-click "why locked" diagnostic panel for Real Robot replay.

## Acceptance Checklist

A Thor exported dataset may unlock Real Robot only when all checks below pass:

- `meta/info.json` declares compatible `action_semantics`.
- The canonical action stream is EE pose + optional gripper, not BOX next-state.
- MuJoCo replay completed with structured metrics and passed thresholds.
- Action contract passed on the exact command stream consumed by replay.
- Persisted validation is current for dataset root, episode, fps, thresholds,
  and action semantics hash/config.
- Preflight passed immediately before hardware replay.
- The operator confirms the derived-action replay warning.

## Current Decision

Keep the current conservative block for exported real-robot replay until P0 is
complete. Derived actions are a valid direction, but the project must first make
which action stream is executed explicit and verify that exact stream.
