# Data Collection GUI Replay Safety Review - 2026-05-18

## Goal

Track review findings and implementation progress for Episode Replay / Replay Controls safety before real-robot replay.

## Progress

| Priority | Finding | Status | Notes |
| --- | --- | --- | --- |
| P0 | Preflight must perform real hardware checks, not only MuJoCo gate state changes. | Done | `Preflight` now runs host-side FR3 ping/arm checks via `fr3_record_preflight.py` before setting `safety=ready`. Can be disabled only with `replay.real_preflight_enabled=false` for tests/dev. |
| P0 | MuJoCo gate must cover more real-robot trajectory risks than pose replay error alone. | Done | Added trajectory contract checks for frame count, pose availability, EE step, gripper range/step, and optional Z bounds. |
| P1 | Frontend mock fallback must not fabricate safety-critical replay success. | Done | Gateway-unavailable MuJoCo, Preflight, Dry Run, and Real Robot paths now fail closed in the frontend. |
| P1 | MuJoCo validation should be persisted and auditable. | Done | Validation records are persisted under dataset `meta/gui_replay_validations.json` and restored only when dataset/episode/fps/thresholds match. |
| P1 | MuJoCo pass must require structured machine-readable result output. | Done | `mujoco_replay_result=...` is required; legacy summary lines can populate metrics but cannot pass gate alone. |
| P2 | Frontend path matching can disagree with backend path resolution. | Done | Frontend now trusts backend `isCurrentForSelection` instead of doing path string equality. |
| P2 | Dataset processing actions must not fail silently when not implemented. | Done | `Generate EE Trajectory` now returns explicit `501 Not Implemented` from the gateway and the frontend shows a `待实现` alert for the selected dataset. |
| P2 | Real replay confirmation should show robot IP, gripper port, dataset, episode, fps, and OTG settings. | Pending | Add a command summary to snapshot or replay status. |
| P2 | Tests should cover process lifecycle and HTTP endpoints, not only helper functions. | Partial | Added structured-result and trajectory-contract unit coverage. Still need mocked process/HTTP endpoint lifecycle tests. |
| P3 | Real replay should expose structured progress and error metrics in GUI. | Pending | Add `real_replay_result` and progress parsing after safety gate is stable. |
| P3 | Gate thresholds should be visible in Replay Controls. | Done | Replay Controls now display current max error metrics with thresholds. |

## Latest Verification

- `PYTHONPATH=src:. pytest tests/scripts/test_data_collection_gui_gateway.py`: 12 passed.
- `npm run build` in `tools/data_collection_gui/frontend`: passed with existing Vite chunk-size warning.
- `PYTHONPATH=src:. python -m py_compile tools/data_collection_gui/gateway.py`: passed.

## Policy Update - 2026-05-18

- MuJoCo replay is a strong recommendation before `Preflight` and `Dry Run`, not a hard requirement.
- `Preflight` may run without a current MuJoCo validation, but still fails closed on hardware preflight errors.
- `Dry Run` may run without a current MuJoCo validation and shows a recommendation message.
- `Real Robot` keeps the hard requirement for current passed MuJoCo validation.

## Acceptance Notes

- Real Robot must remain disabled unless backend says the selected dataset/episode/fps has a current passed MuJoCo validation.
- Preflight must fail closed when hardware checks fail or time out.
- MuJoCo replay exit code 0 is not enough; the structured result line and trajectory contract must also pass.
- Gateway-unavailable frontend paths must not show fake safety success for replay controls.
