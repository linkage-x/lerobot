---
title: 'Nintendo FR3 Teleop'
type: 'feature'
created: '2026-05-09'
status: 'done'
context: []
---

<frozen-after-approval reason="human-owned intent -- do not modify unless human renegotiates">

## Intent

**Problem:** FR3 MuJoCo teleoperation and recording currently support SpaceMouse and Quest3 runtime teleoperators, but not the Nintendo Joy-Con / Pro Controller setup validated by `/home/hanyu/Codes/joycon-robotics/script/ubuntu_nintendo_controller_test.py`.

**Approach:** Add a LeRobot `nintendo` teleoperator that reads Nintendo HID reports with the same report parsing as the Ubuntu smoke test, emits the existing FR3 `enabled + target_* + gripper` action contract, and wire it into `tools/fr3/fr3_mujoco_teleop.py` / `tools/fr3/fr3_mujoco_record.py` through the shared runtime parser.

## Boundaries & Constraints

**Always:** Keep the action contract compatible with `FR3MujocoEnv.step_teleop_action`; use a clutch/enable button model similar to Quest3 controller mode; keep HID access optional so imports and tests work without a connected controller.

**Ask First:** Any change to physical robot control semantics, dataset schema, or destructive changes to existing SpaceMouse/Quest3 behavior.

**Never:** Do not depend on the external `joycon-robotics` repo being importable at runtime; do not replace existing SpaceMouse/Quest3 paths; do not require real hardware for unit tests.

## I/O & Edge-Case Matrix

| Scenario | Input / State | Expected Output / Behavior | Error Handling |
|----------|--------------|---------------------------|----------------|
| Pro Controller active | R/L clutch held, sticks moved | Emits `enabled=True`, scaled `target_x/y/z/wz`, filtered gripper | N/A |
| Clutch released | Sticks moved, no clutch button | Emits `enabled=False`, zero motion, gripper still updates | N/A |
| No HID package/device | User selects `nintendo` without `hidapi` or controller | Clear ImportError/ConnectionError with setup guidance | Raise before teleop loop |

</frozen-after-approval>

## Code Map

- `src/lerobot/teleoperators/nintendo/*` -- new Nintendo HID backend, config, and FR3 action teleoperator.
- `src/lerobot/teleoperators/utils.py` -- teleoperator factory registration.
- `tools/fr3/fr3_mujoco_runtime.py` -- argparse runtime support shared by teleop and record scripts.
- `tools/fr3/fr3_mujoco_teleop.py` / `tools/fr3/fr3_mujoco_record.py` -- user-facing descriptions/log metadata.
- `tests/teleoperators/test_nintendo.py` -- hardware-free mapping and behavior tests.

## Tasks & Acceptance

**Execution:**
- [x] `src/lerobot/teleoperators/nintendo/` -- add config/backend/teleop using Nintendo HID report parsing.
- [x] `src/lerobot/teleoperators/utils.py` and scripts -- register the new teleoperator type.
- [x] `tools/fr3/fr3_mujoco_runtime.py` -- add `--teleop-type nintendo` and Nintendo runtime flags.
- [x] `tests/teleoperators/test_nintendo.py` -- cover clutch, stick mapping, gripper, stale input, and factory creation.

**Acceptance Criteria:**
- Given `--teleop-type nintendo`, when building runtime teleop config, then the config type is `nintendo` and `make_teleoperator_from_config` returns `NintendoTeleop`.
- Given a fake Nintendo Pro Controller reading with clutch held, when `get_action()` runs, then it emits nonzero FR3 target fields and preserves the normalized gripper contract.
- Given no clutch button is held, when sticks are moved, then motion is disabled but gripper button input can still update.

## Verification

**Commands:**
- `pytest tests/teleoperators/test_nintendo.py` -- expected: all pass without physical HID hardware.
- `python -m compileall src/lerobot/teleoperators/nintendo tools/fr3/fr3_mujoco_runtime.py tools/fr3/fr3_mujoco_teleop.py tools/fr3/fr3_mujoco_record.py` -- expected: no syntax errors.
