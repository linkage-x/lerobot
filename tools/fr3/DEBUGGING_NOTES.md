# FR3 MuJoCo Teleop Debugging Notes

## 2026-04-22: SpaceMouse Rotation Not Working

### Symptom
Running `fr3_mujoco_teleop.py --enable-rotation --rotation-scale=100` had no visible effect on end-effector rotation in MuJoCo viewer.

### Root Cause Analysis

**Layer 1: Two Independent Enable Mechanisms**

1. **SpaceMouse side** (`--enable-rotation` / `enable_rotation` in `SpaceMouseTeleopConfig`):
   - Controls whether the SpaceMouse device sends rotation delta values (`target_wx/y/z`) in its action dict.
   - Even when enabled, the actual rotation magnitude is scaled by `rotation_scale_vector`.

2. **IK Solver side** (`lock_orientation` in `FR3MujocoEnvConfig`):
   - Was **hardcoded to `True`** in `fr3_mujoco.py:823`, ignoring any external setting.

```python
# fr3_mujoco.py:823 - BEFORE (hardcoded lock_orientation=True)
target_joints = self._kinematics.inverse_kinematics(current_joints, desired_pose, lock_orientation=True)

# AFTER (uses config)
target_joints = self._kinematics.inverse_kinematics(current_joints, desired_pose, lock_orientation=self.cfg.lock_orientation)
```

**Layer 2: `lock_orientation` Semantics Are Not Binary**

After fixing Layer 1, rotation still didn't work. The `lock_orientation` parameter in `fr3_mujoco.py:179` is **NOT a binary lock**:

```python
resolved_orientation_weight = 1.0 if lock_orientation else 0.01
```

When `lock_orientation=False`, orientation weight is **0.01**, not 0. This means:
- Orientation is still optimized, but with 1/100th the weight of position
- In practice, the IK solver prioritizes position over orientation ~100:1
- Result: rotation commands are effectively ignored

### Solution
To enable end-effector rotation during teleop, BOTH conditions must be satisfied:

```bash
python tools/fr3/fr3_mujoco_teleop.py --enable-rotation --unlock-orientation
```

- `--enable-rotation`: SpaceMouse sends rotation commands
- `--unlock-orientation`: IK solver accepts rotation commands (new flag)

**Remaining Issue**: Even with both flags, the orientation weight of 0.01 is too low for practical use. A proper fix would require either:
1. Setting a higher orientation weight (e.g., 0.5-1.0) when unlocked
2. Adding a separate `orientation_weight` config parameter

### Lesson Learned
1. When a feature has two independent enable signals (source + destination), both must be enabled.
2. A parameter named "lock" that uses a small non-zero weight is not a true binary lock - the name is misleading about actual behavior.
