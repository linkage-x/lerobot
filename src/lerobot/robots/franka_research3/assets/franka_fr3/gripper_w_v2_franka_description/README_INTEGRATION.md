# Corenetic gripper V2 URDF integration

Source archive: `/Users/yanzeyu/Downloads/gripper_w_v2_franka_description.zip`

SHA-256: `6c03f2266e5b66cb9d55875c49b58189c78c751d8aac17ffbdd51ed9aea35ee3`

The vendor URDF is retained unchanged under `urdf/`. It is a gripper component,
not a complete Franka model, and cannot be loaded directly because it contains
two joints whose child links are absent and references one collision mesh that
is not included in the archive.

Run the following commands from the repository root to rebuild the integrated
models:

```bash
.venv/bin/python \
  src/lerobot/robots/franka_research3/assets/franka_fr3/build_fr3_corenetic_gripper_v2_urdf.py

.venv/bin/python \
  src/lerobot/robots/franka_research3/assets/franka_fr3/build_dual_fr3_urdf.py \
  --single-arm-urdf \
  src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_corenetic_gripper_v2.urdf \
  --output \
  src/lerobot/robots/franka_research3/assets/franka_fr3/dual_fr3_corenetic_gripper_v2_p1_p2.urdf
```

Generated files:

- `../fr3_corenetic_gripper_v2.urdf`
- `../dual_fr3_corenetic_gripper_v2_p1_p2.urdf`

The integration builder omits the two stale joints, redirects the missing
gripper-base collision path to the supplied visual mesh, supplies collision
geometry for moving gripper links where it is absent, and repairs placeholder
inertia values rejected by MuJoCo. It also adds the compatibility end-effector
link `corenetic_gripper_ee`.
