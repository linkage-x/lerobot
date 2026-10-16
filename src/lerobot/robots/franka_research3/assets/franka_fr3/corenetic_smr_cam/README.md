# corenetic SMR/camera adaptor meshes

Used by `fr3_corenetic_gripper.urdf` / `.xml` (links `link_corenetic_mount`, `link_corenetic_smr_cam`,
`link_corenetic_smr`). The adaptor replaces the old `link_franka_corenetic_gripper_3` mount (removed on the
robot; its STL is in the vendor package `gripper_w_v2_franka_description` outside the repo, byte-identical to
the one the registration below was done against).

- Units: metres. Frame: **every mesh is in `base_link`** (fixed joints with identity origin), so the
  STLs can be overlaid directly on the gripper meshes.
- Source: the Onshape assembly, assembled offline in FreeCAD (2026-10-10). The adaptor pose comes from
  ICP of `Part 1` onto its twin plate in the old link_3 mesh (RMS 0.26 mm, p95 0.45 mm). Part 1 is
  mounted upside down in Onshape; the seat was found by voxel search. The pose was checked against
  bench photos.
- `mount.STL` = Part 1 (the black bracket) + the two 33x7x5 mm spacers kept from the old link_3.
- The SMR is a 1.5" sphere primitive at the CAD ball centre, base_link (70.201, -0.249, 103.891) mm.
  A laser-tracker fit differs by 4.15 mm (mostly lateral), so this centre is nominal, not a calibration.

Inertials (in the URDF/MJCF):

| link | mass | basis |
|---|---|---|
| `link_corenetic_mount` | 46.5 g | **not weighed**: CAD volume x aluminium 2.70 g/cm3 |
| `link_corenetic_smr_cam` | 187.1 g | weighed per group (camera 53.6, steel plate 35.7, nest 53.4, base+case+pin 44.4 g), spread uniformly over each group's CAD volume |
| `link_corenetic_smr` | 140 g | weighed; inertia of a solid sphere |

Regenerate (outside the repo, `~/cad/corenetic_smr_cam/`): `./fcrun.sh mass_props.py`, then
`./fcrun.sh export_urdf_meshes.py`.
