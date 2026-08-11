# UMI TCP Calibration Fixture Design

This document defines the fixtures for calibrating `cube -> TCP` under the
BOX UMI roadmap. The main requirement is not just fitting a model; the fixture
must support an independent residual/holdout test that can justify the final
TCP uncertainty written to the sidecar.

## First-Principles Choice

The final target is `<= 3 mm` TCP translation error, with static jitter below
`0.5-1 mm`. The camera rig is already below this scale, so the fixture must not
introduce millimeter-scale unobserved bias.

A ball-head rod with a measured radius `r` gives only a sphere constraint:

```text
|| T_world_cube_i * p_cube_tcp - c_world || = r
```

If `r` is only known to `1-2 mm`, that single prior consumes a large part of the
3 mm budget and can be absorbed by the fitted ball center and TCP position. It
is useful as a sanity check, not as the main accuracy certification path.

The main path is fixed-point pivot, but the flat gripper pads should not clamp
the fixed ball directly. The preferred design is a gripper-held calibration
insert with a spherical/conical socket, mated to a table-fixed precision ball:

```text
T_world_cube_i * p_cube_pivot ~= p_world_pivot
```

The pivot point is unknown, but it is fixed. The ball/socket pair provides the
rotation joint; the flat gripper pads only locate and clamp the insert. This
gives a 3D point constraint per frame without forcing a steel ball to slide
against two flat finger pads.

## Fixture Set

CAD generator:

```bash
python third_party/opencv_kalibr/metrology/fixtures/cad/umi_tcp_calibration_fixtures.py
```

Generated files:

```text
outputs/metrology/fixtures/umi_tcp_calibration/
  pivot_socket_insert_45mm_opening_13p85mm_ball_centered_compact.step # next print
  pivot_socket_insert_45mm_opening_13p85mm_ball_centered_compact.stl
  pivot_socket_insert_reference_assembly_13p85mm_ball_centered_compact.step
  pivot_socket_insert_45mm_opening_13p85mm_ball.step # previous field insert
  pivot_socket_insert_45mm_opening_13p85mm_ball.stl
  pivot_socket_insert_reference_assembly_13p85mm_ball.step
  fixed_ball_post_13p85mm_ball.step
  pivot_socket_insert_45mm_opening_25mm_ball.step     # larger-ball reference insert
  pivot_socket_insert_45mm_opening_25mm_ball.stl
  pivot_socket_insert_reference_assembly_25mm_ball.step
  fixed_ball_post_25mm_ball.step
  pivot_socket_insert_30mm_opening_12mm_ball.step     # small exploratory version
  fixed_ball_post_12mm_ball.step
  legacy_pivot_holder_m8_25mm_ball.step
  legacy_pivot_reference_assembly_25mm_ball.step
  opening_spacer_30mm.step
  opening_spacer_60mm.step
  opening_spacer_100mm.step
  rod_stop_collar_8mm.step
```

`pivot_socket_insert_reference_assembly` is for visual review only. The real
table datum should be a purchased hardened steel ball, ball-ended locating pin,
or equivalent precision part.

The old `legacy_pivot_holder_m8_25mm_ball` files are retained for comparison,
but direct flat-pad clamping of a fixed ball is not the preferred path because
it tends to lock, slide, or shift the effective contact point during rotation.

## Main Fixture A: Socket Insert + Fixed Ball Pivot

Purpose:

```text
Calibrate cube -> calibration-insert pivot, then map that pivot to TCP_closed
using the insert's known CAD offset and repeatability benchmark.
```

Mechanical design:

- Gripper-held insert with two flat +/-Y clamp faces and shoulder ribs that set
  repeatable axial position in the flat finger pads. For the current 13.85 mm ball insert,
  the shoulder inner gap is 28.8 mm for a measured 28.4 mm gripper-finger width plus one 0.4 mm nozzle-width allowance.
- Front-facing spherical/conical socket; the intended seated datum ball center
  is the insert CAD origin.
- The next 13.85 mm field variant is `*_centered_compact`: it keeps 45 mm gripper opening, places the socket/ball center at the X midpoint between the two shoulder ribs, and removes the earlier 7.465 mm CAD offset.
- This revision removes redundant material outside the shoulder outer faces: shoulder ribs no longer extend beyond the 45 mm clamp width, and the body length is trimmed to the shoulder outer faces while keeping the material needed around the ball pocket.
- The previous 13.85 mm variant is retained for comparison. A 25 mm variant is retained because a larger ball gives more angular clearance if the hardware allows it; a 12 mm variant exists only for small-clearance trials.
- Table-side fixed ball post should preferably use a purchased hardened steel ball or ball-ended pin. The exported `fixed_ball_post_*` STL/STEP is a printable fit-check body; for final metrology, use a metal datum or prove the printed datum by holdout residual.
- Rotation occurs at the ball/socket interface, not at the flat finger pads.
- The four lobes visible around the socket mouth are retention/guide lips left by the front and vertical clearance cuts. They are not support feet; if they hit the steel ball or limit the required orientation sweep, increase `fork_gap_z` / `lead_in_radius` or trim the lips.

Recommended BOM:

- 1x CNC or high-quality printed insert for first trials; CNC preferred after
  geometry is accepted.
- 1x 13.85 mm hardened steel ball or ball-ended locating pin on a rigid post for the current field fixture. If clearance permits, the 25 mm version remains mechanically better because it raises the pivot, opens the socket geometry, and gives more angular clearance after the gripper clamps the insert.
- 1x aluminum/steel table base, bolted or clamped rigidly.
- Optional dowel pins or a locating fence for repeatable fixture remounting.

Use:

1. Clamp the insert between the UMI finger pads at the insert's nominal opening
   (`45 mm` for the current 13.85 mm and 25 mm ball inserts). Prefer the `centered_compact` 13.85 mm insert for the next print. It uses a 28.8 mm shoulder inner gap for the measured 28.4 mm finger width plus one 0.4 mm nozzle-width allowance, with the socket center centered between the two shoulder ribs.
   The shoulder ribs must contact the finger-pad edges so the insert cannot slide along the pads.
2. Seat the insert socket on the matching fixed 13.85 mm table ball for the current field fixture. Clamp force should be
   just enough to retain the insert; do not preload hard into the socket.
3. Move the UMI through 100-300 frames over 20+ diverse orientations while the
   socket stays seated on the fixed ball.
4. Split into fit/holdout sets and solve fixed-point pivot for the seated ball
   center in cube coordinates.
5. For the `centered_compact` insert, the pivot result is the seated ball/socket center. Map it to `TCP_closed` using the measured/CAD insert-to-TCP offset if required; do not assume the supplier opening retreat is a CAD offset in this part.
   Validate by repeated remove/reclamp runs.

Acceptance:

```text
holdout pivot residual p95 <= 1 mm: can discuss sub-mm/1 mm nominal TCP
holdout pivot residual p95 <= 3 mm: usable for final 3 mm GT budget
holdout pivot residual p95 > 3 mm: fixture/contact/mechanics are not acceptable
```

## Auxiliary Fixture B: Opening Spacers

Purpose:

```text
Fit sparse correction of contact_tcp(g) over gripper opening.
```

The generator creates dumbbell-style spacers:

- Central cylinder diameter = nominal opening.
- End flanges prevent the flat finger pads from sliding along the spacer axis.
- Default diameters: 30, 60, 100 mm.

These spacers do not replace pivot. They validate the mapping from reported gripper opening to true fingertip distance `d`, then the supplier formula provides the TCP retreat:

```text
z_mm = 49.8361 - sqrt(49.8361^2 - 4.0456*d_mm - d_mm^2/4)
contact_tcp(d) = TCP_closed + retreat_axis * z(d) + residual(d)
```

Use:

1. Select 3-5 openings, prioritizing real task openings.
2. For each spacer, lightly clamp the central cylinder at the flanged section.
3. Record the actual `opening_m` reported by the gripper.
4. Collect repeated samples. Fit only the reported-opening-to-true-distance conversion and a low-dimensional residual if the supplier curve leaves systematic error.

Do not fit a free 3D transform per opening unless you have enough independent
validation data; it will overfit the fixture.

## Auxiliary Fixture C: Ball-Head Rod Sanity Check

Purpose:

```text
Check opening direction / rough magnitude only.
```

Physical setup:

- Photography ball head fixed to the table.
- Rigid rod B fixed to the ball head.
- TCP lies on rod B axis.
- Stop collars constrain the axial clamp location.

Constraint:

```text
|| T_world_cube_i * p_cube_tcp - c_world || = r
```

This is a sphere constraint and is weaker than pivot. With `r` known only to
`1-2 mm`, it should not certify final TCP accuracy. Use it to verify statements
like:

```text
d ~= 78.7 mm -> contact TCP retreats about 25 mm by the supplier z(d) curve
```

## CAD Notes

The current CAD is deliberately simple and conservative:

- `build123d` is the source of truth.
- STEP is for CAM/review.
- STL is for quick printing/review.
- Precision comes from metal datum parts and residual validation, not printed
  plastic.

If converting the pivot holder into a CNC part, add:

- Dowel pin holes for repeatable table registration.
- A proper threaded or conical ball-seat feature matching the purchased ball pin.
- A CNC socket insert with inspected clamp-face width and socket location. Start from the 25 mm ball insert unless the gripper workspace forces a smaller ball.
- Engraved fixture ID, insert orientation, and datum direction.

