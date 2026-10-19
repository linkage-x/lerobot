/** The heights a scene reset works at.
 *
 * The pick follows a measurement -- the x/y/z the demonstrations released the peg at -- because
 * a reset has to close the gripper exactly where the peg is, and only the recording knows that.
 *
 * The place height does not. It is one number, set here, and a newly loaded dataset does not
 * move it: the release height is a property of the peg and the fingers holding it, not of the
 * run whose landmarks happen to be loaded, and a height that changes with the dataset is a
 * height nobody chose. `dropWarning` below is what keeps it honest against the pick.
 *
 * The failure both of these exist to prevent: the place height used to be a literal typed into
 * the panel, and it read 0.55. Every reset then carried the peg up to 55 cm, opened the gripper
 * there and dropped it half a metre onto the table, while the pick -- which did follow the
 * measurement -- stayed at 5 cm. Nothing downstream can tell that apart from an operator who
 * meant it: the workspace fence and the reach check both bound the arm, not the drop.
 */

/** Where the pick starts before any dataset has been measured, in metres: the hole, where every
 *  rollout leaves the peg. The same point as `GRASP_LOOP_PICK_XYZ` in tools/fr3/grasp_loop.py --
 *  the grasp loop's fixture pick, which held on every 09-29 fetch.
 *
 *  Was (0.40, 0.00, 0.035), a point nothing stands at. With no measurement to replace it (a view
 *  without demonstration parquet, such as funnel105) it stayed on the map labelled "pick", and on
 *  10-16 a mask was drawn around it as if it were the hole: 130-200 mm left of where the policy
 *  had learned to grasp. */
export const DEFAULT_PICK_XYZ: readonly [number, number, number] = [0.3597, -0.1328, 0.055];

/** The height the peg is released from, in metres. Fixed, and not derived from the
 *  demonstrations -- the panel still shows what they measured, but the place height defaults
 *  to this value instead of being copied from a loaded dataset. */
export const PLACE_Z_M = 0.058;


/** How far above the pick a place height may sit before it counts as a drop, in metres.
 *
 * Has to clear the gap the two defaults above sit apart (3 mm), or a fresh panel -- one with no
 * measurement yet, so the pick falls back to the hole -- opens on a warning about the rig's own
 * setup, which is a banner nobody can act on. It stays far below the 50 cm the constant exists
 * to catch, and far below any release height that would actually let the peg fall. */
export const DROP_WARNING_M = 0.025;

/** What to tell the operator when the place height is not the height the peg came from.
 *
 * Opening the gripper above the surface drops the peg by the difference. A few millimetres is
 * the peg settling out of the fingers; more than that is a fall, and the arm gives no other
 * sign it is about to take one -- the workspace fence and the reach check both pass, because
 * the arm can perfectly well reach a point in mid-air. Better the number is on screen before
 * the reset runs than inferred afterwards from the noise.
 */
export function dropWarning(pickZ: number, placeZ: number): string {
  const drop = placeZ - pickZ;
  if (!Number.isFinite(drop) || drop <= DROP_WARNING_M) return "";
  return `Place z is ${(drop * 1000).toFixed(0)} mm above the pick — the peg is released there and falls that far.`;
}
