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

/** Where the pick starts before any dataset has been measured, in metres. */
export const TABLE_Z_M = 0.035;

/** The height the peg is released from, in metres. Fixed, and not derived from the
 *  demonstrations -- the panel still shows what they measured, but the place height defaults
 *  to this value instead of being copied from a loaded dataset. */
export const PLACE_Z_M = 0.055;


/** How far above the pick a place height may sit before it counts as a drop, in metres. */
export const DROP_WARNING_M = 0.02;

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
