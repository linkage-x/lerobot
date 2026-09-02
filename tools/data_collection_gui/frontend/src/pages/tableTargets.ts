/** Where to send the arm to calibrate one camera against the table.
 *
 * A plane homography needs four points *in general position*: no three of them on one straight
 * line. Three collinear points state the same line twice, the fit is underdetermined, and the
 * matrix that comes out sends much of the image past the horizon rather than onto the table.
 *
 * That constraint is what makes this a module and not a literal. The obvious set -- four corners
 * and the centre -- is fine only while all five are reachable, and the corners nearest the robot
 * are exactly the ones inverse kinematics refuses. Drop one and the centre, which sits on both
 * diagonals, is in a line with the two corners that remain. So the fifth point is pulled off the
 * centre, and the test holds the invariant that every four of these five can stand alone.
 */
export type TableTarget = { x: number; y: number };

export function suggestedTargets(centre: [number, number], spreadM: number): TableTarget[] {
  return [
    { x: centre[0] + spreadM, y: centre[1] + spreadM },
    { x: centre[0] + spreadM, y: centre[1] - spreadM },
    { x: centre[0] - spreadM, y: centre[1] - spreadM },
    { x: centre[0] - spreadM, y: centre[1] + spreadM },
    // Half way out towards +x: away from the robot, so it is the reachable direction, and off
    // every edge and both diagonals by a third of the spread.
    { x: centre[0] + spreadM * 0.5, y: centre[1] }
  ];
}

/** How far the middle point of a triple is off the line through the other two, in metres. */
export function offLineDistance(a: TableTarget, b: TableTarget, c: TableTarget): number {
  const areaTwice = Math.abs((b.x - a.x) * (c.y - a.y) - (b.y - a.y) * (c.x - a.x));
  const longest = Math.max(Math.hypot(b.x - a.x, b.y - a.y), Math.hypot(c.x - a.x, c.y - a.y), Math.hypot(c.x - b.x, c.y - b.y));
  return longest > 0 ? areaTwice / longest : 0;
}
