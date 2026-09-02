import type { SceneResetStroke } from "../types";

/** Which mask a panel should hold once the stored one arrives.
 *
 * The stored region is fetched on mount and the fetch is a round trip, so it can land after the
 * operator has already started drawing. What is on the canvas wins: strokes are drawn against a
 * camera still of the table as it is right now, and replacing them mid-drawing with the region
 * from a previous session is both surprising and, since the reset picks its place point from
 * this mask, a peg put somewhere nobody aimed at.
 *
 * The stored mask therefore only fills an empty canvas -- which is the case the persistence
 * exists for: a page reloaded between rollouts should come back with the region it had.
 */
export function strokesAfterLoad(
  current: SceneResetStroke[],
  saved: SceneResetStroke[]
): SceneResetStroke[] {
  return current.length > 0 ? current : saved;
}
