import type { EventDriver, RolloutGeometry } from "../types";

/** Who drove the moments a rollout is judged by.
 *
 * The rollout-level `intervened` flag says a human was in the rollout somewhere. That is the
 * wrong resolution for both questions this page asks. The map draws one point per rollout and
 * has to say who put the gripper there; the grade turns on whether the policy reached the end
 * of the task by itself. An operator who took the arm after the grasp did not place it, and one
 * who seated the peg did not leave the policy a success.
 *
 * `undefined` means the runtime did not attribute the event -- an older log line -- and is
 * deliberately not folded into "the policy": an unattributed point is unknown, not innocent.
 */

export function landingPointDriver(geometry?: RolloutGeometry): EventDriver | undefined {
  if (!geometry) return undefined;
  // Same precedence as `landingPoint`: the map draws the grasp when there is one and the
  // approach otherwise, so the driver it needs is the driver of whichever it drew.
  if (geometry.graspXyz) return geometry.graspBy;
  if (geometry.approachXyz) return geometry.approachBy;
  return undefined;
}

export function terminalEventDriver(geometry?: RolloutGeometry): EventDriver | undefined {
  if (!geometry) return undefined;
  // The release is the last thing that happens to the object -- the peg going in, the cube being
  // put down. A rollout with no release never got that far, and then the question the caller is
  // asking (did the policy finish it?) is already answered by the stage.
  return geometry.releaseXyz ? geometry.releaseBy : undefined;
}

/** Whether "success" is a grade this rollout is allowed to carry.
 *
 * Blocked, not warned about: the outcome log is what two checkpoints are compared on, and a
 * success the policy did not earn is the one entry more rollouts cannot correct. `acknowledged`
 * is the operator answering the single question the trace cannot -- whether the task was already
 * finished when they took the arm to tidy up -- and the page writes that answer into the note.
 */
export function assistedSuccessBlocked(
  geometry: RolloutGeometry | undefined,
  acknowledged: boolean
): boolean {
  return terminalEventDriver(geometry) === "expert" && !acknowledged;
}
