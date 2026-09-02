import type { RolloutLastParams } from "../types";

/** What the page tells the operator about the settings a new rollout inherited.
 *
 * Almost everything on this page is carried over from the last rollout, and most of it is
 * invisible until you scroll to the field it landed in. The notice is what makes the carry-over
 * something the operator was told rather than something they find out.
 *
 * The SpaceMouse switch gets its own sentence. It used to be the one setting deliberately not
 * remembered: a remembered "yes" opens a second action source onto a loop that is driving a real
 * arm, and nobody decided that this session. It is remembered now -- re-ticking it before every
 * rollout of an afternoon spent collecting corrections is its own kind of noise -- and this
 * sentence is what stands in for the protection that dropped. The arm still does not move
 * without motion confirmation; what the sentence adds is that if takeover came back on, somebody
 * has to be at the rig, said before the run rather than discovered during it.
 */
export function carriedOverNotice(params: RolloutLastParams): string {
  const parts = ["Settings carried over from the last rollout. Motion confirmation is not."];
  if (params.runtimeOptions?.daggerTakeover) {
    parts.push("SpaceMouse takeover is on from last time — somebody has to be at the rig.");
  }
  return parts.join(" ");
}
