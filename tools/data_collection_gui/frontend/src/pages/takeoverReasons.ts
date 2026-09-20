import type { RolloutOutcomeEntry, TakeoverDetail } from "../types";

/** One reason per takeover, which is what this field has always meant and what the control
 *  now makes possible.
 *
 *  The list `blockers` carries is defined as one entry per takeover span, in the order they
 *  happened -- that is what makes "the first one" the reason belonging to the graded stage.
 *  A checkbox set cannot express it: ticking a box twice is the same as ticking it once, so a
 *  rollout rescued three times for the same reason could only ever report it once, and the
 *  backend then dropped the repeat as well. Both halves of that are fixed; this module is the
 *  page's half.
 *
 *  Kept out of the component so the alignment can be checked without a browser, because the
 *  alignment is the whole feature.
 */

/** The reason recorded for a span the operator did not label.
 *
 *  Sent explicitly rather than left blank: blanks are dropped on the way in, which would shift
 *  every later span's reason onto the wrong span -- the exact failure this control exists to
 *  remove. "Recording unknown is allowed; recording nothing loses the question ever having been
 *  asked."
 */
export const UNLABELLED = "unknown";

/** How many reasons this rollout should carry.
 *
 *  One per takeover, or one for the rollout itself when nobody reached in -- that is the field's
 *  older meaning, why it stopped where its stage says it stopped, and it does not disappear
 *  just because there was no takeover.
 */
export function reasonSlots(spanCount: number): number {
  return Math.max(spanCount, 1);
}

/** What to send for `blockers`, given what the operator picked.
 *
 *  Nothing at all when nothing was picked: the backend writes `unknown` for a rollout that fell
 *  short, and a page that sent a full row of `unknown` would turn "nobody was asked" into "asked
 *  and could not say". Once anything is picked the row is sent whole, holes included, because a
 *  partial list is one whose positions no longer mean anything.
 */
export function blockersToSend(selected: string[], spanCount: number): string[] {
  const slots = reasonSlots(spanCount);
  const picked = selected.slice(0, slots);
  if (!picked.some((value) => value)) return [];
  return Array.from({ length: slots }, (_, index) => picked[index] || UNLABELLED);
}

/** How to name one slot to the operator, so they know which reach-in they are labelling.
 *
 *  The step range first because that is what they watched, then the height and what the command
 *  guard said at the last policy step -- the two facts that separate "it was heading for the
 *  wrong place" from "it was being held back by the leash".
 */
export function slotLabel(index: number, detail: TakeoverDetail | undefined): string {
  if (!detail) return `第 ${index + 1} 段`;
  const parts = [`第 ${index + 1} 段`, `步 ${detail.first}–${detail.last}`];
  if (detail.xyz) parts.push(`z ${detail.xyz[2].toFixed(3)}`);
  if (detail.policyStatus && detail.policyStatus !== "pass") parts.push(detail.policyStatus);
  if (detail.stepsLeft !== undefined && detail.stepsLeft <= 0) parts.push("预算用尽");
  return parts.join(" · ");
}

/** How many graded rollouts carry a reason count that did not match their takeovers.
 *
 *  A mismatch is recorded rather than refused, because a grade is perishable -- the operator is
 *  standing at the rig, and a refusal loses that rollout's evidence for good. Recording it is
 *  only half the decision, though: a defect nothing displays is a defect nobody fixes, so the
 *  page carries the count.
 */
export function mismatchCount(entries: RolloutOutcomeEntry[]): number {
  return entries.filter((entry) => entry.takeoverBlockerMismatch).length;
}
