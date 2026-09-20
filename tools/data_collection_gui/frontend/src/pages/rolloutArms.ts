import type { RolloutArm, RolloutOutcomeEntry } from "../types";

/** Which arm a rollout ran, and whether the rig and the operator agreed about how it ended.
 *
 *  Kept out of the component for the reason the unattended panel's decisions are: both of
 *  these are read off a log, both are arithmetic, and neither needs a browser to be wrong.
 *
 *  The two live together because they answer one question between them. Every comparison on
 *  this rig is between arms -- medoid against mean against a single draw, search ring on
 *  against off -- and an arm is only worth reading if the column that grades it can be
 *  trusted. So the page shows the arm it is running next to how often the rig's own verdict
 *  matched the operator standing there.
 */

/** A one-line name for an arm, short enough for a session bar.
 *
 *  Written so two arms are distinguishable at a glance rather than complete: a reader who
 *  wants the pose reads the record, and a reader looking at the bar wants to know they have
 *  not left last night's configuration on.
 */
export function describeArm(arm: RolloutArm | undefined): string {
  if (!arm || Object.keys(arm).length === 0) return "";
  const parts: string[] = [];
  if (arm.actionSamples !== undefined) {
    parts.push(
      arm.actionSamples <= 1
        ? "1 draw"
        : `${arm.actionAggregate ?? "?"} of ${arm.actionSamples}`
    );
  }
  const landings = arm.terminalServoSearchLandings;
  if (landings !== undefined) {
    // 1 landing is not "a search with one point" -- it is the search switched off, which is
    // E5's control arm and the thing every other reading is compared against. Naming it that
    // way is what keeps a reader from filing a control run as a search run.
    parts.push(
      landings <= 1
        ? "fixed pose"
        : `${landings} landings @ ${((arm.terminalServoSearchRingM ?? 0) * 1000).toFixed(1)} mm`
    );
  } else if (arm.terminalServoXyz) {
    parts.push("fixed pose");
  }
  return parts.join(" · ");
}

/** What the rig's verdict means in the operator's vocabulary.
 *
 *  `slip` maps to a failure rather than to its own grade because the operator's scale has no
 *  word for it: a peg that slid up in the fingers is filed as a miss by whoever is watching,
 *  and the point of this comparison is to count how often the two scales disagree, not to
 *  invent a third.
 */
export function verdictAsOutcome(verdict: string | undefined): "success" | "failure" | "" {
  if (verdict === "seated") return "success";
  if (verdict === "standing" || verdict === "slip") return "failure";
  return "";
}

export type VerdictAgreement = {
  /** Rollouts where both the rig and the operator said something comparable. */
  compared: number;
  agreed: number;
  /** Null rather than 0 when nothing is comparable yet, so the page can say "no reading"
   *  instead of "0% agreement", which is a very different sentence. */
  rate: number | null;
  /** Rollout indices where they disagreed, in the order they were graded. These are the only
   *  rows worth opening, and an agreement rate without them is a number nobody can act on. */
  disagreed: number[];
};

/** How often the rig's own verdict matched the operator's grade.
 *
 *  This is the number the unattended loops are accepted on -- the E6 criterion is 95% against
 *  hand labels -- and it can only be computed because the two are stored as separate columns.
 *
 *  `aborted` rollouts are excluded rather than counted as failures. An abort is a statement
 *  about the session, not about the peg: the operator stopped the run, and the descent may
 *  never have happened. Counting them would move this rate around for reasons that have
 *  nothing to do with whether the rig can read its own descents.
 */
export function verdictAgreement(entries: RolloutOutcomeEntry[]): VerdictAgreement {
  let compared = 0;
  let agreed = 0;
  const disagreed: number[] = [];
  for (const entry of entries) {
    const rig = verdictAsOutcome(entry.terminalServo?.verdict);
    if (!rig) continue;
    if (entry.outcome !== "success" && entry.outcome !== "failure") continue;
    compared += 1;
    if (rig === entry.outcome) agreed += 1;
    else disagreed.push(entry.rolloutIndex ?? 0);
  }
  return {
    compared,
    agreed,
    rate: compared ? agreed / compared : null,
    disagreed
  };
}
