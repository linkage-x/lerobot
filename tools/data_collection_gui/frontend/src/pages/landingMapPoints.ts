import type { RolloutGeometry, RolloutOutcomeEntry } from "../types";

/** Turning graded rollouts into the dots on the landing map.
 *
 * Split out of the component because the ordering below is the part that can be wrong without
 * looking wrong: a buried dot and an absent one are the same picture.
 */

export type PlottedOutcome = "success" | "failure" | "aborted" | "pending";

export type PlottedPoint = {
  key: string;
  x: number;
  y: number;
  outcome: PlottedOutcome;
  /** False when the gripper never closed: the point is where it reached, not where it gripped. */
  closed: boolean;
  /** When it was graded. Sorted on so the newest dot is painted last. */
  at: string;
  title: string;
  /** How far along the task's precondition chain it got, when graded on a ladder. */
  stage?: number;
  /** The stage that counted as success on the ladder that graded it. */
  terminalStage?: number;
};

// A sequential ramp rather than a palette. `stage` is ordinal, so how far a rollout got has to
// be readable from the colour itself; categorical hues would throw that ordering away, which is
// precisely what success/failure did to the 20-rollout batch that broke in three places and came
// back one colour. Sampled by position along the chain, so a task with four stages and a task
// with seven both run the full ramp.
const STAGE_RAMP = ["#b3261e", "#d1512b", "#e08214", "#c8a415", "#93a520", "#5b9c3e", "#2f9e5f"];

const OUTCOME_FILL: Record<PlottedOutcome, string> = {
  success: "#2f9e5f",
  failure: "#cf4b3a",
  aborted: "#8a8a8a",
  // Outside the ramp on purpose: an ungraded dot must not read as a middling stage.
  pending: "#3b7dd8"
};

export function stageFill(stage: number, terminalStage: number): string {
  if (terminalStage <= 1) return STAGE_RAMP[STAGE_RAMP.length - 1];
  const t = Math.min(1, Math.max(0, (stage - 1) / (terminalStage - 1)));
  return STAGE_RAMP[Math.round(t * (STAGE_RAMP.length - 1))];
}

export function pointFill(point: PlottedPoint): string {
  // Aborted and not-yet-graded keep their own colour whatever stage they reached: neither is
  // evidence about the policy, and colouring them along the ramp would put them in the tally
  // the ramp exists to show.
  if (point.outcome === "aborted" || point.outcome === "pending") return OUTCOME_FILL[point.outcome];
  if (point.stage === undefined || point.terminalStage === undefined) {
    return OUTCOME_FILL[point.outcome];
  }
  return stageFill(point.stage, point.terminalStage);
}

export function landingPoint(
  geometry: RolloutGeometry | undefined
): [number, number, number] | null {
  if (!geometry) return null;
  // Grasp first, approach as the fallback: a rollout that closed on the object and one that
  // only reached toward it are both worth a dot, and which of the two it was is carried by
  // `closed` rather than by leaving the second kind off the map.
  return geometry.graspXyz ?? geometry.approachXyz ?? null;
}

export function formatMm(value: number): string {
  return `${(value * 1000).toFixed(0)} mm`;
}

export function describe(
  geometry: RolloutGeometry,
  prefix: string,
  grade?: { stage?: number; stageId?: string; terminalStage?: number; blocker?: string }
): string {
  const parts = [prefix];
  // Ahead of the geometry, because it is the part the geometry cannot supply: `lift` and
  // `held steps` cannot tell a held object from a gripper closed on air.
  if (grade?.stage !== undefined) {
    const of = grade.terminalStage !== undefined ? `/${grade.terminalStage}` : "";
    parts.push(`stage ${grade.stage}${of}${grade.stageId ? ` ${grade.stageId}` : ""}`);
    if (grade.blocker && grade.blocker !== "unknown") parts.push(grade.blocker);
  }
  if (!geometry.closed) {
    parts.push("gripper never closed (approach point)");
  } else {
    if (geometry.liftM !== undefined) parts.push(`lift ${formatMm(geometry.liftM)}`);
    if (geometry.descentM !== undefined) parts.push(`insert descent ${formatMm(geometry.descentM)}`);
  }
  return parts.join(" · ");
}

export function buildLandingPoints(
  entries: RolloutOutcomeEntry[],
  pendingIndex: number,
  pendingGeometry?: RolloutGeometry
): PlottedPoint[] {
  const points: PlottedPoint[] = [];
  entries.forEach((entry, index) => {
    const point = landingPoint(entry.geometry);
    if (!point) return;
    const label = entry.rolloutIndex ? `Rollout ${entry.rolloutIndex}` : entry.recordedAt;
    points.push({
      key: `${entry.recordedAt}-${index}`,
      x: point[0],
      y: point[1],
      outcome: entry.outcome,
      closed: entry.geometry?.closed ?? true,
      at: entry.recordedAt,
      stage: entry.stage,
      terminalStage: entry.terminalStage,
      title: [
        describe(entry.geometry ?? {}, `${label} — ${entry.outcome}`, entry),
        entry.note ? `“${entry.note}”` : ""
      ]
        .filter(Boolean)
        .join("\n")
    });
  });
  // Oldest first, so the newest dot is painted last and is never buried. The history arrives
  // newest first, and two rollouts landing on the same spot is the map's most interesting signal
  // rather than a rarity -- a systematic offset puts every point in one place -- so whichever
  // dot is on top has to be the one the operator just graded.
  points.sort((left, right) => (left.at < right.at ? -1 : left.at > right.at ? 1 : 0));
  // The rollout that just finished, drawn before anybody grades it. Without this the map is
  // blank for exactly the seconds the operator is deciding what they saw, which is when they
  // most want to know whether this placement was inside the demonstrated region.
  const pendingPoint = landingPoint(pendingGeometry);
  if (pendingIndex > 0 && pendingPoint) {
    points.push({
      key: `pending-${pendingIndex}`,
      x: pendingPoint[0],
      y: pendingPoint[1],
      outcome: "pending",
      closed: pendingGeometry?.closed ?? true,
      at: "",
      title: describe(pendingGeometry ?? {}, `Rollout ${pendingIndex} — not graded yet`)
    });
  }
  return points;
}
