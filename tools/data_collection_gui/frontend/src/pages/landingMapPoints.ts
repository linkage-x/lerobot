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
};

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

export function describe(geometry: RolloutGeometry, prefix: string): string {
  const parts = [prefix];
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
      title: [
        describe(entry.geometry ?? {}, `${label} — ${entry.outcome}`),
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
