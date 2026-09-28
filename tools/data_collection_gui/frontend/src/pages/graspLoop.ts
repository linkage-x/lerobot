import type { GraspLoopTrial } from "../types";

/** 95% Wilson interval for k of n. Same formula as `wilson_interval` in tools/fr3/grasp_loop.py,
 *  so the number on the page is the number the row file's summary will carry. */
export function wilsonInterval(successes: number, n: number, z = 1.96): [number, number] {
  if (n <= 0) return [0, 1];
  const p = successes / n;
  const denom = 1 + (z * z) / n;
  const centre = (p + (z * z) / (2 * n)) / denom;
  const half = (z * Math.sqrt((p * (1 - p)) / n + (z * z) / (4 * n * n))) / denom;
  return [Math.max(0, centre - half), Math.min(1, centre + half)];
}

function median(values: number[]): number | null {
  if (values.length === 0) return null;
  const sorted = [...values].sort((a, b) => a - b);
  const mid = Math.floor(sorted.length / 2);
  return sorted.length % 2 ? sorted[mid] : (sorted[mid - 1] + sorted[mid]) / 2;
}

/** What `summarize_grasp_loop` grades: a miss, no close, or a collision counts against the arm. */
export const GRADED_VERDICTS = ["held", "empty", "no_close", "collision"];

/** Median of one covariate, split by verdict: held against everything graded that was not. */
export function medianByVerdict(
  trials: GraspLoopTrial[],
  key: "closeAboveTargetMm" | "lateralMm"
): { held: number | null; notHeld: number | null } {
  const pick = (keep: (t: GraspLoopTrial) => boolean) =>
    median(trials.filter(keep).map((t) => t[key]).filter((v): v is number => typeof v === "number"));
  return {
    held: pick((t) => t.verdict === "held"),
    notHeld: pick((t) => t.verdict !== "held" && GRADED_VERDICTS.includes(t.verdict))
  };
}

export type ArmRate = { arm: string; graded: number; held: number; low: number; high: number };

/** Held rate per arm. An interleaved run's pooled rate is a rate of neither arm, so the panel
 *  leads with these. Same grading as `summarize_grasp_loop` (GRADED_VERDICTS). */
export function armRates(trials: GraspLoopTrial[]): ArmRate[] {
  const byArm = new Map<string, { graded: number; held: number }>();
  for (const trial of trials) {
    if (!GRADED_VERDICTS.includes(trial.verdict)) continue;
    const arm = trial.arm ?? "A";
    const entry = byArm.get(arm) ?? { graded: 0, held: 0 };
    entry.graded += 1;
    if (trial.verdict === "held") entry.held += 1;
    byArm.set(arm, entry);
  }
  return [...byArm.entries()]
    .sort(([a], [b]) => a.localeCompare(b))
    .map(([arm, { graded, held }]) => {
      const [low, high] = wilsonInterval(held, graded);
      return { arm, graded, held, low, high };
    });
}

/** v14 (1): n = 50 per layer; 45/50 clears the 90% line, 47/50 counts toward 95%. */
export function gateReading(held: number, graded: number, planned = 50): string {
  if (graded === 0) return "";
  const misses = graded - held;
  if (misses > planned - 45) return `已失手 ${misses} 次，这批过不了 45/${planned}`;
  if (graded >= planned) return held >= 47 ? "≥ 47/50：按 95% 目标记" : held >= 45 ? "≥ 45/50：过 90% 线" : "";
  return `失手 ${misses} / 允许 ${planned - 45}`;
}
