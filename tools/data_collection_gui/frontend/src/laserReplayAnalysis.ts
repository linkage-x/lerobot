import type { LaserReplayRun } from "./types";

export const mm = (value: number | undefined | null, digits = 2) =>
  value == null || !Number.isFinite(value) ? "—" : value.toFixed(digits);

export const vec = (values: number[] | undefined, digits = 1) =>
  values?.length ? `(${values.map((value) => value.toFixed(digits)).join(", ")})` : "—";

// "20261010T032019Z_episode_000000_right" -> "03:20:19Z"
export function runLabel(run: string): string {
  const match = run.match(/^\d{8}T(\d{2})(\d{2})(\d{2})Z/);
  return match ? `${match[1]}:${match[2]}:${match[3]}Z` : run;
}

// replay.pose_command_lead_frames of a run; "—" for traces that predate the column.
export const leadLabel = (frames: number | null | undefined) =>
  frames == null ? "—" : frames === 0 ? "0" : `${frames} fr`;

export function beamLossSummary(run: LaserReplayRun): string {
  if (run.first_loss_frame == null) return "";
  if (run.first_loss_frame < 0) return `Beam held for all ${run.frames} frames.`;
  const sight = run.line_of_sight;
  const lost = `Beam lost at frame ${run.first_loss_frame} of ${run.frames} and not regained in this run`;
  if (!sight) return `${lost}.`;
  const swing =
    `In the gripper frame the line of sight to the tracker stays within ${mm(sight.seen_max_deg, 1)}° ` +
    `of its mean while the beam is on (${mm(sight.trajectory_max_deg, 1)}° over the whole trajectory)`;
  // A loss that follows the gripper happens at the edge of what the sight line had covered; one far
  // inside it would be something in the scene crossing the beam instead.
  const atEdge =
    sight.at_first_loss_deg != null && sight.at_first_loss_deg >= 0.9 * sight.seen_max_deg
      ? `; it was lost at ${mm(sight.at_first_loss_deg, 1)}°, the edge of the directions seen so far. ` +
        "A small swing lost at its edge points at the SMR (aperture turned away from the tracker, or " +
        "something on the gripper clipping the beam): face the SMR at the tracker at the trajectory start."
      : `; it was lost at ${mm(sight.at_first_loss_deg, 1)}°, inside the directions already seen, ` +
        "so look for something in the scene crossing the beam.";
  return `${lost}. ${swing}${atEdge}`;
}

export function errorBudgetSummary(run: LaserReplayRun): string {
  const dec = run.decomposition;
  if (!dec || !run.laser_vs_desired || !run.laser_vs_fk) return "";
  return (
    `Replay error ${mm(dec.laser_vs_desired_rms_mm)} mm RMS = constant offset ${vec(run.laser_vs_desired_bias_mm)} mm ` +
    `+ ${mm(dec.follow_lag_ms, 0)} ms follow lag + ${mm(dec.without_bias_and_lag_rms_mm)} mm rest. ` +
    `The FR3's own readback is off by ${mm(run.laser_vs_fk.rms_mm)} mm RMS ` +
    `(p95 ${mm(run.laser_vs_fk.p95_mm)}), so the trace's own error is the real one to within that.`
  );
}

// SVG polyline points for a series with gaps (null = beam off): one string per contiguous run.
export function polylineSegments(
  values: (number | null)[],
  width: number,
  height: number,
  yMax: number
): string[] {
  const n = values.length;
  if (n < 2 || !(yMax > 0)) return [];
  const segments: string[] = [];
  let current: string[] = [];
  values.forEach((value, index) => {
    if (value == null || !Number.isFinite(value)) {
      if (current.length > 1) segments.push(current.join(" "));
      current = [];
      return;
    }
    const x = (index / (n - 1)) * width;
    const y = height - (Math.min(value, yMax) / yMax) * height;
    current.push(`${x.toFixed(1)},${y.toFixed(1)}`);
  });
  if (current.length > 1) segments.push(current.join(" "));
  return segments;
}

export function niceCeiling(value: number): number {
  if (!(value > 0)) return 1;
  const exponent = Math.floor(Math.log10(value));
  const base = 10 ** exponent;
  for (const step of [1, 2, 2.5, 5, 10]) {
    if (value <= step * base) return step * base;
  }
  return 10 * base;
}
