import type {
  DualFr3EpisodeVerdict,
  DualFr3Executability,
  DualFr3Stats,
  DualReplayJob,
  DualReplayState,
  RobotArm
} from "./types";

export type DualFr3Row = {
  mode: string;
  verdict: "executable" | "not executable" | "not evaluated";
  change: string;
  tlr: string;
  gap: string;
  placement: string;
  timing: string;
  reasons: string[];
  evaluated: string;
};

const MODE_ORDER = ["absolute", "relative"];

function stats(value: DualFr3Stats | undefined, digits = 1): string {
  if (!value) return "—";
  return `${value.p95.toFixed(digits)} / ${value.max.toFixed(digits)}`;
}

function rowFor(mode: string, verdict: DualFr3EpisodeVerdict | null, createdUtc?: string): DualFr3Row {
  const evaluated = createdUtc ?? "";
  if (!verdict) {
    return {
      mode,
      verdict: "not evaluated",
      change: "—",
      tlr: "—",
      gap: "—",
      placement: "—",
      timing: "—",
      reasons: [],
      evaluated
    };
  }
  const change = verdict.retarget_change_position_mm;
  const gap =
    verdict.min_arm_arm_gap_mm != null
      ? `${verdict.min_arm_arm_gap_mm.toFixed(0)} / ${verdict.d_safe_mm != null ? verdict.d_safe_mm.toFixed(0) : "?"}` +
        (verdict.mesh_min_gap_mm != null ? ` (mesh ${verdict.mesh_min_gap_mm.toFixed(0)})` : "")
      : "—";
  return {
    mode,
    verdict: verdict.dual_fr3_executable ? "executable" : "not executable",
    change: change ? `R ${stats(change.right, 2)} · L ${stats(change.left, 2)}` : "—",
    tlr: verdict.t_lr_change_position_mm ? verdict.t_lr_change_position_mm.max.toFixed(1) : "—",
    gap,
    placement: verdict.placement_assumed == null ? "—" : verdict.placement_assumed ? "assumed" : "measured",
    timing:
      verdict.reference_duration_s != null
        ? `${verdict.reference_duration_s.toFixed(1)} s` +
          (verdict.slowdown_vs_nominal != null && verdict.slowdown_vs_nominal > 1.001
            ? ` (×${verdict.slowdown_vs_nominal.toFixed(2)} retimed)`
            : "")
        : "—",
    reasons: verdict.reasons ?? [],
    evaluated
  };
}

// Absolute first, then relative, then whatever else the file holds.
export function dualFr3Rows(payload: DualFr3Executability | null): DualFr3Row[] {
  if (!payload) return [];
  const modes = Object.keys(payload.modes).sort(
    (a, b) => (MODE_ORDER.indexOf(a) + 1 || 99) - (MODE_ORDER.indexOf(b) + 1 || 99) || a.localeCompare(b)
  );
  return modes.map((mode) => rowFor(mode, payload.modes[mode].episode, payload.modes[mode].createdUtc));
}

// --- the operator's half (roadmap D4): Connect -> both arms at frame 0 -> Execute; Abort any time ---

const LIVE: DualReplayState[] = ["connecting", "at_start", "executing"];

export function dualReplayLive(job: DualReplayJob | null | undefined): boolean {
  return Boolean(job?.state && LIVE.includes(job.state));
}

// Which graph edges each arm's targets go through in this mode, one line each (C2's plan, from C4).
export function edgeLines(verdict: DualFr3EpisodeVerdict | null | undefined): string[] {
  const arms = verdict?.mode?.arms ?? {};
  return ["right", "left"]
    .filter((arm) => arms[arm])
    .map((arm) => {
      const edges = arms[arm].edges ?? [];
      const via = edges.length
        ? edges.map((e) => `${e.from_world_frame_id} → ${e.to_world_frame_id} (${e.method ?? "?"})`).join(", ")
        : "none: the dataset is in its base";
      return `${arm} (${arms[arm].root ?? "?"}): ${via}`;
    });
}

// Why the real FR3s cannot be the backend: an arm out of service (B1). "" when both are; the runtime
// still says what else it waits for.
export function realBackendBlocker(arms: RobotArm[] | undefined): string {
  const missing = ["right", "left"].map((name) => arms?.find((row) => row.arm === name)).filter((row) => !row?.inService);
  if (!arms?.length) return "arm registry not read";
  return missing.map((row) => (row ? `${row.arm}: ${row.whyNot}` : "an arm is missing from fr3_arms.json")).join("; ");
}

export type DualReplayControls = {
  connectBlocker: string;
  canExecute: boolean;
  canAbort: boolean;
};

export function dualReplayControls(
  payload: DualFr3Executability | null,
  job: DualReplayJob | null | undefined,
  mode: string,
  backend: string,
  arms: RobotArm[] | undefined
): DualReplayControls {
  const verdict = payload?.modes[mode]?.episode ?? null;
  let connectBlocker = "";
  if (dualReplayLive(job)) connectBlocker = "a two-arm replay is running";
  else if (!verdict) connectBlocker = `no ${mode} verdict for this episode`;
  else if (!verdict.dual_fr3_executable) connectBlocker = `not executable in ${mode}: ${(verdict.reasons ?? []).join("; ")}`;
  else if (backend === "real") connectBlocker = realBackendBlocker(arms);
  return {
    connectBlocker,
    canExecute: job?.state === "at_start",
    canAbort: dualReplayLive(job)
  };
}

export function dualReplayStatus(job: DualReplayJob | null | undefined): string {
  if (!job?.state) return "idle";
  const what = `${job.backend ?? "?"} arms, ${job.mode ?? "?"}, episode ${job.episode ?? "?"}`;
  const report = job.report;
  switch (job.state) {
    case "connecting":
      return `moving both arms to frame 0 (${what})`;
    case "at_start":
      return `both arms at frame 0, waiting for Execute (${what})`;
    case "executing":
      return `executing (${what})`;
    case "finished":
      return `finished (${what}): min gap ${report?.minGapMm ?? "?"} mm, start skew ${report?.startSkewMs ?? "?"} ms`;
    case "stopped":
      return `stopped (${what}): ${(report?.stop ?? []).join("; ") || "?"}` +
        (report?.stoppedTogether === false ? " — the arms did NOT stop together" : "");
    case "error":
      return `refused or failed (${what}): ${job.error || "?"}`;
  }
}
