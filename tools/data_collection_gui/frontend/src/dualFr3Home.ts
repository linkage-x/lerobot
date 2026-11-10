// Both FR3s home (tools/fr3/dual_fr3_move_to_start.py): what the panel shows of a check's plan and what it
// lets the operator do. The plan decides; this only reads it.
import type { DualHomeJob, DualHomePlan, RobotArms } from "./types";

const ARMS = ["right", "left"] as const;
const LIVE = new Set(["checking", "ready", "moving"]);

export function dualHomeLive(job: DualHomeJob | null | undefined): boolean {
  return Boolean(job?.state && LIVE.has(job.state));
}

// Both arms in service, or why the panel cannot home them.
export function dualHomeArmsBlocker(arms: RobotArms | null): string {
  if (!arms) return "Loading the arm registry…";
  const serving = arms.arms.filter((row) => row.inService);
  if (serving.length < 2) return `Both arms must be in service (now: ${serving.map((row) => row.arm).join(", ") || "none"}).`;
  return "";
}

const ORDER_LABELS: Record<string, string> = {
  together: "together",
  right_then_left: "right first, then left",
  left_then_right: "left first, then right"
};

export function orderLabel(name: string): string {
  return ORDER_LABELS[name] ?? name;
}

export type DualHomeVerdict = { tone: "ok" | "no"; headline: string };

export function dualHomeVerdict(plan: DualHomePlan | null | undefined): DualHomeVerdict | null {
  if (!plan) return null;
  if (plan.stages === null) return { tone: "no", headline: "Refused: neither arm moves" };
  if (!plan.stages.length) return { tone: "ok", headline: "Both arms are already at their start joints" };
  const movers = plan.stages.flat();
  if (plan.stages.length === 1) {
    return {
      tone: "ok",
      headline: movers.length === 2 ? "Both arms move at once" : `Only the ${movers[0]} arm moves; the other is home`
    };
  }
  return { tone: "ok", headline: `One after the other: the ${plan.stages[0][0]} arm first, then the ${plan.stages[1][0]}` };
}

export type DualHomeOptionRow = {
  name: string;
  label: string;
  gap: string;
  where: string;
  clear: boolean;
  chosen: boolean;
};

// Every way the arms could go, with its closest approach against what the paths are held to.
export function dualHomeOptionRows(plan: DualHomePlan | null | undefined): DualHomeOptionRow[] {
  if (!plan) return [];
  return ["together", "right_then_left", "left_then_right"]
    .filter((name) => plan.options[name])
    .map((name) => {
      const option = plan.options[name];
      return {
        name,
        label: orderLabel(name),
        gap: `${option.min_gap_mm.toFixed(0)} mm`,
        where: `${option.pair.join(" / ")} · ${option.where}`,
        clear: option.min_gap_mm >= plan.gate_mm,
        chosen: plan.chosen === name
      };
    });
}

export type DualHomeArmRow = { arm: string; ip: string; move: string; table: string; payload: string };

export function dualHomeArmRows(plan: DualHomePlan | null | undefined): DualHomeArmRow[] {
  if (!plan) return [];
  return ARMS.map((arm) => {
    const row = plan.arms[arm];
    const env = row.min_env_gap_mm;
    return {
      arm,
      ip: plan.ip?.[arm] ?? "",
      move: row.moving ? `${row.max_joint_move_deg.toFixed(0)}° (largest: j${row.joint})` : "at start — stays",
      table: env === null ? "—" : `${env.toFixed(0)} mm${row.min_env_with ? ` (${row.min_env_with.join(" / ")})` : ""}`,
      payload: plan.payload?.[arm] ?? ""
    };
  });
}

// The gate the paths were held to, and where it came from.
export function dualHomeGateLine(plan: DualHomePlan | null | undefined): string {
  if (!plan) return "";
  const sigma = plan.placement?.sigma_mm;
  const base = `Arm–arm gate ${plan.d_gate_mm.toFixed(0)} mm${sigma ? ` (3 × the placement's ${sigma.toFixed(0)} mm σ)` : ""}; table ${plan.d_env_mm.toFixed(0)} mm.`;
  if (plan.gate_mm < plan.d_gate_mm) {
    return `${base} The arms stand ${plan.start_gap_mm.toFixed(0)} mm apart now, inside the gate: their way home may not come closer than that (${plan.gate_mm.toFixed(0)} mm).`;
  }
  return base;
}

// What the confirmation says will happen.
export function dualHomeStagesText(plan: DualHomePlan | null | undefined): string {
  if (!plan?.stages?.length) return "";
  const speed = plan.speed_factor ? ` at speed factor ${plan.speed_factor}` : "";
  if (plan.stages.length === 1) {
    const movers = plan.stages[0];
    return movers.length === 2
      ? `Both arms move to their start joints at once${speed}, each in a straight line in joint space.`
      : `The ${movers[0]} arm moves to its start joints${speed}; the ${movers[0] === "right" ? "left" : "right"} stays.`;
  }
  return `The ${plan.stages[0][0]} arm moves to its start joints${speed}; once it is there, the ${plan.stages[1][0]} follows.`;
}

export type DualHomeControls = { canCheck: boolean; canExecute: boolean; canCancel: boolean; cancelStops: boolean };

export function dualHomeControls(job: DualHomeJob | null | undefined, armsBlocker: string): DualHomeControls {
  const live = dualHomeLive(job);
  return {
    canCheck: !live && !armsBlocker,
    canExecute: job?.state === "ready" && Boolean(job.plan?.stages?.length),
    canCancel: live,
    // moving: cancel ends the process group, and both arms stop on a dropped FCI
    cancelStops: job?.state === "moving"
  };
}

// Seconds left before the runtime gives up waiting for Execute; null when nothing waits.
export function dualHomeSecondsLeft(job: DualHomeJob | null | undefined, nowS: number): number | null {
  if (job?.state !== "ready" || !job.readyUnixS || !job.waitS) return null;
  return Math.max(0, Math.round(job.readyUnixS + job.waitS - nowS));
}

export function dualHomeStatus(job: DualHomeJob | null | undefined): string {
  switch (job?.state) {
    case undefined:
      return "idle";
    case "checking":
      return "reading both arms and checking their paths…";
    case "ready":
      return "plan ready — read it, then Execute or Cancel";
    case "moving":
      return "moving — the user stops halt the arms fastest";
    case "done":
      return job.result ? (job.result.ok ? "both arms at their start joints" : "done with errors") : "nothing to move";
    case "cancelled":
      return job.result ? "stopped while moving" : "cancelled; nothing moved";
    case "refused":
      return `refused: ${job.error || "see the plan"}`;
    case "failed":
      return `failed: ${job.error || `exit ${job.exitCode}`}`;
    default:
      return String(job?.state);
  }
}

export type DualHomeResultRow = { stage: number; arm: string; ok: boolean; text: string };

export function dualHomeResultRows(job: DualHomeJob | null | undefined): DualHomeResultRow[] {
  return (job?.result?.stages ?? []).flatMap((stage, index) =>
    Object.entries(stage).map(([arm, row]) => ({
      stage: index + 1,
      arm,
      ok: row.ok,
      text: row.error ? row.error : `${(row.max_error_deg ?? 0).toFixed(2)}° from its start joints`
    }))
  );
}
