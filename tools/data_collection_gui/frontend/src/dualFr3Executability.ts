import type { DualFr3EpisodeVerdict, DualFr3Executability, DualFr3Stats } from "./types";

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
