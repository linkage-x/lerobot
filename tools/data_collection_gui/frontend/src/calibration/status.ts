// Calibration freshness (validity) computation and state-machine helpers.
// Pure functions only — trivially unit-testable and free of React.

import { VALIDITY, type CalibrationKind } from "./config";
import type { CaliMachineState, DotState, ValidityState } from "./types";
import { TERMINAL_STATES, BUSY_STATES } from "./types";

export type Validity = {
  state: ValidityState;
  /** Milliseconds until expiry (negative once overdue); null when never run. */
  remainingMs: number | null;
  /** Absolute expiry time (ms epoch); null when never run. */
  expiresAt: number | null;
};

/**
 * Classify how fresh a calibration is given when it last ran.
 *
 * `valid`    — expiry is further away than the kind's due-soon lead time
 * `due_soon` — inside the due-soon window but not yet expired
 * `overdue`  — past expiry
 * `unknown`  — never run (no timestamp)
 */
export function computeValidity(
  lastRunAt: number | null | undefined,
  kind: CalibrationKind,
  now: number = Date.now(),
): Validity {
  if (lastRunAt == null || !Number.isFinite(lastRunAt)) {
    return { state: "unknown", remainingMs: null, expiresAt: null };
  }
  const { validMs, dueSoonMs } = VALIDITY[kind];
  const expiresAt = lastRunAt + validMs;
  const remainingMs = expiresAt - now;
  let state: ValidityState;
  if (remainingMs <= 0) state = "overdue";
  else if (remainingMs <= dueSoonMs) state = "due_soon";
  else state = "valid";
  return { state, remainingMs, expiresAt };
}

/** Map a validity state onto the shared StatusDot palette. */
export function validityDot(state: ValidityState): DotState {
  switch (state) {
    case "valid":
      return "running";
    case "due_soon":
      return "warning";
    case "overdue":
      return "error";
    case "unknown":
    default:
      return "idle";
  }
}

/** Fold several validities into the worst one (for a section/nav badge). */
export function worstValidity(states: ValidityState[]): ValidityState {
  if (states.includes("overdue")) return "overdue";
  if (states.includes("due_soon")) return "due_soon";
  if (states.length && states.every((s) => s === "valid")) return "valid";
  return "unknown";
}

export function isBusyState(state: CaliMachineState): boolean {
  return BUSY_STATES.includes(state);
}

export function isTerminalState(state: CaliMachineState): boolean {
  return TERMINAL_STATES.includes(state);
}

/** Map a machine state onto the shared StatusDot palette. */
export function machineDot(state: CaliMachineState): DotState {
  if (state === "passed") return "running";
  if (state === "failed") return "error";
  if (state === "cancelled" || state === "idle") return "idle";
  return "warning"; // any in-flight step
}

/**
 * The allowed forward transitions of the calibration state machine (spec §8).
 * Used both by the workflow hook and by tests to guard against illegal jumps.
 * `waiting_for_reboot`/`reconnecting` only apply to the origin (raw) flow.
 */
export const ALLOWED_TRANSITIONS: Record<CaliMachineState, CaliMachineState[]> = {
  idle: ["checking_prerequisites", "cancelled"],
  checking_prerequisites: ["waiting_for_stability", "failed", "cancelled"],
  waiting_for_stability: ["calibrating", "failed", "cancelled"],
  calibrating: ["validating", "waiting_for_reboot", "failed", "cancelled"],
  waiting_for_reboot: ["reconnecting", "cancelled", "failed"],
  reconnecting: ["validating", "failed", "cancelled"],
  validating: ["passed", "failed", "cancelled"],
  passed: ["idle", "checking_prerequisites"],
  failed: ["idle", "checking_prerequisites"],
  cancelled: ["idle", "checking_prerequisites"],
};

export function canTransition(from: CaliMachineState, to: CaliMachineState): boolean {
  return ALLOWED_TRANSITIONS[from]?.includes(to) ?? false;
}

export type ValiditySummary = { overdue: number; dueSoon: number; valid: number; unknown: number };

/**
 * Roll the latest-run time of each calibration kind into overall counts for
 * the sidebar badge. A kind that was never run counts as "unknown" (needs
 * attention) — it is not silently treated as valid.
 */
export function summarizeKinds(
  latestByKind: Partial<Record<CalibrationKind, number | null>>,
  now: number = Date.now(),
): ValiditySummary {
  const summary: ValiditySummary = { overdue: 0, dueSoon: 0, valid: 0, unknown: 0 };
  (Object.keys(VALIDITY) as CalibrationKind[]).forEach((kind) => {
    const last = latestByKind[kind] ?? null;
    const v = computeValidity(last, kind, now);
    summary[v.state === "due_soon" ? "dueSoon" : v.state] += 1;
  });
  return summary;
}
