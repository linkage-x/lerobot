// Domain types for the calibration center. Kept separate from the top-level
// `types.ts` (which mirrors the gateway snapshot) so calibration-only concepts
// don't leak into the transport types.

import type { ReactNode } from "react";
import type { CalibrationKind } from "./config";

export type { CalibrationKind };

/** A 6-channel force/torque reading: [Fx, Fy, Fz, Mx, My, Mz]. */
export type ForceVec = {
  fx: number;
  fy: number;
  fz: number;
  mx: number;
  my: number;
  mz: number;
};

export const FORCE_AXES = ["fx", "fy", "fz", "mx", "my", "mz"] as const;
export type ForceAxis = (typeof FORCE_AXES)[number];

export const FORCE_AXIS_LABELS: Record<ForceAxis, string> = {
  fx: "Fx",
  fy: "Fy",
  fz: "Fz",
  mx: "Mx",
  my: "My",
  mz: "Mz",
};

/** Per-axis pass/fail evaluation against the active thresholds. */
export type AxisEval = {
  axis: ForceAxis;
  label: string;
  value: number;
  /** Newton for forces, N·m for moments — drives the unit shown. */
  kind: "force" | "moment";
  pass: boolean;
  /** Human reason shown when this axis fails (e.g. "超出 ±0.5 N"). */
  reason: string;
};

/** Result of evaluating a whole force vector. */
export type ForceEval = {
  axes: AxisEval[];
  pass: boolean;
  /** First failing axis reason, for a one-line summary. */
  firstFailure: string | null;
};

/**
 * One box's parsed calibration outcome from the streamed CALI_LOG lines.
 * `boxId` is the label emitted by the recorder (`[<box_id>]`, or "box" for a
 * single unnamed box).
 */
export type ParsedBoxResult = {
  boxId: string;
  method: string | null;
  ok: boolean | null;
  rc: number | null;
  error: string | null;
  before: ForceVec | null;
  after: ForceVec | null;
};

/** Aggregate parse of a full calibration log buffer. */
export type ParsedCaliLog = {
  requested: boolean;
  done: boolean;
   /** Terminal CALI_DONE outcome, if seen. */
  outcome: "ok" | "error" | null;
  results: ParsedBoxResult[];
};

// --- Calibration state machine (spec §8) -------------------------------------
export type CaliMachineState =
  | "idle"
  | "checking_prerequisites"
  | "waiting_for_stability"
  | "calibrating"
  | "waiting_for_reboot"
  | "reconnecting"
  | "validating"
  | "passed"
  | "failed"
  | "cancelled";

/** States in which a new run must be blocked (request already in flight). */
export const BUSY_STATES: readonly CaliMachineState[] = [
  "checking_prerequisites",
  "waiting_for_stability",
  "calibrating",
  "waiting_for_reboot",
  "reconnecting",
  "validating",
];

export const TERMINAL_STATES: readonly CaliMachineState[] = ["passed", "failed", "cancelled"];

export const CALI_STATE_LABELS: Record<CaliMachineState, string> = {
  idle: "待机",
  checking_prerequisites: "检查前置条件",
  waiting_for_stability: "稳定性检测中",
  calibrating: "校准中",
  waiting_for_reboot: "等待重新上电",
  reconnecting: "重新连接中",
  validating: "自动验证中",
  passed: "通过",
  failed: "失败",
  cancelled: "已取消",
};

// --- Validity (freshness) badges ---------------------------------------------
export type ValidityState = "valid" | "due_soon" | "overdue" | "unknown";

export const VALIDITY_LABELS: Record<ValidityState, string> = {
  valid: "有效",
  due_soon: "即将过期",
  overdue: "已过期",
  unknown: "未标定",
};

/** Maps a validity/readiness concept onto the shared StatusDot palette. */
export type DotState = "running" | "warning" | "error" | "idle";

// --- Readiness checklist ------------------------------------------------------
export type ReadinessState = "complete" | "pending" | "warning" | "failed" | "unavailable";

export type ReadinessItem = {
  id: string;
  label: string;
  state: ReadinessState;
  /** Short status text (e.g. "已预热 12m" / "1/3" / "--"). */
  detail: string;
  /** Set when the datum is not yet available from the backend. */
  todo?: string;
  /** Optional interactive controls (e.g. start-timer / confirm / reset). */
  action?: ReactNode;
};

// --- Calibration history record (spec §11) -----------------------------------
// The gateway has no history endpoint yet, so records are produced locally by
// the workflow hook and persisted through the repository adapter. The shape is
// defined here so a future backend can serve the same contract.
export type CalibrationRecord = {
  id: string;
  timestamp: number;
  operator: string;
  boxId: string;
  sensorId: string;
  kind: CalibrationKind;
  before: ForceVec | null;
  after: ForceVec | null;
  pass: boolean;
  /** Per-axis (force) or per-side (touch) failure notes. */
  notes: string;
  softwareVersion: string;
};

export type CalibrationHistoryFilter = {
  boxId?: string;
  sensorId?: string;
  kind?: CalibrationKind | "all";
  outcome?: "all" | "pass" | "fail";
};
