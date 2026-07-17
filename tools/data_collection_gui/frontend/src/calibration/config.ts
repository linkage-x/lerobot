// Centralized calibration configuration.
//
// Every magic number the calibration center relies on lives here so thresholds,
// validity windows, staleness and stability tuning are never duplicated across
// JSX. Components and the workflow hook import from this module only.
//
// Units: forces in newtons (N), moments in newton-metres (N·m), time in
// milliseconds unless a constant name says otherwise.

export const MS = { second: 1000, minute: 60_000, hour: 3_600_000, day: 86_400_000 } as const;

/** Per-axis pass/fail limits for a 6D force sensor after calibration. */
export type ForceAxisLimits = {
  /** Absolute-value ceiling for Fx (N). */
  fxMaxN: number;
  /** Absolute-value ceiling for Fy (N). */
  fyMaxN: number;
  /**
   * Fz acceptance. `mode: "abs"` checks |Fz| <= max (raw/origin zeroing).
   * `mode: "target"` checks |Fz - target| <= tol (dynamic keeps a real load).
   */
  fz: { mode: "abs"; maxN: number } | { mode: "target"; targetN: number; tolN: number };
  /** Absolute-value ceiling for Mx/My/Mz (N·m), shared across the three axes. */
  momentMaxNm: number;
};

// --- 6D force: raw/origin zero (sensor hardware, expects fully unloaded) ------
// Origin zeroing drives every channel to ~0, so Fz is an absolute check.
export const ORIGIN_FORCE_LIMITS: ForceAxisLimits = {
  fxMaxN: 0.5,
  fyMaxN: 0.5,
  fz: { mode: "abs", maxN: 0.5 },
  momentMaxNm: 0.01,
};

// --- 6D force: dynamic (filter algorithm, keeps the tool/gravity load) --------
// After dynamic calibration Fz should sit at the loaded target, NOT near zero,
// so this deliberately does not reuse the origin Fz check.
export const DYNAMIC_FORCE_LIMITS: ForceAxisLimits = {
  fxMaxN: 0.5,
  fyMaxN: 0.5,
  fz: { mode: "target", targetN: -5.8, tolN: 0.5 },
  momentMaxNm: 0.01,
};

// --- Tactile (touch pad) ------------------------------------------------------
// After touch re-zero the net force and per-taxel residual target is 0. Real
// hardware never reaches exactly 0, so we accept within an explicit epsilon and
// always surface the epsilon actually used in the UI.
export const TOUCH_TOLERANCE = {
  /** Net |force| (sum over taxels, projected) accepted as "zeroed" (N). */
  netForceEpsilonN: 0.3,
  /** Largest single-taxel residual accepted (0.1 N units, matching fz_0p1N). */
  maxTaxelResidual0p1N: 8,
} as const;

// --- Validity windows: how long each calibration stays trustworthy -----------
// `dueSoonMs` is the lead time before expiry where we warn ("due soon").
export type ValidityWindow = { validMs: number; dueSoonMs: number };

export const VALIDITY: Record<CalibrationKind, ValidityWindow> = {
  // Raw/origin zero: once per day.
  force_origin: { validMs: 1 * MS.day, dueSoonMs: 2 * MS.hour },
  // Dynamic: once per 30 min; warn in the last 5 min (spec §5).
  force_dynamic: { validMs: 30 * MS.minute, dueSoonMs: 5 * MS.minute },
  // Touch: no hard spec cadence; treat like origin (once per day) until the
  // backend exposes a real policy. TODO(backend): confirm touch cadence.
  touch: { validMs: 1 * MS.day, dueSoonMs: 2 * MS.hour },
};

// --- Live-data staleness ------------------------------------------------------
// A preview sample older than this is shown as "stale" and must not be treated
// as a valid live reading (drives the monitor cards + stability sampling).
export const STALE_SAMPLE_MS = 1500;

// --- Pre-calibration stability check -----------------------------------------
// Before firing a calibration we sample the live preview for a short window and
// require the signal to be quiet (peak-to-peak within limits). If we cannot
// collect enough fresh samples the check reports "unavailable" rather than
// pretending the rig is stable.
export const STABILITY = {
  /** How long to sample the live signal before deciding (ms). */
  windowMs: 2500,
  /** Minimum fresh samples required to make a decision. */
  minSamples: 8,
  /** Max peak-to-peak on any force axis to call the rig stable (N). */
  forcePeakToPeakN: 0.4,
  /** Max peak-to-peak on any moment axis to call the rig stable (N·m). */
  momentPeakToPeakNm: 0.02,
  /** Max net-force peak-to-peak for a touch pad to call it stable (N). */
  touchNetPeakToPeakN: 0.6,
} as const;

// --- Async safety -------------------------------------------------------------
/** A calibration request that streams no terminal line within this window fails. */
export const CALIBRATION_TIMEOUT_MS = 45_000;
/** How often we poll the calibration log while a run is active (ms). */
export const CALI_LOG_POLL_MS = 300;
/** How often the monitor/stability sampling polls the box preview (ms). */
export const PREVIEW_POLL_MS = 300;

// --- Calibration kinds --------------------------------------------------------
export type CalibrationKind = "force_origin" | "force_dynamic" | "touch";

export const CALIBRATION_KIND_LABELS: Record<CalibrationKind, string> = {
  force_origin: "六维力原始零点校准",
  force_dynamic: "六维力动态校准",
  touch: "触觉标定",
};

// --- Formatting helpers (single source of truth for units/precision) ---------
const N_DASH = "—";

/** Format a force value in N with fixed precision; null/NaN render as em dash. */
export function fmtN(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return N_DASH;
  return `${value.toFixed(digits)} N`;
}

/** Format a moment value in N·m with fixed precision. */
export function fmtNm(value: number | null | undefined, digits = 3): string {
  if (value == null || !Number.isFinite(value)) return N_DASH;
  return `${value.toFixed(digits)} N·m`;
}

/** Bare fixed-precision number (no unit); null/NaN render as em dash. */
export function fmtNum(value: number | null | undefined, digits = 2): string {
  if (value == null || !Number.isFinite(value)) return N_DASH;
  return value.toFixed(digits);
}

/** Human duration like "4m 12s" / "2h 5m" / "820ms". Negative clamps to 0. */
export function fmtDuration(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return N_DASH;
  const clamped = Math.max(0, ms);
  if (clamped < MS.second) return `${Math.round(clamped)}ms`;
  const totalSec = Math.floor(clamped / 1000);
  const h = Math.floor(totalSec / 3600);
  const m = Math.floor((totalSec % 3600) / 60);
  const s = totalSec % 60;
  if (h > 0) return `${h}h ${m}m`;
  if (m > 0) return `${m}m ${s}s`;
  return `${s}s`;
}

/** Unified wall-clock timestamp (local): "YYYY-MM-DD HH:mm:ss". */
export function fmtTimestamp(ms: number | null | undefined): string {
  if (ms == null || !Number.isFinite(ms)) return N_DASH;
  const d = new Date(ms);
  const pad = (n: number) => String(n).padStart(2, "0");
  return (
    `${d.getFullYear()}-${pad(d.getMonth() + 1)}-${pad(d.getDate())} ` +
    `${pad(d.getHours())}:${pad(d.getMinutes())}:${pad(d.getSeconds())}`
  );
}
