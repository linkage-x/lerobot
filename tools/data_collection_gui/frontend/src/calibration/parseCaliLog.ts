// Parse the recorder's streamed calibration log into structured, per-box,
// per-axis results, and evaluate a force vector against the active thresholds.
//
// The gateway strips the `CALI_LOG `/`CALI_DONE ` (and touch equivalents)
// prefixes before buffering, so a `BoxCaliLog.lines[].line` looks like one of:
//   "6D force software-zero calibration command sent to recorder"   (gateway)
//   "6D force sensor software zero requested"                        (recorder)
//   "[box_a] cali_6d_force_sensor OK (rc=0)"
//   "[box_a] cali_6d_force_sensor FAILED (rc=-3): timeout"
//   "[box_a] before: Fx=0.1200, Fy=-0.8600, Fz=0.0800, Mx=..., My=..., Mz=..."
//   "[box_a] after:  Fx=0.0100, ..."
//   "ok" / "error"   (terminal, with done=true)
// Touch runs never emit before/after vectors.

import type { BoxCaliLogLine } from "../types";
import type { ForceAxisLimits } from "./config";
import type { AxisEval, ForceAxis, ForceEval, ForceVec, ParsedBoxResult, ParsedCaliLog } from "./types";
import { FORCE_AXES, FORCE_AXIS_LABELS } from "./types";

const FORCE_TOKEN_RE = /\b(Fx|Fy|Fz|Mx|My|Mz)=(-?\d+(?:\.\d+)?)/g;
const BOX_LINE_RE = /^\[(?<label>[^\]]*)\]\s+(?<rest>.*)$/;
const OK_RE = /^(?<method>.+?)\s+OK\s+\(rc=(?<rc>-?\d+)\)/;
const FAIL_RE = /^(?<method>.+?)\s+FAILED\s+\(rc=(?<rc>-?\d+)\):\s*(?<error>.*)$/;

/** Parse a "Fx=…, Fy=…" fragment into a ForceVec, or null for "n/a"/empty. */
export function parseForceVec(text: string): ForceVec | null {
  const trimmed = text.trim();
  if (!trimmed || trimmed.toLowerCase() === "n/a") return null;
  const found: Partial<Record<string, number>> = {};
  for (const match of trimmed.matchAll(FORCE_TOKEN_RE)) {
    found[match[1].toLowerCase()] = Number(match[2]);
  }
  const keys = ["fx", "fy", "fz", "mx", "my", "mz"] as const;
  if (!keys.every((k) => typeof found[k] === "number" && Number.isFinite(found[k]))) {
    return null;
  }
  return {
    fx: found.fx as number,
    fy: found.fy as number,
    fz: found.fz as number,
    mx: found.mx as number,
    my: found.my as number,
    mz: found.mz as number,
  };
}

/**
 * Parse a full log buffer (optionally only the tail from `startIndex`) into a
 * per-box result set plus the terminal outcome.
 *
 * Robust to the buffer holding several past runs: results accumulate per box
 * label and later lines overwrite earlier ones, so the last "before/after/OK"
 * for each box wins. Callers that want a single run pass `startIndex` captured
 * at trigger time.
 */
export function parseCaliLog(lines: BoxCaliLogLine[], startIndex = 0): ParsedCaliLog {
  const slice = lines.slice(Math.max(0, startIndex));
  const byBox = new Map<string, ParsedBoxResult>();
  let requested = false;
  let done = false;
  let outcome: "ok" | "error" | null = null;

  const ensure = (label: string): ParsedBoxResult => {
    const existing = byBox.get(label);
    if (existing) return existing;
    const created: ParsedBoxResult = {
      boxId: label,
      method: null,
      ok: null,
      rc: null,
      error: null,
      before: null,
      after: null,
    };
    byBox.set(label, created);
    return created;
  };

  for (const entry of slice) {
    const text = entry.line.trim();
    if (!text) continue;

    if (entry.done && (text === "ok" || text === "error")) {
      done = true;
      outcome = text;
      continue;
    }
    if (/requested$/.test(text) || /command sent to recorder$/.test(text)) {
      requested = true;
      continue;
    }

    const boxMatch = BOX_LINE_RE.exec(text);
    if (!boxMatch?.groups) continue;
    const label = boxMatch.groups.label || "box";
    const rest = boxMatch.groups.rest;
    const box = ensure(label);

    const okMatch = OK_RE.exec(rest);
    if (okMatch?.groups) {
      box.method = okMatch.groups.method;
      box.ok = true;
      box.rc = Number(okMatch.groups.rc);
      continue;
    }
    const failMatch = FAIL_RE.exec(rest);
    if (failMatch?.groups) {
      box.method = failMatch.groups.method;
      box.ok = false;
      box.rc = Number(failMatch.groups.rc);
      box.error = failMatch.groups.error || "unknown error";
      continue;
    }
    if (rest.startsWith("before:")) {
      box.before = parseForceVec(rest.slice("before:".length));
      continue;
    }
    if (rest.startsWith("after:")) {
      box.after = parseForceVec(rest.slice("after:".length));
      continue;
    }
  }

  return { requested, done, outcome, results: [...byBox.values()] };
}

/** Evaluate one axis of a force vector against a limit, producing a reason. */
function evalAxis(axis: ForceAxis, value: number, limits: ForceAxisLimits): AxisEval {
  const label = FORCE_AXIS_LABELS[axis];
  const isMoment = axis === "mx" || axis === "my" || axis === "mz";
  const kind: "force" | "moment" = isMoment ? "moment" : "force";

  if (isMoment) {
    const pass = Math.abs(value) <= limits.momentMaxNm;
    return {
      axis,
      label,
      value,
      kind,
      pass,
      reason: pass ? "" : `|${label}| 超出 ±${limits.momentMaxNm} N·m`,
    };
  }
  if (axis === "fx") {
    const pass = Math.abs(value) <= limits.fxMaxN;
    return { axis, label, value, kind, pass, reason: pass ? "" : `|Fx| 超出 ±${limits.fxMaxN} N` };
  }
  if (axis === "fy") {
    const pass = Math.abs(value) <= limits.fyMaxN;
    return { axis, label, value, kind, pass, reason: pass ? "" : `|Fy| 超出 ±${limits.fyMaxN} N` };
  }
  // fz — mode differs between origin (abs) and dynamic (target).
  if (limits.fz.mode === "abs") {
    const pass = Math.abs(value) <= limits.fz.maxN;
    return { axis, label, value, kind, pass, reason: pass ? "" : `|Fz| 超出 ±${limits.fz.maxN} N` };
  }
  const { targetN, tolN } = limits.fz;
  const pass = Math.abs(value - targetN) <= tolN;
  return {
    axis,
    label,
    value,
    kind,
    pass,
    reason: pass ? "" : `Fz 偏离目标 ${targetN} ± ${tolN} N`,
  };
}

/** Evaluate a whole force vector against the active per-axis thresholds. */
export function evaluateForce(vec: ForceVec, limits: ForceAxisLimits): ForceEval {
  const axes = FORCE_AXES.map((axis) => evalAxis(axis, vec[axis], limits));
  const firstFail = axes.find((a) => !a.pass) ?? null;
  return {
    axes,
    pass: axes.every((a) => a.pass),
    firstFailure: firstFail
      ? `${firstFail.label} = ${firstFail.value.toFixed(firstFail.kind === "moment" ? 3 : 2)}` +
        `${firstFail.kind === "moment" ? " N·m" : " N"}，${firstFail.reason}`
      : null,
  };
}
