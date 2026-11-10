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

  if (axis === "mx") {
    // Mx — mode differs between origin (abs) and dynamic (target load).
    if (limits.mx.mode === "abs") {
      const pass = Math.abs(value) <= limits.mx.maxN;
      return { axis, label, value, kind, pass, reason: pass ? "" : `|Mx| 超出 ±${limits.mx.maxN} N·m` };
    }
    const { targetN, tolN } = limits.mx;
    const pass = Math.abs(value - targetN) <= tolN;
    return { axis, label, value, kind, pass, reason: pass ? "" : `Mx 偏离目标 ${targetN} ± ${tolN} N·m` };
  }
  if (isMoment) {
    // My/Mz — always an absolute ceiling.
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
  return summarize(FORCE_AXES.map((axis) => evalAxis(axis, vec[axis], limits)));
}

/** Per-axis tolerance of a "target"-style limit set, read off the spec's own numbers. */
function axisTolerance(axis: ForceAxis, limits: ForceAxisLimits): number {
  if (axis === "fx") return limits.fxMaxN;
  if (axis === "fy") return limits.fyMaxN;
  if (axis === "fz") return limits.fz.mode === "target" ? limits.fz.tolN : limits.fz.maxN;
  if (axis === "mx") return limits.mx.mode === "target" ? limits.mx.tolN : limits.mx.maxN;
  return limits.momentMaxNm;
}

/**
 * Dynamic (software-zero) acceptance at whatever attitude the BOX rests in.
 *
 * The spec's numbers (Fz -7.784, Mx -0.168, the other four ~0) are the payload's weight as the
 * sensor sees it gripper-down. Tilt the BOX and that same weight leaks into Fx/Fy/My/Mz: 5 deg
 * already puts 0.68 N on Fx and 0.04 N*m on My (0.7935 kg x 9.81 x the 56.5 mm CoM lever), so the
 * fixed numbers really test "is it within ~1.3 deg of vertical", which a handheld BOX standing on its
 * own feet does not meet (11-03, left collector: acc (-0.997, 0.094, 0.025), every failing axis is
 * that tilt to the mN*m). The SDK publishes the payload model at the live IMU attitude as
 * `fxyz_mxyz_gravity`; compare against it with the spec's tolerances. Gripper-down it IS the spec,
 * and its norm is always the payload weight, so a run that zeroed the load away still fails.
 */
export function evaluateForceAgainstLoad(vec: ForceVec, load: ForceVec, limits: ForceAxisLimits): ForceEval {
  return summarize(
    FORCE_AXES.map((axis) => {
      const label = FORCE_AXIS_LABELS[axis];
      const kind: "force" | "moment" = axis.startsWith("m") ? "moment" : "force";
      const tol = axisTolerance(axis, limits);
      const value = vec[axis];
      const expected = load[axis];
      const pass = Math.abs(value - expected) <= tol;
      const unit = kind === "moment" ? "N·m" : "N";
      const digits = kind === "moment" ? 3 : 2;
      return {
        axis,
        label,
        value,
        kind,
        pass,
        reason: pass ? "" : `${label} 偏离当前姿态下的负载重力 ${expected.toFixed(digits)} ± ${tol} ${unit}`,
      };
    }),
  );
}

function summarize(axes: AxisEval[]): ForceEval {
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
