// The calibration state machine (spec §8) as a React hook. One instance drives
// one calibration *type* (origin / dynamic / touch), which the backend performs
// fleet-wide, so the hook samples every relevant device and reports per-box (or
// per-side) results.
//
//   idle → checking_prerequisites → waiting_for_stability → calibrating
//        → [waiting_for_reboot → reconnecting] → validating → passed | failed
//   (cancel from any in-flight state → cancelled)
//
// Nothing here fakes success: a request that streams no terminal line inside
// CALIBRATION_TIMEOUT_MS fails; a stability check that cannot gather fresh
// samples reports "unavailable" rather than "stable".

import { useCallback, useEffect, useRef, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { BoxCaliLog, BoxCaliLogLine, DeviceStatus } from "../types";
import {
  CALIBRATION_TIMEOUT_MS,
  CALI_LOG_POLL_MS,
  PREVIEW_POLL_MS,
  STABILITY,
  TOUCH_TOLERANCE,
  type CalibrationKind,
  type ForceAxisLimits,
} from "./config";
import { evaluateForce, parseCaliLog } from "./parseCaliLog";
import {
  deviceBoxId,
  softwareVersion,
  touchMaxResidual0p1N,
  touchNetForceN,
} from "./adapters";
import { forceVecFromSensor, numberArray } from "./useBoxPreview";
import type {
  CaliMachineState,
  CalibrationRecord,
  ForceEval,
  ForceVec,
} from "./types";

export type StabilityStatus = "idle" | "sampling" | "stable" | "unstable" | "unavailable";

export type StabilityResult = {
  status: StabilityStatus;
  samples: number;
  /** Representative peak-to-peak (worst axis / net); null when undecided. */
  peakToPeak: number | null;
  detail: string;
};

export type ForceBoxResult = {
  boxId: string;
  ok: boolean | null;
  error: string | null;
  before: ForceVec | null;
  after: ForceVec | null;
  evaluation: ForceEval | null;
  pass: boolean;
};

export type TouchSideResult = {
  deviceId: string;
  label: string;
  netN: number | null;
  maxResidual: number | null;
  pass: boolean;
};

export type WorkflowConfig = {
  api: DataCollectionGuiApi;
  kind: CalibrationKind;
  /** Fires the existing backend endpoint (whole fleet). */
  trigger: () => Promise<{ ok: boolean; error?: string }>;
  fetchLog: () => Promise<BoxCaliLog | null>;
  /** Live devices to sample for stability + validation fallback. */
  sampleDevices: DeviceStatus[];
  sampleKind: "force" | "touch";
  /** Origin zeroing needs a power-cycle before re-reading. */
  requiresReboot: boolean;
  /** Per-axis thresholds (force kinds only). */
  limits: ForceAxisLimits | null;
  /** Returns a blocking reason (e.g. "正在录制") or null when clear. */
  guard: () => string | null;
  operator: string;
  onRecord?: (records: CalibrationRecord[]) => void;
};

export type WorkflowApi = {
  state: CaliMachineState;
  error: string | null;
  stability: StabilityResult;
  logLines: BoxCaliLogLine[];
  forceResults: ForceBoxResult[];
  touchResults: TouchSideResult[];
  /** null until validated. */
  overallPass: boolean | null;
  begin: () => void;
  confirm: () => void;
  confirmReboot: () => void;
  cancel: () => void;
  reset: () => void;
};

export type StabilitySample = { t: number; sourceId: string; vec: ForceVec | null; net: number | null };

const IDLE_STABILITY: StabilityResult = { status: "idle", samples: 0, peakToPeak: null, detail: "" };

function peakToPeak(values: number[]): number {
  if (values.length === 0) return 0;
  return Math.max(...values) - Math.min(...values);
}

export function evaluateStability(
  samples: StabilitySample[],
  sampleKind: "force" | "touch",
  sourceIds: string[] = [],
): StabilityResult {
  const fresh = samples.filter((s) => (sampleKind === "force" ? s.vec : s.net) != null);
  const ids = sourceIds.length ? sourceIds : [...new Set(fresh.map((s) => s.sourceId))];
  if (ids.length === 0) {
    return { status: "sampling", samples: 0, peakToPeak: null, detail: "采样中…" };
  }

  const bySource = ids.map((sourceId) => ({
    sourceId,
    samples: fresh.filter((s) => s.sourceId === sourceId),
  }));
  const fewestSamples = Math.min(...bySource.map((group) => group.samples.length));
  if (fewestSamples < STABILITY.minSamples) {
    return { status: "sampling", samples: fewestSamples, peakToPeak: null, detail: "采样中…" };
  }

  if (sampleKind === "force") {
    const axes: (keyof ForceVec)[] = ["fx", "fy", "fz", "mx", "my", "mz"];
    let worstForce = 0;
    let worstMoment = 0;
    let worstSource = "";
    for (const group of bySource) {
      for (const axis of axes) {
        const p2p = peakToPeak(group.samples.map((s) => (s.vec as ForceVec)[axis]));
        if (axis === "mx" || axis === "my" || axis === "mz") {
          if (p2p > worstMoment) worstMoment = p2p;
        } else if (p2p > worstForce) {
          worstForce = p2p;
          worstSource = group.sourceId;
        }
      }
    }
    const stable = worstForce <= STABILITY.forcePeakToPeakN && worstMoment <= STABILITY.momentPeakToPeakNm;
    const suffix = worstSource ? `（${worstSource}）` : "";
    return {
      status: stable ? "stable" : "unstable",
      samples: fewestSamples,
      peakToPeak: worstForce,
      detail: `力峰峰值 ${worstForce.toFixed(2)} N / 限 ${STABILITY.forcePeakToPeakN} N${suffix}`,
    };
  }

  let worstNet = 0;
  let worstSource = "";
  for (const group of bySource) {
    const p2p = peakToPeak(group.samples.map((s) => s.net as number));
    if (p2p > worstNet) {
      worstNet = p2p;
      worstSource = group.sourceId;
    }
  }
  const stable = worstNet <= STABILITY.touchNetPeakToPeakN;
  const suffix = worstSource ? `（${worstSource}）` : "";
  return {
    status: stable ? "stable" : "unstable",
    samples: fewestSamples,
    peakToPeak: worstNet,
    detail: `净力峰峰值 ${worstNet.toFixed(2)} N / 限 ${STABILITY.touchNetPeakToPeakN} N${suffix}`,
  };
}

export function useCalibrationWorkflow(cfg: WorkflowConfig): WorkflowApi {
  const [state, setState] = useState<CaliMachineState>("idle");
  const [error, setError] = useState<string | null>(null);
  const [stability, setStability] = useState<StabilityResult>(IDLE_STABILITY);
  const [logLines, setLogLines] = useState<BoxCaliLogLine[]>([]);
  const [forceResults, setForceResults] = useState<ForceBoxResult[]>([]);
  const [touchResults, setTouchResults] = useState<TouchSideResult[]>([]);
  const [overallPass, setOverallPass] = useState<boolean | null>(null);

  // Timers / mutable run context held in refs so effects stay stable.
  const stabilityTimer = useRef<number | null>(null);
  const stabilityStart = useRef(0);
  const samples = useRef<StabilitySample[]>([]);
  const logTimer = useRef<number | null>(null);
  const logStartIndex = useRef(0);
  const calibrateDeadline = useRef(0);
  const mounted = useRef(true);
  const cfgRef = useRef(cfg);
  cfgRef.current = cfg;
  // Mirror `state` so event handlers (confirm/confirmReboot) can guard on the
  // freshest value WITHOUT doing side effects inside a setState updater — React
  // StrictMode double-invokes updater functions in dev, which would start the
  // polling loop twice and leak the first interval (→ runaway validate → the
  // page "flickers"). Handlers run once, so guarding via a ref is safe.
  const stateRef = useRef(state);
  stateRef.current = state;

  const clearTimers = useCallback(() => {
    if (stabilityTimer.current != null) window.clearInterval(stabilityTimer.current);
    if (logTimer.current != null) window.clearInterval(logTimer.current);
    stabilityTimer.current = null;
    logTimer.current = null;
  }, []);

  useEffect(() => {
    mounted.current = true;
    return () => {
      mounted.current = false;
      clearTimers();
    };
  }, [clearTimers]);

  // --- stability sampling loop (runs while waiting_for_stability) ------------
  const startStabilitySampling = useCallback(() => {
    samples.current = [];
    stabilityStart.current = Date.now();
    setStability({ status: "sampling", samples: 0, peakToPeak: null, detail: "采样中…" });
    const tick = async () => {
      const { api, sampleDevices, sampleKind } = cfgRef.current;
      await Promise.all(
        sampleDevices.map(async (dev) => {
          const preview = await api.fetchBoxPreview(dev.id);
          const sensor = preview?.sensor ?? null;
          const now = Date.now();
          if (sampleKind === "force") {
            samples.current.push({ t: now, sourceId: dev.id, vec: forceVecFromSensor(sensor), net: null });
          } else {
            const net = touchNetForceN(numberArray(sensor?.["fz_0p1N"]));
            samples.current.push({ t: now, sourceId: dev.id, vec: null, net });
          }
        }),
      );
      // Keep only the trailing window.
      const cutoff = Date.now() - STABILITY.windowMs;
      samples.current = samples.current.filter((s) => s.t >= cutoff);
      if (!mounted.current) return;
      const result = evaluateStability(samples.current, sampleKind, sampleDevices.map((dev) => dev.id));
      // If we cannot gather enough fresh samples in twice the window, the live
      // feed is unavailable — say so instead of implying stability.
      if (
        result.status === "sampling" &&
        Date.now() - stabilityStart.current > STABILITY.windowMs * 2
      ) {
        setStability({
          status: "unavailable",
          samples: result.samples,
          peakToPeak: null,
          detail: "无法获取足够的实时采样，稳定性检测不可用",
        });
        return;
      }
      setStability(result);
    };
    tick();
    stabilityTimer.current = window.setInterval(tick, PREVIEW_POLL_MS);
  }, []);

  const begin = useCallback(() => {
    setError(null);
    setForceResults([]);
    setTouchResults([]);
    setOverallPass(null);
    setState("checking_prerequisites");
    // Prerequisite guard (recorder connected / not recording, etc.).
    const reason = cfgRef.current.guard();
    if (reason) {
      setError(reason);
      setState("failed");
      return;
    }
    setState("waiting_for_stability");
    startStabilitySampling();
  }, [startStabilitySampling]);

  // logLines mirror so validate()/poll can read the latest without re-creating
  // the callback each render. Declared before validate() which reads it.
  const logLinesRef = useRef<BoxCaliLogLine[]>([]);
  logLinesRef.current = logLines;

  // --- final validation: parse log + fall back to live for after-values -----
  const validate = useCallback(async () => {
    setState("validating");
    const { api, sampleDevices, sampleKind, limits, kind, operator, onRecord } = cfgRef.current;

    // Live snapshot per box for fallback / touch residuals.
    const liveByBox = new Map<string, { force: ForceVec | null; net: number | null; residual: number | null }>();
    const liveByDevice = new Map<string, { force: ForceVec | null; net: number | null; residual: number | null; label: string; boxId: string }>();
    await Promise.all(
      sampleDevices.map(async (dev) => {
        const preview = await api.fetchBoxPreview(dev.id);
        const sensor = preview?.sensor ?? null;
        const fz = numberArray(sensor?.["fz_0p1N"]);
        const entry = {
          force: forceVecFromSensor(sensor),
          net: touchNetForceN(fz),
          residual: touchMaxResidual0p1N(fz),
        };
        liveByBox.set(deviceBoxId(dev), entry);
        liveByDevice.set(dev.id, { ...entry, label: dev.label || dev.id, boxId: deviceBoxId(dev) });
      }),
    );

    const parsed = parseCaliLog(logLinesRef.current, logStartIndex.current);
    const version = softwareVersion();
    const records: CalibrationRecord[] = [];

    if (sampleKind === "force" && limits) {
      const singleLiveForce = liveByBox.size === 1 ? [...liveByBox.values()][0]?.force ?? null : null;
      const results: ForceBoxResult[] = parsed.results.map((r) => {
        const after = r.after ?? liveByBox.get(r.boxId)?.force ?? singleLiveForce;
        const evaluation = after ? evaluateForce(after, limits) : null;
        const pass = (r.ok !== false) && !!evaluation?.pass;
        records.push({
          id: `${Date.now()}-${r.boxId}-${kind}`,
          timestamp: Date.now(),
          operator,
          boxId: r.boxId,
          sensorId: `${r.boxId || "box"}/box_six_d_force`,
          kind,
          before: r.before,
          after,
          pass,
          notes: pass ? "" : evaluation?.firstFailure ?? r.error ?? "校准未通过",
          softwareVersion: version,
        });
        return { boxId: r.boxId, ok: r.ok, error: r.error, before: r.before, after, evaluation, pass };
      });
      // If the log yielded no per-box lines, fall back to live-only evaluation.
      if (results.length === 0) {
        for (const [id, entry] of liveByDevice) {
          const evaluation = entry.force ? evaluateForce(entry.force, limits) : null;
          const pass = !!evaluation?.pass;
          results.push({ boxId: entry.boxId, ok: null, error: null, before: null, after: entry.force, evaluation, pass });
          records.push({
            id: `${Date.now()}-${id}-${kind}`,
            timestamp: Date.now(),
            operator,
            boxId: entry.boxId,
            sensorId: id,
            kind,
            before: null,
            after: entry.force,
            pass,
            notes: pass ? "" : evaluation?.firstFailure ?? "校准未通过",
            softwareVersion: version,
          });
        }
      }
      if (!mounted.current) return;
      setForceResults(results);
      const all = results.length > 0 && results.every((r) => r.pass) && parsed.outcome !== "error";
      setOverallPass(all);
      onRecord?.(records);
      setState(all ? "passed" : "failed");
      if (!all) setError("部分传感器未通过校准，见下方逐轴结果");
      return;
    }

    // Touch: no per-axis vectors in the log, derive per-side residual from live.
    const results: TouchSideResult[] = [...liveByDevice].map(([deviceId, entry]) => {
      const netOk = entry.net != null && Math.abs(entry.net) <= TOUCH_TOLERANCE.netForceEpsilonN;
      const resOk = entry.residual != null && entry.residual <= TOUCH_TOLERANCE.maxTaxelResidual0p1N;
      const pass = netOk && resOk && parsed.outcome !== "error";
      records.push({
        id: `${Date.now()}-${deviceId}-${kind}`,
        timestamp: Date.now(),
        operator,
        boxId: entry.boxId,
        sensorId: deviceId,
        kind,
        before: null,
        after: null,
        pass,
        notes: pass ? "" : "净力/残差超出容差",
        softwareVersion: version,
      });
      return { deviceId, label: entry.label, netN: entry.net, maxResidual: entry.residual, pass };
    });
    if (!mounted.current) return;
    setTouchResults(results);
    const all = results.length > 0 && results.every((r) => r.pass);
    setOverallPass(all);
    onRecord?.(records);
    setState(all ? "passed" : "failed");
    if (!all) setError("部分触觉传感器未通过校准，见下方结果");
  }, []);

  // --- calibrating: fire trigger, poll the log until terminal / timeout -----
  const startCalibrating = useCallback(async () => {
    const { trigger, fetchLog, requiresReboot } = cfgRef.current;
    setState("calibrating");
    // Anchor parsing to the current buffer length so we only read this run.
    const pre = await fetchLog();
    logStartIndex.current = pre?.lines.length ?? 0;
    if (pre) setLogLines(pre.lines);
    const res = await trigger();
    if (!res.ok) {
      setError(res.error ?? "校准启动失败");
      setState("failed");
      const after = await fetchLog();
      if (after) setLogLines(after.lines);
      return;
    }
    calibrateDeadline.current = Date.now() + CALIBRATION_TIMEOUT_MS;
    const poll = async () => {
      const log = await fetchLog();
      if (!mounted.current) return;
      if (log) setLogLines(log.lines);
      const parsed = log ? parseCaliLog(log.lines, logStartIndex.current) : null;
      const finished = parsed?.done || (log ? !log.running : false);
      if (finished) {
        if (logTimer.current != null) window.clearInterval(logTimer.current);
        logTimer.current = null;
        if (requiresReboot) {
          setState("waiting_for_reboot");
        } else {
          validate();
        }
        return;
      }
      if (Date.now() > calibrateDeadline.current) {
        if (logTimer.current != null) window.clearInterval(logTimer.current);
        logTimer.current = null;
        setError("校准请求超时，未收到完成信号");
        setState("failed");
      }
    };
    poll();
    logTimer.current = window.setInterval(poll, CALI_LOG_POLL_MS);
  }, [validate]);

  const confirm = useCallback(() => {
    // Guard + side effects live in the event handler (runs once), never in a
    // setState updater (StrictMode runs those twice → duplicate/leaked timers).
    if (stateRef.current !== "waiting_for_stability") return;
    if (stabilityTimer.current != null) window.clearInterval(stabilityTimer.current);
    stabilityTimer.current = null;
    startCalibrating(); // sets state → "calibrating" itself
  }, [startCalibrating]);

  const confirmReboot = useCallback(() => {
    if (stateRef.current !== "waiting_for_reboot") return;
    // reconnecting: wait for a device to report a fresh live sample again.
    setState("reconnecting");
    const started = Date.now();
    const poll = async () => {
      const { api, sampleDevices } = cfgRef.current;
      const previews = await Promise.all(sampleDevices.map((d) => api.fetchBoxPreview(d.id)));
      const anyActive = previews.some((p) => p?.active);
      if (!mounted.current) return;
      if (anyActive) {
        if (logTimer.current != null) window.clearInterval(logTimer.current);
        logTimer.current = null;
        validate();
      } else if (Date.now() - started > CALIBRATION_TIMEOUT_MS) {
        if (logTimer.current != null) window.clearInterval(logTimer.current);
        logTimer.current = null;
        setError("重新连接超时，请检查 BOX 是否已重新上电");
        setState("failed");
      }
    };
    poll();
    logTimer.current = window.setInterval(poll, CALI_LOG_POLL_MS);
  }, [validate]);

  const cancel = useCallback(() => {
    clearTimers();
    setState((cur) => (cur === "idle" ? cur : "cancelled"));
  }, [clearTimers]);

  const reset = useCallback(() => {
    clearTimers();
    setError(null);
    setStability(IDLE_STABILITY);
    setForceResults([]);
    setTouchResults([]);
    setOverallPass(null);
    setState("idle");
  }, [clearTimers]);

  return {
    state,
    error,
    stability,
    logLines,
    forceResults,
    touchResults,
    overallPass,
    begin,
    confirm,
    confirmReboot,
    cancel,
    reset,
  };
}
