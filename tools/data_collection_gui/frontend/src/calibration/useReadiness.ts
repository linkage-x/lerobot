// Operator-driven readiness hooks for data the box SDK does not expose:
//  - warm-up: a per-box 30-min timer (start on power-on, or confirm "already
//    warm" to skip), persisted so it survives navigation/refresh.
//  - tactile activation: guides the operator to press each pad to full scale
//    N times, detected from the live max-taxel reading with hysteresis.
// Both persist per box in localStorage keyed by box_id.

import { useCallback, useEffect, useRef, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { DeviceStatus } from "../types";
import {
  PREVIEW_POLL_MS,
  TACTILE_ACTIVATE_0p1N,
  TACTILE_ACTIVATION_TARGET,
  TACTILE_RELEASE_0p1N,
  WARMUP_MS,
  fmtDuration,
} from "./config";
import type { ReadinessState } from "./types";
import { boxSensorSuffix, touchMaxResidual0p1N } from "./adapters";
import { numberArray } from "./useBoxPreview";

function readJSON<T>(key: string, fallback: T): T {
  try {
    const raw = window.localStorage.getItem(key);
    return raw ? (JSON.parse(raw) as T) : fallback;
  } catch {
    return fallback;
  }
}
function writeJSON(key: string, value: unknown): void {
  try {
    window.localStorage.setItem(key, JSON.stringify(value));
  } catch {
    /* storage denied — in-memory state still holds for this session */
  }
}

// --- warm-up -----------------------------------------------------------------
type WarmupPersist = { startMs: number | null; confirmed: boolean };

export type WarmupApi = {
  state: ReadinessState;
  detail: string;
  start: () => void;
  confirm: () => void;
  reset: () => void;
  running: boolean;
};

export function useWarmup(boxId: string, now: number): WarmupApi {
  const key = `lerobot.calibration.warmup.${boxId}`;
  const [persist, setPersist] = useState<WarmupPersist>(() =>
    readJSON<WarmupPersist>(key, { startMs: null, confirmed: false }),
  );

  const update = useCallback(
    (next: WarmupPersist) => {
      setPersist(next);
      writeJSON(key, next);
    },
    [key],
  );

  const start = useCallback(() => update({ startMs: Date.now(), confirmed: false }), [update]);
  const confirm = useCallback(() => update({ startMs: null, confirmed: true }), [update]);
  const reset = useCallback(() => update({ startMs: null, confirmed: false }), [update]);

  let state: ReadinessState;
  let detail: string;
  let running = false;
  if (persist.confirmed) {
    state = "complete";
    detail = "已确认预热完毕";
  } else if (persist.startMs != null) {
    const elapsed = now - persist.startMs;
    if (elapsed >= WARMUP_MS) {
      state = "complete";
      detail = `已预热 ${fmtDuration(elapsed)}`;
    } else {
      state = "pending";
      running = true;
      detail = `预热中 ${fmtDuration(elapsed)} · 剩 ${fmtDuration(WARMUP_MS - elapsed)}`;
    }
  } else {
    state = "pending";
    detail = "未开始";
  }

  return { state, detail, start, confirm, reset, running };
}

// --- tactile full-scale activation -------------------------------------------
export type TactileActivationApi = {
  state: ReadinessState;
  detail: string;
  /** Progress of the limiting (least-pressed) pad; complete once every pad hits target. */
  count: number;
  target: number;
  reset: () => void;
};

/** Short side label for a touch pad ("左"/"右", else the device label). */
function padSideLabel(dev: DeviceStatus): string {
  const suffix = boxSensorSuffix(dev.id).toLowerCase();
  if (suffix.includes("left")) return "左";
  if (suffix.includes("right")) return "右";
  return dev.label || dev.id;
}

export function useTactileActivation(
  api: DataCollectionGuiApi,
  boxId: string,
  touchDevices: DeviceStatus[],
  enabled: boolean,
): TactileActivationApi {
  const key = `lerobot.calibration.tactile-activation.${boxId}`;
  const target = TACTILE_ACTIVATION_TARGET;
  // Per-pad counts (keyed by device id): each pad must be pressed to full scale
  // `target` times, so pressing only one side never completes the readiness.
  const [counts, setCounts] = useState<Record<string, number>>(() =>
    readJSON<Record<string, number>>(key, {}),
  );
  // Per-pad hysteresis latch: true once that pad has released below the low
  // threshold, so a single sustained press on it is counted exactly once.
  const armedRef = useRef<Record<string, boolean>>({});
  const countsRef = useRef(counts);
  countsRef.current = counts;

  const reset = useCallback(() => {
    setCounts({});
    writeJSON(key, {});
    armedRef.current = {};
  }, [key]);

  const deviceIds = touchDevices.map((d) => d.id).join(",");
  const ids = deviceIds ? deviceIds.split(",") : [];
  const perPad = ids.map((id) => Math.min(target, Math.max(0, counts[id] ?? 0)));
  const minCount = perPad.length ? Math.min(...perPad) : 0;
  const complete = perPad.length > 0 && perPad.every((c) => c >= target);

  useEffect(() => {
    if (!enabled || complete || touchDevices.length === 0) return;
    let cancelled = false;
    const idList = deviceIds ? deviceIds.split(",") : [];
    const tick = async () => {
      const previews = await Promise.all(idList.map((id) => api.fetchBoxPreview(id)));
      if (cancelled) return;
      const next = { ...countsRef.current };
      let changed = false;
      idList.forEach((id, i) => {
        const r = touchMaxResidual0p1N(numberArray(previews[i]?.sensor?.["fz_0p1N"]));
        if (r == null) return;
        if (r <= TACTILE_RELEASE_0p1N) {
          armedRef.current[id] = true;
        } else if (r >= TACTILE_ACTIVATE_0p1N && (armedRef.current[id] ?? true)) {
          armedRef.current[id] = false;
          const cur = Math.min(target, (next[id] ?? 0) + 1);
          if (cur !== (next[id] ?? 0)) {
            next[id] = cur;
            changed = true;
          }
        }
      });
      if (changed) {
        countsRef.current = next;
        setCounts(next);
        writeJSON(key, next);
      }
    };
    const timer = window.setInterval(tick, PREVIEW_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [api, deviceIds, enabled, complete, key, target, touchDevices.length]);

  const summary = touchDevices.map((d) => `${padSideLabel(d)} ${Math.min(target, Math.max(0, counts[d.id] ?? 0))}/${target}`).join(" · ");
  return {
    state: complete ? "complete" : perPad.some((c) => c > 0) ? "warning" : "pending",
    detail: complete ? `${summary} 已激活` : `${summary} · 请分别将每个 pad 按压至满量程`,
    count: minCount,
    target,
    reset,
  };
}
