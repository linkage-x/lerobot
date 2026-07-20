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
import { touchMaxResidual0p1N } from "./adapters";
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
  count: number;
  target: number;
  reset: () => void;
};

export function useTactileActivation(
  api: DataCollectionGuiApi,
  boxId: string,
  touchDevices: DeviceStatus[],
  enabled: boolean,
): TactileActivationApi {
  const key = `lerobot.calibration.tactile-activation.${boxId}`;
  const target = TACTILE_ACTIVATION_TARGET;
  const [count, setCount] = useState<number>(() => {
    const n = Number(readJSON<number>(key, 0));
    return Number.isFinite(n) ? Math.max(0, Math.min(target, n)) : 0;
  });
  // Hysteresis latch: true once the pads have released below the low threshold,
  // so a single sustained press is counted exactly once.
  const armedRef = useRef(true);
  const countRef = useRef(count);
  countRef.current = count;

  const reset = useCallback(() => {
    setCount(0);
    writeJSON(key, 0);
    armedRef.current = true;
  }, [key]);

  const deviceIds = touchDevices.map((d) => d.id).join(",");
  const complete = count >= target;

  useEffect(() => {
    if (!enabled || complete || touchDevices.length === 0) return;
    let cancelled = false;
    const ids = deviceIds ? deviceIds.split(",") : [];
    const tick = async () => {
      const previews = await Promise.all(ids.map((id) => api.fetchBoxPreview(id)));
      if (cancelled) return;
      let max = 0;
      for (const p of previews) {
        const r = touchMaxResidual0p1N(numberArray(p?.sensor?.["fz_0p1N"]));
        if (r != null) max = Math.max(max, r);
      }
      if (max <= TACTILE_RELEASE_0p1N) {
        armedRef.current = true;
      } else if (max >= TACTILE_ACTIVATE_0p1N && armedRef.current) {
        armedRef.current = false;
        const next = Math.min(target, countRef.current + 1);
        countRef.current = next;
        setCount(next);
        writeJSON(key, next);
      }
    };
    const timer = window.setInterval(tick, PREVIEW_POLL_MS);
    return () => {
      cancelled = true;
      window.clearInterval(timer);
    };
  }, [api, deviceIds, enabled, complete, key, target, touchDevices.length]);

  return {
    state: complete ? "complete" : count > 0 ? "warning" : "pending",
    detail: complete ? `${count}/${target} 已激活` : `${count}/${target} · 请用力按压触觉至满量程`,
    count,
    target,
    reset,
  };
}
