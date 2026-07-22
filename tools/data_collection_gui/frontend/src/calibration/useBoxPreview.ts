// Shared live-preview polling for a single BOX sensor. Mirrors the polling loop
// the Device Manager's BoxSensorTile uses, but exposes typed force/touch views
// plus a freshness flag so the monitor cards and the stability sampler agree on
// what "stale" means.

import { useEffect, useRef, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { BoxPreviewPayload } from "../types";
import { PREVIEW_POLL_MS, STALE_SAMPLE_MS } from "./config";
import type { ForceVec } from "./types";
import { touchMaxResidual0p1N, touchNetForceN } from "./adapters";

/** Coerce an unknown JSON value to a number[] (empty when not array-like). */
export function numberArray(value: unknown): number[] {
  if (!Array.isArray(value)) return [];
  return value.map((v) => (typeof v === "number" ? v : Number(v))).filter((v) => Number.isFinite(v));
}

/** Extract [Fx,Fy,Fz,Mx,My,Mz] from a preview sensor payload. */
export function forceVecFromSensor(sensor: Record<string, unknown> | null | undefined): ForceVec | null {
  const arr = numberArray(sensor?.["fxyz_mxyz"]);
  if (arr.length < 6) return null;
  return { fx: arr[0], fy: arr[1], fz: arr[2], mx: arr[3], my: arr[4], mz: arr[5] };
}

export type BoxPreviewView = {
  payload: BoxPreviewPayload | null;
  sensor: Record<string, unknown> | null;
  /** Seconds since the sample was produced; null when unknown. */
  staleS: number | null;
  /** True when live and the sample is fresh enough to trust. */
  fresh: boolean;
  /** Parsed 6D force vector (force sensors only). */
  force: ForceVec | null;
  /** Touch pad taxel array in 0.1 N units (touch sensors only). */
  touchFz0p1N: number[];
  touchNetN: number | null;
  touchMaxResidual: number | null;
};

const EMPTY_VIEW: BoxPreviewView = {
  payload: null,
  sensor: null,
  staleS: null,
  fresh: false,
  force: null,
  touchFz0p1N: [],
  touchNetN: null,
  touchMaxResidual: null,
};

function toView(payload: BoxPreviewPayload | null): BoxPreviewView {
  if (!payload) return EMPTY_VIEW;
  const sensor = payload.sensor ?? null;
  const staleS = payload.staleS ?? null;
  const fresh = Boolean(payload.active) && staleS != null && staleS * 1000 <= STALE_SAMPLE_MS;
  const touchFz0p1N = numberArray(sensor?.["fz_0p1N"]);
  return {
    payload,
    sensor,
    staleS,
    fresh,
    force: forceVecFromSensor(sensor),
    touchFz0p1N,
    touchNetN: touchNetForceN(touchFz0p1N),
    touchMaxResidual: touchMaxResidual0p1N(touchFz0p1N),
  };
}

export function useBoxPreview(
  api: DataCollectionGuiApi,
  deviceId: string,
  enabled = true,
): BoxPreviewView {
  const [payload, setPayload] = useState<BoxPreviewPayload | null>(null);
  const mountedRef = useRef(true);

  useEffect(() => {
    mountedRef.current = true;
    if (!enabled) {
      setPayload(null);
      return () => {
        mountedRef.current = false;
      };
    }
    const load = async () => {
      const next = await api.fetchBoxPreview(deviceId);
      if (mountedRef.current) setPayload(next);
    };
    load();
    const timer = window.setInterval(load, PREVIEW_POLL_MS);
    return () => {
      mountedRef.current = false;
      window.clearInterval(timer);
    };
  }, [api, deviceId, enabled]);

  return toView(payload);
}
