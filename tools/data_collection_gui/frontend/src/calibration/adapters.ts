// Adapters that isolate "data the backend does not expose yet" behind an
// explicit availability contract. Nothing here fabricates a passing/complete
// status: when the datum is missing the adapter says so (`unavailable`) and
// carries a TODO describing the backend work required.

import type { DeviceStatus } from "../types";
import type { CalibrationHistoryFilter, CalibrationRecord, ReadinessItem } from "./types";

// --- Software / firmware version ---------------------------------------------
// TODO(backend): expose the recorder/box firmware version in the snapshot or a
// dedicated endpoint. Until then we read Vite's build-time define if present,
// else report unknown.
export function softwareVersion(): string {
  const v = (import.meta.env.VITE_APP_VERSION as string | undefined) ?? "";
  return v || "unknown";
}

// --- Derived touch quantities (from the live preview payload) -----------------
/** Net normal force across a touch pad (N) from per-taxel fz_0p1N (0.1 N units). */
export function touchNetForceN(fz0p1N: number[] | null | undefined): number | null {
  if (!fz0p1N || fz0p1N.length === 0) return null;
  const sum = fz0p1N.reduce((acc, v) => acc + (Number.isFinite(v) ? v : 0), 0);
  return sum / 10;
}

/** Largest single-taxel residual (0.1 N units). */
export function touchMaxResidual0p1N(fz0p1N: number[] | null | undefined): number | null {
  if (!fz0p1N || fz0p1N.length === 0) return null;
  return fz0p1N.reduce((max, v) => (Number.isFinite(v) ? Math.max(max, Math.abs(v)) : max), 0);
}

// --- Readiness adapters -------------------------------------------------------
// The two backend-derivable checks (box connected, calibration validity) are
// assembled by the page from the snapshot + history. The three below have no
// backend source yet and are honestly reported as unavailable.

/** Warm-up: needs a device power-on timestamp the gateway does not report. */
export function readinessWarmup(): ReadinessItem {
  return {
    id: "warmup",
    label: "设备已预热 30 分钟",
    state: "unavailable",
    detail: "--",
    todo: "backend: 暴露 box 上电时间戳以计算预热时长/倒计时",
  };
}

/** Tactile activation: needs a full-scale-activation counter from the SDK. */
export function readinessTactileActivation(): ReadinessItem {
  return {
    id: "tactile-activation",
    label: "触觉已完成 3 次满量程激活",
    state: "unavailable",
    detail: "0/3（不可用）",
    todo: "backend: 暴露触觉满量程激活计数 (0/3..3/3)",
  };
}

/** Environment temp/humidity: no sensor feed in the snapshot. */
export function readinessEnvironment(): ReadinessItem {
  return {
    id: "environment",
    label: "环境温湿度正常",
    state: "unavailable",
    detail: "-- / --",
    todo: "backend: 暴露环境温度/湿度读数",
  };
}

// --- Calibration history repository ------------------------------------------
// TODO(backend): replace with a real history endpoint. The interface is the
// contract the UI codes against; the local implementation persists to
// localStorage so records survive a page refresh on a single workstation.
export interface CalibrationHistoryRepo {
  list(filter?: CalibrationHistoryFilter): CalibrationRecord[];
  add(record: CalibrationRecord): void;
  clear(): void;
}

const HISTORY_KEY = "lerobot.calibration.history.v1";

function matchesFilter(r: CalibrationRecord, f?: CalibrationHistoryFilter): boolean {
  if (!f) return true;
  if (f.boxId && f.boxId !== "all" && r.boxId !== f.boxId) return false;
  if (f.sensorId && f.sensorId !== "all" && r.sensorId !== f.sensorId) return false;
  if (f.kind && f.kind !== "all" && r.kind !== f.kind) return false;
  if (f.outcome && f.outcome !== "all") {
    if (f.outcome === "pass" && !r.pass) return false;
    if (f.outcome === "fail" && r.pass) return false;
  }
  return true;
}

/** localStorage-backed history with an in-memory fallback (SSR/denied storage). */
export class LocalCalibrationHistoryRepo implements CalibrationHistoryRepo {
  private memory: CalibrationRecord[] = [];

  private read(): CalibrationRecord[] {
    try {
      const raw = window.localStorage.getItem(HISTORY_KEY);
      if (!raw) return [];
      const parsed = JSON.parse(raw);
      return Array.isArray(parsed) ? (parsed as CalibrationRecord[]) : [];
    } catch {
      return this.memory;
    }
  }

  private write(records: CalibrationRecord[]): void {
    this.memory = records;
    try {
      window.localStorage.setItem(HISTORY_KEY, JSON.stringify(records));
    } catch {
      /* storage denied — memory fallback already updated */
    }
  }

  list(filter?: CalibrationHistoryFilter): CalibrationRecord[] {
    return this.read()
      .filter((r) => matchesFilter(r, filter))
      .sort((a, b) => b.timestamp - a.timestamp);
  }

  add(record: CalibrationRecord): void {
    const next = [record, ...this.read()].slice(0, 500);
    this.write(next);
  }

  clear(): void {
    this.write([]);
  }
}

// Shared singleton so the sidebar badge and the calibration page read the same
// records without threading the repo through props.
export const historyRepo: CalibrationHistoryRepo = new LocalCalibrationHistoryRepo();

/** Serialize records to a downloadable JSON/CSV blob string. */
export function historyToJson(records: CalibrationRecord[]): string {
  return JSON.stringify(records, null, 2);
}

export function historyToCsv(records: CalibrationRecord[]): string {
  const header = [
    "timestamp",
    "operator",
    "boxId",
    "sensorId",
    "kind",
    "pass",
    "before",
    "after",
    "notes",
    "softwareVersion",
  ];
  const rows = records.map((r) => {
    const vec = (v: CalibrationRecord["before"]) =>
      v ? `Fx=${v.fx} Fy=${v.fy} Fz=${v.fz} Mx=${v.mx} My=${v.my} Mz=${v.mz}` : "";
    const cell = (s: string) => `"${String(s).replace(/"/g, '""')}"`;
    return [
      new Date(r.timestamp).toISOString(),
      r.operator,
      r.boxId,
      r.sensorId,
      r.kind,
      r.pass ? "pass" : "fail",
      vec(r.before),
      vec(r.after),
      r.notes,
      r.softwareVersion,
    ]
      .map(cell)
      .join(",");
  });
  return [header.join(","), ...rows].join("\n");
}

// --- Troubleshooting guidance (spec §9) --------------------------------------
export const FORCE_TROUBLESHOOTING: string[] = [
  "检查 BOX 是否完全空载",
  "检查是否使用标准标定工装",
  "检查设备是否受到按压或移动",
  "检查安装面是否平整",
  "检查转接板是否翘曲",
  "检查螺钉预紧是否一致",
  "检查线缆是否跨接传感器上下两侧",
  "检查是否发生碰撞或过载",
];

export const TOUCH_TROUBLESHOOTING: string[] = [
  "检查触觉表面是否完全空载",
  "检查是否有异物贴附在触觉阵列上",
  "检查触觉线缆连接是否牢固",
  "重新静置后再次校准",
];

// --- Device grouping helpers --------------------------------------------------
/** BOX device ids are namespaced `<box_id>/<sensor>`; strip to the bare sensor. */
export function boxSensorSuffix(deviceId: string): string {
  const slash = deviceId.lastIndexOf("/");
  return slash >= 0 ? deviceId.slice(slash + 1) : deviceId;
}

/** The box id carried on a device row ("" for a single unnamed box). */
export function deviceBoxId(device: DeviceStatus): string {
  const cfg = device.config ?? {};
  const raw = (cfg.box_id as string | undefined) ?? "";
  return raw || "";
}

/** Display name for a box: its id, or "BOX" when a single box is unnamed. */
export function boxDisplayName(boxId: string): string {
  return boxId || "BOX";
}
