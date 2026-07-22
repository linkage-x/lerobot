// Operation log, moved out of the sensor cards into its own panel (spec §10).
// Aggregates the two calibration stdout channels (6D force + touch) into a
// unified, filterable, copyable view that only renders a bounded tail.

import { useMemo, useState } from "react";
import type { BoxCaliLogLine } from "../types";
import { fmtTimestamp } from "./config";

export type CaliLogLevel = "info" | "warn" | "error";
export type CaliLogSensor = "6d_force" | "touch" | "system";

export type CaliLogEntry = {
  ts: number;
  level: CaliLogLevel;
  boxId: string;
  sensor: CaliLogSensor;
  message: string;
  raw: string;
};

const BOX_PREFIX_RE = /^\[(?<label>[^\]]*)\]\s*(?<rest>.*)$/;

function levelOf(line: string): CaliLogLevel {
  const lower = line.toLowerCase();
  if (/failed|error|超出|timeout|超时/.test(lower)) return "error";
  if (/ignored|warn|不可用|陈旧/.test(lower)) return "warn";
  return "info";
}

/** Derive unified log entries from the raw force/touch buffers. */
export function toLogEntries(
  force: BoxCaliLogLine[],
  touch: BoxCaliLogLine[],
): CaliLogEntry[] {
  const build = (lines: BoxCaliLogLine[], sensor: CaliLogSensor): CaliLogEntry[] =>
    lines.map((l) => {
      const match = BOX_PREFIX_RE.exec(l.line);
      const boxId = match?.groups?.label || "";
      const message = match?.groups?.rest || l.line;
      return {
        ts: l.ts * 1000,
        level: levelOf(l.line),
        boxId,
        sensor,
        message,
        raw: l.line,
      };
    });
  return [...build(force, "6d_force"), ...build(touch, "touch")].sort((a, b) => a.ts - b.ts);
}

const SENSOR_LABELS: Record<CaliLogSensor | "all", string> = {
  all: "全部传感器",
  "6d_force": "六维力",
  touch: "触觉",
  system: "系统",
};

const LEVEL_LABELS: Record<CaliLogLevel | "all", string> = {
  all: "全部级别",
  info: "信息",
  warn: "警告",
  error: "错误",
};

const TAIL = 60;

export function CalibrationLogPanel({ entries }: { entries: CaliLogEntry[] }) {
  const [sensor, setSensor] = useState<CaliLogSensor | "all">("all");
  const [level, setLevel] = useState<CaliLogLevel | "all">("all");
  const [cleared, setCleared] = useState<number>(0);
  const [expanded, setExpanded] = useState<Set<number>>(new Set());

  const filtered = useMemo(() => {
    return entries
      .filter((e) => e.ts > cleared)
      .filter((e) => (sensor === "all" ? true : e.sensor === sensor))
      .filter((e) => (level === "all" ? true : e.level === level))
      .slice(-TAIL);
  }, [entries, cleared, sensor, level]);

  const copy = () => {
    const text = filtered
      .map((e) => `${fmtTimestamp(e.ts)}\t${e.level}\t${e.boxId || "-"}\t${e.sensor}\t${e.raw}`)
      .join("\n");
    void navigator.clipboard?.writeText(text);
  };

  const toggle = (idx: number) =>
    setExpanded((prev) => {
      const next = new Set(prev);
      next.has(idx) ? next.delete(idx) : next.add(idx);
      return next;
    });

  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>操作日志</h2>
        <span>{filtered.length} 条（最近 {TAIL}）</span>
      </div>
      <div className="cali-log-controls">
        <select value={sensor} onChange={(e) => setSensor(e.target.value as CaliLogSensor | "all")}>
          {(["all", "6d_force", "touch"] as const).map((s) => (
            <option key={s} value={s}>
              {SENSOR_LABELS[s]}
            </option>
          ))}
        </select>
        <select value={level} onChange={(e) => setLevel(e.target.value as CaliLogLevel | "all")}>
          {(["all", "info", "warn", "error"] as const).map((l) => (
            <option key={l} value={l}>
              {LEVEL_LABELS[l]}
            </option>
          ))}
        </select>
        <button onClick={() => setCleared(Date.now())}>清除视图</button>
        <button onClick={copy}>复制日志</button>
      </div>
      <div className="cali-log-list">
        {filtered.length === 0 ? (
          <div className="cali-log-empty">暂无日志</div>
        ) : (
          filtered.map((e, i) => (
            <div className={`cali-log-row cali-log-${e.level}`} key={`${e.ts}-${i}`}>
              <button className="cali-log-line" onClick={() => toggle(i)} title="点击展开原始消息">
                <span className="cali-log-ts">{fmtTimestamp(e.ts)}</span>
                <span className={`cali-log-badge cali-log-badge-${e.level}`}>{LEVEL_LABELS[e.level]}</span>
                <span className="cali-log-box">{e.boxId || "-"}</span>
                <span className="cali-log-sensor">{SENSOR_LABELS[e.sensor]}</span>
                <span className="cali-log-msg">{e.message}</span>
              </button>
              {expanded.has(i) && <pre className="cali-log-raw">{e.raw}</pre>}
            </div>
          ))
        )}
      </div>
    </section>
  );
}
