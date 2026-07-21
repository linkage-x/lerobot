// Calibration history (spec §11). Records are produced locally by the workflow
// hook and read back through the repository adapter; the UI codes against the
// CalibrationRecord contract so a future backend endpoint drops in unchanged.

import { useMemo, useState } from "react";
import { CALIBRATION_KIND_LABELS, fmtTimestamp, type CalibrationKind } from "./config";
import { historyToCsv, historyToJson } from "./adapters";
import type { CalibrationHistoryFilter, CalibrationRecord, ForceVec } from "./types";
import { FORCE_AXES, FORCE_AXIS_LABELS } from "./types";

function download(name: string, mime: string, content: string) {
  const blob = new Blob([content], { type: mime });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = name;
  a.click();
  URL.revokeObjectURL(url);
}

function vecText(vec: ForceVec | null): string {
  if (!vec) return "—";
  return FORCE_AXES.map((axis) => `${FORCE_AXIS_LABELS[axis]} ${vec[axis].toFixed(axis.startsWith("m") ? 3 : 2)}`).join("  ");
}

export function CalibrationHistory({
  records,
  onClear,
}: {
  records: CalibrationRecord[];
  onClear: () => void;
}) {
  const [filter, setFilter] = useState<CalibrationHistoryFilter>({ kind: "all", outcome: "all" });
  const [openId, setOpenId] = useState<string | null>(null);

  const boxIds = useMemo(() => [...new Set(records.map((r) => r.boxId || "box"))], [records]);

  const filtered = useMemo(
    () =>
      records.filter((r) => {
        if (filter.boxId && filter.boxId !== "all" && (r.boxId || "box") !== filter.boxId) return false;
        if (filter.kind && filter.kind !== "all" && r.kind !== filter.kind) return false;
        if (filter.outcome === "pass" && !r.pass) return false;
        if (filter.outcome === "fail" && r.pass) return false;
        return true;
      }),
    [records, filter],
  );

  return (
    <section className="panel">
      <div className="panel-heading">
        <h2>标定记录</h2>
        <span>{filtered.length} / {records.length} 条</span>
      </div>
      <div className="cali-log-controls">
        <select
          value={filter.boxId ?? "all"}
          onChange={(e) => setFilter((f) => ({ ...f, boxId: e.target.value }))}
        >
          <option value="all">全部 BOX</option>
          {boxIds.map((b) => (
            <option key={b} value={b}>
              {b}
            </option>
          ))}
        </select>
        <select
          value={filter.kind ?? "all"}
          onChange={(e) => setFilter((f) => ({ ...f, kind: e.target.value as CalibrationKind | "all" }))}
        >
          <option value="all">全部类型</option>
          {(Object.keys(CALIBRATION_KIND_LABELS) as CalibrationKind[]).map((k) => (
            <option key={k} value={k}>
              {CALIBRATION_KIND_LABELS[k]}
            </option>
          ))}
        </select>
        <select
          value={filter.outcome ?? "all"}
          onChange={(e) => setFilter((f) => ({ ...f, outcome: e.target.value as "all" | "pass" | "fail" }))}
        >
          <option value="all">全部结果</option>
          <option value="pass">通过</option>
          <option value="fail">失败</option>
        </select>
        <button disabled={!filtered.length} onClick={() => download("calibration-history.json", "application/json", historyToJson(filtered))}>
          导出 JSON
        </button>
        <button disabled={!filtered.length} onClick={() => download("calibration-history.csv", "text/csv", historyToCsv(filtered))}>
          导出 CSV
        </button>
        <button disabled={!records.length} onClick={onClear}>
          清空记录
        </button>
      </div>
      {filtered.length === 0 ? (
        <p className="cali-muted">暂无标定记录（TODO：后端历史接口就绪后改为服务端读取）。</p>
      ) : (
        <div className="cali-history">
          {filtered.map((r) => (
            <div className="cali-history-row" key={r.id}>
              <button className="cali-history-head" onClick={() => setOpenId((id) => (id === r.id ? null : r.id))}>
                <span className={`cali-badge cali-badge-${r.pass ? "running" : "error"}`}>
                  <span className={`status-dot status-${r.pass ? "running" : "error"}`} />
                  {r.pass ? "PASS" : "FAIL"}
                </span>
                <span className="cali-history-time">{fmtTimestamp(r.timestamp)}</span>
                <span>{r.boxId || "box"}</span>
                <span>{CALIBRATION_KIND_LABELS[r.kind]}</span>
                <span className="cali-muted">{r.operator || "—"}</span>
              </button>
              {openId === r.id && (
                <div className="cali-history-detail">
                  <div><b>传感器</b> {r.sensorId}</div>
                  <div><b>校准前</b> <code>{vecText(r.before)}</code></div>
                  <div><b>校准后</b> <code>{vecText(r.after)}</code></div>
                  <div><b>备注</b> {r.notes || "—"}</div>
                  <div><b>软件版本</b> {r.softwareVersion}</div>
                </div>
              )}
            </div>
          ))}
        </div>
      )}
    </section>
  );
}
