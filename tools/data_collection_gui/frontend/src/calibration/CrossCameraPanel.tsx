// Cross-camera consistency: do the calibrated cameras agree about the cube in
// the workspace, right now?
//
// The third of the checks on this page and the only one that is not a
// comparison with the past: the self-check and world continuity both carry an
// extrinsic that was wrong from the start inside their reference. Run it after
// exporting new extrinsics and again before a laser tracker session.
import { useEffect, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { CrossCameraCandidate, CrossCameraReport } from "../types";
import { StatusDot } from "../shared/ui";
import {
  crossCameraRows,
  overallDot,
  overallLabel,
  setChangeSummary,
  staleReason,
  verdictDot,
  verdictLabel,
} from "./crossCamera";

export function CrossCameraPanel({ api, busy }: { api: DataCollectionGuiApi; busy: boolean }) {
  const [report, setReport] = useState<CrossCameraReport | null>(null);
  const [candidates, setCandidates] = useState<CrossCameraCandidate[]>([]);
  const [extrinsicsRun, setExtrinsicsRun] = useState<string | undefined>(undefined);
  const [dataset, setDataset] = useState("");
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");

  const refresh = async () => {
    const payload = await api.fetchCrossCameraCheck();
    if (!payload) return;
    setReport(payload.report ?? null);
    setExtrinsicsRun(payload.extrinsicsRun);
    const list = payload.candidates ?? [];
    setCandidates(list);
    setDataset((current) => current || list[0]?.dataset || "");
  };

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [api]);

  const run = async () => {
    setRunning(true);
    setError("");
    const payload = await api.runCrossCameraCheck(dataset);
    setRunning(false);
    if (!payload.ok || !payload.report) {
      setError(payload.error || "检查失败");
      return;
    }
    setReport(payload.report);
  };

  const rows = crossCameraRows(report);
  const stale = staleReason(report, extrinsicsRun);

  return (
    <section className="panel calibration-panel">
      <div className="panel-heading">
        <h2>跨相机一致性</h2>
        {report ? (
          <span className="state-pill">
            <StatusDot state={overallDot[report.overall]} />
            {overallLabel[report.overall]}
          </span>
        ) : null}
      </div>

      <div className="control-row">
        <select value={dataset} disabled={busy || running} onChange={(e) => setDataset(e.target.value)}>
          {candidates.length === 0 ? <option value="">没有已生成 EE 轨迹的数据集</option> : null}
          {candidates.map((c) => (
            <option key={c.dataset} value={c.dataset}>
              {c.name} · {c.cameras.length} 台相机 · 轨迹 {new Date(c.trajectoryModifiedUnixS * 1000).toLocaleString()}
            </option>
          ))}
        </select>
        <button className="cali-btn-primary" disabled={busy || running || !dataset} onClick={() => void run()}>
          {running ? "检查中…" : "运行检查"}
        </button>
        <button disabled={running} onClick={() => void refresh()}>
          刷新列表
        </button>
      </div>

      {error ? <p className="panel-note error">{error}</p> : null}
      {stale ? <p className="panel-note error">{stale}</p> : null}

      {report ? (
        <p className="panel-note">
          {report.dataset.split("/").pop()} · 检查于 {report.generated_utc}
          {report.sidecar_generated_utc ? ` · 轨迹生成于 ${report.sidecar_generated_utc}` : ""}
          {report.extrinsics_run ? ` · 外参 ${report.extrinsics_run}` : ""}
        </p>
      ) : null}

      {rows.length > 0 ? (
        <div className="check-table calibration-table">
          {rows.map((row) => (
            <div className="check-row" key={row.key}>
              <strong>
                <StatusDot state={verdictDot[row.verdict]} />
                {row.camera}
                {row.serial ? ` · ${row.serial}` : ""}
              </strong>
              <span>
                横向偏差 {row.offset} · {row.detail}
              </span>
              <em>{verdictLabel[row.verdict]}</em>
            </div>
          ))}
        </div>
      ) : null}

      {report
        ? Object.entries(report.cubes).map(([cube, entry]) => (
            <p className="panel-note" key={cube}>
              <StatusDot state={verdictDot[entry.set_changes.verdict]} />
              相机集切换（{cube}）：{setChangeSummary(entry.set_changes)}
            </p>
          ))
        : null}

      {report ? <p className="panel-note">{report.guidance}</p> : null}

      <p className="panel-note">
        用一段 cube 在工作区里移动、同时被至少 3 台相机看到的录制（先生成 EE 轨迹）。外参导出后跑一次，采集跟踪仪数据前再跑一次。
        判据是每台相机垂直于视线方向的恒定偏差（门槛 {report?.thresholds.warn_mm ?? 2}/{report?.thresholds.fail_mm ?? 4} mm）；
        沿视线的深度差主要来自 marker 尺寸，不参与判定。所有相机共同的偏移看不见，只有跟踪仪能测。
      </p>
    </section>
  );
}
