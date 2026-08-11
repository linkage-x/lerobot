// Camera self-check: run it whenever you suspect something got bumped.
//
// It compares each camera's current view against a baseline frame taken when
// the rig was last calibrated. What it observes is the camera's *pose*, so a
// positive means the extrinsics are stale; the lens is untouched by a bump, so
// intrinsics normally survive. That distinction is spelled out in the panel
// because "redo the calibration" is otherwise read as "redo all of it".
import { useEffect, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { RigCheckBaseline, RigCheckReport } from "../types";
import { StatusDot } from "../shared/ui";
import {
  baselineSummary,
  canUpdateBaseline,
  overallDot,
  overallLabel,
  rigCheckRows,
  shouldOfferRecalibration,
  verdictDot,
  verdictLabel,
} from "./rigCheck";

export function RigCheckPanel({
  api,
  busy,
  onRecalibrate,
}: {
  api: DataCollectionGuiApi;
  busy: boolean;
  onRecalibrate: () => void;
}) {
  const [report, setReport] = useState<RigCheckReport | null>(null);
  const [baseline, setBaseline] = useState<RigCheckBaseline | undefined>(undefined);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string>("");

  useEffect(() => {
    let cancelled = false;
    api.fetchRigCheck().then((payload) => {
      if (cancelled || !payload) return;
      setReport(payload.report ?? null);
      setBaseline(payload.baseline);
    });
    return () => {
      cancelled = true;
    };
  }, [api]);

  const run = async (action: "check" | "baseline") => {
    setRunning(true);
    setError("");
    const payload = action === "check" ? await api.runRigCheck() : await api.captureRigCheckBaseline();
    setRunning(false);
    if (!payload.ok) {
      // The common failure is that the recorder owns the cameras. Showing the
      // gateway's own sentence keeps the fix in the message.
      setError(payload.error || payload.hint || "自检失败");
      return;
    }
    if (action === "check") {
      setReport(payload.report ?? null);
      setBaseline(payload.report?.baseline ?? payload.baseline);
    } else {
      setBaseline(payload.baseline);
      setReport(null);
    }
  };

  const rows = rigCheckRows(report);
  const disabled = busy || running;
  const offerRecalibration = shouldOfferRecalibration(report);

  return (
    <section className="panel calibration-panel">
      <div className="panel-heading">
        <h2>相机自检</h2>
        {report ? (
          <span className="state-pill">
            <StatusDot state={overallDot[report.overall]} />
            {overallLabel[report.overall]}
          </span>
        ) : null}
      </div>

      <div className="control-row">
        <button className="cali-btn-primary" disabled={disabled} onClick={() => run("check")}>
          {running ? "自检中…" : "运行自检"}
        </button>
        <button
          disabled={disabled || !canUpdateBaseline(report)}
          title={
            canUpdateBaseline(report)
              ? "把当前画面记为新的比较基线"
              : "检测到移动时不能更新基线——那会把移动后的状态记成参考"
          }
          onClick={() => run("baseline")}
        >
          {baseline?.exists ? "重新采集基线" : "采集基线"}
        </button>
      </div>

      <p className="panel-note">基线：{baselineSummary(report, baseline)}</p>

      {error ? <p className="panel-note error">{error}</p> : null}

      {rows.length > 0 ? (
        <div className="check-table calibration-table">
          {rows.map((row) => (
            <div className="check-row" key={row.camera}>
              <strong>
                <StatusDot state={verdictDot[row.verdict]} />
                {row.camera}
              </strong>
              <span>
                {row.shift}
                {row.impact ? ` · ${row.impact}` : ""}
                {row.note ? ` · ${row.note}` : ""}
              </span>
              <em>{verdictLabel[row.verdict]}</em>
            </div>
          ))}
        </div>
      ) : null}

      {report ? <p className="panel-note">{report.guidance}</p> : null}

      {offerRecalibration ? (
        <div className="control-row">
          <button className="cali-btn-primary" disabled={disabled} onClick={onRecalibrate}>
            前往外参标定
          </button>
          <span className="panel-note">
            仅需重做外参：相机被移动改变的是位姿，镜头未变，内参无需重标。
          </span>
        </div>
      ) : null}

      {!report && !error ? (
        <p className="panel-note">
          用于不定期确认固定相机是否被碰过。判据是与基线相比的变化，而不是相机之间的绝对分歧——
          后者主要由标定板/marker 的几何参数决定，重做标定也不会改善。
        </p>
      ) : null}
    </section>
  );
}
