import { useEffect, useMemo, useState } from "react";
import { api } from "./apiClient";
import {
  beamLossSummary,
  errorBudgetSummary,
  mm,
  niceCeiling,
  polylineSegments,
  runLabel,
  vec
} from "./laserReplayAnalysis";
import type { LaserReplayReport, LaserReplayRun, RealCubeMode, ReplayStatus } from "./types";

const CHART_W = 640;
const CHART_H = 160;
const SERIES = [
  { key: "laser_vs_desired_mm", label: "laser vs desired", color: "#c2410c" },
  { key: "fk_vs_desired_mm", label: "FR3 FK vs desired", color: "#64748b" },
  { key: "laser_vs_fk_mm", label: "laser vs FR3 FK", color: "#0f766e" }
] as const;

function ErrorChart({ run }: { run: LaserReplayRun }) {
  const perFrame = run.per_frame;
  const yMax = useMemo(() => {
    if (!perFrame) return 1;
    const all = SERIES.flatMap((series) => perFrame[series.key]).filter((v): v is number => v != null);
    return niceCeiling(all.length ? Math.max(...all) : 1);
  }, [perFrame]);
  if (!perFrame) return null;
  const n = perFrame.frame_index.length;
  const lossX =
    run.first_loss_frame != null && run.first_loss_frame >= 0 && n > 1
      ? (perFrame.frame_index.indexOf(run.first_loss_frame) / (n - 1)) * CHART_W
      : null;
  return (
    <figure className="laser-error-chart">
      <svg viewBox={`-34 -8 ${CHART_W + 44} ${CHART_H + 26}`} role="img" aria-label={`Per-frame error of ${run.run}`}>
        {[0, 0.5, 1].map((f) => (
          <g key={f}>
            <line x1={0} x2={CHART_W} y1={CHART_H * (1 - f)} y2={CHART_H * (1 - f)} className="laser-error-grid" />
            <text x={-6} y={CHART_H * (1 - f) + 4} textAnchor="end" className="laser-error-axis">
              {(yMax * f).toFixed(yMax * f < 10 && f > 0 ? 1 : 0)}
            </text>
          </g>
        ))}
        <text x={0} y={CHART_H + 16} className="laser-error-axis">frame {perFrame.frame_index[0]}</text>
        <text x={CHART_W} y={CHART_H + 16} textAnchor="end" className="laser-error-axis">
          {perFrame.frame_index[n - 1]}
        </text>
        {lossX != null ? (
          <g>
            <line x1={lossX} x2={lossX} y1={0} y2={CHART_H} className="laser-error-loss" />
            <text x={lossX + 4} y={10} className="laser-error-axis">beam lost · {run.first_loss_frame}</text>
          </g>
        ) : null}
        {SERIES.map((series) =>
          polylineSegments(perFrame[series.key], CHART_W, CHART_H, yMax).map((points, index) => (
            <polyline key={`${series.key}-${index}`} points={points} fill="none" stroke={series.color} strokeWidth={1.4} />
          ))
        )}
      </svg>
      <figcaption>
        {SERIES.map((series) => (
          <span key={series.key}>
            <i style={{ background: series.color }} /> {series.label}
          </span>
        ))}
        <span>mm at the SMR point, per replay frame</span>
      </figcaption>
    </figure>
  );
}

export function LaserReplayAnalysisPanel({
  status,
  busy,
  onAnalyze
}: {
  status: ReplayStatus;
  busy: boolean;
  onAnalyze: (cube: RealCubeMode) => void;
}) {
  const [cube, setCube] = useState<RealCubeMode>(status.realCubeMode ?? "right");
  const [report, setReport] = useState<LaserReplayReport | null>(null);
  const [availableRuns, setAvailableRuns] = useState<string[]>([]);
  const [selectedRun, setSelectedRun] = useState<string>("");
  const datasetPath = status.datasetRoot || status.dataset;
  const analysis = status.laserAnalysis ?? {};
  const running = analysis.status === "running";
  const analysisIsForSelection =
    analysis.datasetRoot === datasetPath && analysis.episode === status.episode && analysis.cube === cube;
  const replayActive = status.state === "replaying" || status.state === "sim_replay";

  useEffect(() => {
    if (!datasetPath) {
      setReport(null);
      setAvailableRuns([]);
      return;
    }
    let mounted = true;
    api.fetchLaserReplayReport(datasetPath, status.episode, cube).then((next) => {
      if (!mounted) return;
      setReport(next.report);
      setAvailableRuns(next.availableRuns);
    });
    return () => {
      mounted = false;
    };
    // A finished replay lands a new run; a finished analysis rewrites the report.
  }, [datasetPath, status.episode, cube, analysis.updatedAt, analysis.status, status.realTrackerState, status.revision]);

  const analysed = (report?.runs ?? []).filter((run) => run.laser_vs_desired);
  const chartRun = analysed.find((run) => run.run === selectedRun) ?? analysed[analysed.length - 1];
  const notAnalysed = availableRuns.filter((name) => !(report?.runs ?? []).some((run) => run.run === name));
  const repeat = report?.repeatability;

  return (
    <section className="panel laser-analysis-panel">
      <div className="panel-heading">
        <h2>Laser vs FR3 Replay Error</h2>
        <span>episode {status.episode} · every real replay of it that recorded the laser tracker</span>
      </div>
      <div className="control-row laser-analysis-controls">
        <label>
          <span>Cube</span>
          <select value={cube} disabled={running} onChange={(event) => setCube(event.target.value as RealCubeMode)}>
            <option value="right">right</option>
            <option value="left">left</option>
          </select>
        </label>
        <button
          disabled={busy || running || replayActive || !availableRuns.length}
          onClick={() => onAnalyze(cube)}
          type="button"
        >
          {running ? "Analysing…" : report ? "Re-analyse laser runs" : "Analyse laser runs"}
        </button>
        <span className="panel-note">
          {availableRuns.length
            ? `${availableRuns.length} run(s) with the tracker` +
              (notAnalysed.length && report ? ` · ${notAnalysed.length} not in the report yet` : "")
            : "No real replay of this episode recorded the laser tracker."}
          {report?.modifiedAt ? ` · report ${report.modifiedAt}` : ""}
        </span>
      </div>
      {analysisIsForSelection && analysis.status === "failed" ? (
        <pre className="real-replay-log">{[analysis.message, ...(analysis.output ?? [])].join("\n")}</pre>
      ) : null}
      {analysisIsForSelection && running ? <p className="panel-note">{analysis.message}</p> : null}
      {report ? (
        <>
          <div className="table-scroll">
            <table className="tracker-strata laser-analysis-table">
              <thead>
                <tr>
                  <th>Run</th>
                  <th>Beam frames used</th>
                  <th>Lost at</th>
                  <th>Laser vs desired RMS / p95 / max</th>
                  <th>FK vs desired RMS</th>
                  <th>Laser vs FK RMS / p95</th>
                  <th>Offset (x, y, z)</th>
                  <th>Lag</th>
                  <th>Rest RMS</th>
                </tr>
              </thead>
              <tbody>
                {report.runs.map((run) =>
                  run.laser_vs_desired ? (
                    <tr
                      key={run.run}
                      className={run === chartRun ? "selected" : undefined}
                      onClick={() => setSelectedRun(run.run)}
                      title={run.run}
                    >
                      <td>{runLabel(run.run)}</td>
                      <td>{run.used_frames} / {run.frames}</td>
                      <td>{run.first_loss_frame != null && run.first_loss_frame >= 0 ? run.first_loss_frame : "—"}</td>
                      <td>
                        {mm(run.laser_vs_desired.rms_mm)} / {mm(run.laser_vs_desired.p95_mm)} / {mm(run.laser_vs_desired.max_mm)}
                      </td>
                      <td>{mm(run.fk_vs_desired?.rms_mm)}</td>
                      <td>{mm(run.laser_vs_fk?.rms_mm)} / {mm(run.laser_vs_fk?.p95_mm)}</td>
                      <td>{vec(run.laser_vs_desired_bias_mm)}</td>
                      <td>{mm(run.decomposition?.follow_lag_ms, 0)} ms</td>
                      <td>{mm(run.decomposition?.without_bias_and_lag_rms_mm)}</td>
                    </tr>
                  ) : (
                    <tr key={run.run} title={run.run}>
                      <td>{runLabel(run.run)}</td>
                      <td colSpan={8} className="laser-analysis-error">{run.error}</td>
                    </tr>
                  )
                )}
              </tbody>
            </table>
          </div>
          <p className="panel-note">
            All in mm at the SMR point; desired = the tracked trajectory replayed, FK = the pose the FR3 reports.
            {repeat?.pairwise_max
              ? ` Run-to-run spread over ${repeat.frames} common frames: p50 ${mm(repeat.pairwise_max.p50_mm)}, ` +
                `p95 ${mm(repeat.pairwise_max.p95_mm)} mm; SMR offset agrees to ${mm(repeat.smr_in_tcp_spread_mm)} mm.`
              : repeat?.note
                ? ` ${repeat.note}.`
                : ""}
          </p>
          {chartRun ? (
            <div className="laser-analysis-detail">
              <ErrorChart run={chartRun} />
              <p className="panel-note">
                <strong>{runLabel(chartRun.run)}</strong> · {errorBudgetSummary(chartRun)}
              </p>
              <p className="panel-note">{beamLossSummary(chartRun)}</p>
              <p className="panel-note">
                SMR in the TCP frame {vec(chartRun.smr_in_tcp_mm)} mm (±{vec(chartRun.smr_in_tcp_sigma_mm, 2)}), fitted
                together with tracker → base over {mm(chartRun.orientation_span_deg, 1)}° of orientation. The fit is to
                the FR3's FK, so any rigid FK error is absorbed and laser vs FK is a lower bound.
              </p>
            </div>
          ) : null}
        </>
      ) : null}
    </section>
  );
}
