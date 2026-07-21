import type { GuiSnapshot } from "../api";
import { Metric, StatusDot, stateLabel } from "../shared/ui";

function calibrationDotState(state: GuiSnapshot["calibration"]["state"]): string {
  if (state === "complete") return "running";
  if (state === "failed") return "error";
  if (state === "running") return "warning";
  return "idle";
}

export function MultiCameraCalibrationPanel({
  status,
  busy,
  onRun,
}: {
  status: GuiSnapshot["calibration"];
  busy: boolean;
  onRun: () => void;
}) {
  const cameraCount = status.cameras.length;
  const disabled = busy || status.state === "running";
  return (
    <section className="panel calibration-panel">
      <div className="panel-heading">
        <h2>Multi-Camera Calibration</h2>
        <span className="state-pill">
          <StatusDot state={calibrationDotState(status.state)} />
          {stateLabel(status.state)}
        </span>
      </div>
      <div className="control-row">
        <button disabled={disabled} onClick={onRun}>
          {status.state === "complete" || status.state === "failed" ? "Re-run Calibration" : "Run Calibration"}
        </button>
        <span className="calibration-pattern">pattern: {status.pattern}</span>
      </div>
      <p className="panel-note">{status.message}</p>
      {cameraCount > 0 ? (
        <div className="check-table calibration-table">
          {status.cameras.map((camera) => (
            <div className="check-row" key={camera.id}>
              <strong>{camera.id}</strong>
              <span>repro {camera.reprojectionMm.toFixed(3)} mm · baseline {camera.baselineMm.toFixed(1)} mm</span>
              <em>{camera.status}</em>
            </div>
          ))}
        </div>
      ) : null}
      <div className="summary-grid">
        <Metric label="Last run" value={status.lastRunAt || "—"} />
        <Metric label="Output" value={status.outputPath || "—"} />
      </div>
    </section>
  );
}
