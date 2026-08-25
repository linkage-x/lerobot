import type { CalibrationStatus } from "../types";
import { solveProgressView } from "./solveProgress";

/** Progress of a running solve. Nothing is drawn when nothing is running. */
export function SolveProgress({ status }: { status: CalibrationStatus }) {
  const view = solveProgressView(status);
  if (!view) return null;
  return (
    <div className="cali-solve">
      <div className="cali-progress">
        <span className="cali-spinner" aria-hidden="true" />
        <b>{view.headline}</b>
        {view.counter ? <span>{view.counter}</span> : null}
        <span className="cali-solve-percent">{view.percent}%</span>
      </div>
      <div className="progress">
        <div
          className={`progress-bar${view.indeterminate ? " progress-bar-waiting" : ""}`}
          style={{ width: `${view.percent}%` }}
        />
      </div>
      <p className="small cali-solve-detail">
        {view.timing}
        {view.detail ? ` · ${view.detail}` : ""}
      </p>
    </div>
  );
}
