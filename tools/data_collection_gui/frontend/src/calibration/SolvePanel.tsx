import { useEffect, useState } from "react";
import type { CalibrationSession, CalibrationStatus } from "../types";
import { intrinsicsNote, solveButtonView, solveTargetView } from "./solvePanel";

export type SolveOptions = { forceRedetect?: boolean; refitIntrinsics?: boolean };

/** The capture the solve will read, and the button that starts (or retries) it. */
export function SolvePanel({
  status,
  session,
  disabled,
  onSolve,
  onPickDataset,
}: {
  status: CalibrationStatus;
  session: CalibrationSession | undefined;
  disabled: boolean;
  onSolve: (options: SolveOptions) => void;
  onPickDataset: (path: string, kind?: "extrinsics" | "intrinsics") => void;
}) {
  // Local, not gateway state: it applies to the next click and nothing else.
  // Detections are reused by default because they are a pure function of the
  // video, so re-deriving them is only ever a way to spend half an hour again.
  const [forceRedetect, setForceRedetect] = useState(false);
  // Off by default: re-fitting is a much longer run (one sweep per camera to
  // decode) and it replaces the intrinsics production is using, which is not
  // something to do as a side effect of re-solving extrinsics.
  const [refitIntrinsics, setRefitIntrinsics] = useState(false);
  const target = solveTargetView(status.solve);
  const button = solveButtonView(status, session);
  // The gateway is the source of truth for which capture is selected, but it
  // only says so on the next snapshot poll; without holding the choice locally
  // the dropdown visibly snaps back to the old capture in between.
  const [pending, setPending] = useState<string | null>(null);
  useEffect(() => {
    if (pending !== null && target.selected === pending) setPending(null);
  }, [pending, target.selected]);
  if (!button.visible) return null;

  return (
    <div className="cali-solve-target">
      <div className="control-row">
        <label className="cali-field">
          <span>将解算</span>
          <select
            value={pending ?? target.selected}
            disabled={disabled}
            onChange={(event) => {
              setPending(event.target.value);
              onPickDataset(event.target.value);
            }}
          >
            <option value="">自动选择</option>
            {target.candidates.map((item) => (
              <option key={item.path} value={item.path}>
                {item.label}
              </option>
            ))}
          </select>
        </label>
        <span className="cali-dev-hint">外参</span>
        <button
          className="cali-btn-primary"
          disabled={disabled || button.disabled}
          onClick={() => onSolve({ forceRedetect, refitIntrinsics })}
        >
          {button.label}
        </button>
      </div>
      <p className="small cali-solve-detail">
        {target.summary}
        {target.origin ? ` · ${target.origin}` : ""}
      </p>

      <label className="cali-check">
        <input
          type="checkbox"
          checked={refitIntrinsics}
          disabled={disabled}
          onChange={(event) => setRefitIntrinsics(event.target.checked)}
        />
        <span>同时重算内参</span>
      </label>
      {refitIntrinsics ? (
        <div className="control-row cali-solve-nested">
          <label className="cali-field">
            <span>内参采集</span>
            <select
              value={status.solve?.intrinsicsDatasetRoot ?? ""}
              disabled={disabled}
              onChange={(event) => onPickDataset(event.target.value, "intrinsics")}
            >
              <option value="">未选择</option>
              {target.candidates.map((item) => (
                <option key={item.path} value={item.path}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
        </div>
      ) : null}
      <p className="small cali-solve-detail">{intrinsicsNote(status.solve, refitIntrinsics)}</p>
      <label className="cali-check">
        <input
          type="checkbox"
          checked={forceRedetect}
          disabled={disabled}
          onChange={(event) => setForceRedetect(event.target.checked)}
        />
        <span>强制重新检测角点（视频没变时不需要——重算一遍是这一步最花时间的部分）</span>
      </label>
    </div>
  );
}
