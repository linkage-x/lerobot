// Guided multi-camera calibration: one intrinsics sweep per camera, then one
// shared extrinsics sweep, then the solve.
//
// The panel this replaces was a single "Run Calibration" button with no
// indication of what to record or how. That is not a UI gap so much as a
// correctness one: the 0804 capture had to be redone from scratch because the
// board never reached the edges of the frame, and nothing on screen said it had
// to. Every step here carries its acceptance criterion.
import { useState } from "react";
import type { DataCollectionGuiApi, GuiSnapshot } from "../api";
import type { CalibrationSession, CalibrationSessionStep } from "../types";
import { Metric, StatusDot, stateLabel } from "../shared/ui";
import {
  BOARD_NOTE,
  STEP_STATUS_DOT,
  STEP_STATUS_LABEL,
  guideFor,
  skipConsequence,
  stepTitle,
} from "./calibrationGuide";

function StepList({ steps, currentIndex }: { steps: CalibrationSessionStep[]; currentIndex: number }) {
  return (
    <div className="check-table calibration-table">
      {steps.map((step, index) => (
        <div className="check-row" key={`${step.kind}-${step.camera || "rig"}-${index}`}>
          <strong>
            <StatusDot state={index === currentIndex ? "warning" : STEP_STATUS_DOT[step.status]} />
            {step.kind === "intrinsics" ? step.camera : "外参 · 协同"}
          </strong>
          <span>
            {step.episodeIndex >= 0 ? `episode ${step.episodeIndex}` : "—"}
            {step.note ? ` · ${step.note}` : ""}
          </span>
          <em>{STEP_STATUS_LABEL[step.status] ?? step.status}</em>
        </div>
      ))}
    </div>
  );
}

export function CalibrationWizard({
  snapshot,
  api,
  busy,
  onSolve,
}: {
  snapshot: GuiSnapshot;
  api: DataCollectionGuiApi;
  busy: boolean;
  onSolve: () => void;
}) {
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const session: CalibrationSession | undefined = snapshot.calibrationSession;
  const status = snapshot.calibration;

  const call = async (fn: () => Promise<{ ok: boolean; error?: string }>) => {
    setPending(true);
    setError("");
    const result = await fn();
    setPending(false);
    if (!result.ok) setError(result.error || "操作失败");
  };

  const disabled = busy || pending;
  const step = session?.steps[session.currentIndex];
  const recording = step?.status === "recording";

  return (
    <section className="panel calibration-panel">
      <div className="panel-heading">
        <h2>多相机标定</h2>
        <span className="state-pill">
          <StatusDot
            state={
              status.state === "complete"
                ? "running"
                : status.state === "failed"
                  ? "error"
                  : status.state === "running"
                    ? "warning"
                    : "idle"
            }
          />
          {stateLabel(status.state)}
        </span>
      </div>

      {!session?.active ? (
        <>
          <p className="panel-note">{status.message}</p>
          <div className="callout">
            <b>开始前先准备好：</b>
            <ol>
              <li>相机已 Connect（标定要用录制器采集，未连接无法开始）。</li>
              <li>{BOARD_NOTE}</li>
              <li>桌面清空到能自由挥板的程度；标定期间不要碰任何一台相机。</li>
            </ol>
            <p>
              流程：<b>逐台录一段内参 → 全部相机协同录一段外参 → 解算</b>。
              视野被遮挡的相机可以跳过，跳过后沿用它现有的内参。
            </p>
          </div>
          <div className="control-row">
            <button className="cali-btn-primary" disabled={disabled} onClick={() => call(() => api.startCalibrationSession())}>
              开始引导标定
            </button>
          </div>
        </>
      ) : (
        <>
          <div className="summary-grid">
            <Metric label="数据集" value={session.datasetName} />
            <Metric label="进度" value={`${session.currentIndex} / ${session.steps.length}`} />
            <Metric label="录制器" value={session.recorderState} />
          </div>

          <p className="panel-note">{session.message}</p>

          {step ? (
            <div className="callout">
              <b>{stepTitle(step.kind, step.camera)}</b>
              <ol>
                {guideFor(step.kind).map((item) => (
                  <li key={item.title}>
                    {item.title}
                    {item.detail ? <div className="small">{item.detail}</div> : null}
                  </li>
                ))}
              </ol>
              <p className="small">{skipConsequence(step.kind)}</p>
            </div>
          ) : null}

          <StepList steps={session.steps} currentIndex={session.currentIndex} />

          {error ? <p className="panel-note error">{error}</p> : null}

          <div className="control-row">
            {step && !recording ? (
              <button
                className="cali-btn-primary"
                disabled={disabled}
                onClick={() => call(() => api.calibrationStepRecord("start"))}
              >
                开始录制本段
              </button>
            ) : null}
            {recording ? (
              <>
                <button
                  className="cali-btn-primary"
                  disabled={disabled}
                  onClick={() => call(() => api.calibrationStepRecord("save"))}
                >
                  保存本段
                </button>
                <button disabled={disabled} onClick={() => call(() => api.calibrationStepRecord("discard"))}>
                  丢弃重录
                </button>
              </>
            ) : null}
            {step && !recording ? (
              <button disabled={disabled} onClick={() => call(() => api.calibrationStepSkip())}>
                跳过这一台
              </button>
            ) : null}
            {session.stage === "ready" ? (
              <button className="cali-btn-primary" disabled={disabled} onClick={onSolve}>
                开始解算
              </button>
            ) : null}
            <button disabled={disabled || recording} onClick={() => call(() => api.cancelCalibrationSession())}>
              退出引导
            </button>
          </div>
        </>
      )}

      {status.cameras.length > 0 ? (
        <>
          <p className="panel-note">
            解算结果 · 每台相机的 BA 重投影残差（<b>不是</b>跟踪精度：单 marker 路径下重投影几乎恒等可满足，
            它只反映标定自身是否自洽）
          </p>
          <div className="check-table calibration-table">
            {status.cameras.map((camera) => (
              <div className="check-row" key={camera.id}>
                <strong>{camera.id}</strong>
                <span>重投影 {camera.reprojectionPx.toFixed(4)} px</span>
                <em>{camera.status}</em>
              </div>
            ))}
          </div>
        </>
      ) : null}

      <div className="summary-grid">
        <Metric label="上次解算" value={status.lastRunAt || "—"} />
        <Metric label="生产内参" value={status.intrinsicsRun || "—"} />
        <Metric label="生产外参" value={status.extrinsicsRun || "—"} />
      </div>
    </section>
  );
}
