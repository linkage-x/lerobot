// Guided multi-camera calibration: one intrinsics sweep per camera, then one
// shared extrinsics sweep, then the solve.
//
// The panel this replaces was a single "Run Calibration" button with no
// indication of what to record or how. That is not a UI gap so much as a
// correctness one: the 0804 capture had to be redone from scratch because the
// board never reached the edges of the frame, and nothing on screen said it had
// to. Every step here carries its acceptance criterion.
import { useEffect, useState } from "react";
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
import { SolvePanel, type SolveOptions } from "./SolvePanel";
import { captureTally } from "./solvePanel";
import { SolveProgress } from "./SolveProgress";
import { StepCameraPreview } from "./StepCameraPreview";
import { previewCameras, previewStatus } from "./stepPreview";
import { pointerPromotionHint, pointerRows } from "./status";
import { PromotionPanel } from "./PromotionPanel";

// Mirrors the gateway's validation (_parse_calibration_segment_seconds); the
// input just stops the obvious mistakes before a round-trip.
const SEGMENT_SECONDS_DEFAULT = 30;
const SEGMENT_SECONDS_MIN = 5;
const SEGMENT_SECONDS_MAX = 300;

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
  onSolve: (options?: SolveOptions) => void;
}) {
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const session: CalibrationSession | undefined = snapshot.calibrationSession;
  const status = snapshot.calibration;
  const promotionHint = pointerPromotionHint(status);

  const call = async (fn: () => Promise<{ ok: boolean; error?: string }>) => {
    setPending(true);
    setError("");
    const result = await fn();
    setPending(false);
    if (!result.ok) setError(result.error || "操作失败");
    return result;
  };

  const disabled = busy || pending;
  const step = session?.steps[session.currentIndex];
  const recording = step?.status === "recording";

  // How long each sweep records. Kept as a draft string so typing "45" does not
  // fire a request per keystroke; the session is the source of truth once it
  // exists and the effect adopts whatever it reports back.
  const sessionSeconds = session?.active ? session.episodeTimeS : undefined;
  const [secondsDraft, setSecondsDraft] = useState(String(SEGMENT_SECONDS_DEFAULT));
  useEffect(() => {
    if (sessionSeconds != null) setSecondsDraft(String(sessionSeconds));
  }, [sessionSeconds]);

  const commitSeconds = async () => {
    const value = Number(secondsDraft);
    if (!Number.isFinite(value) || value <= 0) {
      setSecondsDraft(String(sessionSeconds ?? SEGMENT_SECONDS_DEFAULT));
      return;
    }
    // Before the session exists there is nothing to update: the value is passed
    // to the start call instead.
    if (sessionSeconds == null || value === sessionSeconds) return;
    const result = await call(() => api.setCalibrationSegmentSeconds(value));
    // A rejected length (out of range, or a sweep already recording) must not
    // stay on screen as if it had been accepted.
    if (!result.ok) setSecondsDraft(String(sessionSeconds));
  };

  const secondsField = (
    <label className="cali-seconds">
      <span>每段时长</span>
      <input
        type="number"
        min={SEGMENT_SECONDS_MIN}
        max={SEGMENT_SECONDS_MAX}
        step={5}
        value={secondsDraft}
        disabled={disabled || recording}
        onChange={(event) => setSecondsDraft(event.target.value)}
        onBlur={() => void commitSeconds()}
        onKeyDown={(event) => {
          if (event.key === "Enter") void commitSeconds();
        }}
      />
      <span>秒</span>
    </label>
  );
  // What this step is pointed at, and whether the recorder is publishing frames
  // for it right now. Both answers live in stepPreview.ts so the wording can be
  // pinned by a test.
  const cameras = previewCameras(
    step,
    snapshot.devices.filter((device) => device.kind === "camera").map((device) => device.id),
  );
  const preview = previewStatus(step, session?.recorderState ?? "idle");

  // Solving is an operation on a capture, not the tail of the capture wizard:
  // it is offered whenever a capture exists and nothing is running, including
  // after a failure and outside a session entirely.
  const solvePanel = (
    <SolvePanel
      status={status}
      session={session}
      disabled={disabled}
      onSolve={onSolve}
      onPickDataset={(path, kind) => call(() => api.setCalibrationDataset(path, kind))}
    />
  );

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

      {/* Above the branch: a solve can be launched with or without a guided
          session, and in both cases it is the thing the operator is waiting on. */}
      <SolveProgress status={status} />

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
            {secondsField}
            <button
              className="cali-btn-primary"
              disabled={disabled}
              onClick={() =>
                call(() => api.startCalibrationSession(undefined, Number(secondsDraft) || undefined))
              }
            >
              开始引导标定
            </button>
          </div>
          {solvePanel}
        </>
      ) : (
        <>
          <div className="summary-grid">
            <Metric label="数据集" value={session.datasetName} />
            <Metric label="进度" value={`${session.currentIndex} / ${session.steps.length}`} />
            <Metric label="已录制" value={captureTally(session.steps)} />
            <Metric label="每段时长" value={`${session.episodeTimeS}s`} />
            <Metric label="录制器" value={session.recorderState} />
          </div>

          <p className={`panel-note${session.stage === "failed" ? " error" : ""}`}>{session.message}</p>

          {step ? (
            <div className="callout">
              <b>{stepTitle(step.kind, step.camera)}</b>
              <ol>
                {guideFor(
                  step.kind,
                  session.episodeTimeS,
                  Math.round(snapshot.configSummary.fps * session.episodeTimeS),
                ).map((item) => (
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

          <StepCameraPreview api={api} cameras={cameras} live={preview.live} note={preview.note} />

          {error ? <p className="panel-note error">{error}</p> : null}

          <div className="control-row">
            {step ? secondsField : null}
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
            <button disabled={disabled || recording} onClick={() => call(() => api.cancelCalibrationSession())}>
              退出引导
            </button>
          </div>

          {solvePanel}
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
                <span>
                  重投影 {camera.reprojectionPx.toFixed(4)} px
                  {/* Coverage is the failure reprojection cannot see: a fit is
                      happy to be self-consistent over the middle of the frame. */}
                  {camera.coverage != null ? ` · 边缘覆盖 ${Math.round(camera.coverage * 100)}%` : ""}
                  {camera.intrinsicsNote ? <div className="small">{camera.intrinsicsNote}</div> : null}
                </span>
                <em>{camera.status}</em>
              </div>
            ))}
          </div>
        </>
      ) : null}

      <div className="summary-grid">
        <Metric label="上次解算" value={status.lastRunAt || "—"} />
      </div>

      {/* Two columns, never one. A solve writes its run name into gateway
          memory and never into the tracking config, so a single "生产外参: X"
          line showed a calibration as live while production kept loading the
          previous one — for seven days, with a moved camera in it. */}
      <div className="cali-pointers">
        <div className="cali-pointer-head">
          <span />
          <span>最近解出</span>
          <span>生产实际加载</span>
        </div>
        {pointerRows(status).map((row) => (
          <div key={row.label} className={row.differs ? "cali-pointer-row differs" : "cali-pointer-row"}>
            <span>{row.label}</span>
            <span className="mono">{row.solved || "—"}</span>
            <span className="mono">
              {row.production || "—"}
              {row.differs ? <b> ← 不一致</b> : null}
            </span>
          </div>
        ))}
      </div>
      {promotionHint ? <p className="panel-note cali-pointer-drift">{promotionHint}</p> : null}
      <PromotionPanel
        review={status.promotion}
        disabled={disabled}
        onPromote={(options) => api.promoteCalibration(options)}
      />
    </section>
  );
}
