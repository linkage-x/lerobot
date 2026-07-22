// Calibration operation cards, scoped to one BOX. Origin (raw zero) and dynamic
// (filter) 6D force are deliberately two separate cards with different pass/fail
// criteria, plus a tactile card. Each drives one useCalibrationWorkflow instance
// and shares the confirm modal + result panel. Each card passes its box_id so
// the recorder calibrates only that box (empty = the single box).

import { useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { DeviceStatus } from "../types";
import {
  CALIBRATION_KIND_LABELS,
  DYNAMIC_FORCE_LIMITS,
  ORIGIN_FORCE_LIMITS,
  TOUCH_TOLERANCE,
  fmtDuration,
  fmtN,
  fmtNum,
  fmtTimestamp,
  type CalibrationKind,
  type ForceAxisLimits,
} from "./config";
import { computeValidity, validityDot } from "./status";
import { VALIDITY_LABELS, CALI_STATE_LABELS } from "./types";
import type { CalibrationRecord } from "./types";
import { isBusyState } from "./status";
import {
  FORCE_TROUBLESHOOTING,
  TOUCH_TROUBLESHOOTING,
  boxDisplayName,
} from "./adapters";
import { ForceAxisGrid } from "./BoxSensorViews";
import { CalibrationConfirmModal, CONFIRM_CHECKLIST } from "./ConfirmModal";
import {
  useCalibrationWorkflow,
  type ForceBoxResult,
  type TouchSideResult,
  type WorkflowApi,
} from "./useCalibrationWorkflow";

// --- small shared bits --------------------------------------------------------
function MetaRow({ label, value }: { label: string; value: React.ReactNode }) {
  return (
    <div className="cali-meta-row">
      <span>{label}</span>
      <strong>{value}</strong>
    </div>
  );
}

function ValidityBadge({ records, kind }: { records: CalibrationRecord[]; kind: CalibrationKind }) {
  const last = records[0]?.timestamp ?? null;
  const v = computeValidity(last, kind);
  const dot = validityDot(v.state);
  return (
    <span className={`cali-badge cali-badge-${dot}`}>
      <span className={`status-dot status-${dot}`} />
      {VALIDITY_LABELS[v.state]}
    </span>
  );
}

function ProgressLine({ wf }: { wf: WorkflowApi }) {
  if (!isBusyState(wf.state)) return null;
  return (
    <div className="cali-progress">
      <span className="cali-spinner" aria-hidden="true" />
      {CALI_STATE_LABELS[wf.state]}…
    </div>
  );
}

function useChecklist() {
  const [checked, setChecked] = useState<Record<string, boolean>>({});
  const toggle = (id: string) => setChecked((c) => ({ ...c, [id]: !c[id] }));
  const reset = () => setChecked({});
  return { checked, toggle, reset };
}

// --- result panel (spec §9) ---------------------------------------------------
export function CalibrationResultPanel({
  wf,
  troubleshooting,
  limits,
}: {
  wf: WorkflowApi;
  troubleshooting: string[];
  limits: ForceAxisLimits | null;
}) {
  const [showGuide, setShowGuide] = useState(false);
  if (wf.state !== "passed" && wf.state !== "failed") return null;
  const passed = wf.state === "passed";

  const copyError = () => {
    const text = [
      wf.error ?? "",
      ...wf.forceResults.flatMap((r) => (r.evaluation ? [r.boxId, r.evaluation.firstFailure ?? ""] : [])),
    ]
      .filter(Boolean)
      .join("\n");
    void navigator.clipboard?.writeText(text);
  };

  return (
    <div className={`cali-result ${passed ? "cali-result-pass" : "cali-result-fail"}`}>
      <div className="cali-result-head">
        <span className={`cali-badge cali-badge-${passed ? "running" : "error"}`}>
          <span className={`status-dot status-${passed ? "running" : "error"}`} />
          {passed ? "校准通过" : "校准失败"}
        </span>
        {wf.error && <span className="cali-result-reason">{wf.error}</span>}
      </div>

      {/* Force: per-box, per-axis */}
      {wf.forceResults.map((r: ForceBoxResult) => (
        <div className="cali-result-box" key={r.boxId}>
          <div className="cali-result-box-head">
            <strong>{boxDisplayName(r.boxId)}</strong>
            <span className={`cali-axis-verdict ${r.pass ? "pass" : "fail"}`}>{r.pass ? "PASS" : "FAIL"}</span>
            {r.error && <span className="cali-muted">{r.error}</span>}
          </div>
          {r.evaluation ? (
            <ForceAxisGrid evals={r.evaluation.axes} variant="result" />
          ) : (
            <ForceAxisGrid vec={r.after} limits={limits} variant="result" />
          )}
        </div>
      ))}

      {/* Touch: per-side */}
      {wf.touchResults.map((r: TouchSideResult) => (
        <div className="cali-result-box" key={r.deviceId}>
          <div className="cali-result-box-head">
            <strong>{r.label}</strong>
            <span className={`cali-axis-verdict ${r.pass ? "pass" : "fail"}`}>{r.pass ? "PASS" : "FAIL"}</span>
          </div>
          <div className="cali-touch-result">
            <span>净力 {fmtN(r.netN)}</span>
            <span>最大残差 {fmtNum(r.maxResidual, 1)} (0.1N)</span>
            <span className="cali-muted">容差 ±{TOUCH_TOLERANCE.netForceEpsilonN} N / {TOUCH_TOLERANCE.maxTaxelResidual0p1N} (0.1N)</span>
          </div>
        </div>
      ))}

      {!passed && (
        <div className="cali-troubleshoot">
          <button className="cali-link" onClick={() => setShowGuide((s) => !s)}>
            {showGuide ? "收起排查指南" : "查看详细排查指南"}
          </button>
          {showGuide && (
            <ul>
              {troubleshooting.map((t) => (
                <li key={t}>{t}</li>
              ))}
            </ul>
          )}
        </div>
      )}

      <div className="cali-result-actions">
        <button onClick={() => wf.begin()}>重新检测</button>
        <button className="cali-btn-primary" onClick={() => wf.begin()}>
          重新校准
        </button>
        {!passed && <button onClick={copyError}>复制错误信息</button>}
        <button onClick={() => wf.reset()}>关闭</button>
      </div>
    </div>
  );
}

// --- reboot step (origin only) -----------------------------------------------
function RebootStep({ wf }: { wf: WorkflowApi }) {
  if (wf.state !== "waiting_for_reboot") return null;
  return (
    <div className="cali-reboot">
      <p>
        <b>请给 BOX 重新上电</b>，原始零点写入 MCU 后需断电重启才能生效。重启并等待重新连接后将自动复核原始数据。
      </p>
      <button className="cali-btn-primary" onClick={() => wf.confirmReboot()}>
        我已重新上电，开始复核
      </button>
    </div>
  );
}

// --- 6D force card (origin / dynamic) ----------------------------------------
export function ForceCalibrationCard({
  variant,
  api,
  boxId,
  devices,
  operator,
  guard,
  onRecord,
  history,
}: {
  variant: "origin" | "dynamic";
  api: DataCollectionGuiApi;
  boxId: string;
  devices: DeviceStatus[];
  operator: string;
  guard: () => string | null;
  onRecord: (records: CalibrationRecord[]) => void;
  history: CalibrationRecord[];
}) {
  const isOrigin = variant === "origin";
  const kind: CalibrationKind = isOrigin ? "force_origin" : "force_dynamic";
  const limits = isOrigin ? ORIGIN_FORCE_LIMITS : DYNAMIC_FORCE_LIMITS;
  const functionName = isOrigin ? "cali_6d_force_sensor_origin" : "cali_6d_force_sensor";
  const checklist = useChecklist();

  const wf = useCalibrationWorkflow({
    api,
    kind,
    trigger: isOrigin
      ? () => api.triggerSixDForceOriginCalibration(boxId)
      : () => api.triggerSixDForceCalibration(boxId),
    fetchLog: () => api.fetchBoxCaliLog(),
    sampleDevices: devices,
    sampleKind: "force",
    requiresReboot: isOrigin,
    limits,
    guard,
    operator,
    onRecord,
  });

  const last = history[0]?.timestamp ?? null;
  const validity = computeValidity(last, kind);
  const guardReason = guard();
  const busy = isBusyState(wf.state);
  const showModal = wf.state === "checking_prerequisites" || wf.state === "waiting_for_stability";

  const start = () => {
    checklist.reset();
    wf.begin();
  };

  return (
    <section className="panel cali-op-card">
      <div className="panel-heading">
        <h2>{isOrigin ? "六维力原始零点校准（传感器本身）" : "六维力动态校准（滤波算法）"}</h2>
        <ValidityBadge records={history} kind={kind} />
      </div>

      <div className="cali-meta">
        <MetaRow label="频率" value={isOrigin ? "1 次 / 天" : "1 次 / 30 min"} />
        <MetaRow label="上次校准" value={fmtTimestamp(last)} />
        <MetaRow label="操作者" value={history[0]?.operator || operator || "—"} />
        <MetaRow label="当前状态" value={VALIDITY_LABELS[validity.state]} />
        <MetaRow label="有效截止" value={fmtTimestamp(validity.expiresAt)} />
        <MetaRow label="下次倒计时" value={fmtDuration(validity.remainingMs)} />
      </div>

      {/* Procedure + warnings */}
      <ol className="cali-steps">
        {isOrigin ? (
          <>
            <li>放置到标准标定工装</li>
            <li>确认传感器完全空载</li>
            <li>确认桌面稳定</li>
            <li>执行原始零点校准</li>
            <li>按提示给 BOX 重新上电</li>
            <li>重新连接后自动复核原始数据</li>
          </>
        ) : (
          <>
            <li>确认工具/负载处于标定姿态</li>
            <li>确认桌面稳定、无人触碰</li>
            <li>执行动态校准（滤波）</li>
            <li>自动复核 Fz / Mx 是否落在目标负载区间</li>
          </>
        )}
      </ol>

      {isOrigin ? (
        <div className="cali-warn">
          必须使用标准标定工装 · 禁止手持校零 · 原始校准完成后需重新上电并复核。
          判定：|Fx|,|Fy|,|Fz| ≤ 0.5 N，|Mx|,|My|,|Mz| ≤ 0.01 N·m。
        </div>
      ) : (
        <div className="cali-info">
          动态校准后 Fz 应落在 <b>-7.784 ± 0.5 N</b>、Mx 应落在 <b>-0.168 ± 0.01 N·m</b>（保留负载，不应接近 0）；|Fx|,|Fy| ≤ 0.5 N，|My|,|Mz| ≤ 0.01 N·m。
        </div>
      )}

      {guardReason && <div className="cali-guard">当前不可标定：{guardReason}</div>}

      <div className="cali-op-actions">
        <button className="cali-btn-primary" disabled={busy || Boolean(guardReason)} onClick={start}>
          {isOrigin ? "执行原始零点校准" : "执行动态校准"}
        </button>
        <span className="cali-dev-hint">调用 {functionName}</span>
      </div>

      <ProgressLine wf={wf} />
      <RebootStep wf={wf} />
      <CalibrationResultPanel wf={wf} troubleshooting={FORCE_TROUBLESHOOTING} limits={limits} />

      {showModal && (
        <CalibrationConfirmModal
          title={CALIBRATION_KIND_LABELS[kind]}
          checked={checklist.checked}
          onToggle={checklist.toggle}
          stability={wf.stability}
          functionName={functionName}
          onCancel={() => {
            wf.cancel();
            wf.reset();
          }}
          onConfirm={wf.confirm}
        />
      )}
    </section>
  );
}

// --- tactile card -------------------------------------------------------------
export function TactileCalibrationCard({
  api,
  boxId,
  devices,
  operator,
  guard,
  onRecord,
  history,
}: {
  api: DataCollectionGuiApi;
  boxId: string;
  devices: DeviceStatus[];
  operator: string;
  guard: () => string | null;
  onRecord: (records: CalibrationRecord[]) => void;
  history: CalibrationRecord[];
}) {
  const checklist = useChecklist();
  const wf = useCalibrationWorkflow({
    api,
    kind: "touch",
    // Per-box calibration: a box's two pads share one MCU-side re-zero, so there
    // is one button per box (not per pad). Results are still reported per side.
    trigger: () => api.triggerTouchCalibration(boxId),
    fetchLog: () => api.fetchBoxTouchCaliLog(),
    sampleDevices: devices,
    sampleKind: "touch",
    requiresReboot: false,
    limits: null,
    guard,
    operator,
    onRecord,
  });

  const last = history[0]?.timestamp ?? null;
  const validity = computeValidity(last, "touch");
  const guardReason = guard();
  const busy = isBusyState(wf.state);
  const showModal = wf.state === "checking_prerequisites" || wf.state === "waiting_for_stability";

  const start = () => {
    checklist.reset();
    wf.begin();
  };

  return (
    <section className="panel cali-op-card">
      <div className="panel-heading">
        <h2>触觉标定</h2>
        <ValidityBadge records={history} kind="touch" />
      </div>

      <div className="cali-meta">
        <MetaRow label="上次校准" value={fmtTimestamp(last)} />
        <MetaRow label="当前状态" value={VALIDITY_LABELS[validity.state]} />
        <MetaRow label="下次倒计时" value={fmtDuration(validity.remainingMs)} />
      </div>

      <div className="cali-info">
        按 BOX 整体校准（一次同时归零该 BOX 的两个 Paxini pad）；目标：合力与分布力趋近 0，
        容差 净力 ±{TOUCH_TOLERANCE.netForceEpsilonN} N、单 taxel ≤ {TOUCH_TOLERANCE.maxTaxelResidual0p1N} (0.1N)。
        结果仍按左/右分别显示。
      </div>

      {guardReason && <div className="cali-guard">当前不可标定：{guardReason}</div>}

      <div className="cali-op-actions">
        <button className="cali-btn-primary" disabled={busy || Boolean(guardReason) || devices.length === 0} onClick={start}>
          校准触觉（该 BOX）
        </button>
        <span className="cali-dev-hint">调用 cali_touch_sensor（device_id={boxId || "single"}）</span>
      </div>

      <ProgressLine wf={wf} />
      <CalibrationResultPanel wf={wf} troubleshooting={TOUCH_TROUBLESHOOTING} limits={null} />

      {showModal && (
        <CalibrationConfirmModal
          title="触觉标定"
          checked={checklist.checked}
          onToggle={checklist.toggle}
          stability={wf.stability}
          functionName="cali_touch_sensor"
          onCancel={() => {
            wf.cancel();
            wf.reset();
          }}
          onConfirm={wf.confirm}
        />
      )}
    </section>
  );
}

// Re-export so the page imports cards from one module.
export { CONFIRM_CHECKLIST };
