// Canonical world frame (roadmap 2.4).
//
// What an operator needs from this panel is one sentence -- "is this still the
// same world?" -- and, when it is not, which camera is responsible. The rest is
// evidence for that answer.
//
// Two behaviours are deliberate and not obvious from the UI alone:
//   * checking is read-only. Committing changes what every future calibration
//     is compared against, so it is a separate, confirmed action;
//   * a break is presented as a coordinate-frame event, not as data loss.
//     Episode-relative motion, bimanual pose and contact-local trajectories are
//     unaffected by it; only cross-session absolute replay is.
import { useEffect, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { WorldFrameResponse } from "../types";
import { StatusDot } from "../shared/ui";
import { Modal } from "./ConfirmModal";
import {
  alignmentSummary,
  applyConfirmation,
  applyLabel,
  candidateClusters,
  canApplyWorld,
  canFallBackToGeometry,
  commonModeSummary,
  formatDeg,
  formatMm,
  graphSummary,
  needsOperatorChoice,
  referenceSummary,
  stableSourceSummary,
  worldCameraRows,
  worldReasonLabel,
  worldRoleDot,
  worldRoleLabel,
  worldStateDot,
  worldStateLabel,
  selectableCameras,
} from "./worldFrame";

export function WorldFramePanel({ api, busy }: { api: DataCollectionGuiApi; busy: boolean }) {
  const [payload, setPayload] = useState<WorldFrameResponse | null>(null);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [declared, setDeclared] = useState<string[]>([]);

  useEffect(() => {
    let cancelled = false;
    api.fetchWorldFrame().then((result) => {
      if (!cancelled && result) setPayload(result);
    });
    return () => {
      cancelled = true;
    };
  }, [api]);

  const run = async (action: () => Promise<WorldFrameResponse>) => {
    setRunning(true);
    setError("");
    const result = await action();
    setRunning(false);
    if (!result.ok) {
      setError(result.error || "操作失败");
      // The payload still carries the current state even on failure, so the
      // panel does not go blank while showing an error about it.
      if (result.reference) setPayload({ ...result, ok: true });
      return;
    }
    setPayload(result);
    setDeclared([]);
  };

  const registration = payload?.registration ?? null;
  const reference = payload?.reference;
  const stableSource = payload?.stableSource;
  const rows = worldCameraRows(registration);
  const ambiguous = needsOperatorChoice(registration);
  const disabled = busy || running;

  return (
    <section className="panel calibration-panel">
      <div className="panel-heading">
        <h2>世界坐标系</h2>
        {registration ? (
          <span className="state-pill">
            <StatusDot state={worldStateDot[registration.world_continuity_state]} />
            {worldStateLabel[registration.world_continuity_state]}
          </span>
        ) : null}
      </div>

      <p className="panel-note">基准：{referenceSummary(reference)}</p>

      <div className="control-row">
        <button
          className="cali-btn-primary"
          disabled={disabled || !reference?.exists}
          title={reference?.exists ? "" : "先冻结基准世界系"}
          onClick={() => run(() => api.registerWorldFrame({ stable: declared.length ? declared : undefined }))}
        >
          {running ? "检测中…" : "检测世界系连续性"}
        </button>
        <button
          disabled={disabled || !canApplyWorld(registration)}
          onClick={() => setConfirming(true)}
        >
          {applyLabel(registration)}
        </button>
        <button
          disabled={disabled}
          title={
            reference?.exists
              ? "重新冻结会定义一个新的世界系，历史绝对轨迹仍属于旧的那个"
              : "把当前标定的相机位姿定为唯一世界系"
          }
          onClick={() => run(() => api.freezeWorldFrame(Boolean(reference?.exists)))}
        >
          {reference?.exists ? "重新冻结基准" : "冻结为基准世界系"}
        </button>
        {canFallBackToGeometry(stableSource) ? (
          <button
            disabled={disabled}
            title="不采用相机自检的结论，只用两次解算之间的相机几何变化判定"
            onClick={() => run(() => api.registerWorldFrame({ useRigCheck: false }))}
          >
            改用几何共识判定
          </button>
        ) : null}
      </div>

      {/* Only once there is a verdict to attribute: before any registration has
          run there is no stable set, and describing one would be describing
          nothing. */}
      {stableSource && registration ? (
        <p className="panel-note">{stableSourceSummary(stableSource)}</p>
      ) : null}

      {error ? <p className="panel-note error">{error}</p> : null}

      {registration ? (
        <>
          <p className="panel-note">
            {worldReasonLabel(registration.reason)} · {alignmentSummary(registration)}
          </p>
          {registration.world_frame_id !== registration.reference_world_frame_id ? (
            <p className="panel-note error">
              本次标定属于新的世界系 {registration.world_frame_id}（父世界系{" "}
              {registration.parent_world_frame_id ?? "—"}）。旧数据没有失效，只是与新数据之间的绝对变换未知。
            </p>
          ) : null}
        </>
      ) : null}

      {ambiguous ? (
        <div className="check-table calibration-table">
          <div className="check-row">
            <strong>无法自动判定</strong>
            <span>
              候选相机组：
              {candidateClusters(registration)
                .map((cluster) => cluster.join("+"))
                .join("  或  ")}
              。整组一起被搬动与整组没动在数学上不可区分，请勾选确实没被碰过的相机后重新判定。
            </span>
            <em />
          </div>
          <div className="check-row">
            <strong>未移动的相机</strong>
            <span className="world-camera-picker">
              {selectableCameras(registration).map((camera) => (
                <label key={camera}>
                  <input
                    type="checkbox"
                    checked={declared.includes(camera)}
                    onChange={(event) =>
                      setDeclared((current) =>
                        event.target.checked
                          ? [...current, camera]
                          : current.filter((name) => name !== camera),
                      )
                    }
                  />
                  {camera}
                </label>
              ))}
            </span>
            <em>{declared.length} 台</em>
          </div>
        </div>
      ) : null}

      {rows.length > 0 ? (
        <div className="check-table calibration-table">
          {rows.map((row) => (
            <div className="check-row" key={row.camera}>
              <strong>
                <StatusDot state={worldRoleDot[row.role]} />
                {row.camera}
              </strong>
              <span>{row.detail}</span>
              <em>{worldRoleLabel[row.role]}</em>
            </div>
          ))}
        </div>
      ) : null}

      {registration ? (
        <>
          <p className="panel-note">{registration.guidance}</p>
          <p className="panel-note">{commonModeSummary(registration)}</p>
          <p className="panel-note cali-muted">
            判定阈值：相机两两几何变化 ≤{formatMm(registration.consensus.thresholds.translation_mm)} 且 ≤
            {formatDeg(registration.consensus.thresholds.rotation_deg)}；期望稳定相机数 ≥
            {registration.min_stable_cameras}
            {registration.session?.gauge ? ` · 本次解算 gauge：${registration.session.gauge}` : ""}
          </p>
        </>
      ) : null}

      {graphSummary(payload) ? <p className="panel-note">{graphSummary(payload)}</p> : null}

      {!registration && !error ? (
        <p className="panel-note">
          外参每重标一次，BA 都会自己挑一台相机当基准，世界系就会悄悄换一个——历史绝对轨迹的数字还在，含义已经变了。
          冻结一次基准世界系之后，这里用「哪些相机之间的相对几何没变」把每次新标定重新注册回同一个世界系。
        </p>
      ) : null}

      {confirming && registration ? (
        <Modal
          title={applyLabel(registration)}
          onClose={() => setConfirming(false)}
          footer={
            <>
              <button onClick={() => setConfirming(false)}>取消</button>
              <button
                className="cali-btn-primary"
                onClick={() => {
                  setConfirming(false);
                  void run(() =>
                    api.registerWorldFrame({
                      apply: true,
                      stable: declared.length ? declared : undefined,
                    }),
                  );
                }}
              >
                确认
              </button>
            </>
          }
        >
          <p>{applyConfirmation(registration)}</p>
        </Modal>
      ) : null}
    </section>
  );
}
