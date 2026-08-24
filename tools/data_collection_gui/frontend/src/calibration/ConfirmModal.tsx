// The pre-calibration confirmation dialog: an operator checklist gate and a live
// stability probe. The "开始校准" button stays disabled until every item is checked
// and the rig is not measurably unstable (spec §7).
//
// The modal shell it sits in moved to shared/ui once a second page needed one; it is
// re-exported here so the calibration imports did not all have to move with it.

import { Modal } from "../shared/ui";
import type { StabilityResult } from "./useCalibrationWorkflow";
import { STABILITY } from "./config";

export { Modal };

export const CONFIRM_CHECKLIST: { id: string; label: string }[] = [
  { id: "fixture", label: "BOX 已放入标准标定工装" },
  { id: "unloaded", label: "传感器完全空载" },
  { id: "stable-desk", label: "桌面稳定" },
  { id: "no-touch", label: "当前没有人触碰设备" },
  { id: "no-record", label: "当前没有进行数据采集" },
  { id: "no-handheld", label: "已了解禁止手持校零" },
];

function StabilityBadge({ stability }: { stability: StabilityResult }) {
  const map: Record<StabilityResult["status"], { cls: string; text: string }> = {
    idle: { cls: "idle", text: "未开始" },
    sampling: { cls: "warning", text: `采样中 ${stability.samples}/${STABILITY.minSamples}` },
    stable: { cls: "running", text: "稳定" },
    unstable: { cls: "error", text: "不稳定" },
    unavailable: { cls: "idle", text: "稳定性检测不可用" },
  };
  const { cls, text } = map[stability.status];
  return (
    <div className="cali-stability">
      <span className={`cali-badge cali-badge-${cls}`}>
        <span className={`status-dot status-${cls === "idle" ? "idle" : cls}`} />
        {text}
      </span>
      {stability.detail && <small>{stability.detail}</small>}
    </div>
  );
}

export function CalibrationConfirmModal({
  title,
  checked,
  onToggle,
  stability,
  onCancel,
  onConfirm,
  functionName,
}: {
  title: string;
  checked: Record<string, boolean>;
  onToggle: (id: string) => void;
  stability: StabilityResult;
  onCancel: () => void;
  onConfirm: () => void;
  /** Backend function name, surfaced as dev detail (not on the main button). */
  functionName: string;
}) {
  const allChecked = CONFIRM_CHECKLIST.every((c) => checked[c.id]);
  const stabilityBlocks = stability.status === "unstable" || stability.status === "sampling";
  const canStart = allChecked && !stabilityBlocks;

  return (
    <Modal
      title={title}
      onClose={onCancel}
      footer={
        <>
          <button onClick={onCancel}>取消</button>
          <button className="cali-btn-primary" disabled={!canStart} onClick={onConfirm}>
            开始校准
          </button>
        </>
      }
    >
      <p className="cali-modal-lead">执行校准前，请逐项确认：</p>
      <div className="cali-checklist">
        {CONFIRM_CHECKLIST.map((item) => (
          <label className="cali-check" key={item.id}>
            <input
              type="checkbox"
              checked={Boolean(checked[item.id])}
              onChange={() => onToggle(item.id)}
            />
            <span>{item.label}</span>
          </label>
        ))}
      </div>
      <div className="cali-stability-block">
        <div className="cali-stability-head">稳定性检测（{STABILITY.windowMs / 1000}s 采样）</div>
        <StabilityBadge stability={stability} />
        {stability.status === "unavailable" && (
          <small className="cali-muted">
            无法获得实时采样，无法确认稳定性；如坚持执行请自行确保设备静止。
          </small>
        )}
      </div>
      <div className="cali-dev-hint">开发信息：调用后端 {functionName}</div>
    </Modal>
  );
}
