// Live sensor monitor cards (dark, mirroring the Device Manager tile styling).
// These preserve the existing real-time monitoring capability inside the
// calibration center; they never claim "calibrated" from live data alone.

import type { DataCollectionGuiApi } from "../api";
import type { DeviceStatus } from "../types";
import {
  DYNAMIC_FORCE_LIMITS,
  STATIC_COMPENSATED_FORCE_LIMITS,
  TOUCH_TOLERANCE,
  fmtDuration,
  fmtN,
  fmtNm,
  fmtNum,
} from "./config";
import { useBoxPreview } from "./useBoxPreview";
import { ForceAxisGrid, TouchHeatmap } from "./BoxSensorViews";
import { evaluateForce } from "./parseCaliLog";
import type { ForceVec } from "./types";

function StaleChip({ fresh, staleS }: { fresh: boolean; staleS: number | null }) {
  return (
    <span className={`cali-chip ${fresh ? "cali-chip-ok" : "cali-chip-warn"}`}>
      {fresh ? "实时" : "数据陈旧"}
      {staleS != null && ` · ${fmtDuration(staleS * 1000)}前`}
    </span>
  );
}

function vecNorm(vec: ForceVec | null, axes: readonly (keyof ForceVec)[]): number | null {
  if (!vec) return null;
  return Math.sqrt(axes.reduce((acc, axis) => acc + vec[axis] * vec[axis], 0));
}

export function ForceSensorCard({
  api,
  device,
}: {
  api: DataCollectionGuiApi;
  device: DeviceStatus;
}) {
  const view = useBoxPreview(api, device.id);
  const online = device.state === "running" || view.fresh;
  // Operational reference for out-of-range highlighting: the dynamic limits
  // (Fz is expected loaded during use). Highlight is per-axis, not whole-card.
  const limits = DYNAMIC_FORCE_LIMITS;

  return (
    <div className="cali-monitor cali-monitor-dark">
      <div className="cali-monitor-head">
        <span className="cali-monitor-title">
          <span className={`status-dot status-${online ? "running" : "idle"}`} />
          {device.label || device.id}
        </span>
        <StaleChip fresh={view.fresh} staleS={view.staleS} />
      </div>
      <div className="cali-monitor-body">
        {view.force ? (
          <ForceAxisGrid vec={view.force} limits={limits} variant="monitor" />
        ) : (
          <div className="cali-axis-empty">无力数据（{online ? "等待采样" : "离线"}）</div>
        )}
      </div>
      <div className="cali-monitor-foot">
        <span>采样率 {device.fps || "--"} Hz</span>
        <span>数据源 实时读数</span>
        <span>{online ? "在线" : "离线"}</span>
      </div>
    </div>
  );
}

export function ForceStaticValidationCard({
  api,
  device,
}: {
  api: DataCollectionGuiApi;
  device: DeviceStatus;
}) {
  const view = useBoxPreview(api, device.id);
  const online = device.state === "running" || view.fresh;
  const evalResult = view.forceNoGravity
    ? evaluateForce(view.forceNoGravity, STATIC_COMPENSATED_FORCE_LIMITS)
    : null;
  const ready = Boolean(view.fresh && evalResult?.pass);
  const forceNorm = vecNorm(view.forceNoGravity, ["fx", "fy", "fz"]);
  const momentNorm = vecNorm(view.forceNoGravity, ["mx", "my", "mz"]);
  const rawForceNorm = vecNorm(view.force, ["fx", "fy", "fz"]);
  const chipText = !view.forceNoGravity
    ? "等待补偿数据"
    : !view.fresh
      ? "数据陈旧"
      : evalResult?.pass
        ? "静态通过"
        : "残差偏大";

  return (
    <div className="cali-monitor cali-monitor-dark">
      <div className="cali-monitor-head">
        <span className="cali-monitor-title">
          <span className={`status-dot status-${ready ? "running" : online ? "warning" : "idle"}`} />
          重力补偿静态验证
        </span>
        <span className={`cali-chip ${ready ? "cali-chip-ok" : "cali-chip-warn"}`}>
          {chipText}
        </span>
      </div>
      <div className="cali-monitor-body">
        {view.forceNoGravity ? (
          <ForceAxisGrid
            vec={view.forceNoGravity}
            limits={STATIC_COMPENSATED_FORCE_LIMITS}
            variant="monitor"
          />
        ) : (
          <div className="cali-axis-empty">无补偿后力数据（{online ? "等待采样" : "离线"}）</div>
        )}
      </div>
      <div className="cali-touch-metrics cali-force-static-metrics">
        <div className="cali-touch-metric">
          <span>补偿后 |F|</span>
          <strong>{fmtN(forceNorm)}</strong>
        </div>
        <div className="cali-touch-metric">
          <span>补偿后 |M|</span>
          <strong>{fmtNm(momentNorm)}</strong>
        </div>
        <div className="cali-touch-metric">
          <span>原力 |F|</span>
          <strong>{fmtN(rawForceNorm)}</strong>
        </div>
        <div className="cali-touch-metric">
          <span>阈值</span>
          <strong>1.0 N / 0.030 N·m</strong>
        </div>
      </div>
      <div className="cali-monitor-foot">
        <span>{device.label || device.id}</span>
        <span>空载静止时应接近 0</span>
      </div>
    </div>
  );
}

export function TactileSensorCard({
  api,
  device,
}: {
  api: DataCollectionGuiApi;
  device: DeviceStatus;
}) {
  const view = useBoxPreview(api, device.id);
  const online = device.state === "running" || view.fresh;
  const unloaded =
    view.touchNetN != null && Math.abs(view.touchNetN) <= TOUCH_TOLERANCE.netForceEpsilonN;

  return (
    <div className="cali-monitor cali-monitor-dark">
      <div className="cali-monitor-head">
        <span className="cali-monitor-title">
          <span className={`status-dot status-${online ? "running" : "idle"}`} />
          {device.label || device.id}
        </span>
        <StaleChip fresh={view.fresh} staleS={view.staleS} />
      </div>
      <div className="cali-monitor-media">
        <TouchHeatmap
          fz0p1N={view.touchFz0p1N}
          fx0p1N={view.touchFx0p1N}
          fy0p1N={view.touchFy0p1N}
          model={view.touchModel}
          points={view.touchPoints}
        />
      </div>
      <div className="cali-touch-metrics">
        {/* The Paxini pad only reports per-taxel normal force, so net Fx/Fy are
            not derivable from the preview — shown as "--" rather than 0. */}
        <div className="cali-touch-metric">
          <span>合力 Fz</span>
          <strong>{fmtN(view.touchNetN)}</strong>
        </div>
        <div className="cali-touch-metric">
          <span>合力 Fx/Fy</span>
          <strong className="cali-muted">-- / --</strong>
        </div>
        <div className="cali-touch-metric">
          <span>最大 taxel</span>
          <strong>{fmtNum(view.touchMaxResidual, 1)} (0.1N)</strong>
        </div>
        <div className="cali-touch-metric">
          <span>空载</span>
          <strong>{view.touchNetN == null ? "--" : unloaded ? "是" : "否"}</strong>
        </div>
      </div>
      <div className="cali-monitor-foot">
        <span>采样率 {device.fps || "--"} Hz</span>
        <span>{online ? "在线" : "离线"}</span>
      </div>
    </div>
  );
}
