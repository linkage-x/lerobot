// Overview dashboard: an at-a-glance hub that aggregates cross-page state from
// the existing snapshot (no new backend) and links out to the relevant page.
// It is deliberately read-only except for the one action every page depends on
// -- connecting the recorder session (which streams the box/camera live data
// that Calibration and Device Manager also consume).

import type { GuiSnapshot } from "../api";
import type { PageId } from "../App";
import type { ProcessingStatus } from "../types";
import { StatusDot, Metric, PageHeader, stateLabel, processingStatusLabel } from "../shared/ui";
import { historyRepo } from "../calibration/adapters";
import { summarizeKinds } from "../calibration/status";
import type { CalibrationKind } from "../calibration/types";

// A recorder in any of these states owns the devices and is streaming live data.
const CONNECTED_STATES = ["connecting", "armed", "recording", "review", "saving", "discarding"];
// Actively writing an episode -- disconnecting here would abort it, so the
// global control defers to Live Record's Save/Discard/Stop instead.
const WRITING_STATES = ["recording", "saving", "discarding"];

export function DashboardPage({
  snapshot,
  busy,
  onConnect,
  onNavigate,
}: {
  snapshot: GuiSnapshot;
  busy: boolean;
  onConnect: () => void;
  onNavigate: (page: PageId) => void;
}) {
  const recorderConnected = CONNECTED_STATES.includes(snapshot.recording.state);
  const recorderWriting = WRITING_STATES.includes(snapshot.recording.state);

  const devices = snapshot.devices;
  const online = devices.filter((d) => d.state === "running");
  const offline = devices.filter((d) => d.state !== "running");

  // Calibration readiness: worst-case across the three kinds, read from the same
  // local history the nav badge uses so the numbers agree everywhere.
  const records = historyRepo.list();
  const latestByKind: Partial<Record<CalibrationKind, number | null>> = {
    force_origin: records.find((r) => r.kind === "force_origin")?.timestamp ?? null,
    force_dynamic: records.find((r) => r.kind === "force_dynamic")?.timestamp ?? null,
    touch: records.find((r) => r.kind === "touch")?.timestamp ?? null,
  };
  const cali = summarizeKinds(latestByKind);
  const caliNeeds = cali.overdue + cali.unknown;
  const caliTone = caliNeeds > 0 ? "error" : cali.dueSoon > 0 ? "warning" : "running";
  const caliText = caliNeeds > 0 ? `${caliNeeds} 项需标定` : cali.dueSoon > 0 ? `${cali.dueSoon} 项即将过期` : "全部有效";

  // Data pipeline funnel from the recorded-session processing list.
  const countOf = (...statuses: ProcessingStatus[]) =>
    snapshot.processing.filter((p) => statuses.includes(p.status)).length;
  const funnel: { label: string; value: number }[] = [
    { label: "已录制", value: snapshot.processing.length },
    { label: "待生成 Pose", value: countOf("pose_missing") },
    { label: "生成中", value: countOf("queued", "running") },
    { label: "Pose 就绪", value: countOf("pose_ready") },
    { label: "QC 通过", value: countOf("qc_pass") },
    { label: "QC 警告", value: countOf("qc_warn") },
    { label: "QC 失败/错误", value: countOf("qc_failed", "error") },
  ];
  const latest = snapshot.recordedDatasets.find((d) => d.isLatest) ?? snapshot.recordedDatasets[0] ?? null;
  const exportedCount = snapshot.recordedDatasets.filter((d) => d.datasetKind === "exported").length;

  const activeTask = snapshot.tasks.find((t) => t.id === snapshot.activeTaskId) ?? null;

  return (
    <div className="page-stack">
      <PageHeader title="Overview" subtitle="跨页状态总览与快捷入口（只读聚合，动作跳转到对应页面）" />

      {/* --- session / connection: the one global action every page relies on --- */}
      <section className="panel">
        <div className="panel-heading">
          <h2>会话 · 连接</h2>
          <span>录制器供录制 / 标定 / 设备预览共用</span>
        </div>
        <div className="summary-grid">
          <Metric label="Gateway" value={snapshot.gateway.state} />
          <Metric label="Recorder" value={stateLabel(snapshot.recording.state)} />
          <Metric label="设备在线" value={`${online.length}/${devices.length}`} />
          <Metric label="当前 Task" value={activeTask?.name ?? "未绑定"} />
        </div>
        <div className="control-row">
          {recorderConnected ? (
            <span className="dashboard-inline-status">
              <StatusDot state="running" /> 录制器已连接
              {recorderWriting && <em className="dashboard-muted">（录制进行中，请在 Live Record 停止）</em>}
            </span>
          ) : (
            <button disabled={busy} onClick={onConnect} title="按当前绑定 Task 启动录制器会话">
              连接录制器
            </button>
          )}
          <button onClick={() => onNavigate("live-record")}>去 Live Record →</button>
        </div>
      </section>

      {/* --- calibration readiness --- */}
      <section className="panel">
        <div className="panel-heading">
          <h2>标定就绪</h2>
          <span className={`cali-badge cali-badge-${caliTone}`}>
            <span className={`status-dot status-${caliTone}`} />
            {caliText}
          </span>
        </div>
        <div className="summary-grid">
          <Metric label="需标定" value={caliNeeds} />
          <Metric label="即将过期" value={cali.dueSoon} />
          <Metric label="有效" value={cali.valid} />
          <Metric label="多相机标定" value={stateLabel(snapshot.calibration.state)} />
        </div>
        <div className="control-row">
          <button onClick={() => onNavigate("calibration")}>去 Calibration →</button>
        </div>
      </section>

      {/* --- device health --- */}
      <section className="panel">
        <div className="panel-heading">
          <h2>设备健康</h2>
          <span>{online.length}/{devices.length} 在线</span>
        </div>
        {offline.length === 0 ? (
          <p className="dashboard-muted">全部设备在线。</p>
        ) : (
          <ul className="dashboard-device-list">
            {offline.map((d) => (
              <li key={d.id}>
                <StatusDot state={d.state} />
                <strong>{d.label || d.id}</strong>
                <span className="dashboard-muted">{stateLabel(d.state)}</span>
              </li>
            ))}
          </ul>
        )}
        <div className="control-row">
          <button onClick={() => onNavigate("device-manager")}>去 Device Manager →</button>
        </div>
      </section>

      {/* --- data pipeline funnel --- */}
      <section className="panel">
        <div className="panel-heading">
          <h2>数据管线</h2>
          <span>最近录制：{latest ? latest.name : "—"}</span>
        </div>
        <div className="summary-grid">
          {funnel.map((f) => (
            <Metric key={f.label} label={f.label} value={f.value} />
          ))}
          <Metric label="已导出" value={exportedCount} />
        </div>
        <div className="control-row">
          <button onClick={() => onNavigate("dataset-processing")}>去 Processing →</button>
          <button onClick={() => onNavigate("episode-replay")}>去 Replay →</button>
          <button onClick={() => onNavigate("dataset-export")}>去 Export →</button>
        </div>
        {snapshot.processing.some((p) => p.status === "error") && (
          <p className="dashboard-muted">
            存在处理错误的会话：{snapshot.processing.filter((p) => p.status === "error").length} 个（{processingStatusLabel.error}）
          </p>
        )}
      </section>
    </div>
  );
}
