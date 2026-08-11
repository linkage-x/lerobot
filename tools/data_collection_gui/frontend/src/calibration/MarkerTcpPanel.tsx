import { useState } from "react";
import type { DataCollectionGuiApi, GuiSnapshot } from "../api";
import type { MarkerTcpSample, MarkerTcpSession } from "../types";
import { Metric, StatusDot, stateLabel } from "../shared/ui";

type Side = "left" | "right";

const EMPTY_SESSION: MarkerTcpSession = {
  active: false,
  sessionName: "",
  sessionRoot: "",
  stage: "idle",
  samples: [],
  pendingSampleId: "",
  message: "Marker→TCP repeatability session not started",
  reportPath: ""
};

const stageTone = (stage: string) => {
  if (stage === "failed") return "error";
  if (stage === "done") return "done";
  if (stage === "capture" || stage === "reporting") return "warning";
  return "idle";
};

function sampleLabel(sample: MarkerTcpSample) {
  if (sample.staticTransformPath) return sample.staticTransformPath;
  if (sample.datasetRoot) {
    const episode = sample.episodeIndex >= 0 ? `episode ${sample.episodeIndex}` : "episode —";
    return `${sample.datasetRoot} · ${episode}`;
  }
  return "—";
}

export function MarkerTcpPanel({
  snapshot,
  api,
  busy,
}: {
  snapshot: GuiSnapshot;
  api: DataCollectionGuiApi;
  busy: boolean;
}) {
  const [side, setSide] = useState<Side>("left");
  const [condition, setCondition] = useState("same_mount_01");
  const [staticPath, setStaticPath] = useState("");
  const [error, setError] = useState("");
  const [pending, setPending] = useState(false);
  const session = snapshot.markerTcp ?? EMPTY_SESSION;
  const pendingSample = session.samples.find((sample) => sample.id === session.pendingSampleId);
  const disabled = busy || pending;
  const canReport = session.samples.filter((sample) => sample.status === "registered" && sample.staticTransformPath).length >= 2;

  const call = async (fn: () => Promise<{ ok: boolean; error?: string }>) => {
    setPending(true);
    setError("");
    const result = await fn();
    setPending(false);
    if (!result.ok) setError(result.error || "操作失败");
  };

  return (
    <section className="panel marker-tcp-panel">
      <div className="panel-heading">
        <h2>UMI Marker→TCP 重复性</h2>
        <span className="state-pill">
          <StatusDot state={stageTone(session.stage)} />
          {stateLabel(session.stage)}
        </span>
      </div>

      {!session.active ? (
        <>
          <p className="panel-note">左右 UMI cube 的 marker→TCP 样本入口；不依赖 FR3，也不包含 head cube。</p>
          <div className="control-row">
            <button className="cali-btn-primary" disabled={disabled} onClick={() => call(() => api.startMarkerTcpSession())}>
              开始 UMI 采集会话
            </button>
          </div>
        </>
      ) : (
        <>
          <div className="summary-grid">
            <Metric label="会话" value={session.sessionName || "—"} />
            <Metric label="样本" value={session.samples.length} />
            <Metric label="录制器" value={snapshot.recording.state} />
            <Metric label="报告" value={session.reportPath || "—"} />
          </div>

          <p className="panel-note">{session.message}</p>

          <div className="marker-tcp-controls">
            <label>
              UMI
              <select value={side} disabled={disabled || Boolean(pendingSample)} onChange={(event) => setSide(event.target.value as Side)}>
                <option value="left">left</option>
                <option value="right">right</option>
              </select>
            </label>
            <label>
              条件
              <input
                value={condition}
                disabled={disabled || Boolean(pendingSample)}
                placeholder="same_mount_01"
                onChange={(event) => setCondition(event.target.value)}
              />
            </label>
          </div>

          <div className="control-row">
            {!pendingSample ? (
              <button
                className="cali-btn-primary"
                disabled={disabled || !condition.trim()}
                onClick={() => call(() => api.markerTcpRecordSample("start", side, condition))}
              >
                录制样本
              </button>
            ) : (
              <>
                <button className="cali-btn-primary" disabled={disabled} onClick={() => call(() => api.markerTcpRecordSample("save", side, condition))}>
                  保存样本
                </button>
                <button disabled={disabled} onClick={() => call(() => api.markerTcpRecordSample("discard", side, condition))}>
                  丢弃样本
                </button>
              </>
            )}
            <button disabled={disabled || Boolean(pendingSample)} onClick={() => call(() => api.cancelMarkerTcpSession())}>
              结束会话
            </button>
          </div>

          <div className="marker-tcp-controls marker-tcp-register">
            <label>
              static_transform.json
              <input
                value={staticPath}
                disabled={disabled}
                placeholder="outputs/.../static_transform.json"
                onChange={(event) => setStaticPath(event.target.value)}
              />
            </label>
            <button disabled={disabled || !staticPath.trim()} onClick={() => call(() => api.registerMarkerTcpStaticTransform(staticPath, side, condition))}>
              登记结果
            </button>
            <button className="cali-btn-primary" disabled={disabled || !canReport} onClick={() => call(() => api.runMarkerTcpReport())}>
              生成报告
            </button>
          </div>

          {error ? <p className="panel-note error">{error}</p> : null}

          {session.samples.length ? (
            <div className="check-table calibration-table marker-tcp-table">
              {session.samples.map((sample) => (
                <div className="check-row" key={sample.id}>
                  <strong>
                    <StatusDot state={sample.status === "discarded" ? "error" : sample.status === "recording" ? "recording" : "running"} />
                    {sample.side} · {sample.condition}
                  </strong>
                  <span>{sampleLabel(sample)}</span>
                  <em>{sample.status}</em>
                </div>
              ))}
            </div>
          ) : null}
        </>
      )}
    </section>
  );
}
