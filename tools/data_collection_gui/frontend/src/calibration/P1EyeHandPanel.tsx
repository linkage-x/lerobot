import { useState } from "react";
import type { DataCollectionGuiApi, GuiSnapshot } from "../api";
import type { P1EyeHandStatus } from "../types";
import { Metric, StatusDot, stateLabel } from "../shared/ui";

const EMPTY: P1EyeHandStatus = {
  state: "idle",
  message: "P1_simple_eye_hand_calibration has not run",
  pid: null,
  runDir: "",
  calibrationPath: "",
  candidatePath: "",
  activePath: "",
  log: []
};

const running = new Set(["preparing", "capturing", "detecting", "retargeting"]);

function tone(state: string) {
  if (state === "failed") return "error";
  if (state === "ready" || state === "active") return "done";
  if (running.has(state)) return "warning";
  return "idle";
}

export function P1EyeHandPanel({ snapshot, api, busy }: { snapshot: GuiSnapshot; api: DataCollectionGuiApi; busy: boolean }) {
  const status = snapshot.p1EyeHand ?? EMPTY;
  const [confirmation, setConfirmation] = useState("");
  const [pending, setPending] = useState(false);
  const [error, setError] = useState("");
  const disabled = busy || pending;

  const call = async (fn: () => Promise<{ ok: boolean; error?: string }>) => {
    setPending(true);
    setError("");
    const result = await fn();
    setPending(false);
    if (!result.ok) setError(result.error || "操作失败");
  };

  return (
    <section className="panel p1-eye-hand-panel">
      <div className="panel-heading">
        <h2>P1_simple_eye_hand_calibration</h2>
        <span className="state-pill"><StatusDot state={tone(status.state)} />{stateLabel(status.state)}</span>
      </div>

      <p className="panel-note">
        使用 0720 robot-base extrinsics 数据库中的 50 个原始 pose/joint values：先过滤 TCP Z 低于 0.250469 m 的姿态（当前桌面接触实测 Z=0.100469 m，保留 150 mm 余量），再优先选择离历史可见相机更近且保持 TCP 覆盖多样性的姿态。观测夹爪附近 tag36h11 ID 56/57（有效黑框 55 mm；70 mm 仅为背板尺寸），
        联合重定位 FR3 base 与两个 TCP→tag 常量。通过质量门后只生成 P0 候选，必须再次点击「激活」。
      </p>
      <div className="callout danger-callout">
        <b>HIGH RISK · 真机运动</b>
        <p>
          Base 已移动后，历史 teach-pose 路径没有经过项目侧场景/碰撞模型验证。清空人员、负载、线缆与桌面障碍，并握住物理急停。
          GUI 取消只是软件停止，不能替代急停。
        </p>
      </div>

      <div className="summary-grid">
        <Metric label="阶段" value={status.state} />
        <Metric label="PID" value={status.pid ?? "—"} />
        <Metric label="运行目录" value={status.runDir || "—"} />
        <Metric label="标定结果" value={status.calibrationPath || "—"} />
        <Metric label="P0 候选" value={status.candidatePath || "—"} />
        <Metric label="Active" value={status.activePath || "—"} />
      </div>

      {!running.has(status.state) && (
        <div className="control-row">
          <label>
            输入 <code>P1_MOVE_FR3</code> 授权 50 个姿态的真机运动
            <input value={confirmation} onChange={(event) => setConfirmation(event.target.value)} />
          </label>
          <button
            className="cali-btn-primary"
            disabled={disabled || confirmation !== "P1_MOVE_FR3"}
            onClick={() => call(() => api.startP1EyeHand(confirmation))}
          >
            采集并解算候选
          </button>
          <button
            disabled={disabled || status.state !== "ready"}
            onClick={() => call(() => api.activateP1EyeHand())}
          >
            激活到下一次 P0 unchecked replay
          </button>
        </div>
      )}
      {running.has(status.state) && (
        <div className="control-row">
          <button className="danger" disabled={disabled} onClick={() => call(() => api.cancelP1EyeHand())}>取消（软件停止）</button>
        </div>
      )}

      <p className={status.state === "failed" || error ? "error-text" : "panel-note"}>{error || status.message}</p>
      {status.log.length > 0 && <pre className="marker-tcp-example">{status.log.slice(-40).join("\n")}</pre>}
    </section>
  );
}
