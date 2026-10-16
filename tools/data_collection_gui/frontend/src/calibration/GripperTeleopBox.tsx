// FR3 gripper teleop from the handheld BOX: enter / exit, with the live readout.
// See gripperTeleop.ts for the wording and tools/fr3/fr3_gripper_teleop.py for the loop.
import { useCallback, useEffect, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { GripperTeleopResponse } from "../types";
import { Modal } from "./ConfirmModal";
import { isTeleopRunning, teleopStatusText, teleopWarnings } from "./gripperTeleop";

const POLL_MS = 400;
const LOG_TAIL = 6;

export function GripperTeleopBox({ api, busy }: { api: DataCollectionGuiApi; busy: boolean }) {
  const [payload, setPayload] = useState<GripperTeleopResponse | null>(null);
  const [running, setRunning] = useState("");
  const [error, setError] = useState("");
  const [confirming, setConfirming] = useState(false);

  const refresh = useCallback(async () => {
    const result = await api.fetchGripperTeleop();
    if (result) setPayload(result);
  }, [api]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const active = isTeleopRunning(payload?.job);
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [active, refresh]);

  const act = async (label: string, action: () => Promise<GripperTeleopResponse | null>) => {
    setRunning(label);
    setError("");
    const result = await action();
    setRunning("");
    if (!result) {
      setError("网关没有响应");
      return;
    }
    setPayload(result);
    if (!result.ok) setError(result.error || "操作失败");
  };

  const status = teleopStatusText(payload);
  const disabled = busy || running !== "";
  const log = payload?.job?.log ?? [];

  return (
    <div className="cali-result-box">
      <div className="cali-result-box-head">
        <b>⓪ FR3 夹爪 · 手持遥控</b>
        <span className={status.warn ? "cali-warn" : "cali-muted"}>{status.text}</span>
      </div>
      <p className="cali-muted">
        用来让 FR3 夹爪夹住插件 v2：FR3 夹爪跟着手持夹爪开合，<b>扳机按到底（≥{payload?.graspTriggerClosePct ?? 90}%）就是夹紧</b>
        （和回放同一个规则）。进入后要先把手持夹爪调到与 FR3 当前开度一致才开始跟随。夹紧后<b>保持按住扳机，点「退出遥控」，再松手</b>：
        退出不会卸力，FR3 夹爪留在控制模式、保持最后的目标。只用 BOX 网络、不碰 FCI，所以 FR3 示教模式（下面的 pivot 采集）开着也可以用；
        但录制器连着或真机 replay 在跑时不行（它们占着 BOX 端口），遥控期间它们也会被拒绝。
      </p>
      <div className="cali-op-actions">
        {active ? (
          <button className="cali-btn-primary" disabled={running !== ""} onClick={() => act("stop", () => api.stopGripperTeleop())}>
            {running === "stop" ? "退出中…" : "退出遥控（保持夹紧）"}
          </button>
        ) : (
          <button className="cali-btn-primary" disabled={disabled || payload?.armDeviceId == null} onClick={() => setConfirming(true)}>
            进入遥控
          </button>
        )}
        {payload?.armDeviceId != null ? <span className="cali-muted">FR3 夹爪 BOX {payload.armDeviceId}</span> : null}
      </div>
      {teleopWarnings(payload).map((w) => (
        <p className="cali-warn" key={w}>
          {w}
        </p>
      ))}
      {error ? <pre className="cali-warn">{error}</pre> : null}
      {log.length ? <pre className="marker-tcp-example">{log.slice(-LOG_TAIL).join("\n")}</pre> : null}

      {confirming ? (
        <Modal
          title="FR3 夹爪跟随手持夹爪"
          onClose={() => setConfirming(false)}
          footer={
            <>
              <button onClick={() => setConfirming(false)}>取消</button>
              <button
                className="cali-btn-primary"
                onClick={() => {
                  setConfirming(false);
                  void act("start", () => api.startGripperTeleop());
                }}
              >
                进入遥控
              </button>
            </>
          }
        >
          <p>
            FR3 夹爪会<b>跟着手持夹爪开合</b>，扳机按到底时全力夹紧。手指不要放在 FR3 两指之间。
          </p>
          <p className="cali-muted">机械臂本身不会动。手持夹爪要开机、和 FR3 夹爪 BOX 在同一网络。</p>
        </Modal>
      ) : null}
    </div>
  );
}
