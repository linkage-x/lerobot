// FR3 gripper close/open button: press to close, press again to open. Only shown where an arm has a
// gripper BOX (Thor). The command rides the connected recorder (it owns the BOX port).
// Device Manager shows it as its own panel; the Rollout page embeds it in the online preview, for the
// preview's arm only, so the gripper can be set for a probe without leaving the page.
import { useCallback, useEffect, useState } from "react";
import { api } from "../apiClient";
import { Metric } from "../shared/ui";
import type { ArmGripperResponse } from "../types";
import { armLabel, gripperActionLabel, lastCommandText, nextGripperAction, type GripperAction } from "./armGripper";

const POLL_MS = 500;
// After a command, the jaw takes a moment to travel; the label follows the live width only once it settled.
const SETTLE_MS = 1200;

export function ArmGripperPanel({ embedded = false, arm = "" }: { embedded?: boolean; arm?: string } = {}) {
  const [payload, setPayload] = useState<ArmGripperResponse | null>(null);
  const [busyArm, setBusyArm] = useState("");
  const [error, setError] = useState("");

  const refresh = useCallback(async () => {
    const next = await api.fetchArmGripper();
    if (next) setPayload(next);
  }, []);

  useEffect(() => {
    void refresh();
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [refresh]);

  const arms = (payload?.arms ?? []).filter((row) => !arm || row.arm === arm);
  if (!payload || arms.length === 0) return null;

  const press = async (arm: string, action: GripperAction) => {
    setBusyArm(arm);
    setError("");
    const result = await api.commandArmGripper(arm, action);
    if (!result) {
      setError("网关没有响应");
    } else {
      setPayload(result);
      if (!result.ok) setError(result.error || "操作失败");
    }
    window.setTimeout(() => {
      setBusyArm("");
      void refresh();
    }, result?.ok ? SETTLE_MS : 0);
  };

  const Wrapper = embedded ? "div" : "section";
  return (
    <Wrapper className={embedded ? "arm-gripper-embedded" : "panel"}>
      <div className="panel-heading">
        {embedded ? <h3>FR3 夹爪</h3> : <h2>FR3 夹爪</h2>}
        <span>按一下闭合，再按一下张开</span>
      </div>
      {payload.refusal && <p className="arm-gripper-note">{payload.refusal}</p>}
      {arms.map((row) => {
        const action = nextGripperAction(row, payload.openWidthMm);
        const busy = busyArm === row.arm;
        return (
          <div className="arm-gripper-row" key={row.arm}>
            <div className="row-title">
              <strong>{armLabel(row.arm)}</strong>
              <small>BOX {row.deviceId}</small>
            </div>
            <Metric label="开度" value={row.widthMm == null ? "-" : `${row.widthMm.toFixed(1)} mm`} />
            <button
              type="button"
              disabled={busy || Boolean(payload.refusal)}
              title={payload.refusal || (action === "close" ? "闭合到 replay 的抓取宽度，夹住会持续用力" : `张开到 ${payload.openWidthMm} mm`)}
              onClick={() => void press(row.arm, action)}
            >
              {busy ? "发送中…" : gripperActionLabel(action)}
            </button>
            <small className="arm-gripper-last">{lastCommandText(row.last)}</small>
          </div>
        );
      })}
      {error && <p className="arm-gripper-note arm-gripper-error">{error}</p>}
      <p className="arm-gripper-note">闭合用 replay 的抓取宽度（0 mm），夹到物体会持续夹紧；松手后夹爪保持在控制模式，不会自己张开。</p>
    </Wrapper>
  );
}
