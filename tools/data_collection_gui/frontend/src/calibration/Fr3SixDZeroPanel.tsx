// The 6D zero of an FR3 gripper BOX with the arm as the fixture -- tools/data_collection_gui/fr3_six_d_zero.py.
//
// The vendor procedure zeros the 6D sensor upright, fingertips up, never by hand. On the arm the robot holds
// that attitude: start -> fingertips up (j2/j4/j6 only) -> attitude check -> [origin + re-power + raw check]
// -> dynamic zero -> start -> two re-check poses with the registered 6D fit -> start. The zeros go through
// the recorder (the software zero lives in its process), so unlike the 6D fit this needs Connect.
import { useCallback, useEffect, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import { ArmPicker, useRobotArms } from "../shared/ArmPicker";
import { Modal } from "./ConfirmModal";
import { PayloadJobView } from "./PayloadJobView";
import {
  checkKey,
  isJobActive,
  isZeroJob,
  ZERO_MODE_LABELS,
  ZERO_POSES,
  zeroBlocker,
  type PayloadCalibrationResponse,
  type ZeroCheck,
} from "./payloadCalibration";

const POLL_MS = 1000;

const ZERO_CHECKLIST = [
  { id: "connect", label: "录制器已 Connect，夹爪 BOX 在线（零点经录制器做，软件零点存在录制器进程里）" },
  { id: "start", label: "机械臂在 start 姿态（Move to start）" },
  { id: "overhead", label: "基座正上方约 1.4 m 内没有相机支架、灯架等（指尖会到基座面以上约 1.33 m）" },
  { id: "grip", label: "夹爪空载、开度和做 6D 拟合时相同；线缆有余量，不跨接传感器两侧" },
  { id: "estop", label: "有人站在急停旁边，全程看着" },
];
const ORIGIN_ITEM = { id: "repower", label: "准备好给 BOX 断电再上电（机械臂会停在指尖朝上，页面提示时再断电）" };

function CheckRows({ checks }: { checks: ZeroCheck[] }) {
  return (
    <>
      {checks.map((c) => (
        <div key={c.id} className={c.pass ? "payload-verdict-ok" : "payload-verdict-bad"}>
          {c.pass ? "✓" : "✕"} {c.label}：<span className="payload-line">{c.detail}</span>
        </div>
      ))}
    </>
  );
}

export function Fr3SixDZeroPanel({ api, busy }: { api: DataCollectionGuiApi; busy: boolean }) {
  const arms = useRobotArms();
  const [payload, setPayload] = useState<PayloadCalibrationResponse | null>(null);
  const [arm, setArm] = useState("");
  const [mode, setMode] = useState("dynamic");
  const [table, setTable] = useState<{ edgeM: string; topM: string; marginMm: string } | null>(null);
  const [running, setRunning] = useState("");
  const [error, setError] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [ticked, setTicked] = useState<Record<string, boolean>>({});

  const refresh = useCallback(async () => {
    const next = await api.fetchPayloadCalibration();
    if (next) setPayload(next);
  }, [api]);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  useEffect(() => {
    if (payload && !table) {
      setTable({ edgeM: String(payload.table.edgeM), topM: String(payload.table.topM), marginMm: String(payload.table.marginMm) });
    }
  }, [payload, table]);
  const job = payload?.job ?? {};
  const active = isJobActive(job);
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [active, refresh]);

  const act = async (label: string, action: string, body: Record<string, unknown> = {}) => {
    setRunning(label);
    setError("");
    const result = await api.payloadCalibrationPost(action, body);
    setRunning("");
    if (!result) {
      setError("网关没有响应");
      return;
    }
    if (result.runs) setPayload(result);
    if (!result.ok) setError(result.error || "操作失败");
  };

  const nowS = Date.now() / 1000;
  const check = payload?.checks[checkKey(arm, ZERO_POSES)] ?? null;
  const blocker = zeroBlocker(payload, arm, nowS);
  const armRow = payload?.arms.find((a) => a.arm === arm);
  const disabled = busy || running !== "" || active;
  const mine = isZeroJob(job);
  const checklist = mode === "origin" ? [...ZERO_CHECKLIST, ORIGIN_ITEM] : ZERO_CHECKLIST;
  const allTicked = checklist.every((c) => ticked[c.id]);
  const reports = payload?.zeroReports ?? [];

  const runCheck = () => {
    if (!table) return;
    void act("check", "check", {
      arm,
      poses: ZERO_POSES,
      tableEdgeM: Number(table.edgeM),
      tableTopM: Number(table.topM),
      tableMarginMm: Number(table.marginMm),
    });
  };
  const start = async () => {
    setConfirming(false);
    setTicked({});
    await act("start", "fr3-six-d-zero/start", { arm, mode, confirmed: true });
  };

  return (
    <section className="panel">
      <div className="cali-result-head">
        <h3>夹爪 6D 零点（机械臂当工装，指尖朝上）</h3>
        <div className="cali-op-actions">
          <span className="cali-preview-badge">机械臂会自动运动</span>
          <button className="cali-mini-btn" disabled={running !== ""} onClick={() => void refresh()}>
            刷新
          </button>
        </div>
      </div>
      <p className="cali-muted">
        厂家手册要求 6D 在「竖直、指尖朝上」的工装姿态校零，禁止手持。臂上的夹爪由机械臂摆到这个姿态：从 start 只动 j2/j4/j6
        转到指尖朝上（j7 不转，线缆走向不变）→ 静止后用 IMU 核对姿态 → 校零 → 回 start → 侧放两个朝向用已登记的 6D 拟合复核外力
        ≤0.5 N → 回 start。动态校零每 30 分钟一次；每天第一次选 origin（之后要给 BOX 重新上电，机械臂会停着等）。
        录制器要先 Connect：校零经录制器做，Disconnect 后要重做。
      </p>

      <ol className="cali-steps">
        <li className="cali-step">
          <div className="cali-step-title">① 选臂和校零方式</div>
          <ArmPicker arms={arms} value={arm} onChange={setArm} disabled={disabled} label="机械臂" />
          {armRow && <span className="cali-muted">夹爪 BOX {armRow.gripperBox ?? "未登记"}</span>}
          <div className="payload-choices" role="radiogroup" aria-label="校零方式">
            {(["dynamic", "origin"] as const).map((m) => (
              <label key={m} className={mode === m ? "payload-choice active" : "payload-choice"}>
                <input type="radio" checked={mode === m} disabled={disabled} onChange={() => setMode(m)} />
                {ZERO_MODE_LABELS[m]}
                <span className="cali-muted">{m === "dynamic" ? "（每 30 分钟）" : "（每天第一次，BOX 要预热满 30 分钟）"}</span>
              </label>
            ))}
          </div>
        </li>

        <li className="cali-step">
          <div className="cali-step-title">② 零点路线碰撞检查（含桌面）</div>
          <p className="cali-muted">
            MuJoCo 沿整条路线（start → 指尖朝上 → start → x+ 朝上 → y− 朝上 → start）检查工具与机械臂的间隙和桌面。
            零点姿态很高（指尖在基座面以上约 1.33 m）、在基座轴线后方约 14 cm：头顶的相机支架<b>没有</b>建模，要目测。
          </p>
          {table && (
            <div className="payload-row">
              <label className="cali-field">
                桌沿 x (m)
                <input value={table.edgeM} disabled={disabled} onChange={(e) => setTable({ ...table, edgeM: e.target.value })} />
              </label>
              <label className="cali-field">
                桌面 z (m，高于基座)
                <input value={table.topM} disabled={disabled} onChange={(e) => setTable({ ...table, topM: e.target.value })} />
              </label>
              <label className="cali-field">
                余量 (mm)
                <input value={table.marginMm} disabled={disabled} onChange={(e) => setTable({ ...table, marginMm: e.target.value })} />
              </label>
              <button disabled={disabled || !arm} onClick={runCheck}>
                {running === "check" ? "提交中…" : "运行检查"}
              </button>
            </div>
          )}
          {check && (
            <div className={`cali-result-box ${check.pass ? "cali-result-pass" : "cali-result-fail"}`}>
              <div className="cali-result-box-head">
                <span className={`cali-badge cali-badge-${check.pass ? "running" : "error"}`}>{check.pass ? "通过" : "未通过"}</span>
                <span className="cali-muted">
                  {new Date(check.atUnixS * 1000).toLocaleString()} · 桌沿 {check.table.edgeM} m / 桌面 {check.table.topM} m / 余量{" "}
                  {check.table.marginMm} mm
                </span>
              </div>
              {check.summary.map((line) => (
                <code key={line} className="payload-line">
                  {line}
                </code>
              ))}
            </div>
          )}
        </li>

        <li className="cali-step">
          <div className="cali-step-title">③ 校零（机械臂运动）</div>
          <div className="payload-row">
            <button className="cali-btn-primary" disabled={disabled || blocker !== ""} onClick={() => setConfirming(true)}>
              开始校零
            </button>
            {blocker && !active ? <span className="cali-muted">{blocker}</span> : null}
          </div>
          {mine && job.kind === "six_d_zero" && (
            <>
              {active && job.phase ? <p className="cali-info">当前：{job.phase}</p> : null}
              {active && job.prompt ? <p className="cali-guard">{job.prompt}</p> : null}
              {job.checks && job.checks.length > 0 && (
                <div className="cali-result-box">
                  <CheckRows checks={job.checks} />
                </div>
              )}
              {(job.warnings ?? []).map((w) => (
                <p key={w} className="cali-warn">
                  {w}
                </p>
              ))}
            </>
          )}
          {mine && job.state ? (
            <PayloadJobView job={job} stopping={running === "stop"} onStop={() => void act("stop", "fr3-six-d-zero/stop")} />
          ) : active ? (
            <p className="cali-guard">另一个任务在进行：{job.label}</p>
          ) : null}
        </li>
      </ol>

      {reports.length > 0 && (
        <details>
          <summary className="cali-muted">最近的零点记录（{payload?.zeroReportRoot}）</summary>
          <table className="metric-table">
            <thead>
              <tr>
                <th>时间</th>
                <th>臂 / BOX</th>
                <th>方式</th>
                <th>结果</th>
                <th>检查</th>
              </tr>
            </thead>
            <tbody>
              {reports.map((r) => (
                <tr key={r.name}>
                  <td>
                    <code>{r.created.replace("T", " ")}</code>
                  </td>
                  <td>
                    {r.arm} / {r.deviceId}
                  </td>
                  <td>{ZERO_MODE_LABELS[r.mode] ?? r.mode}</td>
                  <td className={r.pass ? "payload-verdict-ok" : "payload-verdict-bad"}>{r.pass ? "通过" : r.state === "done" ? "未通过" : r.state}</td>
                  <td className="cali-muted">
                    {r.checks.map((c) => `${c.pass ? "✓" : "✕"}${c.label}`).join(" ")}
                    {r.warnings.length ? ` · ${r.warnings.length} 条提醒` : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </details>
      )}

      {error && <p className="cali-warn">{error}</p>}

      {confirming && (
        <Modal
          title="开始 6D 零点（机械臂会动）"
          onClose={() => setConfirming(false)}
          footer={
            <>
              <button onClick={() => setConfirming(false)}>取消</button>
              <button className="cali-btn-primary" disabled={!allTicked} onClick={() => void start()}>
                开始（机械臂会动）
              </button>
            </>
          }
        >
          <p>
            {arm} 臂（{armRow?.robotIp}），夹爪 BOX {armRow?.gripperBox}，{ZERO_MODE_LABELS[mode]}。速度 0.1；任何时候可点「停止」，
            机械臂会停在当时的位置。
          </p>
          <p className="cali-muted">
            BOX 应先上电预热 30 分钟；不满也能开始，任务会按 BOX 时钟记一条警告并写进报告（零点可能随温度漂）。
          </p>
          <div className="cali-confirm-list">
            {checklist.map((c) => (
              <label key={c.id} className="cali-check">
                <input type="checkbox" checked={!!ticked[c.id]} onChange={(e) => setTicked({ ...ticked, [c.id]: e.target.checked })} />
                {c.label}
              </label>
            ))}
          </div>
        </Modal>
      )}
    </section>
  );
}
