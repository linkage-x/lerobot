// BOX 6D payload fit (K) -- tools/thor/gmsl2/box_six_d.py.
//
// What the 6D sensor carries (gripper, rig, brackets) pulls on it with gravity at every attitude;
// K (6x3) maps the IMU's specific force to that wrench, so it can be taken out of every sample.
// K belongs to what is mounted, not to the BOX: every change on the payload side needs a new fit,
// registered in tools/thor/box_devices.json from the time of the change.
//
// Two ways to turn the sensor through its six faces: by hand (a capture here, any BOX), or held by
// the FR3 (the payload panel's "负载 + 6D" run, for the arm's gripper BOX). Both land in the list
// below with their fit, and either can be registered.
import { useCallback, useEffect, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import { PayloadJobView } from "./PayloadJobView";
import {
  captureDeviceChoices,
  faceLabel,
  faceProgress,
  fmtG,
  fmtMm,
  fromLocalInput,
  isJobActive,
  sixDFitNotes,
  sixDVerdict,
  toLocalInput,
  TOOL_CONFIG_SHORT,
  type PayloadCalibrationResponse,
  type SixDFitSummary,
} from "./payloadCalibration";

const POLL_MS = 700;

type Source = {
  key: string; // capture:<dir> | run:<file>
  title: string;
  how: string;
  deviceId: string;
  startedUnix: number | null;
  endEffector: string;
  fit: SixDFitSummary | null;
  registered: boolean;
};

const TOOL_CONFIG_EE: Record<string, string> = {
  gripper: "gripper only (no camera, no SMR nest), on the FR3",
  gripper_cam: "gripper + camera/SMR adaptor, SMR out, on the FR3",
  gripper_cam_smr: "gripper + camera/SMR adaptor + SMR, on the FR3",
};

export function SixDFitPanel({ api, busy }: { api: DataCollectionGuiApi; busy: boolean }) {
  const [payload, setPayload] = useState<PayloadCalibrationResponse | null>(null);
  const [deviceId, setDeviceId] = useState("");
  const [durationS, setDurationS] = useState("300");
  const [endEffector, setEndEffector] = useState("");
  const [running, setRunning] = useState("");
  const [error, setError] = useState("");
  const [output, setOutput] = useState("");
  const [registering, setRegistering] = useState<Source | null>(null);
  const [regEE, setRegEE] = useState("");
  const [regFrom, setRegFrom] = useState("");
  const [regStatus, setRegStatus] = useState("");

  const refresh = useCallback(async () => {
    const next = await api.fetchPayloadCalibration();
    if (next) setPayload(next);
  }, [api]);
  useEffect(() => {
    void refresh();
  }, [refresh]);
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
    setOutput("");
    const result = await api.payloadCalibrationPost(action, body);
    setRunning("");
    if (!result) {
      setError("网关没有响应");
      return null;
    }
    if (result.runs) setPayload(result);
    if (!result.ok) setError(result.error || "操作失败");
    if (result.output) setOutput(result.output);
    return result;
  };

  const boxes = payload?.boxes ?? [];
  const choices = captureDeviceChoices(boxes);
  const disabled = busy || running !== "" || active;
  const captureJob = job.kind === "six_d_capture";
  const live = captureJob ? job.capture ?? null : null;

  const sources: Source[] = [
    ...(payload?.captures ?? []).map((c) => ({
      key: `capture:${c.name}`,
      title: c.name,
      how: "手持",
      deviceId: String(c.deviceId ?? ""),
      startedUnix: c.startedUnix,
      endEffector: c.endEffector,
      fit: c.fit,
      registered: !!c.registered,
    })),
    ...(payload?.runs ?? [])
      .filter((r) => r.sixD)
      .map((r) => ({
        key: `run:${r.name}`,
        title: r.name,
        how: `FR3 托举 · ${r.arm} · ${TOOL_CONFIG_SHORT[r.toolConfig] ?? r.toolConfig}`,
        deviceId: String(r.sixD?.deviceId ?? ""),
        startedUnix: null,
        endEffector: TOOL_CONFIG_EE[r.toolConfig] ?? "",
        fit: r.sixD?.fit ?? null,
        registered: false,
      })),
  ];

  const openRegister = (s: Source) => {
    setRegistering(s);
    setRegEE(s.endEffector);
    setRegFrom(toLocalInput(s.startedUnix));
    setRegStatus("");
  };
  const submitRegister = async () => {
    if (!registering) return;
    const result = await act("register", "six-d/register", {
      source: registering.key,
      deviceId: registering.deviceId,
      endEffector: regEE,
      validFromUnix: fromLocalInput(regFrom),
      status: regStatus,
    });
    if (result?.ok) setRegistering(null);
  };

  return (
    <section className="panel">
      <div className="cali-result-head">
        <h3>6D 负载补偿拟合（K）</h3>
        <div className="cali-op-actions">
          <button className="cali-mini-btn" disabled={running !== ""} onClick={() => void refresh()}>
            刷新
          </button>
        </div>
      </div>
      <p className="cali-muted">
        6D 传感器上挂的东西（夹爪、rig、支架）在每个姿态都会给它一个重力读数。拟合把 IMU 读数映射成这个读数（K，6×3），
        之后每个样本都能扣掉它，只剩外力。<b>K 跟着末端配置走，不跟着 BOX 走</b>：负载侧装/拆任何东西都要重新拟合，
        并从改动那一刻起登记到 <code>{payload?.registryPath ?? "tools/thor/box_devices.json"}</code>。
        手册顺序：上电预热 30 分钟 → 工装上 origin → 重新上电 → 工装上动态校准 → 再拟合。
      </p>

      <ol className="cali-steps">
        <li className="cali-step">
          <div className="cali-step-title">手持采集（任何 BOX）</div>
          <p className="cali-muted">
            握住 BOX 本体或手柄，夹爪悬空、不夹东西、开度不动。依次摆出六个姿态，每个<b>静止 2–3 秒</b>，姿态之间最好多停几个中间角度：
            <b>指尖朝上</b>、<b>指尖朝下</b>；夹爪侧放、<b>两指一上一下</b>，再翻 180° 换另一指在上；夹爪侧放、<b>两指并排</b>（一样高），再翻面。
            偏 30° 以内都算，哪个格子没满就补哪个。
            录制器要先 Disconnect（采集独占 BOX 端口）。从拿起那一刻开始计数。
          </p>
          <div className="payload-row">
            <label className="cali-field">
              BOX
              <select value={deviceId} disabled={disabled} onChange={(e) => setDeviceId(e.target.value)}>
                <option value="">选择 BOX…</option>
                {choices.map((c) => (
                  <option key={c.id} value={c.id}>
                    {c.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="cali-field payload-note">
              末端配置（登记时用）
              <input
                value={endEffector}
                disabled={disabled}
                placeholder="例如：gripper + gripper-holder V3 marker rig v9b (100/101/110/111)"
                onChange={(e) => setEndEffector(e.target.value)}
              />
            </label>
            <label className="cali-field">
              最长 (s)
              <input value={durationS} disabled={disabled} onChange={(e) => setDurationS(e.target.value)} />
            </label>
            <button
              className="cali-btn-primary"
              disabled={disabled || !deviceId}
              onClick={() => void act("capture", "six-d/capture", { deviceId, durationS: Number(durationS), endEffector })}
            >
              开始采集
            </button>
          </div>
          {captureJob && (
            <>
              <div className="face-grid">
                {faceProgress(live?.faces, live?.need).map((f) => (
                  <div key={f.face} className={f.frac >= 1 ? "face-tile done" : "face-tile"}>
                    <strong>{faceLabel(f.face)}</strong>
                    <div className="progress">
                      <div className="progress-bar" style={{ width: `${Math.round(f.frac * 100)}%` }} />
                    </div>
                    <span>{f.n}</span>
                  </div>
                ))}
              </div>
              <p className="cali-muted">
                {live?.picked_up ? "已拿起，正在计数" : "等待拿起（转动超过 30°/s 开始计数）"} · {live?.rows ?? 0} 个样本
                {live?.opening_mm != null ? ` · 开度 ${live.opening_mm} mm` : ""}
                {live?.done ? " · 六个姿态都够了，可以停止" : ""}
              </p>
              {active && (
                <button className="cali-btn-primary" disabled={running === "stop"} onClick={() => void act("stop", "six-d/stop")}>
                  {running === "stop" ? "正在停止…" : "停止并拟合"}
                </button>
              )}
              <PayloadJobView job={job} tail={6} />
            </>
          )}
          {active && !captureJob ? <p className="cali-guard">另一个任务在进行：{job.label}</p> : null}
        </li>

        <li className="cali-step">
          <div className="cali-step-title">FR3 托举（机械臂上的夹爪 BOX）</div>
          <p className="cali-muted">
            在「FR3 机械臂」页的负载辨识里选「负载 + 6D」或「仅 6D」：机械臂把夹爪依次摆成同样六个姿态（指尖朝上/朝下、四个侧放）、每个姿态静止数秒，比手持稳，也不会碰到负载。
            采完自动拟合，结果出现在下面。
          </p>
        </li>

        <li className="cali-step">
          <div className="cali-step-title">拟合结果与登记</div>
          {sources.length === 0 ? (
            <p className="cali-muted">还没有 6D 记录。</p>
          ) : (
            <table className="metric-table payload-runs">
              <thead>
                <tr>
                  <th>记录</th>
                  <th>BOX</th>
                  <th>方式</th>
                  <th>结果</th>
                  <th>刚体读数</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {sources.map((s) => {
                  const v = sixDVerdict(s.fit);
                  return (
                    <tr key={s.key}>
                      <td>
                        <code>{s.title.slice(0, 15)}</code>
                      </td>
                      <td>{s.deviceId}</td>
                      <td>{s.how}</td>
                      <td>
                        <span className={`payload-verdict-${v.tone}`}>{v.text}</span>
                        {sixDFitNotes(s.fit).map((n) => (
                          <div key={n.text} className={n.tone === "warn" ? "cali-warn" : "cali-muted"}>
                            {n.text}
                          </div>
                        ))}
                      </td>
                      <td className="cali-muted">
                        {s.fit && !s.fit.error ? `${fmtG(s.fit.massKg)} · ${fmtMm(s.fit.comInFtM)}` : "—"}
                      </td>
                      <td>
                        <div className="cali-op-actions">
                          <button
                            className="cali-mini-btn"
                            disabled={disabled}
                            onClick={() => void act("fit", "six-d/fit", { source: s.key })}
                          >
                            重新拟合
                          </button>
                          <button
                            className="cali-mini-btn"
                            disabled={disabled || !s.fit || !!s.fit.error || !s.fit.meetsTarget}
                            title={s.fit?.meetsTarget ? "" : "未达标的拟合不登记"}
                            onClick={() => openRegister(s)}
                          >
                            {s.registered ? "已登记" : "登记"}
                          </button>
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          <p className="cali-muted">
            「刚体读数」只是把 K 当成刚体时的质量/质心，用来发现装错，不是称重。登记写的是 Thor 上的文件（旧文件留 .bak）；
            下次 deploy 会用工作站的版本覆盖它，<b>登记后要把 box_devices.json 拷回仓库提交</b>。
          </p>
          {registering && (
            <div className="cali-result-box">
              <div className="cali-result-box-head">
                <strong>登记到 BOX {registering.deviceId}</strong>
                <span className="cali-muted">{registering.title}</span>
              </div>
              <label className="cali-field payload-note">
                末端配置（6D 传感器带的是什么）
                <input value={regEE} onChange={(e) => setRegEE(e.target.value)} />
              </label>
              <label className="cali-field">
                这个配置的装上时间（记录从这一刻起生效，不是拟合时间）
                <input type="datetime-local" value={regFrom} onChange={(e) => setRegFrom(e.target.value)} />
              </label>
              <label className="cali-field payload-note">
                备注（可空）
                <input value={regStatus} onChange={(e) => setRegStatus(e.target.value)} />
              </label>
              <div className="cali-op-actions">
                <button onClick={() => setRegistering(null)}>取消</button>
                <button className="cali-btn-primary" disabled={running !== "" || !regEE || !regFrom} onClick={() => void submitRegister()}>
                  写入 box_devices.json
                </button>
              </div>
            </div>
          )}
        </li>

        <li className="cali-step">
          <div className="cali-step-title">已登记</div>
          <table className="metric-table">
            <thead>
              <tr>
                <th>BOX</th>
                <th>角色</th>
                <th>生效时间</th>
                <th>末端配置</th>
                <th>状态</th>
              </tr>
            </thead>
            <tbody>
              {boxes.flatMap((b) =>
                b.records.length === 0
                  ? [
                      <tr key={b.deviceId}>
                        <td>{b.deviceId}</td>
                        <td>{b.role}{b.hand ? ` ${b.hand}` : ""}</td>
                        <td colSpan={3} className="payload-verdict-warn">
                          没有拟合（这只 BOX 的 6D 不能补偿）
                        </td>
                      </tr>,
                    ]
                  : b.records.map((r, i) => (
                      <tr key={`${b.deviceId}-${r.validFromUnix}`}>
                        <td>{i === 0 ? b.deviceId : ""}</td>
                        <td>{i === 0 ? `${b.role}${b.hand ? ` ${b.hand}` : ""}` : ""}</td>
                        <td>{r.validFrom}</td>
                        <td>{r.endEffector}</td>
                        <td className={r.fitted ? "" : "payload-verdict-warn"}>
                          {r.fitted ? `留一面 ${r.heldoutWorstForceN ?? "—"} N` : "待拟合"}
                        </td>
                      </tr>
                    )),
              )}
            </tbody>
          </table>
        </li>
      </ol>

      {error && <p className="cali-warn">{error}</p>}
      {output && (
        <details>
          <summary className="cali-muted">输出</summary>
          <pre className="payload-log">{output}</pre>
        </details>
      )}
    </section>
  );
}
