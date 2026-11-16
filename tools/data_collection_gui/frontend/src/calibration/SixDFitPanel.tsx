// BOX 6D payload fit (K) -- tools/thor/gmsl2/box_six_d.py.
//
// What the 6D sensor carries (gripper, rig, brackets) pulls on it with gravity at every attitude;
// K (6x3) maps the IMU's specific force to that wrench, so it can be taken out of every sample.
// K belongs to what is mounted, not to the BOX: every change on the payload side needs a new fit,
// registered from the time of the change.
//
// Two ways to turn the sensor through its six faces: by hand (a capture here, any BOX), or held by
// the FR3 (the payload panel's "负载 + 6D" run, for the arm's gripper BOX). Both land in the list
// below with their fit. A hand-held BOX's fit is registered here by whoever captured it, if the
// self-service gate passes (box_six_d_store): into this station's own log, in force at once for the
// export and the live wrench, no deploy. An arm BOX's K goes to the repo through a maintainer.
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
  SIX_D_SELF_SERVICE_DEFAULT,
  sixDFitNotes,
  sixDGateChecks,
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
  /** an FR3 lift run: an arm BOX, registered by a maintainer, not here */
  armRun: boolean;
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
  const choices = captureDeviceChoices(
    boxes,
    (payload?.arms ?? []).map((a) => a.gripperBox),
  );
  const disabled = busy || running !== "" || active;
  const rules = payload?.selfService ?? SIX_D_SELF_SERVICE_DEFAULT;
  const roleOf = (deviceId: string) => boxes.find((b) => b.deviceId === deviceId);
  const selfService = (s: Source) => !s.armRun && (roleOf(s.deviceId)?.selfService ?? roleOf(s.deviceId)?.role === "handheld");
  const gatePasses = (s: Source) => sixDGateChecks(s.fit, rules).every((c) => c.ok);
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
      armRun: false,
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
        armRun: true,
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
        之后每个样本都能扣掉它，只剩外力（导出的 <code>box_six_d_external.*</code> 和 BOX 页的实时外力都用它）。
        <b>K 跟着末端配置走，不跟着 BOX 走</b>：负载侧装/拆任何东西都要重新拟合，并从改动那一刻起登记。
        <b>手持 BOX 自己采、自己登记</b>：拟合过了下面的自动门槛就能登记，写进本站的{" "}
        <code>{payload?.localRegistryPath ?? "outputs/calibration/box_six_d/registry.jsonl"}</code>，导出和实时外力立刻生效，不用找维护者、不用 deploy。
        手册顺序：上电预热 30 分钟 → 工装上 origin → 重新上电 → 工装上动态校准 → 再拟合。
      </p>

      <ol className="cali-steps">
        <li className="cali-step">
          <div className="cali-step-title">手持采集（手持 BOX）</div>
          <p className="cali-muted">
            握住 BOX 本体或手柄，夹爪悬空、不夹东西、开度不动。依次摆出六个姿态，每个<b>静止 2–3 秒</b>，姿态之间最好多停几个中间角度：
            <b>指尖朝上</b>、<b>指尖朝下</b>；夹爪侧放、<b>两指一上一下</b>，再翻 180° 换另一指在上；夹爪侧放、<b>两指并排</b>（一样高），再翻面。
            偏 30° 以内都算，哪个格子没满就补哪个。<b>六面都满之后，回到第一个姿态再静止 3 秒</b>（和第一次隔 30 秒以上）：
            用来测这段时间里读数漂了多少，测不出漂移的拟合不能自助登记。
            BOX 要上电 30 分钟以上再采。录制器要先 Disconnect（采集独占 BOX 端口）。从拿起那一刻开始计数。
            装在机械臂上的夹爪 BOX 不在这个列表里：它装着时没法用手翻面，用下一步的 FR3 托举。
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
                {live?.done
                  ? ` · 六个姿态都够了，${live.revisited ? `${faceLabel(live.revisited)}也复查过，` : ""}可以停止`
                  : live?.faces_done
                    ? " · 六个姿态都够了：回到第一个姿态再静止 3 秒（测漂移），然后停止"
                    : ""}
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
                        {selfService(s) && s.fit && !s.fit.error && !gatePasses(s) && (
                          <div className="cali-warn">
                            不能自助登记：
                            {sixDGateChecks(s.fit, rules)
                              .filter((c) => !c.ok)
                              .map((c) => `${c.label}（${c.detail}）`)
                              .join("；")}
                          </div>
                        )}
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
                          {selfService(s) ? (
                            <button
                              className="cali-mini-btn"
                              disabled={disabled || !gatePasses(s)}
                              title={gatePasses(s) ? "" : "没过自助登记门槛（点开看哪一项）"}
                              onClick={() => openRegister(s)}
                            >
                              {s.registered ? "已登记" : "登记"}
                            </button>
                          ) : (
                            <span className="cali-muted" title="臂上 BOX 的 K 由维护者审过后提交到仓库">
                              维护者登记
                            </span>
                          )}
                        </div>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          <p className="cali-muted">
            「刚体读数」只是把 K 当成刚体时的质量/质心，用来发现装错，不是称重。
            自助登记的门槛（手持 BOX）：留一面 ≤ {rules.heldoutN} N、六个姿态齐、BOX 预热 ≥ {Math.round(rules.warmupS / 60)} 分钟、
            回到同一朝向测过漂移且 ≤ {rules.driftN} N、装上时间不早于这只 BOX 现有最新的登记。
            登记只追加不改写：K 错了就重新采、按同一个装上时间再登记一次，新的那条接管之后所有导出（旧导出的标签链会和新的不同，不会被混在一起）。
            撤销一条登记只有维护者能做。
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
              <ul className="cali-muted">
                {sixDGateChecks(registering.fit, rules).map((c) => (
                  <li key={c.label} className={c.ok ? "" : "cali-warn"}>
                    {c.ok ? "✓" : "✗"} {c.label}：{c.detail}
                  </li>
                ))}
              </ul>
              <p className="cali-muted">
                写进本站登记，从装上时间起生效：之后导出、重新导出这段时间的数据都用这个 K。装上时间之后录的数据要重新导出才会换成新 K。
              </p>
              <div className="cali-op-actions">
                <button onClick={() => setRegistering(null)}>取消</button>
                <button className="cali-btn-primary" disabled={running !== "" || !regEE || !regFrom} onClick={() => void submitRegister()}>
                  登记（立即生效）
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
                <th>来源</th>
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
                        <td colSpan={4} className="payload-verdict-warn">
                          没有拟合（这只 BOX 的 6D 不能补偿）
                        </td>
                      </tr>,
                    ]
                  : b.records.map((r, i) => (
                      <tr key={`${b.deviceId}-${r.validFromUnix}-${r.registeredUnix ?? "repo"}`}>
                        <td>{i === 0 ? b.deviceId : ""}</td>
                        <td>{i === 0 ? `${b.role}${b.hand ? ` ${b.hand}` : ""}` : ""}</td>
                        <td>{r.validFrom}</td>
                        <td>{r.endEffector}</td>
                        <td className="cali-muted">
                          {r.store === "local" ? `本站 · ${r.registered ?? ""}` : "仓库（维护者）"}
                        </td>
                        <td className={r.retired || !r.fitted ? "payload-verdict-warn" : r.inForce ? "payload-verdict-ok" : ""}>
                          {r.retired
                            ? `已撤销：${r.retired.reason ?? ""}`
                            : r.fitted
                              ? `${r.inForce ? "现在生效 · " : ""}留一面 ${r.heldoutWorstForceN ?? "—"} N`
                              : "待拟合"}
                        </td>
                      </tr>
                    )),
              )}
            </tbody>
          </table>
        </li>
      </ol>

      {payload?.registryError && <p className="cali-warn">6D 登记读不了：{payload.registryError}</p>}
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
