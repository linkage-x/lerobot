// BOX 6D payload fit (K) -- tools/thor/gmsl2/box_six_d.py.
//
// What the 6D sensor carries (gripper, rig, brackets) pulls on it with gravity at every attitude;
// K (6x3) maps the IMU's specific force to that wrench, so it can be taken out of every sample.
// K belongs to what is mounted, not to the BOX: every change on the payload side needs a new fit,
// registered from the time of the change.
//
// Two ways to turn the sensor through its six faces: by hand (a capture here, any BOX), or held by
// the FR3 (the payload panel's "负载 + 6D" run, for the arm's gripper BOX). Both land in the list
// below with their fit, and whoever captured it registers it here if the self-service gate passes
// (box_six_d_store), into this station's own log, no deploy. A hand-held BOX's K is in force at once;
// an arm BOX's waits for the next arm-held 6D zero to re-check it. A load that comes and goes (the
// SMR) is switched back to its earlier, already checked K with the 当前负载 picker: no new capture.
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
  sixDRecordState,
  sixDRegisterKind,
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
  /** an FR3 lift run: an arm BOX, its end effector from the run, pending until the arm zero */
  armRun: boolean;
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
  const [loadPick, setLoadPick] = useState<Record<string, string>>({});
  const [loadFrom, setLoadFrom] = useState<Record<string, string>>({});

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
  const registerKind = (s: Source) => sixDRegisterKind(roleOf(s.deviceId)?.role, s.armRun, rules);
  const selfService = (s: Source) => registerKind(s).kind !== "";
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
        startedUnix: r.sixD?.startedUnix ?? null,
        endEffector: r.sixD?.endEffector ?? "",
        fit: r.sixD?.fit ?? null,
        registered: (roleOf(String(r.sixD?.deviceId ?? ""))?.records ?? []).some((rec) =>
          (rec.status ?? "").includes(`FR3 lift run ${r.name}`),
        ),
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
  const applyLoad = async (deviceId: string, key: string) => {
    const choice = roleOf(deviceId)?.loads?.find((c) => c.key === key);
    if (!choice) return;
    await act("switch-back", "six-d/switch-back", {
      deviceId,
      sourceValidFromUnix: choice.validFromUnix,
      sourceRegisteredUnix: choice.registeredUnix,
      validFromUnix: fromLocalInput(loadFrom[deviceId] ?? "") ?? Math.round(Date.now() / 1000),
    });
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
        <b>自己采、自己登记</b>：拟合过了下面的自动门槛就能登记，写进本站的{" "}
        <code>{payload?.localRegistryPath ?? "outputs/calibration/box_six_d/registry.jsonl"}</code>，不用找维护者、不用 deploy。
        手持 BOX 登记后立刻生效；臂上 BOX 登记后<b>待复核</b>，BOX 预热后做一次臂上校零、两次复核都过了才生效。
        装上又拆下的东西（例如臂上的 SMR）不用每次重采：在下面「已登记」里把<b>当前负载</b>切回它以前验证过的那套 K。
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
            机械臂会动：要有人在急停旁边，由人来点开始。BOX 要先上电预热 30 分钟。采完自动拟合，结果出现在下面，过了门槛就能登记。
            末端配置按采集时选的「装了什么」自动写，不用填。登记后先算<b>待复核</b>：等 BOX 预热后在「6D 零点（臂上）」做一次校零，
            两个复核姿态都在 ±0.5 N 内这条 K 才生效；复核没过它就作废（重新采集登记，或把当前负载换回以前的 K）。
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
                        {!selfService(s) && s.fit && !s.fit.error && (
                          <div className="cali-muted">{registerKind(s).why}</div>
                        )}
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
                            <span className="cali-muted" title={registerKind(s).why}>
                              不能在此登记
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
            自助登记的门槛：留一面 ≤ {rules.heldoutN} N、六个姿态齐、BOX 预热 ≥ {Math.round(rules.warmupS / 60)} 分钟、
            回到同一朝向测过漂移且 ≤ {rules.driftN} N、装上时间不早于这只 BOX 现有最新的登记。
            臂上 BOX 另外要：一次跑完的 FR3 托举采集，采集里的 BOX 就是 fr3_arms.json 里这条臂的夹爪 BOX；登记后由臂上校零复核。
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
                <input
                  value={regEE}
                  readOnly={registering.armRun}
                  title={registering.armRun ? "按采集时选的末端配置自动写" : ""}
                  onChange={(e) => setRegEE(e.target.value)}
                />
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
                {registering.armRun
                  ? "写进本站登记，先算待复核：BOX 预热后做一次臂上校零，两次复核都过了才从装上时间起生效；在那之前这个装上时间之后的导出会拒绝，不会悄悄用旧配置的 K。"
                  : "写进本站登记，从装上时间起生效：之后导出、重新导出这段时间的数据都用这个 K。装上时间之后录的数据要重新导出才会换成新 K。"}
              </p>
              <div className="cali-op-actions">
                <button onClick={() => setRegistering(null)}>取消</button>
                <button className="cali-btn-primary" disabled={running !== "" || !regEE || !regFrom} onClick={() => void submitRegister()}>
                  {registering.armRun ? "登记（待臂上校零复核）" : "登记（立即生效）"}
                </button>
              </div>
            </div>
          )}
        </li>

        <li className="cali-step">
          <div className="cali-step-title">已登记</div>
          {boxes
            .filter((b) => b.selfService && (b.loads?.length ?? 0) > 0)
            .map((b) => {
              const loads = b.loads ?? [];
              const current = loads.find((c) => c.inForce);
              const pick = loadPick[b.deviceId] ?? current?.key ?? "";
              const chosen = loads.find((c) => c.key === pick);
              return (
                <div key={`load-${b.deviceId}`} className="payload-row">
                  <label className="cali-field payload-note">
                    BOX {b.deviceId}（{b.role === "fr3_gripper" ? "臂上" : `手持${b.hand ? ` ${b.hand}` : ""}`}）当前负载
                    <select
                      value={pick}
                      disabled={disabled}
                      onChange={(e) => setLoadPick({ ...loadPick, [b.deviceId]: e.target.value })}
                    >
                      {!current && <option value="">（现在没有生效的 K）</option>}
                      {loads.map((c) => (
                        <option key={c.key} value={c.key}>
                          {c.endEffector} · K 拟合自 {c.fittedFrom ?? "—"}
                          {c.inForce ? "（现在生效）" : ""}
                        </option>
                      ))}
                    </select>
                  </label>
                  <label className="cali-field">
                    从什么时候起（装上/取下的时间，空 = 现在）
                    <input
                      type="datetime-local"
                      value={loadFrom[b.deviceId] ?? ""}
                      disabled={disabled}
                      onChange={(e) => setLoadFrom({ ...loadFrom, [b.deviceId]: e.target.value })}
                    />
                  </label>
                  <button
                    className="cali-btn-primary"
                    disabled={disabled || !chosen || chosen.inForce}
                    title={chosen?.inForce ? "这就是现在生效的负载" : ""}
                    onClick={() => void applyLoad(b.deviceId, pick)}
                  >
                    应用这套 K
                  </button>
                </div>
              );
            })}
          <p className="cali-muted">
            「当前负载」只列这只 BOX 已经验证过的 K（每种末端配置最新的一条：维护者登记、手持自助登记、或臂上校零复核过的）。
            应用后从填的时间起生效，立即用于导出和实时外力，不用重新采集。系统看不出负载什么时候换过：
            装上/取下 SMR 或 rig 之后要记得在这里切换，不切就一直用旧的 K，外力会偏。没有出现在列表里的配置要先采集登记。
          </p>
          {boxes
            .filter((b) => b.notInForce && b.records.length > 0)
            .map((b) => (
              <p key={`nif-${b.deviceId}`} className="cali-warn">
                BOX {b.deviceId} 现在没有生效的 K：{b.notInForce}
              </p>
            ))}
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
                        <td className={sixDRecordState(r).tone ? `payload-verdict-${sixDRecordState(r).tone}` : ""}>
                          {sixDRecordState(r).text}
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
