// FR3 end-effector load identification (Desk) -- tools/fr3/fr3_payload_identification_runtime.py.
//
// The arm visits static poses on its own and weighs what hangs below the flange with its joint
// torque sensors. Four steps on one card: what is mounted -> collision check (with the table) ->
// collect (moves the arm; optionally logs the gripper BOX's 6D for the 6D fit) -> Desk configs.
// Nothing here edits Desk or fr3_arms.json: the page writes the Desk JSON to import and shows the
// fr3_arms.json entry to paste.
import { useCallback, useEffect, useMemo, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import { ArmPicker, useRobotArms } from "../shared/ArmPicker";
import { Modal } from "./ConfirmModal";
import { PayloadJobView } from "./PayloadJobView";
import {
  checkKey,
  collectBlocker,
  deskTcpRequest,
  fmtG,
  fmtMm,
  fmtN,
  fr3ArmsPayloadSnippet,
  isJobActive,
  isZeroJob,
  sixDFitNotes,
  sixDVerdict,
  TCP_SOURCE_LABELS,
  TOOL_CONFIG_SHORT,
  type PayloadCalibrationResponse,
  type TcpSource,
} from "./payloadCalibration";

const POLL_MS = 1000;

const COLLECT_CHECKLIST = [
  { id: "start", label: "机械臂已回到 start 姿态（Move to start）" },
  { id: "clear", label: "桌面上、臂周围没有会被碰到的东西（试管架、支架、另一条臂已转开）" },
  { id: "cable", label: "夹爪线缆有余量，腕部转一圈不会被拉扯" },
  { id: "estop", label: "有人站在急停旁边，全程看着" },
  { id: "desk", label: "已记下 Desk 当前的末端负载（采集会把它写进记录，用于闭环核对）" },
];
const SIX_D_CHECKLIST = [
  { id: "disconnect", label: "录制器已 Disconnect（采集要独占 BOX 端口）" },
  { id: "warm", label: "夹爪 BOX 已上电预热 30 分钟" },
  { id: "free", label: "夹爪处于空载（不夹东西），开度固定不动" },
];

export function Fr3PayloadPanel({ api, busy }: { api: DataCollectionGuiApi; busy: boolean }) {
  const arms = useRobotArms();
  const [payload, setPayload] = useState<PayloadCalibrationResponse | null>(null);
  const [arm, setArm] = useState("");
  const [toolConfig, setToolConfig] = useState("");
  const [poses, setPoses] = useState("payload+six_d");
  const [sixD, setSixD] = useState(true);
  const [table, setTable] = useState<{ edgeM: string; topM: string; marginMm: string } | null>(null);
  const [running, setRunning] = useState("");
  const [error, setError] = useState("");
  const [output, setOutput] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [ticked, setTicked] = useState<Record<string, boolean>>({});
  const [note, setNote] = useState("");
  const [picked, setPicked] = useState<string[]>([]);
  const [deskNames, setDeskNames] = useState<Record<string, string>>({});
  const [tcpZ, setTcpZ] = useState("");
  const [tcpSource, setTcpSource] = useState<TcpSource["kind"]>("estimate");
  const [tcpPivot, setTcpPivot] = useState("");
  const [shownDesk, setShownDesk] = useState("");
  const pivotFits = payload?.pivotFits ?? [];

  const refresh = useCallback(async () => {
    const next = await api.fetchPayloadCalibration();
    if (next) setPayload(next);
  }, [api]);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  useEffect(() => {
    if (payload && !table) {
      setTable({
        edgeM: String(payload.table.edgeM),
        topM: String(payload.table.topM),
        marginMm: String(payload.table.marginMm),
      });
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

  const nowS = Date.now() / 1000;
  const check = payload?.checks[checkKey(arm, poses)] ?? null;
  const blocker = collectBlocker(payload, { arm, poses, toolConfig, sixD }, nowS);
  const armRow = payload?.arms.find((a) => a.arm === arm);
  const disabled = busy || running !== "" || active;
  const jobMine = (job.kind === "check" || job.kind === "collect") && !isZeroJob(job);
  const runs = payload?.runs ?? [];
  const pickedRuns = runs.filter((r) => picked.includes(r.name));
  const pickedConfigs = [...new Set(pickedRuns.map((r) => r.toolConfig))];
  const desk = payload?.desks.find((d) => d.name === shownDesk) ?? payload?.desks[0] ?? null;
  const checklist = useMemo(() => [...COLLECT_CHECKLIST, ...(sixD ? SIX_D_CHECKLIST : [])], [sixD]);
  const allTicked = checklist.every((c) => ticked[c.id]);
  const today = new Date().toISOString().slice(0, 10);

  const runCheck = () => {
    if (!table) return;
    void act("check", "check", {
      arm,
      poses,
      tableEdgeM: Number(table.edgeM),
      tableTopM: Number(table.topM),
      tableMarginMm: Number(table.marginMm),
    });
  };
  const startCollect = async () => {
    setConfirming(false);
    setTicked({});
    await act("collect", "collect", { arm, poses, toolConfig, sixD, note });
  };
  const makeDesk = async () => {
    const names: Record<string, string> = {};
    for (const c of pickedConfigs) names[c] = deskNames[c] || `Corenetic ${TOOL_CONFIG_SHORT[c] ?? c} ${today}`;
    const tcp = deskTcpRequest(tcpSource, { zM: tcpZ, pivot: tcpPivot });
    if (!tcp) {
      setError(tcpSource === "pivot" ? "先选一条 pivot 拟合" : "TCP z 要填数字（米）");
      return;
    }
    const result = await act("desk", "desk", { runs: picked, names, tcp });
    if (result?.deskWritten) setShownDesk(result.deskWritten);
  };

  return (
    <section className="panel">
      <div className="cali-result-head">
        <h3>末端负载辨识（Desk 负载）</h3>
        <div className="cali-op-actions">
          <span className="cali-preview-badge">机械臂会自动运动</span>
          <button className="cali-mini-btn" disabled={running !== ""} onClick={() => void refresh()}>
            刷新
          </button>
        </div>
      </div>
      <p className="cali-muted">
        机械臂自己走一组静止姿态，用关节力矩算出法兰下面挂了多重、质心在哪，生成 Franka Desk 的末端配置。
        末端每装/拆一样东西（相机、SMR 靶座、换指尖）都要重做。选「负载 + 6D」时同一趟顺便把夹爪 BOX 的 6D 数据记下来，
        结果在「BOX 传感器」页登记。
      </p>

      <ol className="cali-steps">
        <li className="cali-step">
          <div className="cali-step-title">① 选臂和末端配置</div>
          <ArmPicker arms={arms} value={arm} onChange={setArm} disabled={disabled} label="机械臂" />
          <div className="payload-choices" role="radiogroup" aria-label="末端配置">
            {(payload?.toolConfigs ?? []).map((c) => (
              <label key={c.id} className={toolConfig === c.id ? "payload-choice active" : "payload-choice"}>
                <input type="radio" checked={toolConfig === c.id} disabled={disabled} onChange={() => setToolConfig(c.id)} />
                {c.label}
              </label>
            ))}
          </div>
          <div className="payload-row">
            <label className="cali-field">
              姿态集
              <select value={poses} disabled={disabled} onChange={(e) => setPoses(e.target.value)}>
                {(payload?.poseSets ?? []).map((p) => (
                  <option key={p.id} value={p.id}>
                    {p.label}
                  </option>
                ))}
              </select>
            </label>
            <label className="cali-check">
              <input type="checkbox" checked={sixD} disabled={disabled} onChange={(e) => setSixD(e.target.checked)} />
              同时记录夹爪 BOX 6D{armRow ? `（BOX ${armRow.gripperBox ?? "未登记"}）` : ""}
            </label>
          </div>
          {sixD && poses === "payload" && (
            <p className="cali-warn">负载姿态里工具始终朝下（倾斜 ≤45°），只够拟合一个面；要拟合 6D 请选含「6D」的姿态集。</p>
          )}
        </li>

        <li className="cali-step">
          <div className="cali-step-title">② 碰撞检查（含桌面）</div>
          <p className="cali-muted">
            在 MuJoCo 里沿整条路径检查工具和机械臂自身的间隙，并把桌面当成一个平面：基座坐标系里 x 超过桌沿的部分，
            都要高于桌面加余量。桌沿和桌面高度按 11-06 图像估计偏保守取值；卷尺量过后填实测值。
            桌上的物体、线缆、另一条臂都<b>没有</b>建模，要目测。
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
                <input
                  value={table.marginMm}
                  disabled={disabled}
                  onChange={(e) => setTable({ ...table, marginMm: e.target.value })}
                />
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
          <div className="cali-step-title">③ 采集（机械臂运动）</div>
          <div className="payload-row">
            <label className="cali-field payload-note">
              备注
              <input value={note} disabled={disabled} placeholder="例如：右臂，11-09 拆相机后" onChange={(e) => setNote(e.target.value)} />
            </label>
            <button className="cali-btn-primary" disabled={disabled || blocker !== ""} onClick={() => setConfirming(true)}>
              开始采集
            </button>
            {blocker && !active ? <span className="cali-muted">{blocker}</span> : null}
          </div>
          {jobMine && job.state ? (
            <PayloadJobView job={job} stopping={running === "stop"} onStop={() => void act("stop", "stop")} />
          ) : active ? (
            <p className="cali-guard">另一个任务在进行：{job.label}</p>
          ) : null}
        </li>

        <li className="cali-step">
          <div className="cali-step-title">④ 生成 Desk 配置</div>
          <p className="cali-muted">
            勾选同一末端配置的完整采集（同配置的多条会合并），按「不带关节零偏」拟合，这正是机器人自己的重力补偿要的值（10-12 闭环验证过）。
            导入 Desk 后再采一次，日志里 Desk 的 O_F_ext 平均 Fz 应在 ±0.2 N 以内。
          </p>
          {runs.length === 0 ? (
            <p className="cali-muted">还没有采集（{payload?.runsRoot}）。</p>
          ) : (
            <table className="metric-table payload-runs">
              <thead>
                <tr>
                  <th />
                  <th>采集</th>
                  <th>臂</th>
                  <th>末端</th>
                  <th>姿态</th>
                  <th>6D</th>
                  <th>备注</th>
                </tr>
              </thead>
              <tbody>
                {runs.map((r) => {
                  const usable = r.complete && r.twoWay;
                  const verdict = r.sixD ? sixDVerdict(r.sixD.fit) : null;
                  return (
                    <tr key={r.name} className={usable ? "" : "payload-run-off"}>
                      <td>
                        <input
                          type="checkbox"
                          disabled={!usable || disabled}
                          title={usable ? "" : r.complete ? "单向采集（10-12 早期），有摩擦迟滞，不能用" : "没采完"}
                          checked={picked.includes(r.name)}
                          onChange={(e) =>
                            setPicked((names) => (e.target.checked ? [...names, r.name] : names.filter((n) => n !== r.name)))
                          }
                        />
                      </td>
                      <td>
                        <code>{r.name.slice(0, 15)}</code>
                      </td>
                      <td>{r.arm || r.robotIp}</td>
                      <td>{TOOL_CONFIG_SHORT[r.toolConfig] ?? r.toolConfig}</td>
                      <td>
                        {r.nPoses} {r.complete ? "" : "（未完成）"}
                      </td>
                      <td
                        className={verdict ? `payload-verdict-${verdict.tone}` : "cali-muted"}
                        title={verdict ? [verdict.text, ...sixDFitNotes(r.sixD?.fit).map((n) => n.text)].join("\n") : ""}
                      >
                        {verdict ? (r.sixD?.fit?.meetsTarget ? "✓" : verdict.tone === "bad" ? "✕" : "…") : "—"}
                      </td>
                      <td className="cali-muted">{r.note}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
          {pickedConfigs.length > 0 && (
            <div className="payload-row">
              {pickedConfigs.map((c) => (
                <label key={c} className="cali-field payload-note">
                  Desk 名称（{TOOL_CONFIG_SHORT[c] ?? c}）
                  <input
                    value={deskNames[c] ?? `Corenetic ${TOOL_CONFIG_SHORT[c] ?? c} ${today}`}
                    disabled={disabled}
                    onChange={(e) => setDeskNames({ ...deskNames, [c]: e.target.value })}
                  />
                </label>
              ))}
              <label className="cali-field">
                Desk TCP
                <select value={tcpSource} disabled={disabled} onChange={(e) => setTcpSource(e.target.value as TcpSource["kind"])}>
                  <option value="estimate">
                    {TCP_SOURCE_LABELS.estimate} z {((payload?.tcpEstimateZM ?? 0.19016) * 1000).toFixed(2)} mm
                  </option>
                  <option value="pivot" disabled={!pivotFits.length}>
                    {TCP_SOURCE_LABELS.pivot}
                    {pivotFits.length ? "" : "（还没有）"}
                  </option>
                  <option value="manual">{TCP_SOURCE_LABELS.manual}</option>
                  <option value="template">
                    {TCP_SOURCE_LABELS.template} z {((payload?.tcpTemplateZM ?? 0.17199) * 1000).toFixed(2)} mm
                  </option>
                </select>
              </label>
              {tcpSource === "pivot" && (
                <label className="cali-field">
                  pivot 拟合
                  <select value={tcpPivot} disabled={disabled} onChange={(e) => setTcpPivot(e.target.value)}>
                    <option value="">选一条</option>
                    {pivotFits.map((f) => (
                      <option key={f.name} value={f.name}>
                        {f.name.replace(/\.fit\.json$|\.json$/, "")} · {f.robotIp || "?"} · s {fmtMm(f.sEeMm.map((v) => v / 1000))}
                        {f.residualRmsMm != null ? ` · rms ${f.residualRmsMm} mm` : ""}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {tcpSource === "manual" && (
                <label className="cali-field">
                  TCP z (m)
                  <input value={tcpZ} disabled={disabled} placeholder="0.19016" onChange={(e) => setTcpZ(e.target.value)} />
                </label>
              )}
              <button className="cali-btn-primary" disabled={disabled} onClick={() => void makeDesk()}>
                {running === "desk" ? "拟合中…" : "生成 Desk 配置"}
              </button>
            </div>
          )}
          {pickedConfigs.length > 0 && (
            <p className="cali-muted">
              Desk TCP 取 nov4 球窝中心（和相机标签的 TCP 是同一个点：夹住插件时两指尖端面的中心）。默认值是估计：手持 rig 的 pivot 实测
              167.91 mm（传感器系）+ URDF 法兰到传感器 22.25 mm = 190.16 mm；没在机械臂上测过，手持上有约 12 mm 的堆叠偏差没查清，
              所以可能差到 12 mm。在机械臂上做一次 pivot（FR3 页签「FR3 球窝 pivot」，夹 nov4 插件）后选「FR3 pivot 实测」。旋转沿用模板（yaw −45°）；
              负载和 TCP 互不影响。我们的录制和回放不读 Desk 的 TCP。
            </p>
          )}
          {desk && (
            <div className="payload-desks">
              {(payload?.desks.length ?? 0) > 1 && (
                <label className="cali-field">
                  结果
                  <select value={desk.name} onChange={(e) => setShownDesk(e.target.value)}>
                    {payload?.desks.map((d) => (
                      <option key={d.name} value={d.name}>
                        {d.name}
                      </option>
                    ))}
                  </select>
                </label>
              )}
              {desk.tcpSource && (
                <p className="cali-muted">
                  TCP：{TCP_SOURCE_LABELS[desk.tcpSource.kind] ?? desk.tcpSource.kind}（{desk.tcpSource.text}）
                </p>
              )}
              {desk.configs.map((c) => (
                <div key={c.toolConfig} className="cali-result-box">
                  <div className="cali-result-box-head">
                    <strong>{c.deskName}</strong>
                    <span className="cali-muted">
                      {TOOL_CONFIG_SHORT[c.toolConfig] ?? c.toolConfig} · {c.nPoses} 个姿态 · {c.runs.join(", ")}
                    </span>
                    <a className="cali-mini-btn" href={api.payloadDeskFileUrl(desk.name, c.deskJson)} download>
                      下载 Desk JSON
                    </a>
                  </div>
                  <div className="payload-figures">
                    <div>
                      <span>质量</span>
                      <strong>{fmtG(c.massKg)}</strong>
                      <small>±{c.massStdG} g（噪声）</small>
                    </div>
                    <div>
                      <span>质心（法兰系）</span>
                      <strong>{fmtMm(c.comFlangeM)}</strong>
                    </div>
                    <div>
                      <span>导入后预计 O_F_ext 平均 F</span>
                      <strong>{fmtN(c.predictedForceMeanN)} N</strong>
                      <small>最大 |F| {c.predictedForceMaxN} N</small>
                    </div>
                    <div>
                      <span>CAD 参考</span>
                      <strong>{fmtG(c.cad.mass_kg)}</strong>
                      <small>{fmtMm(c.cad.com_flange_m)}</small>
                    </div>
                  </div>
                  {c.deskAtStart.map((d) => (
                    <div key={d.run} className="cali-muted">
                      采集时 Desk 设的是 {d.m_ee.toFixed(4)} kg：实测平均 F {fmtN(d.measured_force_mean_n)} N（最大 {d.measured_force_max_n} N），
                      复现误差 {d.reproduction_rms}
                    </div>
                  ))}
                  <details>
                    <summary className="cali-muted">带零偏的拟合（物理负载的不确定度参考）和 fr3_arms.json 条目</summary>
                    {Object.entries(c.references).map(([name, r]) => (
                      <div key={name} className="cali-muted">
                        {name}：{fmtG(r.mass_kg)} {fmtMm(r.com_flange_m)}
                      </div>
                    ))}
                    <p className="cali-muted">
                      导入 Desk 并闭环通过后，把下面这段手动填进 <code>tools/fr3/fr3_arms.json</code> 对应臂的 <code>payload</code>（页面不改配置）：
                    </p>
                    <pre className="payload-log">{fr3ArmsPayloadSnippet(c, desk.name)}</pre>
                  </details>
                </div>
              ))}
            </div>
          )}
        </li>
      </ol>

      {error && <p className="cali-warn">{error}</p>}
      {output && (
        <details>
          <summary className="cali-muted">输出</summary>
          <pre className="payload-log">{output}</pre>
        </details>
      )}

      {confirming && (
        <Modal
          title="开始负载辨识采集"
          onClose={() => setConfirming(false)}
          footer={
            <>
              <button onClick={() => setConfirming(false)}>取消</button>
              <button className="cali-btn-primary" disabled={!allTicked} onClick={() => void startCollect()}>
                开始（机械臂会动）
              </button>
            </>
          }
        >
          <p>
            {arm} 臂（{armRow?.robotIp}），{TOOL_CONFIG_SHORT[toolConfig]}，
            {payload?.poseSets.find((p) => p.id === poses)?.label}
            {sixD ? `，同时记录 BOX ${armRow?.gripperBox} 的 6D` : ""}。速度 0.1，每个姿态从两侧各到达一次。
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
