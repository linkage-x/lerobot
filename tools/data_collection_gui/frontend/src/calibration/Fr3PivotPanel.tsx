// FR3 FK pivot and the ball check of the world -> fr3_base edge.
//
// A ball fixed to the table, pivoted about twice: by the FR3 in teaching mode
// (fr3_tcp_pivot.py collect + fit, started here) and by the handheld UMI in an
// ordinary recording. The FR3 half settles corenetic_socket_ee on its own; the
// two together score every edge -- live, single-tag, laser -- at the ball,
// without the laser or any TCP definition. See fr3Pivot.ts for the wording.
import { useCallback, useEffect, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { BallCheckReport, Fr3PivotResponse } from "../types";
import { Modal } from "./ConfirmModal";
import { Fr3PivotAutoSteps } from "./Fr3PivotAutoSteps";
import { GripperTeleopBox } from "./GripperTeleopBox";
import {
  ballCheckVerdict,
  ballEdgeSaveBlocker,
  ballEdgeSummary,
  biasConsistency,
  biasLabel,
  contactText,
  edgeDisplayName,
  episodeKey,
  fmtVec,
  handheldSharedSummary,
  isAnchored,
  isJobActive,
  jointFitLines,
  jobStatusText,
  pivotFitWarnings,
  positionDraftWarnings,
  rankedEdges,
  socketFrameVerdict,
  type PositionDraft,
} from "./fr3Pivot";
import { fmtNum } from "./fr3BaseEdge";

const POLL_MS = 1000;
const LOG_TAIL = 14;

export function Fr3PivotPanel({ api, busy }: { api: DataCollectionGuiApi; busy: boolean }) {
  const [payload, setPayload] = useState<Fr3PivotResponse | null>(null);
  const [running, setRunning] = useState("");
  const [error, setError] = useState("");
  const [output, setOutput] = useState("");
  const [robotIp, setRobotIp] = useState("");
  const [durationS, setDurationS] = useState("");
  const [note, setNote] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [mode, setMode] = useState<"guided" | "auto" | "teach">("guided");
  const [drafts, setDrafts] = useState<PositionDraft[]>([{ pivot: "", episodeKey: "" }]);
  const [checkName, setCheckName] = useState("");
  const [picked, setPicked] = useState<string[]>([]);
  const [perRunCentre, setPerRunCentre] = useState(false);
  const [edgeSaved, setEdgeSaved] = useState("");

  const refresh = useCallback(async () => {
    const result = await api.fetchFr3Pivot();
    if (result) setPayload(result);
  }, [api]);

  useEffect(() => {
    void refresh();
  }, [refresh]);

  const job = payload?.job ?? null;
  const active = isJobActive(job);
  // While collecting or fitting the job lives on the gateway; follow it.
  useEffect(() => {
    if (!active) return;
    const timer = window.setInterval(() => void refresh(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [active, refresh]);

  const act = async (label: string, action: () => Promise<Fr3PivotResponse | null>) => {
    setRunning(label);
    setError("");
    setOutput("");
    const result = await action();
    setRunning("");
    if (!result) {
      setError("网关没有响应");
      return;
    }
    if (result.runs) setPayload(result);
    if (!result.ok) setError(result.error || "操作失败");
    if (result.output) setOutput(result.output);
    if (result.checked) setCheckName(result.checked);
    setEdgeSaved(result.edgeSaved ?? "");
  };

  const runs = payload?.runs ?? [];
  const fitted = runs.filter((r) => r.fk);
  const joints = payload?.joints ?? [];
  const pickedRuns = picked.filter((name) => runs.some((r) => r.name === name));
  const togglePicked = (name: string, on: boolean) =>
    setPicked((names) => (on ? [...names.filter((n) => n !== name), name] : names.filter((n) => n !== name)));
  const episodes = payload?.episodes ?? [];
  const ip = robotIp || payload?.robotIp || "";
  const duration = Number(durationS || payload?.defaultDurationS || 45);
  const [lo, hi] = payload?.durationRangeS ?? [15, 180];
  const disabled = busy || running !== "";
  const draftWarnings = positionDraftWarnings(drafts, runs, episodes);
  const completeDrafts = drafts.filter((d) => d.pivot && d.episodeKey);
  const checks = payload?.checks ?? [];
  const shown: BallCheckReport | null = checks.find((c) => c.name === checkName) ?? checks[0] ?? null;

  const patchDraft = (index: number, patch: Partial<PositionDraft>) =>
    setDrafts((rows) => rows.map((row, i) => (i === index ? { ...row, ...patch } : row)));

  const submitCheck = () => {
    const positions = completeDrafts.map((d) => {
      const ep = episodes.find((e) => episodeKey(e) === d.episodeKey)!;
      return { pivot: d.pivot, dataset: ep.dataset, episode: ep.episode };
    });
    void act("check", () => api.runFr3BallCheck(positions));
  };

  return (
    <section className="panel">
      <div className="cali-result-head">
        <h3>FR3 球窝 pivot · 同球检验注册边</h3>
        <div className="cali-op-actions">
          <span className="cali-preview-badge">不需要激光</span>
          <button className="cali-mini-btn" disabled={disabled} onClick={() => act("refresh", () => api.fetchFr3Pivot())}>
            刷新
          </button>
        </div>
      </div>

      <p className="cali-muted">
        <b>作用：</b>桌上固定一个球，FR3 和手持 UMI 先后套着插件 v2 绕它转。FR3 那次只靠关节角和 URDF，
        给出球窝中心在末端系里的位置（检验 replay 驱动的 <code>corenetic_socket_ee</code> 是否就是球窝），以及球心在 fr3_base 里的位置；
        手持那次是一段普通录制，相机标签给出球心在世界系里的位置。每条注册边把前者映射到世界系，和后者的差就是这条边<b>在球那里</b>的误差。
        两边都各自拟合杠杆臂，所以和 TCP 定义、跟踪仪站位、激光都无关。
      </p>

      <GripperTeleopBox api={api} busy={busy} />

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>① FR3 pivot 采集</b>
          <span className="cali-muted">{jobStatusText(job)}</span>
        </div>
        <div className="cali-op-actions">
          <button className={mode === "guided" ? "cali-btn-primary" : "cali-mini-btn"} disabled={active} onClick={() => setMode("guided")}>
            引导（推荐）
          </button>
          <button className={mode === "auto" ? "cali-btn-primary" : "cali-mini-btn"} disabled={active} onClick={() => setMode("auto")}>
            自动规划
          </button>
          <button className={mode === "teach" ? "cali-btn-primary" : "cali-mini-btn"} disabled={active} onClick={() => setMode("teach")}>
            示教（手扶）
          </button>
          <label className="cali-muted">
            FR3 IP <input value={ip} disabled={disabled || active} onChange={(e) => setRobotIp(e.target.value)} size={14} />
          </label>
          <label className="cali-muted">
            备注 <input value={note} disabled={disabled || active} onChange={(e) => setNote(e.target.value)} placeholder="球 A" size={12} />
          </label>
          {active && job?.mode === "guided" ? (
            <button
              className="cali-btn-primary"
              disabled={running !== "" || job?.state !== "collecting" || job?.finishRequested}
              onClick={() => act("finish", () => api.finishFr3Pivot())}
            >
              完成（保存）
            </button>
          ) : null}
          {active ? (
            <button className="cali-mini-btn" disabled={running !== "" || job?.state !== "collecting"} onClick={() => act("stop", () => api.stopFr3Pivot())}>
              停止（不保存）
            </button>
          ) : null}
        </div>
        {mode !== "teach" ? (
          <Fr3PivotAutoSteps
            api={api}
            payload={payload}
            ip={ip}
            note={note}
            disabled={disabled}
            active={active}
            running={running}
            act={act}
            variant={mode}
          />
        ) : (
          <>
            <p className="cali-muted">
              准备：球固定牢（FR3 够得到、相机也看得到的地方）；FR3 夹爪夹住插件 v2（用上面 ⓪ 夹），与回放时一致；Desk 负载保持带 SMR 的配置；
              球窝坐到球上。开始后机械臂进入<b>示教模式（重力补偿，会变软）</b>，用手扶着：两个方向各倾斜约 ±30°、再绕工具轴转几圈，
              全程球窝不离开球。时间到后自动退出示教模式并拟合。
            </p>
            <div className="cali-op-actions">
              <label className="cali-muted">
                时长（s）
                <input
                  type="number"
                  min={lo}
                  max={hi}
                  value={durationS || String(payload?.defaultDurationS ?? 45)}
                  disabled={disabled || active}
                  onChange={(e) => setDurationS(e.target.value)}
                  style={{ width: "5em" }}
                />
              </label>
              {active ? null : (
                <button className="cali-btn-primary" disabled={disabled || !ip} onClick={() => setConfirming(true)}>
                  开始采集
                </button>
              )}
            </div>
          </>
        )}
        {job?.log?.length ? <pre className="marker-tcp-example">{job.log.slice(-LOG_TAIL).join("\n")}</pre> : null}
      </div>

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>② FR3 pivot 结果</b>
          <span className="cali-muted">{payload?.runsRoot ?? "outputs/fr3_tcp_pivot"}</span>
        </div>
        {runs.length ? (
          <table className="metric-table">
            <thead>
              <tr>
                <th>采集</th>
                <th>球窝中心 @corenetic_gripper_ee（mm）</th>
                <th>corenetic_socket_ee 对不对</th>
                <th>球心 @fr3_base（mm）</th>
                <th>残差 rms · 总转角 · 可用姿态</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {runs.map((run) => {
                const verdict = socketFrameVerdict(run);
                const contact = contactText(run.commanded);
                return (
                  <tr key={run.name}>
                    <td>
                      <label>
                        <input
                          type="checkbox"
                          checked={pickedRuns.includes(run.name)}
                          disabled={!run.fk}
                          onChange={(e) => togglePicked(run.name, e.target.checked)}
                          title="选进合并拟合"
                        />{" "}
                        {run.note || "—"}
                      </label>
                      <div className="cali-muted">{run.name}</div>
                      {run.commanded ? (
                        <div className="cali-muted">
                          {run.commanded.kind === "guided" ? "引导" : "自动"} · {biasLabel(run.commanded.bias_mm)}
                          {isAnchored(run.commanded.anchor_mm) ? ` · 弹簧挂在 ${fmtVec(run.commanded.anchor_mm)} mm` : ""}
                        </div>
                      ) : null}
                      {contact ? <div className={contact.ok ? "cali-muted" : "cali-warn"}>{contact.text}</div> : null}
                    </td>
                    <td>
                      {fmtVec(run.fk?.s_ee_mm, 2)}
                      {run.fk ? <div className="cali-muted">± {fmtVec(run.fk.s_sigma_mm, 2)}</div> : null}
                    </td>
                    <td className={verdict.ok === false ? "cali-warn" : undefined}>{verdict.text}</td>
                    <td>
                      {fmtVec(run.fk?.c_base_mm, 1)}
                      {run.fk?.c_sigma_mm ? <div className="cali-muted">± {fmtVec(run.fk.c_sigma_mm, 2)}</div> : null}
                    </td>
                    <td>
                      {run.fk
                        ? `${fmtNum(run.fk.residual_rms_mm, 2, " mm")} · ${fmtNum(run.fk.span_deg, 0, "°")} · ${run.fk.n_used}/${run.fk.n_total}`
                        : "—"}
                      {pivotFitWarnings(run.fk).map((w) => (
                        <div className="cali-warn" key={w}>
                          {w}
                        </div>
                      ))}
                    </td>
                    <td>
                      <button
                        className="cali-mini-btn"
                        disabled={disabled || active}
                        onClick={() => act(`fit:${run.name}`, () => api.refitFr3Pivot(run.name))}
                      >
                        {running === `fit:${run.name}` ? "拟合中…" : "重新拟合"}
                      </button>
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        ) : (
          <p className="cali-muted">还没有 FR3 pivot 采集。</p>
        )}
        <p className="cali-muted">
          <b>合并拟合：</b>勾选几次采集，一起拟合<b>一个</b>球窝中心。单次只在一个朝向附近转 ±15–30°，工具轴方向的分量定不住（10-14：单次 2.6–6.7 mm）；
          每次从不同朝向伸到球上（绕竖直轴转开 60–90°），合在一起就能定住。工具轴方向靠绕竖直轴摆（yaw）和俯仰，竖直方向靠滚转（roll）和俯仰。
          球没动过就共用一个球心（最强的约束）；动过就勾「每次一个球心」。
        </p>
        <div className="cali-op-actions">
          <label className="cali-muted">
            <input type="checkbox" checked={perRunCentre} onChange={(e) => setPerRunCentre(e.target.checked)} /> 每次一个球心（球动过）
          </label>
          <button
            className="cali-btn-primary"
            disabled={disabled || active || pickedRuns.length < 2}
            onClick={() => act("joint", () => api.fitJointFr3Pivot(pickedRuns, perRunCentre))}
          >
            {running === "joint" ? "拟合中…" : `合并拟合所选（${pickedRuns.length}）`}
          </button>
          {pickedRuns.length ? (
            <button className="cali-mini-btn" disabled={disabled} onClick={() => setPicked([])}>
              清空选择
            </button>
          ) : null}
        </div>
        {joints.length ? (
          <table className="metric-table">
            <thead>
              <tr>
                <th>合并拟合</th>
                <th>球窝中心 @corenetic_socket_ee（mm）</th>
                <th>@corenetic_gripper_ee（mm）</th>
                <th>判断</th>
              </tr>
            </thead>
            <tbody>
              {joints.slice(0, 3).map((j) => (
                <tr key={j.name}>
                  <td>
                    {j.name}
                    <div className="cali-muted">{j.runs.map((r) => r.slice(9, 15)).join(" · ")}</div>
                  </td>
                  <td>
                    {fmtVec(j.socketFrame?.s_ee_mm, 2)}
                    {j.socketFrame?.s_sigma_mm ? <div className="cali-muted">± {fmtVec(j.socketFrame.s_sigma_mm, 2)}</div> : null}
                  </td>
                  <td>{fmtVec(j.fk?.s_ee_mm, 2)}</td>
                  <td>
                    {jointFitLines(j).map((line) => (
                      <div className={line.ok ? "cali-muted" : "cali-warn"} key={line.text}>
                        {line.text}
                      </div>
                    ))}
                    {j.warnings.map((w) => (
                      <div className="cali-warn" key={w}>
                        {w}
                      </div>
                    ))}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : null}
        {biasConsistency(runs).map((v) => (
          <p className={v.seated === true ? undefined : "cali-warn"} key={v.centre}>
            <b>偏置对照（球心 {v.centre}）：</b>
            {v.text}
          </p>
        ))}
      </div>

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>③ 手持同球 · 检验注册边</b>
          <span className="cali-muted">现役边和全部候选边一起算，不改任何东西</span>
        </div>
        <p className="cali-muted">
          <b>球不要动</b>，把 FR3 移开；手持 UMI 夹插件 v2 坐到同一个球上，在录制页正常录一段（约 30 s，先坐稳再开始、转完再停），
          动作同上。一个球位置只能看出该处的平移误差；换 2–3 个相隔约 30 cm 的位置各做一遍，旋转也能看出来（1° 在 30 cm 处约 5 mm）。
          EE 轨迹没生成的话检验时会自动生成。FR3 一侧可以选同一个球上的合并拟合（共用球心），球心比单次稳一倍左右。
        </p>
        <table className="metric-table">
          <thead>
            <tr>
              <th>位置</th>
              <th>FR3 pivot</th>
              <th>手持 episode</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {drafts.map((d, i) => (
              <tr key={i}>
                <td>{i + 1}</td>
                <td>
                  <select value={d.pivot} disabled={disabled} onChange={(e) => patchDraft(i, { pivot: e.target.value })}>
                    <option value="">—</option>
                    {joints.some((j) => j.sharedCentre) ? (
                      <optgroup label="合并拟合（共用球心，更稳）">
                        {joints
                          .filter((j) => j.sharedCentre)
                          .map((j) => (
                            <option key={j.name} value={j.name}>
                              {j.runs.length} 次合并 · {j.name}
                            </option>
                          ))}
                      </optgroup>
                    ) : null}
                    <optgroup label="单次">
                      {fitted.map((r) => (
                        <option key={r.name} value={r.name}>
                          {r.note ? `${r.note} · ` : ""}
                          {r.name}
                        </option>
                      ))}
                    </optgroup>
                  </select>
                </td>
                <td>
                  <select value={d.episodeKey} disabled={disabled} onChange={(e) => patchDraft(i, { episodeKey: e.target.value })}>
                    <option value="">—</option>
                    {episodes.map((ep) => (
                      <option key={episodeKey(ep)} value={episodeKey(ep)}>
                        {ep.datasetName} · ep{ep.episode} · {new Date(ep.recordedUnixS * 1000).toLocaleString()}
                        {ep.world ? "" : " · 无 world 戳"}
                      </option>
                    ))}
                  </select>
                </td>
                <td>
                  <button
                    className="cali-mini-btn"
                    disabled={disabled || drafts.length === 1}
                    onClick={() => setDrafts((rows) => rows.filter((_, j) => j !== i))}
                  >
                    删除
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
        <div className="cali-op-actions">
          <button className="cali-mini-btn" disabled={disabled} onClick={() => setDrafts((rows) => [...rows, { pivot: "", episodeKey: "" }])}>
            添加位置
          </button>
          <button className="cali-btn-primary" disabled={disabled || !completeDrafts.length} onClick={submitCheck}>
            {running === "check" ? "检验中…" : "检验"}
          </button>
        </div>
        {draftWarnings.map((w) => (
          <p className="cali-warn" key={w}>
            {w}
          </p>
        ))}
      </div>

      {shown ? (
        <BallCheckResult
          report={shown}
          checks={checks}
          onPick={setCheckName}
          disabled={disabled}
          saving={running === "ballEdge"}
          onSaveEdge={() => void act("ballEdge", () => api.saveFr3BallEdge(shown.name))}
        />
      ) : null}
      {edgeSaved ? (
        <p>
          已存为候选边 <code>{edgeSaved}</code>：到「world → fr3_base 注册边」面板点刷新，和现役边比较挪动量后再换边。
        </p>
      ) : null}

      {error ? <pre className="cali-warn">{error}</pre> : null}
      {output ? <pre className="marker-tcp-example">{output}</pre> : null}

      {confirming ? (
        <Modal
          title="FR3 进入示教模式"
          onClose={() => setConfirming(false)}
          footer={
            <>
              <button onClick={() => setConfirming(false)}>取消</button>
              <button
                className="cali-btn-primary"
                onClick={() => {
                  setConfirming(false);
                  void act("collect", () => api.startFr3Pivot(ip, duration, note));
                }}
              >
                我扶着机械臂，开始
              </button>
            </>
          }
        >
          <p>
            FR3（{ip}）会进入<b>重力补偿示教模式 {duration.toFixed(0)} s</b>：机械臂变软，要有人<b>一直扶着</b>。
          </p>
          <p className="cali-muted">
            确认球窝已经坐在球上、周围没有障碍、急停在手边。中途可以点「停止（不保存）」，或按急停。
          </p>
        </Modal>
      ) : null}
    </section>
  );
}

function BallCheckResult({
  report,
  checks,
  onPick,
  disabled,
  saving,
  onSaveEdge,
}: {
  report: BallCheckReport;
  checks: BallCheckReport[];
  onPick: (name: string) => void;
  disabled: boolean;
  saving: boolean;
  onSaveEdge: () => void;
}) {
  const ranked = rankedEdges(report);
  const unscored = report.edges.filter((e) => e.error);
  return (
    <div className="cali-result-box">
      <div className="cali-result-box-head">
        <b>④ 检验结果</b>
        <select value={report.name} onChange={(e) => onPick(e.target.value)}>
          {checks.map((c) => (
            <option key={c.name} value={c.name}>
              {c.name}
            </option>
          ))}
        </select>
      </div>
      <p>{ballCheckVerdict(report)}</p>
      {ranked.length ? (
        <table className="metric-table">
          <thead>
            <tr>
              <th>边</th>
              {report.positions.map((_, i) => (
                <th key={i}>位置 {i + 1}：T·c − b（mm，世界系）</th>
              ))}
              <th>rms</th>
              <th>max</th>
            </tr>
          </thead>
          <tbody>
            {ranked.map((edge) => (
              <tr key={edge.name}>
                <td>{edgeDisplayName(edge.name, edge.from_balls)}</td>
                {report.positions.map((_, i) => {
                  const p = edge.positions?.find((q) => q.position === i);
                  return <td key={i}>{p ? `${fmtVec(p.error_mm)} |${p.norm_mm.toFixed(1)}|` : "—"}</td>;
                })}
                <td>{fmtNum(edge.rms_mm, 1)}</td>
                <td>{fmtNum(edge.max_mm, 1)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {unscored.map((e) => (
        <p className="cali-muted" key={e.name}>
          {edgeDisplayName(e.name)}：{e.error}
        </p>
      ))}
      <table className="metric-table">
        <thead>
          <tr>
            <th>位置</th>
            <th>FR3 pivot · 手持 episode</th>
            <th>手持：球心相对生产 TCP（mm）</th>
            <th>手持：残差 p95 · 偏转 · 在球上</th>
            <th>手持：共用杠杆臂后球心移动（mm）</th>
            <th>问题</th>
          </tr>
        </thead>
        <tbody>
          {report.positions.map((p, i) => (
            <tr key={i}>
              <td>{i + 1}</td>
              <td>
                {p.fr3_pivot}
                <div className="cali-muted">
                  {p.dataset} ep{p.episode}
                  {p.cube ? ` · ${p.cube}` : ""}
                </div>
              </td>
              <td title="marker→TCP 常量在这个 cube 上的误差；TCP 定义为球窝中心时应接近 0">
                {fmtVec(p.handheld?.s_mm)}
              </td>
              <td>
                {p.handheld
                  ? `${fmtNum(p.handheld.residual_p95_mm, 2, " mm")} · ${fmtNum(p.handheld.span_deg, 0, "°")} · ${
                      p.handheld.on_ball_fraction == null ? "—" : `${(p.handheld.on_ball_fraction * 100).toFixed(0)}%`
                    }`
                  : "—"}
              </td>
              <td title="各位置共用一个杠杆臂拟合出的球心，减去本位置单独拟合的球心；检验和球位边用的是前者">
                {p.handheld_shared ? `${fmtVec(p.handheld_shared.shift_mm)} |${Math.hypot(...p.handheld_shared.shift_mm).toFixed(1)}|` : "—"}
              </td>
              <td>
                {[...p.reasons, ...p.warnings].map((w) => (
                  <div className={p.reasons.includes(w) ? "cali-warn" : "cali-muted"} key={w}>
                    {w}
                  </div>
                ))}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
      {handheldSharedSummary(report).map((line) => (
        <p className="cali-muted" key={line}>
          {line}
        </p>
      ))}
      {Object.entries(report.ball_edges).map(([world, be]) => {
        const blocker = ballEdgeSaveBlocker(report, world);
        return (
          <div className="cali-op-actions" key={world}>
            <p className={be.verdict === "ok" ? "" : "cali-muted"}>
              只用这些球拟合出的边（{world}）：{ballEdgeSummary(be)}
            </p>
            <button
              className="cali-mini-btn"
              disabled={disabled || blocker !== ""}
              title={blocker || "写成 world → fr3_base 的候选报告；换边在注册边面板里确认"}
              onClick={onSaveEdge}
            >
              {saving ? "保存中…" : "存为候选边"}
            </button>
          </div>
        );
      })}
    </div>
  );
}
