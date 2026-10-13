// The world -> fr3_base registration edge (register_fr3_base.py).
//
// Every fr3_base trajectory -- MuJoCo preview, IK QC, real replay, export --
// goes through this one edge, so it gets its own panel next to the world frame
// it leaves from. Capturing a P0 single-tag run stays at the robot (the arm is
// hand-guided in zero-stiffness mode and the capture tool opens its own
// recorder); the panel solves runs, compares candidates and swaps the edge.
//
// Swapping is separate and confirmed, and a passing verdict alone does not make
// a candidate better than the live edge: see fr3BaseEdge.ts.
import { useEffect, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { Fr3BaseEdgeResponse } from "../types";
import { Modal } from "./ConfirmModal";
import {
  applyConfirmation,
  candidateSpread,
  fmtNum,
  liveEdgeSummary,
  liveEdgeWarnings,
  moveSummary,
  runLabel,
  verdictLabel,
} from "./fr3BaseEdge";

type Candidate = Fr3BaseEdgeResponse["candidates"][number];

export function Fr3BaseEdgePanel({ api, busy }: { api: DataCollectionGuiApi; busy: boolean }) {
  const [payload, setPayload] = useState<Fr3BaseEdgeResponse | null>(null);
  const [running, setRunning] = useState("");
  const [error, setError] = useState("");
  const [output, setOutput] = useState("");
  const [confirming, setConfirming] = useState<Candidate | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.fetchFr3BaseEdge().then((result) => {
      if (!cancelled && result) setPayload(result);
    });
    return () => {
      cancelled = true;
    };
  }, [api]);

  const act = async (label: string, action: () => Promise<Fr3BaseEdgeResponse | null>) => {
    setRunning(label);
    setError("");
    setOutput("");
    const result = await action();
    setRunning("");
    if (!result) {
      setError("网关没有响应");
      return;
    }
    // failures still carry the panel state
    if (result.runs) setPayload(result);
    if (!result.ok) setError(result.error || "操作失败");
    if (result.output) setOutput(result.output);
  };

  const current = payload?.current ?? null;
  const warnings = payload ? liveEdgeWarnings(current, payload.referenceWorld) : [];
  const spread = payload ? candidateSpread(payload.candidates) : null;
  const disabled = busy || running !== "";

  return (
    <section className="panel">
      <div className="cali-result-head">
        <h3>FR3 基座注册边 · world → fr3_base</h3>
        <div className="cali-op-actions">
          <span className="cali-preview-badge">单 tag 手眼</span>
          <button className="cali-mini-btn" disabled={disabled} onClick={() => act("refresh", () => api.fetchFr3BaseEdge())}>
            刷新
          </button>
        </div>
      </div>

      <p className="cali-muted">
        <b>作用：</b>相机世界系里的位姿要交给 FR3，必须经过这条边。MuJoCo 预览、IK QC、真机 replay 和导出到 fr3_base 都走它，
        所以它的误差会<b>原样</b>叠加到每一条 replay 上。这条边由 FR3 工具上的 tag36h11 id6 测出：机械臂停在若干个位姿，
        基准世界系的相机<b>固定在原位不动</b>，一起拟合 T_world_base、T_tcp_tag 和 tag 尺度。
      </p>

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>现役边</b>
          <span className="cali-muted">{current ? `写入 ${current.createdUtc}` : ""}</span>
        </div>
        <p>{liveEdgeSummary(current)}</p>
        {current && current.supersedes > 0 ? (
          <p className="cali-muted">它替换过 {current.supersedes} 条旧边（保存在 supersedes 里）。</p>
        ) : null}
        {warnings.map((w) => (
          <p className="cali-warn" key={w}>
            {w}
          </p>
        ))}
      </div>

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>① 采集</b>
          <span className="cali-muted">在机器人旁边做，不在这里</span>
        </div>
        <p className="cali-muted">
          采集时 FR3 处于零刚度示教模式，需要有人手扶着推动；采集工具会自己打开一个 recorder 占用相机，
          所以先在顶栏<b>断开</b> GUI 的录制器。到 Thor 的终端上运行：
        </p>
        <pre className="marker-tcp-example">{payload?.captureCommand ?? "—"}</pre>
        <p className="cali-muted">
          把机械臂推到工作区里不同的位置和朝向，每停稳一处按一次 Enter；至少 20 次，并让每台相机都看到 tag。
          结果写到 <code>{payload?.runsRoot ?? "outputs/calibration/p0_single_tag_camera_calibration"}</code>，
          采完后点上面的「刷新」，新 run 会出现在下表里。
        </p>
      </div>

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>② 解算</b>
          <span className="cali-muted">只写候选报告，不改现役边</span>
        </div>
        {payload && payload.runs.length ? (
          <table className="metric-table">
            <thead>
              <tr>
                <th>P0 run</th>
                <th>捕获（现场 / 导入）</th>
                <th>P0 自解</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {payload.runs.map((run) => (
                <tr key={run.name}>
                  <td>
                    {run.name}
                    {run.isLiveEdgeSource ? <span className="cali-preview-badge">现役边来源</span> : null}
                  </td>
                  <td>
                    {run.captures}（{run.live} / {run.imported}）
                  </td>
                  <td>{run.p0Status || "—"}</td>
                  <td>
                    <button
                      className="cali-mini-btn"
                      disabled={disabled || !run.solvable}
                      title={run.solvable ? "" : "缺 captures.json 或 P0 自解结果"}
                      onClick={() => act(run.name, () => api.solveFr3BaseEdge(run.name))}
                    >
                      {running === run.name ? "解算中…" : "解算"}
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="cali-muted">还没有 P0 单 tag run。</p>
        )}
        <p className="cali-muted">
          解算时用的是基准世界系自己的相机位姿和内参，P0 自解的结果只用来提供初值。每轮约 1–2 分钟。
        </p>
      </div>

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>③ 候选边</b>
          <span className="cali-muted">样本外误差只说明这一轮自身是否一致，还要看它会把现役边挪多远</span>
        </div>
        {payload && payload.candidates.length ? (
          <table className="metric-table">
            <thead>
              <tr>
                <th>run</th>
                <th>判定</th>
                <th>样本外 p95</th>
                <th>tag 尺度</th>
                <th>重投影 RMSE</th>
                <th>相对现役边（该轮工作区）</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {payload.candidates.map((c) => (
                <tr key={c.name} title={c.reasons.join("；")}>
                  <td>
                    {runLabel(c.p0Run)}
                    <div className="cali-muted">{c.generatedUtc}</div>
                  </td>
                  <td>{verdictLabel(c)}</td>
                  <td>{fmtNum(c.quality.outOfSampleP95Mm, 2, " mm")}</td>
                  <td>{fmtNum(c.quality.tagScale, 4)}</td>
                  <td>{fmtNum(c.quality.rmsePx, 2, " px")}</td>
                  <td>{c.sameAsLive ? "—" : moveSummary(c.moveVsLive)}</td>
                  <td>
                    <button
                      className="cali-mini-btn"
                      disabled={disabled || c.applyBlockers.length > 0}
                      title={c.applyBlockers.join("；")}
                      onClick={() => setConfirming(c)}
                    >
                      替换现役边
                    </button>
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        ) : (
          <p className="cali-muted">还没有解算过的候选。</p>
        )}
        {spread ? (
          <p className={spread.mm >= 2 ? "cali-warn" : "cali-muted"}>
            通过判定的各轮（含现役边）在工作区里最多相差 <b>{spread.mm.toFixed(1)} mm</b>（{spread.a} 与 {spread.b}）。
            这就是这种方法的轮间重复性：差距大于 2 mm 时，单看某一轮「通过」不能说明它比现役边更准，
            换边之前先用激光跟踪仪 replay 做独立验证。
          </p>
        ) : null}
      </div>

      {error ? <pre className="cali-warn">{error}</pre> : null}
      {output ? <pre className="marker-tcp-example">{output}</pre> : null}

      {confirming ? (
        <Modal
          title="替换 world → fr3_base 注册边"
          onClose={() => setConfirming(null)}
          footer={
            <>
              <button onClick={() => setConfirming(null)}>取消</button>
              <button
                className="cali-btn-primary"
                onClick={() => {
                  const chosen = confirming;
                  setConfirming(null);
                  void act("apply", () => api.applyFr3BaseEdge(chosen.name));
                }}
              >
                确认替换
              </button>
            </>
          }
        >
          <p>{applyConfirmation(confirming, current)}</p>
        </Modal>
      ) : null}
    </section>
  );
}
