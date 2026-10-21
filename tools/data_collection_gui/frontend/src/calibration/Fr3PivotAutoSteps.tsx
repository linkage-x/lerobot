// FR3 pivot under impedance, both variants of tools/fr3/fr3_pivot_auto.py: capture the centre with
// the socket seated, then either guide (the socket held pressed on the ball, the operator turns it --
// no model of the surroundings needed) or plan + MuJoCo-check + execute. Wording in fr3Pivot.ts.
import { useEffect, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { Fr3PivotResponse } from "../types";
import { Modal } from "./ConfirmModal";
import { BIAS_PRESETS, anchorText, biasLabel, fmtVec, planSummaryText } from "./fr3Pivot";

const CENTRES_SHOWN = 3;
const PLANS_SHOWN = 6;

export function Fr3PivotAutoSteps({
  api,
  payload,
  ip,
  note,
  disabled,
  active,
  running,
  act,
  variant,
}: {
  api: DataCollectionGuiApi;
  payload: Fr3PivotResponse | null;
  ip: string;
  note: string;
  disabled: boolean;
  active: boolean;
  running: string;
  act: (label: string, action: () => Promise<Fr3PivotResponse | null>) => Promise<void>;
  variant: "guided" | "auto";
}) {
  const centres = payload?.centres ?? [];
  const plans = payload?.plans ?? [];
  const [centre, setCentre] = useState("");
  const [bias, setBias] = useState(0);
  const [confirmPlan, setConfirmPlan] = useState("");
  const [guideS, setGuideS] = useState("");
  const [confirmGuide, setConfirmGuide] = useState(false);
  const [anchored, setAnchored] = useState(false);
  const anchor = payload?.anchor;
  const anchorLine = anchorText(anchor);
  const [glo, ghi] = payload?.guideRangeS ?? [20, 180];
  const guideDuration = Number(guideS || payload?.guideDefaultS || 60);

  // A fresh capture is the one to plan about.
  const latest = centres[0]?.name ?? "";
  useEffect(() => {
    setCentre(latest);
  }, [latest]);

  // The offset belongs to one grip and one fit: a re-clamp (the gateway sees the gripper teleop run
  // after the fit) or a newer fit unticks it, so a stale one is never sent by habit.
  const anchorKey = anchor?.available ? anchor.fitName ?? "" : "";
  // ...except that in the planned variant a fit of a run made after this centre was captured (the probe,
  // same grip) is what the full plan is meant to turn about.
  const anchorRun = anchor?.available ? anchor.run ?? "" : "";
  useEffect(() => {
    setAnchored(variant === "auto" && !!anchorRun && !!centre && anchorRun.slice(0, 15) >= centre.slice(0, 15));
  }, [anchorKey, anchorRun, centre, variant]);

  const chosen = centres.find((c) => c.name === centre);
  const toRun = plans.find((p) => p.name === confirmPlan);
  const locked = disabled || active;

  return (
    <>
      <p className="cali-muted">
        <b>摆球：</b>球心离 FR3 底座水平 0.45–0.65 m、比桌面高 6 cm 以上（10-13 那次约 0.8 m，接近够取极限，哪个方向都转不开）；
        夹爪<b>从侧面</b>伸过去——夹爪朝向和「底座→球」方向夹 60–90°，别正对着往外伸（那样 j6 贴着 panda_py 的 215° 墙）。
        <br />
        <b>1 取球心：</b>先用 ⓪ 让夹爪夹好插件；用 Desk 引导（手腕上的引导键）把夹爪大致水平地伸到球上方，球窝朝下坐到球上；
        松手后点「取当前位置为球心」。只读关节角，机械臂不动；会提示离 panda_py 那道墙太近的关节（j6 默认 215°，这个姿态最容易贴墙，
        贴墙就转不开）。取完到开始之前不要再碰机械臂（开始前会核对，动过就拒绝）。
      </p>
      <div className="cali-op-actions">
        <button className="cali-btn-primary" disabled={locked || !ip} onClick={() => act("capture", () => api.captureFr3PivotCentre(ip, note))}>
          {running === "capture" ? "读取中…" : "取当前位置为球心"}
        </button>
      </div>
      {centres.length ? (
        <table className="metric-table">
          <thead>
            <tr>
              <th>球心</th>
              <th>位置 @fr3_base（mm）</th>
              <th>工具轴离水平</th>
              <th>关节限位</th>
            </tr>
          </thead>
          <tbody>
            {centres.slice(0, CENTRES_SHOWN).map((c) => (
              <tr key={c.name}>
                <td>
                  {c.note || "—"}
                  <div className="cali-muted">{c.name}</div>
                </td>
                <td>{fmtVec(c.centreMm)}</td>
                <td>{c.toolFromLevelDeg == null ? "—" : `${c.toolFromLevelDeg.toFixed(0)}°`}</td>
                <td className={c.wallProblems.length ? "cali-warn" : undefined}>{c.wallProblems.join("；") || "OK"}</td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}

      {variant === "guided" ? (
        <>
      <p className="cali-muted">
        <b>2 引导：</b>开始后机器人用一根软弹簧（600 N/m）把球窝<b>往下压在球上</b>（约 2 N），转动完全放开、肘部也不锁：
        你扶着夹爪绕球转——上下俯仰、绕夹爪自身轴滚转各约 ±20°，再绕过球心的竖直轴左右摆约 ±30°；球窝始终朝下，别翻到侧面；
        避开桌子和支架由你目测。转够了点「完成（保存）」，
        或者到时间自动收尾；外力超过 20 N、球窝被拉离超过 15 mm、关节靠近 panda_py 的墙会自动停下（不保存）。日志里每 2 s 报一次最大倾斜和自转范围。
        <b>同一个球心跑「+3 mm」和「−3 mm」两次</b>：两次拟合一致，才证明结果由球决定。
        <br />
        <b>弹簧挂在拟合的球心：</b>默认弹簧拉的是 corenetic_socket_ee 的原点，它离真正的转动中心约 6–7 mm（10-13），
        每转一下弹簧都在横着顶球（20° 约 1–3 N），会把拟合往原点拉。勾上后弹簧改拉上一次引导拟合出的球心，转动不再受力。
        偏移是在某一次夹持下拟合的：之后用过 ⓪ 遥控（比如 BOX 重启后重新夹），这里会提示夹持可能变了，但仍可以勾——
        插件重新夹的定位在 0.2 mm 以内；要稳妥就先跑一轮<b>不勾</b>的引导、拟合完再勾。
      </p>
      <div className="cali-op-actions">
        <label className="cali-muted">
          球心{" "}
          <select value={centre} disabled={locked} onChange={(e) => setCentre(e.target.value)}>
            <option value="">—</option>
            {centres.map((c) => (
              <option key={c.name} value={c.name}>
                {c.note ? `${c.note} · ` : ""}
                {c.name}
              </option>
            ))}
          </select>
        </label>
        <label className="cali-muted">
          偏置{" "}
          <select value={bias} disabled={locked} onChange={(e) => setBias(Number(e.target.value))}>
            {BIAS_PRESETS.map((b, i) => (
              <option key={b.label} value={i}>
                {b.label}
              </option>
            ))}
          </select>
        </label>
        <label className="cali-muted">
          最长（s）
          <input
            type="number"
            min={glo}
            max={ghi}
            value={guideS || String(payload?.guideDefaultS ?? 60)}
            disabled={locked}
            onChange={(e) => setGuideS(e.target.value)}
            style={{ width: "5em" }}
          />
        </label>
        <label className="cali-muted" title={anchorLine.text}>
          <input
            type="checkbox"
            checked={anchored && !!anchor?.available}
            disabled={locked || !anchor?.available}
            onChange={(e) => setAnchored(e.target.checked)}
          />{" "}
          弹簧挂在上次拟合的球心
        </label>
        <button className="cali-btn-primary" disabled={locked || !chosen || !ip} onClick={() => setConfirmGuide(true)}>
          开始引导
        </button>
      </div>
      <p className={anchorLine.ok ? "cali-muted" : "cali-warn"}>{anchorLine.text}</p>
      {anchorLine.warning ? <p className="cali-warn">{anchorLine.warning}</p> : null}
      {chosen && centre !== latest ? <p className="cali-warn">选的不是最新的球心：开始前会核对球窝还在它那里，不在就拒绝</p> : null}

      {confirmGuide && chosen ? (
        <Modal
          title="FR3 引导 pivot"
          onClose={() => setConfirmGuide(false)}
          footer={
            <>
              <button onClick={() => setConfirmGuide(false)}>取消</button>
              <button
                className="cali-btn-primary"
                onClick={() => {
                  setConfirmGuide(false);
                  const useAnchor = anchored && !!anchor?.available;
                  void act("guide", () => api.guideFr3Pivot(centre, BIAS_PRESETS[bias].mm, guideDuration, ip, note, useAnchor));
                }}
              >
                我扶着夹爪，开始
              </button>
            </>
          }
        >
          <p>
            FR3（{ip}）会把球窝<b>往球里压约 2 N</b> 并保持在球上，最长 {guideDuration.toFixed(0)} s，{biasLabel(BIAS_PRESETS[bias].mm)}。
            <b>转动是放开的</b>：一开始就要扶住夹爪，否则手腕会在重力下慢慢转。
          </p>
          <p>
            {anchored && anchor?.available
              ? `弹簧挂在拟合的球心 ${fmtVec(anchor.sMm)} mm（${anchor.run}）。${anchor.warning ?? ""}`
              : "弹簧挂在 corenetic_socket_ee 的原点。"}
          </p>
          <p className="cali-muted">确认球窝坐在球上、夹爪夹紧、急停在手边。</p>
        </Modal>
      ) : null}
        </>
      ) : (
        <>
      <p className="cali-muted">
        <b>2 试探（演算约 1 分钟，执行 1–1.5 分钟）：</b>杯子真正绕着转的点离 corenetic_socket_ee 的原点多远，随夹持而变（10-19 一次 1.4 mm、一次 8.7 mm）。
        规划绕错的点转，转大了球窝要跟着挪十几毫米，力环吃不下。所以先只在起点附近 20° 内排 6 个驻点、3°/s 慢慢转一圈，跑完自动拟合出球心；
        之后的完整规划会自动勾上「绕上次拟合的球心转」。换了夹持（重夹、加装东西）就重新试探。
        <br />
        <b>3 规划 + 演算（约 1–2 分钟）：</b>先在仿真里把球心周围的姿态（工具轴朝向、俯仰 −20°～+10°、绕工具轴滚转 ±40°）逐格解 IK，
        每格查关节限位（含 panda_py 那组更窄的墙）、自碰撞、离桌面高度（fr3_base 上方 120 mm），以及球窝和夹爪离球的支架多远（支架和球窝按插件 CAD，夹爪按厂商网格；10-13/10-14 的 21 次引导都落在它算出的可行域里）。
        再从能到的格子里挑 14 个驻点，使拟合的不确定度最小，按关节走得最少的顺序连起来，每个驻点停 1.5 s；执行时关节跟着规划走，力环以约 3 N 把球窝压在球上、横向随球滑动，峰值 5°/s。
        整条路径再查一遍限位、速度、j6 摆幅、桌面和支架，全过才能执行。
      </p>
      <div className="cali-op-actions">
        <label className="cali-muted">
          球心{" "}
          <select value={centre} disabled={locked} onChange={(e) => setCentre(e.target.value)}>
            <option value="">—</option>
            {centres.map((c) => (
              <option key={c.name} value={c.name}>
                {c.note ? `${c.note} · ` : ""}
                {c.name}
              </option>
            ))}
          </select>
        </label>
        <label className="cali-muted">
          偏置{" "}
          <select value={bias} disabled={locked} onChange={(e) => setBias(Number(e.target.value))}>
            {BIAS_PRESETS.map((b, i) => (
              <option key={b.label} value={i}>
                {b.label}
              </option>
            ))}
          </select>
        </label>
        <label className="cali-muted" title={anchorLine.text}>
          <input
            type="checkbox"
            checked={anchored && !!anchor?.available}
            disabled={locked || !anchor?.available}
            onChange={(e) => setAnchored(e.target.checked)}
          />{" "}
          绕上次拟合的球心转
        </label>
        <button className="cali-btn-primary" disabled={locked || !chosen} onClick={() => act("probe", () => api.planFr3Pivot(centre, [0, 0, 0], false, true))}>
          {running === "probe" ? "演算中…" : "规划试探"}
        </button>
        <button
          className="cali-btn-primary"
          disabled={locked || !chosen}
          onClick={() => act("plan", () => api.planFr3Pivot(centre, BIAS_PRESETS[bias].mm, anchored && !!anchor?.available))}
        >
          {running === "plan" ? "演算中…" : "规划 + 演算"}
        </button>
      </div>
      {plans.length ? (
        <table className="metric-table">
          <thead>
            <tr>
              <th>规划</th>
              <th>演算结果</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {plans.slice(0, PLANS_SHOWN).map((p) => (
              <tr key={p.name}>
                <td>
                  {p.kind === "probe" ? "试探" : biasLabel(p.biasMm)}
                  <div className="cali-muted">
                    {p.name} · 球心 {p.centre}
                    {p.centre !== latest ? "（不是最新球心）" : ""}
                  </div>
                </td>
                <td className={p.passed ? undefined : "cali-warn"}>{planSummaryText(p)}</td>
                <td>
                  <button className="cali-mini-btn" disabled={locked || !p.passed || !ip} onClick={() => setConfirmPlan(p.name)}>
                    执行
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}

        </>
      )}
      {toRun ? (
        <Modal
          title="FR3 自动绕球转动"
          onClose={() => setConfirmPlan("")}
          footer={
            <>
              <button onClick={() => setConfirmPlan("")}>取消</button>
              <button
                className="cali-btn-primary"
                onClick={() => {
                  setConfirmPlan("");
                  void act("execute", () => api.executeFr3Pivot(toRun.name, ip, note));
                }}
              >
                急停在手边，开始
              </button>
            </>
          }
        >
          <p>
            FR3（{ip}）会<b>自己动</b>：关节跟着规划的关节角、以约 3 N 把球窝压在球上（横向随球滑动、不顶球）绕球转约 {(toRun.durationS ?? 0).toFixed(0)} s，{biasLabel(toRun.biasMm)}。
          </p>
          {toRun.kind === "dwells" && !toRun.anchorFrom ? (
            <p className="cali-warn">
              这份规划绕 corenetic_socket_ee 的原点转，没有绕拟合的球心：夹持让球心偏开几毫米时，转不了多少就会停（10-19 两次都停在 20° 左右）。先跑一次试探。
            </p>
          ) : null}
          <p className="cali-muted">
            确认球窝还坐在取球心时的位置、夹爪夹紧、周围没人没东西、急停在手边。外力超过 10 N、球窝离规划点超过 5 mm、压不住、压力降不下来或横向推不开、关节偏离指令超过 3° 或转得过快、
            关节接近 panda_py 的墙，都会自动停下；也可以随时点「停止」或按急停。
          </p>
        </Modal>
      ) : null}
    </>
  );
}
