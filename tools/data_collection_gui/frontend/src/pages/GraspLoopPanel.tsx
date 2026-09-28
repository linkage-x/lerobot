import type { GraspLoopProgress, RolloutRun } from "../types";
import { armRates, gateReading, medianByVerdict, wilsonInterval } from "./graspLoop";

const VERDICT_COLORS: Record<string, string> = {
  held: "#38a169",
  empty: "#e53e3e",
  no_close: "#dd6b20",
  collision: "#9b2c2c",
  not_graded: "#718096"
};

const ARM_LABELS: Record<string, string> = {
  A: "A 纯策略",
  B: "B 策略 + 漏斗"
};

const VERDICT_LABELS: Record<string, string> = {
  held: "抓住",
  empty: "空抓",
  no_close: "没合手",
  collision: "碰撞保护",
  not_graded: "未评"
};

function fmt(value: number | null | undefined, digits = 1): string {
  return typeof value === "number" ? value.toFixed(digits) : "—";
}

/** The grasp loop's live card: where the run is, what it has measured, and its two controls.
 *
 *  The loop grades itself, so there is nothing to fill in here. What the operator does is start
 *  it, put the peg back when the loop asks, and stop it -- at a trial boundary, which leaves the
 *  peg on the table and the arm homed, or with End session, which stops it where it stands. */
export function GraspLoopPanel({
  run,
  busy,
  onControl
}: {
  run: RolloutRun;
  busy: boolean;
  onControl: (command: "grasp_stop" | "grasp_continue") => void;
}) {
  const progress: GraspLoopProgress | undefined = run.graspLoop;
  const live = run.state !== "complete" && run.state !== "error" && run.state !== "stopped";
  if (!progress || !progress.planned) {
    return (
      <div className="subcard">
        <h4>抓取循环</h4>
        <p className="hint">Loading the policy and checking the target mask… the first trial starts after homing.</p>
      </div>
    );
  }
  const [low, high] = wilsonInterval(progress.held, progress.graded);
  const closeHeight = medianByVerdict(progress.trials, "closeAboveTargetMm");
  const lateral = medianByVerdict(progress.trials, "lateralMm");
  const done = progress.trials.length;
  const rates = armRates(progress.trials);
  const multiArm = rates.length > 1 || (progress.arms ?? "A") !== "A";
  const current = progress.currentTrial;

  return (
    <div className="subcard">
      <h4>抓取循环 · {done} / {progress.planned}</h4>

      {progress.needsOperator && (
        <div className="banner banner-warn" style={{ display: "flex", gap: 12, alignItems: "center", flexWrap: "wrap" }}>
          {progress.needsOperator.startsWith("reflex") ? (
            <span>
              <strong>需要你：</strong>机械臂触发了碰撞保护，已停住。确认手指下没有压着东西、周围安全后点「安全，恢复」——
              机械臂会松爪、竖直抬起、回原位，然后再请你放回销子。
            </span>
          ) : (
            <span>
              <strong>需要你：</strong>销不在它原来的位置了（被碰倒或推走）。把销插回孔里（fixture pick pose），
              手离开工作区后点「已放回，继续」。
            </span>
          )}
          <button type="button" className="primary" disabled={busy} onClick={() => onControl("grasp_continue")}>
            {progress.needsOperator.startsWith("reflex") ? "安全，恢复" : "已放回，继续"}
          </button>
        </div>
      )}

      {progress.halted && (
        <div className="banner banner-error">
          循环停了：{progress.halted}
          {progress.haltDetails ? ` — ${progress.haltDetails}` : ""}
        </div>
      )}

      <div style={{ display: "flex", gap: 24, flexWrap: "wrap", margin: "8px 0" }}>
        {multiArm &&
          rates.map((rate) => (
            <div key={rate.arm}>
              <div className="hint">{ARM_LABELS[rate.arm] ?? rate.arm}</div>
              <strong style={{ fontSize: 20 }}>
                {rate.held} / {rate.graded}
              </strong>
              <div className="hint">
                {Math.round((100 * rate.held) / rate.graded)}% · 95% CI {Math.round(rate.low * 100)}–
                {Math.round(rate.high * 100)}%
                {rate.arm === "B" && gateReading(rate.held, rate.graded) ? ` · ${gateReading(rate.held, rate.graded)}` : ""}
              </div>
            </div>
          ))}
        <div>
          <div className="hint">{multiArm ? "两臂合计（仅供参考）" : "抓住率"}</div>
          <strong style={{ fontSize: 20 }}>
            {progress.graded ? `${progress.held} / ${progress.graded}` : "—"}
          </strong>
          <div className="hint">
            {progress.graded ? `${Math.round((100 * progress.held) / progress.graded)}% · 95% CI ${Math.round(low * 100)}–${Math.round(high * 100)}%` : "no trials graded yet"}
          </div>
        </div>
        <div>
          <div className="hint">合手高度（高于放置点, mm, 中位）</div>
          <strong>抓住 {fmt(closeHeight.held)} · 没抓住 {fmt(closeHeight.notHeld)}</strong>
        </div>
        <div>
          <div className="hint">横向误差（mm, 中位）</div>
          <strong>抓住 {fmt(lateral.held)} · 没抓住 {fmt(lateral.notHeld)}</strong>
        </div>
        <div>
          <div className="hint">现在</div>
          <strong>
            {progress.done
              ? "已结束"
              : progress.needsOperator
                ? "等你放回销"
                : current !== null
                  ? `第 ${current + 1} 条进行中`
                  : live
                    ? "准备下一条"
                    : run.state}
          </strong>
        </div>
      </div>

      {live && !progress.done && (
        <div className="row-actions" style={{ marginBottom: 8 }}>
          <button
            type="button"
            disabled={busy || progress.stopRequested}
            onClick={() => onControl("grasp_stop")}
            title="The trial in flight finishes and is graded; the peg is put down and the arm homed. Then the session ends."
          >
            {progress.stopRequested ? "本条结束后停止（已请求）" : "本条结束后停止"}
          </button>
          <span className="hint">
            要立刻停在原地用顶部的 End session。它会中断当前动作，销可能留在夹爪里。
          </span>
        </div>
      )}

      <div className="table-scroll" style={{ maxHeight: 280 }}>
        <table className="table" style={{ width: "100%", fontSize: 12 }}>
          <thead>
            <tr>
              <th>#</th>
              {multiArm && <th>臂</th>}
              <th>结果</th>
              <th>抬起后宽度</th>
              <th>合手高度 mm</th>
              <th>横向 mm</th>
              <th>用时 s</th>
            </tr>
          </thead>
          <tbody>
            {[...progress.trials].reverse().map((trial) => (
              <tr key={trial.trial}>
                <td>{trial.trial + 1}</td>
                {multiArm && <td>{trial.arm ?? "A"}</td>}
                <td style={{ color: VERDICT_COLORS[trial.verdict] }}>{VERDICT_LABELS[trial.verdict] ?? trial.verdict}</td>
                <td>{fmt(trial.widthLifted, 3)}</td>
                <td>{fmt(trial.closeAboveTargetMm)}</td>
                <td>{fmt(trial.lateralMm)}</td>
                <td>{fmt(trial.trialS, 0)}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {progress.out && (
        <p className="hint">
          记录文件：<code>{progress.out}</code>
        </p>
      )}
    </div>
  );
}
