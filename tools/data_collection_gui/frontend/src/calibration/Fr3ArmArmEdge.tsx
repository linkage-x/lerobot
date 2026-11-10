// fr3_base -> fr3_base_right from balls both FR3s pivoted about (fr3_arm_arm_ball.py, roadmap B3).
//
// Each row is one ball position: a pivot of the left arm and one of the right arm about the same ball,
// the ball not moved in between. Both centres are FK, so the edge carries no camera error; it is written
// as a candidate of the FR3 base panel, which compares it with any live edge and applies it (checking
// the loop through the world when both world edges exist).
import { useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { Fr3PivotResponse, RobotArms } from "../types";
import { armIp } from "../shared/robotArms";
import { ARM_ARM_MIN_POSITIONS, type ArmArmDraft, armArmDraftWarnings, pivotsOfArm } from "./fr3Pivot";

type Props = {
  api: DataCollectionGuiApi;
  payload: Fr3PivotResponse | null;
  arms: RobotArms | null;
  disabled: boolean;
  running: boolean;
  act: (label: string, action: () => Promise<Fr3PivotResponse | null>) => Promise<void>;
};

function PivotSelect({
  value,
  options,
  disabled,
  onChange,
}: {
  value: string;
  options: ReturnType<typeof pivotsOfArm>;
  disabled: boolean;
  onChange: (value: string) => void;
}) {
  return (
    <select value={value} disabled={disabled} onChange={(e) => onChange(e.target.value)}>
      <option value="">—</option>
      {options.joints.length ? (
        <optgroup label="合并拟合（共用球心）">
          {options.joints.map((j) => (
            <option key={j.name} value={j.name}>
              {j.runs.length} 次合并 · {j.name}
            </option>
          ))}
        </optgroup>
      ) : null}
      <optgroup label="单次">
        {options.runs.map((r) => (
          <option key={r.name} value={r.name}>
            {r.note ? `${r.note} · ` : ""}
            {r.name}
          </option>
        ))}
      </optgroup>
    </select>
  );
}

export function Fr3ArmArmEdge({ api, payload, arms, disabled, running, act }: Props) {
  const [drafts, setDrafts] = useState<ArmArmDraft[]>([{ left: "", right: "" }]);
  const runs = payload?.runs ?? [];
  const joints = payload?.joints ?? [];
  const leftIp = armIp(arms, "left");
  const rightIp = armIp(arms, "right");
  const left = pivotsOfArm(runs, joints, leftIp);
  const right = pivotsOfArm(runs, joints, rightIp);
  const warnings = armArmDraftWarnings(drafts, leftIp, rightIp);
  const complete = drafts.filter((d) => d.left && d.right);
  const patch = (i: number, p: Partial<ArmArmDraft>) =>
    setDrafts((rows) => rows.map((row, j) => (j === i ? { ...row, ...p } : row)));

  return (
    <div className="cali-result-box">
      <div className="cali-result-box-head">
        <b>两臂同球 · fr3_base → fr3_base_right</b>
        <span className="cali-muted">两边都是 FK，不经过相机</span>
      </div>
      <p className="cali-muted">
        同一个球<b>不动</b>，左臂（{leftIp || "—"}）和右臂（{rightIp || "—"}）各绕它做一次 pivot（上面的采集，选对臂），
        换 ≥ {ARM_ARM_MIN_POSITIONS} 个两臂都够得着、相隔约 30 cm 且不共线的位置。下拉框只列出该臂拟合过的 pivot。
        解出的边存为「FR3 基座注册边」面板的候选，在那里比较、确认后换边。
      </p>
      <table className="metric-table">
        <thead>
          <tr>
            <th>位置</th>
            <th>左臂 pivot</th>
            <th>右臂 pivot</th>
            <th />
          </tr>
        </thead>
        <tbody>
          {drafts.map((d, i) => (
            <tr key={i}>
              <td>{i + 1}</td>
              <td>
                <PivotSelect value={d.left} options={left} disabled={disabled} onChange={(v) => patch(i, { left: v })} />
              </td>
              <td>
                <PivotSelect value={d.right} options={right} disabled={disabled} onChange={(v) => patch(i, { right: v })} />
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
        <button className="cali-mini-btn" disabled={disabled} onClick={() => setDrafts((rows) => [...rows, { left: "", right: "" }])}>
          添加位置
        </button>
        <button
          className="cali-btn-primary"
          disabled={disabled || complete.length < 3}
          onClick={() => void act("armArm", () => api.solveFr3ArmArmEdge(complete))}
        >
          {running ? "解算中…" : "解算两臂边"}
        </button>
      </div>
      {warnings.map((w) => (
        <p className="cali-warn" key={w}>
          {w}
        </p>
      ))}
      {payload?.armArmEdge ? (
        <p>
          已存为候选边 <code>{payload.armArmEdge}</code>：到「FR3 基座注册边」面板点刷新，看判定和留一误差后再换边。
        </p>
      ) : null}
    </div>
  );
}
