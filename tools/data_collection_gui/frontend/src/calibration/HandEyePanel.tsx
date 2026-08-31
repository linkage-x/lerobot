// Hand-eye (AX = XB): the rotation half of the marker rig -> TCP constant.
//
// This panel exists because the pivot fixture next to it structurally cannot
// answer the question. A ball-and-socket pivot is a 3-DoF spherical joint, so
// the mounting rotation sits in the null space of what it observes: more frames
// never help. The production bundle therefore still carries a *declared*
// rotation_sigma_deg = 2.0 -- 9.1 mm on the current lever, the largest single
// line in a 3 mm budget and the only one that was never measured at all.
//
// Two things drive the layout:
//   * the acquisition does not exist yet (it is blocked on the new gripper-rig
//     design), so the capture planner comes first. Today the useful output of
//     this item is the *specification* of the capture, not a constant;
//   * a refusal is a result. "not observable" and "mis-associated" are
//     successful runs of a tool that declines to emit a number, and they are
//     rendered as prominently as a success -- an underdetermined fit that gets
//     shown as an answer is exactly how this project previously shipped a
//     confident +29.3 mm that the URDF later contradicted by 50 mm.
import { useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { HandEyePlanResponse, HandEyeReport, HandEyeSolveResponse } from "../types";
import { Metric } from "../shared/ui";
import { Modal } from "./ConfirmModal";

const PAIRS_EXAMPLE = `{
  "schema": "hand_eye_pose_pairs/v1",
  "units": { "translation": "m" },
  "poses": [
    {
      "name": "p00",

      // 机器人侧：FK 给出的 base -> flange，4x4 行主序，平移单位米
      "T_base_flange": [
        [ 1, 0, 0, 0.412 ],
        [ 0, 1, 0, -0.03 ],
        [ 0, 0, 1, 0.318 ],
        [ 0, 0, 0, 1     ]
      ],

      // 视觉侧：多相机 BA 给出的 world -> rig，同一个位姿、同一时刻
      // 也可以写成 { "quat_xyzw": [...], "xyz_m": [...] }
      "T_world_rig": { "quat_xyzw": [0, 0, 0, 1], "xyz_m": [0.9, 0.1, 0.4] }
    }
  ]
}`;

const CAD_EXAMPLE = `{
  "T_flange_box": [
    [ 1, 0, 0, 0.000 ],
    [ 0, 1, 0, 0.000 ],
    [ 0, 0, 1, 0.041 ],
    [ 0, 0, 0, 1     ]
  ]
}`;

function fmt(value: number | null | undefined, digits = 3, unit = ""): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value.toFixed(digits)}${unit}`;
}

function UsageModal({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="Hand-eye 使用说明" onClose={onClose} footer={<button onClick={onClose}>关闭</button>}>
      <p className="cali-modal-lead">
        它测的是 <b>rig 相对夹爪/BOX 的安装旋转</b>，也就是 marker→TCP 常量里旋转的那一半。
        现役 bundle 的这一项 <b>从来没有被测量过</b>：left 是从 24 个轴对齐 mount 里按「物理上合理」挑的，
        right 直接继承 left，<code>rotation_sigma_deg = 2.0</code> 是一个声明。按现役杠杆臂折算是 <b>9.1 mm</b>，
        3 mm 预算的三倍，也是预算表上<b>最大的单项</b>。
      </p>

      <h4>为什么不能用旁边的 pivot 治具测</h4>
      <p>
        球窝是 <b>3-DoF 球副</b>，旋转落在观测的<b>零空间</b>里——这是结构性不可观测，加帧数、加时间都无效。
        必须换一个能让 rig 跟着<b>已知</b>刚体运动一起动的装置，机器人法兰就是。
      </p>

      <h4>采集怎么做</h4>
      <ol>
        <li>把 BOX（连同 marker rig）装到 FR3 法兰上，装夹要刚，中途<b>不能松动</b>——松了就是换了一个待测量。</li>
        <li>
          驱动机器人走一组姿态。<b>关键不是走多少个点，是转过多少个不同的转轴</b>：全部绕同一根轴转，
          AX=XB 在数学上就解不出旋转，而最小二乘照样会给你一个干净的矩阵。至少要有两根明显不平行的转轴。
        </li>
        <li>
          <b>每个姿态停下来</b>，静止采若干秒。视觉侧逐帧姿态噪声按 √N 平均下去，这是把每位姿噪声压到
          0.1° 量级最便宜的办法，也是下面「采集规划」那张表能不能成立的前提。
        </li>
        <li>全程 rig 必须在多相机视野内，且尽量被多台相机同时看到。</li>
        <li>
          每个姿态记两份位姿：机器人 FK 的 <code>T_base_flange</code>，和多相机 BA 的 <code>T_world_rig</code>，
          按姿态一一对应写进一个 JSON。
        </li>
      </ol>

      <h4>输入文件格式</h4>
      <ul>
        <li>路径可以是绝对路径，或相对 gateway 仓库根目录。</li>
        <li>两个位姿都支持 <b>4×4 行主序</b>齐次矩阵，或 <code>{"{ quat_xyzw, xyz_m }"}</code>，可以混用。</li>
        <li>平移单位<b>米</b>；旋转块会校验正交且 det=+1，不合格会点名报错而不是静默接受。</li>
        <li>示例里的 <code>//</code> 注释只是说明，真实 JSON 不能带注释。</li>
      </ul>
      <pre className="marker-tcp-example">{PAIRS_EXAMPLE}</pre>

      <h4>T_flange_box（可选，但不会替你猜）</h4>
      <p>
        URDF 里 <code>joint_link_marker_cube</code> 和 <code>joint_lt_gripper_base</code> 的 parent 都是{" "}
        <code>link_box</code>，marker 挂在 BOX 本体上而不是夹爪上，所以生产要的是 <code>R_box_rig</code>；
        而 hand-eye 解出的是 <code>T_flange_rig</code>，中间差的 <code>T_flange_box</code> <b>只出现一次、不会抵消</b>，
        是这条链上唯一残留的 CAD 输入。不给它，结果里 <code>T_box_rig</code> 会被标成{" "}
        <code>missing</code> 而<b>不会用默认值顶替</b>——一个假设错的安装面会静默地把答案整体转过去，
        看起来和一次好的解算一模一样。
      </p>
      <pre className="marker-tcp-example">{CAD_EXAMPLE}</pre>

      <h4>结果怎么读</h4>
      <ul>
        <li>
          <b>先看可观测性</b>，再看数值。转轴 s2/s1 小于 0.05 就意味着这次采集根本没有约束住旋转，
          此时任何漂亮的残差都不算数。
        </li>
        <li>
          <b>残差不是精度</b>。判据是留一位姿 holdout 与 bootstrap 里较大的那个，
          目标 ≤0.2°（即杠杆末端 ≤1 mm），≤0.5° 为可接受下限。
        </li>
        <li>
          解出来<b>不会自动进生产</b>。2026-08-27 查出过一次：解算导出完全正确、界面显示已生效，
          而生产 yaml 从头到尾没被改过，错了七天。所以这里只出数和证据，改生产是另一次明确的动作。
        </li>
      </ul>
    </Modal>
  );
}

function VerdictBox({ report, returncode }: { report: HandEyeReport | null; returncode?: number }) {
  const status = report?.verdict?.status ?? "unknown";
  const why = report?.verdict?.why ?? "";
  const tone: Record<string, { label: string; className: string; advice: string }> = {
    ok: {
      label: "已解出并落在预算内",
      className: "callout ok",
      advice: "下一步是决定要不要提升到生产——那是一次单独的、要人点的动作。",
    },
    not_observable: {
      label: "不可观测 · 没有出数",
      className: "callout warn",
      advice: "这次采集的转轴太集中。补一组绕明显不同转轴的位姿再解，不要试图用更多同轴位姿凑。",
    },
    mis_associated: {
      label: "两路位姿对不上 · 没有出数",
      className: "callout warn",
      advice:
        "共轭不改变旋转角，所以 |A| 与 |B| 必须相等。对不上通常是 FK 与视觉的位姿顺序错位或漏了一个姿态。",
    },
    insufficient_motions: {
      label: "可用运动太少 · 没有出数",
      className: "callout warn",
      advice: "过滤掉过小/过大的相对旋转后剩不下两组独立运动。加位姿，或放宽角度过滤。",
    },
    solved_but_out_of_budget: {
      label: "解出来了，但超预算",
      className: "callout warn",
      advice: "数值可用作参考，但不能作为生产常量。按「采集规划」加位姿或降低逐位姿姿态噪声。",
    },
    no_uncertainty_estimate: {
      label: "解出来了，但没有不确定度 · 不算测量",
      className: "callout warn",
      advice:
        "位姿太少，holdout 和 bootstrap 都给不出 σ。没有 σ 的常量不能进预算表——先加位姿。",
    },
  };
  const shown = tone[status] ?? {
    label: `未知结果 (returncode ${returncode ?? "?"})`,
    className: "callout warn",
    advice: "查看下面的原始输出。",
  };
  return (
    <div className={shown.className}>
      <b>{shown.label}</b>
      {why && <p className="cali-muted">{why}</p>}
      <p className="cali-muted">{shown.advice}</p>
    </div>
  );
}

export function HandEyePanel({ api, busy }: { api: DataCollectionGuiApi; busy: boolean }) {
  const [showUsage, setShowUsage] = useState(false);
  const [pairsPath, setPairsPath] = useState("");
  const [cadPath, setCadPath] = useState("");
  const [leverMm, setLeverMm] = useState("102.3");
  const [planPoses, setPlanPoses] = useState("6,8,12,16,24,32");
  const [planNoise, setPlanNoise] = useState("0.10");
  const [running, setRunning] = useState<"" | "plan" | "solve">("");
  const [plan, setPlan] = useState<HandEyePlanResponse | null>(null);
  const [solve, setSolve] = useState<HandEyeSolveResponse | null>(null);

  const report = solve?.report ?? null;
  const disabled = busy || running !== "";

  async function onPlan() {
    setRunning("plan");
    setPlan(await api.runHandEyePlan({ poses: planPoses, poseNoiseDeg: planNoise, leverMm }));
    setRunning("");
  }

  async function onSolve() {
    setRunning("solve");
    setSolve(await api.runHandEyeSolve({ pairsPath, tFlangeBoxPath: cadPath, leverMm }));
    setRunning("");
  }

  return (
    <section className="panel">
      <div className="cali-result-head">
        <h3>Hand-eye 手眼标定 · rig→TCP 旋转</h3>
        <div className="cali-op-actions">
          <span className="cali-preview-badge">未接入生产</span>
          <button className="cali-mini-btn" onClick={() => setShowUsage(true)}>
            使用说明
          </button>
        </div>
      </div>

      {/* 作用：一段，先说这项在测什么、为什么必须是它。 */}
      <p className="cali-muted">
        <b>作用：</b>测出 marker rig 装在 BOX 上的<b>安装旋转</b>——marker→TCP 常量里旋转的那一半。
        现役 bundle 这一项是<b>声明值 2.0°</b>而非测量，按杠杆臂折算 <b>9.1 mm</b>，是 3 mm 误差预算里
        <b>最大的单项</b>。旁边的 pivot 治具测不了它：球窝是 3-DoF 球副，旋转在观测的零空间里，加帧数无效。
        本工具只吃「机器人 FK ↔ 视觉 BA」的位姿对，不读 rig 几何，所以夹爪换设计不会让它作废。
      </p>

      {/* 采集规划在解算前面：采集本身还被新夹爪方案阻塞，现在能产出的是它的验收规格。 */}
      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>① 采集规划</b>
          <span className="cali-muted">采集还没做，先算清楚它要长什么样</span>
        </div>
        <div className="cali-op-grid">
          <label className="cali-field">
            候选位姿数
            <input value={planPoses} onChange={(e) => setPlanPoses(e.target.value)} placeholder="6,8,12,16" />
          </label>
          <label className="cali-field">
            逐位姿姿态噪声 (deg)
            <input value={planNoise} onChange={(e) => setPlanNoise(e.target.value)} placeholder="0.10" />
          </label>
          <label className="cali-field">
            杠杆臂 (mm)
            <input value={leverMm} onChange={(e) => setLeverMm(e.target.value)} placeholder="102.3" />
          </label>
          <button className="cali-btn-primary" disabled={disabled} onClick={onPlan}>
            {running === "plan" ? "计算中…" : "算需要多少位姿"}
          </button>
        </div>
        <p className="cali-muted">
          姿态噪声是<b>输入</b>不是测量：它把每一行一起放缩，所以这张表要当<b>形状</b>看（曲线在哪里变平），
          不是当承诺看。每个位姿静止多采几秒、把逐帧噪声按 √N 平均下去，是压低这个输入最便宜的办法。
        </p>

        {plan?.ok && plan.plan?.rows && (
          <table className="metric-table">
            <thead>
              <tr>
                <th>位姿数</th>
                <th>旋转误差 p50</th>
                <th>p95</th>
                <th>杠杆末端 p95</th>
                <th>判定</th>
              </tr>
            </thead>
            <tbody>
              {plan.plan.rows.map((row) => (
                <tr key={row.num_poses}>
                  <td>{row.num_poses}</td>
                  <td>{fmt(row.rotation_error_p50_deg, 3, "°")}</td>
                  <td>{fmt(row.rotation_error_p95_deg, 3, "°")}</td>
                  <td>{fmt(row.lever_equivalent_p95_mm, 2, " mm")}</td>
                  <td>
                    {row.meets_target ? "达标 (≤0.2°)" : row.meets_acceptable ? "可接受 (≤0.5°)" : "超预算"}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
        {plan && !plan.ok && <p className="cali-warn">{plan.error}</p>}
      </div>

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>② 解算</b>
          <span className="cali-muted">采完之后，把位姿对喂进来</span>
        </div>
        <div className="cali-op-grid">
          <label className="cali-field">
            pose-pair JSON
            <input
              value={pairsPath}
              onChange={(e) => setPairsPath(e.target.value)}
              placeholder="outputs/metrology/hand_eye/pairs_20260901.json"
            />
          </label>
          <label className="cali-field">
            T_flange_box JSON（可选，缺了不猜）
            <input value={cadPath} onChange={(e) => setCadPath(e.target.value)} placeholder="留空则 T_box_rig 标为 missing" />
          </label>
          <button className="cali-btn-primary" disabled={disabled || !pairsPath.trim()} onClick={onSolve}>
            {running === "solve" ? "解算中…" : "解算"}
          </button>
        </div>

        {solve && <VerdictBox report={report} returncode={solve.returncode} />}

        {report?.observability && (
          <div className="cali-meta-row">
            <Metric
              label="可观测性"
              value={report.observability.ok ? "ok" : "不可观测"}
            />
            <Metric label="转轴 s2/s1" value={fmt(report.observability.rotation_axis_rank_ratio, 4)} />
            <Metric label="平移 s3/s1" value={fmt(report.observability.translation_rank_ratio, 4)} />
            <Metric label="运动数" value={`${report.motions?.num_motions ?? "—"} / ${report.motions?.num_candidate_pairs ?? "—"}`} />
          </div>
        )}

        {report?.solution && (
          <div className="cali-meta-row">
            <Metric label="残差 旋转 p95" value={fmt(report.solution.residual_rotation_deg?.p95, 4, "°")} />
            <Metric label="残差 平移 p95" value={fmt(report.solution.residual_translation_mm?.p95, 3, " mm")} />
            <Metric label="留一位姿 p95" value={fmt(report.holdout?.solution_shift_rotation_deg?.p95, 4, "°")} />
            <Metric label="bootstrap σ" value={fmt(report.bootstrap?.sigma_deg, 4, "°")} />
          </div>
        )}

        {report?.rotation_sigma_deg != null && (
          <div className="callout">
            <b>
              rotation_sigma_deg = {fmt(report.rotation_sigma_deg, 3, "°")} → 杠杆末端{" "}
              {fmt(report.lever_equivalent_mm, 2, " mm")}
            </b>
            <p className="cali-muted">
              取 holdout 与 bootstrap 里<b>较大</b>的那个。它替换的是声明值{" "}
              {fmt(report.against_budget?.replaces_declared_deg, 1, "°")}（
              {fmt(report.against_budget?.replaces_declared_mm, 1, " mm")}），后者从未被测量。
            </p>
          </div>
        )}

        {report?.T_box_rig?.status === "missing" && (
          <p className="cali-warn">
            <b>T_box_rig 未计算：</b>
            {report.T_box_rig.why}
          </p>
        )}

        {report && (
          <p className="cali-muted">
            解算<b>不改任何生产配置</b>。报告写在 <code>{solve?.reportPath || "outputs/metrology/hand_eye/"}</code>
            ，提升到生产是另一次明确的动作。
          </p>
        )}

        {solve && !solve.ok && !report && <p className="cali-warn">{solve.error}</p>}
      </div>

      {showUsage && <UsageModal onClose={() => setShowUsage(false)} />}
    </section>
  );
}
