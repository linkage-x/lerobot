// Laser tracker: where the tracker stands (T_WG) and where the SMR sits on the
// rig (c). Two fits, because they are constants of two different things -- the
// station is a constant of the room and survives sessions, the lever arm is a
// constant of one bolting-on of the plate and does not.
//
// The panel is laid out around what the fit is *entitled to claim*, because the
// arithmetic is a Kabsch and a linear solve and is not where this goes wrong:
//
//   * observability before numbers. c is determined by rotation, so a capture
//     with one axis of rotation leaves a direction of c undetermined while the
//     residual stays small and reassuring;
//   * absorbed modes next to the residual. The fit removes a constant body-frame
//     translation and rotation, which together are exactly the marker->TCP
//     constant -- so this can never certify cube->TCP, however small the rms;
//   * the mount check is attitude *structure*, not residual size. The spread of
//     the per-pose lever arms is algebraically the same number as the rms, so
//     size alone cannot separate a flexing plate from an attitude-dependent
//     pipeline.
import { useEffect, useState } from "react";
import type { DataCollectionGuiApi, GuiSnapshot } from "../api";
import type {
  TrackerMountArtifact,
  TrackerMountCaptureRow,
  TrackerMountReport,
  TrackerMountSolveResponse,
  TrackerMountCapture,
  TrackerMountChainResponse,
  TrackerMountSession,
  TrackerPivotReport,
  TrackerStationReport,
  TrackerValidateResponse,
} from "../types";
import { Metric, StatusDot } from "../shared/ui";
import { Modal } from "./ConfirmModal";
import { TrackerPivotResult } from "./TrackerPivotResult";
import type { RecorderLiveState } from "./trackerMount";
import {
  DWELL_SECONDS_MIN,
  DWELL_TRIM_S,
  PIVOT_SWEEP_SUGGESTED_S,
  POSES_TO_CERTIFY,
  SEGMENTS_SUGGESTED,
  SEGMENT_MIN_S,
  absorbedNote,
  earlySaveWarning,
  protocolLabel,
  segmentLengthVerdict,
  attitudeVerdict,
  captureLabel,
  captureReadiness,
  coverageVerdict,
  fmtMm,
  holdoutVerdict,
  observabilityRows,
  scaleVerdict,
  sigmaVerdict,
  speedStrataRows,
  timeBaseVerdict,
  timingSignal,
  suggestedDwellSeconds,
  trackerMountVerdict,
  validateVerdict,
} from "./trackerMount";

const EMPTY_ROW: TrackerMountCaptureRow = { session: "", dataset: "", episode: "", mountId: "" };

function UsageModal({ onClose }: { onClose: () => void }) {
  return (
    <Modal title="跟踪仪站位 / 杠杆臂 使用说明" onClose={onClose} footer={<button onClick={onClose}>关闭</button>}>
      <p className="cali-modal-lead">
        这一项解的是 <b>T_WG</b>（跟踪仪系 G → 相机 world 系 W）和 <b>c</b>（SMR 球心在 rig 本体系里的位置）。
        两个都<b>不能从 CAD 读</b>：T_WG 取决于跟踪仪这次站在哪，c 取决于这次把板子拧在哪。
      </p>

      <h4>为什么 c 要拟合而不是量</h4>
      <p>
        T_WG 本来就非解不可（6 个未知数），而每个姿态给 3 个方程，再挂 3 个未知数几乎白送。
        解出来的是<b>实际装配</b>，不是设计值——打印件有过 1.6% 尺度偏差的前科，100 mm 力臂上就是 1.6 mm。
        CAD 在这里的角色是<b>初值和粗差判据</b>：解出的 c 和 CAD 差太多说明装错了，而不是"标定不准"。
      </p>

      <h4>采集：要的是姿态，不是位置</h4>
      <p>
        识别 c 靠的是<b>旋转</b>。两个姿态相减消掉 t_WG，剩下 <code>(R_i − R_j) c</code>——
        纯平移的一对<b>什么也不贡献</b>，两根近乎平行的轴会让 c 的某个方向定不下来<b>而残差照样很小</b>。
        所以 station 模式要求绕<b>两根明显不平行</b>的轴转，不满足会直接拒绝而不是给个好看的数。
      </p>
      <ul>
        <li><b>station</b>：一次或多次 session 累积解 T_WG，每个 session 给一个自己的 c。需要姿态多样性。</li>
        <li>
          <b>lever-arm</b>：站位冻结后单独解 c。<b>线性、完全不需要旋转</b>，所以定姿态的 session 也能用——
          这也是"跟踪仪站着不动"值一个 4 倍 σ_c 的原因。
        </li>
        <li>
          <b>mount id</b> 不是标签：拧过一次板子就是新的 id。跨 mount 的两个姿态杠杆臂不同，相减不构成约束，
          写错了不会报错，只会把条件数虚高。
        </li>
      </ul>

      <h4>结果怎么读（顺序不能反）</h4>
      <ol>
        <li><b>先看可观测性</b>。最弱方向增益换算成等效角，太小就是"再换一根轴转"，此时残差不算数。</li>
        <li>
          <b>再看吸收项</b>。拟合会吃掉本体系的常量平移和常量旋转，这两项加起来<b>就是 marker→TCP 常量</b>。
          所以残差再小也<b>不能</b>用来证明 cube→TCP 是对的——那一项得靠 CAD 或 pivot，两个仪器都看不见 TCP。
        </li>
        <li>
          <b>刚性看结构不看幅度</b>。逐姿态 c 的散布和残差 RMS 在代数上是同一个数（c 的最小二乘解正好是
          逐姿态 c 的均值），所以幅度分不开"板子变形"和"相机姿态误差"。能分开的是散布<b>与姿态是否相关</b>。
        </li>
        <li>
          <b>尺度是诊断，不施加</b>。注册故意用刚性 6 参数：放开 7 参数会把相机系统的尺度误差吸进 T_WG，
          残差看着干净，而尺度正是跟踪仪唯一有资格认证的东西。
        </li>
      </ol>

      <h4>C. GT 比较能判什么、判不了什么</h4>
      <p>
        比较的是<b>球心</b>：相机侧 <code>R_W_cube·c + t_W_cube</code>，跟踪仪侧 <code>R_WG·p_G + t_WG</code>。
        <b>TCP 一次都没出现</b>，所以这一步完全不依赖 cube→TCP，不用等它重标就能跑。
      </p>
      <ul>
        <li>
          <b>判得了</b>：随时间/姿态变化的那部分——噪声、漂移、姿态相关性，以及按速度分层出来的时间项。
        </li>
        <li>
          <b>判不了</b>：常量。杠杆臂拟合把本体系的常量平移和常量旋转都吸收了，这两项加起来就是
          marker→TCP 常量。一个整体偏 5 mm 的 cube→TCP 会被 c 原样吸收，残差一样漂亮。
          所以「残差小 ⇒ cube→TCP 没问题」是<b>错的推理</b>；要查那一项只能靠 CAD / pivot。
        </li>
        <li>
          <b>速度分层</b>是几何项和时间项的分界：常量时间偏移按 <code>|v|·Δt</code> 走，静止时为零、随速度线性增长。
          静止段的 p95 就是纯几何误差，涨出来的那部分是时间基准的账。
        </li>
        <li>
          <b>顺带能定曝光符号</b>：同一个 episode 用 −0.5 / 0 / +0.5 各跑一遍，哪个把速度相关的增长压平，
          哪个就是对的符号。默认留空 = 跟随录制器现行值（0.0），因为拿 CLI 默认的 0.5 打分等于给一条
          从来没被生产出来过的轨迹打分。
        </li>
      </ul>
    </Modal>
  );
}

function CaptureRows({
  rows,
  disabled,
  onChange,
  single,
}: {
  rows: TrackerMountCaptureRow[];
  disabled: boolean;
  onChange: (rows: TrackerMountCaptureRow[]) => void;
  single?: boolean;
}) {
  function patch(index: number, key: keyof TrackerMountCaptureRow, value: string) {
    onChange(rows.map((row, i) => (i === index ? { ...row, [key]: value } : row)));
  }
  return (
    <>
      {rows.map((row, index) => (
        <div className="cali-op-grid" key={index}>
          <label className="cali-field">
            tracker session 目录
            <input
              value={row.session}
              disabled={disabled}
              onChange={(e) => patch(index, "session", e.target.value)}
              placeholder="outputs/laser_tracker/20260921_a"
            />
          </label>
          <label className="cali-field">
            数据集目录
            <input
              value={row.dataset}
              disabled={disabled}
              onChange={(e) => patch(index, "dataset", e.target.value)}
              placeholder="outputs/datasets/box_umi_rig0818"
            />
          </label>
          <label className="cali-field">
            episode
            <input
              value={row.episode}
              disabled={disabled}
              onChange={(e) => patch(index, "episode", e.target.value)}
              placeholder="3"
            />
          </label>
          <label className="cali-field">
            mount id（拧一次板子 = 一个 id）
            <input
              value={row.mountId}
              disabled={disabled}
              onChange={(e) => patch(index, "mountId", e.target.value)}
              placeholder="plate_v1_20260921"
            />
          </label>
          {!single && rows.length > 1 && (
            <button
              className="cali-mini-btn"
              disabled={disabled}
              onClick={() => onChange(rows.filter((_, i) => i !== index))}
            >
              删除
            </button>
          )}
        </div>
      ))}
      {!single && (
        <button className="cali-mini-btn" disabled={disabled} onClick={() => onChange([...rows, { ...EMPTY_ROW }])}>
          + 再加一个 session
        </button>
      )}
    </>
  );
}

function Verdict({ result }: { result: TrackerMountSolveResponse | null }) {
  const verdict = trackerMountVerdict(result);
  if (!verdict) return null;
  // A solved pivot's own card already lists them.
  const skipped = result?.kind === "pivot" && result.report ? [] : (result?.skipped ?? []);
  return (
    <div className="cali-result-box">
      <div className="cali-result-box-head">
        <StatusDot state={verdict.dot} />
        <b>{verdict.title}</b>
      </div>
      <p className="cali-muted">{verdict.detail}</p>
      {skipped.length > 0 && (
        <div className="cali-warn">
          <p>
            跳过了 {skipped.length} 段（没停够/晃动超门槛，不参与拟合）：
            {skipped.map((item) => `ep${item.episode}`).join("、")}。这些姿态要的话请重录。
          </p>
          <ul>
            {skipped.map((item) => (
              <li key={item.episode}>
                ep{item.episode}：{item.why}
              </li>
            ))}
          </ul>
        </div>
      )}
      {result?.reportPath && <p className="cali-muted">结果文件：<code>{result.reportPath}</code></p>}
    </div>
  );
}

function Observability({ report }: { report: TrackerStationReport | TrackerMountReport | null }) {
  const rows = observabilityRows(report?.observability);
  if (!rows.length) return null;
  const reasons = report?.observability?.reasons ?? [];
  return (
    <div className="cali-result-box">
      <div className="cali-result-box-head">
        <b>① 可观测性</b>
        <span className="cali-muted">先看这个；不过关的话下面的残差不算数</span>
      </div>
      <table className="metric-table">
        <tbody>
          {rows.map((row) => (
            <tr key={row.label}>
              <td>{row.label}</td>
              <td>{row.value}</td>
              <td className="cali-muted">{row.hint}</td>
            </tr>
          ))}
        </tbody>
      </table>
      {reasons.length > 0 && (
        <p className="cali-warn">这批采集定不出 c：{reasons.join("；")}</p>
      )}
    </div>
  );
}

function MountNumbers({ report }: { report: TrackerMountReport }) {
  const att = attitudeVerdict(report.attitude);
  const sigma = sigmaVerdict(report.sigma);
  const holdout = holdoutVerdict(report);
  const absorbed = absorbedNote(report.absorbed_modes);
  return (
    <>
      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>② 杠杆臂 c</b>
          <span className="cali-muted">{report.mount_id}</span>
        </div>
        <div className="cali-metric-row">
          <Metric label="|c|" value={fmtMm(report.lever_arm_mm, 1)} />
          <Metric label="旋转敏感度" value={`${report.rotation_sensitivity_mm_per_deg.toFixed(2)} mm/度`} />
          <Metric label="残差 RMS" value={fmtMm(report.rms_mm)} />
          <Metric label="残差 max" value={fmtMm(report.max_mm)} />
        </div>
        <p className="cali-muted">
          c = [{report.c_m.map((v) => (v * 1e3).toFixed(2)).join(", ")}] mm。
          单光束跟踪仪之所以能说出一点姿态信息，全靠这个力臂：1 度的 rig 姿态误差把球心挪
          {report.rotation_sensitivity_mm_per_deg.toFixed(2)} mm。
        </p>
        {absorbed && <p className="cali-warn">{absorbed}</p>}
      </div>

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>③ 装夹刚性（看结构，不看幅度）</b>
          <StatusDot state={att.dot} />
          <b>{att.title}</b>
        </div>
        <p className="cali-muted">{att.detail}</p>
        <p className="cali-muted">
          注：逐姿态 c 的散布 {fmtMm(report.per_pose_c_spread_mm)} 与残差 RMS {fmtMm(report.rms_mm)}{" "}
          是<b>同一个数</b>（c 的最小二乘解正好是逐姿态 c 的均值），所以幅度本身不构成第二个检查。
        </p>
      </div>

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>④ 精度与预测</b>
        </div>
        <div className="cali-metric-row">
          <Metric label="位姿数" value={String(report.n_poses)} />
        </div>
        <p>
          <StatusDot state={sigma.dot} /> bootstrap σ_c：{sigma.text}
          <br />
          <StatusDot state={holdout.dot} /> holdout：{holdout.text}
        </p>
        <p className="cali-muted">
          σ_c 是<b>精度不是准确度</b>：它说这次采集把答案钉得多紧，说不了位姿源有没有偏。
          说不了的那部分就是上面的吸收项，重采样多少次也看不见。
        </p>
      </div>
    </>
  );
}

function StationNumbers({ report }: { report: TrackerStationReport }) {
  const sigma = sigmaVerdict(report.sigma);
  const scale = scaleVerdict(report);
  return (
    <>
      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>② 站位 T_WG</b>
          <span className="cali-muted">{report.tracker_station_id || "未命名站位"}</span>
        </div>
        <div className="cali-metric-row">
          <Metric label="残差 RMS" value={fmtMm(report.rms_mm)} />
          <Metric label="留一最大" value={fmtMm(report.leave_one_out_max_mm)} />
          <Metric label="总位姿数" value={String(report.n_poses_total)} />
        </div>
        <p>
          <StatusDot state={sigma.dot} /> bootstrap σ_c：{sigma.text}
        </p>
        <p className="cali-muted">
          这个 T_WG 是<b>房间的常量</b>，跟踪仪不挪、相机外参不重标就一直有效，后续 session 用 lever-arm
          模式冻结它单独解 c 即可。
        </p>
      </div>

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>③ 尺度诊断（未施加）</b>
          <StatusDot state={scale.dot} />
          <b>{scale.text}</b>
        </div>
        <p className="cali-muted">{scale.detail}</p>
      </div>

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>④ 各 session 的 c 与装夹刚性</b>
        </div>
        <table className="metric-table">
          <thead>
            <tr>
              <th>session</th>
              <th>mount</th>
              <th>|c|</th>
              <th>姿态相关性</th>
            </tr>
          </thead>
          <tbody>
            {report.sessions.map((session) => {
              const att = attitudeVerdict(session.attitude);
              return (
                <tr key={`${session.session_id}/${session.mount_id}`}>
                  <td>{session.session_id || "—"}</td>
                  <td>{session.mount_id || "—"}</td>
                  <td>{fmtMm(session.lever_arm_mm, 1)}</td>
                  <td>
                    <StatusDot state={att.dot} /> {att.title}
                  </td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="cali-muted">
          每个 mount 有自己的 c，不会互相串——跨一次重新装夹的两个姿态杠杆臂不同，池化它们不构成约束。
        </p>
      </div>
    </>
  );
}

function ValidateResult({ result }: { result: TrackerValidateResponse | null }) {
  const verdict = validateVerdict(result);
  if (!verdict) return null;
  const report = result?.report ?? null;
  const summary = report?.summary ?? null;
  const coverage = coverageVerdict(report);
  const timeBase = timeBaseVerdict(report);
  const timing = timingSignal(summary);
  const strata = speedStrataRows(summary);
  const absorbed = absorbedNote(summary?.registration?.absorbed_modes);

  return (
    <>
      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <StatusDot state={verdict.dot} />
          <b>{verdict.title}</b>
        </div>
        <p className="cali-muted">{verdict.detail}</p>
        {result?.reportPath && (
          <p className="cali-muted">
            结果文件：<code>{result.reportPath}</code>（Replay 页那一集会自动显示）
          </p>
        )}
      </div>

      {summary && (
        <div className="cali-result-box">
          <div className="cali-result-box-head">
            <b>残差</b>
            <span className="cali-muted">ep{report?.episode} · {report?.target}</span>
          </div>
          <div className="cali-metric-row">
            <Metric label="p50" value={fmtMm(summary.residual_mm.p50_mm)} />
            <Metric label="p95" value={fmtMm(summary.residual_mm.p95_mm)} />
            <Metric label="插值误差上界" value={fmtMm(summary.interp_error_mm_bound)} />
            <Metric label="旋转敏感度" value={`${summary.rotation_sensitivity_mm_per_deg.toFixed(2)} mm/度`} />
          </div>
          <p>
            <StatusDot state={coverage.dot} /> 配对覆盖率：{coverage.text}
            <br />
            <StatusDot state={timeBase.dot} /> 相机时间基准：{timeBase.text}
          </p>
          {timeBase.detail && <p className="cali-muted">{timeBase.detail}</p>}
          {absorbed && <p className="cali-warn">{absorbed}</p>}
        </div>
      )}

      {strata.length > 0 && (
        <div className="cali-result-box">
          <div className="cali-result-box-head">
            <b>按速度分层：几何项 vs 时间项</b>
            <StatusDot state={timing.dot} />
            <b>{timing.text}</b>
          </div>
          <table className="metric-table">
            <thead>
              <tr>
                <th>速度段</th>
                <th>帧数</th>
                <th>p50</th>
                <th>p95</th>
              </tr>
            </thead>
            <tbody>
              {strata.map((row) => (
                <tr key={row.label}>
                  <td>{row.label}</td>
                  <td>{row.count}</td>
                  <td>{row.p50}</td>
                  <td>{row.p95}</td>
                </tr>
              ))}
            </tbody>
          </table>
          <p className="cali-muted">{timing.detail}</p>
          <p className="cali-muted">
            一个常量时间偏移按 <code>|v|·Δt</code> 走：静止时看不见，随速度线性增长。
            把整条轨迹池化成一个 RMS 正好会把它藏起来——大多数帧都接近静止。
          </p>
        </div>
      )}
    </>
  );
}


/**
 * What the gateway is holding on this page's behalf.
 *
 * ``dwellsOnDisk`` is separated from ``dwellsStarted`` on purpose: a take the
 * operator discarded, and one that never started because the redirect was not
 * acknowledged, both leave the press count ahead of what the solve will read.
 * On 2026-09-21 a whole capture was lost to exactly that gap, invisibly.
 */
function SessionLine({ session }: { session: TrackerMountSession }) {
  const solvable = session.stage === "landed";
  const dot = session.stage === "failed" ? "error" : solvable ? "running" : "warning";
  return (
    <div className="cali-result-box">
      <div className="cali-result-box-head">
        <StatusDot state={dot} />
        <b>{session.sessionName}</b>
        <span className="cali-muted">
          {session.kind === "pivot" ? "pivot（E1p）" : "驻点集"} ·{" "}
          {solvable ? "可解算" : session.stage === "failed" ? "无法解算" : "采集中"}
        </span>
      </div>
      <p className="cali-muted">
        已落盘 <b>{session.dwellsOnDisk}</b> 段
        {session.dwellsStarted !== session.dwellsOnDisk && (
          <>（按了 {session.dwellsStarted} 次——差额是没保存成的）</>
        )}
        {session.kind === "pivot"
          ? "（pivot 按姿态格认证，解算后看结果卡片）。"
          : `，距离认证还差 ${Math.max(0, POSES_TO_CERTIFY - session.dwellsOnDisk)} 段的姿态量。`}
        {session.trackerSessionId && <> 跟踪仪 session：{session.trackerSessionId}。</>}
      </p>
      {session.kind !== "pivot" && (session.shortSegments ?? 0) > 0 && (
        <p className="cali-warn">
          <StatusDot state="warning" /> 有 {session.shortSegments} 段录得太短（&lt; {SEGMENT_MIN_S.toFixed(1)}s），
          解算时会被丢掉，对应的姿态要重录。
        </p>
      )}
      {session.message && <p className="cali-muted">{session.message}</p>}
      {session.landedPath && <p className="cali-muted">落地于 {session.landedPath}</p>}
    </div>
  );
}

/** tm_<ts> for a mount capture (its dataset dir is always "tracker_mount"), else the dataset name. */
function captureGroupLabel(capture: TrackerMountCapture): string {
  if (capture.datasetName !== "tracker_mount") return capture.datasetName;
  const parts = capture.dataset.split("/").filter(Boolean);
  return parts[parts.length - 2] ?? capture.datasetName;
}

function GuidedCapture({
  api,
  disabled,
  running,
  captures,
  live,
  session,
  onRefresh,
  setRunning,
}: {
  api: DataCollectionGuiApi;
  disabled: boolean;
  running: string;
  captures: TrackerMountCapture[];
  live: RecorderLiveState;
  session: TrackerMountSession | undefined;
  onRefresh: () => void;
  setRunning: (
    value: "" | "connect" | "start" | "record" | "save" | "disconnect" | "end",
  ) => void;
}) {
  const [seconds, setSeconds] = useState(String(suggestedDwellSeconds()));
  const [kind, setKind] = useState<"dwell" | "pivot">("dwell");
  // When the take in flight started, so an early "save" can be called out
  // while the rig is still standing where it was.
  const [recordStartedAt, setRecordStartedAt] = useState<number | null>(null);
  const activeKind = session?.active ? session.kind ?? "dwell" : kind;
  const lengthVerdict = segmentLengthVerdict(Number(seconds), activeKind);
  // A dwell segment is one pose (4 s); a pivot segment is a sweep (30 s).
  useEffect(() => {
    setSeconds(String(suggestedDwellSeconds(activeKind)));
  }, [activeKind]);
  // A refused press and a started take are not the same news, and the first one
  // used to be rendered in the same muted grey as the instructions -- which is
  // how a lost take reads as "the button is just disabled".
  const [note, setNote] = useState<{ text: string; bad: boolean } | null>(null);
  const readiness = captureReadiness(captures, live);

  async function onConnect() {
    setRunning("connect");
    await api.connectRecording(undefined, true);
    setRunning("");
  }

  async function onStartSession() {
    setRunning("start");
    const result = await api.startTrackerMountSession(kind);
    setNote(
      result.ok
        ? { text: `站位采集 ${result.session?.sessionName ?? ""} 已开始`, bad: false }
        : { text: result.error || "无法开始站位采集", bad: true },
    );
    setRunning("");
    onRefresh();
  }

  async function onEndSession() {
    // Releases the claim on the recorder. Nothing on disk is touched -- said on
    // the button's own line, because "结束" reads like "discard".
    setRunning("end");
    await api.cancelTrackerMountSession();
    setRunning("");
    onRefresh();
  }

  async function onRecord() {
    setRunning("record");
    const result = await api.recordTrackerMountDwell({
      seconds: Number(seconds) || suggestedDwellSeconds(activeKind),
    });
    const howTo =
      activeKind === "pivot"
        ? "连续扫动，插件别抬离球窝；扫够了可以提前保存"
        : "静止到收尾，别提前保存";
    setNote(
      result.ok
        ? { text: `已开始录制，${result.seconds}s 后自动收尾——${howTo}`, bad: false }
        : { text: result.error || "录制失败：录制器没有接受这一段。", bad: true },
    );
    setRecordStartedAt(result.ok ? Date.now() : null);
    setRunning("");
  }

  async function onSave() {
    // The only way to end a take early and keep it. Without this button the
    // sole control on this panel was Disconnect, which discards.
    setRunning("save");
    const early =
      recordStartedAt == null ? null : earlySaveWarning((Date.now() - recordStartedAt) / 1000, activeKind);
    setRecordStartedAt(null);
    await api.stopRecording("save");
    if (early) setNote({ text: early, bad: true });
    setRunning("");
    onRefresh();
  }

  async function onDisconnect() {
    setRunning("disconnect");
    // Disconnect is what seals and lands the tracker session; until it happens
    // every recorded episode is correct and none of them is readable.
    await api.stopRecording("exit");
    setRunning("");
    onRefresh();
  }

  return (
    <div className="cali-result-box">
      <div className="cali-result-box-head">
        <b>① 采集</b>
        <StatusDot state={readiness.dot} />
        <b>{readiness.title}</b>
      </div>

      <p className="cali-muted">{readiness.detail}</p>

      <div className="cali-op-grid">
        <button
          className="cali-mini-btn"
          disabled={disabled || !readiness.canConnect}
          onClick={onConnect}
        >
          {running === "connect" ? "连接中…" : readiness.canConnect ? "Connect（带跟踪仪）" : "已连接"}
        </button>
        <label className="cali-field">
          采集什么
          <select
            value={activeKind}
            disabled={disabled || Boolean(session?.active)}
            onChange={(e) => setKind(e.target.value as "dwell" | "pivot")}
          >
            <option value="dwell">驻点集（求站位 T_WG 与杠杆臂 c）</option>
            <option value="pivot">pivot · E1p（TCP 卡球窝，测生产 TCP）</option>
          </select>
        </label>
        <button
          className="cali-mini-btn"
          disabled={disabled || !readiness.canStartSession}
          onClick={onStartSession}
        >
          {running === "start" ? "开始中…" : activeKind === "pivot" ? "开始一次 pivot 采集" : "开始一次站位采集"}
        </button>
        <label className="cali-field">
          这一段录多久 (s)
          <input value={seconds} disabled={disabled} onChange={(e) => setSeconds(e.target.value)} />
          {lengthVerdict.level !== "ok" && (
            <span className={lengthVerdict.level === "bad" ? "cali-warn" : "cali-muted"}>{lengthVerdict.text}</span>
          )}
        </label>
        <button
          className="cali-btn-primary"
          disabled={disabled || !readiness.canRecord}
          onClick={onRecord}
        >
          {/* Not "录制中": the press first has to get the recorder to confirm
              where this segment goes, and saying "recording" during a handshake
              that can still be refused is how a refused press reads as a take
              that happened. */}
          {running === "record" ? "正在确认采集目录…" : activeKind === "pivot" ? "录一段扫动" : "录一段停驻姿态"}
        </button>
        <button
          className="cali-btn-primary"
          disabled={disabled || !readiness.canSave}
          onClick={onSave}
        >
          {running === "save" ? "保存中…" : "保存本段"}
        </button>
        <button
          className="cali-mini-btn"
          disabled={disabled || !readiness.canDisconnect}
          onClick={onDisconnect}
        >
          {running === "disconnect" ? "收尾中…" : "Disconnect 并落地 session"}
        </button>
        <button
          className="cali-mini-btn"
          disabled={disabled || !readiness.canEndSession}
          onClick={onEndSession}
        >
          {running === "end" ? "结束中…" : "结束采集（不删数据）"}
        </button>
      </div>
      {note && (
        <p className={note.bad ? "cali-warn" : "cali-muted"}>
          {note.bad && <StatusDot state="error" />} {note.text}
        </p>
      )}

      {activeKind === "pivot" ? (
        <p className="cali-muted">
          <b>pivot（E1p）</b>：夹爪夹住相机 pivot 标定用的同一个插件，球头卡进固定在桌上的球窝，<b>SMR 留在钢片上不动</b>。
          <b>连续扫动，不用停</b>：插件在窝里时每个跟踪仪点、每一帧都算数，手抖不要紧；跟踪仪看到 SMR 离开球面就判为抬起，
          那一段不比较。以球窝为支点转：<b>绕光束方向侧倾 ±45–60°</b>（不受接受角限制，它决定球窝中心能定多准），
          前后倾留在 ±25° 内。一段 {PIVOT_SWEEP_SUGGESTED_S}s 左右、扫够了可提前保存；认证按 5° 姿态格数，要 ≥{" "}
          {POSES_TO_CERTIFY} 格。左右两侧各一场，和驻点集用<b>同一个跟踪仪站位</b>。
        </p>
      ) : (
        <>
          <p className="cali-muted">
            <b>一段一个姿态。</b>摆好、手离开，点「录一段」，<b>静止到自动收尾</b>。一个驻点要 {DWELL_SECONDS_MIN}s 静止，
            两端各裁 {DWELL_TRIM_S}s，所以每段至少 {SEGMENT_MIN_S.toFixed(1)}s——提前「保存本段」就会被丢掉。
            拟合要 ≥ <b>{POSES_TO_CERTIFY}</b> 段，要留出复核就录 <b>{SEGMENTS_SUGGESTED}</b> 段。
          </p>
          <p className="cali-muted">
            换姿态时<b>要绕两根明显不平行的轴</b>：纯平移和单轴旋转都定不出 c，而且残差照样很小。
            位置也要在三维里拉开（左右、前后、高低），停的时候要真停住（&lt;2 mm/s）。
          </p>
        </>
      )}
      <p className="cali-muted">
        跟踪仪 session 是<b>一次 Connect 一个</b>（logger 冷启动要 15–16 s，不可能每段重来），
        并且在 <b>Disconnect 时才 seal + land</b>。所以顺序是：
        Connect → 开始站位采集 → 录若干段 → Disconnect → 解算。
      </p>
      {session?.active && <SessionLine session={session} />}

      {captures.length > 0 && (
        <CaptureHistory api={api} disabled={disabled} captures={captures} onRefresh={onRefresh} />
      )}
    </div>
  );
}


/**
 * The recorded list, with batch delete. Deletion is the gateway's call: it
 * plans the whole batch and refuses all of it if one row cannot go (a capture
 * in progress, an un-landed session, part of a recorder dataset), so what the
 * operator selected either goes or stays -- never half.
 */
function CaptureHistory({
  api,
  disabled,
  captures,
  onRefresh,
}: {
  api: DataCollectionGuiApi;
  disabled: boolean;
  captures: TrackerMountCapture[];
  onRefresh: () => void;
}) {
  const [selected, setSelected] = useState<Set<string>>(new Set());
  const [confirming, setConfirming] = useState(false);
  const [deleting, setDeleting] = useState(false);
  const [note, setNote] = useState<{ text: string; bad: boolean } | null>(null);

  // A refreshed list may no longer hold a selected row; never delete what is not shown.
  useEffect(() => {
    const shown = new Set(captures.map((capture) => capture.episodeDir));
    setSelected((prev) => {
      const kept = [...prev].filter((dir) => shown.has(dir));
      return kept.length === prev.size ? prev : new Set(kept);
    });
  }, [captures]);

  const allSelected = captures.length > 0 && captures.every((capture) => selected.has(capture.episodeDir));
  const picked = captures.filter((capture) => selected.has(capture.episodeDir));
  const groups = new Map<string, number>();
  for (const capture of picked) {
    const key = captureGroupLabel(capture);
    groups.set(key, (groups.get(key) ?? 0) + 1);
  }

  function toggle(dir: string) {
    setSelected((prev) => {
      const next = new Set(prev);
      if (next.has(dir)) next.delete(dir);
      else next.add(dir);
      return next;
    });
  }

  function toggleAll() {
    setSelected(allSelected ? new Set() : new Set(captures.map((capture) => capture.episodeDir)));
  }

  async function onDelete() {
    setConfirming(false);
    setDeleting(true);
    const result = await api.deleteTrackerMountCaptures(picked.map((capture) => capture.episodeDir));
    setDeleting(false);
    if (result.ok) {
      const streams = result.removedStreams?.length ?? 0;
      setNote({
        text: `已删除 ${result.deleted ?? picked.length} 段录制${streams ? `，以及 ${streams} 个不再被引用的跟踪仪数据流` : ""}。`,
        bad: false,
      });
      setSelected(new Set());
    } else {
      setNote({ text: result.error || "删除失败", bad: true });
    }
    onRefresh();
  }

  return (
    <>
      <div className="cali-op-grid tracker-capture-actions">
        <span className="cali-muted">已录制 {captures.length} 段，选中 {picked.length} 段</span>
        <button
          className="cali-mini-btn danger"
          disabled={disabled || deleting || picked.length === 0}
          onClick={() => setConfirming(true)}
        >
          {deleting ? "删除中…" : `删除选中（${picked.length}）`}
        </button>
      </div>
      {note && (
        <p className={note.bad ? "cali-warn" : "cali-muted"}>
          {note.bad && <StatusDot state="error" />} {note.text}
        </p>
      )}
      <div className="tracker-capture-scroll">
        <table className="metric-table">
          <thead>
            <tr>
              <th>
                <input
                  type="checkbox"
                  aria-label="全选"
                  checked={allSelected}
                  disabled={disabled || deleting}
                  onChange={toggleAll}
                />
              </th>
              <th>录制</th>
              <th>session</th>
              <th>落地</th>
              <th>光束有效</th>
            </tr>
          </thead>
          <tbody>
            {captures.map((capture) => (
              <tr key={capture.episodeDir} onClick={() => !disabled && !deleting && toggle(capture.episodeDir)}>
                <td>
                  <input
                    type="checkbox"
                    aria-label={`选中 ${captureGroupLabel(capture)} ep${capture.episode}`}
                    checked={selected.has(capture.episodeDir)}
                    disabled={disabled || deleting}
                    onClick={(e) => e.stopPropagation()}
                    onChange={() => toggle(capture.episodeDir)}
                  />
                </td>
                <td>
                  {captureGroupLabel(capture)} ep{capture.episode}
                  {protocolLabel(capture) && <span className="cali-muted"> · {protocolLabel(capture)}</span>}
                </td>
                <td>{capture.sessionId || "—"}</td>
                <td>
                  <StatusDot state={capture.landed ? "running" : "warning"} />
                  {capture.landed ? "已落地" : "待 Disconnect"}
                </td>
                <td>
                  {capture.beamValidFraction < 0
                    ? "—"
                    : `${(capture.beamValidFraction * 100).toFixed(0)}%`}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      {confirming && (
        <Modal
          title="删除选中的录制"
          onClose={() => setConfirming(false)}
          footer={
            <>
              <button className="cali-mini-btn" onClick={() => setConfirming(false)}>
                取消
              </button>
              <button className="cali-mini-btn danger" onClick={onDelete}>
                永久删除 {picked.length} 段
              </button>
            </>
          }
        >
          <p>
            将从 Thor 磁盘上<b>永久删除</b>下面这些录制，不能恢复：
          </p>
          <ul>
            {[...groups.entries()].map(([label, count]) => (
              <li key={label}>
                {label}：{count} 段
              </li>
            ))}
          </ul>
          <p className="cali-muted">
            剩下的录制<b>保留原编号</b>（已有的解算结果按编号引用 episode，重编号会把它们指到别的姿态上）。
            某次采集的录制全删光时整个采集目录一起删；某个跟踪仪数据流不再被任何录制引用时也一起删。
            已经解出来的 T_WG / c 结果文件不受影响。
          </p>
        </Modal>
      )}
    </>
  );
}

export function TrackerMountPanel({
  api,
  busy,
  snapshot,
}: {
  api: DataCollectionGuiApi;
  busy: boolean;
  snapshot: GuiSnapshot;
}) {
  // The same fields Live Record renders. Derived here rather than re-probed, so
  // the two pages cannot answer "is the tracker up" differently.
  const recording = snapshot.recording;
  const recorderState = String(recording?.state ?? "idle");
  // Held by the gateway, not by this component. A reload used to mint a new
  // session name and orphan the dwells already recorded under the old one.
  const mountSession = snapshot.trackerMountSession;
  const live: RecorderLiveState = {
    connected: !["idle", "error"].includes(recorderState),
    trackerEnabled: Boolean(recording?.laserTracker),
    trackerReady: Boolean(recording?.laserTrackerReady),
    trackerDetail: String(recording?.laserTrackerDetail ?? ""),
    trackerBeamBroken: Boolean(recording?.laserTrackerBeamBroken),
    // "review" counts: an episode waiting for save/discard is still a take that
    // Disconnect would throw away.
    episodeInFlight: ["recording", "review", "saving", "discarding"].includes(recorderState),
    sessionActive: Boolean(mountSession?.active),
    sessionName: String(mountSession?.sessionName ?? ""),
  };
  const [showUsage, setShowUsage] = useState(false);
  const [stationRows, setStationRows] = useState<TrackerMountCaptureRow[]>([{ ...EMPTY_ROW }]);
  const [leverRows, setLeverRows] = useState<TrackerMountCaptureRow[]>([{ ...EMPTY_ROW }]);
  const [stationPath, setStationPath] = useState("");
  const [holdout, setHoldout] = useState("5");
  const [worldFrameId, setWorldFrameId] = useState("");
  const [trackerStationId, setTrackerStationId] = useState("");
  const [running, setRunning] = useState<"" | "station" | "lever" | "pivot" | "validate">("");
  const [result, setResult] = useState<TrackerMountSolveResponse | null>(null);
  const [gtDataset, setGtDataset] = useState("");
  const [gtEpisode, setGtEpisode] = useState("");
  const [gtSession, setGtSession] = useState("");
  const [gtMountFit, setGtMountFit] = useState("");
  // Blank means "use the recorder's own constant", resolved gateway-side. An
  // explicit value is how the exposure sign gets measured: run -0.5 / 0 / +0.5
  // and keep the one that flattens the speed-stratified residual.
  const [gtFraction, setGtFraction] = useState("");
  const [gtResult, setGtResult] = useState<TrackerValidateResponse | null>(null);
  const [captures, setCaptures] = useState<TrackerMountCapture[]>([]);
  const [captureRunning, setCaptureRunning] = useState<
    "" | "connect" | "start" | "record" | "save" | "disconnect" | "end"
  >("");
  const [chainMode, setChainMode] = useState<"station" | "lever-arm" | "pivot">("station");
  const [pivotCube, setPivotCube] = useState("left");
  const [pivotBundle, setPivotBundle] = useState("");
  const [pivotMountFit, setPivotMountFit] = useState("");
  const [chainPicked, setChainPicked] = useState<string[]>([]);
  const [chainGt, setChainGt] = useState("");
  const [chainResult, setChainResult] = useState<TrackerMountChainResponse | null>(null);
  const [showManual, setShowManual] = useState(false);
  const [artifacts, setArtifacts] = useState<{ stations: TrackerMountArtifact[]; mounts: TrackerMountArtifact[] }>({
    stations: [],
    mounts: [],
  });

  const disabled = busy || running !== "" || captureRunning !== "";
  const readiness = captureReadiness(captures, live);

  async function refresh() {
    const payload = await api.fetchTrackerMount();
    setArtifacts({ stations: payload.stations ?? [], mounts: payload.mounts ?? [] });
    // Opening on the newest station is the common case: a station is meant to be
    // reused, and retyping its path is how a stale one gets picked by accident.
    if (!stationPath && payload.stations?.length) setStationPath(payload.stations[0].path);
    if (!gtMountFit && payload.mounts?.length) setGtMountFit(payload.mounts[0].path);
    const found = await api.fetchTrackerMountCaptures();
    setCaptures(found.episodes ?? []);
  }

  /** A discovered capture becomes a capture row with no path typing at all. */
  function rowOf(capture: TrackerMountCapture): TrackerMountCaptureRow {
    return {
      session: capture.sessionPath,
      dataset: capture.dataset,
      episode: String(capture.episode),
      // One bolting-on of the plate is one mount id, and one Connect is one
      // tracker session -- so the session id is the honest default, and a
      // re-mount inside one Connect is the case the operator has to override.
      mountId: capture.sessionId || capture.datasetName,
      sessionId: capture.sessionId,
    };
  }

  async function onChain() {
    setRunning(chainMode === "station" ? "station" : chainMode === "pivot" ? "pivot" : "lever");
    const picked = readiness.usable.filter((c) => chainPicked.includes(c.episodeDir));
    // A pivot is static at the TCP: there is no trajectory to grade against it.
    const gt = chainMode === "pivot" ? undefined : readiness.usable.find((c) => c.episodeDir === chainGt);
    setChainResult(
      await api.runTrackerMountChain({
        mode: chainMode,
        rows: picked.map(rowOf),
        station: chainMode === "station" ? undefined : stationPath,
        holdout: chainMode === "lever-arm" ? Number(holdout) || 0 : undefined,
        cube: chainMode === "pivot" ? pivotCube : undefined,
        markerTcp: chainMode === "pivot" ? pivotBundle.trim() : undefined,
        mountFit: chainMode === "pivot" ? pivotMountFit : undefined,
        worldFrameId: worldFrameId.trim(),
        trackerStationId: trackerStationId.trim(),
        validate: gt
          ? {
              dataset: gt.dataset,
              episode: gt.episode,
              session: gt.sessionPath,
              mountFit: chainMode === "station" ? gtMountFit : "",
              exposureFraction: gtFraction.trim() === "" ? "" : Number(gtFraction),
            }
          : undefined,
      }),
    );
    setRunning("");
    void refresh();
  }

  useEffect(() => {
    void refresh();
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  // What lands on disk is not what a button press returns: a take auto-saves
  // seconds after "record" answered, and the tracker stream lands some time
  // after Disconnect. Refreshing only on presses left the list a step behind
  // for good (2026-09-24: 20 saved and landed dwells, none of them pickable).
  // The snapshot poll does see those changes, so the list follows it.
  const diskSignature = [
    mountSession?.sessionName ?? "",
    mountSession?.dwellsOnDisk ?? 0,
    mountSession?.stage ?? "",
    mountSession?.landedPath ?? "",
    recorderState,
  ].join("|");
  useEffect(() => {
    let stale = false;
    void api.fetchTrackerMountCaptures().then((found) => {
      if (!stale && found.ok) setCaptures(found.episodes ?? []);
    });
    return () => {
      stale = true;
    };
  }, [api, diskSignature]);

  async function onStation() {
    setRunning("station");
    setResult(
      await api.runTrackerMountStation({
        rows: stationRows,
        worldFrameId: worldFrameId.trim(),
        trackerStationId: trackerStationId.trim(),
      }),
    );
    setRunning("");
    void refresh();
  }

  async function onLeverArm() {
    setRunning("lever");
    setResult(
      await api.runTrackerMountLeverArm({
        rows: leverRows,
        station: stationPath,
        holdout: Number(holdout) || 0,
      }),
    );
    setRunning("");
    void refresh();
  }

  async function onValidate() {
    setRunning("validate");
    setGtResult(
      await api.runTrackerValidate({
        dataset: gtDataset,
        episode: gtEpisode,
        session: gtSession,
        mountFit: gtMountFit,
        exposureFraction: gtFraction.trim() === "" ? "" : Number(gtFraction),
      }),
    );
    setRunning("");
  }

  const report = (result?.report ?? null) as TrackerStationReport | TrackerMountReport | null;
  const isMount = result?.kind === "lever_arm";

  return (
    <section className="panel">
      <div className="cali-result-head">
        <h3>跟踪仪站位 T_WG + SMR 杠杆臂 c</h3>
        <div className="cali-op-actions">
          <span className="cali-preview-badge">不认证 marker→TCP</span>
          <button className="cali-mini-btn" onClick={() => setShowUsage(true)}>
            使用说明
          </button>
        </div>
      </div>

      <p className="cali-muted">
        <b>作用：</b>把跟踪仪系 G 和相机 world 系 W 注册起来，并解出 SMR 球心在 rig 本体系里的位置 c。
        两个都不能从 CAD 读——T_WG 取决于跟踪仪站在哪，c 取决于这次把板子拧在哪。
        <b>采集要的是姿态多样性不是位置</b>：纯平移的一对姿态对 c 完全没有贡献。
      </p>

      <GuidedCapture
        api={api}
        disabled={disabled}
        running={captureRunning}
        captures={captures}
        live={live}
        session={mountSession}
        onRefresh={() => void refresh()}
        setRunning={setCaptureRunning}
      />

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>② 一键解算</b>
          <span className="cali-muted">选刚录的段，不用填路径</span>
        </div>

        <div className="cali-op-grid">
          <label className="cali-field">
            模式
            <select
              value={chainMode}
              disabled={disabled}
              onChange={(e) => setChainMode(e.target.value as "station" | "lever-arm" | "pivot")}
            >
              <option value="station">station（连 T_WG 一起解；需要姿态多样性）</option>
              <option value="lever-arm">lever-arm（站位已冻结，只解 c；不需要旋转）</option>
              <option value="pivot">pivot · E1p（在 TCP 处测生产 c_TCP；只认位置）</option>
            </select>
          </label>
          {chainMode === "pivot" && (
            <>
              <label className="cali-field">
                哪个 cube
                <select value={pivotCube} disabled={disabled} onChange={(e) => setPivotCube(e.target.value)}>
                  <option value="left">left</option>
                  <option value="right">right</option>
                </select>
              </label>
              <label className="cali-field">
                marker→TCP bundle（留空 = 生产配置里那份）
                <input value={pivotBundle} disabled={disabled} onChange={(e) => setPivotBundle(e.target.value)} placeholder="留空" />
              </label>
              <label className="cali-field">
                同一钢片的 lever-arm 结果（可选，做半径一致性检查）
                <select value={pivotMountFit} disabled={disabled} onChange={(e) => setPivotMountFit(e.target.value)}>
                  <option value="">不做</option>
                  {artifacts.mounts.map((item) => (
                    <option key={item.path} value={item.path}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
            </>
          )}
          {chainMode !== "station" && (
            <>
              <label className="cali-field">
                已冻结的 station
                <select value={stationPath} disabled={disabled} onChange={(e) => setStationPath(e.target.value)}>
                  <option value="">选一个…</option>
                  {artifacts.stations.map((item) => (
                    <option key={item.path} value={item.path}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
              {chainMode === "lever-arm" && (
                <label className="cali-field">
                  holdout 位姿数
                  <input value={holdout} disabled={disabled} onChange={(e) => setHoldout(e.target.value)} />
                </label>
              )}
            </>
          )}
          <label className="cali-field">
            顺便做 GT 比较（可不选；pivot 不做）
            <select value={chainGt} disabled={disabled || chainMode === "pivot"} onChange={(e) => setChainGt(e.target.value)}>
              <option value="">不做</option>
              {readiness.usable.map((capture) => (
                <option key={capture.episodeDir} value={capture.episodeDir}>
                  {captureLabel(capture)}
                </option>
              ))}
            </select>
          </label>
          <label className="cali-field">
            曝光系数（留空 = 跟随录制器）
            <input
              value={gtFraction}
              disabled={disabled}
              onChange={(e) => setGtFraction(e.target.value)}
              placeholder="留空"
            />
          </label>
        </div>

        <div className="cali-result-box">
          <div className="cali-result-box-head">
            <b>用哪几段来拟合</b>
            <span className="cali-muted">
              {chainMode === "station"
                ? "可多选，跨 session 累积"
                : "选同一个 mount 的所有段（一段一个姿态）"}
            </span>
          </div>
          {readiness.usable.length === 0 && <p className="cali-muted">还没有可用的录制。</p>}
          {readiness.usable.map((capture) => (
            <label key={capture.episodeDir} className="cali-field">
              <input
                type="checkbox"
                name="tracker-mount-pick"
                disabled={disabled}
                checked={chainPicked.includes(capture.episodeDir)}
                onChange={(e) =>
                  setChainPicked(
                    e.target.checked
                      ? [...chainPicked, capture.episodeDir]
                      : chainPicked.filter((dir) => dir !== capture.episodeDir),
                  )
                }
              />
              {captureLabel(capture)}
              {protocolLabel(capture) && <span className="cali-muted"> · {protocolLabel(capture)}</span>}
            </label>
          ))}
        </div>

        <button
          className="cali-btn-primary"
          disabled={
            disabled ||
            chainPicked.length === 0 ||
            (chainMode !== "station" && !stationPath) ||
            (chainMode === "pivot" && !pivotCube)
          }
          onClick={onChain}
        >
          {running !== "" ? "解算中…" : chainGt && chainMode !== "pivot" ? "解算 + GT 比较" : "解算"}
        </button>

        <p className="cali-muted">
          拟合和 GT 比较是<b>串起来跑，不是合成一个</b>：拟合必须用<b>不是被评估那条轨迹</b>的停驻姿态，
          所以这里把拟合的产物传给比较，而不是让两步共享状态。拟合没通过就不会去做比较——
          拿一个被拒绝的注册去给轨迹打分，正是把"拒绝"变回"数字"的那条路。
        </p>

        <button className="cali-mini-btn" onClick={() => setShowManual(!showManual)}>
          {showManual ? "收起手填路径" : "手填路径（逃生口）"}
        </button>
      </div>

      {showManual && (
        <>
          <div className="cali-result-box">
            <div className="cali-result-box-head">
              <b>A. 站位 station（手填）</b>
            </div>
            <CaptureRows rows={stationRows} disabled={disabled} onChange={setStationRows} />
            <div className="cali-op-grid">
              <label className="cali-field">
                world frame id
                <input value={worldFrameId} disabled={disabled} onChange={(e) => setWorldFrameId(e.target.value)} />
              </label>
              <label className="cali-field">
                tracker station id
                <input
                  value={trackerStationId}
                  disabled={disabled}
                  onChange={(e) => setTrackerStationId(e.target.value)}
                />
              </label>
              <button className="cali-btn-primary" disabled={disabled} onClick={onStation}>
                {running === "station" ? "解算中…" : "解 T_WG"}
              </button>
            </div>
          </div>

          <div className="cali-result-box">
            <div className="cali-result-box-head">
              <b>B. 杠杆臂 lever-arm（手填）</b>
            </div>
            <CaptureRows rows={leverRows} disabled={disabled} onChange={setLeverRows} single />
            <div className="cali-op-grid">
              <label className="cali-field">
                station JSON
                <select value={stationPath} disabled={disabled} onChange={(e) => setStationPath(e.target.value)}>
                  <option value="">选一个…</option>
                  {artifacts.stations.map((item) => (
                    <option key={item.path} value={item.path}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
              <button className="cali-btn-primary" disabled={disabled || !stationPath} onClick={onLeverArm}>
                {running === "lever" ? "解算中…" : "解 c"}
              </button>
            </div>
          </div>

          <div className="cali-result-box">
            <div className="cali-result-box-head">
              <b>C. GT 比较（手填）</b>
            </div>
            <div className="cali-op-grid">
              <label className="cali-field">
                数据集目录
                <input value={gtDataset} disabled={disabled} onChange={(e) => setGtDataset(e.target.value)} />
              </label>
              <label className="cali-field">
                episode
                <input value={gtEpisode} disabled={disabled} onChange={(e) => setGtEpisode(e.target.value)} />
              </label>
              <label className="cali-field">
                tracker session 目录
                <input value={gtSession} disabled={disabled} onChange={(e) => setGtSession(e.target.value)} />
              </label>
              <label className="cali-field">
                mount-fit JSON
                <select value={gtMountFit} disabled={disabled} onChange={(e) => setGtMountFit(e.target.value)}>
                  <option value="">不用（只有形状，什么都不认证）</option>
                  {artifacts.mounts.map((item) => (
                    <option key={item.path} value={item.path}>
                      {item.name}
                    </option>
                  ))}
                </select>
              </label>
              <button
                className="cali-btn-primary"
                disabled={disabled || !gtDataset || !gtSession}
                onClick={onValidate}
              >
                {running === "validate" ? "比较中…" : "跑 GT 比较"}
              </button>
            </div>
          </div>
        </>
      )}

      <p className="cali-muted">
        <b>GT 比较不碰 cube→TCP。</b>比的是球心：相机侧 <code>R·c + t</code>，跟踪仪侧{" "}
        <code>R_WG·p_G + t_WG</code>，TCP 一次都没出现。而杠杆臂拟合已经吸收了本体系的常量平移和常量旋转——
        那两项加起来<b>就是</b> marker→TCP 常量，所以残差再小也<b>不能</b>反过来证明 cube→TCP 是对的。
      </p>

      <Verdict result={result} />
      <Observability report={report} />
      {report && isMount && <MountNumbers report={report as TrackerMountReport} />}
      {report && !isMount && <StationNumbers report={report as TrackerStationReport} />}

      <div className="cali-result-box">
        <div className="cali-result-box-head">
          <b>C. GT 比较 validate</b>
          <span className="cali-muted">拿上面解出的 T_WG + c，去和跟踪仪比一条真实轨迹</span>
        </div>
        <div className="cali-op-grid">
          <label className="cali-field">
            数据集目录
            <input
              value={gtDataset}
              disabled={disabled}
              onChange={(e) => setGtDataset(e.target.value)}
              placeholder="outputs/datasets/box_umi_rig0818"
            />
          </label>
          <label className="cali-field">
            episode
            <input value={gtEpisode} disabled={disabled} onChange={(e) => setGtEpisode(e.target.value)} />
          </label>
          <label className="cali-field">
            tracker session 目录
            <input
              value={gtSession}
              disabled={disabled}
              onChange={(e) => setGtSession(e.target.value)}
              placeholder="outputs/laser_tracker/20260921_a"
            />
          </label>
          <label className="cali-field">
            mount-fit JSON
            <select value={gtMountFit} disabled={disabled} onChange={(e) => setGtMountFit(e.target.value)}>
              <option value="">不用（只有形状，什么都不认证）</option>
              {artifacts.mounts.map((item) => (
                <option key={item.path} value={item.path}>
                  {item.name}
                </option>
              ))}
            </select>
          </label>
          <label className="cali-field">
            曝光系数（留空 = 跟随录制器）
            <input
              value={gtFraction}
              disabled={disabled}
              onChange={(e) => setGtFraction(e.target.value)}
              placeholder="留空"
            />
          </label>
          <button className="cali-btn-primary" disabled={disabled || !gtDataset || !gtSession} onClick={onValidate}>
            {running === "validate" ? "比较中…" : "跑 GT 比较"}
          </button>
        </div>
        <p className="cali-muted">
          <b>这一步不碰 cube→TCP。</b>比的是球心：相机侧 <code>R·c + t</code>，跟踪仪侧 <code>R_WG·p_G + t_WG</code>，
          TCP 一次都没出现。而且杠杆臂拟合已经吸收了本体系的常量平移和常量旋转——那两项加起来<b>就是</b>
          marker→TCP 常量，所以残差再小也<b>不能</b>反过来证明 cube→TCP 是对的。能判的是随时间/姿态变化的那部分。
        </p>
        <p className="cali-muted">
          曝光系数留空会跟随录制器的 <code>EXPOSURE_CENTER_FRACTION</code>（现在是 0.0），
          而不是 CLI 自己的默认 0.5——用 0.5 打分等于给一条<b>从来没被生产出来过</b>的轨迹打分。
          想定曝光符号就填 −0.5 / 0 / +0.5 各跑一遍，看哪个把下面「按速度分层」那一项压平。
        </p>
        {gtResult?.episodeDir && (
          <p className="cali-muted">
            自动带上了 episode 目录 <code>{gtResult.episodeDir}</code>，相机时刻走硬件 SOF sidecar。
          </p>
        )}
      </div>

      {chainResult?.fit && <Verdict result={chainResult.fit} />}
      {chainResult?.fit?.report && (
        <Observability report={chainResult.fit.report as TrackerStationReport | TrackerMountReport} />
      )}
      {chainResult?.fit?.kind === "lever_arm" && chainResult.fit.report && (
        <MountNumbers report={chainResult.fit.report as TrackerMountReport} />
      )}
      {chainResult?.fit?.kind === "station" && chainResult.fit.report && (
        <StationNumbers report={chainResult.fit.report as TrackerStationReport} />
      )}
      {chainResult?.fit?.kind === "pivot" && chainResult.fit.report && (
        <TrackerPivotResult report={chainResult.fit.report as TrackerPivotReport} />
      )}
      <ValidateResult result={chainResult?.validate ?? gtResult} />

      {showUsage && <UsageModal onClose={() => setShowUsage(false)} />}
    </section>
  );
}
