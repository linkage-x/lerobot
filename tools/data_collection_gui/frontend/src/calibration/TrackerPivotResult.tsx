// One E1p result card, drawn wherever an E1p is run: the tracker-mount panel
// (parked pivot dwells) and the marker->TCP panel (the camera pivot's samples,
// recorded with the tracker on). Same artifact, so one rendering of it.
import type { TrackerPivotReport } from "../types";
import { Metric, StatusDot } from "../shared/ui";
import { cannotSeeLabel, fmtMm, pivotVerdict } from "./trackerMount";

const mm = (v: number[]) => `[${v.map((x) => x.toFixed(2)).join(", ")}]`;

export function TrackerPivotResult({ report }: { report: TrackerPivotReport }) {
  const verdict = pivotVerdict(report);
  const sphere = report.sphere;
  const radiusGap = report.radius_check_mm?.difference_mm;
  const skipped = report.episodes_without_dwells ?? [];
  return (
    <div className="cali-result-box">
      <div className="cali-result-box-head">
        {verdict && <StatusDot state={verdict.dot} />}
        <b>{verdict?.title ?? "E1p"}</b>
        <span className="cali-muted">
          {report.cube} · {report.mount_id} · {report.n_poses} 个姿态
        </span>
      </div>
      {verdict && <p className="cali-muted">{verdict.detail}</p>}
      <div className="cali-metric-row">
        <Metric label="静态 TCP 误差 p95" value={fmtMm(report.static_tcp_error_mm.p95, 2)} />
        <Metric label="max" value={fmtMm(report.static_tcp_error_mm.max, 2)} />
        <Metric label="c_TCP 常量误差" value={fmtMm(report.c_tcp_error_norm_mm, 2)} />
        <Metric label="球窝中心 σ" value={fmtMm(sphere?.center_sigma_norm_mm, 3)} />
      </div>
      <p className="cali-muted">
        c_TCP（cube 系）：生产 {mm(report.c_tcp_production_mm)} mm，跟踪仪实测 {mm(report.c_tcp_measured_mm)} mm，
        差 {mm(report.c_tcp_error_mm)} mm。
        {report.split && (
          <>
            {" "}拆开看：cube 系常量 {mm(report.split.cube_frame_constant_mm)} mm（标定常量的错），world 系常量{" "}
            {mm(report.split.world_frame_constant_mm)} mm（多半来自站位），其余随姿态 RMS{" "}
            {fmtMm(report.split.pose_dependent_rms_mm, 2)}。
          </>
        )}
      </p>
      {report.smr_to_tcp_cube_mm && (
        <p className="cali-muted">
          SMR → TCP（cube 系，TCP 端由跟踪仪定）：<b>{mm(report.smr_to_tcp_cube_mm)} mm</b>，
          |·| = {fmtMm(report.smr_to_tcp_norm_mm, 2)}。轨迹比较用它把球心搬到 TCP；它指向球窝中心，
          离真正的 TCP 还差一个偏距 d。
        </p>
      )}
      {sphere && (
        <p className="cali-muted">
          球面半径 {fmtMm(sphere.radius_mm, 1)}，弱方向增益 {sphere.gain_min.toFixed(3)}
          （绕光束方向侧倾越大越高），弱方向 σ {fmtMm(sphere.center_sigma_weak_mm, 3)}。
        </p>
      )}
      {radiusGap != null && (
        <p className={Math.abs(radiusGap) > 0.5 ? "cali-warn" : "cali-muted"}>
          半径一致性：跟踪仪给的 ρ 与相机侧 |球窝 − c| 相差 {fmtMm(radiusGap, 2)}
          {Math.abs(radiusGap) > 0.5 ? "——站位或那份 lever-arm 有问题，先别引用。" : "。"}
        </p>
      )}
      {skipped.length > 0 && (
        <p className="cali-muted">
          {skipped.length} 段样本里没有够长的停顿，没进拟合：
          {skipped.map((item) => `ep${item.episode}`).join("、")}。
        </p>
      )}
      <p className="cali-warn">这一步看不见：{report.cannot_see.map(cannotSeeLabel).join("；")}。</p>
    </div>
  );
}
