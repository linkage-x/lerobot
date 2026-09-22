// One E1p result card, drawn wherever an E1p is run: the tracker-mount panel
// and the marker->TCP panel (the camera pivot's samples, recorded with the
// tracker on). Same artifact, so one rendering of it.
import type { TrackerPivotReport, TrackerPivotSampling } from "../types";
import { Metric, StatusDot } from "../shared/ui";
import { cannotSeeLabel, fmtMm, pivotVerdict } from "./trackerMount";

const mm = (v: number[]) => `[${v.map((x) => x.toFixed(2)).join(", ")}]`;

export function TrackerPivotResult({ report }: { report: TrackerPivotReport }) {
  const verdict = pivotVerdict(report);
  const sphere = report.sphere;
  const radiusGap = report.radius_check_mm?.difference_mm;
  const skipped = [...(report.episodes_skipped ?? []), ...(report.episodes_without_dwells ?? [])];
  const continuous = sphere?.continuous;
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
        <Metric
          label={report.sampling ? "TCP 误差 p95（按姿态格）" : "静态 TCP 误差 p95"}
          value={fmtMm(report.static_tcp_error_mm.p95, 2)}
        />
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
          {continuous && (
            <>
              {" "}球面用了 {continuous.n_points_seated} / {continuous.n_points} 个跟踪仪点
              （{continuous.n_direction_cells} 个 1° 方向格等权）；离球面超过 {fmtMm(continuous.max_radial_mm, 2)} 的算插件抬起，
              占 {((continuous.lifted_fraction ?? 0) * 100).toFixed(1)}%。
            </>
          )}
        </p>
      )}
      {report.sampling && <SamplingLine sampling={report.sampling} />}
      {radiusGap != null && (
        <p className={Math.abs(radiusGap) > 0.5 ? "cali-warn" : "cali-muted"}>
          半径一致性：跟踪仪给的 ρ 与相机侧 |球窝 − c| 相差 {fmtMm(radiusGap, 2)}
          {Math.abs(radiusGap) > 0.5 ? "——站位或那份 lever-arm 有问题，先别引用。" : "。"}
        </p>
      )}
      {skipped.length > 0 && (
        <p className="cali-muted">
          {skipped.length} 段样本没进比较（没解出位姿、全程断光或没停够）：
          {skipped.map((item) => `ep${item.episode}`).join("、")}。
        </p>
      )}
      <p className="cali-warn">这一步看不见：{report.cannot_see.map(cannotSeeLabel).join("；")}。</p>
    </div>
  );
}

/**
 * The continuous pivot's own numbers: how many frames were compared, the
 * per-frame error (the card's p95 above is per attitude), the same error by how
 * fast the gripper was turning, and how much of a lift the seat gate can see.
 */
function SamplingLine({ sampling }: { sampling: TrackerPivotSampling }) {
  const pf = sampling.per_frame_error_mm;
  const lift = sampling.lift_sensitivity;
  const bins = sampling.by_smr_speed.filter((b) => b.n > 0);
  const range = (b: TrackerPivotSampling["by_smr_speed"][number]) =>
    b.smr_speed_mm_s[1] == null ? `≥${b.smr_speed_mm_s[0]}` : `${b.smr_speed_mm_s[0]}–${b.smr_speed_mm_s[1]}`;
  return (
    <>
      <p className="cali-muted">
        连续采样：{sampling.n_frames_seated} / {sampling.n_frames} 帧在窝内，合成 {sampling.n_attitudes} 个
        {sampling.attitude_cell_deg}° 姿态格（认证按姿态格数，不按帧数）。
        {pf && <> 逐帧误差 p95 {fmtMm(pf.p95, 2)}，max {fmtMm(pf.max, 2)}。</>}
      </p>
      {bins.length > 0 && (
        <p className="cali-muted">
          按 SMR 速度（mm/s）：
          {bins.map((b) => `${range(b)}：${b.n} 帧 p95 ${fmtMm(b.p95, 2)}`).join("；")}。速度越高误差越大，多出来的是动态项。
        </p>
      )}
      {lift && (
        <p className={lift.median < 0.2 ? "cali-warn" : "cali-muted"}>
          抬起检测灵敏度 {lift.median.toFixed(2)}（沿插件方向抬起 1 mm，径向残差变 {lift.median.toFixed(2)} mm）
          {lift.median < 0.2 ? "——几乎看不见抬起，这次结果不能引用。" : "。"}
        </p>
      )}
    </>
  );
}
