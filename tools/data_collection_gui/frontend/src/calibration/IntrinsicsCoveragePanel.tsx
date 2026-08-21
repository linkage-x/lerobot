// How much of each camera's frame the production intrinsics were measured on.
//
// This panel exists because the rig self-check next to it answers a different
// question and says so: a bump moves a camera's *pose*, so it invalidates the
// extrinsics and leaves the lens alone. Nothing else on this page says how much
// of the frame the distortion model is extrapolating over, and the usual
// instinct -- look at the reprojection error -- cannot answer it. That number
// is measured where the board went; the part of the frame the board never
// reached is exactly the part it cannot see.
//
// It reports rather than prescribes. See intrinsicsCoverage.ts for why thin
// coverage on this rig turned out not to be a re-shoot trigger.
import { useEffect, useState } from "react";
import type { DataCollectionGuiApi } from "../api";
import type { IntrinsicsCoverageResponse } from "../types";
import { StatusDot } from "../shared/ui";
import {
  coverageRows,
  coverageSummary,
  coverageVerdictDot,
  coverageVerdictLabel,
  shouldOfferIntrinsicsRecapture,
} from "./intrinsicsCoverage";

export function IntrinsicsCoveragePanel({
  api,
  onRecapture,
}: {
  api: DataCollectionGuiApi;
  onRecapture: () => void;
}) {
  const [payload, setPayload] = useState<IntrinsicsCoverageResponse | null>(null);

  useEffect(() => {
    let cancelled = false;
    api.fetchIntrinsicsCoverage().then((next) => {
      if (!cancelled) setPayload(next);
    });
    return () => {
      cancelled = true;
    };
  }, [api]);

  const rows = coverageRows(payload);
  const target = payload ? `${(payload.coverageTarget * 100).toFixed(0)}%` : "90%";

  return (
    <section className="panel calibration-panel">
      <div className="panel-heading">
        <h2>内参覆盖</h2>
        {payload?.run ? <span className="state-pill">{payload.run}</span> : null}
      </div>

      <p className="panel-note">{coverageSummary(payload)}</p>

      {rows.length > 0 ? (
        <div className="check-table calibration-table">
          <div className="check-row check-row-head">
            <strong>相机</strong>
            <span>覆盖（测到 ≥ {target} 半径为充分）· 畸变折返</span>
            <em>状态</em>
          </div>
          {rows.map((row) => (
            <div className="check-row" key={row.camera}>
              <strong>
                <StatusDot state={coverageVerdictDot[row.verdict]} />
                {row.camera}
              </strong>
              <span>
                {row.coverage} · {row.fold}
                {row.note ? ` · ${row.note}` : ""}
              </span>
              <em>{coverageVerdictLabel[row.verdict]}</em>
            </div>
          ))}
        </div>
      ) : null}

      <p className="panel-note">
        覆盖 = 标定板扫到的最远角点占画幅半径的比例。低于 {target} 时画幅外侧就没有角点约束畸变模型，
        那一圈是外推。<b>这是一项测量，不是判决</b>——外推区只有在相机实际用到那里时才付出代价。
      </p>
      <p className="panel-note">
        <b>要不要重标，看的是「覆盖到哪」对「实际用到哪」。</b>
        2026-08-21 量过两次。cube 数据（20260817_162847）：cam_06 覆盖 79%、cube 最远只到 52% 半径，
        外推区从未进入；cam_09 覆盖 86% 而 cube 中位在 71%、5.4% 的帧超出覆盖。
        刚性 marker rig 跑 manipulation 轨迹（20260821_135926，58447 次检出）：
        <b>七台相机无一角点越出自己的覆盖（0.00%）</b>，用得最远的 cam_12 / cam_07 只到 0.72 / 0.71 半径。
        换任务、换工作区这两个数都会变，判断前先重新量一次。
      </p>
      <p className="panel-note">
        <b>不能用重投影误差代替这一项。</b>held-out 残差是在板出现过的画面区域上算的，
        板没去过的地方它看不见——两套在 held-out 上打平到 0.01 px 的内参，
        实测同一像素反解出的光线在工作距离上仍可差 1–2 mm，而分歧最大的正是边缘覆盖最低的那台。
      </p>
      <p className="panel-note">
        <b>反过来，也别把外参问题当成内参问题。</b>内参外推误差必然随像半径增长、并随可见 marker 增多而收敛。
        cam_09 那 27 mm 两条都不满足（41°→67° 离轴角上 27.3→25.6 mm，1/2/3 个 marker 上 25.3/27.4/26.3 mm），
        84% 是一个固定平移——那是外参，重标内参不会改善它。
      </p>

      {shouldOfferIntrinsicsRecapture(payload) ? (
        <div className="control-row">
          <button className="cali-btn-primary" onClick={onRecapture}>
            前往内参标定
          </button>
          <span className="panel-note">
            畸变模型在画幅内折返，那片像素没有唯一光线，属于拟合缺陷而非用不用得到的问题。
            内参段每台相机各一条 episode，板要扫到<b>四角与画幅边缘</b>并覆盖近远两档距离；
            只在画面中心挥板不会改善覆盖。重标内参后外参需一并重解。
          </span>
        </div>
      ) : null}
    </section>
  );
}
