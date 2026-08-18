// Shared BOX sensor visualizations reused by the live monitor cards and the
// calibration result panels: the tactile heatmap and the per-axis 6D force
// grid (which doubles as the pass/fail table when given an evaluation).

import type { ForceAxisLimits } from "./config";
import { fmtNum } from "./config";
import { evaluateForce } from "./parseCaliLog";
import type { AxisEval, ForceVec } from "./types";
import { FORCE_AXES, FORCE_AXIS_LABELS } from "./types";
import { TouchHeatmapGrid, touchLayoutForSample, touchScaleFromSamples } from "../touchVisualization";

export function TouchHeatmap({
  fz0p1N,
  fx0p1N = [],
  fy0p1N = [],
  model,
  points,
}: {
  fz0p1N: number[];
  fx0p1N?: number[];
  fy0p1N?: number[];
  // Pad geometry reported alongside the frame; without it a 9-taxel M2020
  // frame and an untouched Paxini pad are indistinguishable.
  model?: string;
  points?: number;
}) {
  const sample = { fz: fz0p1N, fx: fx0p1N, fy: fy0p1N, model, points };
  if (fz0p1N.length === 0) {
    return <div className="camera-tile-empty">无触觉采样</div>;
  }
  const layout = touchLayoutForSample(sample);
  return (
    <div className="box-touch-view">
      <TouchHeatmapGrid
        sample={sample}
        scale={touchScaleFromSamples([sample])}
        ariaLabel="触觉实时热力图"
        className="box-touch-fill"
        emptyText="无触觉采样"
      />
      {layout ? <div className="box-touch-legend">{layout.label}</div> : null}
    </div>
  );
}

/**
 * Per-axis 6D force grid. When `limits` is provided each axis is checked and
 * out-of-range axes are highlighted (not the whole card). When `evals` is
 * provided the explicit PASS/FAIL from a calibration run is shown instead.
 */
export function ForceAxisGrid({
  vec,
  limits,
  evals,
  variant = "monitor",
}: {
  vec?: ForceVec | null;
  limits?: ForceAxisLimits | null;
  evals?: AxisEval[] | null;
  variant?: "monitor" | "result";
}) {
  const rows: AxisEval[] =
    evals ??
    (vec && limits ? evaluateForce(vec, limits).axes : null) ??
    (vec
      ? FORCE_AXES.map((axis) => ({
          axis,
          label: FORCE_AXIS_LABELS[axis],
          value: vec[axis],
          kind: axis.startsWith("m") ? ("moment" as const) : ("force" as const),
          pass: true,
          reason: "",
        }))
      : []);

  if (rows.length === 0) {
    return <div className="cali-axis-empty">无数据</div>;
  }

  return (
    <div className="cali-axis-grid">
      {rows.map((row) => {
        const digits = row.kind === "moment" ? 3 : 2;
        const unit = row.kind === "moment" ? "N·m" : "N";
        const cls = row.pass ? "" : "cali-axis-fail";
        return (
          <div className={`cali-axis-row ${cls}`} key={row.axis}>
            <span className="cali-axis-label">{row.label}</span>
            <span className="cali-axis-value">{fmtNum(row.value, digits)}</span>
            <span className="cali-axis-unit">{unit}</span>
            {variant === "result" && (
              <span className={`cali-axis-verdict ${row.pass ? "pass" : "fail"}`}>
                {row.pass ? "PASS" : "FAIL"}
              </span>
            )}
          </div>
        );
      })}
    </div>
  );
}
