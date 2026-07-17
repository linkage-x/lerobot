// Shared BOX sensor visualizations reused by the live monitor cards and the
// calibration result panels: the tactile heatmap and the per-axis 6D force
// grid (which doubles as the pass/fail table when given an evaluation).

import type { ForceAxisLimits } from "./config";
import { fmtNum } from "./config";
import { evaluateForce } from "./parseCaliLog";
import type { AxisEval, ForceVec } from "./types";
import { FORCE_AXES, FORCE_AXIS_LABELS } from "./types";

// Paxini pad hardware layout: 15 rows, 17 columns, corners trimmed. Kept here
// (mirrors the Device Manager tile) so the calibration views don't import from
// the monolithic App.tsx.
export const TOUCH_ROW_LENGTHS = [13, 13, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 17, 13, 13];
export const TOUCH_COLUMNS = 17;

export function touchColor(value: number, scaleMax: number): string {
  const ratio = scaleMax > 0 ? Math.min(1, Math.abs(value) / scaleMax) : 0;
  // dark slate → teal, matching the Device Manager palette
  const light = 18 + ratio * 42;
  return `hsl(${175 - ratio * 12}deg 55% ${light}%)`;
}

export function TouchHeatmap({ fz0p1N }: { fz0p1N: number[] }) {
  if (fz0p1N.length < 239) {
    return <div className="camera-tile-empty">无触觉采样</div>;
  }
  const scaleMax = Math.max(1, ...fz0p1N.map((v) => Math.abs(v)));
  let cursor = 0;
  return (
    <div className="box-touch-fill" aria-label="触觉实时热力图">
      {TOUCH_ROW_LENGTHS.map((length, rowIndex) => {
        const offset = Math.floor((TOUCH_COLUMNS - length) / 2);
        const row = fz0p1N.slice(cursor, cursor + length);
        const startIndex = cursor;
        cursor += length;
        return (
          <div className="touch-row" key={rowIndex}>
            {Array.from({ length: offset }).map((_, i) => (
              <span className="touch-cell touch-cell-empty" key={`pre-${i}`} />
            ))}
            {row.map((value, i) => {
              const pointIndex = startIndex + i + 1;
              return (
                <span
                  className="touch-cell"
                  key={pointIndex}
                  title={`#${pointIndex} fz=${value.toFixed(1)} (0.1N)`}
                  style={{ backgroundColor: touchColor(Math.abs(value), scaleMax) }}
                />
              );
            })}
            {Array.from({ length: TOUCH_COLUMNS - length - offset }).map((_, i) => (
              <span className="touch-cell touch-cell-empty" key={`post-${i}`} />
            ))}
          </div>
        );
      })}
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
