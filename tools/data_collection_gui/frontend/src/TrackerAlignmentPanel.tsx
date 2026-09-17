import { useMemo } from "react";
import type { TrackerAlignment, TrackerAlignmentSummary } from "./types";

/**
 * The laser-tracker comparison for the selected episode.
 *
 * The panel is built around one distinction, because everything else is
 * downstream of it: a residual computed against an *independent* registration is
 * a measurement, and a residual computed against a transform fitted on this very
 * trajectory is a picture. The second is useful -- you cannot see two point
 * clouds in one frame without it -- and it is worth nothing as an accuracy
 * claim, since the fit absorbs exactly the error being asked about. So the
 * verdict line is the first thing rendered and the numbers are visibly demoted
 * when it is not a measurement, rather than the caveat living in a tooltip.
 *
 * Absence is rendered as a calm note, not an error. The tracker is a shared
 * instrument and most episodes will never have an artifact; a red banner on the
 * normal case trains people to ignore the panel.
 */

function fmt(value: number | null | undefined, digits = 3, unit = ""): string {
  if (value === null || value === undefined || !Number.isFinite(value)) return "—";
  return `${value.toFixed(digits)}${unit}`;
}

function StatRow({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="tracker-stat">
      <span className="tracker-stat-label">{label}</span>
      <span className="tracker-stat-value">{value}</span>
      {hint ? <span className="tracker-stat-hint">{hint}</span> : null}
    </div>
  );
}

/** Speed strata first: a constant time offset is `|v| * dt`, so it hides in a pooled mean. */
function StrataTable({ summary }: { summary: TrackerAlignmentSummary }) {
  const rows = useMemo(() => {
    const order = ["speed_rest", "speed_slow", "speed_medium", "speed_fast"];
    const named = Object.keys(summary.strata ?? {});
    const ranked = [
      ...order.filter((k) => named.includes(k)),
      ...named.filter((k) => !order.includes(k)).sort()
    ];
    return ranked.map((key) => ({ key, stats: summary.strata[key] }));
  }, [summary]);

  if (!rows.length) return null;
  return (
    <table className="tracker-strata">
      <thead>
        <tr>
          <th>stratum</th>
          <th>n</th>
          <th>p50 mm</th>
          <th>p95 mm</th>
          <th>max mm</th>
        </tr>
      </thead>
      <tbody>
        {rows.map(({ key, stats }) => (
          <tr key={key}>
            <td>{key.replace(/_/g, " ")}</td>
            <td>{stats?.count ?? "—"}</td>
            <td>{fmt(stats?.p50 as number | null)}</td>
            <td>{fmt(stats?.p95 as number | null)}</td>
            <td>{fmt(stats?.max as number | null)}</td>
          </tr>
        ))}
      </tbody>
    </table>
  );
}

export function TrackerAlignmentPanel({ alignment }: { alignment: TrackerAlignment | null }) {
  if (!alignment) {
    return (
      <section className="panel tracker-panel">
        <div className="panel-heading">
          <h2>Laser tracker GT</h2>
          <span>gateway unreachable</span>
        </div>
      </section>
    );
  }

  if (!alignment.available) {
    return (
      <section className="panel tracker-panel">
        <div className="panel-heading">
          <h2>Laser tracker GT</h2>
          <span>no comparison for this episode</span>
        </div>
        <p className="tracker-absent">
          {alignment.reason}. Run <code>python -m metrology.cli.validate_against_tracker</code> for
          an episode recorded while the tracker was locked on.
        </p>
      </section>
    );
  }

  const { summary, series } = alignment;
  const independent = summary.registration.source === "parked_poses" && summary.registration.certifies_space;
  const covered = alignment.coverage >= alignment.minCoverage;
  const verdict = summary.certifies_space
    ? "measurement"
    : independent
      ? "independent registration, insufficient coverage"
      : "picture only — transform fitted on this trajectory";

  return (
    <section className={`panel tracker-panel ${summary.certifies_space ? "" : "tracker-panel-demoted"}`}>
      <div className="panel-heading">
        <h2>Laser tracker GT</h2>
        <span>
          episode {alignment.episode} · {alignment.target} · {alignment.generatedUtc.slice(0, 19)}Z
        </span>
      </div>

      <div className={`tracker-verdict ${summary.certifies_space ? "ok" : "warn"}`}>
        <strong>{verdict}</strong>
        <span>
          registration: {summary.registration.source.replace(/_/g, " ")}
          {summary.registration.rms_mm !== null ? ` · ${fmt(summary.registration.rms_mm, 2)} mm rms` : ""}
        </span>
      </div>

      {!independent ? (
        <p className="tracker-warning">
          The transform between tracker and camera frames was fitted on this trajectory, so it has
          already absorbed any constant offset and rotation the pipeline carries. The shapes can be
          compared; the absolute error cannot. Register from parked poses to get a number.
        </p>
      ) : null}

      <div className="tracker-stats">
        <StatRow
          label="residual p95"
          value={fmt(summary.residual_mm.p95 as number | null, 3, " mm")}
          hint={`p50 ${fmt(summary.residual_mm.p50 as number | null, 3)} · max ${fmt(summary.residual_mm.max as number | null, 3)}`}
        />
        <StatRow
          label="coverage"
          value={`${(alignment.coverage * 100).toFixed(1)}%`}
          hint={`${summary.n_paired}/${summary.n_camera_frames} frames${covered ? "" : ` · below ${(alignment.minCoverage * 100).toFixed(0)}% gate`}`}
        />
        <StatRow
          label="rotation sensitivity"
          value={fmt(summary.rotation_sensitivity_mm_per_deg, 2, " mm/°")}
          hint={
            summary.rotation_sensitivity_mm_per_deg <= 0
              ? "SMR at the target origin — this residual says nothing about rotation"
              : `lever arm ${fmt(summary.lever_arm_mm, 1)} mm`
          }
        />
        <StatRow
          label="clock rate"
          value={fmt(summary.clock.rate_error_ppm, 1, " ppm")}
          hint="controller → Thor, composed; fitted per session, never assumed"
        />
        <StatRow
          label="time cross-check"
          value={
            summary.time_crosscheck_s === null
              ? "unanswerable"
              : fmt(summary.time_crosscheck_s * 1e3, 2, " ms")
          }
          hint={
            summary.time_crosscheck_s === null
              ? "speed too steady to carry a lag"
              : "positive = tracker leads; a finding, not a correction"
          }
        />
        <StatRow
          label="interp bound"
          value={fmt(summary.interp_error_mm_bound, 4, " mm")}
          hint="resampling the 1 kHz stream onto frame times"
        />
      </div>

      <StrataTable summary={summary} />

      <p className="tracker-footnote">
        {series.residual_mm.length} plotted points
        {alignment.dropoutsRelS.length
          ? ` · ${alignment.dropoutsRelS.length} loss-of-lock window(s) excluded`
          : " · no loss of lock"}
        {" · "}
        <code>{alignment.artifact}</code>
      </p>
    </section>
  );
}
