import { useMemo } from "react";
import type { TrackerAlignment, TrackerAlignmentSummary } from "./types";

/**
 * The laser-tracker comparison for the selected episode.
 *
 * The panel is built around two questions, and they are not the same one:
 *
 * 1. **Is it independent?** A residual computed against a transform fitted on
 *    this very trajectory is a picture, not a measurement -- the fit absorbs
 *    exactly the error being asked about. That is `registration.source`.
 * 2. **What did the independent fit still absorb?** Since the single-nest
 *    interface the lever arm `c` is fitted from parked poses too, and a fit
 *    soaks up every constant it can reach: a constant offset and a constant
 *    rotation in the rig frame, which together *are* the marker-to-TCP
 *    constant. The residual is then a real measurement of something narrower
 *    than "TCP error", and calling it TCP error is the mistake this panel
 *    exists to make impossible. That is `registration.absorbed_modes`.
 *
 * So the verdict line names the narrowest true statement rather than the
 * flattering one, the absorbed modes are spelled out in prose underneath, and
 * the numbers are visibly demoted when the verdict is not a measurement.
 *
 * Absence is rendered as a calm note, not an error. The tracker is a shared
 * instrument and most episodes will never have an artifact; a red banner on the
 * normal case trains people to ignore the panel.
 */

/** Plain-language gloss for each absorbed mode, keyed by the solver's vocabulary. */
const ABSORBED_MODE_TEXT: Record<string, string> = {
  body_frame_constant_translation: "a constant offset in the rig frame",
  body_frame_constant_rotation: "a constant rotation of the rig frame",
  world_frame_constant_translation: "a constant offset in the world frame",
  world_frame_constant_rotation: "a constant rotation of the world frame",
  all_rotation_error_at_fixed_attitude:
    "every orientation error, because attitude was held fixed for this session"
};

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
  const absorbed = summary.registration.absorbed_modes ?? [];
  const shapeOnly = summary.registration.source !== "parked_poses";
  const fixedAttitude = absorbed.includes("all_rotation_error_at_fixed_attitude");
  const absorbsRigConstants = absorbed.includes("body_frame_constant_rotation");
  const covered = alignment.coverage >= alignment.minCoverage;
  const mount = alignment.mountFit;

  // Ordered most-limiting first, so the verdict is always the narrowest true
  // statement: no independence beats no coverage beats a fitted lever arm.
  const verdict = shapeOnly
    ? "picture only — transform fitted on this trajectory"
    : !covered
      ? "independent registration, insufficient coverage"
      : fixedAttitude
        ? "position measurement — attitude was held, so rotation error is invisible"
        : absorbsRigConstants
          ? "measurement at the SMR point — not TCP error"
          : "measurement";
  const verdictTone = shapeOnly || !covered ? "warn" : "ok";

  return (
    <section className={`panel tracker-panel ${verdictTone === "ok" ? "" : "tracker-panel-demoted"}`}>
      <div className="panel-heading">
        <h2>Laser tracker GT</h2>
        <span>
          episode {alignment.episode} · {alignment.target} · {alignment.generatedUtc.slice(0, 19)}Z
        </span>
      </div>

      <div className={`tracker-verdict ${verdictTone}`}>
        <strong>{verdict}</strong>
        <span>
          registration: {summary.registration.source.replace(/_/g, " ")}
          {summary.registration.rms_mm !== null ? ` · ${fmt(summary.registration.rms_mm, 2)} mm rms` : ""}
        </span>
      </div>

      {shapeOnly ? (
        <p className="tracker-warning">
          The transform between tracker and camera frames was fitted on this trajectory, so it has
          already absorbed any constant offset and rotation the pipeline carries. The shapes can be
          compared; the absolute error cannot. Register from parked poses to get a number.
        </p>
      ) : null}

      {!shapeOnly && absorbed.length ? (
        <p className="tracker-warning">
          Fitted away before this residual was computed, so it is silent about them however small
          it is:{" "}
          {absorbed.map((m) => ABSORBED_MODE_TEXT[m] ?? m.replace(/_/g, " ")).join("; ")}.
          {absorbsRigConstants ? (
            <>
              {" "}
              The first two together <em>are</em> the marker&nbsp;→&nbsp;TCP constant, so this
              number is the error of the SMR centre and may not be reported as TCP error. Measuring
              that constant needs three non-collinear nests, or an STS.
            </>
          ) : null}
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
        {mount ? (
          <StatRow
            label="lever arm σ"
            value={fmt(mount.sigma?.c_sigma_norm_mm ?? null, 3, " mm")}
            hint={`bootstrap over ${mount.n_poses ?? "?"} parked poses${
              mount.holdout_rms_mm !== null && mount.holdout_rms_mm !== undefined
                ? ` · holdout ${fmt(mount.holdout_rms_mm, 3)} mm`
                : ""
            }`}
          />
        ) : null}
        {mount ? (
          <StatRow
            label="per-pose c spread"
            value={fmt(mount.per_pose_c_spread_mm ?? null, 3, " mm")}
            hint="structure here is attitude-dependent rig-pose error, not mount flex"
          />
        ) : null}
      </div>

      <StrataTable summary={summary} />

      <p className="tracker-footnote">
        {series.residual_mm.length} plotted points
        {alignment.dropoutsRelS.length
          ? ` · ${alignment.dropoutsRelS.length} loss-of-lock window(s) excluded`
          : " · no loss of lock"}
        {mount?.mount_id ? (
          <>
            {" · mount "}
            <code>{mount.mount_id}</code>
          </>
        ) : null}
        {" · "}
        <code>{alignment.artifact}</code>
      </p>
    </section>
  );
}
