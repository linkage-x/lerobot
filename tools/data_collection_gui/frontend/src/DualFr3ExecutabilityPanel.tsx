import { useEffect, useState } from "react";
import { api } from "./apiClient";
import { dualFr3Rows } from "./dualFr3Executability";
import type { DualFr3Executability, ReplayStatus } from "./types";

// Read-only: the verdicts are computed offline (tools/fr3/dual_fr3_executability.py, roadmap C1-C4) and
// stored next to info.json; this shows the selected episode's.
export function DualFr3ExecutabilityPanel({ status }: { status: ReplayStatus }) {
  const [payload, setPayload] = useState<DualFr3Executability | null>(null);
  const datasetPath = status.datasetRoot || status.dataset;

  useEffect(() => {
    setPayload(null);
    if (!datasetPath) return;
    let mounted = true;
    api.fetchDualFr3Executability(datasetPath, status.episode).then((next) => {
      if (mounted) setPayload(next);
    });
    return () => {
      mounted = false;
    };
  }, [datasetPath, status.episode, status.revision]);

  const rows = dualFr3Rows(payload);
  return (
    <section className="panel dual-fr3-panel">
      <div className="panel-heading">
        <h2>Two-Arm Executability</h2>
        <span>episode {status.episode} · both hands on two FR3s, after the joint retarget</span>
      </div>
      {rows.length ? (
        <>
          <div className="table-scroll">
            <table className="tracker-strata">
              <thead>
                <tr>
                  <th>Mode</th>
                  <th>Verdict</th>
                  <th>TCP change p95 / max (mm)</th>
                  <th>T_LR change max (mm)</th>
                  <th>Min arm gap / d_safe (mm)</th>
                  <th>Placement</th>
                  <th>Reference time</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row) => (
                  <tr key={row.mode} title={row.evaluated ? `evaluated ${row.evaluated}` : undefined}>
                    <td>{row.mode}</td>
                    <td className={row.verdict === "executable" ? "dual-fr3-ok" : "dual-fr3-no"}>{row.verdict}</td>
                    <td>{row.change}</td>
                    <td>{row.tlr}</td>
                    <td>{row.gap}</td>
                    <td>{row.placement}</td>
                    <td>{row.timing}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
          {rows
            .filter((row) => row.reasons.length)
            .map((row) => (
              <p className="panel-note error" key={`${row.mode}-reasons`}>
                <strong>{row.mode}:</strong> {row.reasons.join("; ")}
              </p>
            ))}
          {rows.some((row) => row.placement === "assumed") ? (
            <p className="panel-note">
              The left base is placed by assumption until it is measured (B3); d_safe carries that as a 20 mm sigma.
            </p>
          ) : null}
        </>
      ) : (
        <p className="panel-note">
          No two-arm verdict for this dataset. Run tools/fr3/dual_fr3_executability.py on it (in the FR3
          environment) to retarget every episode for two arms and write the verdicts here.
        </p>
      )}
    </section>
  );
}
