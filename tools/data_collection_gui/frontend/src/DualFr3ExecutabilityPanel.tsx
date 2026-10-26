import { useCallback, useEffect, useState } from "react";
import { api } from "./apiClient";
import { Modal } from "./calibration/ConfirmModal";
import {
  dualFr3Rows,
  dualReplayControls,
  dualReplayLive,
  dualReplayStatus,
  edgeLines
} from "./dualFr3Executability";
import type { DualFr3Executability, DualReplayResponse, ReplayStatus } from "./types";

const POLL_MS = 400;

// The verdicts are computed offline (tools/fr3/dual_fr3_executability.py, roadmap C1-C4) and stored next to
// info.json; this shows the selected episode's. Below them the operator replays it on two arms (D4):
// Connect moves both to frame 0 and holds, Execute starts both on one clock, Abort stops both.
export function DualFr3ExecutabilityPanel({ status }: { status: ReplayStatus }) {
  const [payload, setPayload] = useState<DualFr3Executability | null>(null);
  const [replay, setReplay] = useState<DualReplayResponse | null>(null);
  // Neither is pre-selected: which plan and which arms are the operator's call.
  const [mode, setMode] = useState("");
  const [backend, setBackend] = useState("");
  const [acting, setActing] = useState("");
  const [actionError, setActionError] = useState("");
  const [confirming, setConfirming] = useState(false);
  const datasetPath = status.datasetRoot || status.dataset;

  const refreshJob = useCallback(async () => {
    const next = await api.fetchDualReplayJob();
    if (next) setReplay(next);
  }, []);
  useEffect(() => {
    void refreshJob();
  }, [refreshJob]);
  const job = replay?.job;
  const live = dualReplayLive(job);
  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => void refreshJob(), POLL_MS);
    return () => window.clearInterval(timer);
  }, [live, refreshJob]);

  const act = async (action: "connect" | "execute" | "abort", body: Record<string, unknown> = {}) => {
    setActing(action);
    setActionError("");
    const result = await api.dualReplay(action, body);
    setActing("");
    if (!result) {
      setActionError("Gateway unreachable");
      return;
    }
    setReplay(result);
    if (!result.ok) setActionError(result.error);
  };

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
  const controls = dualReplayControls(payload, job, mode, backend, replay?.arms);
  const verdict = mode ? payload?.modes[mode]?.episode : null;
  const edges = edgeLines(verdict);
  const log = job?.log ?? [];
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
      <div className="dual-fr3-run">
        <h3>Two-arm replay</h3>
        <div className="dual-fr3-run-choices">
          <div role="radiogroup" aria-label="Mode">
            <span>Mode</span>
            {(replay?.modes ?? ["absolute", "relative"]).map((name) => (
              <label key={name}>
                <input type="radio" checked={mode === name} disabled={live} onChange={() => setMode(name)} />
                {name}
              </label>
            ))}
          </div>
          <div role="radiogroup" aria-label="Arms">
            <span>Arms</span>
            <label title="the runtime's fake arms: coordination, gates and timelines on this machine; nothing moves">
              <input type="radio" checked={backend === "fake"} disabled={live} onChange={() => setBackend("fake")} />
              fake (rehearsal)
            </label>
            <label title="both FR3s of tools/fr3/fr3_arms.json">
              <input type="radio" checked={backend === "real"} disabled={live} onChange={() => setBackend("real")} />
              real FR3s
            </label>
          </div>
        </div>
        {edges.length ? (
          <ul className="dual-fr3-edges">
            {edges.map((line) => (
              <li key={line}>{line}</li>
            ))}
          </ul>
        ) : null}
        <div className="control-row">
          <button
            disabled={!mode || !backend || Boolean(controls.connectBlocker) || acting !== ""}
            title={controls.connectBlocker || undefined}
            onClick={() => void act("connect", { mode, backend })}
          >
            {acting === "connect" ? "Connecting…" : "Connect (both to frame 0)"}
          </button>
          <button
            disabled={!controls.canExecute || acting !== ""}
            onClick={() => (job?.backend === "real" ? setConfirming(true) : void act("execute"))}
          >
            Execute
          </button>
          <button disabled={!controls.canAbort || acting === "abort"} onClick={() => void act("abort")}>
            Abort
          </button>
        </div>
        {mode && backend && controls.connectBlocker && !live ? (
          <p className="panel-note error">Connect: {controls.connectBlocker}</p>
        ) : null}
        <p className="panel-note">Two-arm replay: {dualReplayStatus(job)}</p>
        {actionError ? <p className="panel-note error">{actionError}</p> : null}
        {log.length ? <pre className="dual-fr3-log">{log.slice(-8).join("\n")}</pre> : null}
      </div>
      {confirming ? (
        <Modal
          title="Execute on both FR3s"
          onClose={() => setConfirming(false)}
          footer={
            <>
              <button onClick={() => setConfirming(false)}>Cancel</button>
              <button
                className="cali-btn-primary"
                onClick={() => {
                  setConfirming(false);
                  void act("execute");
                }}
              >
                Execute
              </button>
            </>
          }
        >
          <p>
            Both arms start together on one clock and stream the retargeted joint path ({job?.mode}). Either one
            faulting, or the arms coming within the gate distance, stops both. Keep a hand on each emergency stop.
          </p>
        </Modal>
      ) : null}
    </section>
  );
}
