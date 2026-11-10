import { useCallback, useEffect, useState } from "react";
import { api } from "./apiClient";
import { Modal } from "./calibration/ConfirmModal";
import {
  dualHomeArmRows,
  dualHomeArmsBlocker,
  dualHomeControls,
  dualHomeGateLine,
  dualHomeLive,
  dualHomeOptionRows,
  dualHomeResultRows,
  dualHomeSecondsLeft,
  dualHomeStagesText,
  dualHomeStatus,
  dualHomeVerdict
} from "./dualFr3Home";
import type { DualHomeResponse, RobotArms } from "./types";

const POLL_MS = 400;

// Both FR3s to their start joints (tools/fr3/dual_fr3_move_to_start.py). The single-arm button cannot see the
// other arm; this one reads both, checks their straight joint paths home against each other and the table, and
// shows how they would go -- together, one after the other, or not at all -- before anything moves.
export function DualFr3HomePanel({ arms }: { arms: RobotArms | null }) {
  const [response, setResponse] = useState<DualHomeResponse | null>(null);
  const [acting, setActing] = useState("");
  const [actionError, setActionError] = useState("");
  const [confirming, setConfirming] = useState(false);
  const [nowS, setNowS] = useState(() => Date.now() / 1000);

  const refresh = useCallback(async () => {
    const next = await api.fetchDualHome();
    if (next) setResponse(next);
  }, []);
  useEffect(() => {
    void refresh();
  }, [refresh]);
  const job = response?.job;
  const live = dualHomeLive(job);
  useEffect(() => {
    if (!live) return;
    const timer = window.setInterval(() => {
      setNowS(Date.now() / 1000);
      void refresh();
    }, POLL_MS);
    return () => window.clearInterval(timer);
  }, [live, refresh]);

  const act = async (action: "check" | "execute" | "cancel") => {
    setActing(action);
    setActionError("");
    const result = await api.dualHome(action);
    setActing("");
    if (!result) {
      setActionError("Gateway unreachable; nothing was sent to the arms.");
      return;
    }
    setResponse(result);
    if (!result.ok) setActionError(result.error);
  };

  const blocker = dualHomeArmsBlocker(arms);
  const controls = dualHomeControls(job, blocker);
  const plan = job?.plan ?? null;
  const verdict = dualHomeVerdict(plan);
  const armRows = dualHomeArmRows(plan);
  const optionRows = dualHomeOptionRows(plan);
  const results = dualHomeResultRows(job);
  const secondsLeft = dualHomeSecondsLeft(job, nowS);
  const serving = (arms?.arms ?? []).filter((row) => row.inService);
  const log = job?.log ?? [];

  return (
    <section className="panel dual-home-panel">
      <div className="panel-heading">
        <h2>Both Arms to Start</h2>
        <span>
          {serving.map((row) => `${row.arm} ${row.robotIp}`).join(" · ") || "two FR3s"} · paths checked against each other and the table
        </span>
      </div>
      {blocker ? <p className="panel-note">{blocker}</p> : null}
      <div className="control-row">
        <button disabled={!controls.canCheck || acting !== ""} onClick={() => void act("check")} type="button">
          {acting === "check" || job?.state === "checking" ? "Checking…" : "Check (read both arms)"}
        </button>
        <button
          className="danger"
          disabled={!controls.canExecute || acting !== ""}
          onClick={() => setConfirming(true)}
          type="button"
        >
          Move to start…
        </button>
        <button disabled={!controls.canCancel || acting === "cancel"} onClick={() => void act("cancel")} type="button">
          {controls.cancelStops ? "Stop (drops both FCIs)" : "Cancel"}
        </button>
      </div>
      <p className="panel-note">
        Status: {dualHomeStatus(job)}
        {secondsLeft !== null ? ` · the plan lapses in ${secondsLeft} s` : ""}
      </p>
      {actionError ? <p className="panel-note error">{actionError}</p> : null}
      {verdict ? (
        <div className={`dual-home-verdict ${verdict.tone === "ok" ? "dual-fr3-ok" : "dual-fr3-no"}`}>{verdict.headline}</div>
      ) : null}
      {plan?.reasons.length ? (
        <ul className="dual-home-reasons">
          {plan.reasons.map((reason) => (
            <li key={reason}>{reason}</li>
          ))}
        </ul>
      ) : null}
      {armRows.length ? (
        <div className="table-scroll">
          <table className="tracker-strata">
            <thead>
              <tr>
                <th>Arm</th>
                <th>IP</th>
                <th>Way home</th>
                <th>Closest to table / cart</th>
              </tr>
            </thead>
            <tbody>
              {armRows.map((row) => (
                <tr key={row.arm} title={row.payload || undefined}>
                  <td>{row.arm}</td>
                  <td>{row.ip}</td>
                  <td>{row.move}</td>
                  <td>{row.table}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {optionRows.length ? (
        <div className="table-scroll">
          <table className="tracker-strata">
            <thead>
              <tr>
                <th>Order</th>
                <th>Closest the arms come</th>
                <th>Where</th>
                <th />
              </tr>
            </thead>
            <tbody>
              {optionRows.map((row) => (
                <tr key={row.name} className={row.chosen ? "dual-home-chosen" : undefined}>
                  <td>{row.label}</td>
                  <td className={row.clear ? "dual-fr3-ok" : "dual-fr3-no"}>{row.gap}</td>
                  <td>{row.where}</td>
                  <td>{row.chosen ? "chosen" : row.clear ? "clear" : "too close"}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      ) : null}
      {plan ? (
        <p className="panel-note">
          {dualHomeGateLine(plan)} Placement: {plan.placement?.source ?? "—"}
        </p>
      ) : null}
      {results.length ? (
        <ul className="dual-home-results">
          {results.map((row) => (
            <li key={`${row.stage}-${row.arm}`} className={row.ok ? "dual-fr3-ok" : "dual-fr3-no"}>
              stage {row.stage} · {row.arm}: {row.text}
            </li>
          ))}
        </ul>
      ) : null}
      {log.length ? <pre className="dual-fr3-log">{log.slice(-8).join("\n")}</pre> : null}
      {confirming ? (
        <Modal
          title="Move both FR3s to start"
          onClose={() => setConfirming(false)}
          footer={
            <>
              <button onClick={() => setConfirming(false)} type="button">
                Cancel
              </button>
              <button
                className="cali-btn-primary"
                onClick={() => {
                  setConfirming(false);
                  void act("execute");
                }}
                type="button"
              >
                Move
              </button>
            </>
          }
        >
          <p>{dualHomeStagesText(plan)}</p>
          <p>
            Both arms are read again first; if either moved since the check, nothing moves. Clear the space between
            the arms and keep a hand on each user stop.
          </p>
        </Modal>
      ) : null}
    </section>
  );
}
