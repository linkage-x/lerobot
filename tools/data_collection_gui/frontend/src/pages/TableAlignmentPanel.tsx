import { useCallback, useEffect, useRef, useState } from "react";

import { api } from "../apiClient";
import type { TableAlignment } from "../types";
import { suggestedTargets } from "./tableTargets";

/** Ties one camera's pixels to the table, using the robot as the calibration target.
 *
 * The maps on this page are drawn in base x/y and the camera is not: a still is a projection of
 * the table, so a photo stretched to fill a plot box lines up with nothing in it. Four points
 * whose base coordinate is known fix the whole plane, and the arm can supply them itself --
 * it goes to a coordinate, holds still while a frame is frozen, and the operator clicks the
 * tool in that frame. No board, no intrinsics, no hand-eye solve.
 *
 * The base half of every pair comes from the runtime's report of where it actually got to,
 * never from the coordinate that was typed here. A probe that was refused, that timed out, or
 * that stopped short would otherwise be recorded as if it had arrived, and the error would be
 * invisible in exactly the way a calibration must not be.
 *
 * What to click is not a matter of taste. The coordinate the arm reports belongs to the tool
 * frame -- on this rig `pika_gripper_ee`, the midpoint of the two fingers' working point -- and
 * every point sharing that x/y is a whole vertical line in the image. Clicking anywhere else on
 * that line, its footprint on the table included, fits the plane *that* point sweeps instead.
 * Which is why the plane height is a real choice rather than a formality: the fit is exact at
 * the height it was probed at, and a feature h below it is displaced in the warped still by
 * about h times the tangent of the camera's angle off vertical. On a side camera that is a
 * centimetre of error per centimetre of height.
 */

const PROBE_POLL_MS = 1000;
const PROBE_TIMEOUT_MS = 90_000;

function numberOr(value: string, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

export function TableAlignmentPanel({
  cameraKey,
  planeZDefault,
  centre,
  disabled,
  disabledReason,
  onAlignmentChange
}: {
  cameraKey: string;
  /** The height the points are probed at. The demonstrations' release height by default: that
   *  is the plane the pegs and the plotted landing points live on. */
  planeZDefault: number;
  /** Where the task happens, used to seed the suggested probe targets. */
  centre: [number, number];
  disabled?: boolean;
  disabledReason?: string;
  onAlignmentChange?: (alignment: TableAlignment | null) => void;
}) {
  const [alignment, setAlignment] = useState<TableAlignment | null>(null);
  const [planeZ, setPlaneZ] = useState(planeZDefault.toFixed(3));
  const [spread, setSpread] = useState("0.080");
  const [targetX, setTargetX] = useState(centre[0].toFixed(3));
  const [targetY, setTargetY] = useState(centre[1].toFixed(3));
  const [confirmMotion, setConfirmMotion] = useState(false);
  const [busy, setBusy] = useState(false);
  const [probing, setProbing] = useState(false);
  // Carried as a flag rather than sniffed out of the text: whether something failed is known
  // where it happened, and matching on words gets it wrong the first time one changes.
  const [message, setMessage] = useState<{ text: string; failed: boolean } | null>(null);
  const [frameNonce, setFrameNonce] = useState(0);
  const [frameSize, setFrameSize] = useState<{ width: number; height: number } | null>(null);
  const changeRef = useRef(onAlignmentChange);
  changeRef.current = onAlignmentChange;

  const load = useCallback(async () => {
    const next = await api.fetchTableAlignment(cameraKey);
    setAlignment(next);
    changeRef.current?.(next);
    return next;
  }, [cameraKey]);

  useEffect(() => {
    void load();
  }, [load]);

  // The plane height is a property of the stored points, not of this form: once anything is
  // recorded the field reports what those points were taken at rather than offering to change
  // it, because changing it would describe them wrongly.
  const lockedPlaneZ = (alignment?.points.length ?? 0) > 0 ? alignment?.planeZ : undefined;
  useEffect(() => {
    if (lockedPlaneZ !== undefined) setPlaneZ(lockedPlaneZ.toFixed(3));
  }, [lockedPlaneZ]);

  // The demonstrations' geometry arrives a poll or two after this panel mounts, so the height
  // and the targets follow it until something is recorded against them.
  useEffect(() => {
    if (lockedPlaneZ !== undefined) return;
    setPlaneZ(planeZDefault.toFixed(3));
    setTargetX(centre[0].toFixed(3));
    setTargetY(centre[1].toFixed(3));
    // centre is a fresh array on every render of the parent; its contents are the dependency.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [lockedPlaneZ, planeZDefault, centre[0], centre[1]]);

  const probe = async () => {
    setMessage(null);
    setBusy(true);
    const xyz: [number, number, number] = [
      numberOr(targetX, centre[0]),
      numberOr(targetY, centre[1]),
      numberOr(planeZ, planeZDefault)
    ];
    const result = await api.probeTablePoint(cameraKey, xyz);
    setBusy(false);
    if (!result.ok) {
      setMessage({ text: result.error || "The probe was refused.", failed: true });
      return;
    }
    setProbing(true);
    setMessage({
      text: "The arm is on its way to the point. It will hold there while a still is taken.",
      failed: false
    });
    const deadline = Date.now() + PROBE_TIMEOUT_MS;
    // Polled rather than pushed: the runtime reports through its log, and the one fact this
    // needs from it -- "the still on disk is the one you asked for" -- is exactly what the
    // alignment endpoint already answers.
    while (Date.now() < deadline) {
      await new Promise((resolve) => setTimeout(resolve, PROBE_POLL_MS));
      const next = await load();
      if (next?.probeFrame && !next.probeFrame.moving && next.probeFrame.ready) {
        setProbing(false);
        setFrameNonce((value) => value + 1);
        setMessage({
        text: "Click the midpoint of the closed fingertips — not its footprint on the table.",
        failed: false
      });
        return;
      }
    }
    setProbing(false);
    setMessage({
      text:
        "The probe did not finish in time — the arm reports every step it takes, so check the " +
        "rollout log for which one stalled. A corner it cannot solve inverse kinematics for is " +
        "the usual cause; type a reachable x/y and probe that instead. Four points anywhere on " +
        "the table work, as long as no three of them are in a line.",
      failed: true
    });
  };

  const recordClick = async (event: React.MouseEvent<HTMLImageElement>) => {
    const image = event.currentTarget;
    if (!image.naturalWidth || !image.naturalHeight) return;
    const rect = image.getBoundingClientRect();
    const u = ((event.clientX - rect.left) / rect.width) * image.naturalWidth;
    const v = ((event.clientY - rect.top) / rect.height) * image.naturalHeight;
    setBusy(true);
    const result = await api.recordTablePoint(cameraKey, {
      u,
      v,
      imageWidth: image.naturalWidth,
      imageHeight: image.naturalHeight
    });
    setBusy(false);
    if (!result.ok) {
      setMessage({ text: result.error || "That point was not recorded.", failed: true });
      return;
    }
    setAlignment(result);
    changeRef.current?.(result);
    setMessage({
      text: result.calibrated
        ? `Recorded. ${result.points.length} points, worst residual ${result.maxResidualMm.toFixed(1)} mm.`
        : `Recorded. ${result.points.length} of ${result.minPoints} points needed for a fit.`,
      failed: false
    });
  };

  const removePoint = async (index: number) => {
    setBusy(true);
    const result = await api.deleteTablePoint(cameraKey, index);
    setBusy(false);
    if (!result.ok) {
      setMessage({ text: result.error || "That point was not deleted.", failed: true });
      return;
    }
    setAlignment(result);
    changeRef.current?.(result);
  };

  const clearAll = async () => {
    setBusy(true);
    const result = await api.clearTableAlignment(cameraKey);
    setBusy(false);
    if (!result.ok) {
      setMessage({ text: result.error || "The alignment was not cleared.", failed: true });
      return;
    }
    setAlignment(result);
    changeRef.current?.(result);
    setMessage({
      text: "Cleared. The maps have no camera backdrop until four points are recorded again.",
      failed: false
    });
  };

  const points = alignment?.points ?? [];
  const frameReady = Boolean(alignment?.probeFrame?.ready) && !probing;
  const cannotProbe = busy || probing || disabled || !confirmMotion;

  return (
    <section className="card">
      <div className="card-head">
        <h3>Table alignment — {cameraKey} camera</h3>
        <div className="row-actions">
          <button type="button" className="ghost" onClick={() => void clearAll()} disabled={busy || points.length === 0}>
            Clear points
          </button>
        </div>
      </div>

      {alignment?.calibrated ? (
        <div className="banner banner-ok">
          Aligned from {points.length} points on the z={alignment.planeZ.toFixed(3)} m plane; worst
          residual {alignment.maxResidualMm.toFixed(1)} mm. The maps draw the camera underneath
          their points.
        </div>
      ) : alignment?.fitError ? (
        <div className="banner banner-warn">
          {points.length} points recorded and no mapping yet: {alignment.fitError}
        </div>
      ) : (
        <div className="banner">
          Not aligned. The maps draw their grid alone until {alignment?.minPoints ?? 4} points are
          recorded — a stretched photo behind them would line up with nothing.
        </div>
      )}

      <p className="hint">
        The arm is the calibration target: it takes its closed gripper to a base coordinate you
        choose, holds while a still is frozen, and you click <strong>the midpoint of the two
        closed fingertips</strong> in that still — that point, not its footprint on the table.
        The arm reports the fingertip midpoint&apos;s coordinate, so clicking anything else on the
        same vertical shifts the whole fit. Spread the points around the working area, and keep no
        three of them in a straight line — three on a line fit no plane, so the centre of a
        square cannot replace a corner the arm cannot reach.
      </p>
      <p className="hint">
        Plane z decides which height the projection is exact at: a feature h below the probed
        plane lands about h·tan(camera tilt) away in the warped still, which on a side view is
        centimetres. So probe at the height of what you want lined up — the tabletop for the hole
        and anything lying on it, the release height for the peg as the gripper holds it.
      </p>

      <div className="row-actions">
        <label className="field inline">
          <span>Plane z</span>
          <input
            value={planeZ}
            onChange={(event) => setPlaneZ(event.target.value)}
            readOnly={lockedPlaneZ !== undefined}
            inputMode="decimal"
          />
        </label>
        <label className="field inline">
          <span>Probe x</span>
          <input value={targetX} onChange={(event) => setTargetX(event.target.value)} inputMode="decimal" />
        </label>
        <label className="field inline">
          <span>Probe y</span>
          <input value={targetY} onChange={(event) => setTargetY(event.target.value)} inputMode="decimal" />
        </label>
        <label className="field inline">
          <span>Spread</span>
          <input value={spread} onChange={(event) => setSpread(event.target.value)} inputMode="decimal" />
        </label>
      </div>

      <div className="row-actions table-align-targets">
        {suggestedTargets(centre, numberOr(spread, 0.08)).map((target) => (
          <button
            key={`${target.x.toFixed(3)}:${target.y.toFixed(3)}`}
            type="button"
            className="ghost"
            onClick={() => {
              setTargetX(target.x.toFixed(3));
              setTargetY(target.y.toFixed(3));
            }}
          >
            {target.x.toFixed(2)}, {target.y.toFixed(2)}
          </button>
        ))}
      </div>

      <label className="checkbox confirm-motion">
        <input type="checkbox" checked={confirmMotion} onChange={(event) => setConfirmMotion(event.target.checked)} />
        <span>The cell is clear. Probing moves the arm to the point and closes the gripper.</span>
      </label>
      {disabledReason ? <p className="hint">{disabledReason}</p> : null}
      <div className="row-actions">
        <button type="button" onClick={() => void probe()} disabled={cannotProbe}>
          {probing ? "Probing…" : "Move arm to point & capture"}
        </button>
      </div>
      {message ? (
        <div className={message.failed ? "banner banner-error" : "banner banner-ok"}>{message.text}</div>
      ) : null}

      {frameReady ? (
        <div className="table-align-frame">
          <img
            src={api.tableProbeFrameUrl(cameraKey, frameNonce)}
            alt={`${cameraKey} camera still at the probed point`}
            onClick={(event) => void recordClick(event)}
            onLoad={(event) =>
              setFrameSize({
                width: event.currentTarget.naturalWidth,
                height: event.currentTarget.naturalHeight
              })
            }
          />
          {frameSize ? (
            <svg viewBox={`0 0 ${frameSize.width} ${frameSize.height}`} preserveAspectRatio="none">
              {points.map((point, index) => (
                <g key={`${point.u.toFixed(1)}:${point.v.toFixed(1)}`} className="table-align-marker">
                  <circle cx={point.u} cy={point.v} r={7} />
                  <line x1={point.u - 12} x2={point.u + 12} y1={point.v} y2={point.v} />
                  <line x1={point.u} x2={point.u} y1={point.v - 12} y2={point.v + 12} />
                  <text x={point.u + 10} y={point.v - 10}>
                    {index + 1}
                  </text>
                </g>
              ))}
            </svg>
          ) : null}
        </div>
      ) : (
        <p className="hint">
          {probing
            ? "Waiting for the arm to reach the point…"
            : "No still yet. Probe a point and one appears here to click."}
        </p>
      )}

      {points.length > 0 ? (
        <table className="table table-align-points">
          <thead>
            <tr>
              <th>#</th>
              <th>base x, y (m)</th>
              <th>pixel u, v</th>
              <th>residual</th>
              <th />
            </tr>
          </thead>
          <tbody>
            {points.map((point, index) => (
              <tr key={`${point.x.toFixed(4)}:${point.y.toFixed(4)}`}>
                <td>{index + 1}</td>
                <td>
                  {point.x.toFixed(3)}, {point.y.toFixed(3)}
                </td>
                <td>
                  {point.u.toFixed(0)}, {point.v.toFixed(0)}
                </td>
                <td>
                  {alignment?.residualsMm?.[index] !== undefined
                    ? `${alignment.residualsMm[index].toFixed(1)} mm`
                    : "—"}
                </td>
                <td>
                  <button type="button" className="ghost" onClick={() => void removePoint(index)} disabled={busy}>
                    Delete
                  </button>
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      ) : null}
      {points.length >= (alignment?.minPoints ?? 4) && points.length < (alignment?.recommendedPoints ?? 5) ? (
        <p className="hint">
          Four points fit exactly, so the residuals above are zero whether the clicks were good or
          not. Probe a fifth to get a number that can tell you.
        </p>
      ) : null}
    </section>
  );
}
