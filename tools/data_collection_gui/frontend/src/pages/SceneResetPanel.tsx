import { useEffect, useMemo, useRef, useState } from "react";

import type {
  DemoLandingPoint,
  RolloutLandmarks,
  SceneResetRequest,
  SceneResetStroke,
  TableWindow
} from "../types";
import type { PlottedPoint } from "./landingMapPoints";
import { windowForPlot } from "./tableWindow";

const SIZE = 360;
const PAD = 28;
const FIXED_LIFT_M = 0.08;
const DEFAULT_FRAME = { minX: 0.18, maxX: 0.70, minY: -0.45, maxY: 0.45 };

function numberOr(value: string, fallback: number): number {
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed : fallback;
}

function formatM(value: number): string {
  return `${(value * 1000).toFixed(0)} mm`;
}

function frameFor(
  points: DemoLandingPoint[],
  strokes: SceneResetStroke[],
  pick: [number, number] | null,
  referencePoints: PlottedPoint[]
) {
  const xs = [DEFAULT_FRAME.minX, DEFAULT_FRAME.maxX];
  const ys = [DEFAULT_FRAME.minY, DEFAULT_FRAME.maxY];
  points.forEach((point) => {
    xs.push(point.graspXyz[0]);
    ys.push(point.graspXyz[1]);
  });
  if (pick) {
    xs.push(pick[0]);
    ys.push(pick[1]);
  }
  referencePoints.forEach((point) => {
    xs.push(point.x);
    ys.push(point.y);
  });
  strokes.forEach((stroke) => {
    xs.push(stroke.x - stroke.radiusM, stroke.x + stroke.radiusM);
    ys.push(stroke.y - stroke.radiusM, stroke.y + stroke.radiusM);
  });
  const minX = Math.min(...xs);
  const maxX = Math.max(...xs);
  const minY = Math.min(...ys);
  const maxY = Math.max(...ys);
  const span = Math.max(maxX - minX, maxY - minY, 0.1) * 1.08;
  const cx = (minX + maxX) / 2;
  const cy = (minY + maxY) / 2;
  const plot = SIZE - PAD * 2;
  const scale = plot / span;
  return {
    scale,
    span,
    toScreenX: (x: number) => SIZE / 2 + (x - cx) * scale,
    toScreenY: (y: number) => SIZE / 2 - (y - cy) * scale,
    toWorldX: (x: number) => cx + (x - SIZE / 2) / scale,
    toWorldY: (y: number) => cy - (y - SIZE / 2) / scale
  };
}

export function SceneResetPanel({
  title = "Scene reset",
  landmarks = {},
  tableViewUrl,
  backgroundLabel = "side camera",
  backgroundHint = "",
  referencePoints = [],
  referenceLabel = "historical rollout landings",
  referenceSourceControl,
  busy,
  disabled,
  disabledReason,
  onReset
}: {
  title?: string;
  landmarks?: RolloutLandmarks;
  /** Builds the URL of the camera still re-projected onto exactly this rectangle of table.
   *
   *  Absent while the camera has no table calibration, and then the map draws no backdrop at
   *  all. That is the honest state: a photo stretched to fill the plot box lines up with
   *  nothing in it, and painting a target region against one is aiming at the wrong table. */
  tableViewUrl?: (window: TableWindow, width: number, height: number) => string;
  backgroundLabel?: string;
  /** Why there is no backdrop, when there is none. */
  backgroundHint?: string;
  referencePoints?: PlottedPoint[];
  referenceLabel?: string;
  referenceSourceControl?: React.ReactNode;
  busy: boolean;
  disabled?: boolean;
  disabledReason?: string;
  onReset: (request: SceneResetRequest) => Promise<{ ok: boolean; error?: string }>;
}) {
  const [pickX, setPickX] = useState("0.40");
  const [pickY, setPickY] = useState("0.00");
  const [pickZ, setPickZ] = useState("0.035");
  // The pick pose follows the measured place point until an operator overrides it, and then
  // stops: a nudge that a newly loaded dataset silently undoes is worse than no default.
  const [pickFollowsDemos, setPickFollowsDemos] = useState(true);
  const [targetZ, setTargetZ] = useState("0.55");
  const [liftM] = useState(FIXED_LIFT_M.toFixed(2));
  const [brushRadius, setBrushRadius] = useState("0.035");
  const [returnToStart, setReturnToStart] = useState(true);
  const [confirmMotion, setConfirmMotion] = useState(false);
  const [strokes, setStrokes] = useState<SceneResetStroke[]>([]);
  const [message, setMessage] = useState("");
  const [drawing, setDrawing] = useState(false);
  const [backgroundLoaded, setBackgroundLoaded] = useState(false);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const demoPoints = landmarks.points ?? [];
  // Where the demonstrations left the peg. A reset picks it up from there, so this is the pick
  // pose, measured from the recording rather than typed from memory.
  const measuredPick = landmarks.placeXyz;
  const measuredPickKey = measuredPick ? measuredPick.map((value) => value.toFixed(5)).join(",") : "";
  const pickXyz = useMemo<[number, number] | null>(() => {
    const x = Number(pickX);
    const y = Number(pickY);
    return Number.isFinite(x) && Number.isFinite(y) ? [x, y] : null;
  }, [pickX, pickY]);
  const frame = useMemo(
    () => frameFor(demoPoints, strokes, pickXyz, referencePoints),
    [demoPoints, strokes, pickXyz, referencePoints]
  );
  // The rectangle of table this plot is showing. Handed to the server so the still comes back
  // already covering it: alignment is then a property of the request rather than something the
  // image and the points have to agree on separately.
  const viewWindow = useMemo<TableWindow>(
    () => windowForPlot(frame, { left: PAD, top: PAD, right: SIZE - PAD, bottom: SIZE - PAD }),
    [frame]
  );
  const plotSize = SIZE - PAD * 2;
  const backgroundUrl = tableViewUrl ? tableViewUrl(viewWindow, plotSize, plotSize) : "";

  const applyMeasuredPick = (xyz: [number, number, number]) => {
    setPickX(xyz[0].toFixed(3));
    setPickY(xyz[1].toFixed(3));
    setPickZ(xyz[2].toFixed(3));
  };

  useEffect(() => {
    if (!measuredPick || !pickFollowsDemos) return;
    applyMeasuredPick(measuredPick);
    // measuredPickKey, not measuredPick: the landmarks object is rebuilt on every poll, and
    // depending on its identity would overwrite an in-progress edit once a second.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [measuredPickKey, pickFollowsDemos]);

  const editPick = (setter: (value: string) => void) => (value: string) => {
    setPickFollowsDemos(false);
    setter(value);
  };
  const gridValues = useMemo(() => {
    const values: number[] = [];
    for (let value = -0.5; value <= 0.75; value += 0.05) values.push(value);
    return values;
  }, []);

  const addStrokeFromEvent = (event: React.PointerEvent<SVGSVGElement>) => {
    const svg = svgRef.current;
    if (!svg) return;
    const rect = svg.getBoundingClientRect();
    const screenX = ((event.clientX - rect.left) / rect.width) * SIZE;
    const screenY = ((event.clientY - rect.top) / rect.height) * SIZE;
    if (screenX < PAD || screenX > SIZE - PAD || screenY < PAD || screenY > SIZE - PAD) return;
    const next = {
      x: frame.toWorldX(screenX),
      y: frame.toWorldY(screenY),
      radiusM: Math.max(0.005, numberOr(brushRadius, 0.035))
    };
    setStrokes((current) => {
      const last = current[current.length - 1];
      if (last && Math.hypot(last.x - next.x, last.y - next.y) < next.radiusM * 0.35) return current;
      return [...current, next].slice(-512);
    });
  };

  const runReset = async () => {
    setMessage("");
    const request: SceneResetRequest = {
      pickXyz: [numberOr(pickX, 0.4), numberOr(pickY, 0), numberOr(pickZ, 0.035)],
      targetZ: numberOr(targetZ, 0.55),
      liftM: FIXED_LIFT_M,
      approachClearanceM: FIXED_LIFT_M,
      returnToStart,
      mask: { strokes }
    };
    const result = await onReset(request);
    setMessage(result.ok ? "Scene reset command sent." : result.error || "Scene reset failed.");
  };

  const cannotRun = busy || disabled || !confirmMotion || strokes.length === 0;

  return (
    <section className="card scene-reset-card">
      <div className="card-head">
        <h3>{title}</h3>
        <div className="row-actions">
          <button type="button" className="ghost" onClick={() => setStrokes([])} disabled={busy || strokes.length === 0}>
            Clear mask
          </button>
        </div>
      </div>

      <div className="scene-reset-layout">
        <div className="scene-reset-map">
          <div className="scene-reset-canvas">
          <svg
            ref={svgRef}
            viewBox={`0 0 ${SIZE} ${SIZE}`}
            onPointerDown={(event) => {
              setDrawing(true);
              addStrokeFromEvent(event);
              event.currentTarget.setPointerCapture(event.pointerId);
            }}
            onPointerMove={(event) => {
              if (drawing) addStrokeFromEvent(event);
            }}
            onPointerUp={() => setDrawing(false)}
            onPointerCancel={() => setDrawing(false)}
            role="img"
            aria-label="Scene reset target mask"
          >
            {backgroundUrl ? (
              // Inside the SVG, at the plot rectangle, in the plot's own coordinates: the image
              // covers exactly the base-frame window it was requested for, so a peg in it and a
              // point drawn over it are the same millimetres. preserveAspectRatio is off because
              // the window is the contract -- letterboxing it would re-introduce the offset this
              // whole path exists to remove.
              <image
                href={backgroundUrl}
                x={PAD}
                y={PAD}
                width={plotSize}
                height={plotSize}
                preserveAspectRatio="none"
                className="table-view-image"
                // Nothing publishes a frame until a runtime owns the cameras, so a miss here is
                // ordinary. Hide the element rather than leave a broken image under the mask,
                // and show it again as soon as a frame arrives.
                onError={() => setBackgroundLoaded(false)}
                onLoad={() => setBackgroundLoaded(true)}
                style={{ visibility: backgroundLoaded ? "visible" : "hidden" }}
              />
            ) : null}
            <rect
              x={PAD}
              y={PAD}
              width={plotSize}
              height={plotSize}
              className={`landing-map-frame ${backgroundLoaded && backgroundUrl ? "scene-reset-frame-with-bg" : ""}`}
            />
            {gridValues.map((value) => {
              const x = frame.toScreenX(value);
              const y = frame.toScreenY(value);
              return (
                <g key={value.toFixed(2)}>
                  {x >= PAD && x <= SIZE - PAD ? <line x1={x} x2={x} y1={PAD} y2={SIZE - PAD} className="landing-map-grid" /> : null}
                  {y >= PAD && y <= SIZE - PAD ? <line x1={PAD} x2={SIZE - PAD} y1={y} y2={y} className="landing-map-grid" /> : null}
                </g>
              );
            })}
            {demoPoints.map((point) => (
              <circle key={point.episode} cx={frame.toScreenX(point.graspXyz[0])} cy={frame.toScreenY(point.graspXyz[1])} r={2.8} className="landing-map-demo" />
            ))}
            {referencePoints.map((point) => (
              <g key={`reference-${point.key}`} className="scene-reset-reference-point">
                <circle
                  cx={frame.toScreenX(point.x)}
                  cy={frame.toScreenY(point.y)}
                  r={2.9}
                  className="scene-reset-reference-dot"
                />
                <title>{point.title}</title>
              </g>
            ))}
            {pickXyz ? (
              <g className="scene-reset-pick-marker">
                <circle cx={frame.toScreenX(pickXyz[0])} cy={frame.toScreenY(pickXyz[1])} r={5} />
                <line
                  x1={frame.toScreenX(pickXyz[0]) - 8}
                  x2={frame.toScreenX(pickXyz[0]) + 8}
                  y1={frame.toScreenY(pickXyz[1])}
                  y2={frame.toScreenY(pickXyz[1])}
                />
                <line
                  x1={frame.toScreenX(pickXyz[0])}
                  x2={frame.toScreenX(pickXyz[0])}
                  y1={frame.toScreenY(pickXyz[1]) - 8}
                  y2={frame.toScreenY(pickXyz[1]) + 8}
                />
                <text x={frame.toScreenX(pickXyz[0]) + 10} y={frame.toScreenY(pickXyz[1]) - 8}>
                  pick
                </text>
              </g>
            ) : null}
            {strokes.map((stroke, index) => (
              <circle
                key={`${index}-${stroke.x.toFixed(3)}-${stroke.y.toFixed(3)}`}
                cx={frame.toScreenX(stroke.x)}
                cy={frame.toScreenY(stroke.y)}
                r={stroke.radiusM * frame.scale}
                className="scene-reset-mask-stroke"
              />
            ))}
          </svg>
          </div>
          <p className="hint">
            Paint the allowed target area in base x/y.{" "}
            {backgroundUrl && backgroundLoaded
              ? `${backgroundLabel} is projected onto the table plane, so the picture and the points are the same millimetres; `
              : backgroundHint
                ? `${backgroundHint} `
                : ""}
            samples keep z fixed.
            {referencePoints.length ? ` Reference: ${referenceLabel} (${referencePoints.length}).` : ""}
          </p>
        </div>

        <div className="scene-reset-form">
          {referenceSourceControl ? (
            <div className="scene-reset-reference-control">{referenceSourceControl}</div>
          ) : null}
          <div className="row-actions">
            <label className="field inline"><span>Pick x</span><input value={pickX} onChange={(event) => editPick(setPickX)(event.target.value)} inputMode="decimal" /></label>
            <label className="field inline"><span>Pick y</span><input value={pickY} onChange={(event) => editPick(setPickY)(event.target.value)} inputMode="decimal" /></label>
            <label className="field inline"><span>Pick z</span><input value={pickZ} onChange={(event) => editPick(setPickZ)(event.target.value)} inputMode="decimal" /></label>
          </div>
          <div className="row-actions">
            <label className="field inline"><span>Place z</span><input value={targetZ} onChange={(event) => setTargetZ(event.target.value)} inputMode="decimal" /></label>
            <label className="field inline"><span>Lift</span><input value={liftM} readOnly inputMode="decimal" /></label>
            <label className="field inline"><span>Brush</span><input value={brushRadius} onChange={(event) => setBrushRadius(event.target.value)} inputMode="decimal" /></label>
          </div>
          <p className="hint">Brush radius {formatM(numberOr(brushRadius, 0.035))}; lift/descent is fixed at 80 mm for QC.</p>
          {measuredPick ? (
            <>
              <p className="hint">
                Pick is measured from where {demoPoints.length} demonstration(s) released the peg:
                x={measuredPick[0].toFixed(3)}, y={measuredPick[1].toFixed(3)}, z=
                {measuredPick[2].toFixed(3)} m. The reset goes there first and grasps, then carries
                the peg to a sampled point in the painted region.
                {pickFollowsDemos
                  ? " Edit any Pick field to nudge it."
                  : " Edited by hand — it no longer tracks the demonstrations."}
              </p>
              {!pickFollowsDemos ? (
                <div className="row-actions">
                  <button
                    type="button"
                    className="ghost"
                    onClick={() => {
                      setPickFollowsDemos(true);
                      applyMeasuredPick(measuredPick);
                    }}
                  >
                    Use the measured pick
                  </button>
                </div>
              ) : null}
            </>
          ) : (
            <p className="hint">
              No demonstration geometry for this dataset yet, so Pick is whatever you type here.
            </p>
          )}
          <label className="checkbox">
            <input type="checkbox" checked={returnToStart} onChange={(event) => setReturnToStart(event.target.checked)} />
            <span>Return arm to start after placing</span>
          </label>
          <label className="checkbox confirm-motion">
            <input type="checkbox" checked={confirmMotion} onChange={(event) => setConfirmMotion(event.target.checked)} />
            <span>The cell is clear. Scene reset moves the arm and gripper.</span>
          </label>
          {disabledReason ? <p className="hint">{disabledReason}</p> : null}
          {message ? <div className={message.includes("failed") ? "banner banner-error" : "banner banner-ok"}>{message}</div> : null}
          <button type="button" onClick={() => void runReset()} disabled={cannotRun}>
            Reset scene
          </button>
        </div>
      </div>
    </section>
  );
}
