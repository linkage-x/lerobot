import { useMemo, useRef, useState } from "react";

import type { DemoLandingPoint, RolloutLandmarks, SceneResetRequest, SceneResetStroke } from "../types";

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

function frameFor(points: DemoLandingPoint[], strokes: SceneResetStroke[]) {
  const xs = [DEFAULT_FRAME.minX, DEFAULT_FRAME.maxX];
  const ys = [DEFAULT_FRAME.minY, DEFAULT_FRAME.maxY];
  points.forEach((point) => {
    xs.push(point.graspXyz[0]);
    ys.push(point.graspXyz[1]);
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
  backgroundImageUrl,
  backgroundLabel = "side camera",
  busy,
  disabled,
  disabledReason,
  onReset
}: {
  title?: string;
  landmarks?: RolloutLandmarks;
  backgroundImageUrl?: string;
  backgroundLabel?: string;
  busy: boolean;
  disabled?: boolean;
  disabledReason?: string;
  onReset: (request: SceneResetRequest) => Promise<{ ok: boolean; error?: string }>;
}) {
  const [pickX, setPickX] = useState("0.40");
  const [pickY, setPickY] = useState("0.00");
  const [pickZ, setPickZ] = useState("0.035");
  const [targetZ, setTargetZ] = useState("0.035");
  const [liftM] = useState(FIXED_LIFT_M.toFixed(2));
  const [brushRadius, setBrushRadius] = useState("0.035");
  const [returnToStart, setReturnToStart] = useState(true);
  const [confirmMotion, setConfirmMotion] = useState(false);
  const [strokes, setStrokes] = useState<SceneResetStroke[]>([]);
  const [message, setMessage] = useState("");
  const [drawing, setDrawing] = useState(false);
  const svgRef = useRef<SVGSVGElement | null>(null);

  const demoPoints = landmarks.points ?? [];
  const frame = useMemo(() => frameFor(demoPoints, strokes), [demoPoints, strokes]);
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
      targetZ: numberOr(targetZ, 0.035),
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
            {backgroundImageUrl ? (
              <img src={backgroundImageUrl} alt={`${backgroundLabel} reset reference`} className="scene-reset-bg-img" />
            ) : null}
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
            <rect
              x={PAD}
              y={PAD}
              width={SIZE - PAD * 2}
              height={SIZE - PAD * 2}
              className={`landing-map-frame ${backgroundImageUrl ? "scene-reset-frame-with-bg" : ""}`}
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
            Paint the allowed target area in base x/y. {backgroundImageUrl ? `${backgroundLabel} is a desaturated reference layer; ` : ""}
            samples keep z fixed.
          </p>
        </div>

        <div className="scene-reset-form">
          <div className="row-actions">
            <label className="field inline"><span>Pick x</span><input value={pickX} onChange={(event) => setPickX(event.target.value)} inputMode="decimal" /></label>
            <label className="field inline"><span>Pick y</span><input value={pickY} onChange={(event) => setPickY(event.target.value)} inputMode="decimal" /></label>
            <label className="field inline"><span>Pick z</span><input value={pickZ} onChange={(event) => setPickZ(event.target.value)} inputMode="decimal" /></label>
          </div>
          <div className="row-actions">
            <label className="field inline"><span>Place z</span><input value={targetZ} onChange={(event) => setTargetZ(event.target.value)} inputMode="decimal" /></label>
            <label className="field inline"><span>Lift</span><input value={liftM} readOnly inputMode="decimal" /></label>
            <label className="field inline"><span>Brush</span><input value={brushRadius} onChange={(event) => setBrushRadius(event.target.value)} inputMode="decimal" /></label>
          </div>
          <p className="hint">Brush radius {formatM(numberOr(brushRadius, 0.035))}; lift/descent is fixed at 80 mm for QC.</p>
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
