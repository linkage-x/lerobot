import { useEffect, useMemo, useRef, type PointerEvent as ReactPointerEvent } from "react";

type Series = { name: string; values: number[] };

type Bounds = { min: number; max: number };

function computeBounds(values: number[]): Bounds {
  let min = Infinity;
  let max = -Infinity;
  for (const value of values) {
    if (!Number.isFinite(value)) continue;
    if (value < min) min = value;
    if (value > max) max = value;
  }
  if (!Number.isFinite(min) || !Number.isFinite(max)) {
    return { min: 0, max: 1 };
  }
  if (Math.abs(max - min) < 1e-9) {
    const pad = Math.max(Math.abs(max), 1) * 0.1 || 0.5;
    return { min: min - pad, max: max + pad };
  }
  return { min, max };
}

function SeriesRow({
  series,
  currentFrame,
  totalFrames,
  onSeek,
  rowHeight
}: {
  series: Series;
  currentFrame: number;
  totalFrames: number;
  onSeek: (frame: number) => void;
  rowHeight: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const bounds = useMemo(() => computeBounds(series.values), [series.values]);
  const currentValue = series.values[currentFrame];

  useEffect(() => {
    const canvas = canvasRef.current;
    if (!canvas) return;
    const ctx = canvas.getContext("2d");
    if (!ctx) return;
    const dpr = window.devicePixelRatio || 1;
    const width = canvas.clientWidth;
    const height = canvas.clientHeight;
    if (canvas.width !== width * dpr || canvas.height !== height * dpr) {
      canvas.width = width * dpr;
      canvas.height = height * dpr;
    }
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, width, height);

    ctx.fillStyle = "#f8fafb";
    ctx.fillRect(0, 0, width, height);

    const span = bounds.max - bounds.min || 1;
    const yPad = 4;
    const usable = height - yPad * 2;
    const xScale = width / Math.max(totalFrames - 1, 1);

    if (bounds.min < 0 && bounds.max > 0) {
      const zero = yPad + ((bounds.max - 0) / span) * usable;
      ctx.strokeStyle = "#e1e6ea";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(0, zero);
      ctx.lineTo(width, zero);
      ctx.stroke();
    }

    ctx.strokeStyle = "#0f766e";
    ctx.lineWidth = 1.2;
    ctx.beginPath();
    let pathStarted = false;
    for (let i = 0; i < series.values.length; i++) {
      const value = series.values[i];
      if (!Number.isFinite(value)) continue;
      const x = i * xScale;
      const y = yPad + ((bounds.max - value) / span) * usable;
      if (!pathStarted) {
        ctx.moveTo(x, y);
        pathStarted = true;
      } else {
        ctx.lineTo(x, y);
      }
    }
    ctx.stroke();

    if (totalFrames > 0) {
      const px = Math.min(width - 1, Math.max(0, currentFrame * xScale));
      ctx.strokeStyle = "#c2410c";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, height);
      ctx.stroke();
      if (Number.isFinite(currentValue)) {
        const py = yPad + ((bounds.max - currentValue) / span) * usable;
        ctx.fillStyle = "#c2410c";
        ctx.beginPath();
        ctx.arc(px, py, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [bounds, currentFrame, currentValue, series.values, totalFrames]);

  function handlePointer(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (event.buttons === 0 && event.type !== "pointerdown") return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / Math.max(rect.width, 1);
    const frame = Math.round(Math.max(0, Math.min(1, ratio)) * Math.max(totalFrames - 1, 0));
    onSeek(frame);
  }

  return (
    <div className="series-row">
      <div className="series-label">
        <span>{series.name}</span>
        <strong>{Number.isFinite(currentValue) ? currentValue.toFixed(3) : "—"}</strong>
      </div>
      <canvas
        className="series-canvas"
        onPointerDown={handlePointer}
        onPointerMove={handlePointer}
        ref={canvasRef}
        style={{ height: rowHeight }}
      />
      <div className="series-range">
        <span>{bounds.min.toFixed(2)}</span>
        <span>{bounds.max.toFixed(2)}</span>
      </div>
    </div>
  );
}

export function SeriesPlot({
  title,
  names,
  pickValue,
  currentFrame,
  totalFrames,
  onSeek,
  rowHeight = 28
}: {
  title: string;
  names: string[];
  pickValue: (frameIndex: number, dim: number) => number;
  currentFrame: number;
  totalFrames: number;
  onSeek: (frame: number) => void;
  rowHeight?: number;
}) {
  const series = useMemo<Series[]>(() => {
    return names.map((name, dim) => {
      const values: number[] = new Array(totalFrames);
      for (let i = 0; i < totalFrames; i++) {
        values[i] = pickValue(i, dim);
      }
      return { name, values };
    });
  }, [names, pickValue, totalFrames]);

  return (
    <section className="panel series-panel">
      <div className="panel-heading">
        <h2>{title}</h2>
        <span>{names.length} dims · {totalFrames} frames</span>
      </div>
      <div className="series-list">
        {series.map((entry) => (
          <SeriesRow
            key={entry.name}
            series={entry}
            currentFrame={currentFrame}
            totalFrames={totalFrames}
            onSeek={onSeek}
            rowHeight={rowHeight}
          />
        ))}
      </div>
    </section>
  );
}
