import { useEffect, useMemo, useRef, type PointerEvent as ReactPointerEvent } from "react";

type Series = { name: string; values: number[]; color: string; component: string };

type SeriesNameItem = { name: string; dim: number; component: string };

export type SeriesNameGroup = { key: string; name: string; items: SeriesNameItem[] };

type Bounds = { min: number; max: number };

const SERIES_COLORS = ["#0f766e", "#2563eb", "#7c3aed", "#c2410c", "#b45309", "#4d7c0f"];
const COMPONENT_ORDER = new Map(
  ["w", "x", "y", "z", "roll", "pitch", "yaw", "fx", "fy", "fz", "mx", "my", "mz"].map((item, index) => [
    item,
    index
  ])
);

function componentOrder(component: string): number {
  return COMPONENT_ORDER.get(component.toLowerCase()) ?? Number.MAX_SAFE_INTEGER;
}

function splitFeatureName(name: string): { prefix: string; leaf: string } {
  const dot = name.lastIndexOf(".");
  if (dot < 0) {
    return { prefix: "", leaf: name };
  }
  return { prefix: name.slice(0, dot), leaf: name.slice(dot + 1) };
}

function joinFeatureName(prefix: string, leaf: string): string {
  return prefix ? `${prefix}.${leaf}` : leaf;
}

function parseVectorComponent(name: string): { key: string; name: string; component: string } | null {
  const { prefix, leaf } = splitFeatureName(name);

  const compactForce = leaf.match(/^([fm])([xyz])$/i);
  if (compactForce) {
    return {
      key: joinFeatureName(prefix, compactForce[1]),
      name: joinFeatureName(prefix, compactForce[1]),
      component: compactForce[2].toLowerCase()
    };
  }

  const touchSummary = leaf.match(/^(.*?_f)([xyz])(_.*)$/i);
  if (touchSummary) {
    return {
      key: joinFeatureName(prefix, `${touchSummary[1]}${touchSummary[3]}`),
      name: joinFeatureName(prefix, `${touchSummary[1]}${touchSummary[3]}`),
      component: touchSummary[2].toLowerCase()
    };
  }

  const eulerAngle = leaf.match(/^(roll|pitch|yaw)(_.+)$/i);
  if (eulerAngle) {
    return {
      key: joinFeatureName(prefix, `rpy${eulerAngle[2]}`),
      name: joinFeatureName(prefix, `rpy${eulerAngle[2]}`),
      component: eulerAngle[1].toLowerCase()
    };
  }

  const separated = leaf.match(/^(.*?)[_-]([wxyz])(_.*)?$/i);
  if (separated) {
    return {
      key: joinFeatureName(prefix, `${separated[1]}${separated[3] ?? ""}`),
      name: joinFeatureName(prefix, `${separated[1]}${separated[3] ?? ""}`),
      component: separated[2].toLowerCase()
    };
  }

  return null;
}

export function groupSeriesNames(names: string[]): SeriesNameGroup[] {
  const candidates = new Map<string, SeriesNameGroup>();
  const candidateByDim = new Map<number, string>();

  names.forEach((name, dim) => {
    const parsed = parseVectorComponent(name);
    if (!parsed) return;
    const group = candidates.get(parsed.key) ?? { key: parsed.key, name: parsed.name, items: [] };
    group.items.push({ name, dim, component: parsed.component });
    candidates.set(parsed.key, group);
    candidateByDim.set(dim, parsed.key);
  });

  const vectorGroups = new Set(
    Array.from(candidates)
      .filter(([, group]) => new Set(group.items.map((item) => item.component.toLowerCase())).size >= 2)
      .map(([key]) => key)
  );
  const emitted = new Set<string>();
  const groups: SeriesNameGroup[] = [];

  names.forEach((name, dim) => {
    const candidateKey = candidateByDim.get(dim);
    if (!candidateKey || !vectorGroups.has(candidateKey)) {
      groups.push({ key: name, name, items: [{ name, dim, component: "" }] });
      return;
    }
    if (emitted.has(candidateKey)) return;
    const group = candidates.get(candidateKey);
    if (!group) return;
    emitted.add(candidateKey);
    groups.push({
      ...group,
      items: [...group.items].sort((a, b) => componentOrder(a.component) - componentOrder(b.component) || a.dim - b.dim)
    });
  });

  return groups;
}

function computeBounds(series: Series[]): Bounds {
  let min = Infinity;
  let max = -Infinity;
  for (const entry of series) {
    for (const value of entry.values) {
      if (!Number.isFinite(value)) continue;
      if (value < min) min = value;
      if (value > max) max = value;
    }
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
  group,
  currentFrame,
  totalFrames,
  onSeek,
  rowHeight
}: {
  group: { name: string; series: Series[] };
  currentFrame: number;
  totalFrames: number;
  onSeek: (frame: number) => void;
  rowHeight: number;
}) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const bounds = useMemo(() => computeBounds(group.series), [group.series]);
  const currentValues = useMemo(() => group.series.map((series) => series.values[currentFrame]), [group.series, currentFrame]);

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

    for (const series of group.series) {
      ctx.strokeStyle = series.color;
      ctx.lineWidth = group.series.length > 1 ? 1.3 : 1.2;
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
    }

    if (totalFrames > 0) {
      const px = Math.min(width - 1, Math.max(0, currentFrame * xScale));
      ctx.strokeStyle = "#111827";
      ctx.lineWidth = 1;
      ctx.beginPath();
      ctx.moveTo(px, 0);
      ctx.lineTo(px, height);
      ctx.stroke();
      for (const [index, currentValue] of currentValues.entries()) {
        const series = group.series[index];
        if (!Number.isFinite(currentValue)) continue;
        const py = yPad + ((bounds.max - currentValue) / span) * usable;
        ctx.fillStyle = series.color;
        ctx.beginPath();
        ctx.arc(px, py, 2.5, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }, [bounds, currentFrame, currentValues, group.series, totalFrames]);

  function handlePointer(event: ReactPointerEvent<HTMLCanvasElement>) {
    if (event.buttons === 0 && event.type !== "pointerdown") return;
    const rect = event.currentTarget.getBoundingClientRect();
    const ratio = (event.clientX - rect.left) / Math.max(rect.width, 1);
    const frame = Math.round(Math.max(0, Math.min(1, ratio)) * Math.max(totalFrames - 1, 0));
    onSeek(frame);
  }

  return (
    <div className="series-row">
      <div className="series-label" title={group.series.map((series) => series.name).join("\n")}>
        <span>{group.name}</span>
        <div className="series-values">
          {group.series.map((series, index) => {
            const currentValue = currentValues[index];
            return (
              <strong key={series.name} style={{ color: series.color }}>
                {series.component ? `${series.component} ` : ""}
                {Number.isFinite(currentValue) ? currentValue.toFixed(3) : "—"}
              </strong>
            );
          })}
        </div>
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
  const groups = useMemo(() => {
    return groupSeriesNames(names).map((group) => {
      return {
        key: group.key,
        name: group.name,
        series: group.items.map((item, index) => {
          const values: number[] = new Array(totalFrames);
          for (let i = 0; i < totalFrames; i++) {
            values[i] = pickValue(i, item.dim);
          }
          return {
            name: item.name,
            values,
            color: SERIES_COLORS[index % SERIES_COLORS.length],
            component: item.component
          };
        })
      };
    });
  }, [names, pickValue, totalFrames]);

  return (
    <section className="panel series-panel">
      <div className="panel-heading">
        <h2>{title}</h2>
        <span>{names.length} dims · {groups.length} rows · {totalFrames} frames</span>
      </div>
      <div className="series-list">
        {groups.map((entry) => (
          <SeriesRow
            key={entry.key}
            group={entry}
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
