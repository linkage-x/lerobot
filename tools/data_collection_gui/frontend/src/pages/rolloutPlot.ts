// Pure helpers of the Rollout page's small SVG plots (rolloutPlot.test.ts).

export type Box = { x0: number; x1: number; y0: number; y1: number };

/** A box around every finite point, padded by `pad` of its larger side (a square-ish view). */
export function boundsOf(points: (number | null)[][], pad = 0.1): Box | null {
  const finite = points.filter((p) => p.length >= 2 && p[0] != null && p[1] != null && Number.isFinite(p[0]) && Number.isFinite(p[1])) as number[][];
  if (finite.length === 0) return null;
  let x0 = Math.min(...finite.map((p) => p[0]));
  let x1 = Math.max(...finite.map((p) => p[0]));
  let y0 = Math.min(...finite.map((p) => p[1]));
  let y1 = Math.max(...finite.map((p) => p[1]));
  const side = Math.max(x1 - x0, y1 - y0, 1e-3);
  const cx = (x0 + x1) / 2;
  const cy = (y0 + y1) / 2;
  const half = (side * (1 + 2 * pad)) / 2;
  x0 = cx - half;
  x1 = cx + half;
  y0 = cy - half;
  y1 = cy + half;
  return { x0, x1, y0, y1 };
}

/** SVG path of `points` in a `w` x `h` box (y up), skipping gaps. */
export function pathOf(points: (number | null)[][], box: Box, w: number, h: number): string {
  let d = "";
  let pen = false;
  for (const p of points) {
    if (p.length < 2 || p[0] == null || p[1] == null || !Number.isFinite(p[0]) || !Number.isFinite(p[1])) {
      pen = false;
      continue;
    }
    const x = ((p[0] - box.x0) / (box.x1 - box.x0)) * w;
    const y = h - ((p[1] - box.y0) / (box.y1 - box.y0)) * h;
    d += `${pen ? "L" : "M"}${x.toFixed(1)},${y.toFixed(1)} `;
    pen = true;
  }
  return d.trim();
}

/** Series (index, value) as plot points for a by-step curve; nulls become gaps. */
export function stepSeries(values: (number | null)[]): (number | null)[][] {
  return values.map((v, i) => [i + 1, v]);
}
