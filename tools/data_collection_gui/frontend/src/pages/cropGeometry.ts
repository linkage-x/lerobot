/**
 * Crop rectangle math for the Camera Crop picker, in *source pixels*.
 *
 * Every rectangle that leaves this module is already legal for the exporter: even x/y/w/h
 * (H.264 chroma subsampling refuses odd geometry) and inside the frame. Doing it here rather
 * than at submit time is what lets the operator drag freely -- the box snaps as it moves
 * instead of the build failing minutes later on a rectangle that was one pixel wide.
 */

export type CropRect = { x: number; y: number; w: number; h: number };

/** Corner and edge grips, named by compass direction. */
export type CropHandle = "nw" | "n" | "ne" | "e" | "se" | "s" | "sw" | "w";

/** What a pointer press on the overlay starts: a grip, a whole-box move, or a fresh box. */
export type CropDragMode = CropHandle | "move" | "new";

export type CropPoint = { x: number; y: number };

/** Smallest crop the exporter accepts. Even, and large enough to still be visible. */
export const CROP_MIN_SIZE = 16;

export function evenFloor(value: number): number {
  return Math.max(0, Math.floor(value / 2) * 2);
}

function evenRound(value: number): number {
  return Math.max(0, Math.round(value / 2) * 2);
}

export function fullFrameCrop(frameW: number, frameH: number): CropRect {
  return { x: 0, y: 0, w: evenFloor(frameW), h: evenFloor(frameH) };
}

export function isFullFrame(rect: CropRect, frameW: number, frameH: number): boolean {
  const full = fullFrameCrop(frameW, frameH);
  return rect.x === 0 && rect.y === 0 && rect.w === full.w && rect.h === full.h;
}

/**
 * Snap a rectangle to even pixels and pull it inside the frame.
 *
 * A box dragged past the edge is clamped rather than rejected: the operator is aiming at the
 * subject, not at the frame bounds, and an out-of-range rectangle is only ever discovered by
 * the exporter refusing the build.
 */
export function normalizeCrop(rect: CropRect, frameW: number, frameH: number): CropRect {
  const maxW = evenFloor(frameW);
  const maxH = evenFloor(frameH);
  if (maxW < 2 || maxH < 2) return { x: 0, y: 0, w: 0, h: 0 };
  const minW = Math.min(CROP_MIN_SIZE, maxW);
  const minH = Math.min(CROP_MIN_SIZE, maxH);
  // The origin is anchored first and the size trimmed to what is left of the frame. Clamping
  // the other way round would slide a box that was dragged past the right edge back to the
  // left, moving the ROI the operator was pointing at instead of stopping it at the edge.
  const x = Math.min(maxW - minW, Math.max(0, evenRound(rect.x)));
  const y = Math.min(maxH - minH, Math.max(0, evenRound(rect.y)));
  const w = Math.min(maxW - x, Math.max(minW, evenRound(rect.w)));
  const h = Math.min(maxH - y, Math.max(minH, evenRound(rect.h)));
  return { x, y, w, h };
}

/** The box spanned by a press point and the current pointer position. */
export function cropFromCorners(a: CropPoint, b: CropPoint, frameW: number, frameH: number): CropRect {
  return normalizeCrop(
    {
      x: Math.min(a.x, b.x),
      y: Math.min(a.y, b.y),
      w: Math.abs(b.x - a.x),
      h: Math.abs(b.y - a.y)
    },
    frameW,
    frameH
  );
}

/**
 * Slide a box without resizing it.
 *
 * The size is preserved on purpose: dragging a chosen ROI to the edge should stop there, not
 * shrink. `normalizeCrop` would shrink it, so the offset is clamped before it is applied.
 */
export function moveCrop(rect: CropRect, dx: number, dy: number, frameW: number, frameH: number): CropRect {
  const base = normalizeCrop(rect, frameW, frameH);
  const maxW = evenFloor(frameW);
  const maxH = evenFloor(frameH);
  return {
    ...base,
    x: Math.min(maxW - base.w, Math.max(0, evenRound(base.x + dx))),
    y: Math.min(maxH - base.h, Math.max(0, evenRound(base.y + dy)))
  };
}

/** Drag one grip; the opposite edge (or corner) stays put. */
export function resizeCrop(
  rect: CropRect,
  handle: CropHandle,
  point: CropPoint,
  frameW: number,
  frameH: number
): CropRect {
  const base = normalizeCrop(rect, frameW, frameH);
  let left = base.x;
  let top = base.y;
  let right = base.x + base.w;
  let bottom = base.y + base.h;
  if (handle.includes("w")) left = point.x;
  if (handle.includes("e")) right = point.x;
  if (handle.includes("n")) top = point.y;
  if (handle.includes("s")) bottom = point.y;
  // Past-the-anchor drags flip rather than collapse, which is what a pointer that has crossed
  // the opposite edge looks like to the hand holding it.
  return cropFromCorners({ x: left, y: top }, { x: right, y: bottom }, frameW, frameH);
}

/**
 * What a press at `point` should start. `tolerance` is the grip radius in source pixels, so the
 * caller converts from the CSS pixels the pointer actually moved in.
 */
export function cropHitTest(rect: CropRect, point: CropPoint, tolerance: number): CropDragMode {
  const nearLeft = Math.abs(point.x - rect.x) <= tolerance;
  const nearRight = Math.abs(point.x - (rect.x + rect.w)) <= tolerance;
  const nearTop = Math.abs(point.y - rect.y) <= tolerance;
  const nearBottom = Math.abs(point.y - (rect.y + rect.h)) <= tolerance;
  const insideX = point.x >= rect.x - tolerance && point.x <= rect.x + rect.w + tolerance;
  const insideY = point.y >= rect.y - tolerance && point.y <= rect.y + rect.h + tolerance;
  if (insideX && insideY) {
    const vertical = nearTop ? "n" : nearBottom ? "s" : "";
    const horizontal = nearLeft ? "w" : nearRight ? "e" : "";
    const handle = `${vertical}${horizontal}`;
    if (handle) return handle as CropHandle;
    if (point.x > rect.x && point.x < rect.x + rect.w && point.y > rect.y && point.y < rect.y + rect.h) {
      return "move";
    }
  }
  return "new";
}

export const CROP_HANDLES: CropHandle[] = ["nw", "n", "ne", "e", "se", "s", "sw", "w"];

const HANDLE_CURSORS: Record<CropHandle, string> = {
  nw: "nwse-resize",
  n: "ns-resize",
  ne: "nesw-resize",
  e: "ew-resize",
  se: "nwse-resize",
  s: "ns-resize",
  sw: "nesw-resize",
  w: "ew-resize"
};

export function cropCursor(mode: CropDragMode): string {
  if (mode === "move") return "move";
  if (mode === "new") return "crosshair";
  return HANDLE_CURSORS[mode];
}

/**
 * The crop the FR3 side camera usually wants: the right-hand slab of the frame that holds the
 * workspace, with the empty ceiling above it dropped. Kept as a starting point to drag from,
 * not as an answer -- the rig moves, and the picker is there to see whether it still fits.
 */
export function sideRoiCrop(key: string, frameW: number, frameH: number): CropRect {
  if (!key.endsWith(".side")) {
    return fullFrameCrop(frameW, frameH);
  }
  if (frameW >= 640 && frameH >= 480) {
    return normalizeCrop(
      { x: 224, y: 0, w: Math.min(416, frameW - 224), h: Math.min(346, frameH) },
      frameW,
      frameH
    );
  }
  const x = evenFloor(frameW * 0.35);
  return normalizeCrop({ x, y: 0, w: frameW - x, h: frameH * 0.72 }, frameW, frameH);
}

/** Percentage box for CSS, so the overlay tracks the rendered image at any display size. */
export function cropStyleBox(
  rect: CropRect,
  frameW: number,
  frameH: number
): { left: string; top: string; width: string; height: string } {
  const pctX = (value: number) => `${frameW > 0 ? (value / frameW) * 100 : 0}%`;
  const pctY = (value: number) => `${frameH > 0 ? (value / frameH) * 100 : 0}%`;
  return { left: pctX(rect.x), top: pctY(rect.y), width: pctX(rect.w), height: pctY(rect.h) };
}
