import { describe, expect, it } from "vitest";

import { windowForPlot } from "./tableWindow";

/** A plot frame of the shape both maps build: linear, base x rightward, base y upward. */
function frameFor(centreX: number, centreY: number, scale: number, originX: number, originY: number) {
  return {
    toScreenX: (x: number) => originX + (x - centreX) * scale,
    toScreenY: (y: number) => originY - (y - centreY) * scale,
    toWorldX: (screenX: number) => centreX + (screenX - originX) / scale,
    toWorldY: (screenY: number) => centreY + (originY - screenY) / scale
  };
}

describe("windowForPlot", () => {
  const frame = frameFor(0.44, 0.0, 370, 180, 190);
  const rect = { left: 28, top: 28, right: 332, bottom: 332 };

  it("names the rectangle whose corners are the plot's own corners", () => {
    const window = windowForPlot(frame, rect);

    // The whole contract: the backdrop covers this window, so the window has to map back onto
    // the pixels the points are drawn in. A round trip that is off by anything is an offset
    // between the picture and every point on it.
    expect(frame.toScreenX(window.minX)).toBeCloseTo(rect.left, 9);
    expect(frame.toScreenX(window.maxX)).toBeCloseTo(rect.right, 9);
    expect(frame.toScreenY(window.maxY)).toBeCloseTo(rect.top, 9);
    expect(frame.toScreenY(window.minY)).toBeCloseTo(rect.bottom, 9);
  });

  it("puts base y's minimum at the bottom of the plot, not the top", () => {
    const window = windowForPlot(frame, rect);

    expect(window.minY).toBeLessThan(window.maxY);
    expect(window.minX).toBeLessThan(window.maxX);
  });

  it("handles a plot box that is not square", () => {
    const window = windowForPlot(frame, { left: 52, top: 16, right: 444, bottom: 420 });

    // Taller than it is wide in pixels, and the window has to be taller than it is wide in
    // metres by the same ratio -- the served image is stretched to the box either way, so a
    // mismatch here is a squashed table.
    expect((window.maxY - window.minY) / (window.maxX - window.minX)).toBeCloseTo(404 / 392, 9);
  });
});
