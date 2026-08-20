import { describe, expect, it } from "vitest";

import {
  CROP_MIN_SIZE,
  cropFromCorners,
  cropHitTest,
  fullFrameCrop,
  isFullFrame,
  moveCrop,
  normalizeCrop,
  resizeCrop,
  sideRoiCrop
} from "./cropGeometry";

const W = 640;
const H = 480;

describe("normalizeCrop", () => {
  it("snaps to even pixels, which is what H.264 can actually encode", () => {
    // An odd crop is not refused by the picker but by the exporter, minutes into a build.
    expect(normalizeCrop({ x: 101, y: 7, w: 315, h: 201 }, W, H)).toEqual({
      x: 102,
      y: 8,
      w: 316,
      h: 202
    });
  });

  it("trims a box dragged past the right edge instead of sliding it left", () => {
    // Sliding would move the ROI away from what the operator was pointing at.
    expect(normalizeCrop({ x: 400, y: 0, w: 400, h: 480 }, W, H)).toEqual({
      x: 400,
      y: 0,
      w: 240,
      h: 480
    });
  });

  it("keeps the box at least CROP_MIN_SIZE so it stays grabbable", () => {
    const rect = normalizeCrop({ x: 100, y: 100, w: 1, h: 0 }, W, H);
    expect(rect.w).toBe(CROP_MIN_SIZE);
    expect(rect.h).toBe(CROP_MIN_SIZE);
  });

  it("reports an empty rect for a camera with no known size", () => {
    expect(normalizeCrop({ x: 0, y: 0, w: 10, h: 10 }, 0, 0)).toEqual({ x: 0, y: 0, w: 0, h: 0 });
  });
});

describe("cropFromCorners", () => {
  it("spans the two points whichever way the drag went", () => {
    const forward = cropFromCorners({ x: 100, y: 80 }, { x: 400, y: 300 }, W, H);
    const backward = cropFromCorners({ x: 400, y: 300 }, { x: 100, y: 80 }, W, H);
    expect(forward).toEqual(backward);
    expect(forward).toEqual({ x: 100, y: 80, w: 300, h: 220 });
  });

  it("clamps a drag that ran off the frame", () => {
    expect(cropFromCorners({ x: -50, y: -30 }, { x: 900, y: 700 }, W, H)).toEqual({
      x: 0,
      y: 0,
      w: W,
      h: H
    });
  });
});

describe("moveCrop", () => {
  it("stops at the edge without shrinking", () => {
    const rect = { x: 300, y: 200, w: 300, h: 240 };
    const moved = moveCrop(rect, 500, 500, W, H);
    expect(moved).toEqual({ x: 340, y: 240, w: 300, h: 240 });
  });

  it("stops at the origin without shrinking", () => {
    expect(moveCrop({ x: 40, y: 40, w: 200, h: 160 }, -500, -500, W, H)).toEqual({
      x: 0,
      y: 0,
      w: 200,
      h: 160
    });
  });
});

describe("resizeCrop", () => {
  const rect = { x: 200, y: 100, w: 200, h: 200 };

  it("moves the dragged edge and leaves the opposite one alone", () => {
    expect(resizeCrop(rect, "e", { x: 500, y: 0 }, W, H)).toEqual({ x: 200, y: 100, w: 300, h: 200 });
    expect(resizeCrop(rect, "n", { x: 0, y: 40 }, W, H)).toEqual({ x: 200, y: 40, w: 200, h: 260 });
  });

  it("moves both edges of a corner grip", () => {
    expect(resizeCrop(rect, "se", { x: 460, y: 380 }, W, H)).toEqual({
      x: 200,
      y: 100,
      w: 260,
      h: 280
    });
  });

  it("flips when the pointer crosses the anchored edge", () => {
    // The hand has crossed the far edge; collapsing to zero width would drop the box instead.
    expect(resizeCrop(rect, "w", { x: 500, y: 0 }, W, H)).toEqual({
      x: 400,
      y: 100,
      w: 100,
      h: 200
    });
  });

  it("clamps a grip dragged out of the frame", () => {
    expect(resizeCrop(rect, "e", { x: 5000, y: 0 }, W, H).w).toBe(W - 200);
  });
});

describe("cropHitTest", () => {
  const rect = { x: 200, y: 100, w: 200, h: 200 };

  it("prefers a grip over a move near the border", () => {
    expect(cropHitTest(rect, { x: 202, y: 102 }, 8)).toBe("nw");
    expect(cropHitTest(rect, { x: 300, y: 300 }, 8)).toBe("s");
    expect(cropHitTest(rect, { x: 400, y: 200 }, 8)).toBe("e");
  });

  it("moves from the middle and draws a new box from outside", () => {
    expect(cropHitTest(rect, { x: 300, y: 200 }, 8)).toBe("move");
    expect(cropHitTest(rect, { x: 50, y: 50 }, 8)).toBe("new");
  });
});

describe("sideRoiCrop", () => {
  it("keeps the workspace slab of a 640x480 side view", () => {
    expect(sideRoiCrop("observation.images.side", W, H)).toEqual({ x: 224, y: 0, w: 416, h: 346 });
  });

  it("leaves any other camera at full frame", () => {
    const rect = sideRoiCrop("observation.images.ee", W, H);
    expect(rect).toEqual(fullFrameCrop(W, H));
    expect(isFullFrame(rect, W, H)).toBe(true);
  });
});
