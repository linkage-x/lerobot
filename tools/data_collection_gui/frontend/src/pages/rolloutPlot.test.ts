import { describe, expect, it } from "vitest";
import { boundsOf, pathOf, stepSeries } from "./rolloutPlot";

describe("rollout plots", () => {
  it("bounds every finite point, square and padded", () => {
    const box = boundsOf([[0, 0], [2, 1], [null, 5], [Number.NaN, 3]], 0);
    expect(box).toEqual({ x0: 0, x1: 2, y0: -0.5, y1: 1.5 });
    expect(boundsOf([[null, null]])).toBeNull();
  });

  it("draws y up and breaks the line at gaps", () => {
    const box = { x0: 0, x1: 10, y0: 0, y1: 10 };
    expect(pathOf([[0, 0], [10, 10], [null, 1], [5, 5]], box, 100, 100)).toBe("M0.0,100.0 L100.0,0.0 M50.0,50.0");
  });

  it("numbers chunk steps from 1", () => {
    expect(stepSeries([2, null, 4])).toEqual([[1, 2], [2, null], [3, 4]]);
  });
});
