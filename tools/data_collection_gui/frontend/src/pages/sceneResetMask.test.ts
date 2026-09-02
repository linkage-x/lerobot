import { describe, expect, it } from "vitest";

import { strokesAfterLoad } from "./sceneResetMask";

const stroke = (x: number) => ({ x, y: 0, radiusM: 0.035 });

describe("strokesAfterLoad", () => {
  it("fills an empty canvas with the stored region", () => {
    expect(strokesAfterLoad([], [stroke(0.4)])).toEqual([stroke(0.4)]);
  });

  it("leaves a canvas the operator has already drawn on alone", () => {
    // The failure this prevents: the load lands mid-stroke and the reset then samples its place
    // point from a region nobody on this shift drew.
    const drawn = [stroke(0.5)];

    expect(strokesAfterLoad(drawn, [stroke(0.4)])).toBe(drawn);
  });

  it("accepts an empty stored region as an answer", () => {
    // Cleared is a thing an operator said, not a thing that failed to load.
    expect(strokesAfterLoad([], [])).toEqual([]);
  });
});
