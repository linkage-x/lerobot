import { describe, expect, it } from "vitest";

import { offLineDistance, suggestedTargets } from "./tableTargets";

const CENTRE: [number, number] = [0.36, -0.133];
const SPREAD = 0.08;
// The gateway refuses a triple whose middle point is under 5 mm off the line through the other
// two, so a suggestion set that only just clears that is a suggestion set that fails on the rig.
const MIN_OFF_LINE_M = 0.005;

function quadruples<T>(items: T[]): T[][] {
  const out: T[][] = [];
  for (let a = 0; a < items.length; a += 1)
    for (let b = a + 1; b < items.length; b += 1)
      for (let c = b + 1; c < items.length; c += 1)
        for (let d = c + 1; d < items.length; d += 1) out.push([items[a], items[b], items[c], items[d]]);
  return out;
}

describe("suggestedTargets", () => {
  it("offers a set in which any four points can be the whole calibration", () => {
    // One unreachable corner is the normal case, not the exception: whichever four survive have
    // to fit a plane on their own.
    const targets = suggestedTargets(CENTRE, SPREAD);
    expect(targets).toHaveLength(5);

    for (const quad of quadruples(targets)) {
      for (let i = 0; i < 4; i += 1)
        for (let j = i + 1; j < 4; j += 1)
          for (let k = j + 1; k < 4; k += 1)
            expect(offLineDistance(quad[i], quad[j], quad[k])).toBeGreaterThan(MIN_OFF_LINE_M);
    }
  });

  it("would have caught the centre point that could not be fitted", () => {
    const corner: [number, number] = [0.44, -0.053];
    const opposite = { x: 0.28, y: -0.213 };
    const centre = { x: 0.36, y: -0.133 };

    expect(offLineDistance({ x: corner[0], y: corner[1] }, opposite, centre)).toBeCloseTo(0, 9);
  });

  it("keeps the spread it is given", () => {
    const targets = suggestedTargets(CENTRE, 0.05);

    expect(targets[0]).toEqual({ x: 0.36 + 0.05, y: -0.133 + 0.05 });
    expect(targets[4].y).toBeCloseTo(-0.133, 9);
  });
});
