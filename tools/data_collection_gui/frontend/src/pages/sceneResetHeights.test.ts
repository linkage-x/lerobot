import { describe, expect, it } from "vitest";

import panelSource from "./SceneResetPanel.tsx?raw";
import { PLACE_Z_M, TABLE_Z_M, dropWarning } from "./sceneResetHeights";

describe("TABLE_Z_M", () => {
  it("is a height on the table, not a height above it", () => {
    // The regression these constants come from: the place height was 0.55, so the reset
    // released the peg 50 cm over the table. Any plausible table-top height is a few
    // centimetres, and one that is not is a fall.
    expect(TABLE_Z_M).toBeGreaterThan(0);
    expect(TABLE_Z_M).toBeLessThan(0.1);
  });
});

describe("PLACE_Z_M", () => {
  it("is a height on the table, not a height above it", () => {
    expect(PLACE_Z_M).toBeGreaterThan(0);
    expect(PLACE_Z_M).toBeLessThan(0.1);
  });

  it("is the 55 mm the rig is set up to release from", () => {
    // Pinned rather than derived: the release height is a property of the peg and the fingers,
    // and the whole point of this constant is that nothing recalculates it per dataset.
    expect(PLACE_Z_M).toBeCloseTo(0.055, 6);
  });

  it("does not, with the default pick, order a drop", () => {
    // The two defaults have to be a pair that the panel's own warning is happy with, or every
    // fresh session opens on a banner nobody can act on.
    expect(dropWarning(TABLE_Z_M, PLACE_Z_M)).toBe("");
  });
});

describe("the place height in the panel", () => {
  it("starts at the default rather than at a measurement", () => {
    expect(panelSource).toContain("useState(PLACE_Z_M.toFixed(3))");
  });

  it("is not written by the effect that applies the measured pick", () => {
    // The measurement lands on the pick because only the recording knows where the peg is.
    // Letting the same effect touch the place height is how a dataset loading in the background
    // silently changes the height the gripper opens at.
    const effect = panelSource.slice(panelSource.indexOf("if (!measuredPick) return;"));

    expect(effect.slice(0, effect.indexOf("}, [measuredPickKey"))).not.toContain("setTargetZ");
  });

  it("cannot be copied from the measured place height", () => {
    expect(panelSource).not.toContain("Use the measured place height");
    expect(panelSource).not.toContain("setTargetZ(measuredPick");
  });
});

describe("dropWarning", () => {
  it("says nothing when the peg is put back on the surface it came from", () => {
    expect(dropWarning(0.051, 0.051)).toBe("");
  });

  it("says nothing about the last few millimetres of settling", () => {
    expect(dropWarning(0.035, 0.05)).toBe("");
  });

  it("reports the fall the operator is about to order", () => {
    // The height that shipped: pick on the table, place half a metre over it.
    expect(dropWarning(0.051, 0.55)).toContain("499 mm");
  });

  it("says nothing about placing lower than the pick", () => {
    // Below the surface is the arm pressing down, which the reach check and the fence own.
    expect(dropWarning(0.051, 0.03)).toBe("");
  });

  it("says nothing when a field is mid-edit and parses to nothing", () => {
    expect(dropWarning(Number.NaN, 0.55)).toBe("");
  });
});
