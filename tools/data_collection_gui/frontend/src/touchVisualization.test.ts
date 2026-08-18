import { describe, expect, it } from "vitest";

import {
  M2020_TOUCH_UNIT_COUNT,
  PAXINI_TOUCH_POINTS,
  PAXINI_TOUCH_UNIT_COUNT,
  TOUCH_MODEL_M2020,
  TOUCH_MODEL_PAXINI,
  touchCellColor,
  touchLayoutForCount,
  touchLayoutForSample,
  touchModelForSample,
  touchSampleActivePoints,
  touchSampleHasShear,
  touchScaleFromSamples,
  touchShearArrow,
} from "./touchVisualization";

describe("touchVisualization", () => {
  it("keeps the Paxini map matched to the 239 vendor XYZ coordinates", () => {
    const layout = touchLayoutForCount(PAXINI_TOUCH_UNIT_COUNT);
    const xs = PAXINI_TOUCH_POINTS.map((point) => point.xMm);
    const ys = PAXINI_TOUCH_POINTS.map((point) => point.yMm);

    expect(PAXINI_TOUCH_POINTS).toHaveLength(239);
    expect(PAXINI_TOUCH_POINTS[0]).toMatchObject({ index: 1, xMm: -12.34316937, yMm: 0.08614981 });
    expect(PAXINI_TOUCH_POINTS[238]).toMatchObject({ index: 239, xMm: 12.34367992, yMm: 51.5705778 });
    expect(Math.min(...xs)).toBeCloseTo(-12.34367992, 6);
    expect(Math.max(...xs)).toBeCloseTo(12.34367992, 6);
    expect(Math.min(...ys)).toBeCloseTo(0.08614981, 6);
    expect(Math.max(...ys)).toBeCloseTo(51.86232744, 6);
    expect(layout.label).toContain("XYZ map");
  });

  it("lays the M2020 patch out as the datasheet 3x3", () => {
    const layout = touchLayoutForCount(M2020_TOUCH_UNIT_COUNT);

    expect(M2020_TOUCH_UNIT_COUNT).toBe(9);
    expect(layout.columns).toBe(3);
    expect(layout.rowLengths).toEqual([3, 3, 3]);
    expect(layout.label).toContain("M2020");
  });

  it("trusts the model tag over array length when identifying a pad", () => {
    // The BOX SDK hands every pad over in one fixed 239-slot array, so an
    // untouched Paxini frame and a zero-padded M2020 frame look identical by
    // length alone -- only the tag separates them.
    const tagged = { fz: new Array(239).fill(0), model: TOUCH_MODEL_M2020, points: 9 };
    expect(touchModelForSample(tagged)).toBe(TOUCH_MODEL_M2020);
    expect(touchLayoutForSample(tagged)?.columns).toBe(3);

    const untagged239 = { fz: new Array(239).fill(0) };
    expect(touchModelForSample(untagged239)).toBe(TOUCH_MODEL_PAXINI);
    expect(touchLayoutForSample(untagged239)?.unitCount).toBe(PAXINI_TOUCH_UNIT_COUNT);

    const untagged9 = { fz: new Array(9).fill(0) };
    expect(touchModelForSample(untagged9)).toBe(TOUCH_MODEL_M2020);
    expect(touchLayoutForSample(untagged9)?.columns).toBe(3);

    expect(touchLayoutForSample({ fz: [] })).toBeNull();
    expect(touchModelForSample({ fz: new Array(64).fill(0) })).toBeNull();
  });

  it("uses the dense 50 x 10 layout for 500-cell DAS tactile images", () => {
    const layout = touchLayoutForCount(500);

    expect(layout.columns).toBe(10);
    expect(layout.rowLengths).toHaveLength(50);
    expect(layout.rowLengths.reduce((total, length) => total + length, 0)).toBe(500);
  });

  it("detects shear channels and colors them with hue", () => {
    const sample = { fz: [0, 3], fx: [4, 0], fy: [0, 0] };
    const scale = touchScaleFromSamples([sample]);

    expect(touchSampleHasShear(sample)).toBe(true);
    expect(touchSampleActivePoints(sample)).toBe(2);
    expect(touchCellColor(3, 4, 0, scale).startsWith("hsl(")).toBe(true);
    expect(touchCellColor(3, 0, 0, scale).startsWith("rgb(")).toBe(true);
  });

  it("maps tactile shear to an xy-plane arrow", () => {
    const scale = touchScaleFromSamples([{ fz: [0, 0], fx: [4, 0], fy: [0, 4] }]);

    expect(touchShearArrow(4, 0, scale)).toMatchObject({ angleDeg: -0, lengthPx: 24, opacity: 0.92 });
    expect(touchShearArrow(0, 4, scale)).toMatchObject({ angleDeg: -90, lengthPx: 24, opacity: 0.92 });
    expect(touchShearArrow(0, 0, scale)).toMatchObject({ angleDeg: 0, lengthPx: 0, opacity: 0 });
  });
});
