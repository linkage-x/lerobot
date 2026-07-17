import { describe, it, expect } from "vitest";
import {
  fmtN,
  fmtNm,
  fmtDuration,
  fmtTimestamp,
  ORIGIN_FORCE_LIMITS,
  DYNAMIC_FORCE_LIMITS,
} from "./config";

describe("formatting helpers handle null/NaN without faking zero", () => {
  it("renders em dash for null/NaN", () => {
    expect(fmtN(null)).toBe("—");
    expect(fmtN(NaN)).toBe("—");
    expect(fmtNm(undefined)).toBe("—");
    expect(fmtDuration(null)).toBe("—");
    expect(fmtTimestamp(null)).toBe("—");
  });
  it("formats forces and moments with fixed precision + unit", () => {
    expect(fmtN(0.5)).toBe("0.50 N");
    expect(fmtNm(0.012)).toBe("0.012 N·m");
  });
  it("formats durations across scales", () => {
    expect(fmtDuration(820)).toBe("820ms");
    expect(fmtDuration(45_000)).toBe("45s");
    expect(fmtDuration(4 * 60_000 + 12_000)).toBe("4m 12s");
    expect(fmtDuration(2 * 3_600_000 + 5 * 60_000)).toBe("2h 5m");
  });
});

describe("threshold constants are distinct between origin and dynamic", () => {
  it("origin zeroes Fz (abs) while dynamic targets a load", () => {
    expect(ORIGIN_FORCE_LIMITS.fz.mode).toBe("abs");
    expect(DYNAMIC_FORCE_LIMITS.fz.mode).toBe("target");
    if (DYNAMIC_FORCE_LIMITS.fz.mode === "target") {
      expect(DYNAMIC_FORCE_LIMITS.fz.targetN).toBe(-5.8);
      expect(DYNAMIC_FORCE_LIMITS.fz.tolN).toBe(0.5);
    }
  });
});
