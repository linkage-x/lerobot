import { describe, expect, it } from "vitest";
import type { RigCheckReport } from "../types";
import {
  baselineSummary,
  canUpdateBaseline,
  formatImpact,
  formatShift,
  rigCheckRows,
  shouldOfferRecalibration,
} from "./rigCheck";

function report(overrides: Partial<RigCheckReport> = {}): RigCheckReport {
  return {
    generated_utc: "2026-08-05T02:00:00Z",
    overall: "ok",
    guidance: "相机未移动，外参仍然有效。",
    moved_cameras: [],
    thresholds_px: { warn: 2, fail: 5 },
    cameras: {},
    ...overrides,
  };
}

describe("rig self-check presentation", () => {
  it("offers recalibration only for a confirmed move", () => {
    expect(shouldOfferRecalibration(report({ overall: "moved" }))).toBe(true);
    for (const overall of ["ok", "partial", "suspect", "inconclusive", "unknown"] as const) {
      expect(shouldOfferRecalibration(report({ overall }))).toBe(false);
    }
  });

  it("refuses to re-baseline over a detected move", () => {
    // Capturing a new baseline after a real move would record the moved rig as
    // the reference and erase the evidence.
    expect(canUpdateBaseline(report({ overall: "moved" }))).toBe(false);
    expect(canUpdateBaseline(report({ overall: "suspect" }))).toBe(true);
    expect(canUpdateBaseline(null)).toBe(true);
  });

  it("shows the equivalent angle next to the pixel shift", () => {
    expect(
      formatShift({ status: "measured", verdict: "moved", shift_px_median: 11.93, equivalent_rotation_deg: 0.678 }),
    ).toBe("11.93 px · ≈0.678°");
  });

  it("shows a dash rather than a number when nothing was measured", () => {
    expect(formatShift({ status: "unknown", verdict: "unknown", reason: "too dark" })).toBe("—");
    expect(formatImpact({ status: "unknown", verdict: "unknown" })).toBe("");
  });

  it("keeps the reason visible for cameras that could not be checked", () => {
    const rows = rigCheckRows(
      report({
        overall: "partial",
        cameras: {
          cam_06: { status: "measured", verdict: "ok", shift_px_median: 0.04 },
          cam_09: { status: "unknown", verdict: "unknown", reason: "matched region covers only 8% of the frame" },
        },
      }),
    );
    expect(rows.map((r) => r.camera)).toEqual(["cam_06", "cam_09"]);
    expect(rows[1].note).toContain("8%");
    expect(rows[1].shift).toBe("—");
  });

  it("says plainly when there is no baseline to compare against", () => {
    expect(baselineSummary(null)).toContain("尚未采集基线");
    expect(
      baselineSummary(
        report({
          baseline: {
            exists: true,
            captured_at: "2026-08-04T15:03:00Z",
            intrinsics_run: "thor_gmsl2_selfcal_0804_fisheye_intrinsics",
          },
        }),
      ),
    ).toContain("thor_gmsl2_selfcal_0804_fisheye_intrinsics");
  });
});
