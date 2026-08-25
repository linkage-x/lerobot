import { describe, expect, it } from "vitest";
import type { IntrinsicsCoverageCamera, IntrinsicsCoverageResponse } from "../types";
import {
  coverageNote,
  coverageRows,
  coverageSummary,
  coverageVerdict,
  formatFold,
  shouldOfferIntrinsicsRecapture,
} from "./intrinsicsCoverage";

function response(cameras: IntrinsicsCoverageCamera[]): IntrinsicsCoverageResponse {
  return {
    ok: true,
    run: "thor_gmsl2_selfcal_0804_fisheye_intrinsics",
    coverageTarget: 0.9,
    foldMarginWarnDeg: 5,
    cameras,
  };
}

// The real 0804 production set, which is what the panel renders today.
const PRODUCTION: IntrinsicsCoverageCamera[] = [
  { camera: "cam_06", coverage: 0.7876, foldMarginDeg: 2.1, foldsInsideFrame: false },
  { camera: "cam_07", coverage: 0.9234, foldMarginDeg: null, foldsInsideFrame: false },
  { camera: "cam_08", coverage: 0.9558, foldMarginDeg: null, foldsInsideFrame: false },
  { camera: "cam_09", coverage: 0.8625, foldMarginDeg: null, foldsInsideFrame: false },
  { camera: "cam_12", coverage: 0.8245, foldMarginDeg: null, foldsInsideFrame: false },
  { camera: "cam_13", coverage: 0.9013, foldMarginDeg: null, foldsInsideFrame: false },
  { camera: "cam_14", coverage: 0.9544, foldMarginDeg: null, foldsInsideFrame: false },
];

describe("coverageVerdict", () => {
  it("treats an unmeasured camera as undecidable rather than as passing", () => {
    const camera = { camera: "cam_01" };
    expect(coverageVerdict(camera, response([]))).toBe("unknown");
    expect(coverageNote(camera, response([]))).toContain("不能当作达标");
  });

  it("reports thin coverage as extrapolation without calling for a re-shoot", () => {
    // cam_06 is the least-covered camera in production at 79%. An earlier
    // version of this panel sent it to a re-shoot on that number alone; the
    // cube it has to track never passes 52% of its frame radius, so its
    // unmeasured band is never entered and the re-shoot would buy nothing.
    expect(coverageVerdict(PRODUCTION[0], response(PRODUCTION))).toBe("extrapolated");
    expect(coverageNote(PRODUCTION[0], response(PRODUCTION))).toContain("实际用到那一圈");
    expect(shouldOfferIntrinsicsRecapture(response([PRODUCTION[0]]))).toBe(false);
  });

  it("passes a camera that reached the target and never folds", () => {
    expect(coverageVerdict(PRODUCTION[2], response(PRODUCTION))).toBe("covered");
    expect(coverageNote(PRODUCTION[2], response(PRODUCTION))).toBe("");
  });

  it("calls a model that folds inside its own frame a defect, however well covered", () => {
    const folded = { camera: "cam_x", coverage: 0.99, foldMarginDeg: -3, foldsInsideFrame: true };
    expect(coverageVerdict(folded, response([folded]))).toBe("folded");
    expect(coverageNote(folded, response([folded]))).toContain("无唯一光线");
  });

  it("flags a fold that sits just outside the corner even at full coverage", () => {
    const tight = { camera: "cam_y", coverage: 0.95, foldMarginDeg: 2.1, foldsInsideFrame: false };
    expect(coverageVerdict(tight, response([tight]))).toBe("extrapolated");
    expect(coverageNote(tight, response([tight]))).toContain("2.1°");
  });
});

describe("formatFold", () => {
  it("renders a model that never folds as such, not as a missing number", () => {
    expect(formatFold({ camera: "cam_07", coverage: 0.92, foldMarginDeg: null })).toBe("无折返");
  });

  it("renders nothing when coverage itself was never measured", () => {
    expect(formatFold({ camera: "cam_01" })).toBe("—");
  });
});

describe("coverageSummary", () => {
  it("names the extrapolating cameras and leaves the decision open", () => {
    const summary = coverageSummary(response(PRODUCTION));
    expect(summary).toContain("3 台外侧为外推");
    expect(summary).toContain("cam_06");
    expect(summary).toContain("取决于这些相机实际用到多远");
    expect(summary).not.toContain("建议重标");
  });

  it("puts a folded model ahead of any coverage counting", () => {
    const folded = [{ camera: "cam_x", coverage: 0.5, foldMarginDeg: -3, foldsInsideFrame: true }, ...PRODUCTION];
    expect(coverageSummary(response(folded))).toContain("画幅内折返，需重标：cam_x");
  });

  it("says so plainly when nothing is being extrapolated", () => {
    const all = PRODUCTION.map((camera) => ({ ...camera, coverage: 0.95, foldMarginDeg: null }));
    expect(coverageSummary(response(all))).toContain("无外推区");
  });

  it("surfaces the gateway's own error rather than an empty table", () => {
    const payload = { ...response([]), error: "找不到内参目录：outputs/calibration/x" };
    expect(coverageSummary(payload)).toContain("找不到内参目录");
  });
});

describe("shouldOfferIntrinsicsRecapture", () => {
  it("offers a re-shoot only for a defect in the fit, not for thin coverage", () => {
    expect(shouldOfferIntrinsicsRecapture(response(PRODUCTION))).toBe(false);
    const folded = [{ camera: "cam_x", coverage: 0.99, foldMarginDeg: -3, foldsInsideFrame: true }];
    expect(shouldOfferIntrinsicsRecapture(response(folded))).toBe(true);
    expect(shouldOfferIntrinsicsRecapture(null)).toBe(false);
  });
});

describe("coverageRows", () => {
  it("quantifies the unmeasured band in the note, since that is what a re-shoot would fix", () => {
    const rows = coverageRows(response(PRODUCTION));
    expect(rows[0].coverage).toBe("79%");
    expect(rows[0].note).toContain("21%");
    expect(rows[0].fold).toBe("余量 2.1°");
  });
});
