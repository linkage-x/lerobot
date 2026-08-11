import { describe, expect, it } from "vitest";
import { EXTRINSICS_GUIDE, INTRINSICS_GUIDE, guideFor, skipConsequence, stepTitle } from "./calibrationGuide";

describe("calibration operator guidance", () => {
  it("tells the operator the one thing that invalidated the 0804 capture", () => {
    // Coverage of the frame edge is the failure that forced a full recapture,
    // so it must be stated as an instruction and as a live check.
    const text = INTRINSICS_GUIDE.map((s) => `${s.title}${s.detail ?? ""}`).join("\n");
    expect(text).toContain("四个角");
    expect(text).toContain("边框");
  });

  it("explains why the extrinsics sweep must be seen by several cameras at once", () => {
    const text = EXTRINSICS_GUIDE.map((s) => `${s.title}${s.detail ?? ""}`).join("\n");
    expect(text).toContain("同时");
  });

  it("separates the two capture kinds", () => {
    expect(guideFor("intrinsics")).toBe(INTRINSICS_GUIDE);
    expect(guideFor("extrinsics")).toBe(EXTRINSICS_GUIDE);
    expect(stepTitle("intrinsics", "cam_06")).toContain("cam_06");
    expect(stepTitle("extrinsics", "")).toContain("协同");
  });

  it("says skipping a camera is fine but skipping the rig sweep is not", () => {
    expect(skipConsequence("intrinsics")).toContain("沿用现有内参");
    expect(skipConsequence("extrinsics")).toContain("不能跳过");
  });
});
