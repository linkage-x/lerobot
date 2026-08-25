import { describe, expect, it } from "vitest";
import {
  EXTRINSICS_GUIDE,
  INTRINSICS_GUIDE,
  guideFor,
  segmentEndGuide,
  skipConsequence,
  stepTitle,
} from "./calibrationGuide";

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
    expect(guideFor("intrinsics").slice(0, INTRINSICS_GUIDE.length)).toEqual(INTRINSICS_GUIDE);
    expect(guideFor("extrinsics").slice(0, EXTRINSICS_GUIDE.length)).toEqual(EXTRINSICS_GUIDE);
    expect(stepTitle("intrinsics", "cam_06")).toContain("cam_06");
    expect(stepTitle("extrinsics", "")).toContain("协同");
  });

  it("ends both guides with how the segment actually stops", () => {
    // The recorder closes the episode itself at dataset.episode_time_s. The old
    // text ("录 30–60 秒后点「保存本段」") promised something it will not do, and
    // following it walked the operator into "Cannot save while recorder is armed".
    for (const kind of ["intrinsics", "extrinsics"] as const) {
      const last = guideFor(kind, 10, 600).at(-1)!;
      expect(last.title).toContain("10 秒");
      expect(last.title).toContain("600 帧");
      expect(last.title).toContain("自动结束并保存");
    }
  });

  it("falls back to untimed wording when the recorder has no episode limit", () => {
    expect(segmentEndGuide(0, 0).title).toBe("挥完后点「保存本段」");
    expect(guideFor("intrinsics").at(-1)!.title).toBe("挥完后点「保存本段」");
  });

  it("says skipping a camera is fine but skipping the rig sweep is not", () => {
    expect(skipConsequence("intrinsics")).toContain("沿用现有内参");
    expect(skipConsequence("extrinsics")).toContain("不能跳过");
  });
});
