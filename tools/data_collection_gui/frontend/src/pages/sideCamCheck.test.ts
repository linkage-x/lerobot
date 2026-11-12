import { describe, expect, it } from "vitest";

import type { SideCamCheckResult } from "../types";
import { describePose, describeReference, verdictBadgeClass } from "./sideCamCheck";

const result: SideCamCheckResult = {
  verdict: "shifted",
  verdictText: "轻微移动",
  reason: "",
  pose: {
    dxPx: -4.5,
    dyPx: 2,
    rotDeg: 0.4,
    shiftPx: 4.9,
    scorePx: 0.3,
    coverage: 0.812,
    tapePixels: 2300,
    referenceTapePixels: 3100,
    atSearchEdge: false
  }
};

describe("side camera tape check", () => {
  it("colours the verdicts from calm to alarming, unknown never green", () => {
    expect(verdictBadgeClass("ok")).toContain("running");
    expect(verdictBadgeClass("shifted")).toContain("warning");
    expect(verdictBadgeClass("moved")).toContain("error");
    expect(verdictBadgeClass("unknown")).toContain("idle");
    expect(verdictBadgeClass(undefined)).toContain("idle");
  });

  it("states the pose with signs and the share of the tape it rests on", () => {
    expect(describePose(result)).toBe("水平 -4.5 px，竖直 +2.0 px，旋转 +0.4°；胶带可见 2300 px，覆盖基准 81%");
    expect(describePose({ ...result, pose: null })).toBe("");
  });

  it("says whether the reference is the training-era default or a captured still", () => {
    expect(describeReference(null)).toBe("还没有基准画面");
    expect(
      describeReference({
        camera: "side",
        reference: { kind: "default", note: "默认基准：10-27", path: "x" },
        last: null,
        history: []
      })
    ).toBe("默认基准：10-27");
    expect(
      describeReference({
        camera: "side",
        reference: { kind: "saved", capturedAt: "2026-11-11T15:04:00+08:00", path: "x" },
        last: null,
        history: []
      })
    ).toMatch(/^基准：11-11 \d\d:04 手动采集$/);
  });
});
