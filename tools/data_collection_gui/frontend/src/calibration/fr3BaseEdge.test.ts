import { describe, expect, it } from "vitest";
import type { Fr3BaseEdgeResponse } from "../types";
import {
  applyConfirmation,
  candidateSpread,
  liveEdgeSummary,
  liveEdgeWarnings,
  moveSummary,
  verdictLabel,
} from "./fr3BaseEdge";

type Candidate = Fr3BaseEdgeResponse["candidates"][number];

const current: NonNullable<Fr3BaseEdgeResponse["current"]> = {
  world: "world_20260928_063531",
  createdUtc: "2026-10-09T06:55:38Z",
  method: "fr3_single_tag_hand_eye",
  p0Run: "outputs/calibration/p0_single_tag_camera_calibration/manual_run_20261002T014903Z",
  intrinsicsRun: "calib_20260928_143107_intrinsics",
  fromReferenceWorld: true,
  referenceUnchanged: true,
  supersedes: 0,
  edgeCount: 1,
  quality: { captures: 137, outOfSampleP95Mm: 4.17, tagScale: 0.9912, folds: [] },
};

function candidate(run: string, meanMm: [number, number, number] | null, overrides: Partial<Candidate> = {}): Candidate {
  return {
    name: `${run}__x.json`,
    generatedUtc: "2026-10-13T08:00:00Z",
    p0Run: run,
    verdict: "ok",
    reasons: [],
    intrinsicsRun: "calib_20260928_143107_intrinsics",
    quality: { captures: 85, outOfSampleP95Mm: 5.55, tagScale: 0.9908, folds: [] },
    moveVsLive: meanMm
      ? { points: 85, meanMm, meanNormMm: Math.hypot(...meanMm), maxMm: Math.hypot(...meanMm) + 0.5, rotationDeg: 0.25 }
      : null,
    sameAsLive: false,
    applyBlockers: [],
    ...overrides,
  };
}

describe("fr3BaseEdge", () => {
  it("summarises the live edge with its run and its out-of-sample number", () => {
    expect(liveEdgeSummary(current)).toContain("manual_run_20261002T014903Z");
    expect(liveEdgeSummary(current)).toContain("p95 4.17 mm");
    expect(liveEdgeSummary(null)).toBe("没有注册边");
  });

  it("says what a missing or stale edge does to the FR3", () => {
    expect(liveEdgeWarnings(null, "w")[0]).toContain("都会被拒绝");
    expect(liveEdgeWarnings(current, "world_20260928_063531")).toEqual([]);
    expect(liveEdgeWarnings({ ...current, fromReferenceWorld: false }, "world_new")[0]).toContain("world_new");
    expect(liveEdgeWarnings({ ...current, referenceUnchanged: false }, "w")[0]).toContain("改过");
  });

  it("reports the move in the workspace with its direction", () => {
    const move = candidate("run", [-2, -0.3, -3.8]).moveVsLive;
    expect(moveSummary(move)).toMatch(/^平均 4\.3 mm \(-2\.0, -0\.3, -3\.8\)/);
    expect(moveSummary(null)).toBe("—");
  });

  // The 2026-10-13 numbers at the replay trajectory: both 10-10 runs pass, sit
  // 3.9 mm apart, and the first is 4.3 mm from the live edge (the 10-02 run
  // re-solved to itself).
  it("shows how far passing candidates are from each other, the live edge included", () => {
    const spread = candidateSpread([
      candidate("manual_run_20261010T000513Z", [-2.0, -0.3, -3.8]),
      candidate("manual_run_20261010T070331Z", [-0.8, -3.5, -1.9]),
      candidate("manual_run_20261002T014903Z", [0, 0, 0], { sameAsLive: true }),
    ]);
    expect(spread?.mm).toBeCloseTo(Math.hypot(2.0, 0.3, 3.8), 6);
    expect([spread?.a, spread?.b].sort()).toEqual(["manual_run_20261002T014903Z", "manual_run_20261010T000513Z"]);
  });

  it("leaves rejected candidates out of the spread and needs two to compare", () => {
    expect(candidateSpread([candidate("a", [9, 9, 9], { verdict: "rejected" })])).toBeNull();
    expect(candidateSpread([candidate("a", [3, 4, 0])])).toEqual({ mm: 5, a: "a", b: "现役边" });
  });

  it("labels and confirms in terms of what changes", () => {
    expect(verdictLabel(candidate("a", [0, 0, 0], { sameAsLive: true }))).toBe("= 现役边");
    expect(verdictLabel(candidate("a", null, { verdict: "rejected" }))).toBe("未通过");
    const text = applyConfirmation(candidate("manual_run_20261010T070331Z", [-0.8, -3.5, -1.9]), current);
    expect(text).toContain("manual_run_20261010T070331Z");
    expect(text).toContain("已经导出的数据集");
    expect(text).toContain("提交");
  });
});
