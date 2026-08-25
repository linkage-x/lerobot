import { describe, expect, it } from "vitest";
import type { CalibrationProgress, CalibrationStatus } from "../types";
import { formatDuration, solveProgressView } from "./solveProgress";

function status(
  state: CalibrationStatus["state"],
  progress?: Partial<CalibrationProgress>,
): CalibrationStatus {
  return {
    state,
    pattern: "ChArUco",
    lastRunAt: "",
    message: "",
    cameras: [],
    outputPath: "",
    progress: progress
      ? {
          stepIndex: 1,
          stepCount: 3,
          label: "检测 ChArUco 角点…",
          done: 0,
          total: 0,
          fraction: 0,
          detail: "",
          startedAt: 1_700_000_000,
          elapsedS: 0,
          etaS: 0,
          ...progress,
        }
      : undefined,
  };
}

describe("calibration solve progress", () => {
  it("draws nothing unless a solve is running", () => {
    expect(solveProgressView(status("idle", { stepIndex: 1 }))).toBeNull();
    expect(solveProgressView(status("complete", { stepIndex: 3, fraction: 1 }))).toBeNull();
    // A gateway older than this field sends no progress at all.
    expect(solveProgressView(status("running"))).toBeNull();
  });

  it("counts videos while the detection step can say how many there are", () => {
    const view = solveProgressView(
      status("running", { done: 30, total: 120, fraction: 0.2, elapsedS: 300, etaS: 1200 }),
    )!;
    expect(view.percent).toBe(20);
    expect(view.counter).toBe("30 / 120 个视频");
    expect(view.headline).toContain("步骤 1/3");
    expect(view.indeterminate).toBe(false);
    expect(view.timing).toBe("已用 5 分 · 预计还需 20 分");
  });

  it("marks the bundle step as unable to advance rather than faking movement", () => {
    // calibrate_extrinsics prints nothing between "seeded poses" and the
    // result, so there is no unit to count. A bar that crept anyway would be
    // indistinguishable from one that is actually progressing.
    const view = solveProgressView(
      status("running", { stepIndex: 2, label: "多相机联合 BA…", total: 0, fraction: 0.8 }),
    )!;
    expect(view.indeterminate).toBe(true);
    expect(view.counter).toBe("");
    expect(view.percent).toBe(80);
  });

  it("never shows a full bar while the solve is still running", () => {
    const view = solveProgressView(status("running", { stepIndex: 3, fraction: 1, total: 0 }))!;
    expect(view.percent).toBe(99);
  });

  it("says the remaining time is unknown instead of guessing at zero", () => {
    const view = solveProgressView(status("running", { elapsedS: 8, etaS: 0 }))!;
    expect(view.timing).toBe("已用 8 秒 · 剩余时间还算不出来");
  });

  it("formats durations the way an operator reads a wall clock", () => {
    expect(formatDuration(0)).toBe("0 秒");
    expect(formatDuration(59.4)).toBe("59 秒");
    expect(formatDuration(60)).toBe("1 分");
    expect(formatDuration(125)).toBe("2 分 5 秒");
    expect(formatDuration(3720)).toBe("1 小时 2 分");
  });
});
