import { describe, expect, it } from "vitest";
import type { CrossCameraReport } from "../types";
import { crossCameraRows, formatOffset, setChangeSummary, staleReason } from "./crossCamera";

const report: CrossCameraReport = {
  generated_utc: "2026-09-28T10:00:00Z",
  dataset: "/d/bench",
  thresholds: { warn_mm: 2, fail_mm: 4, step_budget_mm: 3 },
  camera_serials: { cam_13: "H120K-I05130029" },
  cubes: {
    right: {
      frames_with_cube: 1201,
      cameras: {
        cam_13: {
          frames: 1178,
          verdict: "fail",
          lateral_offset_mm: 4.48,
          lateral_offset_least_squares_mm: 4.19,
          scatter_p95_mm: 1.69,
          along_p50_mm: -2.07,
          rotation_p50_deg: 0.66,
        },
        cam_06: { frames: 12, verdict: "unknown", reason: "只有 12 帧和至少 2 台别的相机同时看到 cube" },
        cam_12: { frames: 1195, verdict: "warn", lateral_offset_mm: 3.63, lateral_offset_least_squares_mm: 2.1 },
      },
      set_changes: {
        verdict: "fail",
        changes: 24,
        flicker_changes: 57,
        step_p50_mm: 1.61,
        step_p95_mm: 5.39,
        steady_p50_mm: 0.47,
        steady_p95_mm: 2.72,
        budget_mm: 3,
      },
    },
  },
  overall: "fail",
  guidance: "cam_13 …",
  extrinsics_run: "calib_20260923_cam13refit",
};

describe("cross-camera rows", () => {
  it("lists every camera with its serial, and says why an unjudged one was not judged", () => {
    const rows = crossCameraRows(report);
    expect(rows.map((r) => r.camera)).toEqual(["cam_06", "cam_12", "cam_13"]);
    expect(rows[2].serial).toBe("H120K-I05130029");
    expect(rows[0].verdict).toBe("unknown");
    expect(rows[0].detail).toContain("12 帧");
  });

  it("shows the least-squares split only when it tells a different story", () => {
    expect(formatOffset(report.cubes.right.cameras.cam_13)).toBe("4.5 mm");
    expect(formatOffset(report.cubes.right.cameras.cam_12)).toBe("3.6 mm（最小二乘 2.1 mm）");
  });

  it("marks depth as not judged, so a marker-size offset is not read as a miss", () => {
    const row = crossCameraRows(report).find((r) => r.camera === "cam_13");
    expect(row?.detail).toContain("深度 -2.1 mm（不参与判定）");
  });
});

describe("set-change summary", () => {
  it("puts the step next to its floor and counts flicker separately", () => {
    const text = setChangeSummary(report.cubes.right.set_changes);
    expect(text).toContain("24 次切换");
    expect(text).toContain("p95 5.4 mm");
    expect(text).toContain("不切换时 0.5 mm / 2.7 mm");
    expect(text).toContain("57 次逐帧闪烁");
  });
});

describe("staleness", () => {
  it("flags a result about extrinsics that have since been replaced", () => {
    expect(staleReason(report, "calib_20260930_new")).toContain("calib_20260923_cam13refit");
    expect(staleReason(report, "calib_20260923_cam13refit")).toBe("");
  });
});
