import { describe, expect, it } from "vitest";
import { beamLossSummary, errorBudgetSummary, niceCeiling, polylineSegments, runLabel } from "./laserReplayAnalysis";
import type { LaserReplayRun } from "./types";

// 2026-10-10 20261010T032019Z, ep0 right of thor_gmsl2_10ch_v1_20261009_102307.
const run: LaserReplayRun = {
  run: "20261010T032019Z_episode_000000_right",
  run_dir: "/x",
  frames: 920,
  used_frames: 463,
  first_loss_frame: 473,
  laser_vs_desired: { n: 463, rms_mm: 7.91, p95_mm: 11.71, max_mm: 15.06 },
  laser_vs_fk: { n: 463, rms_mm: 0.28, p95_mm: 0.47, max_mm: 0.7 },
  laser_vs_desired_bias_mm: [0.2, -2.1, -5.5],
  decomposition: {
    follow_lag_frames: 3,
    follow_lag_ms: 150,
    laser_vs_desired_rms_mm: 7.91,
    without_bias_rms_mm: 5.24,
    without_bias_and_lag_rms_mm: 2.95,
    fk_without_bias_and_lag_rms_mm: 2.87
  },
  line_of_sight: { seen_max_deg: 7.33, trajectory_max_deg: 13.6, at_first_loss_deg: 7.45, first_loss_new_direction_deg: 0.18 }
};

describe("laser replay analysis text", () => {
  it("labels runs by their UTC start", () => {
    expect(runLabel(run.run)).toBe("03:20:19Z");
    expect(runLabel("odd")).toBe("odd");
  });

  it("splits the replay error into offset, lag and rest", () => {
    const text = errorBudgetSummary(run);
    expect(text).toContain("7.91 mm RMS");
    expect(text).toContain("(0.2, -2.1, -5.5) mm");
    expect(text).toContain("150 ms follow lag");
    expect(text).toContain("0.28 mm RMS");
  });

  it("points at the SMR when the beam is lost at the edge of a small swing", () => {
    expect(beamLossSummary(run)).toContain("face the SMR at the tracker");
    const inside = { ...run, line_of_sight: { ...run.line_of_sight!, at_first_loss_deg: 2.0 } };
    expect(beamLossSummary(inside)).toContain("something in the scene");
    expect(beamLossSummary({ ...run, first_loss_frame: -1 })).toBe("Beam held for all 920 frames.");
  });
});

describe("chart helpers", () => {
  it("breaks a series where the beam was off", () => {
    expect(polylineSegments([1, 2, null, 3, 4, null], 50, 10, 4)).toEqual(["0.0,7.5 10.0,5.0", "30.0,2.5 40.0,0.0"]);
    expect(polylineSegments([1], 50, 10, 4)).toEqual([]);
  });

  it("rounds the axis up to a readable number", () => {
    expect(niceCeiling(15.06)).toBe(20);
    expect(niceCeiling(0.7)).toBe(1);
    expect(niceCeiling(0)).toBe(1);
  });
});
