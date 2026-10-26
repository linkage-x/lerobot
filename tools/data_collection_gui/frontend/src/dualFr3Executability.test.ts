import { describe, expect, it } from "vitest";
import {
  dualFr3Rows,
  dualReplayControls,
  dualReplayStatus,
  edgeLines,
  realBackendBlocker
} from "./dualFr3Executability";
import type { DualFr3Executability, RobotArm } from "./types";

const stats = (p95: number, max: number) => ({ p50: 0, p95, max });

describe("dualFr3Rows", () => {
  it("lists absolute before relative and reads each mode's row", () => {
    const payload: DualFr3Executability = {
      datasetRoot: "/d",
      episode: 1,
      modes: {
        relative: { summary: {}, episode: null, createdUtc: "2026-10-26T03:00:00Z" },
        absolute: {
          summary: { episodes: 2, executable: 1 },
          createdUtc: "2026-10-26T04:00:00Z",
          episode: {
            dual_fr3_executable: false,
            reasons: ["arms 10 mm apart at frame 3"],
            placement_assumed: true,
            d_safe_mm: 75.6,
            min_arm_arm_gap_mm: 10.2,
            mesh_min_gap_mm: 14.0,
            retarget_change_position_mm: { right: stats(0.01, 0.13), left: stats(3.2, 21.5) },
            t_lr_change_position_mm: stats(3, 21.47),
            reference_duration_s: 31.3,
            slowdown_vs_nominal: 1.2
          }
        }
      }
    };
    const rows = dualFr3Rows(payload);
    expect(rows.map((row) => row.mode)).toEqual(["absolute", "relative"]);
    expect(rows[0]).toMatchObject({
      verdict: "not executable",
      gap: "10 / 76 (mesh 14)",
      placement: "assumed",
      tlr: "21.5",
      change: "R 0.01 / 0.13 · L 3.20 / 21.50",
      timing: "31.3 s (×1.20 retimed)",
      reasons: ["arms 10 mm apart at frame 3"]
    });
    expect(rows[1].verdict).toBe("not evaluated");
  });

  it("is empty without a payload", () => {
    expect(dualFr3Rows(null)).toEqual([]);
  });
});

describe("two-arm replay controls (roadmap D4)", () => {
  const arms: RobotArm[] = [
    { arm: "right", inService: true, whyNot: "", robotIp: "192.168.11.102", gripperBox: 1, payload: null, startJointsRad: null, replayConfig: "" },
    { arm: "left", inService: false, whyNot: "B1: not installed", robotIp: "", gripperBox: null, payload: null, startJointsRad: null, replayConfig: "" }
  ];
  const edge = { from_world_frame_id: "fr3_base_left", to_world_frame_id: "fr3_base", method: "assumed_placement" };
  const payload: DualFr3Executability = {
    datasetRoot: "/d",
    episode: 0,
    modes: {
      absolute: {
        summary: {},
        episode: {
          dual_fr3_executable: true,
          reasons: [],
          mode: { arms: { right: { root: "fr3_base", edges: [] }, left: { root: "fr3_base_left", edges: [edge] } } }
        }
      },
      relative: { summary: {}, episode: { dual_fr3_executable: false, reasons: ["arms 10 mm apart"] } }
    }
  };

  it("connects only an episode C4 cleared, and the real arms only once both are in service", () => {
    expect(dualReplayControls(payload, null, "absolute", "fake", arms).connectBlocker).toBe("");
    expect(dualReplayControls(payload, null, "relative", "fake", arms).connectBlocker).toBe("not executable in relative: arms 10 mm apart");
    expect(dualReplayControls(payload, null, "absolute", "real", arms).connectBlocker).toBe("left: B1: not installed");
    expect(realBackendBlocker(arms.map((row) => ({ ...row, inService: true })))).toBe("");
    expect(dualReplayControls(null, null, "absolute", "fake", arms).connectBlocker).toBe("no absolute verdict for this episode");
  });

  it("executes only at frame 0 and aborts whenever it runs", () => {
    const at = dualReplayControls(payload, { state: "at_start" }, "absolute", "fake", arms);
    expect(at).toMatchObject({ canExecute: true, canAbort: true, connectBlocker: "a two-arm replay is running" });
    expect(dualReplayControls(payload, { state: "executing" }, "absolute", "fake", arms)).toMatchObject({ canExecute: false, canAbort: true });
    expect(dualReplayControls(payload, { state: "finished" }, "absolute", "fake", arms)).toMatchObject({ canExecute: false, canAbort: false });
  });

  it("names the edges each arm goes through", () => {
    expect(edgeLines(payload.modes.absolute.episode)).toEqual([
      "right (fr3_base): none: the dataset is in its base",
      "left (fr3_base_left): fr3_base_left → fr3_base (assumed_placement)"
    ]);
  });

  it("says why a run stopped, and whether the arms stopped together", () => {
    expect(dualReplayStatus({ state: "stopped", backend: "fake", mode: "absolute", episode: 2, report: { stop: ["operator abort (SIGTERM)"], stoppedTogether: false } }))
      .toBe("stopped (fake arms, absolute, episode 2): operator abort (SIGTERM) — the arms did NOT stop together");
    expect(dualReplayStatus(null)).toBe("idle");
  });

  it("adds D2's clock verdict to a finished run, as an estimate", () => {
    const finished = (clock: object) =>
      dualReplayStatus({ state: "finished", backend: "fake", mode: "absolute", episode: 2, report: { minGapMm: 74, startSkewMs: 0.1, clock } });
    expect(finished({ status: "failed", verdict: "start skew +40.52 ms >= one period (20.00 ms); drift +0.11 ms over 0.8 s", tLrFromSkewMmP95: 16.76 }))
      .toBe("finished (fake arms, absolute, episode 2): min gap 74 mm, send skew 0.1 ms; clocks FAILED: " +
        "start skew +40.52 ms >= one period (20.00 ms); drift +0.11 ms over 0.8 s, T_LR ≈ 16.8 mm p95 (estimate)");
    expect(finished({ status: "no robot clock from left" })).toMatch(/; clocks: no robot clock from left$/);
  });
});
