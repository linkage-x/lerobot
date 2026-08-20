import { describe, expect, it } from "vitest";
import type {
  CalibrationSession,
  CalibrationSessionStep,
  CalibrationSolve,
  CalibrationStatus,
} from "../types";
import { captureTally, intrinsicsNote, solveButtonView, solveTargetView } from "./solvePanel";

function status(
  state: CalibrationStatus["state"],
  solve?: Partial<CalibrationSolve>,
): CalibrationStatus {
  return {
    state,
    pattern: "ChArUco",
    lastRunAt: "",
    message: "",
    cameras: [],
    outputPath: "",
    solve: solve
      ? {
          datasetRoot: "",
          datasetName: "",
          episodes: 0,
          source: "none",
          candidates: [],
          ...solve,
        }
      : undefined,
  };
}

function session(stage: CalibrationSession["stage"]): CalibrationSession {
  return {
    active: true,
    stage,
    datasetName: "thor_gmsl2_10ch_v1_20260820_152528",
    datasetRoot: "/data/thor_gmsl2_10ch_v1_20260820_152528",
    currentIndex: 11,
    message: "",
    episodeTimeS: 30,
    recorderState: "armed",
    steps: [],
  };
}

const CAPTURE: Partial<CalibrationSolve> = {
  datasetRoot: "/data/thor_gmsl2_10ch_v1_20260820_152528",
  datasetName: "thor_gmsl2_10ch_v1_20260820_152528",
  episodes: 11,
  source: "session",
};

describe("solve target", () => {
  it("names the capture that will be read, not the one that was recorded", () => {
    const view = solveTargetView({ ...CAPTURE, candidates: [] } as CalibrationSolve);
    expect(view.selected).toBe("/data/thor_gmsl2_10ch_v1_20260820_152528");
    expect(view.summary).toContain("11 段");
    expect(view.origin).toBe("本次引导录制的采集");
  });

  it("says what to do when there is nothing solvable rather than going blank", () => {
    const view = solveTargetView(undefined);
    expect(view.selected).toBe("");
    expect(view.summary).toContain("先录一段");
  });

  it("reports a pick that has gone missing instead of silently solving another", () => {
    // A capture named explicitly is an instruction. Substituting a different
    // one would produce a calibration nobody could trace back to its input.
    const view = solveTargetView({ source: "missing", candidates: [] } as unknown as CalibrationSolve);
    expect(view.summary).toContain("读不到");
  });

  it("labels each candidate with what distinguishes it", () => {
    const view = solveTargetView({
      ...CAPTURE,
      candidates: [
        { path: "/data/a", name: "capture_a", episodes: 11, updatedAt: "2026-08-20 15:25" },
      ],
    } as CalibrationSolve);
    expect(view.candidates[0].label).toBe("capture_a · 11 段 · 2026-08-20 15:25");
  });
});

describe("solve button", () => {
  it("is offered again after a failure, because the capture is still on disk", () => {
    // The regression this exists for: the button was gated on stage "ready",
    // and a failed solve moved the session to "failed" -- so a solve that died
    // on a missing module made an intact 11-episode capture unusable.
    const view = solveButtonView(status("failed", CAPTURE), session("failed"));
    expect(view.visible).toBe(true);
    expect(view.label).toBe("重新解算");
    expect(view.disabled).toBe(false);
  });

  it("hides while the solve is running, where the progress bar is the answer", () => {
    expect(solveButtonView(status("running", CAPTURE), session("solving")).visible).toBe(false);
  });

  it("hides while sweeps are still being captured", () => {
    expect(solveButtonView(status("idle", CAPTURE), session("capture")).visible).toBe(false);
  });

  it("is offered with no session at all, on whatever capture is selected", () => {
    const view = solveButtonView(status("idle", CAPTURE), undefined);
    expect(view.visible).toBe(true);
    expect(view.label).toBe("开始解算");
  });

  it("offers a re-solve after a successful one", () => {
    expect(solveButtonView(status("complete", CAPTURE), session("done")).label).toBe("重新解算");
  });

  it("is disabled, not hidden, when nothing is selectable", () => {
    const view = solveButtonView(status("idle"), undefined);
    expect(view.visible).toBe(true);
    expect(view.disabled).toBe(true);
  });
});

describe("capture tally", () => {
  it("counts what landed, not how far the wizard walked", () => {
    // The 2026-08-20 session showed "进度 11/11" with exactly one episode on
    // disk: ten intrinsics steps were skipped. The solve reads episodes.
    const steps = [
      ...Array.from({ length: 10 }, () => ({ status: "skipped" })),
      { status: "captured" },
    ] as CalibrationSessionStep[];
    expect(captureTally(steps)).toBe("1 段 · 跳过 10");
  });

  it("says nothing about skips when there were none", () => {
    expect(captureTally([{ status: "captured" }, { status: "captured" }] as CalibrationSessionStep[])).toBe(
      "2 段",
    );
  });
});

describe("intrinsics note", () => {
  it("says which intrinsics a plain re-solve will ship", () => {
    // The solve has only ever run the extrinsics half; saying so is what stops
    // an operator recording ten intrinsics sweeps that nothing will read.
    const note = intrinsicsNote(
      { ...(CAPTURE as CalibrationSolve), candidates: [], intrinsicsRun: "thor_gmsl2_selfcal_0804_fisheye_intrinsics" },
      false,
    );
    expect(note).toContain("沿用");
    expect(note).toContain("thor_gmsl2_selfcal_0804_fisheye_intrinsics");
  });

  it("names the capture the re-fit would read, and what it overwrites", () => {
    const note = intrinsicsNote(
      {
        ...(CAPTURE as CalibrationSolve),
        candidates: [],
        intrinsicsDatasetRoot: "/data/i",
        intrinsicsDatasetName: "thor_gmsl2_10ch_v1_20260820_151356",
        intrinsicsEpisodes: 7,
        intrinsicsRun: "thor_gmsl2_selfcal_0804_fisheye_intrinsics",
      },
      true,
    );
    expect(note).toContain("thor_gmsl2_10ch_v1_20260820_151356");
    expect(note).toContain("7 段");
    expect(note).toContain("覆盖");
  });

  it("asks for the capture when the box is ticked without one", () => {
    const note = intrinsicsNote({ ...(CAPTURE as CalibrationSolve), candidates: [] }, true);
    expect(note).toContain("还没选内参采集");
    expect(note).toContain("四角");
  });
});
