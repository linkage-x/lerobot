import { describe, expect, it } from "vitest";
import {
  HEARTBEAT_STALE_S,
  mapPoints,
  outcomeLabel,
  projectRows,
  runHealth,
  verdictCounts
} from "./unattendedRuns";
import type { UnattendedRun } from "../types";

function run(overrides: Partial<UnattendedRun> = {}): UnattendedRun {
  return {
    id: "terminal_trials_x",
    kind: "terminal_trials",
    state: "running",
    startedAt: 1000,
    unitsDone: 4,
    unitsPlanned: 60,
    stopRequested: false,
    alive: true,
    lastRowAgeS: 5,
    dir: "/tmp/x",
    pid: 4242,
    argv: [],
    logPath: "/tmp/x/run.log",
    stopReason: "",
    plan: {} as UnattendedRun["plan"],
    rows: [],
    rowsTotal: 9,
    summary: null,
    ...overrides
  };
}

describe("the two things a single status light cannot say", () => {
  it("reports a live process that has stopped writing, which is the case that costs a night", () => {
    const health = runHealth(run({ lastRowAgeS: HEARTBEAT_STALE_S + 60 }), 2000);
    expect(health.divergence).toContain("moving and not recording");
    expect(health.heartbeats.find((beat) => beat.label === "controller")?.ok).toBe(true);
    expect(health.heartbeats.find((beat) => beat.label === "last write")?.ok).toBe(false);
  });

  it("does not cry wolf during an ordinary unrecorded leg", () => {
    const health = runHealth(run({ lastRowAgeS: HEARTBEAT_STALE_S - 1 }), 2000);
    expect(health.divergence).toBe("");
    expect(health.heartbeats.every((beat) => beat.ok)).toBe(true);
  });

  it("reports a run that started and produced nothing at all", () => {
    const health = runHealth(run({ lastRowAgeS: null, rowsTotal: 0, startedAt: 1000 }), 1000 + HEARTBEAT_STALE_S + 10);
    expect(health.divergence).toContain("produced nothing");
  });

  it("does not treat a finished run's quiet as a stall", () => {
    const health = runHealth(run({ state: "complete", alive: false, lastRowAgeS: 9999 }), 99999);
    expect(health.divergence).toBe("");
    expect(health.heartbeats.find((beat) => beat.label === "last write")?.ok).toBe(true);
  });
});

describe("crashed is never folded into complete", () => {
  it("says nobody has read why, because a row count cannot tell the two apart", () => {
    expect(outcomeLabel({ state: "crashed", summary: null })).toContain("nobody has read");
    expect(outcomeLabel({ state: "complete", summary: { ok: true } })).toContain("finished");
  });

  it("names the condition an orderly stop stopped on", () => {
    expect(outcomeLabel({ state: "halted", summary: { haltedOn: "slip_streak" } })).toContain("slip_streak");
  });

  it("does not claim a condition it was not given", () => {
    expect(outcomeLabel({ state: "halted", summary: null })).toContain("unnamed");
  });

  it("marks a crashed run's divergence too", () => {
    expect(runHealth(run({ state: "crashed", alive: false, lastRowAgeS: 30 }), 2000).divergence)
      .toContain("never wrote a summary");
  });
});

describe("a verdict travels with the numbers it came from", () => {
  it("carries the readings so a row can be argued with rather than believed", () => {
    const [unit] = projectRows([
      { kind: "trial", index: 3, ok: true, verdict: "seated", trialKind: "offset", aboveTargetMm: 1.42, settleMm: 0.6, offsetMm: 4 }
    ]);
    expect(unit.verdict).toBe("seated");
    expect(unit.readings).toEqual([
      ["above target", "1.42 mm"],
      ["settle", "0.60 mm"],
      ["offset", "4.0 mm"]
    ]);
  });

  it("surfaces how far the hole moved, which is a reading in its own right", () => {
    const [unit] = projectRows([
      { kind: "trial", index: 0, ok: true, verdict: "seated", referenceStepMm: 0.63 }
    ]);
    expect(unit.readings).toContainEqual(["hole moved", "0.63 mm"]);
  });

  it("skips rows that are not units of work, and failed ones", () => {
    expect(
      projectRows([
        { kind: "grasp", stage: "regrip" },
        { kind: "summary", ok: true },
        { kind: "trial", index: 1, ok: false, error: "servo failed" },
        { kind: "cycle", index: 2, ok: true, verdict: "held", cycleKind: "recovery" }
      ]).map((unit) => unit.index)
    ).toEqual([2]);
  });

  it("counts verdicts commonest first", () => {
    const units = projectRows([
      { kind: "cycle", index: 0, ok: true, verdict: "held" },
      { kind: "cycle", index: 1, ok: true, verdict: "held" },
      { kind: "cycle", index: 2, ok: true, verdict: "empty" }
    ]);
    expect(verdictCounts(units)).toEqual([
      ["held", 2],
      ["empty", 1]
    ]);
  });
});

describe("the map carries three layers, and the third is the one usually missing", () => {
  const schedule = [
    { index: 0, placeXyz: [0.36, -0.14, 0.055] },
    { index: 1, placeXyz: [0.37, -0.13, 0.055] },
    { index: 2, placeXyz: [0.38, -0.12, 0.055] }
  ];

  it("shows planned, done-by-verdict, and where it stopped", () => {
    const units = projectRows([
      { kind: "cycle", index: 0, ok: true, verdict: "held" },
      { kind: "cycle", index: 1, ok: true, verdict: "empty" }
    ]);
    expect(mapPoints(schedule, units, 1).map((point) => point.status)).toEqual([
      "held",
      "halt",
      "planned"
    ]);
  });

  it("leaves every point planned before anything has run", () => {
    expect(mapPoints(schedule, [], null).every((point) => point.status === "planned")).toBe(true);
  });

  it("ignores schedule entries with no point to draw", () => {
    expect(mapPoints([{ index: 0 }, ...schedule], [], null)).toHaveLength(3);
  });

  it("reads a terminal trial's aim when there is no placement", () => {
    expect(mapPoints([{ index: 0, aimXyz: [0.3599, -0.1333, 0.0523] }], [], null)[0].x).toBeCloseTo(0.3599);
  });
});
