import { describe, expect, it } from "vitest";
import { describeArm, verdictAgreement, verdictAsOutcome } from "./rolloutArms";
import type { RolloutOutcomeEntry } from "../types";

function graded(overrides: Partial<RolloutOutcomeEntry> = {}): RolloutOutcomeEntry {
  return {
    recordedAt: "2026-09-20T10:00:00Z",
    checkpointId: "job_a/030000",
    outcome: "success",
    mode: "real",
    steps: 400,
    note: "",
    logPath: "",
    ...overrides
  };
}

describe("describeArm", () => {
  it("names the control arm as what it is rather than as a search with one point", () => {
    // 1 landing is E5 exactly, and E5 is what every other reading is compared against. A bar
    // reading "1 landing" is how a control run gets filed as a search run.
    expect(describeArm({ actionSamples: 1, terminalServoSearchLandings: 1 })).toBe(
      "1 draw · fixed pose"
    );
  });

  it("distinguishes two arms of the same comparison at a glance", () => {
    expect(
      describeArm({
        actionSamples: 8,
        actionAggregate: "medoid",
        terminalServoSearchLandings: 9,
        terminalServoSearchRingM: 0.007
      })
    ).toBe("medoid of 8 · 9 landings @ 7.0 mm");
  });

  it("says nothing when the runtime announced nothing", () => {
    // Empty, not "default": silence is a runtime older than the announce, and printing a
    // default here would invent the one fact this field exists to record.
    expect(describeArm(undefined)).toBe("");
    expect(describeArm({})).toBe("");
  });
});

describe("verdictAsOutcome", () => {
  it("files a slip as a miss, because the operator's scale has no other word for it", () => {
    expect(verdictAsOutcome("slip")).toBe("failure");
    expect(verdictAsOutcome("standing")).toBe("failure");
    expect(verdictAsOutcome("seated")).toBe("success");
  });

  it("is empty for a runtime that named no verdict", () => {
    expect(verdictAsOutcome(undefined)).toBe("");
  });
});

describe("verdictAgreement", () => {
  it("counts only rollouts where both said something comparable", () => {
    const rate = verdictAgreement([
      graded({ rolloutIndex: 1, outcome: "success", terminalServo: { verdict: "seated" } }),
      graded({ rolloutIndex: 2, outcome: "failure", terminalServo: { verdict: "standing" } }),
      // No verdict: an older runtime, not a disagreement.
      graded({ rolloutIndex: 3, outcome: "failure" })
    ]);

    expect(rate.compared).toBe(2);
    expect(rate.agreed).toBe(2);
    expect(rate.rate).toBe(1);
  });

  it("names the rollouts that disagreed, because a rate alone cannot be acted on", () => {
    const rate = verdictAgreement([
      graded({ rolloutIndex: 4, outcome: "success", terminalServo: { verdict: "slip" } }),
      graded({ rolloutIndex: 5, outcome: "success", terminalServo: { verdict: "seated" } })
    ]);

    expect(rate.compared).toBe(2);
    expect(rate.agreed).toBe(1);
    expect(rate.rate).toBe(0.5);
    expect(rate.disagreed).toEqual([4]);
  });

  it("leaves an aborted rollout out instead of calling it a miss", () => {
    // An abort is a statement about the session, not about the peg -- the descent may never
    // have happened. Counting it would move this rate for reasons unrelated to what it
    // measures.
    const rate = verdictAgreement([
      graded({ rolloutIndex: 6, outcome: "aborted", terminalServo: { verdict: "standing" } })
    ]);

    expect(rate.compared).toBe(0);
    expect(rate.rate).toBeNull();
  });

  it("reports no reading rather than zero agreement before anything is comparable", () => {
    expect(verdictAgreement([]).rate).toBeNull();
  });
});
