import { describe, expect, it } from "vitest";
import {
  UNLABELLED,
  blockersToSend,
  mismatchCount,
  reasonSlots,
  slotLabel
} from "./takeoverReasons";
import type { RolloutOutcomeEntry } from "../types";

describe("reasonSlots", () => {
  it("is one per takeover", () => {
    expect(reasonSlots(3)).toBe(3);
  });

  it("is still one when nobody reached in", () => {
    // The field's older meaning -- why it stopped where its stage says it stopped -- does not
    // disappear just because there was no takeover.
    expect(reasonSlots(0)).toBe(1);
  });
});

describe("blockersToSend", () => {
  it("keeps a reason that happened twice, twice", () => {
    // The case the old checkbox set could not express and the old backend dropped. Three
    // rescues for the same reason are three data points, and the most common reason is
    // precisely the one that recurs.
    expect(blockersToSend(["policy_action", "policy_action", "perception"], 3)).toEqual([
      "policy_action",
      "policy_action",
      "perception"
    ]);
  });

  it("fills a skipped span rather than shifting the ones after it", () => {
    expect(blockersToSend(["", "perception"], 2)).toEqual([UNLABELLED, "perception"]);
  });

  it("sends nothing when nothing was picked", () => {
    // Not a row of `unknown`: that would turn "nobody was asked" into "asked and could not say".
    expect(blockersToSend(["", ""], 2)).toEqual([]);
    expect(blockersToSend([], 3)).toEqual([]);
  });

  it("never sends more reasons than there are spans", () => {
    expect(blockersToSend(["a", "b", "c"], 2)).toEqual(["a", "b"]);
  });
});

describe("slotLabel", () => {
  it("names the reach-in by what the operator watched", () => {
    expect(
      slotLabel(0, {
        index: 0,
        first: 41,
        last: 58,
        xyz: [0.36, -0.13, 0.1123],
        policyStatus: "step_limited"
      })
    ).toBe("第 1 段 · 步 41–58 · z 0.112 · step_limited");
  });

  it("says nothing it was not told", () => {
    // A runtime older than the per-span lines, or a line lost to a truncated log.
    expect(slotLabel(1, undefined)).toBe("第 2 段");
    // `pass` is not a trigger, so printing it would put a reason where there was none.
    expect(slotLabel(0, { index: 0, first: 4, last: 9, policyStatus: "pass" })).toBe(
      "第 1 段 · 步 4–9"
    );
  });
});

describe("mismatchCount", () => {
  function graded(overrides: Partial<RolloutOutcomeEntry> = {}): RolloutOutcomeEntry {
    return {
      recordedAt: "2026-09-20T10:00:00Z",
      checkpointId: "job_a/030000",
      outcome: "failure",
      mode: "real",
      steps: 400,
      note: "",
      logPath: "",
      ...overrides
    };
  }

  it("counts the grades whose reasons did not line up with their takeovers", () => {
    // Recorded rather than refused -- but a defect nothing displays is a defect nobody fixes.
    expect(
      mismatchCount([
        graded({ takeoverBlockerMismatch: { spans: 2, blockers: 1 } }),
        graded(),
        graded({ takeoverBlockerMismatch: { spans: 3, blockers: 1 } })
      ])
    ).toBe(2);
  });

  it("is zero on a log that has none", () => {
    expect(mismatchCount([graded()])).toBe(0);
  });
});
