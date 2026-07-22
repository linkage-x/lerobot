import { describe, it, expect } from "vitest";
import { computeValidity, canTransition, summarizeKinds, worstValidity } from "./status";
import { MS } from "./config";

describe("computeValidity (dynamic: valid 30min, due-soon last 5min)", () => {
  const now = 1_000_000_000_000;
  it("is unknown when never run", () => {
    expect(computeValidity(null, "force_dynamic", now).state).toBe("unknown");
  });
  it("is valid with more than 5 min remaining", () => {
    const last = now - 10 * MS.minute; // 20 min left of 30
    expect(computeValidity(last, "force_dynamic", now).state).toBe("valid");
  });
  it("is due_soon within the last 5 min", () => {
    const last = now - 26 * MS.minute; // 4 min left
    expect(computeValidity(last, "force_dynamic", now).state).toBe("due_soon");
  });
  it("is overdue past 30 min", () => {
    const last = now - 31 * MS.minute;
    const v = computeValidity(last, "force_dynamic", now);
    expect(v.state).toBe("overdue");
    expect(v.remainingMs).toBeLessThan(0);
  });
  it("origin uses a daily window", () => {
    const last = now - 2 * MS.hour;
    expect(computeValidity(last, "force_origin", now).state).toBe("valid");
  });
});

describe("state machine transitions", () => {
  it("allows the happy path", () => {
    expect(canTransition("idle", "checking_prerequisites")).toBe(true);
    expect(canTransition("checking_prerequisites", "waiting_for_stability")).toBe(true);
    expect(canTransition("waiting_for_stability", "calibrating")).toBe(true);
    expect(canTransition("calibrating", "validating")).toBe(true);
    expect(canTransition("calibrating", "waiting_for_reboot")).toBe(true);
    expect(canTransition("waiting_for_reboot", "reconnecting")).toBe(true);
    expect(canTransition("reconnecting", "validating")).toBe(true);
    expect(canTransition("validating", "passed")).toBe(true);
    expect(canTransition("validating", "failed")).toBe(true);
  });
  it("rejects illegal jumps", () => {
    expect(canTransition("idle", "calibrating")).toBe(false);
    expect(canTransition("passed", "validating")).toBe(false);
    expect(canTransition("waiting_for_stability", "passed")).toBe(false);
  });
  it("allows cancel from any in-flight state", () => {
    expect(canTransition("calibrating", "cancelled")).toBe(true);
    expect(canTransition("reconnecting", "cancelled")).toBe(true);
  });
});

describe("summarizeKinds", () => {
  const now = 1_000_000_000_000;
  it("treats never-run kinds as unknown (needs attention)", () => {
    const s = summarizeKinds({ force_origin: null, force_dynamic: null, touch: null }, now);
    expect(s.unknown).toBe(3);
    expect(s.valid).toBe(0);
  });
  it("counts a mix of valid / due-soon / overdue", () => {
    const s = summarizeKinds(
      {
        force_origin: now - 1 * MS.hour, // valid (daily)
        force_dynamic: now - 26 * MS.minute, // due soon
        touch: now - 2 * MS.day, // overdue (daily)
      },
      now,
    );
    expect(s.valid).toBe(1);
    expect(s.dueSoon).toBe(1);
    expect(s.overdue).toBe(1);
  });
});

describe("worstValidity", () => {
  it("prefers the most severe state", () => {
    expect(worstValidity(["valid", "overdue", "due_soon"])).toBe("overdue");
    expect(worstValidity(["valid", "due_soon"])).toBe("due_soon");
    expect(worstValidity(["valid", "valid"])).toBe("valid");
    expect(worstValidity([])).toBe("unknown");
  });
});
