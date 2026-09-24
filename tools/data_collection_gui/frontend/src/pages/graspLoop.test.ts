import { describe, expect, it } from "vitest";

import { armRates, gateReading, medianByVerdict, wilsonInterval } from "./graspLoop";

describe("wilsonInterval", () => {
  it("matches the backend for the 09-22 hand-graded baseline", () => {
    const [low, high] = wilsonInterval(11, 38);
    expect(low).toBeCloseTo(0.17, 2);
    expect(high).toBeCloseTo(0.448, 2);
  });

  it("is the whole range before anything is graded", () => {
    expect(wilsonInterval(0, 0)).toEqual([0, 1]);
  });
});

describe("medianByVerdict", () => {
  it("splits held from not-held and ignores ungraded trials", () => {
    const row = (verdict: string, closeAboveTargetMm: number | null) => ({
      trial: 0,
      verdict,
      widthLifted: null,
      closeAboveTargetMm,
      lateralMm: null,
      trialS: null
    });
    const trials = [row("held", 10), row("held", 20), row("empty", 30), row("no_close", null), row("not_graded", 99)];
    expect(medianByVerdict(trials, "closeAboveTargetMm")).toEqual({ held: 15, notHeld: 30 });
  });
});

describe("armRates", () => {
  const row = (trial: number, arm: string | undefined, verdict: string) => ({
    trial,
    arm,
    verdict,
    widthLifted: null,
    closeAboveTargetMm: null,
    lateralMm: null,
    trialS: null
  });

  it("counts each arm on its own and leaves ungraded trials out", () => {
    const rates = armRates([row(0, "B", "held"), row(1, "A", "empty"), row(2, "A", "held"), row(3, "B", "not_graded"), row(4, undefined, "no_close")]);
    expect(rates.map(({ arm, graded, held }) => [arm, graded, held])).toEqual([
      ["A", 3, 1],
      ["B", 1, 1]
    ]);
  });
});

describe("gateReading", () => {
  it("reads the v14 pass marks on a batch of 50", () => {
    expect(gateReading(47, 50)).toBe("≥ 47/50：按 95% 目标记");
    expect(gateReading(45, 50)).toBe("≥ 45/50：过 90% 线");
    expect(gateReading(20, 26)).toContain("过不了");
    expect(gateReading(18, 20)).toBe("失手 2 / 允许 5");
  });
});
