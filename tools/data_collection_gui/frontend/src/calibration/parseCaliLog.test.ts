import { describe, it, expect } from "vitest";
import { parseForceVec, parseCaliLog, evaluateForce } from "./parseCaliLog";
import { ORIGIN_FORCE_LIMITS, DYNAMIC_FORCE_LIMITS } from "./config";
import type { BoxCaliLogLine } from "../types";

const line = (text: string, done = false, ts = 1): BoxCaliLogLine => ({ ts, line: text, done });

describe("parseForceVec", () => {
  it("parses a full Fx..Mz fragment", () => {
    const v = parseForceVec("Fx=0.1200, Fy=-0.8600, Fz=0.0800, Mx=0.0010, My=-0.0020, Mz=0.0000");
    expect(v).toEqual({ fx: 0.12, fy: -0.86, fz: 0.08, mx: 0.001, my: -0.002, mz: 0 });
  });
  it("returns null for n/a and incomplete vectors", () => {
    expect(parseForceVec("n/a")).toBeNull();
    expect(parseForceVec("Fx=0.1, Fy=0.2")).toBeNull();
  });
});

describe("parseCaliLog", () => {
  it("parses a single-box run with before/after and terminal ok", () => {
    const lines = [
      line("6D force software-zero calibration command sent to recorder"),
      line("6D force sensor software zero requested"),
      line("[box] cali_6d_force_sensor OK (rc=0)"),
      line("[box] before: Fx=1.0000, Fy=0.0000, Fz=-5.0000, Mx=0.0000, My=0.0000, Mz=0.0000"),
      line("[box] after:  Fx=0.0100, Fy=0.0100, Fz=0.0200, Mx=0.0000, My=0.0000, Mz=0.0000"),
      line("ok", true),
    ];
    const parsed = parseCaliLog(lines);
    expect(parsed.requested).toBe(true);
    expect(parsed.done).toBe(true);
    expect(parsed.outcome).toBe("ok");
    expect(parsed.results).toHaveLength(1);
    const box = parsed.results[0];
    expect(box.boxId).toBe("box");
    expect(box.ok).toBe(true);
    expect(box.after?.fz).toBeCloseTo(0.02);
    expect(box.before?.fz).toBeCloseTo(-5);
  });

  it("parses a multi-box run and a FAILED box with rc/error", () => {
    const lines = [
      line("[box_a] cali_6d_force_sensor OK (rc=0)"),
      line("[box_a] after:  Fx=0.0100, Fy=0.0100, Fz=0.0200, Mx=0.0000, My=0.0000, Mz=0.0000"),
      line("[box_b] cali_6d_force_sensor FAILED (rc=-3): timeout"),
      line("error", true),
    ];
    const parsed = parseCaliLog(lines);
    expect(parsed.results).toHaveLength(2);
    const b = parsed.results.find((r) => r.boxId === "box_b");
    expect(b?.ok).toBe(false);
    expect(b?.rc).toBe(-3);
    expect(b?.error).toBe("timeout");
    expect(parsed.outcome).toBe("error");
  });

  it("honors startIndex to only read the latest run", () => {
    const lines = [
      line("[box] before: Fx=9.0000, Fy=0.0000, Fz=0.0000, Mx=0.0000, My=0.0000, Mz=0.0000"),
      line("ok", true),
      line("[box] after:  Fx=0.0100, Fy=0.0100, Fz=0.0200, Mx=0.0000, My=0.0000, Mz=0.0000"),
    ];
    const parsed = parseCaliLog(lines, 2);
    expect(parsed.results[0].before).toBeNull();
    expect(parsed.results[0].after?.fx).toBeCloseTo(0.01);
  });
});

describe("evaluateForce — origin vs dynamic Fz/Mx criteria differ", () => {
  // Dynamic keeps the residual tool/gravity load: Fz ≈ -7.784, Mx ≈ -0.168.
  const loaded = { fx: 0.1, fy: 0.1, fz: -7.784, mx: -0.168, my: 0, mz: 0 };
  const zeroed = { fx: 0.1, fy: 0.1, fz: 0.08, mx: 0, my: 0, mz: 0 };

  it("dynamic accepts Fz = -7.784 ± 0.5 and Mx = -0.168 ± 0.01 (not near zero)", () => {
    expect(evaluateForce(loaded, DYNAMIC_FORCE_LIMITS).pass).toBe(true);
    // dynamic must NOT treat a near-zero Fz/Mx as valid
    expect(evaluateForce(zeroed, DYNAMIC_FORCE_LIMITS).pass).toBe(false);
  });

  it("dynamic rejects an Mx that drifted off its target", () => {
    const mxOff = { fx: 0.1, fy: 0.1, fz: -7.784, mx: -0.2, my: 0, mz: 0 };
    const dyn = evaluateForce(mxOff, DYNAMIC_FORCE_LIMITS);
    expect(dyn.pass).toBe(false);
    expect(dyn.axes.find((a) => a.axis === "mx")?.pass).toBe(false);
  });

  it("origin accepts a near-zero Fz and rejects the loaded Fz", () => {
    expect(evaluateForce(zeroed, ORIGIN_FORCE_LIMITS).pass).toBe(true);
    expect(evaluateForce(loaded, ORIGIN_FORCE_LIMITS).pass).toBe(false);
  });

  it("flags an out-of-range Fy on both criteria with a reason", () => {
    const skewed = { fx: 0.1, fy: 0.86, fz: -7.784, mx: -0.168, my: 0, mz: 0 };
    const dyn = evaluateForce(skewed, DYNAMIC_FORCE_LIMITS);
    expect(dyn.pass).toBe(false);
    expect(dyn.firstFailure).toContain("Fy");
    const fyAxis = dyn.axes.find((a) => a.axis === "fy");
    expect(fyAxis?.pass).toBe(false);
  });

  it("flags an out-of-range moment", () => {
    const skewed = { fx: 0, fy: 0, fz: 0, mx: 0.05, my: 0, mz: 0 };
    const res = evaluateForce(skewed, ORIGIN_FORCE_LIMITS);
    expect(res.axes.find((a) => a.axis === "mx")?.pass).toBe(false);
  });
});
