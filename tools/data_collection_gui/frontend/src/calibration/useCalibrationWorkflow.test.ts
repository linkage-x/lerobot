import { describe, expect, it } from "vitest";

import { ORIGIN_RECHECK, STABILITY } from "./config";
import {
  addRecheckSample,
  evaluateStability,
  medianForce,
  newOriginRecheck,
  recheckDone,
  recheckProgress,
  type StabilitySample,
} from "./useCalibrationWorkflow";

function touchSample(sourceId: string, net: number): StabilitySample {
  return { t: Date.now(), sourceId, vec: null, net };
}

describe("evaluateStability", () => {
  it("evaluates touch stability per pad instead of mixing left/right baselines", () => {
    const samples: StabilitySample[] = [];
    for (let i = 0; i < STABILITY.minSamples; i += 1) {
      samples.push(touchSample("box_touch_left", 1.0 + i * 0.01));
      samples.push(touchSample("box_touch_right", 4.8 + i * 0.01));
    }

    const result = evaluateStability(samples, "touch", ["box_touch_left", "box_touch_right"]);

    expect(result.status).toBe("stable");
    expect(result.peakToPeak).toBeCloseTo(0.07, 6);
  });

  it("still reports the worst unstable touch pad", () => {
    const samples: StabilitySample[] = [];
    for (let i = 0; i < STABILITY.minSamples; i += 1) {
      samples.push(touchSample("box_touch_left", 1.0 + i * 0.01));
      samples.push(touchSample("box_touch_right", 4.8 + i * 0.2));
    }

    const result = evaluateStability(samples, "touch", ["box_touch_left", "box_touch_right"]);

    expect(result.status).toBe("unstable");
    expect(result.peakToPeak).toBeCloseTo(1.4, 6);
    expect(result.detail).toContain("box_touch_right");
  });
});

describe("origin re-check", () => {
  const raw = (fx: number) => [fx, 0, 0, 0, 0, 0];
  const sample = (uptimeS: number, fx = 0.1) => ({
    timestamp: uptimeS * 1e6,
    fxyz_mxyz: [8, 16, -4, -1.5, 0.3, -0.2], // the software zero's shift, which a power cycle keeps
    fxyz_mxyz_raw: raw(fx),
  });
  const t0 = 1_000_000;

  it("reads raw from a restarted BOX, past the boot, not the shifted channel", () => {
    const recheck = newOriginRecheck(sample(1440), t0);
    for (let i = 0; i < 12; i += 1) {
      addRecheckSample(recheck, sample(70 + i * 0.3, 0.1 + i * 0.01), t0 + 100_000 + i * 300);
    }

    expect(recheck.rebooted).toBe(true);
    expect(recheck.samples).toHaveLength(11); // the first poll only sets the clock
    expect(recheckDone(recheck, t0 + 100_000 + 11 * 300)).toBe(true);
    expect(medianForce(recheck.samples).fx).toBeCloseTo(0.16, 6);
    expect(medianForce(recheck.samples).fz).toBe(0);
  });

  it("takes nothing from a BOX that was never re-powered", () => {
    const recheck = newOriginRecheck(sample(1440), t0);
    for (let i = 0; i < 20; i += 1) {
      addRecheckSample(recheck, sample(1450 + i * 0.3), t0 + 10_000 + i * 300);
    }

    expect(recheck.rebooted).toBe(false);
    expect(recheck.samples).toHaveLength(0);
    expect(recheckProgress([recheck])).toContain("等待");
  });

  it("takes nothing from the last sample a BOX left in the preview when it went off", () => {
    const recheck = newOriginRecheck(sample(1440), t0);
    for (let i = 0; i < 20; i += 1) addRecheckSample(recheck, sample(1441), t0 + 30_000 + i * 300);

    expect(recheck.samples).toHaveLength(0);
  });

  it("sees a re-power even when the origin itself ran just after a boot", () => {
    // the 11-06 16:46:38 origin: 27 s of uptime, so a later boot soon reads more than that
    const recheck = newOriginRecheck(sample(27), t0);
    addRecheckSample(recheck, sample(40), t0 + 60_000);
    addRecheckSample(recheck, sample(40.3), t0 + 60_300);

    expect(recheck.rebooted).toBe(true);
    expect(recheck.samples).toHaveLength(0);
    expect(recheckProgress([recheck])).toContain(`满 ${ORIGIN_RECHECK.minUptimeS} s`);
  });

  it("does not finish on too short a window or too few samples", () => {
    const recheck = newOriginRecheck(sample(1440), t0);
    addRecheckSample(recheck, sample(70), t0 + 100_000);
    addRecheckSample(recheck, sample(70.3), t0 + 100_300);

    expect(recheckDone(recheck, t0 + 100_300 + ORIGIN_RECHECK.windowMs)).toBe(false);
    for (let i = 2; i < 8; i += 1) addRecheckSample(recheck, sample(70 + i * 0.3), t0 + 100_000 + i * 300);
    expect(recheckDone(recheck, t0 + 100_300)).toBe(false);
    expect(recheckDone(recheck, t0 + 100_300 + ORIGIN_RECHECK.windowMs)).toBe(true);
  });
});
