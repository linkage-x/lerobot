import { describe, expect, it } from "vitest";

import { STABILITY } from "./config";
import { evaluateStability, type StabilitySample } from "./useCalibrationWorkflow";

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
