import { describe, expect, it } from "vitest";

import {
  assistedSuccessBlocked,
  landingPointDriver,
  terminalEventDriver
} from "./rolloutAttribution";

describe("who drove the moments a rollout is judged by", () => {
  it("attributes the point the map actually draws", () => {
    expect(
      landingPointDriver({ graspXyz: [0.3, -0.2, 0.05], graspBy: "policy", approachBy: "expert" })
    ).toBe("policy");
  });

  it("falls back to the approach when the gripper never closed", () => {
    expect(landingPointDriver({ approachXyz: [0.3, -0.2, 0.09], approachBy: "expert" })).toBe(
      "expert"
    );
  });

  it("leaves an unattributed point unknown rather than calling it the policy's", () => {
    expect(landingPointDriver({ graspXyz: [0.3, -0.2, 0.05] })).toBeUndefined();
  });

  it("reads the terminal event off the release, not off the grasp", () => {
    expect(
      terminalEventDriver({
        graspXyz: [0.3, -0.2, 0.05],
        graspBy: "policy",
        releaseXyz: [0.36, -0.13, 0.05],
        releaseBy: "expert"
      })
    ).toBe("expert");
  });

  it("has no terminal event to attribute when nothing was ever released", () => {
    expect(terminalEventDriver({ graspXyz: [0.3, -0.2, 0.05], graspBy: "expert" })).toBeUndefined();
  });
});

describe("what a rollout the operator finished may be graded", () => {
  const operatorFinished = {
    graspXyz: [0.3, -0.2, 0.05] as [number, number, number],
    graspBy: "policy" as const,
    releaseXyz: [0.36, -0.13, 0.05] as [number, number, number],
    releaseBy: "expert" as const
  };

  it("refuses success when the operator drove the terminal event", () => {
    expect(assistedSuccessBlocked(operatorFinished, false)).toBe(true);
  });

  it("allows it once the operator says the policy had already finished", () => {
    expect(assistedSuccessBlocked(operatorFinished, true)).toBe(false);
  });

  it("does not block a rollout the policy finished, whoever grasped", () => {
    expect(
      assistedSuccessBlocked(
        { ...operatorFinished, graspBy: "expert", releaseBy: "policy" },
        false
      )
    ).toBe(false);
  });

  it("does not block a rollout nobody attributed", () => {
    expect(assistedSuccessBlocked({ releaseXyz: [0.36, -0.13, 0.05] }, false)).toBe(false);
  });
});
