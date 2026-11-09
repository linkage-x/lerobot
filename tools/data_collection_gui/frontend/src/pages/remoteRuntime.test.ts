import { describe, expect, it } from "vitest";
import { remoteRuntimeOptions } from "./remoteRuntime";

describe("remoteRuntimeOptions", () => {
  it("drops the pi0.5 knobs and turns RTC off", () => {
    expect(
      remoteRuntimeOptions({
        taskPrompt: "insert the peg",
        rtcMode: "auto",
        rtcExecutionHorizon: 16,
        commandEmaAlpha: 0.5,
        actionSamples: 4,
        terminalServoPose: "0.4,0,0.2"
      })
    ).toEqual({ taskPrompt: "insert the peg", rtcMode: "disabled" });
  });

  it("keeps the grasp loop's own settings", () => {
    expect(
      remoteRuntimeOptions({
        rtcMode: "auto",
        rtcReplanQueueSize: 34,
        graspLoopTrials: 20,
        graspLoopSeed: "3",
        graspLoopArms: "A",
        graspLoopAttended: true,
        graspLoopInsertPose: undefined
      })
    ).toEqual({
      taskPrompt: undefined,
      rtcMode: "disabled",
      graspLoopTrials: 20,
      graspLoopSeed: "3",
      graspLoopArms: "A",
      graspLoopAttended: true
    });
  });
});
