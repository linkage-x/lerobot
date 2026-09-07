import { describe, expect, it } from "vitest";

import type { RolloutRun } from "../types";
import { sessionAvailability, sessionNote } from "./rolloutSessionControls";

function run(overrides: Partial<RolloutRun> = {}): RolloutRun {
  return {
    state: "waiting",
    mode: "interactive",
    checkpointId: "ck",
    checkpointPath: "/ck",
    policy: "pi05",
    datasetRoot: "/data",
    targetFrameName: "pika_tcp",
    robotIp: "192.168.1.1",
    cameraKeys: ["side"],
    interactive: true,
    movesArm: true,
    step: 0,
    maxSteps: 0,
    commandStatus: "pass",
    clampedSteps: 0,
    leashedSteps: 0,
    rolloutIndex: 3,
    pendingOutcomeFor: 0,
    lastLines: [],
    logPath: "",
    message: "",
    ...overrides
  } as RolloutRun;
}

describe("which session controls may be pressed", () => {
  it("offers Start only between rollouts, never while one is running", () => {
    expect(sessionAvailability(run({ state: "waiting" }), false, true).canStart).toBe(true);
    expect(sessionAvailability(run({ state: "rolling" }), false, true).canStart).toBe(false);
  });

  it("does not offer Start while the runtime is still loading the policy", () => {
    // The click would be read by the listener thread and then cleared when the loop reaches its
    // wait: it looks like it worked and nothing happens.
    expect(sessionAvailability(run({ state: "starting" }), false, true).canStart).toBe(false);
    expect(sessionNote(run({ state: "starting" }))).toContain("Loading the policy");
  });

  it("offers Stop only while a rollout is actually rolling", () => {
    expect(sessionAvailability(run({ state: "rolling" }), false, true).canStop).toBe(true);
    expect(sessionAvailability(run({ state: "waiting" }), false, true).canStop).toBe(false);
  });

  it("refuses a scene reset the panel itself would refuse", () => {
    // The bar fires the panel's request. Enabling it on a panel with no painted mask or no
    // motion confirmation would be the bar inventing an arm movement nobody confirmed.
    expect(sessionAvailability(run({ state: "waiting" }), false, false).canResetScene).toBe(false);
    expect(sessionAvailability(run({ state: "waiting" }), false, true).canResetScene).toBe(true);
    expect(sessionAvailability(run({ state: "rolling" }), false, true).canResetScene).toBe(false);
  });

  it("presses nothing while a command is already in flight", () => {
    const busy = sessionAvailability(run({ state: "waiting" }), true, true);

    expect(busy).toEqual({
      canStart: false,
      canStop: false,
      canHome: false,
      canResetScene: false,
      canEnd: false
    });
  });

  it("gives a non-interactive run only the way out", () => {
    const smoke = sessionAvailability(run({ interactive: false, state: "rolling" }), false, true);

    expect(smoke.canEnd).toBe(true);
    expect(smoke.canStart || smoke.canStop || smoke.canHome || smoke.canResetScene).toBe(false);
  });

  it("keeps End session available while the arm is carrying out a command", () => {
    // Homing and resetting are the two states in which the arm is moving on its own, which is
    // exactly when an operator most needs a way to stop the session.
    expect(sessionAvailability(run({ state: "homing" }), false, true).canEnd).toBe(true);
    expect(sessionAvailability(run({ state: "resetting" }), false, true).canEnd).toBe(true);
    expect(sessionNote(run({ state: "homing" }))).toContain("start pose");
  });

  it("controls nothing when there is no session", () => {
    expect(sessionAvailability(null, false, true).canEnd).toBe(false);
    expect(sessionAvailability(run({ state: "complete" }), false, true).canEnd).toBe(false);
    expect(sessionNote(run({ state: "waiting" }))).toBe("");
  });
});
