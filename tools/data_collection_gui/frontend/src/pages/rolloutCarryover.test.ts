import { describe, expect, it } from "vitest";

import { carriedOverNotice } from "./rolloutCarryover";
import rolloutPageSource from "./RolloutPage.tsx?raw";

describe("carriedOverNotice", () => {
  it("always says the motion gate was not carried", () => {
    // The one thing the operator must not assume was inherited.
    expect(carriedOverNotice({ mode: "real" })).toContain("Motion confirmation is not.");
  });

  it("says so when takeover came back on", () => {
    // A remembered "yes" to a SpaceMouse is a second action source onto a moving arm. The
    // switch is inside a collapsed subcard, so without this the operator can start a rollout
    // that will hand them the arm without ever having seen the box ticked.
    const notice = carriedOverNotice({
      mode: "real",
      runtimeOptions: { daggerTakeover: true }
    });

    expect(notice).toContain("somebody has to be at the rig");
  });

  it("stays quiet about takeover when it is off", () => {
    expect(carriedOverNotice({ mode: "real", runtimeOptions: { daggerTakeover: false } })).not.toContain(
      "SpaceMouse"
    );
    expect(carriedOverNotice({ mode: "real", runtimeOptions: {} })).not.toContain("SpaceMouse");
    expect(carriedOverNotice({})).not.toContain("SpaceMouse");
  });
});

describe("the takeover switch across rollouts", () => {
  it("comes back from the last rollout instead of starting off every time", () => {
    // The switch lives in a subcard that a rollout can be started without ever scrolling to, so
    // there is nothing on screen to notice it was silently forgotten -- only a SpaceMouse that
    // does nothing when the operator reaches for it mid-rollout.
    expect(rolloutPageSource).toContain("setDaggerTakeover(options.daggerTakeover)");
  });

  it("is not reset by the checkpoint it came back with", () => {
    // Picking a checkpoint clears the prompt and the RTC knobs, because settings tuned for one
    // policy are the wrong ones for another. Takeover is a statement about who is at the rig,
    // not about the policy, so it must not be in that reset.
    const reset = rolloutPageSource.slice(
      rolloutPageSource.indexOf("skipDefaultsForRef.current = \"\";")
    );

    expect(reset.slice(0, reset.indexOf("}, [selected?.id]);"))).not.toContain(
      "setDaggerTakeover"
    );
  });
});
