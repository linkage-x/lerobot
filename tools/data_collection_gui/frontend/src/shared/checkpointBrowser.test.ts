import { describe, expect, it } from "vitest";

import {
  checkpointMatches,
  formatBytes,
  successRate,
  verdictClass
} from "./CheckpointBrowser";
import rolloutPageSource from "../pages/RolloutPage.tsx?raw";
import type { Checkpoint } from "../types";

function makeCheckpoint(overrides: Partial<Checkpoint> = {}): Checkpoint {
  return {
    id: "eeframe_20260813_act_baseline/020000",
    jobName: "eeframe_20260813_act_baseline",
    stepLabel: "020000",
    step: 20000,
    isLast: false,
    aliasOf: "",
    path: "/repo/outputs/train/eeframe_20260813_act_baseline/checkpoints/020000",
    pretrainedPath: "/repo/outputs/train/x/checkpoints/020000/pretrained_model",
    policyType: "act",
    cameras: ["ee", "side"],
    totalSteps: 20000,
    datasetRepoId: "local/eeframe_fr3_spacemouse_20260813_160401__delta_ee_from_prev_cmd",
    datasetRoot: "/repo/outputs/exports/training_views/eeframe",
    recordedDatasetRoot: "/repo/outputs/exports/training_views/eeframe",
    sizeBytes: 206485583,
    modifiedAt: 1787128687,
    view: { exists: true, fps: 30, episodes: 20, actionMode: "delta_ee_from_prev_cmd" },
    contract: { targetFrameName: "pika_gripper_ee", robotIp: "192.168.1.206" },
    inferenceConfigPath: "",
    issues: [],
    verdict: "ok",
    outcomes: null,
    hostId: "local",
    hostLabel: "This machine",
    ...overrides
  };
}

describe("success rate", () => {
  it("counts successes against graded rollouts only", () => {
    const checkpoint = makeCheckpoint({
      outcomes: { success: 7, failure: 3, aborted: 0, total: 10 }
    });

    expect(successRate(checkpoint)).toBe("7/10");
  });

  it("keeps aborted rollouts out of the denominator", () => {
    // Stopping because someone walked into the cell says nothing about the policy. Counting
    // it as a failure would make a good checkpoint look worse every time the cell was busy.
    const checkpoint = makeCheckpoint({
      outcomes: { success: 7, failure: 3, aborted: 5, total: 15 }
    });

    expect(successRate(checkpoint)).toBe("7/10");
  });

  it("does not divide by zero when only aborted rollouts exist", () => {
    const checkpoint = makeCheckpoint({
      outcomes: { success: 0, failure: 0, aborted: 2, total: 2 }
    });

    expect(successRate(checkpoint)).toBe("0/0 (2 aborted)");
  });

  it("shows a dash for a checkpoint nobody has run", () => {
    expect(successRate(makeCheckpoint())).toBe("—");
  });
});

describe("checkpoint search", () => {
  const checkpoint = makeCheckpoint();

  it("matches everything on an empty query", () => {
    expect(checkpointMatches(checkpoint, "")).toBe(true);
    expect(checkpointMatches(checkpoint, "   ")).toBe(true);
  });

  it("matches the fields an operator would actually recall", () => {
    expect(checkpointMatches(checkpoint, "act")).toBe(true);
    expect(checkpointMatches(checkpoint, "eeframe")).toBe(true);
    expect(checkpointMatches(checkpoint, "spacemouse")).toBe(true);
    expect(checkpointMatches(checkpoint, "delta_ee")).toBe(true);
    expect(checkpointMatches(checkpoint, "side")).toBe(true);
  });

  it("is case insensitive", () => {
    expect(checkpointMatches(checkpoint, "ACT")).toBe(true);
    expect(checkpointMatches(checkpoint, "EEFrame")).toBe(true);
  });

  it("does not match an unrelated query", () => {
    expect(checkpointMatches(checkpoint, "diffusion")).toBe(false);
  });
});

describe("verdict styling", () => {
  it("maps a blocking verdict to the error tone", () => {
    // The verdict drives the only visual difference between a checkpoint that may be run and
    // one that may not, so this mapping is load-bearing rather than cosmetic.
    expect(verdictClass("block")).toBe("error");
    expect(verdictClass("warn")).toBe("warn");
    expect(verdictClass("ok")).toBe("ok");
  });
});

describe("byte formatting", () => {
  it("reports a checkpoint in megabytes and a run in gigabytes", () => {
    expect(formatBytes(206485583)).toBe("206 MB");
    expect(formatBytes(2400000000)).toBe("2.4 GB");
    expect(formatBytes(0)).toBe("—");
  });
});

describe("rollout page safety gating", () => {
  it("keeps motion confirmation out of the start payload's defaults", () => {
    // The gateway refuses a motion mode without confirmMotion, but the page must not send a
    // hardcoded `true` -- that would move the confirmation to a place nobody sees.
    expect(rolloutPageSource).not.toContain("confirmMotion: true");
    expect(rolloutPageSource).toContain("confirmMotion,");
  });

  it("resets motion confirmation when the mode or checkpoint changes", () => {
    // A confirmation is a statement about one specific run. Carrying it across a change of
    // checkpoint would let an operator arm a rollout they never looked at.
    expect(rolloutPageSource).toContain("setConfirmMotion(false);");
    expect(rolloutPageSource).toContain("}, [modeId, selected?.id]);");
  });

  it("disables Start until every gate is satisfied", () => {
    expect(rolloutPageSource).toContain("(mode.movesArm && !confirmMotion)");
    expect(rolloutPageSource).toContain("(blocking.length > 0 && !overrideContract)");
  });

  it("keeps pi0.5+LoRA runtime defaults on the Rollout page", () => {
    expect(rolloutPageSource).toContain("pi0.5+LoRA first rollout defaults");
    expect(rolloutPageSource).toContain("runtimeOptions: rolloutRuntimeOptions");
    expect(rolloutPageSource).toContain("RTC auto stays disabled for ACT-style policies");
    expect(rolloutPageSource).toContain("Show advanced rollout knobs");
  });
});
