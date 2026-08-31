import { describe, expect, it } from "vitest";
import {
  FR3_BASELINE_ACTION_LOSS_WEIGHTS,
  PI05_FULL_ACTION_EXPERT_LORA_TARGETS,
  describeLoraTargetModules,
  isFiveDimFr3DeltaActionMode,
  parseActionLossWeightInputs,
  separateActionLossWeights,
  withActionLossWeights
} from "./trainingPolicyConfig";

describe("LoRA target summaries", () => {
  it("distinguishes the compact pi0.5 default from the measured full-expert baseline", () => {
    expect(describeLoraTargetModules("pi05", "").title).toContain("compact q/v");
    const baseline = describeLoraTargetModules("pi05", PI05_FULL_ACTION_EXPERT_LORA_TARGETS);
    expect(baseline.title).toContain("full action-expert");
    expect(baseline.detail).toContain("q/k/v/o");
    expect(baseline.detail).toContain("gate/up/down");
  });

  it("names all-linear, suffix-list, and custom-regex capacity choices", () => {
    expect(describeLoraTargetModules("pi05", "all-linear").title).toContain("All linear");
    expect(describeLoraTargetModules("pi05", "q_proj,v_proj").title).toContain("2 targets");
    expect(describeLoraTargetModules("pi05", ".*expert.*").title).toContain("Custom target regex");
  });
});

describe("FR3 action loss weights", () => {
  it("recognizes the two five-dimensional delta action contracts", () => {
    expect(isFiveDimFr3DeltaActionMode("delta_ee_from_prev_cmd")).toBe(true);
    expect(isFiveDimFr3DeltaActionMode("delta_ee_from_current")).toBe(true);
    expect(isFiveDimFr3DeltaActionMode("absolute_ee")).toBe(false);
    expect(isFiveDimFr3DeltaActionMode(undefined)).toBe(false);
  });

  it("accepts the measured baseline and returns numeric values", () => {
    expect(parseActionLossWeightInputs(FR3_BASELINE_ACTION_LOSS_WEIGHTS)).toEqual({
      weights: [1, 1, 1, 0.2, 2]
    });
  });

  it("rejects missing, negative, non-finite, and all-zero weights", () => {
    expect(parseActionLossWeightInputs(["1", "", "1", "0.2", "2"]).error).toContain("dy");
    expect(parseActionLossWeightInputs(["1", "1", "-1", "0.2", "2"]).error).toContain("dz");
    expect(parseActionLossWeightInputs(["1", "1", "1", "Infinity", "2"]).error).toContain("drz");
    expect(parseActionLossWeightInputs(["0", "0", "0", "0", "0"]).error).toContain("above zero");
  });

  it("merges the labelled values into the generic policy config", () => {
    const config = withActionLossWeights('{"optimizer_lr":0.0001}', [1, 1, 1, 0.2, 2]);

    expect(JSON.parse(config)).toEqual({
      optimizer_lr: 0.0001,
      action_loss_weights: [1, 1, 1, 0.2, 2]
    });
  });

  it("lets the labelled controls replace a stale JSON copy", () => {
    const config = withActionLossWeights('{"action_loss_weights":[9,9,9,9,9]}', [1, 1, 1, 0.2, 2]);

    expect(JSON.parse(config).action_loss_weights).toEqual([1, 1, 1, 0.2, 2]);
  });

  it("removes the pi0.5-only vector for a view with a different action dimension", () => {
    expect(
      withActionLossWeights(
        '{"optimizer_lr":0.0001,"action_loss_weights":[1,1,1,0.2,2]}',
        null
      )
    ).toBe('{"optimizer_lr":0.0001}');
  });

  it("extracts checkpoint history into the five labelled controls", () => {
    const separated = separateActionLossWeights(
      '{"optimizer_lr":0.0001,"action_loss_weights":[1,1,1,0.2,2]}'
    );

    expect(separated).toEqual({
      actionLossWeights: ["1.0", "1.0", "1.0", "0.2", "2.0"],
      policyConfig: '{"optimizer_lr":0.0001}'
    });
  });

  it("reports malformed generic JSON before a training job starts", () => {
    expect(() => withActionLossWeights("{bad json", [1, 1, 1, 1, 1])).toThrow(
      "Policy config must be valid JSON"
    );
    expect(() => withActionLossWeights("[]", [1, 1, 1, 1, 1])).toThrow(
      "Policy config must be a JSON object"
    );
  });
});
