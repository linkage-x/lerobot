export type ActionLossWeightInputs = [string, string, string, string, string];

export const ACTION_LOSS_WEIGHT_DIMENSIONS = [
  { key: "dx", label: "X translation" },
  { key: "dy", label: "Y translation" },
  { key: "dz", label: "Z translation" },
  { key: "drz", label: "Z-axis rotation" },
  { key: "gripper", label: "Gripper open / close" }
] as const;

/** The measured FR3 pi0.5 baseline, kept separate from the upstream equal-weight default. */
export const FR3_BASELINE_ACTION_LOSS_WEIGHTS: ActionLossWeightInputs = [
  "1.0",
  "1.0",
  "1.0",
  "0.2",
  "2.0"
];

/** No action_loss_weights in an older pi0.5 run means upstream's equal weighting. */
export const UNIFORM_ACTION_LOSS_WEIGHTS: ActionLossWeightInputs = [
  "1.0",
  "1.0",
  "1.0",
  "1.0",
  "1.0"
];

/** The broad LoRA target set used by the measured FR3 L4 baseline. */
export const PI05_FULL_ACTION_EXPERT_LORA_TARGETS =
  "(.*\\.gemma_expert\\..*\\.(self_attn\\.(q|k|v|o)_proj|mlp\\.(gate|up|down)_proj)|model\\.(state_proj|action_in_proj|action_out_proj|action_time_mlp_in|action_time_mlp_out))";

export type LoraTargetSummary = {
  detail: string;
  title: string;
};

/** Turn the PEFT target syntax into the capacity choice the operator is actually making. */
export function describeLoraTargetModules(policy: string, targetModules: string): LoraTargetSummary {
  const spec = targetModules.trim();
  if (!spec) {
    if (policy === "pi05") {
      return {
        title: "Policy default · compact q/v LoRA",
        detail:
          "Adapts the action expert's q_proj and v_proj plus the robot state/action interface. This is the smaller pi0.5 default, not the full-action-expert FR3 baseline."
      };
    }
    return {
      title: "Policy default",
      detail: `${policy} chooses its own LoRA target modules at training time.`
    };
  }
  if (spec === PI05_FULL_ACTION_EXPERT_LORA_TARGETS) {
    return {
      title: "FR3 baseline · full action-expert LoRA",
      detail:
        "Adapts q/k/v/o attention, gate/up/down MLP layers, and the robot state/action interface. This is the broader target set used by the measured L4 baseline."
    };
  }
  if (spec === "all-linear") {
    return {
      title: "All linear layers · widest LoRA",
      detail:
        "Adapts every eligible linear layer PEFT can see, including layers outside the action expert. Expect substantially more trainable parameters and memory use."
    };
  }
  if (spec.includes(",")) {
    const count = spec
      .split(",")
      .map((item) => item.trim())
      .filter(Boolean).length;
    return {
      title: `Custom suffix list · ${count} target${count === 1 ? "" : "s"}`,
      detail:
        "PEFT matches modules whose names end with one of these entries. Verify the names against the selected policy before starting training."
    };
  }
  return {
    title: "Custom target regex",
    detail:
      "PEFT treats this single value as a regular expression over module names. A typo can match no layers; compare the resolved trainable-parameter count with the intended run."
  };
}

export function isFiveDimFr3DeltaActionMode(actionMode: string | undefined): boolean {
  return actionMode === "delta_ee_from_prev_cmd" || actionMode === "delta_ee_from_current";
}

export type ParsedActionLossWeights = {
  error?: string;
  weights?: number[];
};

/** Validate at the form boundary so a bad vector never starts a remote sync and training job. */
export function parseActionLossWeightInputs(values: readonly string[]): ParsedActionLossWeights {
  if (values.length !== ACTION_LOSS_WEIGHT_DIMENSIONS.length) {
    return { error: `Expected ${ACTION_LOSS_WEIGHT_DIMENSIONS.length} action loss weights.` };
  }

  const weights: number[] = [];
  for (let index = 0; index < values.length; index += 1) {
    const raw = values[index].trim();
    const dimension = ACTION_LOSS_WEIGHT_DIMENSIONS[index];
    if (!raw) return { error: `${dimension.key} needs a weight.` };
    const value = Number(raw);
    if (!Number.isFinite(value)) return { error: `${dimension.key} must be a finite number.` };
    if (value < 0) return { error: `${dimension.key} must be zero or greater.` };
    weights.push(value);
  }
  if (weights.every((value) => value === 0)) {
    return { error: "At least one action dimension must have a weight above zero." };
  }
  return { weights };
}

function parsePolicyConfigObject(policyConfig: string): Record<string, unknown> {
  if (!policyConfig.trim()) return {};
  let parsed: unknown;
  try {
    parsed = JSON.parse(policyConfig);
  } catch (error) {
    const detail = error instanceof Error ? error.message : String(error);
    throw new Error(`Policy config must be valid JSON: ${detail}`);
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    throw new Error("Policy config must be a JSON object.");
  }
  return { ...(parsed as Record<string, unknown>) };
}

/**
 * Make the labelled controls the single owner of action_loss_weights.
 *
 * Passing null removes a stale pi0.5-only key when the selected view is not the FR3 5-D delta
 * contract. This prevents a copied config from failing later against a 7-D absolute action.
 */
export function withActionLossWeights(
  policyConfig: string,
  weights: readonly number[] | null
): string {
  const parsed = parsePolicyConfigObject(policyConfig);
  if (weights === null) delete parsed.action_loss_weights;
  else parsed.action_loss_weights = [...weights];
  return Object.keys(parsed).length > 0 ? JSON.stringify(parsed) : "";
}

function displayWeight(value: number): string {
  return Number.isInteger(value) ? value.toFixed(1) : String(value);
}

export type SeparatedPolicyConfig = {
  actionLossWeights?: ActionLossWeightInputs;
  policyConfig: string;
};

/** Pull a valid vector out of copied history so it appears in the five labelled controls. */
export function separateActionLossWeights(policyConfig: string): SeparatedPolicyConfig {
  if (!policyConfig.trim()) return { policyConfig: "" };
  let parsed: unknown;
  try {
    parsed = JSON.parse(policyConfig);
  } catch {
    return { policyConfig };
  }
  if (typeof parsed !== "object" || parsed === null || Array.isArray(parsed)) {
    return { policyConfig };
  }

  const config = { ...(parsed as Record<string, unknown>) };
  const candidate = config.action_loss_weights;
  if (!Array.isArray(candidate) || candidate.length !== ACTION_LOSS_WEIGHT_DIMENSIONS.length) {
    return { policyConfig };
  }
  if (!candidate.every((value) => typeof value === "number" && Number.isFinite(value) && value >= 0)) {
    return { policyConfig };
  }
  if (candidate.every((value) => value === 0)) return { policyConfig };

  delete config.action_loss_weights;
  return {
    actionLossWeights: candidate.map(displayWeight) as ActionLossWeightInputs,
    policyConfig: Object.keys(config).length > 0 ? JSON.stringify(config) : ""
  };
}
