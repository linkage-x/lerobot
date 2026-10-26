// Pick the FR3 a request drives (roadmap B2). Starts empty on purpose; see robotArms.ts.
import { useEffect, useState } from "react";
import { api } from "../apiClient";
import type { RobotArms } from "../types";
import { armChoices, armPrompt } from "./robotArms";

export function useRobotArms(): RobotArms | null {
  const [arms, setArms] = useState<RobotArms | null>(null);
  useEffect(() => {
    let mounted = true;
    api.fetchRobotArms().then((next) => {
      if (mounted) setArms(next ?? { ok: false, required: true, arms: [], error: "gateway unreachable" });
    });
    return () => {
      mounted = false;
    };
  }, []);
  return arms;
}

export function ArmPicker({
  arms,
  value,
  onChange,
  disabled = false,
  both = false,
  label = "Arm"
}: {
  arms: RobotArms | null;
  value: string;
  onChange: (arm: string) => void;
  disabled?: boolean;
  both?: boolean;
  label?: string;
}) {
  const choices = armChoices(arms, both);
  return (
    <div className="arm-picker" role="radiogroup" aria-label={label}>
      <span>{label}</span>
      {choices.map((choice) => (
        <label key={choice.value} title={choice.title} className={choice.enabled ? "" : "arm-picker-off"}>
          <input
            type="radio"
            checked={value === choice.value}
            disabled={disabled || !choice.enabled}
            onChange={() => onChange(choice.value)}
          />
          {choice.label}
        </label>
      ))}
      {!choices.some((choice) => choice.value === value) ? <em className="arm-picker-prompt">{armPrompt(arms)}</em> : null}
    </div>
  );
}
