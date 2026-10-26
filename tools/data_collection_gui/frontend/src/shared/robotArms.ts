// Which FR3 a request drives (roadmap B2). The gateway refuses a request on Thor that names no arm, so the
// pickers start empty: with two arms a pre-filled one is the wrong arm moving.
import type { RobotArm, RobotArms } from "../types";

export type ArmChoice = { value: string; label: string; enabled: boolean; title: string };

// The IP of a chosen arm in service; "" for none, an unknown arm, or one not in service.
export function armIp(arms: RobotArms | null, arm: string): string {
  const found = arms?.arms.find((row) => row.arm === arm);
  return found && found.inService ? found.robotIp : "";
}

// The arm an IP belongs to (the gateway resolves the same way); "" when no arm in service has it.
export function armOfIp(arms: RobotArms | null, ip: string): string {
  return arms?.arms.find((row) => row.inService && row.robotIp === ip.trim())?.arm ?? "";
}

function describe(row: RobotArm): string {
  const box = row.gripperBox != null ? ` · BOX ${row.gripperBox}` : "";
  return row.inService ? `${row.robotIp}${box}` : `not in service: ${row.whyNot}`;
}

// Every arm, in service or not (one that is not stays visible, disabled, with its reason); `both` adds
// "every arm in service" for the preflight.
export function armChoices(arms: RobotArms | null, both = false): ArmChoice[] {
  const rows = arms?.arms ?? [];
  const choices: ArmChoice[] = rows.map((row) => ({
    value: row.arm,
    label: row.arm,
    enabled: row.inService,
    title: describe(row)
  }));
  const serving = rows.filter((row) => row.inService);
  if (both && serving.length) {
    choices.push({
      value: "both",
      label: serving.length > 1 ? "both" : `all in service (${serving[0].arm})`,
      enabled: true,
      title: `preflight ${serving.map((row) => row.arm).join(" and ")}, one after the other`
    });
  }
  return choices;
}

// What the pickers say when nothing is chosen yet, or why nothing can be.
export function armPrompt(arms: RobotArms | null): string {
  if (!arms) return "Reading tools/fr3/fr3_arms.json…";
  if (!arms.ok) return `Arm registry unreadable: ${arms.error}`;
  if (!arms.arms.some((row) => row.inService)) return "No FR3 is in service (tools/fr3/fr3_arms.json).";
  return "Choose the arm: there is no default.";
}
