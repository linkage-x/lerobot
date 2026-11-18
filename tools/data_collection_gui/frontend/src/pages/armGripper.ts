// Device Manager close/open button for an arm's gripper: what the next press does and how the
// last one went. The gateway route is /api/fr3/arm-gripper (it rides the connected recorder).
import type { ArmGripperCommand, ArmGripperRow } from "../types";

export type GripperAction = "close" | "open";

// Below this much short of the open width the jaw counts as not open (closed, or holding a cube).
const OPEN_MARGIN_MM = 8;

/**
 * The live width decides when there is one: it also covers what the button did not do (teleop, a
 * BOX reboot that handed the jaw to its own trigger). Without it, the opposite of the last command
 * that went through; with neither, close.
 */
export function nextGripperAction(row: ArmGripperRow, openWidthMm: number): GripperAction {
  if (row.widthMm != null) return row.widthMm >= openWidthMm - OPEN_MARGIN_MM ? "close" : "open";
  const last = row.last;
  if (last?.ok && last.action) return last.action === "close" ? "open" : "close";
  return "close";
}

export function gripperActionLabel(action: GripperAction): string {
  return action === "close" ? "闭合" : "张开";
}

export function armLabel(arm: string): string {
  return arm === "right" ? "右臂" : arm === "left" ? "左臂" : arm || "FR3";
}

export function lastCommandText(last: ArmGripperCommand | null): string {
  if (!last) return "";
  const what = last.action ? gripperActionLabel(last.action) : "指令";
  if (last.ok === null) return `${what}：等录制器回话…`;
  if (!last.ok) return `${what}失败：${last.error || "未知原因"}`;
  const target = last.targetM == null ? "" : `目标 ${(last.targetM * 1e3).toFixed(1)} mm`;
  const mode = last.modeBefore != null && last.modeBefore !== 1 ? "，先切到控制模式" : "";
  const at = last.at ? ` · ${new Date(last.at * 1000).toLocaleTimeString()}` : "";
  return `已${what}（${target}${mode}）${at}`;
}
