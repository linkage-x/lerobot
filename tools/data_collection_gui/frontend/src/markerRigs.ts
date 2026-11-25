import type { MarkerRig } from "./types";

function monthDay(utc: string): string {
  const date = new Date(utc);
  if (Number.isNaN(date.getTime())) return "";
  return `${String(date.getMonth() + 1).padStart(2, "0")}-${String(date.getDate()).padStart(2, "0")}`;
}

/** When the rig was on its BOX, for a card title: "（11-25 起）", "（11-25 前）", "（11-06–11-25）" or "". */
export function markerRigPeriod(rig: Pick<MarkerRig, "mountedFromUtc" | "mountedUntilUtc">): string {
  const from = rig.mountedFromUtc ? monthDay(rig.mountedFromUtc) : "";
  const until = rig.mountedUntilUtc ? monthDay(rig.mountedUntilUtc) : "";
  if (from && until) return `（${from}–${until}）`;
  if (from) return `（${from} 起）`;
  if (until) return `（${until} 前）`;
  return "";
}

/** A rig's dropdown text: label, ids and period. */
export function markerRigOptionText(rig: MarkerRig): string {
  return `${rig.title}${markerRigPeriod(rig)}`;
}

export function isMarkerRigTarget(rigs: MarkerRig[] | undefined, target: string | null | undefined): boolean {
  return Boolean(target) && (rigs ?? []).some((rig) => rig.target === target);
}
