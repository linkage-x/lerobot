import type { SideCamCheckResult, SideCamCheckStatus, SideCamVerdict } from "../types";

const BADGE: Record<SideCamVerdict, string> = {
  ok: "cali-badge cali-badge-running",
  shifted: "cali-badge cali-badge-warning",
  moved: "cali-badge cali-badge-error",
  unknown: "cali-badge cali-badge-idle"
};

export function verdictBadgeClass(verdict: SideCamVerdict | undefined): string {
  return verdict ? BADGE[verdict] ?? BADGE.unknown : BADGE.unknown;
}

function signed(value: number, digits = 1): string {
  return `${value >= 0 ? "+" : ""}${value.toFixed(digits)}`;
}

/** One line of numbers for a fitted pose, in the image's own terms. */
export function describePose(result: SideCamCheckResult | null | undefined): string {
  const pose = result?.pose;
  if (!pose) return "";
  return (
    `水平 ${signed(pose.dxPx)} px，竖直 ${signed(pose.dyPx)} px，旋转 ${signed(pose.rotDeg)}°；` +
    `胶带可见 ${pose.tapePixels} px，覆盖基准 ${Math.round(pose.coverage * 100)}%`
  );
}

/** Where the reference came from: the training-era default, or a still someone captured. */
export function describeReference(status: SideCamCheckStatus | null): string {
  const reference = status?.reference;
  if (!reference) return "还没有基准画面";
  if (reference.kind === "default") return reference.note || `默认基准：${reference.source ?? ""}`;
  return `基准：${formatTime(reference.capturedAt)} 手动采集`;
}

export function formatTime(iso: string | undefined): string {
  if (!iso) return "";
  const when = new Date(iso);
  if (Number.isNaN(when.getTime())) return iso;
  const pad = (n: number) => String(n).padStart(2, "0");
  return `${pad(when.getMonth() + 1)}-${pad(when.getDate())} ${pad(when.getHours())}:${pad(when.getMinutes())}`;
}
