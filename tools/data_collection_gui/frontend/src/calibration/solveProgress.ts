// One bar out of three subprocesses, and what it is allowed to claim.
//
// The solve runs ChArUco detection, then the bundle adjustment, then the
// export. Only the first can say how much work it has (one unit per recorded
// video); the other two report nothing until they finish. So the bar is
// weighted by step and moves within a step only when that step actually
// reports a unit -- a bar that creeps on a timer teaches the operator to stop
// reading it, which is worse than a bar that is honestly still.
//
// The numbers themselves come from the gateway, including elapsed: the rig's
// clock and the operator's laptop have been observed minutes apart, so a
// browser-side stopwatch would disagree with the log it is meant to explain.
import type { CalibrationStatus } from "../types";

export type SolveProgressView = {
  percent: number;
  headline: string;
  counter: string;
  detail: string;
  timing: string;
  /** The running step reports no units of its own, so the bar cannot advance. */
  indeterminate: boolean;
};

export function formatDuration(seconds: number): string {
  const total = Math.max(0, Math.round(seconds));
  if (total < 60) return `${total} 秒`;
  const minutes = Math.floor(total / 60);
  const restSeconds = total % 60;
  if (minutes < 60) return restSeconds ? `${minutes} 分 ${restSeconds} 秒` : `${minutes} 分`;
  const hours = Math.floor(minutes / 60);
  return `${hours} 小时 ${minutes % 60} 分`;
}

/** The bar, or null when there is nothing running to draw one for. */
export function solveProgressView(status: CalibrationStatus): SolveProgressView | null {
  const progress = status.progress;
  if (status.state !== "running" || !progress || progress.stepIndex <= 0) return null;

  const indeterminate = progress.total <= 0;
  // Never 100% while it is still running: the last step reports no units, and a
  // full bar that keeps spinning is exactly the "am I stuck?" this replaces.
  const percent = Math.max(0, Math.min(99, Math.round(progress.fraction * 100)));
  const counter = indeterminate ? "" : `${progress.done} / ${progress.total} 个视频`;
  const eta =
    progress.etaS > 0 ? `预计还需 ${formatDuration(progress.etaS)}` : "剩余时间还算不出来";
  return {
    percent,
    headline: `步骤 ${progress.stepIndex}/${progress.stepCount} · ${progress.label}`,
    counter,
    detail: progress.detail,
    timing: `已用 ${formatDuration(progress.elapsedS)} · ${eta}`,
    indeterminate,
  };
}
