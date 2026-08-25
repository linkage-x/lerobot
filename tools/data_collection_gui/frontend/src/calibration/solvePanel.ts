// Which capture gets solved, and when the solve button is offered.
//
// The wizard used to treat capture and solve as one pipeline: the solve button
// existed only in stage "ready", and a failed solve moved the session to
// "failed", where no button exists at all. That made a failure terminal for a
// capture that was sitting intact on disk -- and the fallback that would have
// found it again requires "calib" in the directory name, which the recorder
// never puts there (it names captures after the rig:
// thor_gmsl2_10ch_v1_20260820_152528).
//
// So: the capture is the durable thing and the solve is a retryable operation
// on it. The button is offered whenever there is something to solve and nothing
// running, and the capture it will read is named on screen rather than inferred.
import type {
  CalibrationSession,
  CalibrationSessionStep,
  CalibrationSolve,
  CalibrationStatus,
} from "../types";

export type SolveTargetView = {
  /** Empty when there is nothing solvable; the button is then disabled. */
  selected: string;
  summary: string;
  origin: string;
  candidates: { path: string; label: string }[];
};

const ORIGIN: Record<string, string> = {
  session: "本次引导录制的采集",
  manual: "你指定的采集",
  auto: "自动挑到的最新标定采集",
  missing: "指定的采集读不到了",
  none: "",
};

export function solveTargetView(solve: CalibrationSolve | undefined): SolveTargetView {
  const candidates = (solve?.candidates ?? []).map((item) => ({
    path: item.path,
    label: `${item.name}${item.episodes ? ` · ${item.episodes} 段` : ""}${
      item.updatedAt ? ` · ${item.updatedAt}` : ""
    }`,
  }));
  if (!solve || !solve.datasetRoot) {
    return {
      selected: "",
      summary:
        solve?.source === "missing"
          ? "指定的采集读不到（目录不存在，或里面没有 episodes/）"
          : "没有可解算的采集——先录一段，或在下拉里选一份",
      origin: ORIGIN[solve?.source ?? "none"] ?? "",
      candidates,
    };
  }
  return {
    selected: solve.datasetRoot,
    summary: solve.episodes ? `${solve.episodes} 段 episode` : "已录制的采集",
    origin: ORIGIN[solve.source] ?? "",
    candidates,
  };
}

/** What re-fitting intrinsics would do, in the terms the operator chose between.
 *
 * The solve has only ever run the extrinsics half: detect -> calibrate_extrinsics
 * -> export, with intrinsics reused from the production run. The wizard walks
 * the operator through one intrinsics sweep per camera all the same, so those
 * recordings were landing on disk and then never being read by anything. This
 * says plainly which of the two is about to happen.
 */
export function intrinsicsNote(solve: CalibrationSolve | undefined, refit: boolean): string {
  if (!refit) {
    return solve?.intrinsicsRun
      ? `内参沿用现有的 ${solve.intrinsicsRun}，本次只解外参`
      : "本次只解外参，内参沿用现有标定";
  }
  if (!solve?.intrinsicsDatasetRoot) {
    return "还没选内参采集——需要逐台相机各录一段、板子走到画面四角的那种采集";
  }
  const episodes = solve.intrinsicsEpisodes ? `${solve.intrinsicsEpisodes} 段` : "";
  return `将从 ${solve.intrinsicsDatasetName}${episodes ? `（${episodes}）` : ""} 重新拟合内参，覆盖现有的 ${solve.intrinsicsRun || "内参"}`;
}

export type SolveButtonView = { visible: boolean; label: string; disabled: boolean };

/** Whether to offer the solve, and whether this would be a first go or a retry. */
export function solveButtonView(
  status: CalibrationStatus,
  session: CalibrationSession | undefined,
): SolveButtonView {
  // While it runs, the progress bar is the answer; while a sweep is still being
  // captured, the set is not complete yet.
  if (status.state === "running" || session?.stage === "solving") {
    return { visible: false, label: "", disabled: true };
  }
  if (session?.active && session.stage === "capture") {
    return { visible: false, label: "", disabled: true };
  }
  const retry = status.state === "failed" || status.state === "complete";
  return {
    visible: true,
    label: retry ? "重新解算" : "开始解算",
    disabled: !status.solve?.datasetRoot,
  };
}

/** How many sweeps actually landed, as opposed to how far the wizard walked.
 *
 * "进度 11/11" says every step was dealt with, which includes being skipped --
 * the 2026-08-20 session read 11/11 with exactly one episode on disk. What the
 * solve reads is episodes, so that is what the panel has to show.
 */
export function captureTally(steps: CalibrationSessionStep[]): string {
  const captured = steps.filter((step) => step.status === "captured").length;
  const skipped = steps.filter((step) => step.status === "skipped").length;
  return `${captured} 段${skipped ? ` · 跳过 ${skipped}` : ""}`;
}
