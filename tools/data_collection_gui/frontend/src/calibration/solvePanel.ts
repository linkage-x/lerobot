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
  IntrinsicsPreflight,
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


/** Whether re-fitting and exporting from this capture is doomed, and why.
 *
 * `export_production_calibration` builds a whole intrinsics run from one report
 * and has no way to carry a camera forward from the run in production, so every
 * camera with video in the capture must come out of the fit with a usable model.
 * On this rig cam_02/cam_03 point away from the board area and detect nothing in
 * every episode, which makes "re-fit the whole rig and export" structurally
 * impossible rather than unlucky -- and the failure lands at the *last* step,
 * after both captures have been decoded. Saying it before the click is worth an
 * hour every time it fires.
 *
 * It only blocks when production already ships intrinsics: a first calibration
 * of a fresh rig has nothing to extend and nothing to lose.
 */
export type PreflightView = { blocking: boolean; message: string; hint: string };

export function preflightView(
  preflight: IntrinsicsPreflight | undefined,
  refit: boolean,
  experiment: boolean,
): PreflightView {
  if (!refit || experiment || !preflight?.blocking) {
    return { blocking: false, message: "", hint: "" };
  }
  const names = preflight.uncalibrated.join("、");
  return {
    blocking: true,
    message:
      `${names} 没有在产内参，重算后必须各自拟合出可用模型才能导出——` +
      `任何一台看不到板都会让整轮在最后一步作废，已解码的部分全部白跑。` +
      `而且导出只写这份报告里的相机，当前在产的 ${preflight.production.length} 台不会被保留。`,
    hint: "勾上「只解算，不导出」跑这一轮：BA 照常解出这些相机并给残差，只是不写进生产。",
  };
}

/** What experiment mode is for, said where the checkbox is. */
export const EXPERIMENT_NOTE =
  "解算并给出残差，但不写生产标定、不动内外参指针、不清空 rig-check 基线。" +
  "测一颗新镜头、试一份采集时用这个——生产标定不该是实验的副作用。";
