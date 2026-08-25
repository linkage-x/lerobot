// What the preview beside the record button may show, and what it must say.
//
// The constraint that shapes this file: on the GMSL2 rig the preview JPEGs are
// published by the recorder itself, out of the synchronized cluster it has
// already captured (`preview_frame_bus_*`, every 12th frame -> ~5 Hz), and only
// while a viewer is asking. No second Argus client is opened, which is what
// makes an idle preview free.
//
// During an episode there is deliberately nothing to show: `START` clears the
// recorder's preview flag and `start_episode()` unlinks the JPEGs, because
// publishing them would mean copying full-resolution NV12 out of the same loop
// that feeds the encoders -- the one place a stall drops frames and fails the
// episode's full-cluster gate. So the preview freezes on its last frame and
// says why, rather than polling into 503s or pretending the feed is live.
import type { CalibrationSessionStep } from "../types";

/** Recorder states in which the cameras are connected but no episode is open. */
export const PREVIEW_LIVE_STATES = ["armed", "review"];

/** Which cameras this step is about: one for intrinsics, all of them for the rig. */
export function previewCameras(
  step: CalibrationSessionStep | undefined,
  cameras: string[],
): string[] {
  if (!step) return [];
  if (step.kind === "intrinsics") return step.camera ? [step.camera] : [];
  // The extrinsics sweep is judged by how often several cameras see the board
  // at the same instant, so showing one of them would show the wrong thing.
  return cameras;
}

export type PreviewStatus = { live: boolean; note: string };

export function previewStatus(
  step: CalibrationSessionStep | undefined,
  recorderState: string,
): PreviewStatus {
  if (!step) return { live: false, note: "" };
  if (step.status === "recording") {
    if (PREVIEW_LIVE_STATES.includes(recorderState)) {
      // The recorder closes an episode itself once the length it was given for
      // it elapses, and goes straight back to armed; the step is still
      // "recording" only because nobody has clicked 保存本段 yet. The preview is
      // free again -- and the operator has to be told the segment is already
      // over, because waving the board on from here records nothing.
      return {
        live: true,
        note:
          "本段已经结束：录满设定时长后录制器自动收尾了，继续挥板不会再录进去。" +
          "点「保存本段」把它登记下来，或「丢弃重录」。",
      };
    }
    return {
      live: false,
      note:
        "录制中：预览已暂停，画面停在开录前的最后一帧。" +
        "录制期间每一帧都归录制器——插一路预览要从编码环路里额外拷贝整帧，可能挤掉录制帧。",
    };
  }
  if (!PREVIEW_LIVE_STATES.includes(recorderState)) {
    return {
      live: false,
      note:
        recorderState === "connecting"
          ? "相机连接中，稍候即可看到画面。"
          : "相机未连接：先到「采集」页点 Connect，回来再录。",
    };
  }
  return {
    live: true,
    note:
      step.kind === "intrinsics"
        ? "开录前先对着预览走一遍挥板路径：板要能到达画面四角。0804 那次就是因为板没到边，整套内参重录。"
        : "开录前确认每台相机都能看到板：外参靠的是多台相机在同一时刻看到同一块板。",
  };
}
