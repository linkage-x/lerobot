import { describe, expect, it } from "vitest";
import type { CalibrationSessionStep } from "../types";
import { previewCameras, previewStatus } from "./stepPreview";

function step(overrides: Partial<CalibrationSessionStep> = {}): CalibrationSessionStep {
  return { kind: "intrinsics", camera: "cam_06", status: "pending", episodeIndex: -1, note: "", ...overrides };
}

describe("calibration step preview", () => {
  it("shows the one camera an intrinsics sweep is about", () => {
    expect(previewCameras(step(), ["cam_06", "cam_07"])).toEqual(["cam_06"]);
  });

  it("shows every camera for the shared extrinsics sweep", () => {
    expect(previewCameras(step({ kind: "extrinsics", camera: "" }), ["cam_06", "cam_07"])).toEqual([
      "cam_06",
      "cam_07",
    ]);
  });

  it("stops polling while an episode is open", () => {
    // Not a UI nicety: the recorder publishes no preview frames during an
    // episode, and making it do so would copy whole frames out of the loop
    // that feeds the encoders.
    const status = previewStatus(step({ status: "recording" }), "recording");
    expect(status.live).toBe(false);
    expect(status.note).toContain("录制中");
  });

  it("says the segment is over once the recorder auto-closed it and re-armed", () => {
    // The recorder ends an episode itself once its length elapses and goes back
    // to armed; the step stays "recording" until the operator clicks 保存本段.
    // Showing "录制中：预览已暂停" then is simply false — the preview is free
    // again and the board is no longer being recorded.
    const status = previewStatus(step({ status: "recording" }), "armed");
    expect(status.live).toBe(true);
    expect(status.note).toContain("已经结束");
    expect(status.note).toContain("保存本段");
  });

  it("goes live only once the cameras are connected and idle", () => {
    expect(previewStatus(step(), "armed").live).toBe(true);
    expect(previewStatus(step(), "review").live).toBe(true);
    expect(previewStatus(step(), "idle").live).toBe(false);
    // Saving is the recorder trimming and muxing the episode it just took;
    // asking it for previews then adds load exactly at the wrong moment.
    expect(previewStatus(step(), "saving").live).toBe(false);
  });

  it("says what to do rather than that nothing is available", () => {
    expect(previewStatus(step(), "idle").note).toContain("Connect");
    expect(previewStatus(step(), "armed").note).toContain("四角");
  });
});
