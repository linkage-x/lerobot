import type { DeviceStatus } from "./types";

export const handheldConfigSummary = {
  configPath: "tools/handheld/handheld_record_example.yaml",
  repoId: "local/handheld_multimodal_v1",
  root: "outputs/datasets/handheld_multimodal_v1",
  fps: 30,
  episodeTimeS: 10,
  targetFrames: 300,
  numEpisodes: "unlimited",
  video: true,
  streamingEncoding: true,
  vcodec: "h264",
  softSync: false,
  rerun: {
    displayData: true,
    savePath: "(not set)"
  }
};

export const initialDevices: DeviceStatus[] = [
  { id: "cam_0", kind: "camera", label: "Hikrobot DA9342700", state: "idle", fps: 30, latencyMs: 12, detail: "1280x720 GigE" },
  { id: "cam_1", kind: "camera", label: "Hikrobot DA9342716", state: "idle", fps: 30, latencyMs: 13, detail: "1280x720 GigE" },
  { id: "cam_2", kind: "camera", label: "Hikrobot DA9342685", state: "idle", fps: 30, latencyMs: 12, detail: "1280x720 GigE" },
  { id: "cam_3", kind: "camera", label: "Hikrobot DA9342471", state: "idle", fps: 30, latencyMs: 15, detail: "1280x720 GigE" },
  { id: "cam_4", kind: "camera", label: "Hikrobot DA9342477", state: "idle", fps: 30, latencyMs: 14, detail: "1280x720 GigE" },
  { id: "cam_5", kind: "camera", label: "Hikrobot DA9342673", state: "idle", fps: 30, latencyMs: 14, detail: "1280x720 GigE" },
  { id: "cam_6", kind: "camera", label: "Hikrobot DA9342615", state: "idle", fps: 30, latencyMs: 16, detail: "1280x720 GigE" },
  { id: "cam_7", kind: "camera", label: "Hikrobot DA9342583", state: "idle", fps: 30, latencyMs: 15, detail: "1280x720 GigE" },
  { id: "pika_left_realsense", kind: "camera", label: "RealSense 315122271700", state: "idle", fps: 30, latencyMs: 18, detail: "640x480 RGB" },
  { id: "pika_right_realsense", kind: "camera", label: "RealSense 315122271805", state: "idle", fps: 30, latencyMs: 18, detail: "640x480 RGB" },
  { id: "pika_left", kind: "handheld_gripper", label: "Pika Sense left", state: "idle", fps: 120, latencyMs: 5, detail: "/dev/ttyUSB1" },
  { id: "pika_right", kind: "handheld_gripper", label: "Pika Sense right", state: "idle", fps: 120, latencyMs: 5, detail: "/dev/ttyUSB0" }
];
