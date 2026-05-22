import type { DeviceStatus } from "./types";

// Frontend mock fallback used when the gateway is unreachable. Production
// state comes from /api/state, which `tools/data_collection_gui/gateway.py`
// builds from `tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml` -- the GMSL2
// 11-camera + BOX 采集板 rig that replaced the old Hikrobot + Pika setup.
export const handheldConfigSummary = {
  configPath: "tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml",
  repoId: "local/thor_gmsl2_11ch_v1",
  root: "outputs/datasets/thor_gmsl2_11ch_v1",
  fps: 60,
  episodeTimeS: 10,
  targetFrames: 600,
  numEpisodes: "unlimited",
  video: true,
  streamingEncoding: false,
  vcodec: "h265",
  softSync: false,
  rerun: {
    displayData: true,
    savePath: "(not set)"
  }
};

// Device list mirrors the YAML defaults. Camera and box-sensor presence
// is determined at Connect time -- the gateway pushes the real subset to
// the frontend via the "Cameras:" / "Box devices:" recorder-output lines,
// so anything listed here that is not actually attached transitions from
// "idle" -> "error" once Connect runs.
export const initialDevices: DeviceStatus[] = [
  { id: "cam_00", kind: "camera", label: "GMSL2 sensor-id 0", state: "idle", fps: 60, latencyMs: 0, detail: "1920x1080 argus" },
  { id: "cam_01", kind: "camera", label: "GMSL2 sensor-id 1", state: "idle", fps: 60, latencyMs: 0, detail: "1920x1080 argus" },
  { id: "cam_02", kind: "camera", label: "GMSL2 sensor-id 2", state: "idle", fps: 60, latencyMs: 0, detail: "1920x1080 argus" },
  { id: "cam_03", kind: "camera", label: "GMSL2 sensor-id 3", state: "idle", fps: 60, latencyMs: 0, detail: "1920x1080 argus" },
  { id: "cam_04", kind: "camera", label: "GMSL2 sensor-id 4", state: "idle", fps: 60, latencyMs: 0, detail: "1920x1080 argus" },
  { id: "cam_05", kind: "camera", label: "GMSL2 sensor-id 5", state: "idle", fps: 60, latencyMs: 0, detail: "1920x1080 argus" },
  { id: "cam_06", kind: "camera", label: "GMSL2 sensor-id 6", state: "idle", fps: 60, latencyMs: 0, detail: "1920x1080 argus" },
  { id: "cam_07", kind: "camera", label: "GMSL2 sensor-id 7", state: "idle", fps: 60, latencyMs: 0, detail: "1920x1080 argus" },
  { id: "cam_08", kind: "camera", label: "GMSL2 sensor-id 8", state: "idle", fps: 60, latencyMs: 0, detail: "1920x1080 argus" },
  { id: "cam_09", kind: "camera", label: "GMSL2 sensor-id 9", state: "idle", fps: 60, latencyMs: 0, detail: "1920x1080 argus" },
  { id: "cam_10", kind: "camera", label: "GMSL2 sensor-id 10", state: "idle", fps: 60, latencyMs: 0, detail: "1920x1080 argus" },
  { id: "box_gripper", kind: "box_collection", label: "BOX gripper (distance)", state: "idle", fps: 20, latencyMs: 0, detail: "UDP 192.168.2.60:15000" },
  { id: "box_imu", kind: "box_collection", label: "BOX IMU (acc/gyr/euler/quat)", state: "idle", fps: 20, latencyMs: 0, detail: "UDP 192.168.2.60:15000" },
  { id: "box_trigger", kind: "box_collection", label: "BOX trigger travel", state: "idle", fps: 20, latencyMs: 0, detail: "UDP 192.168.2.60:15000" },
  { id: "box_six_d_force", kind: "box_collection", label: "BOX 6D force", state: "idle", fps: 20, latencyMs: 0, detail: "UDP 192.168.2.60:15000" },
  { id: "box_touch_left", kind: "box_collection", label: "Paxini touch pad L", state: "idle", fps: 20, latencyMs: 0, detail: "UDP 192.168.2.60:15000" },
  { id: "box_touch_right", kind: "box_collection", label: "Paxini touch pad R", state: "idle", fps: 20, latencyMs: 0, detail: "UDP 192.168.2.60:15000" }
];
