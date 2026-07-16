import type { DeviceStatus } from "./types";

// Frontend mock fallback used when the gateway is unreachable. Production
// state comes from /api/state, which `tools/data_collection_gui/gateway.py`
// builds from `tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml` -- the GMSL2
// detected-camera GMSL2 + BOX rig that replaced the old Hikrobot + Pika setup.
export const handheldConfigSummary = {
  configPath: "tools/thor/gmsl2/thor_gmsl2_11ch_example.yaml",
  repoId: "local/thor_gmsl2_Nch_v1",
  root: "outputs/datasets/thor_gmsl2_Nch_v1",
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
  },
  recorderScript: "tools/thor/gmsl2/thor_record.py",
  rigType: "gmsl2" as const,
  hardwareSync: {
    enabled: true,
    fps: 60,
    trigMode: 1,
    pwmChip: "pwmchip6",
    pwmId: 0
  },
  cameraDefaults: {
    codec: "h265",
    bitrateKbps: 20000,
    width: 1920,
    height: 1080,
    pipeline: "argus",
    exposureUs: 9999,
    gain: 320,
    iframeInterval: 60,
    container: "mkv"
  }
};

// Fallback device list for mock mode (gateway unreachable). When the real
// gateway is running on Thor it detects actually locked cameras via
// check_max96726_locks.sh and only surfaces those.
export const initialDevices: DeviceStatus[] = [];
