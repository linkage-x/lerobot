import { handheldConfigSummary, initialDevices } from "./defaultHandheldConfig";
import type {
  CollectionTask,
  EpisodeAnnotation,
  CalibrationStatus,
  CalibrationSession,
  MarkerTcpSession,
  ConfigSummary,
  DatasetExportStatus,
  DatasetFramePreview,
  DependencyInstall,
  DeploymentProfile,
  DeviceStatus,
  BoxPreviewPayload,
  BoxCaliLog,
  EventLogItem,
  GatewayStatus,
  ProcessingItem,
  ProcessingStatus,
  RecordedDataset,
  RecordingStatus,
  ReplayStatus,
  ReplayTimeline,
  RigCheckResponse,
  MujocoCubeMode,
  RealCubeMode,
  RealEndEffectorMode,
  MujocoPreview,
  RolloutLiveFrames,
  RealSensePreviewStatus,
  TrajectoryPoint,
  TeleopStatus,
  TeleopGains,
  TeleopGainValues,
  Checkpoint,
  CheckpointListing,
  RolloutOutcomeEntry,
  RolloutRun,
  RolloutStartRequest,
  RolloutStatusPayload,
  TrainingHost,
  TrainingMachine,
  TrainingRun,
  TrainingStartRequest,
  RolloutLastParams,
  RolloutLandmarks,
  TableAlignment,
  TableWindow,
  SceneResetRequest,
  SceneResetResult,
  SceneResetStroke,
  TrainingHistoryEntry,
  TrainingView,
  DaggerMergeCheck,
  DaggerMergeRequest,
  TrainingWandbStatus,
  TaskLadder,
} from "./types";

export type CameraCropSpecs = Record<string, [number, number, number, number]>;

const wait = (ms: number) => new Promise((resolve) => window.setTimeout(resolve, ms));

const defaultMujocoValidation = (datasetRoot: string, episode: number, fps: number): ReplayStatus["mujocoValidation"] => ({
  status: "not_run",
  datasetRoot: "",
  episode,
  fps,
  exitCode: null,
  completedFrames: 0,
  totalFrames: 0,
  avgPositionErrorMm: null,
  maxPositionErrorMm: null,
  avgRotationErrorDeg: null,
  maxRotationErrorDeg: null,
  maxPositionThresholdMm: 20,
  maxRotationThresholdDeg: 15,
  hasStructuredResult: false,
  trajectoryContract: {},
  isCurrentForSelection: false,
  message: datasetRoot ? "Run MuJoCo replay before real-robot replay." : "Select a recorded dataset before validation.",
  updatedAt: "",
  cubeMode: "left"
});

/**
 * Mirrors `teleop:` in tools/fr3/fr3_record_config.yaml, and the sim teleop script's own flag
 * defaults. Only the offline mock adapter reads these; against a live gateway both arrive in the
 * snapshot, which is the copy that is actually authoritative.
 */
const FR3_RECORD_CONFIG_TELEOP_GAINS: TeleopGainValues = {
  translation_scale: 0.000615,
  rotation_scale: 0.000648,
  scale_x: null,
  scale_y: null,
  scale_z: null,
  // Every axis follows its global since the rig switched to pika_gripper_ee. The sim script below
  // still zeroes all three rotation axes -- that divergence is the point of showing both columns.
  scale_wx: null,
  scale_wy: null,
  scale_wz: null
};

/** Mirrors gateway.FR3_TELEOP_AXIS_CALIBRATION, which mirrors the teleoperator. Mock adapter only. */
const FR3_TELEOP_AXIS_CALIBRATION: TeleopGains["axisCalibration"] = {
  scale_x: 1,
  scale_y: 0.9414634146341463,
  scale_z: 0.5902439024390244,
  scale_wx: 1,
  scale_wy: 0.9490740740740741,
  scale_wz: 0.9259259259259259
};

const FR3_SIM_TELEOP_GAINS: TeleopGainValues = {
  translation_scale: 0.001845,
  rotation_scale: 0.001944,
  scale_x: null,
  scale_y: null,
  scale_z: null,
  scale_wx: 0,
  scale_wy: 0,
  scale_wz: 0
};

export type GuiSnapshot = {
  deployment: DeploymentProfile;
  gateway: GatewayStatus;
  configSummary: ConfigSummary;
  devices: DeviceStatus[];
  recording: RecordingStatus;
  replay: ReplayStatus;
  teleop: TeleopStatus;
  teleopGains: TeleopGains;
  annotation: EpisodeAnnotation;
  calibration: CalibrationStatus;
  calibrationSession?: CalibrationSession;
  markerTcp?: MarkerTcpSession;
  datasetExport: DatasetExportStatus;
  recordedDatasets: RecordedDataset[];
  processing: ProcessingItem[];
  trajectory: TrajectoryPoint[];
  events: EventLogItem[];
  tasks: CollectionTask[];
  activeTaskId?: string;
  training?: TrainingRun;
  notice?: string;
};

/** A command the gateway rejected. Held until read so the 1s snapshot poll cannot bury it. */
export type CommandFailure = { endpoint: string; command: string; message: string };

export class DataCollectionGuiApi {
  private readonly apiBase = import.meta.env.VITE_GUI_API_BASE ?? "";
  private usingRemote = false;
  private commandFailure: CommandFailure | null = null;
  private snapshot: GuiSnapshot = {
    deployment: {
      profile: "thor",
      label: "Thor Acquisition",
      capabilities: ["gmsl2", "box", "imu", "tactile", "force_torque", "recording"],
      defaultRoute: "live-record"
    },
    gateway: {
      configPath: handheldConfigSummary.configPath,
      pid: null,
      state: "mock",
      processElapsedS: null,
      datasetsRoot: "outputs/datasets"
    },
    configSummary: {
      configPath: handheldConfigSummary.configPath,
      repoId: handheldConfigSummary.repoId,
      root: handheldConfigSummary.root,
      fps: handheldConfigSummary.fps,
      episodeTimeS: handheldConfigSummary.episodeTimeS,
      targetFrames: handheldConfigSummary.targetFrames,
      numEpisodes: "unlimited",
      video: handheldConfigSummary.video,
      streamingEncoding: handheldConfigSummary.streamingEncoding,
      vcodec: handheldConfigSummary.vcodec,
      softSync: handheldConfigSummary.softSync,
      rerun: handheldConfigSummary.rerun,
      recorderScript: handheldConfigSummary.recorderScript,
      rigType: handheldConfigSummary.rigType,
      hardwareSync: handheldConfigSummary.hardwareSync,
      cameraDefaults: handheldConfigSummary.cameraDefaults
    },
    devices: initialDevices,
    recording: {
      state: "idle",
      datasetRoot: handheldConfigSummary.root,
      repoId: handheldConfigSummary.repoId,
      episodeIndex: 0,
      savedEpisodes: 0,
      frameIndex: 0,
      targetFrames: handheldConfigSummary.targetFrames,
      queueDepth: 0,
      message: "Ready to launch handheld recorder"
    },
    replay: {
      state: "idle",
      dataset: handheldConfigSummary.repoId,
      episode: 0,
      frameIndex: 0,
      totalFrames: handheldConfigSummary.targetFrames,
      fps: handheldConfigSummary.fps,
      trackingErrorMm: 0,
      safety: "locked",
      message: "Start the local gateway to load recorded trajectory data",
      datasetRoot: handheldConfigSummary.root,
      datasetKind: "recorded",
      sourcePath: "",
      dataStatus: "missing",
      trajectoryKind: "none",
      totalEpisodes: 0,
      episodeOptions: [],
      recordedFrames: 0,
      diagnostics: [],
      pid: null,
      lastOutput: "",
      mujocoCubeMode: "left",
      mujocoValidation: defaultMujocoValidation(handheldConfigSummary.root, 0, handheldConfigSummary.fps),
      targetFrameName: "pika_gripper_ee"
    },
    teleop: {
      state: "idle",
      backend: "mujoco",
      inputDevice: "spacemouse",
      robotModel: "fr3_pika_gripper",
      urdfPath: "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper.urdf",
      simXmlPath: "src/lerobot/robots/franka_research3/assets/franka_fr3/fr3_pika_gripper_scene.xml",
      targetFrameName: "pika_gripper_ee",
      pid: null,
      message: "Start the local gateway to run FR3 Pika MuJoCo teleop",
      command: [],
      realRobotReady: false,
      cameraViews: [
        { id: "external", label: "External", source: "D435I", fps: 30, deviceId: "side" },
        { id: "wrist", label: "Wrist", source: "D405", fps: 30, deviceId: "ee" }
      ]
    },
    teleopGains: {
      values: { ...FR3_RECORD_CONFIG_TELEOP_GAINS },
      configDefaults: { ...FR3_RECORD_CONFIG_TELEOP_GAINS },
      simDefaults: { ...FR3_SIM_TELEOP_GAINS },
      axisCalibration: { ...FR3_TELEOP_AXIS_CALIBRATION },
      overridden: [],
      absMax: 0.01
    },
    annotation: {
      datasetRoot: handheldConfigSummary.root,
      episode: 0,
      taskPrompt: handheldConfigSummary.repoId,
      outcome: "unreviewed",
      quality: "unreviewed",
      includeInTraining: true,
      tags: [],
      notes: "",
      annotator: "",
      updatedAt: "",
      source: "default",
      segments: [],
      reviewStatus: "pending",
      reviewComment: ""
    },
    calibration: {
      state: "idle",
      pattern: "ChArUco 12x9 · 30 mm (charuco_400)",
      lastRunAt: "",
      message: "Run calibration to refresh extrinsics",
      cameras: [],
      outputPath: ""
    },
    markerTcp: {
      active: false,
      sessionName: "",
      sessionRoot: "",
      stage: "idle",
      samples: [],
      pendingSampleId: "",
      message: "Marker→TCP repeatability session not started",
      reportPath: ""
    },
    datasetExport: {
      state: "idle",
      target: "lerobot_v3",
      datasetRoot: handheldConfigSummary.root,
      outputPath: `${handheldConfigSummary.root}/exports/lerobot_v3`,
      selectedEpisodes: 0,
      totalFrames: 0,
      includeRaw: true,
      includeDebug: false,
      includeTraining: true,
      message: "Select a task or QC-passed dataset to export",
      manifest: [
        "raw/videos/*.mp4",
        "raw/events.jsonl",
        "training/lerobot_v3"
      ]
    },
    recordedDatasets: [],
    processing: [],
    trajectory: [],
    tasks: [],
    events: [
      {
        id: "boot",
        time: new Date().toLocaleTimeString(),
        level: "info",
        message: "GUI initialized with mock gateway adapter"
      }
    ]
  };

  async getSnapshot(): Promise<GuiSnapshot> {
    const remote = await this.getRemoteSnapshot();
    if (remote) {
      return remote;
    }
    await wait(120);
    this.snapshot = this.withFrontendFallbacks(this.snapshot);
    return structuredClone(this.snapshot);
  }

  async connectRecording(
    backend?: "real" | "sim",
    episodeTimeS?: number,
    fps?: number,
    numEpisodes?: number
  ): Promise<GuiSnapshot> {
    // The workstation profile picks between the hardware FR3 and its MuJoCo twin here; the
    // Thor profile has a single rig and sends no backend at all.
    const params = new URLSearchParams();
    if (backend) params.set("backend", backend);
    if (episodeTimeS != null && Number.isFinite(episodeTimeS)) {
      params.set("episode_time_s", String(episodeTimeS));
    }
    if (fps != null && Number.isFinite(fps)) {
      params.set("fps", String(fps));
    }
    if (numEpisodes != null && Number.isFinite(numEpisodes)) {
      params.set("num_episodes", String(numEpisodes));
    }
    const endpoint = params.toString()
      ? `/api/handheld/record/connect?${params.toString()}`
      : "/api/handheld/record/connect";
    const remote = await this.postRemoteSnapshot(endpoint);
    if (remote) {
      return remote;
    }
    await wait(220);
    this.snapshot.recording = {
      ...this.snapshot.recording,
      state: "armed",
      pid: 4242,
      frameIndex: 0,
      queueDepth: 0,
      message: "Devices connected; ready to start episode"
    };
    this.snapshot.devices = this.snapshot.devices.map((device) => ({ ...device, state: "running" }));
    this.log("info", "Handheld devices connected");
    return this.getSnapshot();
  }

  async startRecording(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/handheld/record/start");
    if (remote) {
      return remote;
    }
    await wait(180);
    this.snapshot.recording = {
      ...this.snapshot.recording,
      state: "recording",
      frameIndex: 0,
      queueDepth: 1,
      message: `Recording episode via ${handheldConfigSummary.configPath}`
    };
    this.snapshot.devices = this.snapshot.devices.map((device) => ({ ...device, state: "running" }));
    this.log("info", "Episode recording started");
    return this.getSnapshot();
  }

  async setRecordingStartPose(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/handheld/record/set-start-pose");
    if (remote) {
      return remote;
    }
    await wait(120);
    this.snapshot.recording = {
      ...this.snapshot.recording,
      message: "Start pose capture requested"
    };
    this.log("info", "FR3 start pose capture requested");
    return this.getSnapshot();
  }

  async resetRecordingStartPose(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/handheld/record/reset-start-pose");
    if (remote) {
      return remote;
    }
    await wait(120);
    this.snapshot.recording = {
      ...this.snapshot.recording,
      message: "Start pose reset requested"
    };
    this.log("info", "FR3 start pose reset to the configured default requested");
    return this.getSnapshot();
  }

  async resetRecordingScene(request: SceneResetRequest): Promise<GuiSnapshot> {
    const remote = await this.postRemoteJsonSnapshot("/api/handheld/record/scene-reset", request);
    if (remote) {
      return remote;
    }
    await wait(160);
    this.snapshot.recording = {
      ...this.snapshot.recording,
      message: "Scene reset needs the live FR3 gateway; mock mode did not move the arm"
    };
    this.log("warn", "Scene reset blocked because gateway is unavailable");
    return this.getSnapshot();
  }

  async stopRecording(action: "save" | "discard" | "exit"): Promise<GuiSnapshot> {
    const endpoint =
      action === "save"
        ? "/api/handheld/record/stop-save"
        : action === "discard"
          ? "/api/handheld/record/stop-discard"
          : "/api/handheld/record/exit";
    const remote = await this.postRemoteSnapshot(endpoint);
    if (remote) {
      return remote;
    }
    await wait(180);
    const saved = action === "save" ? this.snapshot.recording.savedEpisodes + 1 : this.snapshot.recording.savedEpisodes;
    const exiting = action === "exit";
    this.snapshot.recording = {
      ...this.snapshot.recording,
      state: exiting ? "idle" : "armed",
      savedEpisodes: saved,
      episodeIndex: saved,
      frameIndex: 0,
      queueDepth: 0,
      pid: exiting ? null : this.snapshot.recording.pid,
      message: action === "save" ? "Episode saved; ready for next episode" : action === "discard" ? "Episode discarded; ready for next episode" : "Session stopped"
    };
    this.snapshot.devices = this.snapshot.devices.map((device) => ({ ...device, state: exiting ? "idle" : "running" }));
    this.log(action === "save" ? "info" : "warn", `Recording command: ${action}`);
    return this.getSnapshot();
  }

  async preflightReplay(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/replay/preflight");
    if (remote) {
      return remote;
    }
    await wait(220);
    this.snapshot.replay = {
      ...this.snapshot.replay,
      state: "aborted",
      safety: "fault",
      message: "Gateway unavailable; preflight cannot be mocked for real-robot safety"
    };
    this.log("error", "Replay preflight blocked because gateway is unavailable");
    return this.getSnapshot();
  }

  async selectRecordedDataset(path: string): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/replay/select-dataset?path=${encodeURIComponent(path)}`);
    if (remote) {
      return remote;
    }
    await wait(140);
    const selected = this.snapshot.recordedDatasets.find((dataset) => dataset.path === path);
    this.snapshot.replay = {
      ...this.snapshot.replay,
      state: "idle",
      safety: "locked",
      episode: 0,
      frameIndex: 0,
      trackingErrorMm: 0,
      dataset: path,
      datasetRoot: path,
      sourcePath: selected?.sourcePath ?? "",
      dataStatus: selected?.dataStatus ?? "missing",
      totalEpisodes: selected?.totalEpisodes ?? 0,
      episodeOptions: Array.from({ length: selected?.totalEpisodes ?? 0 }, (_item, index) => index),
      recordedFrames: selected?.totalFrames ?? 0,
      totalFrames: selected?.totalFrames || this.snapshot.replay.totalFrames,
      message: selected ? `Selected recorded dataset: ${selected.name}` : `Selected recorded dataset: ${path}`,
      mujocoValidation: defaultMujocoValidation(path, 0, this.snapshot.replay.fps)
    };
    this.log("info", `Selected replay dataset: ${path}`);
    return this.getSnapshot();
  }

  async selectReplayEpisode(episode: number): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/replay/select-episode?episode=${episode}`);
    if (remote) {
      return remote;
    }
    await wait(120);
    this.snapshot.replay = {
      ...this.snapshot.replay,
      state: "idle",
      safety: "locked",
      episode,
      frameIndex: 0,
      trackingErrorMm: 0,
      message: `Selected episode ${episode}`,
      mujocoValidation: defaultMujocoValidation(this.snapshot.replay.datasetRoot ?? this.snapshot.replay.dataset, episode, this.snapshot.replay.fps)
    };
    this.log("info", `Selected replay episode: ${episode}`);
    return this.getSnapshot();
  }

  async deleteReplayEpisode(episode: number): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/replay/delete-episode?episode=${episode}`);
    if (remote) {
      return remote;
    }
    // No gateway: deletion rewrites on-disk parquet/videos and cannot be mocked.
    // Surface the failure instead of pretending the episode is gone.
    await wait(140);
    this.snapshot.replay = {
      ...this.snapshot.replay,
      message: "Gateway unavailable; episode deletion needs the backend and cannot be mocked"
    };
    this.log("error", `Delete episode ${episode} blocked because gateway is unavailable`);
    return this.getSnapshot();
  }

  async startMujocoReplay(cubeMode: MujocoCubeMode): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/replay/start-mujoco?cube=${encodeURIComponent(cubeMode)}`);
    if (remote) {
      return remote;
    }
    await wait(220);
    this.snapshot.replay = {
      ...this.snapshot.replay,
      state: "aborted",
      safety: "fault",
      frameIndex: 0,
      trackingErrorMm: 0,
      pid: null,
      mujocoCubeMode: cubeMode,
      message: "Gateway unavailable; MuJoCo validation cannot be mocked for real-robot safety",
      mujocoValidation: defaultMujocoValidation(
        this.snapshot.replay.datasetRoot ?? this.snapshot.replay.dataset,
        this.snapshot.replay.episode,
        this.snapshot.replay.fps
      )
    };
    this.log("error", "MuJoCo validation blocked because gateway is unavailable");
    return this.getSnapshot();
  }

  async approveMujocoReplay(cubeMode: MujocoCubeMode): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/replay/approve-mujoco?cube=${encodeURIComponent(cubeMode)}`);
    if (remote) return remote;
    throw new Error("Gateway unavailable; MuJoCo report approval cannot be completed.");
  }

  async startRealCubeReplay(
    cubeMode: RealCubeMode,
    robotIp: string,
    endEffectorMode: RealEndEffectorMode,
    overrideMujocoFailure = false
  ): Promise<GuiSnapshot> {
    const params = new URLSearchParams({
      cube: cubeMode,
      robot_ip: robotIp,
      end_effector: endEffectorMode,
      override_mujoco_failure: String(overrideMujocoFailure)
    });
    const remote = await this.postRemoteSnapshot(`/api/replay/start-real?${params.toString()}`);
    if (remote) return remote;
    throw new Error("Gateway unavailable; real-robot replay cannot start.");
  }

  async fetchMujocoPreview(
    datasetPath: string,
    episode: number,
    cubeMode: MujocoCubeMode
  ): Promise<MujocoPreview | null> {
    try {
      const response = await fetch(
        `${this.apiBase}/api/replay/mujoco-preview?path=${encodeURIComponent(datasetPath)}&episode=${episode}&cube=${encodeURIComponent(cubeMode)}`,
        { headers: { Accept: "application/json" } }
      );
      if (!response.ok) {
        return null;
      }
      return (await response.json()) as MujocoPreview;
    } catch {
      return null;
    }
  }

  async saveEpisodeAnnotation(annotation: EpisodeAnnotation): Promise<GuiSnapshot> {
    const remote = await this.postRemoteJsonSnapshot("/api/annotation/save", annotation);
    if (remote) {
      return remote;
    }
    await wait(160);
    this.snapshot.annotation = {
      ...annotation,
      tags: this.normalizeTags(annotation.tags),
      updatedAt: new Date().toISOString(),
      source: "manual"
    };
    this.log("info", `Annotation saved: episode ${annotation.episode}`);
    return this.getSnapshot();
  }

  async startSimTeleop(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/teleop/start-sim");
    if (remote) {
      return remote;
    }
    await wait(180);
    this.snapshot.teleop = {
      ...this.snapshot.teleop,
      state: "error",
      message: "Gateway unavailable; FR3 MuJoCo teleop must be started by the backend",
      pid: null
    };
    this.log("error", "FR3 MuJoCo teleop blocked because gateway is unavailable");
    return this.getSnapshot();
  }

  async startRealTeleop(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/teleop/start-real");
    if (remote) {
      return remote;
    }
    await wait(180);
    this.snapshot.teleop = {
      ...this.snapshot.teleop,
      state: "error",
      backend: "real",
      message: "Gateway unavailable; FR3 real teleop must be started by the backend",
      pid: null
    };
    this.log("error", "FR3 real teleop could not reach the gateway");
    return this.getSnapshot();
  }

  async stopTeleop(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/teleop/stop");
    if (remote) {
      return remote;
    }
    await wait(120);
    this.snapshot.teleop = {
      ...this.snapshot.teleop,
      state: "idle",
      pid: null,
      message: "FR3 Pika teleop stopped"
    };
    this.log("warn", "FR3 teleop stop requested");
    return this.getSnapshot();
  }

  /**
   * Replace the whole gain override set. An empty object resets every gain to the recorder config,
   * which is what Reset sends. Takes effect on the next teleop or recording spawn: the teleoperator
   * reads its gains once, at construction, so a running session keeps the ones it started with.
   */
  async setTeleopGains(gains: TeleopGainValues): Promise<GuiSnapshot> {
    const remote = await this.postRemoteJsonSnapshot("/api/teleop/gains", gains);
    if (remote) {
      return remote;
    }
    this.snapshot.teleopGains = {
      ...this.snapshot.teleopGains,
      values: { ...this.snapshot.teleopGains.configDefaults, ...gains },
      overridden: Object.keys(gains) as TeleopGains["overridden"]
    };
    this.log("warn", "Gateway unavailable; SpaceMouse gains changed in the mock adapter only");
    return this.getSnapshot();
  }

  async runCalibration(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/calibration/run");
    if (remote) {
      return remote;
    }
    // Offline, calibration cannot be faked: it reads recorded video and solves a
    // bundle adjustment. The old fallback invented reprojection numbers, which
    // is worse than saying nothing -- a green table implies the rig was verified.
    await wait(120);
    this.snapshot.calibration = {
      ...this.snapshot.calibration,
      state: "failed",
      lastRunAt: new Date().toISOString(),
      message: "离线模式无法标定：需要网关读取录制数据并运行 BA",
      cameras: []
    };
    this.log("warn", "Calibration unavailable without a gateway");
    return this.getSnapshot();
  }

  async setDatasetsRoot(path: string): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/processing/datasets-root?path=${encodeURIComponent(path)}`);
    if (remote) {
      if (remote.notice) {
        window.alert(remote.notice);
      }
      return remote;
    }
    await wait(120);
    this.snapshot.gateway = { ...this.snapshot.gateway, datasetsRoot: path };
    this.log("info", `Mock: datasets root set to ${path}`);
    return this.getSnapshot();
  }

  async queueTrajGen(path: string): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/processing/traj-gen?path=${encodeURIComponent(path)}`);
    if (remote) {
      return remote;
    }
    await wait(140);
    const message = "Gateway unavailable: cannot start EE trajectory generation.";
    window.alert(message);
    this.snapshot.processing = this.snapshot.processing.map((item) =>
      item.path === path
        ? {
            ...item,
            status: "pose_missing" as ProcessingStatus,
            message,
            logTail: [...item.logTail, `[traj-gen] ${message}`]
          }
        : item
    );
    this.log("warn", `Traj-gen unavailable: ${path}`);
    return this.getSnapshot();
  }

  async runQc(path: string): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/processing/qc?path=${encodeURIComponent(path)}`);
    if (remote) {
      return remote;
    }
    await wait(160);
    this.snapshot.processing = this.snapshot.processing.map((item) =>
      item.path === path
        ? {
            ...item,
            status: "qc_pass" as ProcessingStatus,
            qcSummary: "schema + sync + completeness pass",
            validFramesPct: item.validFramesPct ?? 96.4,
            message: "QC passed; available for export",
            logTail: [...item.logTail, "[qc] all checks passed"]
          }
        : item
    );
    this.log("info", `QC passed: ${path}`);
    return this.getSnapshot();
  }

  videoUrl(datasetPath: string, cameraKey: string, episode?: number): string {
    const params = new URLSearchParams({ path: datasetPath, key: cameraKey });
    if (episode != null) {
      params.set("episode", String(episode));
    }
    return `${this.apiBase}/api/replay/video?${params.toString()}`;
  }

  /**
   * One decoded still out of a recorded episode.
   *
   * Not `videoUrl`: a v3 dataset packs a whole chunk of episodes into one mp4 (tens of MB),
   * and the crop picker needs a single frame of it. The gateway seeks and decodes; this
   * fetches ~30 KB.
   */
  datasetFrameUrl(datasetPath: string, cameraKey: string, episode: number, frame: number): string {
    const params = new URLSearchParams({
      path: datasetPath,
      key: cameraKey,
      episode: String(episode),
      frame: String(frame)
    });
    return `${this.apiBase}/api/replay/frame.jpg?${params.toString()}`;
  }

  async fetchDatasetFramePreview(datasetPath: string): Promise<DatasetFramePreview | null> {
    try {
      const response = await fetch(
        `${this.apiBase}/api/replay/frame-info?path=${encodeURIComponent(datasetPath)}`,
        { headers: { Accept: "application/json" } }
      );
      if (!response.ok) return null;
      return (await response.json()) as DatasetFramePreview;
    } catch {
      return null;
    }
  }

  mujocoVideoUrl(datasetPath: string, episode: number, cubeMode: MujocoCubeMode): string {
    const params = new URLSearchParams({
      path: datasetPath,
      episode: String(episode),
      cube: cubeMode
    });
    return `${this.apiBase}/api/replay/mujoco-video?${params.toString()}`;
  }

  realSenseSnapshotUrl(cacheKey: number, cameraKey?: string): string {
    const params = new URLSearchParams({ t: String(cacheKey) });
    if (cameraKey) {
      params.set("key", cameraKey);
    }
    return `${this.apiBase}/api/replay/realsense.jpg?${params.toString()}`;
  }

  // ------------------------------------------------------------- training ---
  //
  // These do not go through the snapshot poll. A machine probe shells out to ssh and a
  // sync runs rsync; putting either on the 1s snapshot path would stall every other page
  // behind a host that is merely slow to answer.

  private async trainingGet<T>(path: string, params?: Record<string, string>): Promise<T | null> {
    try {
      const qs = params ? `?${new URLSearchParams(params).toString()}` : "";
      const response = await fetch(`${this.apiBase}${path}${qs}`, {
        headers: { Accept: "application/json" }
      });
      if (!response.ok) return null;
      return (await response.json()) as T;
    } catch {
      return null;
    }
  }

  private async trainingPost<T>(path: string, body: unknown): Promise<T & { ok: boolean; error?: string }> {
    try {
      const response = await fetch(`${this.apiBase}${path}`, {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(body)
      });
      const payload = (await response.json().catch(() => ({}))) as Record<string, unknown>;
      return {
        ...(payload as T),
        ok: response.ok && payload.ok !== false,
        error: typeof payload.error === "string" ? payload.error : undefined
      };
    } catch (error) {
      return {
        ...({} as T),
        ok: false,
        error: error instanceof Error ? error.message : String(error)
      };
    }
  }

  async fetchTrainingHosts(): Promise<TrainingHost[]> {
    const payload = await this.trainingGet<{ hosts: TrainingHost[] }>("/api/training/hosts");
    return payload?.hosts ?? [];
  }

  async fetchTrainingMachine(
    hostId: string
  ): Promise<{ machine: TrainingMachine; wandb: TrainingWandbStatus } | null> {
    return this.trainingGet<{ machine: TrainingMachine; wandb: TrainingWandbStatus }>(
      "/api/training/machine",
      { host: hostId }
    );
  }

  /** The extras to install come from the probe, not from the page: the page shows what the
   *  machine reported and asks for exactly that back. */
  async installTrainingDeps(hostId: string, extras: string[]) {
    return this.trainingPost<{ install?: DependencyInstall }>("/api/training/deps/install", {
      hostId,
      extras
    });
  }

  async fetchDependencyInstall(): Promise<DependencyInstall | null> {
    const payload = await this.trainingGet<{ install: DependencyInstall }>("/api/training/deps");
    return payload?.install ?? null;
  }

  async fetchTrainingViews(): Promise<TrainingView[]> {
    const payload = await this.trainingGet<{ views: TrainingView[] }>("/api/training/views");
    return payload?.views ?? [];
  }

  /** Past runs with the settings to reproduce them, newest first. */
  async fetchTrainingHistory(): Promise<TrainingHistoryEntry[]> {
    const payload = await this.trainingGet<{ runs: TrainingHistoryEntry[] }>(
      "/api/training/history"
    );
    return payload?.runs ?? [];
  }

  async fetchTrainingStatus(): Promise<TrainingRun | null> {
    const payload = await this.trainingGet<{ training: TrainingRun }>("/api/training/status");
    return payload?.training ?? null;
  }

  async addTrainingHost(host: {
    label: string;
    sshTarget: string;
    repoDir: string;
    pythonPath: string;
  }) {
    return this.trainingPost<{ host?: TrainingHost }>("/api/training/hosts/add", host);
  }

  async removeTrainingHost(hostId: string) {
    return this.trainingPost<Record<string, never>>("/api/training/hosts/remove", { hostId });
  }

  /** The key is written, never read back: the response only says whether one is stored. */
  async setTrainingWandbKey(hostId: string, apiKey: string) {
    return this.trainingPost<{ wandb?: TrainingWandbStatus }>("/api/training/wandb", {
      hostId,
      apiKey
    });
  }

  async clearTrainingWandbKey(hostId: string) {
    return this.trainingPost<{ wandb?: TrainingWandbStatus }>("/api/training/wandb", {
      hostId,
      clear: true
    });
  }

  async syncTrainingHost(hostId: string) {
    return this.trainingPost<{ sync?: { message?: string; changedCount?: number; detail?: string[] } }>(
      "/api/training/sync",
      { hostId }
    );
  }

  async startTraining(request: TrainingStartRequest) {
    return this.trainingPost<{ training?: TrainingRun }>("/api/training/start", request);
  }

  async stopTraining() {
    return this.trainingPost<{ training?: TrainingRun }>("/api/training/stop", {});
  }

  /** Whether these datasets can be merged into one policy-ready view, without writing anything.
   *
   *  Its own call rather than a flag on the merge: the answer is a list of refusals the
   *  operator can act on (QC not passed, a prompt that differs, a schema that does not line
   *  up), and those have to be readable *before* a job starts rewriting a dataset. The gateway
   *  runs the merge script with --check-only and hands back its JSON plus the paths it
   *  resolved, so what the page shows is what a merge would actually run on.
   */
  async checkDaggerMerge(request: DaggerMergeRequest) {
    return this.trainingPost<DaggerMergeCheck>("/api/training/dagger-merge/check", request);
  }

  /** Start the merge. The gateway re-runs the check first and refuses on anything it finds, so
   *  a stale "compatible" on the page cannot become a started job. Progress lands in the
   *  snapshot's datasetExport, the same slot a training-view build uses -- they write to the
   *  same place and only one may run at a time. */
  async startDaggerMerge(request: DaggerMergeRequest) {
    return this.trainingPost<{ merge?: DaggerMergeCheck }>("/api/training/dagger-merge/start", request);
  }

  // --------------------------------------------------- checkpoints & rollout ---
  //
  // Same reasoning as the training calls above: a checkpoint scan on a remote host runs
  // over ssh, and a fetch is an rsync of a few hundred megabytes. Neither belongs on the
  // snapshot poll.

  async fetchCheckpoints(hostId: string): Promise<CheckpointListing | null> {
    return this.trainingGet<CheckpointListing>("/api/checkpoints", { host: hostId });
  }

  async fetchCheckpoint(hostId: string, checkpointId: string): Promise<Checkpoint | null> {
    const listing = await this.fetchCheckpoints(hostId);
    return listing?.checkpoints.find((item) => item.id === checkpointId) ?? null;
  }

  async fetchCheckpointToLocal(hostId: string, checkpointId: string) {
    return this.trainingPost<{ localPath?: string; transferredCount?: number; message?: string }>(
      "/api/checkpoints/fetch",
      { hostId, checkpointId }
    );
  }

  async deleteCheckpoint(checkpointId: string) {
    return this.trainingPost<{ freedBytes?: number; message?: string }>("/api/checkpoints/delete", {
      checkpointId
    });
  }

  // Separate route rather than a list on /delete: the batch answers per checkpoint, so its
  // reply carries `failed` alongside `freedBytes` and the caller has to render both.
  async deleteCheckpoints(checkpointIds: string[]) {
    return this.trainingPost<{
      deleted?: string[];
      failed?: { checkpointId: string; error: string }[];
      freedBytes?: number;
      message?: string;
    }>("/api/checkpoints/delete-many", { checkpointIds });
  }

  async fetchRolloutStatus(): Promise<RolloutStatusPayload | null> {
    return this.trainingGet<RolloutStatusPayload>("/api/rollout/status");
  }

  async fetchRolloutOutcomes(): Promise<RolloutOutcomeEntry[]> {
    const payload = await this.trainingGet<{ entries: RolloutOutcomeEntry[] }>(
      "/api/rollout/outcomes"
    );
    return payload?.entries ?? [];
  }

  /** Where the demonstrations grasped and released, as the backdrop for rollout landing points.
   *  Reduced from the checkpoint's own dataset, so the first call after a gateway restart pays
   *  for one pass over its parquet and every later one is served from memory. */
  async fetchRolloutLandmarks(datasetRoot?: string): Promise<RolloutLandmarks> {
    const query = datasetRoot ? `?dataset=${encodeURIComponent(datasetRoot)}` : "";
    const payload = await this.trainingGet<{ landmarks: RolloutLandmarks }>(
      `/api/rollout/landmarks${query}`
    );
    return payload?.landmarks ?? {};
  }

  /** The grading ladders the repo ships, so the outcome prompt can offer this task's stages. */
  async fetchTaskLadders(): Promise<TaskLadder[]> {
    const payload = await this.trainingGet<{ ladders: TaskLadder[] }>("/api/rollout/ladders");
    return payload?.ladders ?? [];
  }

  /** The previous rollout's settings, including which checkpoint it ran, so a new one starts
   *  where the last left off. Never carries the safety gates; those are re-asked every time. */
  async fetchRolloutLastParams(): Promise<RolloutLastParams> {
    const payload = await this.trainingGet<{ params: RolloutLastParams }>(
      "/api/rollout/last-params"
    );
    return payload?.params ?? {};
  }

  async startRollout(request: RolloutStartRequest) {
    return this.trainingPost<{ rollout?: RolloutRun }>("/api/rollout/start", request);
  }

  /** One control word to the running rollout's stdin: the runtime reads it as a keypress.
   *  `takeover` is the latch, not the ordinary way in -- moving the SpaceMouse takes the arm on
   *  its own. The page sends it because a rollout the gateway launched has its stdin held as a
   *  pipe, which no keyboard backend can read: the `t` key a terminal operator would press does
   *  not exist here, and without this the browser session has no way to hold the arm still. */
  async controlRollout(command: "start" | "stop" | "home" | "quit" | "takeover") {
    return this.trainingPost<{ rollout?: RolloutRun }>("/api/rollout/control", { command });
  }

  async resetRolloutScene(request: SceneResetRequest): Promise<SceneResetResult> {
    return this.trainingPost<SceneResetResult>("/api/rollout/scene-reset", request);
  }

  /** Arm state since `after`, for drawing the rollout while it runs.
   *
   *  Polled rather than pushed: this gateway speaks request/response everywhere else, and a
   *  fifth of a second of latency is invisible on an arm that takes seconds to reach for
   *  something. Frames arrive in batches and are played out at the rate they were produced. */
  async fetchRolloutLiveFrames(after: number): Promise<RolloutLiveFrames | null> {
    return this.trainingGet<RolloutLiveFrames>(
      `/api/rollout/live-frames?after=${encodeURIComponent(String(Math.max(0, Math.trunc(after))))}`
    );
  }

  async stopRollout() {
    return this.trainingPost<{ rollout?: RolloutRun }>("/api/rollout/stop", {});
  }

  /** The landing point is deliberately not a parameter: the gateway attaches the one the
   *  runtime measured, so a grade can never be filed against a place the arm did not go. */
  async recordRolloutOutcome(record: {
    checkpointId?: string;
    /** Omitted when a stage is sent: the gateway derives success from the stage, and an
     *  outcome sent alongside one it disagrees with is refused rather than silently kept. */
    outcome?: "success" | "failure" | "aborted";
    mode?: string;
    steps?: number;
    note?: string;
    taskLadder?: string;
    stageId?: string;
    blocker?: string;
  }) {
    return this.trainingPost<{ entry?: RolloutOutcomeEntry }>("/api/rollout/outcome", record);
  }

  /** Cache-busted so the browser re-requests each poll rather than serving the first frame. */
  rolloutCameraUrl(cameraKey: string, nonce: number): string {
    return `${this.apiBase}/api/rollout/camera.jpg?camera=${encodeURIComponent(cameraKey)}&t=${nonce}`;
  }

  // A still of the parked cell, in camera pixels. Separate from rolloutCameraUrl because that
  // one is a liveness view and refuses a frame older than a few seconds -- between rollouts
  // every frame is older than that, and a still is what a reference layer wants anyway.
  cameraStillUrl(cameraKey: string, nonce: number): string {
    const params = new URLSearchParams({ camera: cameraKey, t: String(nonce) });
    return `${this.apiBase}/api/rollout/camera-still.jpg?${params.toString()}`;
  }

  /** The same still re-projected onto the table, cropped to exactly the base-frame rectangle
   *  the caller is plotting. The window is sent rather than assumed because the maps rescale
   *  around their data; anything the server had to guess would drift from the points drawn
   *  over it. 409 while the camera has never been aligned, which is why callers treat a failed
   *  load as "no backdrop" rather than as an error. */
  tableViewUrl(
    cameraKey: string,
    window: TableWindow,
    width: number,
    height: number,
    nonce: number
  ): string {
    const params = new URLSearchParams({
      camera: cameraKey,
      minX: window.minX.toFixed(5),
      maxX: window.maxX.toFixed(5),
      minY: window.minY.toFixed(5),
      maxY: window.maxY.toFixed(5),
      width: String(Math.round(width)),
      height: String(Math.round(height)),
      t: String(nonce)
    });
    return `${this.apiBase}/api/rollout/table-view.jpg?${params.toString()}`;
  }

  /** The frozen still taken while the arm stood at the last probed point. */
  tableProbeFrameUrl(cameraKey: string, nonce: number): string {
    const params = new URLSearchParams({ camera: cameraKey, t: String(nonce) });
    return `${this.apiBase}/api/rollout/table-probe-frame.jpg?${params.toString()}`;
  }

  async fetchTableAlignment(cameraKey: string): Promise<TableAlignment | null> {
    try {
      const response = await fetch(
        `${this.apiBase}/api/rollout/table-alignment?camera=${encodeURIComponent(cameraKey)}`,
        { headers: { Accept: "application/json" } }
      );
      if (!response.ok) return null;
      return (await response.json()) as TableAlignment;
    } catch {
      return null;
    }
  }

  async probeTablePoint(cameraKey: string, xyz: [number, number, number]) {
    return this.trainingPost<Record<string, never>>("/api/rollout/table-probe", {
      camera: cameraKey,
      x: xyz[0],
      y: xyz[1],
      z: xyz[2]
    });
  }

  async recordTablePoint(
    cameraKey: string,
    click: { u: number; v: number; imageWidth: number; imageHeight: number }
  ) {
    return this.trainingPost<TableAlignment>("/api/rollout/table-point", {
      camera: cameraKey,
      ...click
    });
  }

  async deleteTablePoint(cameraKey: string, index: number) {
    return this.trainingPost<TableAlignment>("/api/rollout/table-point/delete", {
      camera: cameraKey,
      index
    });
  }

  /** The target region stored on the rig, or none.
   *
   *  Rig-side rather than in the browser, for the same reason the table calibration is: the mask
   *  is a region of a physical table measured against a camera bolted above it, and it has to be
   *  the same region whichever machine, browser or page an operator opens next.
   */
  async fetchSceneResetMask(): Promise<SceneResetStroke[]> {
    const payload = await this.trainingGet<{ strokes: SceneResetStroke[] }>("/api/scene-reset/mask");
    return payload?.strokes ?? [];
  }

  async saveSceneResetMask(strokes: SceneResetStroke[]) {
    return this.trainingPost<{ strokes: SceneResetStroke[] }>("/api/scene-reset/mask", { strokes });
  }

  async clearTableAlignment(cameraKey: string) {
    return this.trainingPost<TableAlignment>("/api/rollout/table-clear", { camera: cameraKey });
  }

  async fetchRealSenseStatus(): Promise<RealSensePreviewStatus | null> {
    try {
      const response = await fetch(`${this.apiBase}/api/replay/realsense-status`, {
        headers: { Accept: "application/json" }
      });
      if (!response.ok) return null;
      return (await response.json()) as RealSensePreviewStatus;
    } catch {
      return null;
    }
  }

  teleopCameraUrl(viewId: string): string {
    const params = new URLSearchParams({ view: viewId, t: Date.now().toString() });
    return `${this.apiBase}/api/teleop/camera.jpg?${params.toString()}`;
  }

  cameraSnapshotUrl(deviceId: string): string {
    const params = new URLSearchParams({ key: deviceId, t: Date.now().toString() });
    return `${this.apiBase}/api/device-preview/camera.jpg?${params.toString()}`;
  }

  async fetchBoxPreview(deviceId: string): Promise<BoxPreviewPayload | null> {
    try {
      const params = new URLSearchParams({ device: deviceId });
      const response = await fetch(`${this.apiBase}/api/device-preview/box?${params.toString()}`, {
        headers: { Accept: "application/json" }
      });
      if (!response.ok) {
        return null;
      }
      return (await response.json()) as BoxPreviewPayload;
    } catch {
      return null;
    }
  }

  // The three box calibrations take an optional box_id: empty (single-box rigs)
  // keeps the whole-fleet behavior; a namespace id restricts the run to one box
  // (the calibration center's per-device button).
  private boxCaliUrl(path: string, boxId: string): string {
    const qs = boxId ? `?${new URLSearchParams({ box_id: boxId }).toString()}` : "";
    return `${this.apiBase}${path}${qs}`;
  }

  async triggerSixDForceCalibration(boxId = ""): Promise<{ ok: boolean; error?: string }> {
    try {
      const response = await fetch(this.boxCaliUrl("/api/device/calibrate-6dforce", boxId), {
        method: "POST",
        headers: { Accept: "application/json" }
      });
      const body = (await response.json().catch(() => ({}))) as { ok?: boolean; error?: string };
      return { ok: response.ok && body.ok !== false, error: body.error };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  async triggerSixDForceOriginCalibration(boxId = ""): Promise<{ ok: boolean; error?: string }> {
    try {
      const response = await fetch(this.boxCaliUrl("/api/device/calibrate-6dforce-origin", boxId), {
        method: "POST",
        headers: { Accept: "application/json" }
      });
      const body = (await response.json().catch(() => ({}))) as { ok?: boolean; error?: string };
      return { ok: response.ok && body.ok !== false, error: body.error };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  async triggerTouchCalibration(boxId = ""): Promise<{ ok: boolean; error?: string }> {
    try {
      const response = await fetch(this.boxCaliUrl("/api/device/calibrate-touch", boxId), {
        method: "POST",
        headers: { Accept: "application/json" }
      });
      const body = (await response.json().catch(() => ({}))) as { ok?: boolean; error?: string };
      return { ok: response.ok && body.ok !== false, error: body.error };
    } catch (error) {
      return { ok: false, error: error instanceof Error ? error.message : String(error) };
    }
  }

  async fetchBoxCaliLog(): Promise<BoxCaliLog | null> {
    try {
      const response = await fetch(`${this.apiBase}/api/device/box-cali-log`, {
        headers: { Accept: "application/json" }
      });
      if (!response.ok) {
        return null;
      }
      return (await response.json()) as BoxCaliLog;
    } catch {
      return null;
    }
  }

  // The self-check reads and writes the same three things: the last result, a
  // fresh run, and a re-captured baseline. Errors carry a message rather than
  // being swallowed, because "cameras are busy" is the most common outcome and
  // silently returning null would look like the check itself was broken.
  // Guided calibration session. Each of these returns the gateway's own error
  // string on refusal ("正在录制中，请先保存或丢弃"), which is the actionable part.
  private async calibrationSessionPost(path: string): Promise<{ ok: boolean; error?: string }> {
    try {
      const response = await fetch(`${this.apiBase}${path}`, {
        method: "POST",
        headers: { Accept: "application/json" }
      });
      const payload = (await response.json()) as { ok?: boolean; error?: string };
      return { ok: response.ok && payload.ok !== false, error: payload.error };
    } catch (error) {
      return { ok: false, error: String(error) };
    }
  }

  async startCalibrationSession(cameras?: string[]): Promise<{ ok: boolean; error?: string }> {
    const query = cameras?.length ? `?cameras=${encodeURIComponent(cameras.join(","))}` : "";
    return this.calibrationSessionPost(`/api/calibration/session/start${query}`);
  }

  async calibrationStepRecord(action: "start" | "save" | "discard"): Promise<{ ok: boolean; error?: string }> {
    return this.calibrationSessionPost(`/api/calibration/session/record?action=${action}`);
  }

  async calibrationStepSkip(): Promise<{ ok: boolean; error?: string }> {
    return this.calibrationSessionPost("/api/calibration/session/skip");
  }

  async cancelCalibrationSession(): Promise<{ ok: boolean; error?: string }> {
    return this.calibrationSessionPost("/api/calibration/session/cancel");
  }

  async startMarkerTcpSession(): Promise<{ ok: boolean; error?: string }> {
    return this.calibrationSessionPost("/api/calibration/marker-tcp/start");
  }

  async cancelMarkerTcpSession(): Promise<{ ok: boolean; error?: string }> {
    return this.calibrationSessionPost("/api/calibration/marker-tcp/cancel");
  }

  async markerTcpRecordSample(
    action: "start" | "save" | "discard",
    side: "left" | "right",
    condition: string
  ): Promise<{ ok: boolean; error?: string }> {
    const params = new URLSearchParams({ action, side, condition });
    return this.calibrationSessionPost(`/api/calibration/marker-tcp/record?${params.toString()}`);
  }

  async registerMarkerTcpStaticTransform(
    path: string,
    side: "left" | "right",
    condition: string
  ): Promise<{ ok: boolean; error?: string }> {
    const params = new URLSearchParams({ path, side, condition });
    return this.calibrationSessionPost(`/api/calibration/marker-tcp/register?${params.toString()}`);
  }

  async runMarkerTcpReport(): Promise<{ ok: boolean; error?: string }> {
    return this.calibrationSessionPost("/api/calibration/marker-tcp/report");
  }

  async fetchRigCheck(): Promise<RigCheckResponse | null> {
    try {
      const response = await fetch(`${this.apiBase}/api/calibration/rig-check`, {
        headers: { Accept: "application/json" }
      });
      if (!response.ok) {
        return null;
      }
      return (await response.json()) as RigCheckResponse;
    } catch {
      return null;
    }
  }

  async runRigCheck(): Promise<RigCheckResponse> {
    try {
      const response = await fetch(`${this.apiBase}/api/calibration/rig-check`, {
        method: "POST",
        headers: { Accept: "application/json" }
      });
      const payload = (await response.json()) as RigCheckResponse;
      return { ...payload, ok: response.ok && payload.ok !== false };
    } catch (error) {
      return { ok: false, error: String(error), report: null };
    }
  }

  async captureRigCheckBaseline(): Promise<RigCheckResponse> {
    try {
      const response = await fetch(`${this.apiBase}/api/calibration/rig-check/baseline`, {
        method: "POST",
        headers: { Accept: "application/json" }
      });
      const payload = (await response.json()) as RigCheckResponse;
      return { ...payload, ok: response.ok && payload.ok !== false };
    } catch (error) {
      return { ok: false, error: String(error), report: null };
    }
  }

  async fetchBoxTouchCaliLog(): Promise<BoxCaliLog | null> {
    try {
      const response = await fetch(`${this.apiBase}/api/device/box-touch-cali-log`, {
        headers: { Accept: "application/json" }
      });
      if (!response.ok) {
        return null;
      }
      return (await response.json()) as BoxCaliLog;
    } catch {
      return null;
    }
  }

  async fetchReplayTimeline(datasetPath: string, episode?: number): Promise<ReplayTimeline | null> {
    try {
      const params = new URLSearchParams({ path: datasetPath });
      if (episode != null) {
        params.set("episode", String(episode));
      }
      const response = await fetch(`${this.apiBase}/api/replay/timeline?${params.toString()}`, {
        headers: { Accept: "application/json" }
      });
      if (!response.ok) {
        return null;
      }
      return (await response.json()) as ReplayTimeline;
    } catch {
      return null;
    }
  }

  async abortReplay(): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot("/api/replay/abort");
    if (remote) {
      return remote;
    }
    await wait(140);
    this.snapshot.replay = {
      ...this.snapshot.replay,
      state: "aborted",
      safety: "locked",
      message: "Replay aborted; robot command stream stopped"
    };
    this.log("warn", "Replay aborted");
    return this.getSnapshot();
  }

  async createTask(task: Partial<CollectionTask>): Promise<GuiSnapshot> {
    const remote = await this.postRemoteJsonSnapshot("/api/tasks/create", task);
    if (remote) {
      return remote;
    }
    await wait(140);
    const newTask: CollectionTask = {
      id: `task-${Date.now()}`,
      name: task.name ?? "",
      description: task.description ?? "",
      targetEpisodes: task.targetEpisodes ?? 0,
      completedEpisodes: 0,
      status: "pending",
      assignee: task.assignee ?? "",
      datasetRepoId: task.datasetRepoId ?? "",
      tags: task.tags ?? [],
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString()
    };
    this.snapshot.tasks = [...this.snapshot.tasks, newTask];
    this.log("info", `Created task: ${newTask.name}`);
    return this.getSnapshot();
  }

  async updateTask(task: Partial<CollectionTask>): Promise<GuiSnapshot> {
    const remote = await this.postRemoteJsonSnapshot("/api/tasks/update", task);
    if (remote) {
      return remote;
    }
    await wait(120);
    this.snapshot.tasks = this.snapshot.tasks.map((t) =>
      t.id === task.id ? { ...t, ...task, updatedAt: new Date().toISOString() } : t
    );
    this.log("info", `Updated task: ${task.name ?? task.id}`);
    return this.getSnapshot();
  }

  async deleteTask(taskId: string): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/tasks/delete?id=${encodeURIComponent(taskId)}`);
    if (remote) {
      return remote;
    }
    await wait(120);
    this.snapshot.tasks = this.snapshot.tasks.filter((t) => t.id !== taskId);
    if (this.snapshot.activeTaskId === taskId) {
      this.snapshot.activeTaskId = "";
    }
    this.log("info", `Deleted task: ${taskId}`);
    return this.getSnapshot();
  }

  async exportTask(taskId: string): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/tasks/export?id=${encodeURIComponent(taskId)}`);
    if (remote) {
      return remote;
    }
    await wait(120);
    this.snapshot.datasetExport = {
      ...this.snapshot.datasetExport,
      state: "exporting",
      target: "lerobot_v3",
      message: `Consolidating sessions for task ${taskId}… (mock)`
    };
    this.log("info", `Started v3 export for task: ${taskId}`);
    return this.getSnapshot();
  }

  async exportApprovedDataset(
    paths: string[],
    actionMode?: string,
    acknowledgeWarnings = false,
    cameraCrops?: CameraCropSpecs,
    viewFps?: number,
    taskPrompt?: string
  ): Promise<GuiSnapshot> {
    // The workstation profile reuses this endpoint to build a training view, and picks the
    // action contract with actionMode; Thor sends none and gets the raw->v3 consolidation.
    // `paths` is repeated once per source rather than joined: dataset paths are absolute, and
    // any separator would have to be a character that cannot appear in one.
    const params = new URLSearchParams();
    for (const path of paths) {
      params.append("paths", path);
    }
    // Kept for the Thor consolidation and for any gateway that predates the multi-source build.
    if (paths.length > 0) {
      params.set("path", paths[0]);
    }
    if (actionMode) {
      params.set("action_mode", actionMode);
    }
    if (acknowledgeWarnings) {
      // The gateway refuses a QC-warned dataset without this, so the operator sees the warnings
      // before they decide rather than finding an unexplained absence in this list.
      params.set("acknowledge_warnings", "1");
    }
    if (cameraCrops && Object.keys(cameraCrops).length > 0) {
      params.set("camera_crops", JSON.stringify(cameraCrops));
    }
    if (viewFps !== undefined) {
      // Sent even when it matches the source rate, so the built view records that a rate was
      // chosen rather than inherited -- that is what a later merge needs to know.
      params.set("view_fps", String(viewFps));
    }
    if (taskPrompt && taskPrompt.trim()) {
      // Sent only when the operator typed one. Absent means "keep the prompt the recorder
      // wrote", which is the only honest default: this string is tokenized into every training
      // sample of a language-conditioned policy, so inventing one here would be inventing data.
      params.set("task_prompt", taskPrompt.trim());
    }
    const query = `?${params.toString()}`;
    const remote = await this.postRemoteSnapshot(`/api/datasets/export${query}`);
    if (remote) {
      return remote;
    }
    await wait(120);
    const primary = paths[0] ?? "";
    this.snapshot.datasetExport = {
      ...this.snapshot.datasetExport,
      state: "exporting",
      target: "lerobot_v3",
      datasetRoot: primary,
      datasetRoots: [...paths],
      outputPath: `${primary}/exports/lerobot_v3`,
      message: `Exporting ${paths.length} approved dataset(s)… (mock)`
    };
    this.log("info", `Started approved dataset v3 export: ${paths.join(", ")}`);
    return this.getSnapshot();
  }

  async setActiveTask(taskId: string): Promise<GuiSnapshot> {
    const remote = await this.postRemoteSnapshot(`/api/tasks/activate?id=${encodeURIComponent(taskId)}`);
    if (remote) {
      return remote;
    }
    await wait(80);
    this.snapshot.activeTaskId = taskId;
    this.log("info", taskId ? `Recording bound to task: ${taskId}` : "Cleared active recording task");
    return this.getSnapshot();
  }

  tick(): GuiSnapshot {
    if (this.usingRemote) {
      return structuredClone(this.snapshot);
    }

    if (this.snapshot.recording.state === "recording") {
      const nextFrame = Math.min(this.snapshot.recording.frameIndex + 3, this.snapshot.recording.targetFrames);
      this.snapshot.recording = {
        ...this.snapshot.recording,
        frameIndex: nextFrame,
        queueDepth: (this.snapshot.recording.queueDepth + 1) % 7,
        state: nextFrame >= this.snapshot.recording.targetFrames ? "review" : "recording",
        message: nextFrame >= this.snapshot.recording.targetFrames ? "Episode reached target duration; save or discard" : "Capturing frames"
      };
    }

    if (this.snapshot.replay.state === "sim_replay" || this.snapshot.replay.state === "replaying") {
      const nextFrame = Math.min(this.snapshot.replay.frameIndex + 2, this.snapshot.replay.totalFrames);
      this.snapshot.replay = {
        ...this.snapshot.replay,
        frameIndex: nextFrame,
        trackingErrorMm: this.snapshot.replay.state === "replaying" ? 2 + Math.sin(nextFrame / 18) * 1.2 : 0,
        state: nextFrame >= this.snapshot.replay.totalFrames ? "complete" : this.snapshot.replay.state,
        safety: nextFrame >= this.snapshot.replay.totalFrames ? "locked" : this.snapshot.replay.safety,
        pid: nextFrame >= this.snapshot.replay.totalFrames ? null : this.snapshot.replay.pid,
        message: nextFrame >= this.snapshot.replay.totalFrames ? "Replay complete" : this.snapshot.replay.message
      };
    }

    return structuredClone(this.snapshot);
  }

  private log(level: EventLogItem["level"], message: string) {
    this.snapshot.events = [
      {
        id: `${Date.now()}-${message}`,
        time: new Date().toLocaleTimeString(),
        level,
        message
      },
      ...this.snapshot.events
    ].slice(0, 12);
  }

  private async getRemoteSnapshot(): Promise<GuiSnapshot | null> {
    try {
      const response = await fetch(`${this.apiBase}/api/snapshot`, {
        headers: { Accept: "application/json" }
      });
      if (!response.ok) {
        throw new Error(`HTTP ${response.status}`);
      }
      const snapshot = (await response.json()) as GuiSnapshot;
      this.snapshot = this.withFrontendFallbacks(snapshot);
      this.usingRemote = true;
      return structuredClone(this.snapshot);
    } catch {
      this.usingRemote = false;
      this.log("warn", "Gateway unavailable; using mock adapter");
      return null;
    }
  }

  private async postRemoteSnapshot(endpoint: string): Promise<GuiSnapshot | null> {
    try {
      const response = await fetch(`${this.apiBase}${endpoint}`, {
        method: "POST",
        headers: { Accept: "application/json" }
      });
      if (!response.ok) {
        const message = await this.remoteErrorMessage(response);
        this.applyRemoteCommandError(endpoint, message);
        this.usingRemote = true;
        return structuredClone(this.snapshot);
      }
      const snapshot = (await response.json()) as GuiSnapshot;
      this.snapshot = this.withFrontendFallbacks(snapshot);
      this.usingRemote = true;
      return structuredClone(this.snapshot);
    } catch {
      this.usingRemote = false;
      this.log("warn", "Gateway command failed; using mock adapter");
      return null;
    }
  }

  private async postRemoteJsonSnapshot(endpoint: string, payload: unknown): Promise<GuiSnapshot | null> {
    try {
      const response = await fetch(`${this.apiBase}${endpoint}`, {
        method: "POST",
        headers: { Accept: "application/json", "Content-Type": "application/json" },
        body: JSON.stringify(payload)
      });
      if (!response.ok) {
        const message = await this.remoteErrorMessage(response);
        this.applyRemoteCommandError(endpoint, message);
        this.usingRemote = true;
        return structuredClone(this.snapshot);
      }
      const snapshot = (await response.json()) as GuiSnapshot;
      this.snapshot = this.withFrontendFallbacks(snapshot);
      this.usingRemote = true;
      return structuredClone(this.snapshot);
    } catch {
      this.usingRemote = false;
      this.log("warn", "Gateway command failed; using mock adapter");
      return null;
    }
  }

  private async remoteErrorMessage(response: Response): Promise<string> {
    try {
      const payload = await response.json();
      if (payload && typeof payload.error === "string") {
        return payload.error;
      }
    } catch {
      // Fall back to HTTP status text below.
    }
    return `HTTP ${response.status}`;
  }

  /**
   * Take the last rejected command, if any.
   *
   * The per-page fields written below live in the snapshot, which the 1s poll replaces with the
   * gateway's own state -- so a rejection was visible for less than a poll interval and a failed
   * command looked like a command that did nothing. This survives until someone reads it.
   */
  consumeCommandFailure(): CommandFailure | null {
    const failure = this.commandFailure;
    this.commandFailure = null;
    return failure;
  }

  private applyRemoteCommandError(endpoint: string, message: string) {
    const command = endpoint.split("?")[0].split("/").filter(Boolean).at(-1) ?? "command";
    this.commandFailure = { endpoint, command, message };
    if (endpoint.includes("/handheld/record/")) {
      this.snapshot.recording = {
        ...this.snapshot.recording,
        state: "error",
        message: `Recording ${command} failed: ${message}`
      };
    } else if (endpoint.includes("/replay/")) {
      this.snapshot.replay = {
        ...this.snapshot.replay,
        state: "aborted",
        safety: "fault",
        message: `Replay ${command} failed: ${message}`
      };
    } else if (endpoint.includes("/tasks/export") || endpoint.includes("/datasets/export")) {
      this.snapshot.datasetExport = {
        ...this.snapshot.datasetExport,
        state: "error",
        message: `Export ${command} failed: ${message}`
      };
    } else if (endpoint.includes("/processing/datasets-root")) {
      window.alert(`Datasets Root save failed: ${message}`);
    } else if (endpoint.includes("/processing/traj-gen")) {
      const fallbackMessage = "Generate EE Trajectory failed.";
      const displayMessage = message || fallbackMessage;
      const targetPath = new URLSearchParams(endpoint.split("?")[1] ?? "").get("path");
      window.alert(displayMessage);
      this.snapshot.processing = this.snapshot.processing.map((item) =>
        !targetPath || item.path === targetPath
          ? {
              ...item,
              status: "pose_missing" as ProcessingStatus,
              message: displayMessage,
              logTail: [...item.logTail, `[traj-gen] ${displayMessage}`]
            }
          : item
      );
    }
    this.log("error", `${command} failed: ${message}`);
  }

  private withFrontendFallbacks(snapshot: GuiSnapshot): GuiSnapshot {
    const recordedDatasets =
      snapshot.recordedDatasets?.length || snapshot.gateway.state !== "mock"
        ? snapshot.recordedDatasets ?? []
        : this.mockRecordedDatasets(snapshot);
    const processing = snapshot.processing ?? this.snapshot.processing ?? [];
    const activeAnnotationPath = snapshot.replay.datasetRoot || snapshot.replay.dataset || snapshot.recording.datasetRoot;
    const annotation =
      snapshot.annotation?.datasetRoot === activeAnnotationPath && snapshot.annotation.episode === snapshot.replay.episode
        ? snapshot.annotation
        : this.mockAnnotation(snapshot, false);
    const annotationWithDefaults: typeof annotation = {
      ...annotation,
      segments: annotation.segments ?? [],
      reviewStatus: annotation.reviewStatus ?? "pending",
      reviewComment: annotation.reviewComment ?? ""
    };
    return {
      ...snapshot,
      replay: {
        ...snapshot.replay,
        mujocoValidation:
          snapshot.replay.mujocoValidation ??
          defaultMujocoValidation(snapshot.replay.datasetRoot || snapshot.replay.dataset, snapshot.replay.episode, snapshot.replay.fps)
      },
      recordedDatasets,
      processing,
      annotation: annotationWithDefaults,
      tasks: snapshot.tasks ?? this.snapshot.tasks ?? [],
      calibration: snapshot.calibration ?? this.snapshot.calibration,
      teleopGains: snapshot.teleopGains ?? this.snapshot.teleopGains,
      markerTcp: snapshot.markerTcp ?? this.snapshot.markerTcp,
      datasetExport: snapshot.datasetExport ?? {
        ...this.snapshot.datasetExport,
        datasetRoot: snapshot.replay.datasetRoot || snapshot.recording.datasetRoot,
        outputPath: `${snapshot.replay.datasetRoot || snapshot.recording.datasetRoot}/exports/${this.snapshot.datasetExport.target}`,
        selectedEpisodes: Math.max(snapshot.replay.totalEpisodes ?? 0, snapshot.recording.savedEpisodes),
        totalFrames: snapshot.replay.recordedFrames || snapshot.trajectory.length,
        message: "Export planning is available in the frontend mock adapter"
      }
    };
  }

  private mockAnnotation(snapshot: GuiSnapshot, reuseExisting = true): EpisodeAnnotation {
    const datasetRoot = snapshot.replay.datasetRoot || snapshot.replay.dataset || snapshot.recording.datasetRoot;
    const existing = this.snapshot.annotation;
    if (reuseExisting && existing?.datasetRoot === datasetRoot && existing.episode === snapshot.replay.episode) {
      return existing;
    }
    return {
      datasetRoot,
      episode: snapshot.replay.episode ?? 0,
      taskPrompt: snapshot.configSummary.repoId,
      outcome: "unreviewed",
      quality: "unreviewed",
      includeInTraining: true,
      tags: [],
      notes: "",
      annotator: "",
      updatedAt: "",
      source: "default",
      segments: [],
      reviewStatus: "pending",
      reviewComment: ""
    };
  }

  private normalizeTags(tags: string[]): string[] {
    return tags.map((tag) => tag.trim()).filter(Boolean).slice(0, 12);
  }

  private mockRecordedDatasets(snapshot: GuiSnapshot): RecordedDataset[] {
    const root = snapshot.replay.datasetRoot || snapshot.recording.datasetRoot;
    if (!root) {
      return [];
    }
    return [
      {
        path: root,
        name: root.split("/").filter(Boolean).at(-1) ?? root,
        updatedAt: new Date().toLocaleString(),
        updatedAtMs: Date.now(),
        totalEpisodes: snapshot.replay.totalEpisodes ?? snapshot.recording.savedEpisodes,
        totalFrames: snapshot.replay.recordedFrames ?? snapshot.trajectory.length,
        dataStatus: snapshot.replay.dataStatus ?? (snapshot.trajectory.length ? "loaded" : "missing"),
        sourcePath: snapshot.replay.sourcePath ?? "",
        isLatest: true
      }
    ];
  }

}
